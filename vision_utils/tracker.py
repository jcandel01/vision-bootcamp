"""Tracker multiobjeto desde cero: Kalman + húngaro (SORT), con opción ByteTrack y apariencia (ReID).

Es la solución de referencia de los ejercicios del notebook 06; el notebook 07 y el proyecto final la reutilizan.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .mot_metrics import iou_matrix


def xyxy_to_cxcywh(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], np.float64)


def cxcywh_to_xyxy(s):
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBox:
    """Filtro de Kalman de velocidad constante. Estado x = [cx, cy, w, h, vx, vy, vw, vh].

    El ruido se escala con la altura de la caja (como ByteTrack/BoT-SORT): una persona grande (cerca)
    se mueve más píxeles por frame que una pequeña (lejos).
    """
    std_pos, std_vel = 1 / 20, 1 / 160

    def __init__(self, box_xyxy):
        z = xyxy_to_cxcywh(box_xyxy)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)          # pos += vel (dt = 1 frame)
        self.H = np.eye(4, 8)               # solo observamos la posición/tamaño
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        self.P = np.diag(np.r_[np.full(4, (2 * self.std_pos * h) ** 2), np.full(4, (10 * self.std_vel * h) ** 2)])

    def predict(self):
        h = max(self.x[3], 1.0)
        Q = np.diag(np.r_[np.full(4, (self.std_pos * h) ** 2), np.full(4, (self.std_vel * h) ** 2)])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)   # evitar tamaños negativos
        return cxcywh_to_xyxy(self.x)

    def update(self, box_xyxy):
        z = xyxy_to_cxcywh(box_xyxy)
        R = np.diag(np.full(4, (self.std_pos * z[3]) ** 2))
        S = self.H @ self.P @ self.H.T + R                  # incertidumbre de la innovación
        K = self.P @ self.H.T @ np.linalg.inv(S)            # ganancia de Kalman
        self.x = self.x + K @ (z - self.H @ self.x)         # corrige con la medida
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self):
        return cxcywh_to_xyxy(self.x)


class Track:
    _next_id = 1

    def __init__(self, box, score, emb=None):
        self.kf = KalmanBox(box)
        self.id = None                   # se asigna al confirmar → IDs consecutivos sin huecos
        self.hits, self.age, self.time_since_update = 1, 1, 0
        self.score = score
        self.emb = emb
        self.history = []                # (frame, x1, y1, x2, y2)

    def confirm(self):
        if self.id is None:
            self.id = Track._next_id
            Track._next_id += 1

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, box, score, emb=None, ema=0.9):
        self.kf.update(box)
        self.hits += 1
        self.time_since_update = 0
        self.score = score
        if emb is not None:
            if self.emb is None:
                self.emb = emb
            else:                         # media móvil exponencial de la apariencia
                e = ema * self.emb + (1 - ema) * emb
                self.emb = e / np.linalg.norm(e)


class Tracker:
    """
    mode="sort": una sola asociación con las detecciones de score >= high_thr.
    mode="byte": ByteTrack — 1ª asociación con detecciones altas, 2ª con las bajas (low_thr <= s < high_thr)
                 solo para tracks que siguen sin emparejar y estaban activos en el frame anterior.
    use_reid:    fusiona coste de IoU con distancia coseno de embeddings y permite recuperar tracks perdidos
                 lejos de su predicción si la apariencia coincide (estilo DeepSORT/BoT-SORT).
    """

    def __init__(self, mode="byte", high_thr=0.5, low_thr=0.1, new_thr=0.6, iou_thr=0.3, low_iou_thr=0.5,
                 max_age=30, min_hits=3, use_reid=False, reid_weight=0.5, app_thr=0.35, ema=0.9):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.tracks = []
        self.frame = 0
        Track._next_id = 1

    # ---------- asociación
    def _cost(self, tracks, boxes, embs):
        pred = np.array([t.kf.box for t in tracks]).reshape(-1, 4)
        cost = 1 - iou_matrix(pred, boxes)
        gate = cost > 1 - self.iou_thr
        if self.use_reid and embs is not None and len(tracks) and len(boxes):
            T = np.array([t.emb for t in tracks])
            app = 1 - T @ embs.T
            # puerta espacial que crece con el tiempo perdido (permite reencontrar tras oclusión)
            c_t = np.array([[(t.kf.x[0]), t.kf.x[1], t.kf.x[3], t.time_since_update] for t in tracks])
            c_d = (boxes[:, :2] + boxes[:, 2:]) / 2
            dist = np.linalg.norm(c_t[:, None, :2] - c_d[None], axis=2) / c_t[:, None, 2]
            near = dist < (1.0 + 0.15 * c_t[:, 3:4])
            cost = self.reid_weight * app + (1 - self.reid_weight) * np.where(gate, 1.0, cost)
            gate = (app > self.app_thr) | ~near | (gate & (app > self.app_thr * 0.6))
        cost = np.where(gate, 1e5, cost)
        return cost

    def _match(self, tracks, boxes, embs, iou_thr=None):
        if not len(tracks) or not len(boxes):
            return [], list(range(len(tracks))), list(range(len(boxes)))
        old = self.iou_thr
        if iou_thr is not None:
            self.iou_thr = iou_thr
        C = self._cost(tracks, boxes, embs)
        self.iou_thr = old
        r, c = linear_sum_assignment(C)
        m = [(i, j) for i, j in zip(r, c) if C[i, j] < 1e5]
        mt, md = {i for i, _ in m}, {j for _, j in m}
        return m, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    # ---------- paso por frame
    def update(self, dets, embs=None):
        """dets: (N,5) x1 y1 x2 y2 score; embs: (N,D) opcional. Devuelve (M,7): id x1 y1 x2 y2 score det_idx."""
        self.frame += 1
        dets = np.asarray(dets, np.float64).reshape(-1, 5)
        for t in self.tracks:
            t.predict()

        hi = dets[:, 4] >= self.high_thr
        lo = (dets[:, 4] >= self.low_thr) & ~hi
        idx_hi, idx_lo = np.where(hi)[0], np.where(lo)[0]
        E = (lambda ix: embs[ix] if embs is not None else None)

        # 1ª asociación: detecciones altas vs todos los tracks
        m1, ut, ud = self._match(self.tracks, dets[idx_hi, :4], E(idx_hi))
        matched = [(self.tracks[i], idx_hi[j]) for i, j in m1]
        unmatched_tracks = [self.tracks[i] for i in ut]
        unmatched_hi = [idx_hi[j] for j in ud]

        # 2ª asociación (ByteTrack): detecciones bajas vs tracks activos sin emparejar, solo IoU
        if self.mode == "byte" and len(idx_lo):
            active = [t for t in unmatched_tracks if t.time_since_update == 1]
            reid, self.use_reid = self.use_reid, False      # baja confianza → apariencia poco fiable
            m2, _, _ = self._match(active, dets[idx_lo, :4], None, iou_thr=self.low_iou_thr)
            self.use_reid = reid
            for i, j in m2:
                matched.append((active[i], idx_lo[j]))
            got = {id(active[i]) for i, _ in m2}
            unmatched_tracks = [t for t in unmatched_tracks if id(t) not in got]

        for t, j in matched:
            t.update(dets[j, :4], dets[j, 4], None if embs is None else embs[j], self.ema)
            if t.hits >= self.min_hits or self.frame <= self.min_hits:
                t.confirm()

        # tracks nuevos solo con detecciones altas y con score suficiente
        for j in unmatched_hi:
            if dets[j, 4] >= self.new_thr:
                t = Track(dets[j, :4], dets[j, 4], None if embs is None else embs[j])
                if self.frame <= self.min_hits:
                    t.confirm()
                self.tracks.append(t)

        # borrar: tentativos que fallan una vez, y tracks perdidos más de max_age frames
        self.tracks = [t for t in self.tracks
                       if not (t.id is None and t.time_since_update > 0) and t.time_since_update <= self.max_age]

        out = []
        for t, j in matched:
            if t.id is not None and t.time_since_update == 0:
                b = t.kf.box
                t.history.append((self.frame, *b))
                out.append([t.id, *b, t.score, j])
        return np.array(out, np.float64).reshape(-1, 7)


def run_tracker(scene, **kw):
    """Ejecuta Tracker sobre una SyntheticScene.

    Devuelve array MOT (frame, id, x1, y1, x2, y2, score, det_idx); det_idx = índice de la detección en
    scene.dets[frame-1] (permite recuperar su embedding en scene.det_emb).
    """
    trk = Tracker(**kw)
    rows = []
    for f in range(scene.n_frames):
        out = trk.update(scene.dets[f], scene.det_emb[f] if kw.get("use_reid") else None)
        for r in out:
            rows.append([f + 1, *r])
    return np.array(rows, np.float64).reshape(-1, 8)
