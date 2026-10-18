"""Escena sintética de tracking con ground truth: personas que se mueven, se cruzan y se ocluyen.

Permite medir trackers con métricas reales (MOTA, IDF1, HOTA) sin descargar MOT17.
Cada persona tiene además un "embedding de apariencia" (simula la salida de un modelo ReID):
vector unitario propio + ruido, y algunas parejas se parecen mucho (mismo uniforme).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticScene:
    gt: np.ndarray            # (K, 6): frame, id, x1, y1, x2, y2
    dets: list                # por frame: array (N, 5) x1 y1 x2 y2 score
    det_emb: list             # por frame: array (N, D) embeddings de apariencia (L2-normalizados)
    det_gt_id: list           # por frame: id GT de cada detección (-1 = falso positivo) — solo para análisis
    n_frames: int
    size: tuple = (1280, 720)
    meta: dict = field(default_factory=dict)


def make_scene(n_people=18, n_frames=400, size=(1280, 720), seed=0, emb_dim=64,
               miss_rate=0.04, fp_per_frame=0.6, box_noise=0.04, emb_noise=0.35, n_similar_pairs=3,
               exit_prob=0.002):
    rng = np.random.default_rng(seed)
    W, H = size
    # --- estado inicial
    h = rng.uniform(70, 130, n_people)
    w = h * rng.uniform(0.35, 0.45, n_people)
    pos = np.c_[rng.uniform(80, W - 80, n_people), rng.uniform(150, H - 30, n_people)]   # pie (cx, y2)
    speed = rng.uniform(1.5, 5.0, n_people)
    ang = rng.uniform(0, 2 * np.pi, n_people)
    turn = rng.normal(0, 0.03, n_people)                     # curvatura → movimiento NO lineal
    # apariencia: algunas parejas casi iguales (mismo equipo)
    base = rng.normal(size=(n_people, emb_dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    for k in range(n_similar_pairs):
        a, b = 2 * k, 2 * k + 1
        base[b] = base[a] / np.linalg.norm(base[a]) + 0.5 * rng.normal(size=emb_dim) / np.sqrt(emb_dim)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    visible = np.ones(n_people, bool)
    away = np.zeros(n_people, int)

    gt, dets, det_emb, det_gt = [], [], [], []
    for f in range(1, n_frames + 1):
        # --- movimiento
        ang += turn + rng.normal(0, 0.05, n_people)
        turn += rng.normal(0, 0.003, n_people)
        turn = np.clip(turn, -0.06, 0.06)
        vel = np.c_[np.cos(ang), np.sin(ang) * 0.6] * speed[:, None]
        pos += vel
        # rebote en los bordes
        for d, lo, hi in [(0, 40, W - 40), (1, 120, H - 10)]:
            out = (pos[:, d] < lo) | (pos[:, d] > hi)
            pos[out, d] = np.clip(pos[out, d], lo, hi)
            if d == 0:
                ang[out] = np.pi - ang[out]
            else:
                ang[out] = -ang[out]
        # salidas temporales del campo (desapariciones)
        leave = visible & (rng.random(n_people) < exit_prob)
        visible[leave] = False
        away[leave] = rng.integers(20, 60, leave.sum())
        away[~visible] -= 1
        back = ~visible & (away <= 0)
        visible[back] = True

        persp = 0.6 + 0.4 * pos[:, 1] / H                      # más lejos (arriba) → más pequeño
        hh, ww = h * persp, w * persp
        boxes = np.c_[pos[:, 0] - ww / 2, pos[:, 1] - hh, pos[:, 0] + ww / 2, pos[:, 1]]
        for i in np.where(visible)[0]:
            gt.append([f, i + 1, *boxes[i]])

        # --- oclusión: fracción de cada caja tapada por personas más cercanas (y2 mayor)
        idx = np.where(visible)[0]
        occ = np.zeros(n_people)
        for i in idx:
            for j in idx:
                if j == i or boxes[j, 3] <= boxes[i, 3]:
                    continue
                ix = max(0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]))
                iy = max(0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
                occ[i] = max(occ[i], ix * iy / (ww[i] * hh[i]))

        fd, fe, fg = [], [], []
        for i in idx:
            if occ[i] > 0.75 or rng.random() < miss_rate:
                continue
            score = rng.uniform(0.6, 0.95) if occ[i] < 0.35 else rng.uniform(0.12, 0.5)
            n = box_noise * hh[i]
            b = boxes[i] + rng.normal(0, n, 4)
            fd.append([*b, score])
            e = base[i] + emb_noise * (1 + 2 * occ[i]) * rng.normal(size=emb_dim) / np.sqrt(emb_dim)   # ocluido → peor apariencia
            fe.append(e / np.linalg.norm(e))
            fg.append(i + 1)
        for _ in range(rng.poisson(fp_per_frame)):
            hh0 = rng.uniform(40, 120)
            x, y = rng.uniform(0, W - 50), rng.uniform(100, H - hh0)
            fd.append([x, y, x + hh0 * 0.4, y + hh0, rng.uniform(0.1, 0.55)])
            e = rng.normal(size=emb_dim)
            fe.append(e / np.linalg.norm(e))
            fg.append(-1)
        dets.append(np.array(fd, np.float32).reshape(-1, 5))
        det_emb.append(np.array(fe, np.float32).reshape(-1, emb_dim))
        det_gt.append(np.array(fg, int))

    return SyntheticScene(np.array(gt, np.float32), dets, det_emb, det_gt, n_frames, size,
                          meta=dict(n_people=n_people, seed=seed))


def draw_scene(scene, frame_idx, tracks=None, ax=None):
    """Dibuja GT (gris) y opcionalmente tracks (color por id) en un frame (1-indexado)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    W, H = scene.size
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    g = scene.gt[scene.gt[:, 0] == frame_idx]
    for _, i, x1, y1, x2, y2 in g:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="0.7", lw=1))
    if tracks is not None:
        t = tracks[tracks[:, 0] == frame_idx]
        cmap = plt.get_cmap("tab20")
        for _, i, x1, y1, x2, y2 in t[:, :6]:
            c = cmap(int(i) % 20)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=2))
            ax.text(x1, y1 - 3, str(int(i)), color=c, fontsize=8)
    ax.set_title(f"frame {frame_idx}")
