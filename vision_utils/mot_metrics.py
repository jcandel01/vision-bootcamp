"""Métricas de tracking multiobjeto implementadas desde cero (CLEAR-MOT, IDF1, HOTA).

Formato de entrada: arrays (K, >=6) con columnas frame, id, x1, y1, x2, y2.
- CLEAR (MOTA/MOTP/IDSW/Frag/MT/ML) sigue la definición de Bernardin & Stiefelhagen 2008.
- IDF1 sigue Ristani et al. 2016 (asignación global de identidades).
- HOTA sigue Luiten et al. 2020, con emparejamiento por frame vía húngaro sobre IoU
  (versión simplificada: TrackEval añade un término de asociación al coste; los valores
  pueden diferir ligeramente. Para publicar resultados usa TrackEval).
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _by_frame(arr):
    out = defaultdict(lambda: (np.zeros(0, int), np.zeros((0, 4))))
    for f in np.unique(arr[:, 0]).astype(int):
        r = arr[arr[:, 0] == f]
        out[f] = (r[:, 1].astype(int), r[:, 2:6].astype(np.float64))
    return out


def clear_mot(gt, pr, thr=0.5):
    G, P = _by_frame(gt), _by_frame(pr)
    frames = sorted(set(G) | set(P))
    last = {}                                   # gt_id -> último pred_id emparejado
    tracked = defaultdict(list)                 # gt_id -> lista de bool por frame en que existe
    tp = fp = fn = idsw = 0
    ious = []
    for f in frames:
        gi, gb = G[f]; pi, pb = P[f]
        M = iou_matrix(gb, pb)
        matches = []
        used_g, used_p = set(), set()
        # 1) mantener emparejamientos previos si siguen siendo válidos (continuidad, como en CLEAR)
        for a, g in enumerate(gi):
            if g in last:
                hit = np.where(pi == last[g])[0]
                if len(hit) and M[a, hit[0]] >= thr and hit[0] not in used_p:
                    matches.append((a, hit[0])); used_g.add(a); used_p.add(hit[0])
        # 2) húngaro sobre el resto
        rg = [a for a in range(len(gi)) if a not in used_g]
        rp = [b for b in range(len(pi)) if b not in used_p]
        if rg and rp:
            sub = M[np.ix_(rg, rp)]
            r, c = linear_sum_assignment(-sub)
            for i, j in zip(r, c):
                if sub[i, j] >= thr:
                    matches.append((rg[i], rp[j]))
        mg = {a for a, _ in matches}
        for a, b in matches:
            g, p = gi[a], pi[b]
            if g in last and last[g] != p:
                idsw += 1
            last[g] = p
            ious.append(M[a, b])
        for a, g in enumerate(gi):
            tracked[g].append(a in mg)
        tp += len(matches); fp += len(pi) - len(matches); fn += len(gi) - len(matches)
    n_gt = tp + fn
    # fragmentación = nº de veces que un track GT se recupera tras haberse perdido
    frag = sum(int(np.sum((np.array(v[1:]) == 1) & (np.array(v[:-1]) == 0))) for v in tracked.values())
    ratios = np.array([np.mean(v) for v in tracked.values()])
    return dict(MOTA=1 - (fn + fp + idsw) / max(n_gt, 1), MOTP=float(np.mean(ious)) if ious else 0.0,
                TP=tp, FP=fp, FN=fn, IDSW=idsw, Frag=frag,
                MT=int((ratios >= 0.8).sum()), ML=int((ratios <= 0.2).sum()), n_gt_ids=len(tracked))


def idf1(gt, pr, thr=0.5):
    G, P = _by_frame(gt), _by_frame(pr)
    gids, pids = np.unique(gt[:, 1]).astype(int), np.unique(pr[:, 1]).astype(int)
    gix = {g: i for i, g in enumerate(gids)}; pix = {p: i for i, p in enumerate(pids)}
    overlap = np.zeros((len(gids), len(pids)))
    for f in set(G) & set(P):
        gi, gb = G[f]; pi, pb = P[f]
        M = iou_matrix(gb, pb) >= thr
        for a, b in zip(*np.nonzero(M)):
            overlap[gix[gi[a]], pix[pi[b]]] += 1
    r, c = linear_sum_assignment(-overlap)
    idtp = overlap[r, c].sum()
    idfn, idfp = len(gt) - idtp, len(pr) - idtp
    return dict(IDF1=2 * idtp / max(2 * idtp + idfp + idfn, 1), IDP=idtp / max(len(pr), 1),
                IDR=idtp / max(len(gt), 1))


def hota(gt, pr, alphas=np.arange(0.05, 0.96, 0.05)):
    G, P = _by_frame(gt), _by_frame(pr)
    frames = sorted(set(G) | set(P))
    sims = {f: iou_matrix(G[f][1], P[f][1]) for f in frames}
    gt_count = defaultdict(int); pr_count = defaultdict(int)
    for f in frames:
        for g in G[f][0]: gt_count[g] += 1
        for p in P[f][0]: pr_count[p] += 1
    res = defaultdict(list)
    for al in alphas:
        tp = fn = fp = 0
        pairs = defaultdict(int)
        loc = []
        for f in frames:
            gi, _ = G[f]; pi, _ = P[f]; S = sims[f]
            m = []
            if S.size:
                r, c = linear_sum_assignment(-S)
                m = [(i, j) for i, j in zip(r, c) if S[i, j] >= al]
            for i, j in m:
                pairs[(gi[i], pi[j])] += 1
                loc.append(S[i, j])
            tp += len(m); fn += len(gi) - len(m); fp += len(pi) - len(m)
        det_a = tp / max(tp + fn + fp, 1)
        # AssA: para cada TP, cuánto se parece la trayectoria GT a la predicha con la que se emparejó
        ass = [n * n / (gt_count[g] + pr_count[p] - n) for (g, p), n in pairs.items()]  # n · A(c)
        ass_a = sum(ass) / max(tp, 1)
        res["HOTA"].append(np.sqrt(det_a * ass_a)); res["DetA"].append(det_a); res["AssA"].append(ass_a)
        res["LocA"].append(np.mean(loc) if loc else 0)
    return {k: float(np.mean(v)) for k, v in res.items()}


def evaluate(gt, pr, name="tracker"):
    out = {"tracker": name}
    out.update(hota(gt, pr)); out.update(idf1(gt, pr)); out.update(clear_mot(gt, pr))
    return out
