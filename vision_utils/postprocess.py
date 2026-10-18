"""Post-procesado offline de tracks: interpolación de huecos y stitching de tracklets por apariencia."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def interpolate_tracks(mot, max_gap=20):
    out = [mot[:, :6]]
    for tid in np.unique(mot[:, 1]):
        t = mot[mot[:, 1] == tid]
        t = t[np.argsort(t[:, 0])]
        for a, b in zip(t[:-1], t[1:]):
            gap = int(b[0] - a[0])
            if 1 < gap <= max_gap:
                for k in range(1, gap):
                    w = k / gap
                    out.append(np.r_[a[0] + k, tid, (1 - w) * a[2:6] + w * b[2:6]][None])
    res = np.concatenate(out)
    return res[np.lexsort((res[:, 1], res[:, 0]))]


def stitch(mot, emb_of_row, max_gap=150, max_speed=8.0, app_thr=0.25):
    ids = np.unique(mot[:, 1]).astype(int)
    info = {}
    for tid in ids:
        m = mot[:, 1] == tid
        t = mot[m]; o = np.argsort(t[:, 0]); t = t[o]
        e = emb_of_row[m][o]
        hi = t[:, 6] >= 0.5 if t.shape[1] > 6 else np.ones(len(t), bool)   # solo detecciones fiables
        e = e[hi].mean(0) if hi.any() else e.mean(0)
        foot = lambda r: np.array([(r[2] + r[4]) / 2, r[5]])
        info[tid] = dict(start=t[0, 0], end=t[-1, 0], p0=foot(t[0]), p1=foot(t[-1]), emb=e / np.linalg.norm(e))
    C = np.full((len(ids), len(ids)), 1e5)
    for a, ta in enumerate(ids):
        for b, tb in enumerate(ids):
            A, B = info[ta], info[tb]
            gap = B["start"] - A["end"]
            if a == b or not (0 < gap <= max_gap):
                continue
            if np.linalg.norm(B["p0"] - A["p1"]) > max_speed * gap + 50:
                continue
            d = 1 - A["emb"] @ B["emb"]
            if d < app_thr:
                C[a, b] = d
    r, c = linear_sum_assignment(C)
    nxt = {ids[i]: ids[j] for i, j in zip(r, c) if C[i, j] < 1e5}
    # encadenar: A→B→C pasan todos a ser A
    prev = {v: k for k, v in nxt.items()}
    root = {}
    for tid in ids:
        x = tid
        while x in prev:
            x = prev[x]
        root[tid] = x
    out = mot.copy()
    out[:, 1] = [root[int(t)] for t in mot[:, 1]]
    return out, nxt
