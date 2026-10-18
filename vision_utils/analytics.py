"""Analítica temporal de trayectorias (soluciones del notebook 09)."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 25


def kinematics(xy, fps=FPS, window=25):
    """xy: (T,2) con NaN en huecos → DataFrame t, x, y, vx, vy, speed_kmh (Savitzky–Golay)."""
    df = pd.DataFrame(xy, columns=["x", "y"]).interpolate(limit_direction="both")
    out = pd.DataFrame({"t": np.arange(len(df)) / fps})
    window = min(window, len(df) if len(df) % 2 else len(df) - 1)   # ventana impar ≤ longitud del track
    if window < 5:                                                    # track demasiado corto para suavizar
        for c in ["x", "y"]:
            out[c] = df[c].values
            out["v" + c] = np.gradient(df[c].values) * fps if len(df) > 1 else 0.0
        out["speed_kmh"] = np.hypot(out.vx, out.vy) * 3.6
        return out
    for c in ["x", "y"]:
        out[c] = savgol_filter(df[c].values, window, 2)
        out["v" + c] = savgol_filter(df[c].values, window, 2, deriv=1) * fps
    out["speed_kmh"] = np.hypot(out.vx, out.vy) * 3.6
    return out


def detect_events(signal, enter, exit, min_len):
    events, active, start = [], False, 0
    for i, s in enumerate(signal):
        if not active and s < enter:
            active, start = True, i
        elif active and s > exit:
            active = False
            if i - start >= min_len:
                events.append((start, i))
    if active and len(signal) - start >= min_len:
        events.append((start, len(signal)))
    return events


def player_summary(k, sprint_kmh=24, min_sprint_s=1.0, fps=FPS):
    fast = (k.speed_kmh > sprint_kmh).values
    # nº de sprints = tramos contiguos por encima del umbral con duración mínima
    edges = np.diff(np.r_[0, fast.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    n_sprints = int(((ends - starts) / fps >= min_sprint_s).sum())
    return {"distancia_m": round(k.speed_kmh.sum() / 3.6 / fps, 1), "vel_max_kmh": round(k.speed_kmh.max(), 1),
            "vel_media_kmh": round(k.speed_kmh.mean(), 1), "n_sprints": n_sprints,
            "dist_alta_intensidad_m": round(k.speed_kmh[k.speed_kmh > 19.8].sum() / 3.6 / fps, 1)}
