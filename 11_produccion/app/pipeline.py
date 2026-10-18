"""Pipeline de vídeo → tracks estructurados. Pequeño, tipado y configurable: el estilo que se espera en producción."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    weights: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    conf: float = 0.1
    classes: tuple[int, ...] = (0,)
    device: str | None = None          # None → auto (cuda > mps > cpu)
    max_frames: int | None = None
    homography: list[list[float]] | None = None   # 3×3 imagen → campo (m), opcional


@dataclass
class TrackPoint:
    frame: int
    t: float
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    field_x: float | None = None
    field_y: float | None = None


@dataclass
class PipelineResult:
    fps_video: float
    n_frames: int
    processing_fps: float
    points: list[TrackPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        ids = sorted({p.track_id for p in self.points})
        return {"fps_video": self.fps_video, "n_frames": self.n_frames,
                "processing_fps": round(self.processing_fps, 2), "n_tracks": len(ids),
                "points": [asdict(p) for p in self.points]}


def auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_frames(path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"No se puede abrir el vídeo: {path}")
    try:
        n = 0
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            yield frame
    finally:
        cap.release()


def to_field(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


class VideoTracker:
    """Detector + tracker. El modelo se carga una vez (en el arranque del servicio), no por petición."""

    def __init__(self, cfg: PipelineConfig):
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = cfg.device or auto_device()
        self.model = YOLO(cfg.weights)
        self.H = np.array(cfg.homography, float) if cfg.homography else None
        log.info("modelo %s cargado en %s", cfg.weights, self.device)

    def run(self, video: str | Path, max_frames: int | None = None) -> PipelineResult:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        self.model.predictor = None          # reinicia el estado del tracker entre vídeos
        points: list[TrackPoint] = []
        t0, n = time.perf_counter(), 0
        for n, frame in enumerate(read_frames(video, max_frames or self.cfg.max_frames), start=1):
            r = self.model.track(frame, persist=True, tracker=self.cfg.tracker, imgsz=self.cfg.imgsz,
                                 conf=self.cfg.conf, classes=list(self.cfg.classes), device=self.device,
                                 verbose=False)[0]
            if r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            field_xy = None
            if self.H is not None:
                feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
                field_xy = to_field(self.H, feet)
            for k, (b, i, s) in enumerate(zip(boxes, ids, scores)):
                fx, fy = (float(field_xy[k, 0]), float(field_xy[k, 1])) if field_xy is not None else (None, None)
                points.append(TrackPoint(n, round(n / fps, 3), int(i), *map(float, b), float(s), fx, fy))
        dt = time.perf_counter() - t0
        log.info("procesados %d frames en %.1fs", n, dt)
        return PipelineResult(fps, n, n / dt if dt else 0.0, points)
