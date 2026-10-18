"""API REST mínima de tracking de vídeo.

Arranque local:   cd 11_produccion && uvicorn app.main:app --port 8000
Documentación:    http://localhost:8000/docs  (Swagger generado por FastAPI)
Prueba:           curl -F "file=@../data/people.mp4" "http://localhost:8000/track?max_frames=100"
"""
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .pipeline import PipelineConfig, VideoTracker

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

WEIGHTS = os.getenv("WEIGHTS", str(Path(__file__).resolve().parents[2] / "data" / "yolo11n.pt"))
state: dict = {}
# El modelo + estado del tracker no se pueden compartir entre peticiones simultáneas → un vídeo cada vez por
# proceso. Para escalar: varios workers/réplicas (uno por GPU) o una cola de trabajos (Celery/RQ/SQS).
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tracker"] = VideoTracker(PipelineConfig(weights=WEIGHTS))   # cargar el modelo UNA vez al arrancar
    yield
    state.clear()


app = FastAPI(title="Vision Tracking API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "tracker" in state}


@app.post("/track")
def track(file: UploadFile = File(...), max_frames: int | None = Query(None, ge=1, le=10_000)):
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(415, "Formato de vídeo no soportado")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            with _lock:
                result = state["tracker"].run(tmp.name, max_frames=max_frames)
        except ValueError as e:
            raise HTTPException(400, str(e))
    log.info("track: %s → %d tracks", file.filename, len({p.track_id for p in result.points}))
    return result.to_dict()
