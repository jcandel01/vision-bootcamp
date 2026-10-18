"""Tests de la API. Ejecutar desde 11_produccion/:  python -m pytest -q"""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import to_field

VIDEO = Path(__file__).resolve().parents[2] / "data" / "people.mp4"


def test_to_field_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(to_field(np.eye(3), pts), pts)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # el "with" ejecuta el lifespan (carga del modelo)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_rejects_non_video(client):
    r = client.post("/track", files={"file": ("a.txt", b"hola", "text/plain")})
    assert r.status_code == 415


@pytest.mark.skipif(not VIDEO.exists(), reason="falta data/people.mp4 (ejecuta la celda SETUP del notebook 11)")
def test_track_video(client):
    with open(VIDEO, "rb") as f:
        r = client.post("/track?max_frames=20", files={"file": ("people.mp4", f, "video/mp4")})
    body = r.json()
    assert r.status_code == 200
    assert body["n_frames"] == 20 and body["n_tracks"] > 0
    p = body["points"][0]
    assert {"frame", "track_id", "x1", "y1", "x2", "y2", "score"} <= p.keys()
