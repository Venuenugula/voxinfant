"""FastAPI service for VoxInfant.

Endpoints:
    GET  /health    -> liveness + whether the model is loaded
    POST /predict    -> multipart upload of a .wav/.mp3 -> prediction JSON

Run:
    uvicorn api.app:app --reload --port 8000
    # (from the project root, with `src` on PYTHONPATH — see README)

The model is loaded lazily on the first /predict call so the server starts even
before artifacts exist. If artifacts are missing you get a clear 503.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from voxinfant.config import MODELS_DIR
from voxinfant.inference import CryAnalyzer

app = FastAPI(
    title="VoxInfant API",
    version="0.1.0",
    description="Experimental infant-cry analysis. Research preview — not a medical device.",
)

# Permissive CORS for the Next.js dev frontend; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXT = {".wav", ".mp3"}
_analyzer: CryAnalyzer | None = None


def _get_analyzer() -> CryAnalyzer:
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = CryAnalyzer(MODELS_DIR).load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _analyzer


@app.get("/health")
def health() -> dict:
    loaded = _analyzer is not None and _analyzer.is_loaded
    artifacts_present = os.path.exists(os.path.join(MODELS_DIR, "voxinfant_ensemble.pkl"))
    return {
        "status": "ok",
        "model_loaded": loaded,
        "artifacts_present": artifacts_present,
        "version": app.version,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use .wav or .mp3.")

    analyzer = _get_analyzer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        prediction = analyzer.predict(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    result = prediction.to_dict()
    result["disclaimer"] = "Experimental research output. Not a substitute for professional medical advice."
    return result
