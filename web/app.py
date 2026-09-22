"""AudioMind web server.

FastAPI REST API over the retrieval/recommender stack plus a static
Spotify-style frontend served from web/static.

Set AUDIOMIND_FORCE_FALLBACK=1 to skip the CLAP download and use the
deterministic fallback embedder (useful for offline demos/tests).
"""

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core.database import (
    get_all_tracks,
    get_database_dump,
    get_track_by_id,
    save_track_record,
)
from core.dsp import extract_dsp_features
from core.models import AudioIntelligenceModel, generate_semantic_profile
from core.retrieval import HybridRetrievalEngine, Recommender

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
}

app = FastAPI(title="AudioMind API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_force_fallback = os.getenv("AUDIOMIND_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
model = AudioIntelligenceModel(fallback_mode=_force_fallback)
engine = HybridRetrievalEngine(model=model)
recommender = Recommender()


class SearchRequest(BaseModel):
    query: str
    duration_max: Optional[float] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    min_fractal_dim: Optional[float] = None
    max_fractal_dim: Optional[float] = None
    genre: Optional[str] = None
    top_k: int = 8


def _public_track(row: dict) -> dict:
    """JSON-safe track dict: strip embedding blobs, ISO-encode datetimes."""
    out = {
        k: v for k, v in row.items()
        if k not in ("audio_embedding", "text_embedding")
    }
    if out.get("created_at") is not None:
        out["created_at"] = out["created_at"].isoformat()
    return out


@app.get("/api/tracks")
def list_tracks():
    return [_public_track(t) for t in get_all_tracks()]


@app.post("/api/search")
def search(request: SearchRequest):
    filters = {
        k: v for k, v in request.model_dump().items()
        if k not in ("query", "top_k") and v is not None
    }
    t0 = time.perf_counter()
    try:
        results = engine.search(request.query, filters=filters, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Search failed: {exc}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    total_corpus = len(get_all_tracks())
    # Attach pipeline telemetry to each track for UI inspection while preserving list schema
    for r in results:
        r["pipeline_telemetry"] = {
            "total_corpus": total_corpus,
            "candidates_returned": len(results),
            "latency_ms": latency_ms,
            "top_k": request.top_k,
            "fusion_weights": {"semantic": 0.70, "acoustic": 0.30},
            "embedding_dim": 512,
            "active_filters": filters,
        }
    return [_public_track(r) for r in results]


@app.get("/api/database/inspect")
def inspect_database():
    """Return raw database tables and relational schema inspection data."""
    return get_database_dump()


@app.get("/api/pipeline/stats")
def pipeline_stats():
    total_tracks = len(get_all_tracks())
    return {
        "engine": "reAudio Hybrid 2-Stage Retrieval",
        "total_indexed_tracks": total_tracks,
        "sample_rate_hz": 22050,
        "embedding_model": "CLAP (Contrastive Language-Audio Pretraining)",
        "embedding_dim": 512,
        "fusion_weights": {"semantic_vector": 0.70, "acoustic_alignment": 0.30},
        "dsp_metrics": [
            {"name": "Higuchi Fractal Dimension (HFD)", "domain": "Non-linear time-series complexity", "range": "1.0 to 2.0", "formula": "OLS slope of ln(L(k)) vs ln(1/k), Higuchi 1988"},
            {"name": "Spectral Centroid", "domain": "Frequency center of mass (brightness)", "unit": "Hz", "formula": "sum(f * S(f)) / sum(S(f))"},
            {"name": "Spectral Fractal Exponent", "domain": "1/f^beta spectral power decay", "unit": "dimensionless", "formula": "Welch PSD log-log regression"},
            {"name": "Zero-Crossing Rate", "domain": "High-frequency sign transitions", "unit": "rate", "formula": "librosa.feature.zero_crossing_rate"},
            {"name": "Tempo & Beat Tracking", "domain": "Rhythmic periodicity", "unit": "BPM", "formula": "Dynamic programming beat tracker"},
        ],
        "pipeline_stages": [
            {"step": 1, "name": "Relational SQL Pruning", "type": "Deterministic Pre-Filter", "desc": "Enforces hard boundaries (BPM bounds, max duration, Higuchi limits) using SQLite indexes."},
            {"step": 2, "name": "Multimodal Vector Re-Ranking", "type": "Neural Representation", "desc": "Projects cleaned query into 512-D latent space and computes cosine dot product with pre-indexed audio embeddings."},
            {"step": 3, "name": "Acoustic Score Fusion", "type": "Multi-Objective Ranking", "desc": "Calculates final score = 70% Semantic Vector Similarity + 30% Physical Acoustic Constraint Alignment."},
            {"step": 4, "name": "Grounded Diagnostics", "type": "Explainable AI", "desc": "Compares retrieved DSP properties against user query intent to generate human-readable physical match reasons."}
        ]
    }


@app.get("/api/recommend/{track_id}")
def recommend(track_id: str):
    if get_track_by_id(track_id) is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
    return [_public_track(t) for t in recommender.recommend_similar(track_id, top_k=4)]


@app.get("/api/presets")
def presets():
    return recommender.get_game_dev_presets()


@app.get("/api/audio/{track_id}")
def audio(track_id: str):
    row = get_track_by_id(track_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
    path = Path(row["file_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    media_type = AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "audio/wav")
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/api/upload")
def upload_track(file: UploadFile = File(...),
                 title: str = Form(...),
                 artist: str = Form("Unknown")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. "
                   f"Allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))}",
        )

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "track"
    track_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    dest = config.AUDIO_DIR / f"{track_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        features = extract_dsp_features(str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not decode audio file: {exc}")

    profile = generate_semantic_profile(features)
    audio_embed = model.embed_audio(str(dest))
    semantic_text = (f"{profile['primary_genre']} {profile['mood']} "
                     f"{' '.join(profile['tags'])} {title} {artist}")
    text_embed = model.embed_text(semantic_text)

    track = {
        "track_id": track_id,
        "title": title,
        "artist": artist,
        "duration": features["duration"],
        "sample_rate": features["sample_rate"],
        "bitrate": features["bitrate"],
        "channels": features["channels"],
        "file_path": str(dest),
    }
    save_track_record(track, features, profile,
                      audio_embed=audio_embed, text_embed=text_embed)
    return _public_track(get_track_by_id(track_id))


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
