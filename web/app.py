"""AudioMind web server.

FastAPI REST API over the retrieval/recommender stack plus a static
Spotify-style frontend served from web/static.

Set AUDIOMIND_FORCE_FALLBACK=1 to skip the CLAP download and use the
deterministic fallback embedder (useful for offline demos/tests).
"""

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core.database import get_all_tracks, get_track_by_id, save_track_record
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
    try:
        results = engine.search(request.query, filters=filters, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Search failed: {exc}")
    return [_public_track(r) for r in results]


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
