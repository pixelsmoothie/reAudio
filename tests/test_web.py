import io
import os

os.environ.setdefault("AUDIOMIND_FORCE_FALLBACK", "1")

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import config
from core.database import init_db, save_track_record
from core.models import AudioIntelligenceModel
from web.app import app


SEED_TRACKS = [
    {
        "track_id": "boss", "title": "Iron Tyrant", "artist": "Vortex Synth",
        "genre": "Industrial", "mood": "Aggressive & Chaotic",
        "tags": ["Boss Fight", "Industrial", "Distortion"],
        "embed": "heavy aggressive industrial boss battle chaotic distorted metal",
        "duration": 90.0, "bpm": 135.0, "hfd": 1.75, "rms": 0.25,
    },
    {
        "track_id": "ambient", "title": "Sunken Cavern", "artist": "Shadow Echo",
        "genre": "Ambient", "mood": "Calm & Atmospheric",
        "tags": ["Dungeon", "Drone", "Eerie"],
        "embed": "smooth calm ambient drone peaceful eerie exploration dungeon",
        "duration": 200.0, "bpm": 65.0, "hfd": 1.12, "rms": 0.03,
    },
    {
        "track_id": "village", "title": "Willowbrook", "artist": "Lute & Leaf",
        "genre": "Acoustic", "mood": "Organic & Melodic",
        "tags": ["Village", "Acoustic", "Harmonic"],
        "embed": "warm organic acoustic folk village gentle peaceful melodic",
        "duration": 80.0, "bpm": 92.0, "hfd": 1.18, "rms": 0.06,
    },
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh in-memory DB per test, seeded with 3 tracks; uploads are
    redirected to tmp_path so the real data/ folder is untouched."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine_override=engine)
    model = AudioIntelligenceModel(fallback_mode=True)

    # a real WAV on disk so /api/audio can stream it
    boss_wav = tmp_path / "iron_tyrant_boss.wav"
    t = np.linspace(0.0, 2.0, 22050 * 2, endpoint=False)
    rng = np.random.default_rng(3)
    sf.write(boss_wav, 0.5 * np.sin(2 * np.pi * 110 * t)
             + 0.05 * rng.standard_normal(len(t)), 22050)

    for spec in SEED_TRACKS:
        track = {
            "track_id": spec["track_id"],
            "title": spec["title"],
            "artist": spec["artist"],
            "duration": spec["duration"],
            "sample_rate": 22050,
            "bitrate": 352800,
            "channels": 1,
            "file_path": str(boss_wav),
        }
        features = {
            "bpm": spec["bpm"],
            "rms_energy": spec["rms"],
            "spectral_centroid": 2000.0,
            "spectral_rolloff": 5000.0,
            "zero_crossing_rate": 0.05,
            "dynamic_range_db": 14.0,
            "higuchi_fractal_dimension": spec["hfd"],
            "katz_fractal_dimension": 1.3,
            "spectral_fractal_beta": 1.2,
        }
        semantic = {
            "primary_genre": spec["genre"],
            "mood": spec["mood"],
            "tags": spec["tags"],
            "generated_description": f"Seed description for {spec['title']}.",
        }
        save_track_record(track, features, semantic,
                          audio_embed=model.embed_text(spec["embed"]),
                          text_embed=model.embed_text(spec["embed"] + " soundtrack"))

    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def test_list_tracks(client):
    res = client.get("/api/tracks")
    assert res.status_code == 200
    tracks = res.json()
    assert isinstance(tracks, list) and len(tracks) == 3
    for track in tracks:
        for key in ("track_id", "title", "artist", "duration", "bpm",
                    "higuchi_fractal_dimension", "primary_genre", "mood", "tags"):
            assert key in track


def test_presets(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()
    assert set(presets) == {"Main Menu", "Exploration", "Village", "Dungeon",
                            "Boss Fight", "Credits"}
    for preset in presets.values():
        assert preset["query"] and preset["description"]


def test_search_returns_ranked_results_with_explanations(client):
    res = client.post("/api/search",
                      json={"query": "heavy aggressive industrial boss battle", "top_k": 3})
    assert res.status_code == 200
    results = res.json()
    assert results, "expected at least one search hit"
    assert results[0]["track_id"] == "boss"

    scores = [r["match_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert 0.0 <= r["match_score"] <= 100.0
        explanation = r["explanation"]
        assert explanation["score_pct"] == r["match_score"]
        assert explanation["summary"]
        assert isinstance(explanation["insights"], list) and explanation["insights"]


def test_search_with_relational_filters(client):
    res = client.post("/api/search", json={"query": "any music", "duration_max": 100})
    assert res.status_code == 200
    ids = {r["track_id"] for r in res.json()}
    assert ids == {"boss", "village"}  # ambient (200s) pruned by duration


def test_recommend_existing_and_missing(client):
    res = client.get("/api/recommend/boss")
    assert res.status_code == 200
    similar = res.json()
    ids = [r["track_id"] for r in similar]
    assert "boss" not in ids
    assert len(ids) == 2
    for r in similar:
        assert "similarity" in r

    assert client.get("/api/recommend/does-not-exist").status_code == 404


def test_audio_streaming(client):
    res = client.get("/api/audio/boss")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/wav")
    assert len(res.content) > 1000
    assert client.get("/api/audio/missing-track").status_code == 404


def test_upload_indexes_synthetic_wav(client):
    buf = io.BytesIO()
    sr = 22050
    t = np.arange(sr * 3) / sr
    rng = np.random.default_rng(5)
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.08 * rng.standard_normal(len(t))
    sf.write(buf, signal, sr, format="WAV")
    buf.seek(0)

    res = client.post(
        "/api/upload",
        files={"file": ("synthetic_tone.wav", buf, "audio/wav")},
        data={"title": "Synthetic Tone", "artist": "Pytest Suite"},
    )
    assert res.status_code == 200, res.text
    track = res.json()
    assert track["title"] == "Synthetic Tone"
    assert track["artist"] == "Pytest Suite"
    assert track["duration"] == pytest.approx(3.0, abs=0.1)
    assert 1.0 <= track["higuchi_fractal_dimension"] <= 2.0
    assert track["bpm"] >= 0.0
    assert track["primary_genre"] and track["mood"]
    assert isinstance(track["tags"], list) and track["tags"]
    assert track["generated_description"]

    # indexed and fully retrievable
    listing = client.get("/api/tracks").json()
    assert any(t["track_id"] == track["track_id"] for t in listing)
    audio = client.get(f"/api/audio/{track['track_id']}")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")


def test_upload_rejects_unsupported_extension(client):
    res = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
        data={"title": "Bad File"},
    )
    assert res.status_code == 400
