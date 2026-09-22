import numpy as np
import pytest
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.pool import StaticPool

from core.database import (
    AudioFeatures,
    SemanticMetadata,
    Track,
    blob_to_vector,
    filter_tracks_sql,
    get_all_tracks,
    get_session,
    get_track_by_id,
    init_db,
    save_track_record,
    vector_to_blob,
)


@pytest.fixture()
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine_override=engine)
    yield engine
    engine.dispose()


def make_track_dicts(track_id, title, duration, bpm, higuchi, genre,
                     mood="calm", tags=None):
    track = {
        "track_id": track_id,
        "title": title,
        "artist": "Tester",
        "duration": duration,
        "sample_rate": 22050,
        "bitrate": 352800,
        "channels": 1,
        "file_path": f"data/sample_audio/{track_id}.wav",
    }
    features = {
        "bpm": bpm,
        "rms_energy": 0.2,
        "spectral_centroid": 1500.0,
        "spectral_rolloff": 4000.0,
        "zero_crossing_rate": 0.05,
        "dynamic_range_db": 12.0,
        "higuchi_fractal_dimension": higuchi,
        "katz_fractal_dimension": 1.3,
        "spectral_fractal_beta": 1.1,
    }
    semantic = {
        "primary_genre": genre,
        "mood": mood,
        "tags": tags if tags is not None else ["electronic"],
        "generated_description": f"Description of {title}",
    }
    return track, features, semantic


def test_init_db_creates_tables(db_engine):
    tables = set(inspect(db_engine).get_table_names())
    assert {"tracks", "audio_features", "semantic_metadata"} <= tables

    # init_db made the override engine the active one
    session = get_session()
    try:
        assert session.get_bind() is db_engine
    finally:
        session.close()


def test_vector_serialization_roundtrip():
    vec = np.array([0.1, -0.25, 3.75, 1e-6], dtype=np.float64)
    blob = vector_to_blob(vec)
    assert isinstance(blob, bytes)

    out = blob_to_vector(blob)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert np.allclose(out, vec)

    assert blob_to_vector(None) is None


def test_save_and_retrieve_track_with_embeddings(db_engine):
    track, features, semantic = make_track_dicts(
        "t-embed", "Embed Test", 120.5, 128.0, 1.62, "Techno",
        mood="driving", tags=["techno", "dark"],
    )
    audio_vec = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    text_vec = np.arange(4, dtype=np.float32) * 0.5

    returned_id = save_track_record(
        track, features, semantic,
        audio_embed=audio_vec, text_embed=text_vec,
    )
    assert returned_id == "t-embed"

    got = get_track_by_id("t-embed")
    assert got is not None
    assert got["title"] == "Embed Test"
    assert got["artist"] == "Tester"
    assert got["duration"] == 120.5
    assert got["sample_rate"] == 22050
    assert got["bitrate"] == 352800
    assert got["channels"] == 1
    assert got["bpm"] == 128.0
    assert got["higuchi_fractal_dimension"] == 1.62
    assert got["primary_genre"] == "Techno"
    assert got["mood"] == "driving"
    assert got["tags"] == ["techno", "dark"]
    assert got["generated_description"] == "Description of Embed Test"

    assert isinstance(got["audio_embedding"], np.ndarray)
    assert got["audio_embedding"].dtype == np.float32
    assert np.allclose(got["audio_embedding"], audio_vec)
    assert np.allclose(got["text_embedding"], text_vec)

    assert get_track_by_id("does-not-exist") is None


def test_save_track_record_defaults(db_engine):
    track, features, semantic = make_track_dicts("t-def", "Defaults", 60.0, 100.0, 1.4, "Pop")
    track = {k: v for k, v in track.items() if k not in ("artist", "channels")}
    save_track_record(track, features, semantic)

    got = get_track_by_id("t-def")
    assert got["artist"] == "Unknown Artist"
    assert got["channels"] == 1


def test_save_track_record_upserts(db_engine):
    track, features, semantic = make_track_dicts("t-up", "First", 100.0, 120.0, 1.5, "Rock")
    save_track_record(track, features, semantic)

    track2 = dict(track, title="Second")
    features2 = dict(features, bpm=125.0)
    save_track_record(track2, features2, semantic)

    with get_session() as session:
        track_count = session.scalar(select(func.count()).select_from(Track))
        feat_count = session.scalar(select(func.count()).select_from(AudioFeatures))
        sem_count = session.scalar(select(func.count()).select_from(SemanticMetadata))
    assert (track_count, feat_count, sem_count) == (1, 1, 1)

    got = get_track_by_id("t-up")
    assert got["title"] == "Second"
    assert got["bpm"] == 125.0


def test_cascade_delete(db_engine):
    track, features, semantic = make_track_dicts("t-del", "Doomed", 55.0, 100.0, 1.4, "Metal")
    save_track_record(track, features, semantic)

    with get_session() as session:
        session.delete(session.get(Track, "t-del"))
        session.commit()
        assert session.get(AudioFeatures, "t-del") is None
        assert session.get(SemanticMetadata, "t-del") is None

    assert get_track_by_id("t-del") is None


def test_get_all_tracks(db_engine):
    assert get_all_tracks() == []

    for tid in ("t-1", "t-2"):
        t, f, s = make_track_dicts(tid, tid, 100.0, 120.0, 1.5, "Rock")
        save_track_record(t, f, s)

    tracks = get_all_tracks()
    assert len(tracks) == 2
    assert {t["track_id"] for t in tracks} == {"t-1", "t-2"}
    for t in tracks:
        assert t["bpm"] == 120.0
        assert t["tags"] == ["electronic"]
        assert t["audio_embedding"] is None


def test_filter_tracks_sql(db_engine):
    specs = [
        # (id, duration, bpm, higuchi, genre, mood, tags)
        ("A", 90.0, 130.0, 1.7, "Industrial", "aggressive", ["heavy", "mechanical"]),
        ("B", 180.0, 80.0, 1.1, "Ambient", "dreamy", ["soft", "drone"]),
        ("C", 60.0, 140.0, 1.65, "Industrial", "energetic", ["rhythmic"]),
    ]
    for tid, dur, bpm, hfd, genre, mood, tags in specs:
        t, f, s = make_track_dicts(tid, f"Track {tid}", dur, bpm, hfd, genre,
                                   mood=mood, tags=tags)
        save_track_record(t, f, s)

    assert len(get_all_tracks()) == 3

    # duration_max AND min_fractal_dim -> A and C, B excluded
    result = filter_tracks_sql(duration_max=120, min_fractal_dim=1.5)
    assert {t["track_id"] for t in result} == {"A", "C"}

    # genre filter -> B only
    result = filter_tracks_sql(genre="Ambient")
    assert {t["track_id"] for t in result} == {"B"}

    # bpm range -> A and C
    result = filter_tracks_sql(bpm_min=100, bpm_max=145)
    assert {t["track_id"] for t in result} == {"A", "C"}

    # max_fractal_dim -> B only
    result = filter_tracks_sql(max_fractal_dim=1.5)
    assert {t["track_id"] for t in result} == {"B"}

    # duration range -> A only
    result = filter_tracks_sql(duration_min=70, duration_max=100)
    assert {t["track_id"] for t in result} == {"A"}

    # mood filter -> B only
    result = filter_tracks_sql(mood="dreamy")
    assert {t["track_id"] for t in result} == {"B"}

    # tag keyword search -> A only
    result = filter_tracks_sql(tag_keyword="mechanical")
    assert {t["track_id"] for t in result} == {"A"}

    # combined filters that match nothing
    assert filter_tracks_sql(genre="Ambient", bpm_min=120) == []

    # no filters -> everything
    assert len(filter_tracks_sql()) == 3
