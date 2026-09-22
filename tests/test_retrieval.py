import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from core.database import init_db, save_track_record
from core.explainer import RAGExplainer
from core.models import AudioIntelligenceModel
from core.retrieval import HybridRetrievalEngine, QueryParser, Recommender


TRACK_SPECS = [
    {
        "track_id": "boss", "title": "Iron Tyrant Boss Theme",
        "duration": 90.0, "bpm": 135.0, "higuchi": 1.75, "rms": 0.25,
        "genre": "Industrial", "mood": "Aggressive & Chaotic",
        "tags": ["Boss Fight", "Industrial", "Distortion"],
        "embed": "heavy aggressive industrial boss battle chaotic distorted metal",
    },
    {
        "track_id": "ambient", "title": "Sunken Cavern Drone",
        "duration": 200.0, "bpm": 65.0, "higuchi": 1.12, "rms": 0.03,
        "genre": "Ambient", "mood": "Calm & Atmospheric",
        "tags": ["Exploration", "Dungeon", "Eerie"],
        "embed": "smooth calm ambient drone peaceful eerie exploration dungeon",
    },
    {
        "track_id": "synth", "title": "Neon Highway Combat",
        "duration": 110.0, "bpm": 124.0, "higuchi": 1.45, "rms": 0.20,
        "genre": "Synthwave", "mood": "Driving & Tense",
        "tags": ["Combat", "Night Drive", "Retro Synth"],
        "embed": "heavy driving synthwave combat action retro synth aggressive battle neon",
    },
    {
        "track_id": "village", "title": "Willowbrook Morning",
        "duration": 80.0, "bpm": 90.0, "higuchi": 1.18, "rms": 0.06,
        "genre": "Acoustic", "mood": "Organic & Melodic",
        "tags": ["Village", "Harmonic", "Main Menu"],
        "embed": "warm organic acoustic folk village gentle peaceful melodic harmonic",
    },
]


@pytest.fixture()
def setup():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine_override=engine)
    model = AudioIntelligenceModel(fallback_mode=True)
    retrieval_engine = HybridRetrievalEngine(model=model)
    recommender = Recommender()

    for spec in TRACK_SPECS:
        track = {
            "track_id": spec["track_id"],
            "title": spec["title"],
            "artist": "Test Artist",
            "duration": spec["duration"],
            "sample_rate": 22050,
            "bitrate": 352800,
            "channels": 1,
            "file_path": f"data/sample_audio/{spec['track_id']}.wav",
        }
        features = {
            "bpm": spec["bpm"],
            "rms_energy": spec["rms"],
            "spectral_centroid": 2000.0,
            "spectral_rolloff": 5000.0,
            "zero_crossing_rate": 0.04,
            "dynamic_range_db": 14.0,
            "higuchi_fractal_dimension": spec["higuchi"],
            "katz_fractal_dimension": 1.3,
            "spectral_fractal_beta": 1.2,
        }
        semantic = {
            "primary_genre": spec["genre"],
            "mood": spec["mood"],
            "tags": spec["tags"],
            "generated_description": f"Auto description for {spec['title']}",
        }
        save_track_record(
            track, features, semantic,
            audio_embed=model.embed_text(spec["embed"]),
        )

    yield model, retrieval_engine, recommender
    engine.dispose()


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

@pytest.fixture()
def parser():
    return QueryParser()


def test_parse_boss_fight_duration(parser):
    parsed = parser.parse_query("intense boss fight under 2 minutes")
    assert parsed["duration_max"] == 120.0
    assert parsed["min_fractal_dim"] == 1.50
    assert parsed["genre"] is None
    assert "boss fight" in parsed["cleaned_query"]
    assert "under 2 minutes" not in parsed["cleaned_query"]


def test_parse_calm_ambient_village(parser):
    parsed = parser.parse_query("calm ambient village")
    assert parsed["max_fractal_dim"] == 1.30
    assert parsed["genre"] == "Ambient"
    assert parsed["duration_max"] is None
    assert parsed["bpm_min"] is None and parsed["bpm_max"] is None


def test_parse_duration_variants(parser):
    assert parser.parse_query("epic music under 120s")["duration_max"] == 120.0
    assert parser.parse_query("epic music less than 90s")["duration_max"] == 90.0
    assert parser.parse_query("epic music under 1 min")["duration_max"] == 60.0
    assert parser.parse_query("epic music below 1.5 minutes")["duration_max"] == 90.0


def test_parse_bpm_constraints(parser):
    parsed = parser.parse_query("fast tempo synthwave over 120 bpm")
    assert parsed["bpm_min"] == 120.0
    assert parsed["genre"] == "Synthwave"
    assert "fast tempo" not in parsed["cleaned_query"]
    assert "over 120 bpm" not in parsed["cleaned_query"]

    parsed = parser.parse_query("slow tempo peaceful drone under 90s")
    assert parsed["bpm_max"] == 95.0
    assert parsed["duration_max"] == 90.0
    assert parsed["max_fractal_dim"] == 1.30
    assert parsed["genre"] == "Ambient"
    assert "slow tempo" not in parsed["cleaned_query"]

    parsed = parser.parse_query("something under 100 bpm")
    assert parsed["bpm_max"] == 100.0
    assert parsed["duration_max"] is None  # "under 100 bpm" is tempo, not duration


def test_parse_no_constraints(parser):
    parsed = parser.parse_query("emotional melodic theme")
    assert parsed["duration_max"] is None
    assert parsed["bpm_min"] is None
    assert parsed["bpm_max"] is None
    assert parsed["min_fractal_dim"] is None
    assert parsed["max_fractal_dim"] is None
    assert parsed["genre"] is None
    assert parsed["cleaned_query"] == "emotional melodic theme"


# ---------------------------------------------------------------------------
# HybridRetrievalEngine
# ---------------------------------------------------------------------------

def test_search_boss_query_strict_filter(setup):
    _, engine, _ = setup
    results = engine.search("intense boss fight under 2 minutes", top_k=3)

    # min_fractal 1.50 + duration_max 120 -> only the boss track qualifies
    assert len(results) == 1
    top = results[0]
    assert top["track_id"] == "boss"
    assert 0.0 <= top["match_score"] <= 100.0
    assert top["acoustic_alignment"] == pytest.approx(1.0)

    explanation = top["explanation"]
    assert explanation["score_pct"] == top["match_score"]
    assert isinstance(explanation["insights"], list) and explanation["insights"]


def test_search_semantic_ranking_smooth(setup):
    _, engine, _ = setup
    results = engine.search("dark eerie mysterious calm peaceful atmosphere", top_k=4)

    # max_fractal 1.30 -> ambient (1.12) and village (1.18) survive pruning;
    # shared vocabulary ranks ambient first
    ids = [r["track_id"] for r in results]
    assert {"ambient", "village"} <= set(ids)
    assert results[0]["track_id"] == "ambient"

    scores = [r["match_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_explicit_filter_override(setup):
    _, engine, _ = setup
    results = engine.search("any music at all", filters={"genre": "Ambient"})
    assert len(results) == 1
    assert results[0]["track_id"] == "ambient"


def test_search_empty_strict_falls_back_to_all(setup):
    _, engine, _ = setup
    # "boss fight" + "drone" set contradictory fractal constraints ->
    # strict SQL filter matches nothing -> engine falls back to all tracks
    results = engine.search("chaotic distorted glitch smooth peaceful drone", top_k=10)
    assert len(results) == 4
    scores = [r["match_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all("explanation" in r for r in results)


def test_search_top_k_respected(setup):
    _, engine, _ = setup
    results = engine.search("music", top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def test_recommend_similar_to_boss(setup):
    _, _, recommender = setup
    results = recommender.recommend_similar("boss", top_k=3)
    ids = [r["track_id"] for r in results]
    assert "boss" not in ids
    assert len(ids) == 3
    assert ids[0] == "synth"  # shares heavy/aggressive/battle vocabulary
    assert results[0]["similarity"] > 0.0

    top_two = recommender.recommend_similar("boss", top_k=2)
    assert [r["track_id"] for r in top_two] == ids[:2]


def test_recommend_similar_to_village(setup):
    _, _, recommender = setup
    results = recommender.recommend_similar("village", top_k=4)
    ids = [r["track_id"] for r in results]
    assert "village" not in ids
    assert ids[0] == "ambient"  # shares "peaceful" vocabulary


def test_recommend_similar_missing_track(setup):
    _, _, recommender = setup
    assert recommender.recommend_similar("ghost-track") == []


def test_game_dev_presets(setup):
    _, engine, recommender = setup
    presets = recommender.get_game_dev_presets()
    assert set(presets) == {"Main Menu", "Exploration", "Village", "Dungeon",
                            "Boss Fight", "Credits"}
    for preset in presets.values():
        assert preset["query"] and preset["description"]

    # preset queries run end-to-end through the engine
    boss_hits = engine.search(presets["Boss Fight"]["query"])
    assert boss_hits[0]["track_id"] == "boss"
    menu_hits = engine.search(presets["Main Menu"]["query"])
    assert menu_hits[0]["track_id"] == "ambient"
    village_hits = engine.search(presets["Village"]["query"])
    assert village_hits[0]["track_id"] == "village"


# ---------------------------------------------------------------------------
# RAGExplainer
# ---------------------------------------------------------------------------

def test_rag_explainer_full():
    track = {
        "track_id": "boss", "title": "Iron Tyrant",
        "higuchi_fractal_dimension": 1.72, "bpm": 130.0, "rms_energy": 0.22,
        "duration": 95.0, "primary_genre": "Industrial", "similarity": 0.61,
    }
    parsed = {
        "cleaned_query": "intense boss fight", "duration_max": 120.0,
        "bpm_min": None, "bpm_max": None, "min_fractal_dim": 1.5,
        "max_fractal_dim": None, "genre": None,
    }
    out = RAGExplainer().explain(
        "intense boss fight under 2 minutes", track, 87.54, parsed
    )
    assert set(out) == {"score_pct", "summary", "insights"}
    assert out["score_pct"] == 87.5
    assert "Iron Tyrant" in out["summary"]
    assert any("Higuchi" in i for i in out["insights"])
    assert any("BPM" in i for i in out["insights"])
    assert any("duration" in i.lower() for i in out["insights"])


def test_rag_explainer_minimal_track():
    track = {"track_id": "x", "title": "Unknown", "bpm": 100.0,
             "rms_energy": 0.10, "duration": 75.0}
    out = RAGExplainer().explain("something atmospheric", track, 42.0, None)
    assert out["score_pct"] == 42.0
    assert isinstance(out["summary"], str) and out["summary"]
    assert len(out["insights"]) >= 2  # tempo/energy + semantic alignment
