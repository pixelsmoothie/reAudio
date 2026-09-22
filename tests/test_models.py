import numpy as np
import pytest
import soundfile as sf

from core.models import EMBED_DIM, AudioIntelligenceModel, generate_semantic_profile


@pytest.fixture(scope="module")
def model():
    return AudioIntelligenceModel(fallback_mode=True)


@pytest.fixture(scope="module")
def sine_wave():
    t = np.linspace(0.0, 2.0, 48000 * 2, endpoint=False)
    return 0.4 * np.sin(2 * np.pi * 220.0 * t)


def assert_unit_embedding(emb):
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32
    assert emb.shape == (EMBED_DIM,)
    assert emb.ndim == 1
    assert float(np.linalg.norm(emb.astype(np.float64))) == pytest.approx(1.0, abs=1e-5)


def test_embed_audio_from_array(model, sine_wave):
    assert_unit_embedding(model.embed_audio(sine_wave))


def test_embed_audio_2d_stereo(model, sine_wave):
    stereo = np.stack([sine_wave, 0.5 * sine_wave])  # (2, N)
    assert_unit_embedding(model.embed_audio(stereo, sr=48000))


def test_embed_audio_from_file(model, sine_wave, tmp_path):
    path = tmp_path / "sine.wav"
    sf.write(path, sine_wave, 48000)
    assert_unit_embedding(model.embed_audio(str(path)))


def test_embed_audio_deterministic(model, sine_wave):
    a = model.embed_audio(sine_wave)
    b = model.embed_audio(sine_wave)
    assert np.array_equal(a, b)


def test_embed_text(model):
    emb = model.embed_text("dark pounding industrial metal with distorted drums")
    assert_unit_embedding(emb)

    other = model.embed_text("soft ambient wind chimes at dawn")
    assert not np.allclose(emb, other)


def test_embed_text_deterministic(model):
    text = "neon city night drive"
    assert np.array_equal(model.embed_text(text), model.embed_text(text))


def test_compute_similarity(model, sine_wave):
    audio_emb = model.embed_audio(sine_wave)
    sim = model.compute_similarity("heavy distorted guitars", audio_emb)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_compute_similarity_self_is_one(model):
    emb = model.embed_text("an identical embedding vector")
    sim = model.compute_similarity("an identical embedding vector", emb)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_auto_fallback_when_model_load_fails(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def boom(*args, **kwargs):
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(transformers.ClapModel, "from_pretrained", boom)
    monkeypatch.setattr(transformers.ClapProcessor, "from_pretrained", boom)

    m = AudioIntelligenceModel()
    assert m.fallback_mode is True
    assert_unit_embedding(m.embed_text("graceful degradation"))
    assert_unit_embedding(m.embed_audio(np.zeros(4800)))


def _profile(hfd, bpm, rms, centroid=1500.0, beta=1.0):
    feats = {
        "higuchi_fractal_dimension": hfd,
        "bpm": bpm,
        "rms_energy": rms,
        "spectral_centroid": centroid,
        "spectral_fractal_beta": beta,
    }
    return generate_semantic_profile(feats)


def test_profile_high_fractal_industrial():
    profile = _profile(hfd=1.75, bpm=150.0, rms=0.25)
    assert set(profile) == {"primary_genre", "mood", "tags", "generated_description"}
    assert profile["primary_genre"] in ("Industrial Electronic", "Glitch / Metal")
    assert profile["mood"] == "Aggressive & Chaotic"
    assert "Boss Fight" in profile["tags"]
    assert "BPM" in profile["generated_description"]


def test_profile_low_fractal_ambient():
    profile = _profile(hfd=1.12, bpm=70.0, rms=0.03)
    assert profile["primary_genre"] in ("Ambient Drone", "Atmospheric")
    assert profile["mood"] == "Calm & Atmospheric"
    assert {"Dungeon", "Exploration"} & set(profile["tags"])


def test_profile_synthwave():
    profile = _profile(hfd=1.40, bpm=128.0, rms=0.20)
    assert profile["primary_genre"] == "Cyberpunk Synthwave"
    assert profile["mood"] == "Driving & Tense"
    assert "Combat" in profile["tags"]


def test_profile_default_acoustic():
    profile = _profile(hfd=1.35, bpm=100.0, rms=0.05)
    assert profile["primary_genre"] == "Acoustic / Orchestral"
    assert profile["mood"] == "Organic & Melodic"
    assert "Village" in profile["tags"]


def test_profile_tags_are_plain_strings():
    profile = _profile(hfd=1.7, bpm=150.0, rms=0.3)
    assert all(isinstance(t, str) for t in profile["tags"])
    assert all(isinstance(v, str) for v in profile.values() if not isinstance(v, list))
