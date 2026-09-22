import numpy as np
import pytest
import soundfile as sf

from core.dsp import (
    extract_dsp_features,
    higuchi_fractal_dimension,
    katz_fractal_dimension,
    spectral_fractal_exponent,
)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def sine_wave():
    t = np.linspace(0.0, 2.0, 8192)
    return np.sin(2 * np.pi * 10.0 * t)


@pytest.fixture(scope="module")
def white_noise(rng):
    return rng.standard_normal(8192)


def test_higuchi_sine_is_smooth(sine_wave):
    d = higuchi_fractal_dimension(sine_wave)
    assert isinstance(d, float)
    assert 1.0 <= d <= 1.1


def test_higuchi_noise_is_rough(white_noise):
    d = higuchi_fractal_dimension(white_noise)
    assert 1.8 <= d <= 2.0


def test_higuchi_edge_cases():
    assert higuchi_fractal_dimension(np.array([])) == 1.0
    assert higuchi_fractal_dimension(np.zeros(1000)) == 1.0
    # signal shorter than 2*k_max should not raise
    d = higuchi_fractal_dimension(np.ones(20) + np.linspace(0, 1, 20))
    assert 1.0 <= d <= 2.0


def test_katz_fractal_dimension(sine_wave, white_noise):
    for sig in (sine_wave, white_noise):
        d = katz_fractal_dimension(sig)
        assert isinstance(d, float)
        assert d >= 1.0
    assert katz_fractal_dimension(np.zeros(100)) == 1.0


def test_spectral_fractal_exponent(sine_wave, white_noise):
    b_sine = spectral_fractal_exponent(sine_wave)
    b_noise = spectral_fractal_exponent(white_noise)
    assert isinstance(b_sine, float) and np.isfinite(b_sine)
    assert isinstance(b_noise, float) and np.isfinite(b_noise)
    # white noise has a flat spectrum (beta ~ 0); a pure tone is concentrated
    assert abs(b_noise) < 1.0


def test_extract_dsp_features(tmp_path, rng):
    sr = 22050
    t = np.linspace(0.0, 3.0, int(sr * 3))
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(len(t))
    path = tmp_path / "synthetic.wav"
    sf.write(path, signal, sr)

    feats = extract_dsp_features(str(path))

    expected_keys = {
        "duration", "sample_rate", "channels", "bitrate", "bpm",
        "rms_energy", "spectral_centroid", "spectral_rolloff",
        "zero_crossing_rate", "dynamic_range_db",
        "higuchi_fractal_dimension", "katz_fractal_dimension",
        "spectral_fractal_beta",
    }
    assert set(feats.keys()) == expected_keys

    for key, value in feats.items():
        assert not isinstance(value, np.generic), f"{key} is a NumPy scalar"
        assert np.isfinite(value), f"{key} is not finite"

    assert feats["duration"] == pytest.approx(3.0, abs=0.05)
    assert feats["sample_rate"] == 22050
    assert feats["channels"] == 1
    assert feats["bpm"] >= 0.0
    assert feats["rms_energy"] > 0.0
    assert 0.0 < feats["spectral_centroid"] < sr / 2
    assert feats["spectral_rolloff"] > 0.0
    assert feats["zero_crossing_rate"] > 0.0
    assert feats["dynamic_range_db"] > 0.0
    assert 1.0 <= feats["higuchi_fractal_dimension"] <= 2.0
    assert feats["katz_fractal_dimension"] >= 1.0
