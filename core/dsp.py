"""AudioMind DSP & Fractal Analysis Engine.

Extracts audio features and computes fractal dimensions (Higuchi, Katz)
and the 1/f^beta spectral exponent for audio signals.
"""

import numpy as np
from scipy.signal import welch
import librosa
import soundfile as sf


def higuchi_fractal_dimension(signal: np.ndarray, k_max: int = 16) -> float:
    """Calculate the Higuchi Fractal Dimension (HFD) of a 1D audio waveform.

    HFD measures acoustic roughness and geometric complexity:
      ~1.0: smooth, pure tone or calm ambient
      ~1.5: standard pop/rock/synthwave
      ~1.8 - 2.0: highly distorted, chaotic industrial or noise
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)

    # Base cases: empty audio or dead silence flatline
    if n == 0 or np.allclose(signal, signal[0]):
        return 1.0

    # Ensure step size k does not exceed half the signal length
    k_max = min(k_max, max(1, n // 2))
    if k_max < 1:
        return 1.0

    ln_lengths = np.zeros(k_max)
    ln_inv_k = np.zeros(k_max)

    # Analyze signal curve length across step sizes k = 1 to k_max
    for idx, k in enumerate(range(1, k_max + 1)):
        sub_lengths = []
        for m in range(k):
            # Take every k-th sample starting at offset m
            sub_series = signal[m::k]
            num_steps = len(sub_series) - 1
            if num_steps < 1:
                continue

            # Total vertical distance traveled by the waveform
            vertical_distance = np.sum(np.abs(np.diff(sub_series)))

            # Higuchi (1988) normalized curve length formula
            norm_factor = (n - 1) / (num_steps * (k ** 2))
            sub_lengths.append(vertical_distance * norm_factor)

        # Average length across all offsets for this step size k
        avg_length = np.mean(sub_lengths) if sub_lengths else 1e-12
        if avg_length <= 0:
            avg_length = 1e-12

        ln_lengths[idx] = np.log(avg_length)
        ln_inv_k[idx] = np.log(1.0 / k)

    # Fractal dimension is the slope of ln(length) vs ln(1/k)
    slope = np.polyfit(ln_inv_k, ln_lengths, 1)[0]
    return float(np.clip(slope, 1.0, 2.0))


def katz_fractal_dimension(signal: np.ndarray) -> float:
    """Calculate Katz Fractal Dimension (ratio of curve path length to max planar distance)."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal) - 1
    if n < 1:
        return 1.0

    path_length = float(np.sum(np.abs(np.diff(signal))))
    max_distance = float(np.max(np.abs(signal - signal[0])))

    if max_distance == 0.0 or path_length == 0.0:
        return 1.0

    return float(np.log10(n) / (np.log10(n) + np.log10(max_distance / path_length)))


def spectral_fractal_exponent(signal: np.ndarray, sr: int = 22050) -> float:
    """Estimate the 1/f^beta spectral power exponent from Welch's Power Spectral Density."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) < 2:
        return 0.0

    freqs, psd = welch(signal, fs=sr, nperseg=min(len(signal), 1024))

    # Keep only positive, non-zero frequencies and power bins
    valid = (freqs > 0) & (psd > 0) & np.isfinite(psd)
    if np.count_nonzero(valid) < 2:
        return 0.0

    # Linear fit of log10(power) vs log10(frequency)
    log_f = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])
    slope = np.polyfit(log_f, log_psd, 1)[0]
    return float(-slope)


def extract_dsp_features(audio_path: str) -> dict:
    """Extract standard audio metrics and fractal features from an audio file."""
    signal, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Basic file metadata via soundfile
    channels = 1
    bitrate = int(sr * 16)
    try:
        info = sf.info(audio_path)
        channels = int(info.channels)
        bits = 16
        if info.subtype:
            if "24" in info.subtype:
                bits = 24
            elif "32" in info.subtype:
                bits = 32
        bitrate = int(info.samplerate * bits * channels)
    except Exception:
        pass

    # Detect tempo (BPM)
    try:
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        bpm = 0.0

    signal = np.asarray(signal, dtype=np.float64)

    # Standard MIR features
    rms_energy = float(np.sqrt(np.mean(signal ** 2)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(signal)))

    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    dynamic_range_db = float(20.0 * np.log10(peak / (rms_energy + 1e-9))) if peak > 0 else 0.0

    # Limit fractal analysis to first 30 seconds for speed
    analysis_segment = signal[: min(len(signal), sr * 30)]

    return {
        "duration": float(librosa.get_duration(y=signal, sr=sr)),
        "sample_rate": int(sr),
        "channels": channels,
        "bitrate": int(bitrate),
        "bpm": bpm,
        "rms_energy": rms_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "dynamic_range_db": dynamic_range_db,
        "higuchi_fractal_dimension": float(higuchi_fractal_dimension(analysis_segment)),
        "katz_fractal_dimension": float(katz_fractal_dimension(analysis_segment)),
        "spectral_fractal_beta": float(spectral_fractal_exponent(analysis_segment, sr=sr)),
    }
