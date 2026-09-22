"""AudioMind procedural demo audio generator.

Synthesizes 8 diverse game-music-style WAV tracks with pure NumPy
(sine/harmonic partials, FM, saw/square voices, colored noise beds,
tanh overdrive, bitcrush, FFT reverb) into data/sample_audio/, then runs
the full ingestion pipeline: DSP/fractal feature extraction, multimodal
CLAP (or fallback) embeddings, and upsert into the relational database.

Each synthesizer exposes a `roughness` knob that scales a track-specific
noise bed; the pipeline bisects on it so every track lands inside its
target Higuchi Fractal Dimension range deterministically.

Usage:
    python synthesize_demo_data.py            # try real CLAP, auto-fallback
    python synthesize_demo_data.py --fallback # force deterministic fallback
"""

import argparse
import sys

import numpy as np
import soundfile as sf

import config
from core.database import init_db, save_track_record
from core.dsp import extract_dsp_features, higuchi_fractal_dimension
from core.models import AudioIntelligenceModel

SR = 22050


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------

def _axis(dur: float) -> np.ndarray:
    return np.arange(int(round(dur * SR))) / SR


def _peak_normalize(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 1e-12 else x


def _rms_normalize(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    r = np.sqrt(np.mean(x ** 2))
    return x * (target / r) if r > 1e-12 else x


def _soften_attack(x: np.ndarray, attack_sec: float = 0.004) -> np.ndarray:
    n = min(int(attack_sec * SR), len(x) // 2)
    if n > 0:
        x[:n] *= np.linspace(0.0, 1.0, n)
    return x


def _fade_edges(x: np.ndarray, fade: float = 0.03) -> np.ndarray:
    n = min(int(fade * SR), len(x) // 4)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _fft_filter(x: np.ndarray, lo_hz: float = None, hi_hz: float = None) -> np.ndarray:
    """Zero-phase band filter via rFFT (4th-order style rolloffs)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    gain = np.ones_like(freqs)
    if lo_hz is not None:
        gain *= 1.0 / (1.0 + (lo_hz / np.maximum(freqs, 1e-9)) ** 4)
    if hi_hz is not None:
        gain *= 1.0 / (1.0 + (np.maximum(freqs, 1e-9) / hi_hz) ** 4)
    gain[0] = 0.0
    return np.fft.irfft(spec * gain, len(x))


def _colored_noise(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Noise with power spectrum ~ 1/f^alpha (0=white, 1=pink, 2=brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.maximum(freqs[1:], 1.0) ** (alpha / 2.0)
    return np.fft.irfft(spec * shape, n)


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]


def _reverb(x: np.ndarray, tail: float = 1.5, level: float = 0.3,
            seed: int = 7) -> np.ndarray:
    """Cheap Schroeder-style reverb: convolve with decaying band-limited noise."""
    n_ir = int(tail * SR)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n_ir) * np.exp(-np.arange(n_ir) / (0.18 * SR))
    ir = _fft_filter(ir, lo_hz=150, hi_hz=7000)
    wet = _fft_convolve(x, ir)
    wet = _peak_normalize(wet[: len(x)])
    return x + level * wet


def _place(buf: np.ndarray, start_sec: float, wave: np.ndarray, gain: float = 1.0):
    i = int(round(start_sec * SR))
    if i >= len(buf) or i < 0:
        return
    j = min(i + len(wave), len(buf))
    buf[i:j] += gain * wave[: j - i]


def _finalize(music: np.ndarray, roughness: float, rng: np.random.Generator,
              alpha: float = 1.0, lo_hz: float = None, hi_hz: float = None) -> np.ndarray:
    """Normalize the musical bed, then add the tunable roughness noise layer
    whose RMS is `roughness` × the music RMS."""
    music = _fade_edges(_peak_normalize(music, 0.95))
    if roughness > 0:
        bed = _colored_noise(len(music), alpha, rng)
        bed = _fft_filter(bed, lo_hz=lo_hz, hi_hz=hi_hz)
        bed = _rms_normalize(bed)
        music = music + roughness * np.sqrt(np.mean(music ** 2)) * bed
    return _peak_normalize(music, 0.9)


# ---------------------------------------------------------------------------
# Voice building blocks
# ---------------------------------------------------------------------------

def _pluck(f: float, dur: float, decay: float = 5.0,
           partials=((1, 1.0), (2, 0.5), (3, 0.25))) -> np.ndarray:
    t = _axis(dur)
    x = sum(g * np.sin(2 * np.pi * f * k * t) for k, g in partials)
    x *= np.exp(-decay * t)
    return _soften_attack(x, 0.003)


def _piano(f: float, dur: float, decay: float = 1.8) -> np.ndarray:
    t = _axis(dur)
    x = (np.sin(2 * np.pi * f * t)
         + 0.40 * np.sin(4 * np.pi * f * t)
         + 0.18 * np.sin(6 * np.pi * f * t)
         + 0.08 * np.sin(8 * np.pi * f * t)) * np.exp(-decay * t)
    return _soften_attack(x, 0.006)


def _woodwind(f: float, dur: float, vib_hz: float = 5.2,
              vib_depth: float = 0.25) -> np.ndarray:
    t = _axis(dur)
    phase = 2 * np.pi * f * t + vib_depth * np.sin(2 * np.pi * vib_hz * t)
    x = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    env = np.ones(len(t))
    na, nr = int(0.03 * SR), int(0.06 * SR)
    env[:na] *= np.linspace(0, 1, na)
    env[-nr:] *= np.linspace(1, 0, nr)
    return x * env


def _pad(freqs, dur: float, attack: float = 1.0, release: float = 1.0,
         tremolo_hz: float = 0.0, tremolo_depth: float = 0.3,
         detune_hz: float = 0.0, phase_mod_hz: float = 0.0,
         phase_mod_depth: float = 0.0) -> np.ndarray:
    t = _axis(dur)
    x = np.zeros_like(t)
    for f in freqs:
        for d in ((-detune_hz, detune_hz) if detune_hz > 0 else (0.0,)):
            phase = 2 * np.pi * (f + d) * t
            if phase_mod_hz > 0:
                phase = phase + phase_mod_depth * np.sin(2 * np.pi * phase_mod_hz * t + f)
            x += np.sin(phase)
    n = len(t)
    env = np.ones(n)
    na, nr = min(int(attack * SR), n // 2), min(int(release * SR), n // 2)
    env[:na] *= np.linspace(0, 1, na)
    env[n - nr:] *= np.linspace(1, 0, nr)
    if tremolo_hz > 0:
        env *= 1.0 - tremolo_depth + tremolo_depth * np.sin(2 * np.pi * tremolo_hz * t)
    return x * env


def _saw_note(f: float, dur: float, decay: float = 12.0,
              lo_hz: float = None, hi_hz: float = None) -> np.ndarray:
    t = _axis(dur)
    x = 2.0 * ((f * t) % 1.0) - 1.0
    if lo_hz is not None or hi_hz is not None:
        x = _fft_filter(x, lo_hz=lo_hz, hi_hz=hi_hz)
    x *= np.exp(-decay * t)
    return _soften_attack(x, 0.002)


def _square_note(f: float, dur: float, decay: float = 15.0) -> np.ndarray:
    t = _axis(dur)
    x = np.sign(np.sin(2 * np.pi * f * t)) * np.exp(-decay * t)
    return _soften_attack(x, 0.001)


def _bitcrush(x: np.ndarray, levels: float = 6.0) -> np.ndarray:
    return np.round(x * levels) / levels


def _fm_note(f_carrier: float, dur: float, f_mod: float, index: float,
             decay: float, sweep: float = 0.0) -> np.ndarray:
    t = _axis(dur)
    fc = f_carrier * (1.0 - sweep * t / max(dur, 1e-9))
    phase = 2 * np.pi * np.cumsum(fc) / SR + index * np.sin(2 * np.pi * f_mod * t)
    x = np.sin(phase) * np.exp(-decay * t)
    return _soften_attack(x, 0.003)


def _kick(dur: float = 0.35, f_start: float = 140.0, f_end: float = 45.0,
          decay: float = 9.0) -> np.ndarray:
    t = _axis(dur)
    f = f_end + (f_start - f_end) * np.exp(-t * 18.0)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-decay * t)


def _noise_burst(dur: float, decay: float = 30.0, lo_hz: float = None,
                 hi_hz: float = None, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    n = int(dur * SR)
    x = _fft_filter(rng.standard_normal(n), lo_hz=lo_hz, hi_hz=hi_hz)
    x = _peak_normalize(x) * np.exp(-decay * np.arange(n) / SR)
    return _soften_attack(x, 0.002)


def _sub_pulse(f: float, dur: float = 0.30, decay: float = 8.0) -> np.ndarray:
    """Soft low sine thump — a beat pulse that stays smooth (low HFD)."""
    return _pluck(f, dur, decay, partials=((1, 1.0), (2, 0.2)))


def _strum(freqs, dur: float, decay: float = 6.0, stagger: float = 0.012) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        offset = int(i * stagger * SR)
        if offset >= n:
            continue
        p = _pluck(f, dur - offset / SR, decay)
        out[offset:] += p[: n - offset]
    return out


# ---------------------------------------------------------------------------
# Track synthesizers (roughness knob = noise-bed level)
# ---------------------------------------------------------------------------

def synth_iron_tyrant(roughness: float = 0.20) -> np.ndarray:
    """130 BPM industrial boss theme: overdriven sub-bass, kicks, metallic
    glitch percussion. Target HFD 1.65-1.80."""
    rng = np.random.default_rng(101)
    beat = 60.0 / 130.0
    n_beats = 32
    music = np.zeros(int(beat * n_beats * SR))
    roots = [55.0, 55.0, 65.41, 49.0]
    for b in range(n_beats):
        t0 = b * beat
        sub = _pluck(roots[b % 4], beat * 1.8, decay=2.2, partials=((1, 1.0), (2, 0.35)))
        _place(music, t0, np.tanh(4.0 * 1.8 * sub), 0.55)
        _place(music, t0, _kick(0.40, 150, 42, 10), 0.9)
        if b % 2 == 1:
            _place(music, t0 + beat * 0.5,
                   _noise_burst(0.12, 28, 2500, 9000, rng), 0.5)
        if b % 4 == 2:
            _place(music, t0 + beat * 0.75,
                   _noise_burst(0.08, 40, 4000, 11000, rng), 0.35)
        if b % 8 == 6:
            stab = np.tanh(2.5 * _pad([220, 261.63, 329.63], beat * 2,
                                      attack=0.01, release=0.3))
            _place(music, t0, stab, 0.30)
    for i in range(n_beats * 2):
        if i % 4 != 0:  # off-eighth metallic ticks
            _place(music, i * beat / 2,
                   _noise_burst(0.03, 90, 7000, 14000, rng), 0.18)
    return _finalize(music, roughness, rng, alpha=0.0, lo_hz=1500, hi_hz=10000)


def synth_whispering_ruins(roughness: float = 0.12) -> np.ndarray:
    """70 BPM dungeon drone: sub-bass rumble, minor pads with tremolo,
    subterranean wind. Target HFD 1.20-1.28."""
    rng = np.random.default_rng(102)
    beat = 60.0 / 70.0
    dur = beat * 16
    t = _axis(dur)
    music = 0.4 * (np.sin(2 * np.pi * 40 * t) + 0.7 * np.sin(2 * np.pi * 55 * t)
                   + 0.5 * np.sin(2 * np.pi * 27.5 * t))
    music *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t)
    _place(music, 0, _pad([220, 261.63, 329.63], beat * 8, attack=2.0, release=2.0,
                          tremolo_hz=0.6, tremolo_depth=0.35, detune_hz=0.6), 0.30)
    _place(music, beat * 8, _pad([174.61, 220, 261.63], beat * 8, attack=2.0,
                                 release=2.0, tremolo_hz=0.5, tremolo_depth=0.30,
                                 detune_hz=0.6), 0.30)
    for bar in (1, 3):
        _place(music, bar * 4 * beat, _fm_note(880, 2.5, 0.9, 1.2, 1.2), 0.10)
    for b in range(16):  # low heartbeat pulse keeps the 70 BPM grid detectable
        gain = 0.60 if b % 4 == 0 else 0.45
        _place(music, b * beat, _sub_pulse(55.0, 0.30, 8.0), gain)
    return _finalize(music, roughness, rng, alpha=1.0, lo_hz=80, hi_hz=4000)


def synth_sunnyvale_village(roughness: float = 0.08) -> np.ndarray:
    """92 BPM gentle folk: harp-like plucked arpeggios over a warm pink bed.
    Target HFD 1.12-1.22."""
    rng = np.random.default_rng(103)
    beat = 60.0 / 92.0
    n_bars = 6
    music = np.zeros(int(round(4 * beat * n_bars * SR)))
    chords = [[130.81, 164.81, 196.0], [196.0, 246.94, 293.66],
              [110.0, 130.81, 164.81], [87.31, 110.0, 130.81],
              [130.81, 164.81, 196.0], [196.0, 246.94, 293.66]]
    for bar, chord in enumerate(chords):
        t0 = bar * 4 * beat
        for i in range(8):  # up-down arpeggio in eighth notes
            f = chord[[0, 1, 2, 1][i % 4]] * (2.0 if i % 3 == 2 else 1.0)
            _place(music, t0 + i * beat / 2, _pluck(f, beat * 1.6, 4.5), 0.26)
        for b_i in range(4):  # bass pluck every beat for a steady folk pulse
            gain = 0.65 if b_i == 0 else 0.52
            _place(music, t0 + b_i * beat,
                   _pluck(chord[0] / 2, beat * 1.4, 3.5,
                          partials=((1, 1.0), (2, 0.4))), gain)
        _place(music, t0, _pad(chord, 4 * beat, attack=0.5, release=1.0,
                               tremolo_hz=0.9, tremolo_depth=0.2), 0.10)
    return _finalize(music, roughness, rng, alpha=1.0, hi_hz=6000)


def synth_neon_highway(roughness: float = 0.14) -> np.ndarray:
    """124 BPM synthwave: four-on-the-floor kick, syncopated saw bass,
    bright chord stabs. Target HFD 1.40-1.50."""
    rng = np.random.default_rng(104)
    beat = 60.0 / 124.0
    n_beats = 32
    music = np.zeros(int(round(beat * n_beats * SR)))
    sixteenth = beat / 4
    bass_roots = [110.0, 87.31, 130.81, 98.0]
    stab_chords = [[220, 261.63, 329.63], [174.61, 220, 261.63],
                   [261.63, 329.63, 392.0], [196, 246.94, 293.66]]
    pattern = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    for b in range(n_beats):
        t0 = b * beat
        _place(music, t0, _kick(0.30, 150, 42, 14), 1.0)
        bar = b // 4
        for s, on in enumerate(pattern):
            if on:
                f = bass_roots[bar % 4]
                _place(music, t0 + s * sixteenth,
                       _saw_note(f, sixteenth * 1.8, 14, lo_hz=60, hi_hz=900), 0.42)
        if b % 2 == 1:
            chord = stab_chords[bar % 4]
            stab = np.tanh(1.5 * _pad(chord, 0.45, attack=0.005, release=0.25,
                                      detune_hz=1.2))
            _place(music, t0 + beat * 0.5, stab, 0.26)
        for s in range(4):
            _place(music, t0 + s * sixteenth,
                   _noise_burst(0.03, 80, 8000, 15000, rng), 0.12)
    return _finalize(music, roughness, rng, alpha=0.5, lo_hz=300, hi_hz=6000)


def synth_cosmic_sanctuary(roughness: float = 0.05) -> np.ndarray:
    """65 BPM celestial menu theme: detuned phase-modulated sine pads with a
    long reverb tail. Target HFD 1.08-1.16."""
    rng = np.random.default_rng(105)
    beat = 60.0 / 65.0
    dur = beat * 16
    music = np.zeros(int(round(dur * SR)))
    _place(music, 0, _pad([130.81, 261.63, 329.63, 392.0, 493.88], beat * 8,
                          attack=2.5, release=2.5, tremolo_hz=0.25,
                          tremolo_depth=0.2, detune_hz=0.8, phase_mod_hz=0.13,
                          phase_mod_depth=1.8), 0.32)
    _place(music, beat * 8, _pad([87.31, 174.61, 261.63, 329.63, 440.0], beat * 8,
                                 attack=2.5, release=2.5, tremolo_hz=0.22,
                                 tremolo_depth=0.2, detune_hz=0.8, phase_mod_hz=0.11,
                                 phase_mod_depth=1.6), 0.32)
    shimmer = np.sin(2 * np.pi * 1567.98 * _axis(dur)
                     + 0.8 * np.sin(2 * np.pi * 0.21 * _axis(dur)))
    music += 0.05 * shimmer * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * _axis(dur)))
    for b in range(16):  # faint celestial pulse anchors the 65 BPM grid
        gain = 0.30 if b % 4 == 0 else 0.20
        _place(music, b * beat, _sub_pulse(65.41, 0.26, 12.0), gain)
    music = _reverb(music, tail=2.5, level=0.40, seed=15)
    return _finalize(music, roughness, rng, alpha=1.0, hi_hz=5000)


def synth_glitch_overlord(roughness: float = 0.12) -> np.ndarray:
    """155 BPM extreme combat: bitcrushed square arps, FM screeches,
    stutter transients. Target HFD 1.75-1.90."""
    rng = np.random.default_rng(106)
    beat = 60.0 / 155.0
    n_beats = 32
    sixteenth = beat / 4
    music = np.zeros(int(round(beat * n_beats * SR)))
    arp = [220.0, 277.18, 329.63, 440.0, 554.37, 659.26]
    for b in range(n_beats):
        t0 = b * beat
        _place(music, t0, _kick(0.30, 170, 50, 12), 0.9)
        if b % 4 == 2:
            _place(music, t0, _noise_burst(0.15, 22, 1500, 8000, rng), 0.5)
        for s in range(4):
            f = arp[(b * 4 + s) % len(arp)]
            note = _bitcrush(_square_note(f, sixteenth * 0.85, 18), levels=5.0)
            _place(music, t0 + s * sixteenth, note, 0.26)
        if b % 2 == 0:
            _place(music, t0 + sixteenth,
                   _fm_note(1200, 0.4, 173, 8, 6, sweep=0.5), 0.22)
        if rng.random() < 0.35:  # stutter transients
            for k in range(2):
                _place(music, t0 + beat - (k + 1) * sixteenth / 2,
                       _noise_burst(0.02, 120, 3000, 12000, rng), 0.3)
    return _finalize(music, roughness, rng, alpha=0.0, lo_hz=400, hi_hz=11000)


def synth_tavern_hearth(roughness: float = 0.09) -> np.ndarray:
    """105 BPM medieval tavern: woodwind melody, lute strums, tambourine.
    Target HFD 1.18-1.28."""
    rng = np.random.default_rng(107)
    beat = 60.0 / 105.0
    n_bars = 7
    music = np.zeros(int(round(4 * beat * n_bars * SR)))
    scale = [293.66, 329.63, 392.0, 440.0, 523.25, 587.33]
    chords = [[196, 246.94, 293.66], [130.81, 164.81, 196.0],
              [110.0, 164.81, 220.0], [98.0, 123.47, 196.0]]
    melody_bars = [[0, 2, 1, 3], [4, 3, 2, 0], [1, 2, 3, 4], [5, 4, 3, 2],
                   [0, 1, 2, 4], [3, 4, 5, 3], [2, 1, 0, 1]]
    for bar in range(n_bars):
        t0 = bar * 4 * beat
        chord = chords[bar % 4]
        for i, deg in enumerate(melody_bars[bar]):
            f = scale[deg % len(scale)]
            _place(music, t0 + i * beat, _woodwind(f, beat * 0.95), 0.30)
        _place(music, t0, _strum(chord, beat * 1.8, 6.0), 0.30)
        _place(music, t0 + 2 * beat, _strum(chord, beat * 1.8, 6.0), 0.26)
        _place(music, t0, _pluck(chord[0] / 2, beat * 1.9, 3.5,
                                 partials=((1, 1.0), (2, 0.4))), 0.42)
        for half in (0.5, 1.5, 2.5, 3.5):  # tambourine jingles
            _place(music, t0 + half * beat,
                   _noise_burst(0.05, 25, 5000, 12000, rng), 0.16)
    return _finalize(music, roughness, rng, alpha=1.0, hi_hz=6000)


def synth_starlight_horizon(roughness: float = 0.07) -> np.ndarray:
    """80 BPM emotional credits theme: piano chords + swelling strings.
    Target HFD 1.14-1.24."""
    rng = np.random.default_rng(108)
    beat = 60.0 / 80.0
    chord_len = 5 * beat  # 4 chords over 20 beats
    dur = chord_len * 4
    music = np.zeros(int(round(dur * SR)))
    prog = [[130.81, 164.81, 196.0, 261.63], [98.0, 196.0, 246.94, 293.66],
            [110.0, 220.0, 261.63, 329.63], [87.31, 174.61, 220.0, 261.63]]
    for i, chord in enumerate(prog):
        t0 = i * chord_len
        for f in chord:
            _place(music, t0, _piano(f, chord_len * 0.95, 1.8), 0.24)
        _place(music, t0 + 2.5 * beat, _piano(chord[-1] * 2, chord_len * 0.5, 2.2), 0.16)
        strings = _pad(chord, chord_len, attack=3.5, release=2.0, detune_hz=0.5,
                       tremolo_hz=0.35, tremolo_depth=0.15)
        strings = _fft_filter(strings, hi_hz=1200)
        _place(music, t0, strings, 0.22)
        _place(music, t0, _pad([chord[0] / 2], chord_len, attack=2.0,
                               release=2.0), 0.30)
    for b in range(20):  # soft orchestral bass pulse anchors the 80 BPM grid
        gain = 0.30 if b % 5 == 0 else 0.22
        _place(music, b * beat, _sub_pulse(49.0, 0.28, 10.0), gain)
    music = _reverb(music, tail=2.0, level=0.25, seed=21)
    return _finalize(music, roughness, rng, alpha=1.0, hi_hz=5000)


# ---------------------------------------------------------------------------
# Track catalog
# ---------------------------------------------------------------------------

TRACKS = [
    {
        "filename": "iron_tyrant_boss.wav", "synth": synth_iron_tyrant,
        "title": "Iron Tyrant", "artist": "Vortex Synth",
        "genre": "Industrial", "mood": "Aggressive & Chaotic",
        "tags": ["Boss Fight", "Industrial", "Cyberpunk", "Distortion"],
        "description": "Overdriven sub-bass and metallic glitch percussion driving a relentless 130 BPM assault.",
        "use": "climactic boss encounters and high-chaos combat",
        "hfd_range": (1.65, 1.80),
    },
    {
        "filename": "whispering_ruins_dungeon.wav", "synth": synth_whispering_ruins,
        "title": "Whispering Ruins", "artist": "Shadow Echo",
        "genre": "Ambient", "mood": "Calm & Atmospheric",
        "tags": ["Dungeon", "Eerie", "Drone", "Exploration"],
        "description": "Deep sub-bass rumble and tremolo minor pads drifting beneath subterranean winds at 70 BPM.",
        "use": "exploration and dungeon ambience",
        "hfd_range": (1.20, 1.28),
    },
    {
        "filename": "sunnyvale_village.wav", "synth": synth_sunnyvale_village,
        "title": "Sunnyvale Meadow", "artist": "Lute & Leaf",
        "genre": "Acoustic", "mood": "Organic & Melodic",
        "tags": ["Village", "Acoustic", "Harmonic", "Peaceful"],
        "description": "Gentle 92 BPM harp arpeggios and warm acoustic partials over a soft pink-noise meadow.",
        "use": "peaceful village hubs and safe havens",
        "hfd_range": (1.12, 1.22),
    },
    {
        "filename": "neon_highway_synthwave.wav", "synth": synth_neon_highway,
        "title": "Neon Highway", "artist": "Kavinsky Waves",
        "genre": "Synthwave", "mood": "Driving & Tense",
        "tags": ["Combat", "Retro Synth", "Action", "Night Drive"],
        "description": "Four-on-the-floor kick, syncopated saw bass and bright retro stabs at 124 BPM.",
        "use": "combat, pursuits and neon night drives",
        "hfd_range": (1.40, 1.50),
    },
    {
        "filename": "cosmic_sanctuary_menu.wav", "synth": synth_cosmic_sanctuary,
        "title": "Cosmic Sanctuary", "artist": "Aetheria",
        "genre": "Ambient", "mood": "Calm & Atmospheric",
        "tags": ["Main Menu", "Ambient", "Smooth", "Celestial"],
        "description": "Detuned phase-modulated sine pads floating through a long ethereal reverb tail at 65 BPM.",
        "use": "main menus and celestial world-building",
        "hfd_range": (1.08, 1.16),
    },
    {
        "filename": "glitch_overlord_combat.wav", "synth": synth_glitch_overlord,
        "title": "Glitch Overlord", "artist": "Null Pointer",
        "genre": "Industrial", "mood": "Aggressive & Chaotic",
        "tags": ["Boss Fight", "Glitch", "High Chaos", "Intense"],
        "description": "Bitcrushed square arpeggios, alien FM screeches and stutter transients at a frantic 155 BPM.",
        "use": "final bosses and extreme intensity combat",
        "hfd_range": (1.75, 1.90),
    },
    {
        "filename": "tavern_hearth_folk.wav", "synth": synth_tavern_hearth,
        "title": "Tavern Hearth", "artist": "Bardic Fellowship",
        "genre": "Acoustic", "mood": "Organic & Melodic",
        "tags": ["Village", "Exploration", "Acoustic", "Folk"],
        "description": "Woodwind melodies, lute strums and tambourine jingles around a 105 BPM medieval hearth.",
        "use": "taverns, festivals and cozy hubs",
        "hfd_range": (1.18, 1.28),
    },
    {
        "filename": "starlight_horizon_credits.wav", "synth": synth_starlight_horizon,
        "title": "Starlight Horizon", "artist": "Symphonic Soul",
        "genre": "Orchestral", "mood": "Organic & Melodic",
        "tags": ["Credits", "Emotional", "Melodic", "Orchestral"],
        "description": "Slow piano chords and swelling string harmonics closing the journey at 80 BPM.",
        "use": "credits, finales and reflective cutscenes",
        "hfd_range": (1.14, 1.24),
    },
]


# ---------------------------------------------------------------------------
# Roughness auto-tuning + ingestion pipeline
# ---------------------------------------------------------------------------

def tune_roughness(synth_fn, target_d: float, lo: float = 0.0, hi: float = 3.0,
                   iters: int = 9) -> float:
    """Bisect the roughness knob so the synthesized signal's Higuchi FD
    approaches target_d (D increases monotonically with the noise bed)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = higuchi_fractal_dimension(synth_fn(mid))
        if d < target_d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_semantic_text(spec: dict, description: str) -> str:
    return (f"{spec['title']} by {spec['artist']}. {spec['genre']} - "
            f"{spec['mood']}. Tags: {', '.join(spec['tags'])}. {description}")


def ingest_track(spec: dict, model: AudioIntelligenceModel) -> dict:
    """Synthesize, tune roughness, write WAV, extract features, embed, save."""
    target_d = sum(spec["hfd_range"]) / 2.0
    roughness = tune_roughness(spec["synth"], target_d)
    signal = spec["synth"](roughness)

    filepath = config.AUDIO_DIR / spec["filename"]
    sf.write(filepath, signal, SR, subtype="PCM_16")

    feats = extract_dsp_features(str(filepath))
    audio_embed = model.embed_audio(str(filepath))

    description = (
        f"{spec['description']} Measured near {feats['bpm']:.0f} BPM with "
        f"Higuchi fractal dimension {feats['higuchi_fractal_dimension']:.2f}, "
        f"RMS energy {feats['rms_energy']:.2f} and spectral centroid "
        f"{feats['spectral_centroid']:.0f} Hz. "
        f"Suggested use: {spec['use']}."
    )
    semantic = {
        "primary_genre": spec["genre"],
        "mood": spec["mood"],
        "tags": spec["tags"],
        "generated_description": description,
    }
    text_embed = model.embed_text(build_semantic_text(spec, description))

    track = {
        "track_id": spec["filename"].rsplit(".", 1)[0],
        "title": spec["title"],
        "artist": spec["artist"],
        "duration": feats["duration"],
        "sample_rate": feats["sample_rate"],
        "bitrate": feats["bitrate"],
        "channels": feats["channels"],
        "file_path": str(filepath),
    }
    save_track_record(track, feats, semantic,
                      audio_embed=audio_embed, text_embed=text_embed)

    return {
        "title": spec["title"],
        "bpm": feats["bpm"],
        "duration": feats["duration"],
        "hfd": feats["higuchi_fractal_dimension"],
        "hfd_range": spec["hfd_range"],
        "roughness": roughness,
        "path": filepath,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate AudioMind demo audio + DB")
    parser.add_argument("--fallback", action="store_true",
                        help="force the deterministic fallback embedder (no CLAP download)")
    args = parser.parse_args(argv)

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()

    print("Loading embedding model (falls back automatically if offline)...")
    model = AudioIntelligenceModel(
        fallback_mode=args.fallback
    )
    print(f"Model ready (fallback_mode={model.fallback_mode}).\n")

    rows = []
    for spec in TRACKS:
        print(f"Synthesizing '{spec['title']}' ...", flush=True)
        rows.append(ingest_track(spec, model))

    print("\n{:<22} {:>8} {:>10} {:>12} {:>14} {:>10}".format(
        "TITLE", "BPM", "DURATION", "HIGUCHI FD", "TARGET FD", "ROUGHNESS"))
    print("-" * 82)
    for r in rows:
        lo, hi = r["hfd_range"]
        print("{:<22} {:>8.1f} {:>9.1f}s {:>12.3f} {:>10.2f}-{:<4.2f} {:>10.3f}".format(
            r["title"][:22], r["bpm"], r["duration"], r["hfd"], lo, hi, r["roughness"]))

    in_range = sum(1 for r in rows if r["hfd_range"][0] <= r["hfd"] <= r["hfd_range"][1])
    print("-" * 82)
    print(f"SUCCESS: {len(rows)}/{len(rows)} tracks synthesized into {config.AUDIO_DIR}")
    print(f"Higuchi FD within target range: {in_range}/{len(rows)}")
    print(f"Database populated at {config.DB_PATH}")
    return 0 if in_range == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
