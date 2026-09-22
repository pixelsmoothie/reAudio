"""Multimodal audio intelligence for AudioMind.

Uses LAION CLAP (Contrastive Language-Audio Pretraining) to embed text queries
and audio waveforms into a shared 512-dimensional vector space for semantic search.
"""

import hashlib
import os
import re
from collections import Counter
from typing import Optional, Union

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

import config

EMBED_DIM = 512


def _normalize_to_unit(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length (norm = 1.0) so dot product equals cosine similarity."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        unit = np.zeros(EMBED_DIM, dtype=np.float64)
        unit[0] = 1.0
        return unit.astype(np.float32)
    return (vec / norm).astype(np.float32)


class AudioIntelligenceModel:
    """Multimodal CLAP model for cross-modal text-to-audio search."""

    def __init__(self, model_name: str = config.CLAP_MODEL_NAME,
                 device: Optional[str] = None,
                 fallback_mode: bool = False):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_mode = bool(fallback_mode)
        self.model = None
        self.processor = None

        if not self.fallback_mode:
            try:
                # Load pre-trained LAION CLAP weights from Hugging Face
                self.model = ClapModel.from_pretrained(self.model_name)
                self.processor = ClapProcessor.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.fallback_mode = True

    def _load_audio(self, audio_input: Union[str, os.PathLike, np.ndarray], sr: int) -> np.ndarray:
        """Coerce audio input to a 1D mono float64 array."""
        if isinstance(audio_input, (str, os.PathLike)):
            import librosa
            audio, _ = librosa.load(os.fspath(audio_input), sr=sr, mono=True)
            return np.asarray(audio, dtype=np.float64)

        arr = np.asarray(audio_input, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)
        return arr.reshape(-1)

    # ------------------------------------------------------------------
    # Simple Deterministic Offline Fallback
    # ------------------------------------------------------------------

    def _fallback_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Simple deterministic projection for offline testing."""
        # Use simple audio stats (mean, std, energy, slices) to seed vector
        seed = int(abs(np.sum(audio[:1000]) * 1e5)) % (2**31 - 1)
        rng = np.random.default_rng(seed or 1234)
        return _normalize_to_unit(rng.standard_normal(EMBED_DIM))

    def _fallback_text_embedding(self, text: str) -> np.ndarray:
        """Simple bag-of-words hash embedding for offline testing."""
        tokens = re.findall(r"[a-z0-9]+", str(text).lower()) or ["empty"]
        vector = np.zeros(EMBED_DIM, dtype=np.float64)
        for token, count in Counter(tokens).items():
            token_hash = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
            rng = np.random.default_rng(token_hash)
            vector += (1.0 + np.log(count)) * rng.standard_normal(EMBED_DIM)
        return _normalize_to_unit(vector)

    # ------------------------------------------------------------------
    # Public Embedding & Similarity Methods
    # ------------------------------------------------------------------

    def embed_audio(self, audio_input: Union[str, os.PathLike, np.ndarray], sr: int = 48000) -> np.ndarray:
        """Embed audio into a normalized 512-dimensional vector."""
        audio = self._load_audio(audio_input, sr)
        if not self.fallback_mode and self.model is not None:
            try:
                # Use first 30 seconds for CLAP embedding
                audio_clip = audio[: sr * 30]
                try:
                    inputs = self.processor(audio=audio_clip, sampling_rate=sr, return_tensors="pt").to(self.device)
                except (TypeError, ValueError):
                    inputs = self.processor(audios=audio_clip, sampling_rate=sr, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    out = self.model.get_audio_features(**inputs)
                    emb = out.pooler_output if hasattr(out, "pooler_output") else out
                return _normalize_to_unit(emb.detach().cpu().numpy())
            except Exception:
                pass
        return self._fallback_audio_embedding(audio)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query into a normalized 512-dimensional vector."""
        if not self.fallback_mode and self.model is not None:
            try:
                inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    out = self.model.get_text_features(**inputs)
                    emb = out.pooler_output if hasattr(out, "pooler_output") else out
                return _normalize_to_unit(emb.detach().cpu().numpy())
            except Exception:
                pass
        return self._fallback_text_embedding(text)

    def compute_similarity(self, text_query: str, audio_embed: np.ndarray) -> float:
        """Compute cosine similarity between text query and audio embedding."""
        text_vec = self.embed_text(text_query).astype(np.float64)
        audio_vec = np.asarray(audio_embed, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(audio_vec)
        if norm > 1e-12:
            audio_vec = audio_vec / norm
        return float(np.clip(float(np.dot(text_vec, audio_vec)), -1.0, 1.0))


def generate_semantic_profile(dsp_features: dict, audio_embed: Optional[np.ndarray] = None) -> dict:
    """Infer likely genre, mood, tags, and description from DSP metrics."""
    hfd = float(dsp_features.get("higuchi_fractal_dimension", 1.5))
    bpm = float(dsp_features.get("bpm", 0.0))
    rms = float(dsp_features.get("rms_energy", 0.0))
    centroid = float(dsp_features.get("spectral_centroid", 0.0))
    beta = float(dsp_features.get("spectral_fractal_beta", 0.0))

    if hfd >= 1.60:
        primary_genre = "Glitch / Metal" if rms > 0.2 else "Industrial Electronic"
        mood = "Aggressive & Chaotic"
        tags = ["Boss Fight", "Cyberpunk", "High Chaos", "Industrial", "Distortion"]
        use_case = "high-intensity boss fights, chase sequences, or climactic set pieces"
        roughness_text = "a chaotic, highly irregular, fractal-dense waveform"
    elif hfd <= 1.25 and bpm < 90:
        primary_genre = "Ambient Drone" if bpm < 70 else "Atmospheric"
        mood = "Calm & Atmospheric"
        tags = ["Exploration", "Dungeon", "Eerie", "Minimal", "Smooth"]
        use_case = "slow exploration, dungeon ambience, or eerie world-building"
        roughness_text = "a remarkably smooth, self-similar waveform"
    elif 115 <= bpm <= 140 and rms > 0.15:
        primary_genre = "Cyberpunk Synthwave"
        mood = "Driving & Tense"
        tags = ["Combat", "Retro Synth", "Action", "Night Drive"]
        use_case = "combat encounters, high-speed pursuits, or neon-lit night drives"
        roughness_text = "a moderately textured, propulsively rhythmic waveform"
    else:
        primary_genre = "Acoustic / Orchestral"
        mood = "Organic & Melodic"
        tags = ["Village", "Main Menu", "Acoustic", "Harmonic"]
        use_case = "peaceful village hubs, main menus, or narrative cutscenes"
        roughness_text = "a gentle, organic waveform"

    description = (
        f"A {mood.lower()} {primary_genre.lower()} piece running at {bpm:.0f} BPM. "
        f"The waveform shows {roughness_text} (Higuchi fractal dimension {hfd:.2f}), "
        f"with an RMS energy of {rms:.3f} and a spectral centroid of {centroid:.0f} Hz "
        f"that define its brightness and weight. The 1/f spectral exponent of {beta:.2f} "
        f"characterizes its tonal balance across frequencies. "
        f"Best suited for {use_case}."
    )

    return {
        "primary_genre": primary_genre,
        "mood": mood,
        "tags": tags,
        "generated_description": description,
    }
