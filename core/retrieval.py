"""Hybrid SQL + vector retrieval engine for AudioMind.

Two-stage search:
1. SQL relational pruning (hard filters: duration, BPM, genre, fractal bounds).
2. Vector cosine re-ranking (semantic alignment) + acoustic fit score.
"""

import re
from typing import Optional

import numpy as np

from core.database import filter_tracks_sql, get_all_tracks, get_track_by_id
from core.explainer import RAGExplainer
from core.models import AudioIntelligenceModel


def _cosine(a, b) -> float:
    """Compute cosine similarity between two vectors, returning a float in [-1.0, 1.0]."""
    if a is None or b is None:
        return 0.0

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    # Dot product of unit vectors equals cosine similarity
    return float(np.dot(a / norm_a, b / norm_b))


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace character ranges with spaces to preserve original string coordinates."""
    chars = list(text)
    for start, end in spans:
        for i in range(start, min(end, len(chars))):
            chars[i] = " "
    return "".join(chars)


class QueryParser:
    """Extracts numeric and genre constraints from natural language search queries."""

    DURATION_UNITS = {
        "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
        "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
        "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    }

    # Regex patterns for duration and tempo
    DURATION_RE = re.compile(
        r"\b(?:under|less\s+than|below|shorter\s+than|at\s+most|up\s+to|max(?:imum)?)\s+"
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)?\b",
        re.IGNORECASE,
    )
    BPM_OVER_RE = re.compile(
        r"\b(?:over|above|more\s+than|greater\s+than|faster\s+than|>=?)\s*(\d+(?:\.\d+)?)\s*bpm\b",
        re.IGNORECASE,
    )
    BPM_UNDER_RE = re.compile(
        r"\b(?:under|below|less\s+than|slower\s+than|<=?)\s*(\d+(?:\.\d+)?)\s*bpm\b",
        re.IGNORECASE,
    )
    BPM_FAST_RE = re.compile(r"\b(?:fast|high)\s+(?:tempo|bpm)\b", re.IGNORECASE)
    BPM_SLOW_RE = re.compile(r"\b(?:slow|low)\s+(?:tempo|bpm)\b", re.IGNORECASE)

    # Keywords for fractal roughness and genres
    CHAOTIC_KEYWORDS = ("chaotic", "harsh", "distorted", "heavy", "boss fight", "intense", "glitch")
    SMOOTH_KEYWORDS = ("smooth", "ambient", "peaceful", "calm", "relaxing", "gentle", "drone")

    GENRE_MAP = {
        "industrial": "Industrial", "synthwave": "Synthwave", "acoustic": "Acoustic",
        "ambient": "Ambient", "metal": "Metal", "electronic": "Electronic",
        "orchestral": "Orchestral", "orchestra": "Orchestral", "folk": "Folk",
        "drone": "Ambient", "jazz": "Jazz", "rock": "Rock", "cinematic": "Cinematic",
        "hip hop": "Hip Hop", "lofi": "LoFi", "lo-fi": "LoFi",
    }

    def parse_query(self, query: str) -> dict:
        """Parse natural language query into hard constraints + clean semantic text."""
        raw = str(query or "").strip()
        lowered = raw.lower()
        spans = []

        # 1. Parse BPM (tempo)
        bpm_min = None
        bpm_max = None

        m = self.BPM_OVER_RE.search(lowered)
        if m:
            bpm_min = float(m.group(1))
            spans.append(m.span())

        m = self.BPM_UNDER_RE.search(lowered)
        if m:
            bpm_max = float(m.group(1))
            spans.append(m.span())

        m = self.BPM_FAST_RE.search(lowered)
        if m:
            bpm_min = 120.0 if bpm_min is None else max(bpm_min, 120.0)
            spans.append(m.span())

        m = self.BPM_SLOW_RE.search(lowered)
        if m:
            bpm_max = 95.0 if bpm_max is None else min(bpm_max, 95.0)
            spans.append(m.span())

        # 2. Parse Duration (masking out BPM first to avoid "under 120 bpm" duration confusion)
        duration_max = None
        masked_for_duration = _mask_spans(lowered, spans)
        m = self.DURATION_RE.search(masked_for_duration)
        if m:
            val = float(m.group(1))
            unit = (m.group(2) or "s").lower()
            duration_max = val * self.DURATION_UNITS.get(unit, 1.0)
            spans.append(m.span())

        # 3. Infer Fractal Roughness bounds from keywords
        min_fractal_dim = 1.50 if any(kw in lowered for kw in self.CHAOTIC_KEYWORDS) else None
        max_fractal_dim = 1.30 if any(kw in lowered for kw in self.SMOOTH_KEYWORDS) else None

        # 4. Infer Genre
        genre = None
        best_pos = len(lowered) + 1
        for keyword, mapped_genre in self.GENRE_MAP.items():
            pos = lowered.find(keyword)
            if pos != -1 and pos < best_pos:
                best_pos, genre = pos, mapped_genre

        # 5. Clean query text for the AI embedder (strip numbers and units)
        cleaned = re.sub(r"\s+", " ", _mask_spans(raw, spans)).strip(" ,.;:-")
        if not cleaned:
            cleaned = raw

        return {
            "cleaned_query": cleaned,
            "duration_max": duration_max,
            "bpm_min": bpm_min,
            "bpm_max": bpm_max,
            "min_fractal_dim": min_fractal_dim,
            "max_fractal_dim": max_fractal_dim,
            "genre": genre,
        }


class HybridRetrievalEngine:
    """Two-stage retrieval engine: SQL relational pruning + vector re-ranking."""

    FILTER_KEYS = ("duration_min", "duration_max", "bpm_min", "bpm_max",
                   "min_fractal_dim", "max_fractal_dim", "genre", "mood",
                   "tag_keyword")

    SEMANTIC_WEIGHT = 0.70  # 70% weight for vector similarity
    ACOUSTIC_WEIGHT = 0.30  # 30% weight for numeric constraints fit

    def __init__(self, model: AudioIntelligenceModel = None, session=None):
        self.model = model if model is not None else AudioIntelligenceModel()
        self.session = session
        self.parser = QueryParser()
        self.explainer = RAGExplainer()

    def _acoustic_alignment(self, track: dict, constraints: dict) -> float:
        """Calculate how well a track satisfies numeric constraints [0.0 to 1.0]."""
        scores = []

        # Rules: (track_field, constraint_key, is_minimum, tolerance_scale)
        rules = [
            ("higuchi_fractal_dimension", "min_fractal_dim", True,  0.25),
            ("higuchi_fractal_dimension", "max_fractal_dim", False, 0.25),
            ("bpm",                       "bpm_min",         True,  20.0),
            ("bpm",                       "bpm_max",         False, 20.0),
            ("duration",                  "duration_max",    False, 30.0),
        ]

        for field, key, is_min, scale in rules:
            target = constraints.get(key)
            actual = track.get(field)
            if target is not None and actual is not None:
                diff = (float(actual) - float(target)) if is_min else (float(target) - float(actual))
                # 1.0 if condition is satisfied; linear decay towards 0.0 if violated
                score = 1.0 if diff >= 0 else max(0.0, 1.0 + diff / scale)
                scores.append(score)

        return float(sum(scores) / len(scores)) if scores else 0.5

    def search(self, query: str, filters: Optional[dict] = None,
               top_k: int = 5, session=None) -> list[dict]:
        """Execute two-stage hybrid search."""
        session = session if session is not None else self.session

        # Step 1: Parse user intent and combine with explicit filters
        parsed = self.parser.parse_query(query)
        constraints = {k: v for k, v in parsed.items() if k in self.FILTER_KEYS and v is not None}
        if filters:
            constraints.update({k: v for k, v in filters.items() if k in self.FILTER_KEYS and v is not None})

        # Step 2: SQL Relational Pruning (fallback to all tracks if filters match nothing)
        candidates = filter_tracks_sql(session=session, **constraints)
        if not candidates:
            candidates = get_all_tracks(session=session)

        # Step 3: Embed clean query text with AI model
        text_for_embedding = parsed["cleaned_query"] or query
        query_emb = self.model.embed_text(text_for_embedding)

        # Step 4: Re-rank survivors by combining vector similarity + acoustic alignment
        results = []
        for track in candidates:
            # Multimodal similarity: fuse acoustic audio match with semantic metadata match
            aud_sim = _cosine(query_emb, track.get("audio_embedding"))
            txt_sim = _cosine(query_emb, track.get("text_embedding"))
            similarity = max(aud_sim, 0.5 * aud_sim + 0.5 * txt_sim)

            alignment = self._acoustic_alignment(track, constraints)

            # 70/30 score fusion into match percentage (0% to 100%)
            fused = self.SEMANTIC_WEIGHT * similarity + self.ACOUSTIC_WEIGHT * alignment
            score_pct = round(min(100.0, max(0.0, fused * 100.0)), 1)

            row = dict(track)
            row["similarity"] = round(float(similarity), 4)
            row["acoustic_alignment"] = round(float(alignment), 4)
            row["match_score"] = score_pct

            # Step 5: Grounded RAG explanation
            row["explanation"] = self.explainer.explain(query, row, score_pct, parsed)
            results.append(row)

        # Sort descending by match score, then similarity
        results.sort(key=lambda r: (-r["match_score"], -r["similarity"], r["track_id"]))
        return results[: max(1, int(top_k))]


class Recommender:
    """Finds similar tracks using cosine similarity in the vector space."""

    def __init__(self, session=None):
        self.session = session

    def recommend_similar(self, track_id: str, top_k: int = 4, session=None) -> list[dict]:
        """Find nearest neighbor tracks for a given track_id."""
        session = session if session is not None else self.session
        target = get_track_by_id(track_id, session=session)
        if target is None:
            return []

        results = []
        for track in get_all_tracks(session=session):
            if track["track_id"] == track_id:
                continue
            similarity = _cosine(target.get("audio_embedding"), track.get("audio_embedding"))
            row = dict(track)
            row["similarity"] = round(float(similarity), 4)
            row["match_score"] = round(min(100.0, max(0.0, similarity * 100.0)), 1)
            results.append(row)

        results.sort(key=lambda r: (-r["similarity"], r["track_id"]))
        return results[: max(1, int(top_k))]

    @staticmethod
    def get_game_dev_presets() -> dict:
        """Pre-configured archetype queries for game audio design."""
        return {
            "Main Menu": {
                "query": "peaceful ambient celestial theme with slow tempo",
                "description": "Atmospheric introduction music",
            },
            "Exploration": {
                "query": "organic atmospheric environmental background music",
                "description": "Open world traversal theme",
            },
            "Village": {
                "query": "warm organic acoustic folk music with gentle rhythm",
                "description": "Safe haven hub music",
            },
            "Dungeon": {
                "query": "dark eerie drone with deep sub-bass and tension",
                "description": "Hostile mysterious underground music",
            },
            "Boss Fight": {
                "query": "heavy aggressive industrial boss battle with chaotic distortion",
                "description": "High-intensity climactic combat",
            },
            "Credits": {
                "query": "emotional melodic orchestral closing theme",
                "description": "Reflective finale music",
            },
        }
