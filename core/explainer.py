"""RAG-style match explainer for AudioMind search results.

Generates human-readable diagnostic insights explaining WHY a song matched
a user's natural language search query.
"""

from typing import Optional


class RAGExplainer:
    """Generates clear diagnostic explanations for hybrid search results."""

    def explain(self, query: str, track: dict, score: float,
                parsed_constraints: Optional[dict] = None) -> dict:
        """Create structured match insights for a track."""
        parsed = parsed_constraints or {}
        score_pct = round(float(score), 1)
        title = track.get("title", "this track")
        insights = []

        # 1. Semantic Similarity Insight (AI embedding alignment)
        sim = track.get("similarity")
        if sim is None:
            insights.append(f'Semantic alignment: audio embedding of "{title}" aligns with query intent "{query}".')
        else:
            sim = float(sim)
            strength = "strong" if sim >= 0.5 else ("moderate" if sim >= 0.2 else "weak")
            insights.append(
                f'Semantic alignment: {strength} CLAP cosine similarity ({sim:.2f}) '
                f'between the query "{query}" and the track\'s audio embedding.'
            )

        # 2. Fractal Texture Insight (Waveform roughness)
        hfd = track.get("higuchi_fractal_dimension")
        if hfd is not None:
            hfd = float(hfd)
            if hfd >= 1.60:
                texture = "extreme acoustic roughness and distortion"
            elif hfd >= 1.45:
                texture = "textured, rhythmically complex detail"
            elif hfd >= 1.25:
                texture = "mild organic texture"
            else:
                texture = "a smooth, calm, self-similar surface"

            note = f"Higuchi Fractal Dimension of {hfd:.2f} indicates {texture}"
            if parsed.get("min_fractal_dim") and hfd >= float(parsed["min_fractal_dim"]):
                note += ", matching the requested chaotic/intense mood"
            elif parsed.get("max_fractal_dim") and hfd <= float(parsed["max_fractal_dim"]):
                note += ", matching the requested calm/smooth mood"
            insights.append(note + ".")

        # 3. Tempo & Loudness Verification
        bpm = track.get("bpm")
        rms = track.get("rms_energy")
        if bpm is not None and rms is not None:
            energy_desc = "driving high-energy" if float(rms) > 0.15 else "low-energy ambient"
            insights.append(f"Tempo is {float(bpm):.0f} BPM with {float(rms):.2f} RMS energy ({energy_desc}).")

        # 4. Filter Verification (Duration, BPM, Genre)
        dur = track.get("duration")
        dur_max = parsed.get("duration_max")
        if dur is not None and dur_max is not None:
            if float(dur) <= float(dur_max):
                insights.append(f"Duration {float(dur):.0f}s satisfies the requested <= {float(dur_max):.0f}s limit.")
            else:
                insights.append(f"Duration {float(dur):.0f}s was relaxed during ranking (limit: {float(dur_max):.0f}s).")

        bpm_min = parsed.get("bpm_min")
        if bpm is not None and bpm_min is not None:
            status = "satisfied" if float(bpm) >= float(bpm_min) else "relaxed"
            insights.append(f"BPM minimum constraint {status}: {float(bpm):.0f} vs {float(bpm_min):.0f} target.")

        genre = parsed.get("genre")
        if genre:
            track_genre = track.get("primary_genre")
            match_status = "matches" if track_genre == genre else "partially matches"
            insights.append(f"Genre: '{track_genre}' {match_status} requested '{genre}'.")

        # Summary line
        summary = (
            f'"{title}" matches "{query}" with {score_pct:.1f}% confidence based on '
            f"semantic alignment and acoustic fit."
        )

        return {
            "score_pct": score_pct,
            "summary": summary,
            "insights": insights,
        }
