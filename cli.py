"""AudioMind command-line interface.

Subcommands:
    ingest     Index a single audio file or a directory of audio files.
    search     Hybrid (SQL + vector) intent search with RAG explanations.
    recommend  Embedding-similar track recommendations.
    stats      Library statistics (duration, genres, BPM, fractal ranges).
    serve      Start the FastAPI web server.

Set AUDIOMIND_FORCE_FALLBACK=1 (or pass --force-fallback) to use the
deterministic offline embedder instead of downloading CLAP weights.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import config
from core.database import get_all_tracks, get_track_by_id, init_db, save_track_record
from core.dsp import extract_dsp_features
from core.models import AudioIntelligenceModel, generate_semantic_profile
from core.retrieval import HybridRetrievalEngine, Recommender

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

_model: AudioIntelligenceModel | None = None


def get_model() -> AudioIntelligenceModel:
    """Lazy-load the embedding model so fast commands (stats) stay instant."""
    global _model
    if _model is None:
        force = os.getenv("AUDIOMIND_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
        print("Loading embedding model (CLAP with automatic offline fallback)...")
        _model = AudioIntelligenceModel(fallback_mode=force)
        mode = "deterministic fallback" if _model.fallback_mode else _model.model_name
        print(f"Embedder ready: {mode}")
    return _model


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> int:
    target = Path(args.path).expanduser().resolve()
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    elif target.is_file():
        if target.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"Unsupported file type '{target.suffix}'. "
                  f"Allowed: {', '.join(sorted(AUDIO_EXTENSIONS))}")
            return 1
        files = [target]
    else:
        print(f"Path not found: {target}")
        return 1

    if not files:
        print("No audio files found at that path.")
        return 1

    init_db()
    model = get_model()

    rows = []
    for path in files:
        try:
            features = extract_dsp_features(str(path))
        except Exception as exc:
            print(f"  ! skipping {path.name}: could not decode ({exc})")
            continue

        track_id = path.stem[:64]
        title = path.stem.replace("_", " ").replace("-", " ").strip().title()
        profile = generate_semantic_profile(features)
        audio_embed = model.embed_audio(str(path))
        semantic_text = (f"{profile['primary_genre']} {profile['mood']} "
                         f"{' '.join(profile['tags'])} {title}")
        text_embed = model.embed_text(semantic_text)

        track = {
            "track_id": track_id,
            "title": title,
            "artist": args.artist,
            "duration": features["duration"],
            "sample_rate": features["sample_rate"],
            "bitrate": features["bitrate"],
            "channels": features["channels"],
            "file_path": str(path),
        }
        save_track_record(track, features, profile,
                          audio_embed=audio_embed, text_embed=text_embed)
        rows.append((title, features))
        print(f"  + indexed {path.name}")

    print(f"\nIndexed {len(rows)} file(s) into {config.DB_PATH}\n")
    header = f"{'#':>3}  {'Title':<28} {'Duration':>9} {'BPM':>7} {'Higuchi FD':>11}"
    print(header)
    print("-" * len(header))
    for i, (title, feats) in enumerate(rows, start=1):
        print(f"{i:>3}  {title[:28]:<28} {feats['duration']:>8.1f}s "
              f"{feats['bpm']:>7.1f} {feats['higuchi_fractal_dimension']:>11.3f}")
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args) -> int:
    filters = {}
    for arg_name, filter_key in (
        ("max_duration", "duration_max"),
        ("min_bpm", "bpm_min"),
        ("max_bpm", "bpm_max"),
        ("min_hfd", "min_fractal_dim"),
        ("max_hfd", "max_fractal_dim"),
        ("genre", "genre"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            filters[filter_key] = value

    engine = HybridRetrievalEngine(model=get_model())
    results = engine.search(args.query, filters=filters or None, top_k=args.top_k)
    if not results:
        print("No tracks matched. Try relaxing filters or changing the query.")
        return 0

    header = (f"{'Rank':>4}  {'Title':<24} {'Artist':<18} {'Genre':<12} "
              f"{'BPM':>6} {'HFD':>5} {'Match':>8}")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        print(f"{i:>4}  {r['title'][:24]:<24} {r['artist'][:18]:<18} "
              f"{r['primary_genre'][:12]:<12} {r['bpm']:>6.1f} "
              f"{r['higuchi_fractal_dimension']:>5.2f} {r['match_score']:>7.1f}%")

    print()
    for i, r in enumerate(results, start=1):
        explanation = r["explanation"]
        print(f"#{i} {r['title']} — {explanation['score_pct']}% match")
        print(f"   {explanation['summary']}")
        for insight in explanation["insights"]:
            print(f"   • {insight}")
        print()
    return 0


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

def cmd_recommend(args) -> int:
    target = get_track_by_id(args.track_id)
    if target is None:
        print(f"Track '{args.track_id}' not found in the database.")
        return 1

    recommender = Recommender()
    results = recommender.recommend_similar(args.track_id, top_k=args.top_k)
    if not results:
        print("No other tracks in the library to compare against.")
        return 0

    print(f"Tracks similar to \"{target['title']}\" ({target['primary_genre']}):\n")
    header = f"{'Rank':>4}  {'Title':<28} {'Artist':<20} {'Genre':<12} {'Similarity':>10} {'Match':>8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        print(f"{i:>4}  {r['title'][:28]:<28} {r['artist'][:20]:<20} "
              f"{r['primary_genre'][:12]:<12} {r['similarity']:>10.4f} "
              f"{r['match_score']:>7.1f}%")
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def cmd_stats(args) -> int:
    tracks = get_all_tracks()
    if not tracks:
        print("The library is empty. Ingest audio with: python cli.py ingest <path>")
        return 0

    total_duration = sum(t["duration"] for t in tracks)
    genres = Counter(t["primary_genre"] for t in tracks)
    bpms = [t["bpm"] for t in tracks]
    hfds = [t["higuchi_fractal_dimension"] for t in tracks]

    print("AudioMind Library Statistics")
    print("=" * 46)
    print(f"Total tracks          : {len(tracks)}")
    hours, rem = divmod(total_duration, 3600)
    print(f"Total library duration: {int(hours)}h {rem / 60:.1f}m ({total_duration:.1f}s)")
    print(f"BPM range             : {min(bpms):.1f} - {max(bpms):.1f}")
    print(f"Higuchi FD min/mean/max: {min(hfds):.3f} / "
          f"{sum(hfds) / len(hfds):.3f} / {max(hfds):.3f}")

    print("\nGenre distribution:")
    for genre, count in genres.most_common():
        bar = "#" * max(1, round(40 * count / len(tracks)))
        print(f"  {genre:<18} {count:>3}  {100 * count / len(tracks):>5.1f}%  {bar}")

    smooth = sum(1 for d in hfds if d < 1.35)
    textured = sum(1 for d in hfds if 1.35 <= d < 1.6)
    rough = sum(1 for d in hfds if d >= 1.6)
    print("\nFractal texture distribution:")
    print(f"  Smooth/Harmonic  (FD < 1.35) : {smooth:>3}")
    print(f"  Textured/Rhythmic (1.35-1.6) : {textured:>3}")
    print(f"  Rough/Distorted (FD >= 1.6)  : {rough:>3}")
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    import uvicorn

    print(f"Starting AudioMind web server at http://{args.host}:{args.port}")
    uvicorn.run("web.app:app", host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiomind",
        description="AudioMind CLI — fractal-aware semantic audio retrieval",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a file or directory of audio")
    p_ingest.add_argument("path", help="audio file or directory")
    p_ingest.add_argument("--artist", default="Unknown Artist",
                          help="artist credit for ingested files")
    p_ingest.add_argument("--force-fallback", action="store_true",
                          help="use the deterministic offline embedder (no CLAP download)")

    p_search = sub.add_parser("search", help="hybrid natural-language search")
    p_search.add_argument("query", help="natural language intent query")
    p_search.add_argument("--max-duration", type=float, default=None)
    p_search.add_argument("--min-bpm", type=float, default=None)
    p_search.add_argument("--max-bpm", type=float, default=None)
    p_search.add_argument("--min-hfd", type=float, default=None,
                          help="minimum Higuchi fractal dimension (rougher)")
    p_search.add_argument("--max-hfd", type=float, default=None,
                          help="maximum Higuchi fractal dimension (smoother)")
    p_search.add_argument("--genre", default=None)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--force-fallback", action="store_true")

    p_recommend = sub.add_parser("recommend", help="similar-track recommendations")
    p_recommend.add_argument("track_id")
    p_recommend.add_argument("--top-k", type=int, default=4)

    sub.add_parser("stats", help="library statistics")

    p_serve = sub.add_parser("serve", help="start the web server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "force_fallback", False):
        os.environ["AUDIOMIND_FORCE_FALLBACK"] = "1"

    handlers = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "recommend": cmd_recommend,
        "stats": cmd_stats,
        "serve": cmd_serve,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
