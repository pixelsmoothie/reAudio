# reAudio

> **Fractal-Aware Semantic Audio Retrieval & Discovery — search sound by intent, not just tags.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Librosa](https://img.shields.io/badge/librosa-0.10+-F25CA2?logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00?logo=sqlalchemy&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-52%20passing-1DB954)

reAudio indexes any audio library by combining **mathematical fractal analysis** of the waveform, **multimodal CLAP embeddings**, and a **relational SQL layer** into a two-stage hybrid retrieval engine. Ask for *"terrifying cyberpunk boss fight under 2 minutes"* and it prunes the library with exact SQL constraints (duration, BPM, fractal roughness), re-ranks survivors by cross-modal semantic similarity, and explains every match with grounded, retrieval-augmented insights.

---

## System Architecture

```mermaid
flowchart TB
    subgraph ING["Ingestion Pipeline (cli.py / web /api/upload / synthesize_demo_data.py)"]
        A["Audio Files<br/>WAV · MP3 · FLAC · OGG"] --> B["core/dsp.py<br/>DSP + Fractal Analysis<br/>(Higuchi · Katz · 1/f^β)"]
        A --> C["core/models.py<br/>CLAP Embedder<br/>(offline fallback)"]
        B --> D["generate_semantic_profile<br/>genre / mood / tags"]
        B --> E[("core/database.py<br/>SQLite / MySQL<br/>tracks · features · semantic")]
        C --> E
        D --> E
    end

    subgraph RET["Two-Stage Hybrid Retrieval (core/retrieval.py)"]
        F["Natural Language Query<br/><i>'eerie dungeon drone under 2 min'</i>"] --> G["QueryParser<br/>regex → duration / BPM /<br/>fractal / genre constraints"]
        G --> H["Stage 1: filter_tracks_sql<br/>relational candidate pruning"]
        H --> I["Stage 2: cosine re-ranking<br/>text embedding ↔ audio embedding"]
        I --> J["Score fusion<br/>0.70 × semantic + 0.30 × acoustic alignment"]
        J --> K["RAGExplainer<br/>grounded match insights"]
    end

    E --> H
    K --> L["Web UI (FastAPI + vanilla JS)<br/>player · visualizer · upload · presets"]
    K --> M["CLI (cli.py)"]
```

---

## Core Pillars & Differentiators

### 1. Mathematical Fractal Analysis — measuring roughness, not just loudness

Conventional MIR features (BPM, spectral centroid, RMS) describe *what* the audio is but not how it *feels*. reAudio adds three nonlinear time-series descriptors computed in `core/dsp.py`:

| Metric | What it captures | Range in practice |
|---|---|---|
| **Higuchi Fractal Dimension** | Waveform self-similarity / roughness across lags k=1..16 — slope of ln⟨L(k)⟩ vs ln(1/k) | `1.00` pure sine (calm) → `2.00` white noise (chaos) |
| **Katz Fractal Dimension** | Planar curve complexity: `log10(n) / (log10(n) + log10(d/L))` | Shape complexity of the waveform |
| **1/f^β Spectral Exponent** | Power-law slope of the Welch PSD — tonal vs noisy balance | β≈0 white noise, β≈2 brown noise |

Because Higuchi FD compresses "acoustic chaos vs harmonic calm" into a single **filterable scalar**, it can live in a SQL column and power queries like `min_fractal_dim=1.5` — something a pure vector search cannot express. The demo synthesizer even uses it as a *closed-loop control signal*, bisecting a noise-bed knob until each procedurally generated track lands in its target roughness window (e.g. boss theme D≈1.73, ambient drone D≈1.12).

### 2. Multimodal Audio Intelligence — zero-shot text to audio search

`AudioIntelligenceModel` wraps **LAION CLAP** (`laion/larger_clap_music_and_speech`), a contrastively-trained audio/text encoder fine-tuned on music datasets. Audio and text are projected into the same 512-dim space, so any free-text query retrieves audio **without ever training on your library** — true zero-shot cross-modal retrieval.

Production hardening: if CLAP weights can't be downloaded (offline, rate-limited, no GPU), the model degrades transparently to a **deterministic fallback embedder** — a fixed-seed random projection of mel/spectral/fractal features for audio, and a hashing-trick bag-of-words embedder for text — both L2-normalized onto the same unit hypersphere. Every downstream component (search, recommendations, API) works identically either way, which makes tests and demos 100% reproducible.

### 3. Relational SQL Layer — why SQL beats vector-only search for hard constraints

"Under **2 minutes**", "**above 120** BPM", "genre = **Industrial**" are *exact predicates*, not similarities. A vector-only stack (Pinecone/FAISS) can only approximate them post-hoc, and can silently return a 9-minute track for an "under 2 minutes" query. reAudio models constraints as first-class columns in SQLAlchemy 2.0 (`Track`, `AudioFeatures`, `SemanticMetadata`) — SQLite by default, MySQL-compatible via `DATABASE_URL` — so:

- **Correctness**: `WHERE duration <= 120 AND bpm >= 120` is guaranteed, not probabilistic.
- **Explainability**: every filter maps to a human-readable verification line in the match explanation.
- **Portability**: standard SQLAlchemy, no vector-database vendor lock-in; embeddings live as blobs alongside rows.

### 4. Two-Stage Hybrid Retrieval & RAG Explainer

`HybridRetrievalEngine.search()` runs a retrieve-then-rank pipeline:

1. **Parse** — regex/heuristics lift numeric + genre constraints out of the query, leaving a clean semantic string for the embedder.
2. **Relational pruning** — `filter_tracks_sql` executes dynamic joined queries; if strict filters return nothing, it gracefully falls back to the full library so semantics can still rank.
3. **Vector re-ranking** — cosine similarity between the query embedding and each candidate's stored audio embedding.
4. **Fusion** — `match_score = 0.70 × semantic + 0.30 × acoustic alignment` (a margin-based score of how well BPM/HFD/duration satisfy the parsed constraints), rendered as a 0–100 % match.
5. **Explain** — `RAGExplainer` grounds its insights in *retrieved facts*: CLAP similarity, Higuchi FD interpretation, BPM/RMS energy verification, and per-constraint satisfaction checks. Every result answers *"why did this match?"*.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Seed the demo library: 8 procedurally synthesized tracks
#    (industrial boss, dungeon drone, village folk, synthwave, ...)
python synthesize_demo_data.py            # add --fallback to skip CLAP download

# 3. Launch the web app
python cli.py serve                       # or: python web/app.py
# open http://127.0.0.1:8000

# 4. Ingest your own music
python cli.py ingest "D:/Music/Game OST" --artist "Your Name"
```

Offline / CI environments: set `AUDIOMIND_FORCE_FALLBACK=1` (or pass `--force-fallback`) to skip the CLAP download and use the deterministic embedder. Custom database: set `DATABASE_URL`, e.g. `mysql+pymysql://user:pass@host/audiomind`.

### CLI Usage

```bash
# Index a single file or an entire directory
python cli.py ingest track.wav
python cli.py ingest ./library --artist "Studio Name"

# Hybrid intent search (constraints are parsed AND flag-controllable)
python cli.py search "terrifying cyberpunk boss fight under 2 minutes"
python cli.py search "calm exploration ambient" --max-duration 120 --max-hfd 1.3 --top-k 5

# Embedding-similar recommendations
python cli.py recommend iron_tyrant_boss --top-k 4

# Library statistics: duration, genres, BPM range, fractal distribution
python cli.py stats

# Start the web server
python cli.py serve --host 0.0.0.0 --port 8000
```

### REST API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/tracks` | All indexed tracks with DSP features + tags |
| `POST` | `/api/search` | Hybrid search: `{"query": "...", "duration_max": 120, "bpm_min": 120, "min_fractal_dim": 1.5, "genre": "Industrial", "top_k": 8}` |
| `GET` | `/api/recommend/{track_id}` | Top-4 embedding-similar tracks |
| `GET` | `/api/presets` | 6 game-dev archetype presets (Main Menu → Credits) |
| `GET` | `/api/audio/{track_id}` | Stream the audio file |
| `POST` | `/api/upload` | Upload + auto-index real audio (MP3/WAV/FLAC/OGG) |

### Web UI Highlights

- **Intent search bar** with Game-Developer-Mode preset pills (Main Menu, Exploration, Village, Dungeon, Boss Fight, Credits).
- **Collapsible filter panel**: duration slider, BPM range, and a Higuchi Fractal Roughness slider (Smooth 1.0 to Chaotic 2.0).
- **Track cards** with a color-coded fractal badge — Smooth/Harmonic (FD < 1.35), Textured/Rhythmic (1.35–1.6), Rough/Distorted (>= 1.6) — plus match-score bars and a *"Why did this match?"* AI explanation modal.
- **Player bar** with a live Web Audio API canvas visualizer.
- **Drag-and-drop upload**: real files are analyzed (BPM, spectral centroid, Higuchi FD), embedded, auto-tagged, and indexed on the fly.

---

## Project Structure

```
reAudio/
├── config.py                    # paths, DATABASE_URL, CLAP model name
├── cli.py                       # ingest / search / recommend / stats / serve
├── synthesize_demo_data.py      # procedural demo-track generator + ingestion
├── core/
│   ├── dsp.py                   # Higuchi FD, Katz FD, 1/f^β, feature extraction
│   ├── database.py              # SQLAlchemy 2.0 models, upserts, SQL filters
│   ├── models.py                # CLAP wrapper + deterministic fallback embedder
│   ├── retrieval.py             # QueryParser, HybridRetrievalEngine, Recommender
│   └── explainer.py             # RAG-style match explainer
├── web/
│   ├── app.py                   # FastAPI REST API + static mount
│   └── static/                  # index.html, styles.css, app.js
├── data/sample_audio/           # synthesized + ingested audio
└── tests/                       # 52 pytest tests (dsp, database, models, retrieval, web)
```

## Testing

```bash
pytest tests/ -v          # 52 tests: DSP math, DB upserts/filters, embeddings,
                          # hybrid retrieval, FastAPI endpoints (incl. upload)
```

---

## Technical Architectural Highlights

**1. Why use fractal dimensions instead of just standard audio features?**
Standard features (BPM, spectral centroid, RMS) are *local* descriptors — they average away the texture that distinguishes a chaotic distorted wall of sound from a smooth harmonic pad at the same tempo and loudness. The Higuchi Fractal Dimension quantifies self-similarity across multiple time lags: you reconstruct the waveform at lags k=1..16, measure normalized curve length ⟨L(k)⟩ at each scale, and the slope of ln⟨L(k)⟩ vs ln(1/k) is the dimension. A pure sine sits at D≈1.0 (a smooth curve), white noise at D≈2.0 (space-filling roughness). Crucially it's a single bounded scalar, so it can be stored in SQL and used as an exact filter — bridging nonlinear signal analysis and relational querying.

**2. Why hybrid SQL + vector retrieval instead of a pure vector database?**
Because user intent mixes two fundamentally different query types. "Boss fight music" is fuzzy/semantic — nearest-neighbor in embedding space is right. "Under 2 minutes, above 120 BPM" is exact/relational — approximate nearest-neighbor can't guarantee it and filtering after retrieval degrades recall unpredictably. reAudio executes the exact predicates as SQL over joined relational tables first (correctness + index-friendly pruning), then re-ranks only the survivors by cosine similarity. This also gives clean explainability (each filter becomes a verification line) and avoids vector-DB vendor lock-in.

**3. How does CLAP enable zero-shot search, and what happens offline?**
CLAP is contrastively pre-trained on hundreds of thousands of audio/text pairs so that matching pairs land close together in a shared 512-dim space. At index time we embed each track's audio; at query time we embed the text with the same encoder — no per-library training, no tags required. Cosine similarity in that space is relevance. For robustness, our wrapper catches any load/inference failure (no network, missing weights, runtime error) and transparently substitutes a deterministic fallback: a fixed-seed projection of mel/spectral/fractal features for audio and a hashing-trick text embedder, both L2-normalized to the same unit hypersphere. The retrieval API is identical in both modes, so the system degrades gracefully rather than failing.

**4. Explain the RAG explainer — how is it "retrieval-augmented"?**
Instead of asking an LLM to free-form justify a match, the explainer is *grounded strictly in retrieved structured facts*: the parsed query constraints, the fused match score, the CLAP cosine similarity, and the track's actual DSP values. It renders those facts as natural-language insights — the Higuchi FD value is interpreted ("1.72 indicates extreme acoustic roughness, matching the requested chaotic character"), BPM/RMS are verified against tempo intent, and every relational filter gets an explicit satisfied/relaxed line.

**5. How would you scale this to a million-track library?**
Three fronts. *Storage/retrieval*: move embeddings from SQLite blobs to pgvector or a dedicated ANN index (FAISS/HNSW) with product quantization; keep the relational filters in the same Postgres so stage-1 pruning stays exact and index-backed. *Ingestion*: decouple analysis into an asynchronous worker queue (Celery/RQ workers) since DSP + embedding is CPU/GPU-bound. *Serving*: ship the CLAP encoder behind a batching inference service (TorchServe/Triton), cache query embeddings for repeated intents, and precompute "preset" queries.
