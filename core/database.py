"""Relational database layer for AudioMind.

SQLAlchemy 2.0 models for tracks, audio features, and semantic metadata.
Stores vector embeddings as raw binary blobs in SQLite/MySQL.
"""

import datetime
import json
from typing import Optional

import numpy as np
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

import config


# ---------------------------------------------------------------------------
# 1. ORM Table Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Track(Base):
    """Core audio track metadata."""
    __tablename__ = "tracks"

    track_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    artist: Mapped[str] = mapped_column(String(255), nullable=False, default="Unknown Artist")
    duration: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    bitrate: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    # 1-to-1 relationships with CASCADE delete
    features: Mapped["AudioFeatures"] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan", single_parent=True
    )
    semantic: Mapped["SemanticMetadata"] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan", single_parent=True
    )


class AudioFeatures(Base):
    """Acoustic and fractal analysis features extracted from the waveform."""
    __tablename__ = "audio_features"

    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.track_id", ondelete="CASCADE"), primary_key=True)
    bpm: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    rms_energy: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_centroid: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_rolloff: Mapped[float] = mapped_column(Float, nullable=False)
    zero_crossing_rate: Mapped[float] = mapped_column(Float, nullable=False)
    dynamic_range_db: Mapped[float] = mapped_column(Float, nullable=False)
    higuchi_fractal_dimension: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    katz_fractal_dimension: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_fractal_beta: Mapped[float] = mapped_column(Float, nullable=False)

    track: Mapped["Track"] = relationship(back_populates="features")


class SemanticMetadata(Base):
    """Semantic tags, mood, genre, and 512-dim AI vector embeddings."""
    __tablename__ = "semantic_metadata"

    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.track_id", ondelete="CASCADE"), primary_key=True)
    primary_genre: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    mood: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    generated_description: Mapped[str] = mapped_column(Text, nullable=False)
    audio_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    text_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    track: Mapped["Track"] = relationship(back_populates="semantic")


# ---------------------------------------------------------------------------
# 2. Vector BLOB Serialization Helpers
# ---------------------------------------------------------------------------

def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serialize a numpy vector to raw float32 bytes for BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vector(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize raw bytes back into a 1D float32 numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Database Engine & Session Management
# ---------------------------------------------------------------------------

_active_engine: Optional[Engine] = None


def _create_engine_from_url(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def _get_active_engine() -> Engine:
    global _active_engine
    if _active_engine is None:
        _active_engine = _create_engine_from_url(config.DATABASE_URL)
    return _active_engine


def init_db(engine_override: Optional[Engine] = None) -> Engine:
    """Create all tables in the database."""
    global _active_engine
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = engine_override if engine_override is not None else _get_active_engine()
    _active_engine = engine
    Base.metadata.create_all(engine)
    return engine


def get_session(engine_override: Optional[Engine] = None) -> Session:
    """Create a new SQLAlchemy session."""
    engine = engine_override if engine_override is not None else _get_active_engine()
    return Session(bind=engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# 4. Row Dictionary Converter
# ---------------------------------------------------------------------------

def _row_to_dict(track: Track, features: Optional[AudioFeatures], semantic: Optional[SemanticMetadata]) -> dict:
    """Convert joined database records into a single unified track dictionary."""
    data = {
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "duration": track.duration,
        "sample_rate": track.sample_rate,
        "bitrate": track.bitrate,
        "channels": track.channels,
        "file_path": track.file_path,
        "created_at": track.created_at,
        "bpm": features.bpm if features else None,
        "rms_energy": features.rms_energy if features else None,
        "spectral_centroid": features.spectral_centroid if features else None,
        "spectral_rolloff": features.spectral_rolloff if features else None,
        "zero_crossing_rate": features.zero_crossing_rate if features else None,
        "dynamic_range_db": features.dynamic_range_db if features else None,
        "higuchi_fractal_dimension": features.higuchi_fractal_dimension if features else None,
        "katz_fractal_dimension": features.katz_fractal_dimension if features else None,
        "spectral_fractal_beta": features.spectral_fractal_beta if features else None,
        "primary_genre": semantic.primary_genre if semantic else None,
        "mood": semantic.mood if semantic else None,
        "tags": json.loads(semantic.tags_json or "[]") if semantic else [],
        "generated_description": semantic.generated_description if semantic else None,
        "audio_embedding": blob_to_vector(semantic.audio_embedding) if semantic else None,
        "text_embedding": blob_to_vector(semantic.text_embedding) if semantic else None,
    }
    return data


def _base_query():
    """Build the base joined SQL query for all 3 tables."""
    return (
        select(Track, AudioFeatures, SemanticMetadata)
        .outerjoin(AudioFeatures, Track.track_id == AudioFeatures.track_id)
        .outerjoin(SemanticMetadata, Track.track_id == SemanticMetadata.track_id)
    )


# ---------------------------------------------------------------------------
# 5. Data Access Functions (Upsert, Get, Filter)
# ---------------------------------------------------------------------------

def _upsert_record(session: Session, model_class, pkey_val: str, field_data: dict):
    """Simple helper to insert or update a record by primary key."""
    record = session.get(model_class, pkey_val)
    valid_cols = set(model_class.__table__.columns.keys())
    clean_data = {k: v for k, v in field_data.items() if k in valid_cols and k != "track_id"}

    if record is None:
        record = model_class(track_id=pkey_val, **clean_data)
        session.add(record)
    else:
        for key, val in clean_data.items():
            setattr(record, key, val)


def save_track_record(track_dict: dict, features_dict: dict, semantic_dict: dict,
                      audio_embed: Optional[np.ndarray] = None,
                      text_embed: Optional[np.ndarray] = None,
                      session: Optional[Session] = None) -> str:
    """Upsert a track, its audio features, and semantic metadata."""
    owns_session = session is None
    session = get_session() if owns_session else session

    try:
        track_id = track_dict["track_id"]

        # 1. Upsert Track
        _upsert_record(session, Track, track_id, track_dict)

        # 2. Upsert AudioFeatures
        if features_dict:
            _upsert_record(session, AudioFeatures, track_id, features_dict)

        # 3. Upsert SemanticMetadata
        if semantic_dict:
            sem_data = dict(semantic_dict)
            if "tags" in sem_data:
                sem_data["tags_json"] = json.dumps(sem_data["tags"], ensure_ascii=False)
            if audio_embed is not None:
                sem_data["audio_embedding"] = vector_to_blob(audio_embed)
            if text_embed is not None:
                sem_data["text_embedding"] = vector_to_blob(text_embed)
            _upsert_record(session, SemanticMetadata, track_id, sem_data)

        if owns_session:
            session.commit()
        return track_id
    finally:
        if owns_session:
            session.close()


def get_track_by_id(track_id: str, session: Optional[Session] = None) -> Optional[dict]:
    """Retrieve a single track by its track_id."""
    owns_session = session is None
    session = get_session() if owns_session else session
    try:
        row = session.execute(_base_query().where(Track.track_id == track_id)).first()
        return _row_to_dict(*row) if row else None
    finally:
        if owns_session:
            session.close()


def get_all_tracks(session: Optional[Session] = None) -> list[dict]:
    """Return all tracks in the database."""
    owns_session = session is None
    session = get_session() if owns_session else session
    try:
        rows = session.execute(_base_query().order_by(Track.track_id)).all()
        return [_row_to_dict(*row) for row in rows]
    finally:
        if owns_session:
            session.close()


def filter_tracks_sql(duration_min: Optional[float] = None,
                      duration_max: Optional[float] = None,
                      bpm_min: Optional[float] = None,
                      bpm_max: Optional[float] = None,
                      min_fractal_dim: Optional[float] = None,
                      max_fractal_dim: Optional[float] = None,
                      genre: Optional[str] = None,
                      mood: Optional[str] = None,
                      tag_keyword: Optional[str] = None,
                      session: Optional[Session] = None) -> list[dict]:
    """Execute dynamic SQL filtering across joined tables for all provided constraints."""
    owns_session = session is None
    session = get_session() if owns_session else session

    try:
        stmt = _base_query()

        # Apply numeric range filters
        if duration_min is not None:
            stmt = stmt.where(Track.duration >= duration_min)
        if duration_max is not None:
            stmt = stmt.where(Track.duration <= duration_max)
        if bpm_min is not None:
            stmt = stmt.where(AudioFeatures.bpm >= bpm_min)
        if bpm_max is not None:
            stmt = stmt.where(AudioFeatures.bpm <= bpm_max)
        if min_fractal_dim is not None:
            stmt = stmt.where(AudioFeatures.higuchi_fractal_dimension >= min_fractal_dim)
        if max_fractal_dim is not None:
            stmt = stmt.where(AudioFeatures.higuchi_fractal_dimension <= max_fractal_dim)

        # Apply categorical filters
        if genre is not None:
            stmt = stmt.where(SemanticMetadata.primary_genre == genre)
        if mood is not None:
            stmt = stmt.where(SemanticMetadata.mood == mood)
        if tag_keyword is not None:
            stmt = stmt.where(SemanticMetadata.tags_json.ilike(f"%{tag_keyword}%"))

        rows = session.execute(stmt.order_by(Track.track_id)).all()
        return [_row_to_dict(*row) for row in rows]
    finally:
        if owns_session:
            session.close()


def get_database_dump(session: Optional[Session] = None) -> dict:
    """Return raw table dumps and schema metadata for database exploration."""
    owns_session = session is None
    session = get_session() if owns_session else session
    try:
        t_rows = session.execute(select(Track).order_by(Track.track_id)).scalars().all()
        tracks_data = [
            {
                "track_id": t.track_id,
                "title": t.title,
                "artist": t.artist,
                "duration": round(t.duration, 2),
                "sample_rate": t.sample_rate,
                "bitrate": t.bitrate,
                "channels": t.channels,
                "file_path": t.file_path,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in t_rows
        ]

        f_rows = session.execute(select(AudioFeatures).order_by(AudioFeatures.track_id)).scalars().all()
        features_data = [
            {
                "track_id": f.track_id,
                "bpm": round(f.bpm, 1),
                "rms_energy": round(f.rms_energy, 4),
                "spectral_centroid": round(f.spectral_centroid, 1),
                "spectral_rolloff": round(f.spectral_rolloff, 1),
                "zero_crossing_rate": round(f.zero_crossing_rate, 4),
                "dynamic_range_db": round(f.dynamic_range_db, 1),
                "higuchi_fractal_dimension": round(f.higuchi_fractal_dimension, 4),
                "katz_fractal_dimension": round(f.katz_fractal_dimension, 4),
                "spectral_fractal_beta": round(f.spectral_fractal_beta, 4),
            }
            for f in f_rows
        ]

        s_rows = session.execute(select(SemanticMetadata).order_by(SemanticMetadata.track_id)).scalars().all()
        semantic_data = []
        for s in s_rows:
            tags = []
            try:
                tags = json.loads(s.tags_json)
            except Exception:
                pass

            audio_preview = []
            if s.audio_embedding:
                arr = blob_to_vector(s.audio_embedding)
                if arr is not None:
                    audio_preview = [round(float(x), 4) for x in arr[:6]]

            text_preview = []
            if s.text_embedding:
                arr = blob_to_vector(s.text_embedding)
                if arr is not None:
                    text_preview = [round(float(x), 4) for x in arr[:6]]

            semantic_data.append({
                "track_id": s.track_id,
                "primary_genre": s.primary_genre,
                "mood": s.mood,
                "tags": tags,
                "generated_description": s.generated_description,
                "audio_embedding": {
                    "bytes": len(s.audio_embedding) if s.audio_embedding else 0,
                    "dims": 512,
                    "dtype": "float32",
                    "preview": audio_preview,
                },
                "text_embedding": {
                    "bytes": len(s.text_embedding) if s.text_embedding else 0,
                    "dims": 512,
                    "dtype": "float32",
                    "preview": text_preview,
                },
            })

        return {
            "tables": {
                "tracks": tracks_data,
                "audio_features": features_data,
                "semantic_metadata": semantic_data,
            },
            "summary": {
                "engine": "SQLite 3.x / SQLAlchemy 2.0 ORM",
                "tables_count": 3,
                "total_tracks": len(tracks_data),
                "foreign_keys": "audio_features.track_id -> tracks.track_id (CASCADE), semantic_metadata.track_id -> tracks.track_id (CASCADE)",
                "indexed_columns": ["tracks.duration", "audio_features.bpm", "audio_features.higuchi_fractal_dimension", "semantic_metadata.primary_genre", "semantic_metadata.mood"],
                "vector_storage": "Raw L2-normalized 512-dim float32 binary BLOBs (2,048 bytes per embedding)",
            }
        }
    finally:
        if owns_session:
            session.close()
