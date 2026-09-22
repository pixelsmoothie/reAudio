import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "sample_audio"
DB_PATH = DATA_DIR / "reaudio.db" if (DATA_DIR / "reaudio.db").exists() else (DATA_DIR / "audiomind.db" if (DATA_DIR / "audiomind.db").exists() else DATA_DIR / "reaudio.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
CLAP_MODEL_NAME = "laion/larger_clap_music_and_speech"
