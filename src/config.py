"""Central place for all settings. Other files import values from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- API / models ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
TEMPERATURE = 0.6

# --- Paths ---
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# --- Chunking ---
CHUNK_SIZE = 1000      # max characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks

# --- Retrieval ---
TOP_K = 3              # how many chunks to retrieve
MAX_DISTANCE = 0.80    # chunks farther than this are treated as irrelevant     # placeholder: we calibrate it later
RELATIVE_MARGIN = 0.10  # also drop chunks much worse than the best match
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def require_api_key() -> str:
    """Return the API key, or raise a clear error if it's missing."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith(("your_", "paste_")):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Create a .env file in the project "
            "root (copy .env.example) and paste your key from "
            "https://aistudio.google.com/apikey"
        )
    return GOOGLE_API_KEY