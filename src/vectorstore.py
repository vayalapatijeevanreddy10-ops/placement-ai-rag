"""Build, save and load the FAISS vector store."""

import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import VECTORSTORE_DIR

BATCH_SIZE = 32


def _retry(fn, attempts: int = 4):
    """Run fn(); on failure (e.g. rate limit) wait and try again."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = 5 * (2 ** attempt)
            print(f"  Embedding call failed ({exc}). Retrying in {wait}s...")
            time.sleep(wait)


def build_vectorstore(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed chunks in small batches (avoids rate limits) and index them."""
    vectorstore = None
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = _retry(lambda b=batch: FAISS.from_documents(b, embeddings))
        else:
            _retry(lambda b=batch: vectorstore.add_documents(b))
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Path = VECTORSTORE_DIR) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_vectorstore(embeddings: Embeddings, path: Path = VECTORSTORE_DIR) -> FAISS | None:
    """Load a saved index, or return None if none exists yet."""
    if not (Path(path) / "index.faiss").exists():
        return None
    # The index is stored with pickle, so only load files YOU created.
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )