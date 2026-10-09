"""Load PDF / TXT / DOCX files into LangChain Document objects."""

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.config import SUPPORTED_EXTENSIONS


def load_document(file_path: str | Path, category: str = "General") -> list[Document]:
    """Read one file and return its Documents with clean metadata."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Use: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()

    # Drop pages with no text (blank pages, or scanned images)
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError(
            f"No readable text found in '{path.name}'. "
            "If it is a scanned PDF, it needs OCR first."
        )

    for doc in docs:
        clean = {"source": path.name, "category": category}
        if ext == ".pdf":
            # PyPDFLoader counts pages from 0; humans count from 1
            clean["page_number"] = doc.metadata.get("page", 0) + 1
        doc.metadata = clean  # drop everything else (producer, dates, ...)

    return docs