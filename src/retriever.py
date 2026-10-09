"""Find the chunks most relevant to a question."""

import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import MAX_DISTANCE, RELATIVE_MARGIN, TOP_K

TEMPORARY_ERROR_MARKERS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")


def _with_retry(fn, attempts: int = 4):
    """Retry only temporary Google errors (503 / 429), waiting longer each time."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            is_temporary = any(m in str(exc) for m in TEMPORARY_ERROR_MARKERS)
            if not is_temporary or attempt == attempts - 1:
                raise
            wait = 5 * (2 ** attempt)   # 5s, 10s, 20s
            print(f"  Google service busy. Retrying in {wait}s...")
            time.sleep(wait)


def get_retriever(vectorstore: FAISS, k: int = TOP_K):
    """The standard LangChain retriever (as in the assignment brief)."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
    category: str | None = None,
) -> list[tuple[Document, float]]:
    """Similarity search: returns (chunk, distance) pairs. Lower distance = closer."""
    kwargs = {}
    if category and category != "All":
        kwargs["filter"] = {"category": category}   # metadata filtering

    return _with_retry(
        lambda: vectorstore.similarity_search_with_score(question, k=k, **kwargs)
    )


def retrieve(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
    margin: float = RELATIVE_MARGIN,
    category: str | None = None,
) -> list[tuple[Document, float]]:
    """Return chunks that are close enough to the question.

    Rule 1: distance must be under max_distance (else: nothing relevant).
    Rule 2: distance must be within `margin` of the best match
            (drops weak neighbours so sources stay precise).
    """
    results = search(vectorstore, question, k=k, category=category)

    close_enough = [(doc, score) for doc, score in results if score <= max_distance]
    if not close_enough:
        return []

    best = min(score for _, score in close_enough)
    return [(doc, score) for doc, score in close_enough if score <= best + margin]


def source_label(doc: Document) -> str:
    """Human-readable citation, e.g. 'java.pdf - Page 5'."""
    name = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page_number")
    return f"{name} - Page {page}" if page else name