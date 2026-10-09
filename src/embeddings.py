"""Create the embedding model (text -> vector)."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL, require_api_key


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    require_api_key()
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)