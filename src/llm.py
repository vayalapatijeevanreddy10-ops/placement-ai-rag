"""Create the Gemini chat model."""

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL, require_api_key


def get_llm() -> ChatGoogleGenerativeAI:
    require_api_key()
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL)