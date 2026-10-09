"""Smoke test: can we reach Gemini through LangChain?"""

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL, require_api_key


def main() -> None:
    require_api_key()
    print(f"Using model: {GEMINI_MODEL}")

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    chain = llm | StrOutputParser()  # model output -> plain string

    answer = chain.invoke("Reply with exactly these two words: Setup OK")
    print("Gemini says:", answer)


if __name__ == "__main__":
    main()