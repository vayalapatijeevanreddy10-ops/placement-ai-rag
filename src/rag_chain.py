"""The complete RAG pipeline: question -> retrieve -> prompt -> Gemini -> answer."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.prompts import NOT_FOUND_MESSAGE, RAG_PROMPT
from src.retriever import retrieve, source_label


def format_context(results: list[tuple[Document, float]]) -> str:
    """Join the retrieved chunks into one context string."""
    parts = [f"[{source_label(doc)}]\n{doc.page_content}" for doc, _ in results]
    return "\n\n---\n\n".join(parts)


def answer_question(
    vectorstore: FAISS,
    llm: BaseChatModel,
    question: str,
    category: str | None = None,
) -> dict:
    """Return {'answer': str, 'sources': [ {label, snippet, distance}, ... ]}."""
    results = retrieve(vectorstore, question, category=category)

    # Nothing relevant found: don't even call the LLM.
    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": format_context(results), "question": question}
    )

    # If Gemini says it couldn't find the answer, don't show misleading sources.
    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    sources, seen = [], set()
    for doc, distance in results:
        label = source_label(doc)
        if label in seen:
            continue
        seen.add(label)
        sources.append(
            {
                "label": label,
                "snippet": doc.page_content[:300].strip(),
                "distance": round(float(distance), 3),
            }
        )
    return {"answer": answer, "sources": sources}