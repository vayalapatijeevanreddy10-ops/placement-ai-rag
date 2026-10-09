"""Prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND_MESSAGE = (
    "I could not find this information in the uploaded "
    "placement preparation material."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI placement preparation assistant.\n"
            "Answer the student's question using ONLY the provided context.\n"
            "If the answer cannot be found in the context, reply exactly with:\n"
            f'"{NOT_FOUND_MESSAGE}"\n'
            "Do not invent information or use outside knowledge.\n"
            "Be clear and concise. Use bullet points for lists of topics.",
        ),
        (
            "human",
            "Context:\n{context}\n\nStudent Question:\n{question}\n\nAnswer:",
        ),
    ]
)