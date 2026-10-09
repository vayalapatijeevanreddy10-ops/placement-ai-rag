import streamlit as st

st.set_page_config(
    page_title="RAG Test",
    page_icon="🎓"
)

st.title("🎓 RAG Full Pipeline Test")

# -----------------------------
# Load models
# -----------------------------

st.write("### 1. Loading models...")

from src.embeddings import get_embeddings
from src.llm import get_llm

embeddings = get_embeddings()
llm = get_llm()

st.success("Models loaded")


# -----------------------------
# Load vectorstore
# -----------------------------

st.write("### 2. Loading vectorstore...")

from src.vectorstore import load_vectorstore

vectorstore = load_vectorstore(embeddings)

if vectorstore is None:
    st.error("Vectorstore is None")
    st.stop()

st.success(
    f"Vectorstore loaded — {vectorstore.index.ntotal} vectors"
)


# -----------------------------
# Load RAG function
# -----------------------------

st.write("### 3. Loading RAG chain...")

from src.rag_chain import answer_question

st.success("RAG chain loaded")


# -----------------------------
# Question
# -----------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    st.chat_message(
        "user"
    ).write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                result = answer_question(
                    vectorstore,
                    llm,
                    question,
                    category="All"
                )

                st.write(
                    result["answer"]
                )

                st.write("### Sources")

                for source in result.get(
                    "sources",
                    []
                ):

                    st.write(
                        source
                    )

            except Exception as e:

                st.error(
                    "RAG ERROR"
                )

                st.exception(e)