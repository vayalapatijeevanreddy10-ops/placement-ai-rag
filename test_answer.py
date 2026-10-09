import streamlit as st

st.title("RAG Answer Diagnostic")

st.write("1. Starting")

# Load embeddings
from src.embeddings import get_embeddings

try:
    embeddings = get_embeddings()
    st.success("2. Embeddings loaded")
except Exception as e:
    st.error("Embeddings failed")
    st.exception(e)
    st.stop()

# Load LLM
from src.llm import get_llm

try:
    llm = get_llm()
    st.success("3. Gemini LLM loaded")
except Exception as e:
    st.error("Gemini LLM failed")
    st.exception(e)
    st.stop()

# Load vector store
from src.vectorstore import load_vectorstore

try:
    vectorstore = load_vectorstore(embeddings)

    if vectorstore is None:
        st.error("4. Vectorstore is None")
        st.stop()

    st.success("4. Vectorstore loaded")
    st.write("Number of vectors:", vectorstore.index.ntotal)

except Exception as e:
    st.error("Vectorstore failed")
    st.exception(e)
    st.stop()

# Load RAG chain
from src.rag_chain import answer_question

st.success("5. RAG function imported")

# Test question
question = st.text_input(
    "Enter a question",
    "What is polymorphism in Java?"
)

if st.button("Test RAG", type="primary"):

    st.write("6. Calling answer_question()...")

    try:
        result = answer_question(
            vectorstore,
            llm,
            question,
            category="All"
        )

        st.success("7. RAG answer generated successfully!")

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        st.write(result.get("sources", []))

    except Exception as e:
        st.error("8. RAG answer failed")
        st.exception(e)