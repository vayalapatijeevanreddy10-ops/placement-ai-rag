import streamlit as st

st.title("RAG Model Diagnostic")

st.write("1. Starting")

from src.embeddings import get_embeddings
st.write("2. Embedding function imported")

try:
    embeddings = get_embeddings()
    st.success("3. Embedding model initialized successfully")
    st.write(embeddings)
except Exception as e:
    st.error("Embedding model failed")
    st.exception(e)
    st.stop()


from src.llm import get_llm
st.write("4. LLM function imported")

try:
    llm = get_llm()
    st.success("5. Gemini LLM initialized successfully")
    st.write(llm)
except Exception as e:
    st.error("Gemini LLM failed")
    st.exception(e)
    st.stop()

st.success("🎉 Both models initialized successfully!")