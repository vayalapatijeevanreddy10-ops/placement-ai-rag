import streamlit as st

st.title("RAG Diagnostic")

st.write("1. Streamlit loaded")

from src.config import UPLOAD_DIR, VECTORSTORE_DIR

st.write("2. config.py loaded")

from src.embeddings import get_embeddings

st.write("3. embeddings.py imported")

from src.llm import get_llm

st.write("4. llm.py imported")

from src.loaders import load_document

st.write("5. loaders.py imported")

from src.splitter import split_documents

st.write("6. splitter.py imported")

from src.vectorstore import (
    build_vectorstore,
    load_vectorstore,
    save_vectorstore,
)

st.write("7. vectorstore.py imported")

from src.rag_chain import answer_question

st.write("8. rag_chain.py imported")

st.success("All RAG modules imported successfully!")