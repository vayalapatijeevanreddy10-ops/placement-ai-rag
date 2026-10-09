import streamlit as st

st.title("RAG Vector Store Diagnostic")

st.write("1. Starting")

from src.embeddings import get_embeddings
st.write("2. Importing embeddings")

try:
    embeddings = get_embeddings()
    st.success("3. Embeddings initialized")
except Exception as e:
    st.error("Embedding initialization failed")
    st.exception(e)
    st.stop()

from src.vectorstore import load_vectorstore
st.write("4. Vectorstore function imported")

try:
    vectorstore = load_vectorstore(embeddings)

    if vectorstore is None:
        st.warning("5. Vectorstore returned None")
    else:
        st.success("5. Vectorstore loaded successfully")
        st.write(vectorstore)

        try:
            st.write("Number of vectors:", vectorstore.index.ntotal)
        except Exception:
            st.write("Could not read vector count.")

except Exception as e:
    st.error("Vectorstore loading failed")
    st.exception(e)
    st.stop()

st.success("🎉 Vectorstore diagnostic completed!")