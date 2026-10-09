import streamlit as st

st.title("🔥 NEW APP.PY IS RUNNING")
st.write("If you can see this, Streamlit is executing this exact app.py file.")

from src.embeddings import get_embeddings

from src.embeddings import get_embeddings
from src.llm import get_llm
from src.vectorstore import load_vectorstore
from src.rag_chain import answer_question


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Placement Preparation Assistant",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("🎓 AI Placement Preparation Assistant")

st.caption(
    "Answers come only from your uploaded placement documents."
)


# ---------------------------------------
# Initialize models
# ---------------------------------------

@st.cache_resource
def initialize_models():

    embeddings = get_embeddings()
    llm = get_llm()

    return embeddings, llm


try:

    with st.spinner("Initializing AI models..."):

        embeddings, llm = initialize_models()

except Exception as e:

    st.error("Failed to initialize AI models.")

    st.exception(e)

    st.stop()


# ---------------------------------------
# Load vector database
# ---------------------------------------

try:

    vectorstore = load_vectorstore(
        embeddings
    )

except Exception as e:

    st.error("Failed to load vector database.")

    st.exception(e)

    st.stop()


if vectorstore is None:

    st.warning(
        "No vector database found."
    )

    st.stop()


# ---------------------------------------
# Knowledge base information
# ---------------------------------------

st.success(
    f"Knowledge base loaded successfully — "
    f"{vectorstore.index.ntotal} chunks"
)


# ---------------------------------------
# Chat history
# ---------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------
# Display previous messages
# ---------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ---------------------------------------
# Chat input
# ---------------------------------------

question = st.chat_input(
    "Ask a question about your placement material..."
)


# ---------------------------------------
# RAG
# ---------------------------------------

if question:

    # Display user question

    with st.chat_message("user"):

        st.markdown(question)


    # Save user question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            try:

                result = answer_question(
                    vectorstore,
                    llm,
                    question,
                    category="All"
                )

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

                sources = result.get(
                    "sources",
                    []
                )

            except Exception as e:

                answer = (
                    "An error occurred while "
                    "generating the answer."
                )

                sources = []

                st.exception(e)


        # Display answer

        st.markdown(answer)


        # Display sources

        if sources:

            with st.expander(
                f"📎 Sources ({len(sources)})"
            ):

                for source in sources:

                    st.markdown(
                        f"**{source.get('label', 'Source')}**"
                    )

                    if source.get("snippet"):

                        st.caption(
                            source["snippet"]
                        )

                    if "distance" in source:

                        st.caption(
                            f"Distance: "
                            f"{source['distance']:.4f}"
                        )


    # Save assistant answer

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )