import streamlit as st

st.set_page_config(
    page_title="AI Placement Preparation Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("AI Placement Preparation Assistant")
st.caption("Answers come only from your uploaded documents.")

with st.sidebar:
    st.title("🎓 Placement Assistant")

    st.subheader("📚 Upload documents")

    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    category = st.selectbox(
        "Category",
        ["Technical", "Aptitude", "HR", "Resume", "General"]
    )

    if st.button(
        "Process documents",
        type="primary",
        disabled=not uploaded
    ):
        st.success("Files received!")

    st.divider()

    st.subheader("Knowledge base")

    st.metric("Chunks indexed", 15)

st.write("### Try asking a question")

question = st.chat_input(
    "Ask your placement preparation question..."
)

if question:
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        st.write("This is a UI test. RAG response will appear here.")