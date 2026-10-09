from pathlib import Path

from src.config import UPLOAD_DIR, VECTORSTORE_DIR
from src.loaders import load_document
from src.splitter import split_documents
import streamlit as st

from src.embeddings import get_embeddings
from src.llm import get_llm
from src.vectorstore import load_vectorstore
from src.rag_chain import answer_question

CATEGORIES = ["All", "Technical", "Aptitude", "HR", "Resume", "General"]

st.set_page_config(page_title="AI Placement Preparation Assistant", page_icon="🎓")

st.markdown(
    """
    <style>
    /* Sidebar: subtle gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B26 0%, #0E1117 100%);
        border-right: 1px solid #262D3D;
    }

    /* Hero header */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #7C5CFF 0%, #3B82F6 60%, #06B6D4 100%);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(124, 92, 255, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p  { margin: 6px 0 0 0; opacity: 0.9; font-size: 1rem; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: #161B26;
        border: 1px solid #262D3D;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* Buttons: rounded with a hover lift */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #333B52;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #7C5CFF;
        box-shadow: 0 6px 16px rgba(124, 92, 255, 0.3);
    }

    /* Sidebar stat card */
    .stat-card {
        padding: 14px 16px;
        border-radius: 12px;
        background: #1B2233;
        border: 1px solid #2A3248;
        margin-bottom: 12px;
    }
    .stat-card .num { font-size: 1.8rem; font-weight: 700; color: #A78BFA; }
    .stat-card .lbl { font-size: 0.8rem; opacity: 0.7; }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load backend once (cached) ----------
@st.cache_resource
def load_backend():
    embeddings = get_embeddings()
    llm = get_llm()
    vectorstore = load_vectorstore(embeddings)
    return embeddings, llm, vectorstore


embeddings, llm, vectorstore = load_backend()

if vectorstore is None:
    st.error("Vectorstore is None")
    st.stop()


# ---------- Chat history ----------
# Each message: {"role": "user"/"assistant", "content": str, "sources": list}
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎓 Placement Assistant")

    # ----- Upload -----
    st.subheader("📚 Upload documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )
    upload_category = st.selectbox("Category", CATEGORIES[1:])  # skips "All"

    if st.button("Process documents"):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            with st.spinner("Processing documents..."):
                for f in uploaded_files:
                    # 1. Save the file to disk
                    save_path = Path(UPLOAD_DIR) / f.name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(f.getbuffer())

                    # 2. Load it into LangChain Documents
                    docs = load_document(str(save_path))

                    # 3. Tag every page with the chosen category
                    for d in docs:
                        d.metadata["category"] = upload_category

                    # 4. Split into chunks, embed, and add to the existing index
                    chunks = split_documents(docs)
                    vectorstore.add_documents(chunks)

                # 5. Save the updated index to disk
                vectorstore.save_local(str(VECTORSTORE_DIR))
            st.success(f"Added {len(uploaded_files)} file(s).")

    st.divider()

    # ----- Knowledge base -----
    st.subheader("Knowledge Base")
    st.markdown(
    f"""
    <div class="stat-card">
        <div class="num">{vectorstore.index.ntotal}</div>
        <div class="lbl">CHUNKS INDEXED</div>
    </div>
    """,
    unsafe_allow_html=True,
)
    indexed_files = sorted(
        {
            Path(d.metadata.get("source", "unknown")).name
            for d in vectorstore.docstore._dict.values()
        }
    )
    st.write("Files")
    for name in indexed_files:
        st.write(f"• {name}")

    category = st.selectbox("Search category", CATEGORIES)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main page ----------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 AI Placement Preparation Assistant</h1>
        <p>Answers come only from your uploaded documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# Show the old messages first
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for source in msg["sources"]:
                    st.write(source)

# New question
question = st.chat_input("Ask your placement question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = answer_question(vectorstore, llm, question, category=category)

        st.write(result["answer"])

        sources = result.get("sources") or []
        if sources:
            with st.expander("📎 Sources"):
                for source in sources:
                    st.write(source)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": sources}
    )