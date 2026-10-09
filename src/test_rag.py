"""Direct diagnostic test script (no Streamlit wrapper)."""

import sys
print("1. Python starting...")

try:
    from src.config import GOOGLE_API_KEY, GEMINI_MODEL
    print(f"2. Config loaded. Model target: {GEMINI_MODEL}")
except Exception as e:
    print(f"FAILED at Config: {e}")
    sys.exit(1)

try:
    print("3. Loading Hugging Face Embeddings model...")
    from src.embeddings import get_embeddings
    embed = get_embeddings()
    # Run a quick test vector computation
    v = embed.embed_query("test query")
    print(f"   Embeddings loaded successfully! Vector length: {len(v)}")
except Exception as e:
    print(f"FAILED at Embeddings: {e}")
    sys.exit(1)

try:
    print("4. Initializing Gemini LLM API connection...")
    from src.llm import get_llm
    llm = get_llm()
    res = llm.invoke("Say 'API Connection Successful'")
    print(f"   Gemini Response: {res.content}")
except Exception as e:
    print(f"FAILED at Gemini LLM: {e}")
    sys.exit(1)

print("\n--- ALL SYSTEMS OPERATIONAL ---")