import sys

from src.embeddings import get_embeddings
from src.retriever import search, source_label
from src.vectorstore import load_vectorstore


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is polymorphism in Java?"
    vectorstore = load_vectorstore(get_embeddings())
    if vectorstore is None:
        print("No index found. Run scripts.build_index first.")
        return

    print(f"Question: {question}\n")
    for doc, distance in search(vectorstore, question, k=5):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"distance={distance:.3f} | {source_label(doc)}\n    {preview}...\n")


if __name__ == "__main__":
    main()