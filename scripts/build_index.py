"""Build the FAISS index from one or more files and save it to disk."""

import argparse

from src.embeddings import get_embeddings
from src.loaders import load_document
from src.splitter import split_documents
from src.vectorstore import build_vectorstore, save_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="PDF / TXT / DOCX files")
    parser.add_argument("--category", default="General")
    args = parser.parse_args()

    all_chunks = []
    for file in args.files:
        docs = load_document(file, category=args.category)
        chunks = split_documents(docs)
        print(f"{file}: {len(docs)} page(s) -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print("Embedding and indexing (this can take a minute)...")
    vectorstore = build_vectorstore(all_chunks, get_embeddings())
    save_vectorstore(vectorstore)
    print(f"Done. Indexed {len(all_chunks)} chunks into the 'vectorstore' folder.")


if __name__ == "__main__":
    main()