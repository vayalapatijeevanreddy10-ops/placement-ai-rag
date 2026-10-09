import sys

from src.loaders import load_document
from src.splitter import split_documents


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.try_chunks data/yourfile.pdf")
        return

    docs = load_document(sys.argv[1], category="Technical")
    print(f"Loaded {len(docs)} page(s)/section(s)")
    print("First document metadata:", docs[0].metadata)

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    print("Sample chunk metadata:", chunks[0].metadata)
    print("--- Sample chunk text ---")
    print(chunks[0].page_content[:500])


if __name__ == "__main__":
    main()