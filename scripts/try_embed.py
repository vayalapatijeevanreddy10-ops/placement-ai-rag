import numpy as np

from src.embeddings import get_embeddings


def cosine(a, b) -> float:
    """Similarity between two vectors: 1.0 = same direction, 0 = unrelated."""
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    emb = get_embeddings()
    query_vec = emb.embed_query("What is polymorphism in Java?")
    print("Vector length:", len(query_vec))
    print("First 5 numbers:", [round(x, 4) for x in query_vec[:5]])

    texts = [
        "Polymorphism lets one interface take many forms in object-oriented programming.",
        "Method overriding lets a subclass redefine a method of its parent class.",
        "A balanced diet includes fruits, vegetables and protein.",
    ]
    vectors = emb.embed_documents(texts)

    print("\nSimilarity to the question (higher = more similar):")
    for text, vec in zip(texts, vectors):
        print(f"  {cosine(query_vec, vec):.3f}  {text}")


if __name__ == "__main__":
    main()