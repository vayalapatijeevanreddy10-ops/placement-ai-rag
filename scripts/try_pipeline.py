from src.embeddings import get_embeddings
from src.llm import get_llm
from src.rag_chain import answer_question
from src.vectorstore import load_vectorstore


def main() -> None:
    vectorstore = load_vectorstore(get_embeddings())
    if vectorstore is None:
        print("No index found. Run scripts.build_index first.")
        return
    llm = get_llm()

    print("Ask questions (type 'quit' to exit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        result = answer_question(vectorstore, llm, question)
        print(f"\nAI: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  - {s['label']}  (distance {s['distance']})")


if __name__ == "__main__":
    main()