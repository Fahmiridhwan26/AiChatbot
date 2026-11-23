from pinecone.exceptions import PineconeException

from embedder import embed_texts
from store import NAMESPACE, get_index


def matches_to_results(response):
    """Turn Pinecone's reply into the simple dictionaries the rest of our code uses."""
    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "source": meta.get("source", "unknown"),
            "heading": meta.get("heading"),
            "page": meta.get("page"),
            "text": meta.get("text", ""),
            "score": match.score,
        })
    return results


def retrieve(question, index, top_k=2):
    """Find the top_k chunks most similar in meaning to the question."""
    question_vector = embed_texts([question])[0].tolist()
    response = index.query(
        vector=question_vector,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,    # without this, Pinecone returns only ids and scores
    )
    return matches_to_results(response)


if __name__ == "__main__":
    index = get_index()
    print("Ask a question (type 'quit' to stop)\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break

        try:
            results = retrieve(question, index)
        except PineconeException as e:
            print(f"Search failed: {e}\n")
            continue

        for r in results:
            page = f", page {int(r['page'])}" if r["page"] is not None else ""
            print(f"\n[{r['score']:.2f}] {r['heading'] or ''} ({r['source']}{page})")
            print(r["text"])
        print()