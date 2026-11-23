import os
import time
from pinecone import ServerlessSpec
from pinecone.exceptions import PineconeException

from loader import load_all_documents
from chunker import chunk_documents
from embedder import embed_texts
from store import INDEX_NAME, NAMESPACE, get_client


def build_records(chunks, vectors):
    """Pair each chunk with its vector in the shape Pinecone expects."""
    records = []
    for chunk, vector in zip(chunks, vectors):
        # Pinecone metadata cannot contain None, so we only add fields that have a value
        metadata = {"source": chunk["source"], "text": chunk["text"]}
        if chunk.get("heading"):
            metadata["heading"] = chunk["heading"]
        if chunk.get("page") is not None:
            metadata["page"] = chunk["page"]

        records.append({
            "id": chunk["id"],
            "values": vector.tolist(),   # numpy numbers -> plain Python floats
            "metadata": metadata,
        })
    return records


def ensure_index(pc, dimension):
    """Create the index if it doesn't exist. If it does, check that its size matches."""
    if pc.has_index(INDEX_NAME):
        existing = pc.describe_index(INDEX_NAME).dimension
        if existing != dimension:
            raise RuntimeError(
                f"Index '{INDEX_NAME}' has dimension {existing}, but our embeddings have "
                f"{dimension}. Create a new index with dimension {dimension}."
            )
        print(f"Using existing index '{INDEX_NAME}'.")
    else:
        print(f"Creating index '{INDEX_NAME}' (this can take a minute)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def count_in_namespace(index):
    """How many records are stored in our namespace right now?"""
    stats = index.describe_index_stats()
    summary = stats.namespaces.get(NAMESPACE)
    return summary.vector_count if summary else 0


def wait_for_count(index, expected, timeout=30):
    """Stats can lag behind writes, so check until the count matches what we expect."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if count_in_namespace(index) == expected:
            return True
        time.sleep(1)
    return False


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    docs = load_all_documents(os.path.join(BASE_DIR, "docs"))
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit("No chunks found. Check the docs folder.")
    print(f"Prepared {len(chunks)} chunks from {len(docs)} sections.")

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    records = build_records(chunks, vectors)

    try:
        pc = get_client()
        ensure_index(pc, vectors.shape[1])
        index = pc.Index(INDEX_NAME)

        # Start fresh: remove old records in this namespace so deleted text can't linger
        if count_in_namespace(index) > 0:
            index.delete(delete_all=True, namespace=NAMESPACE)
            wait_for_count(index, 0)
            print("Cleared the old records in the namespace.")

        index.upsert(vectors=records, namespace=NAMESPACE, batch_size=100, show_progress=False)
        print(f"Sent {len(records)} records to Pinecone.")

        if wait_for_count(index, len(records)):
            print(f"Done. The namespace '{NAMESPACE}' now holds {count_in_namespace(index)} records.")
        else:
            print("Records were sent but are not visible yet. Wait a few seconds and run retriever.py.")
    except (PineconeException, RuntimeError) as e:
        print(f"Pinecone step failed: {e}")
    