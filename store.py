import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "policy-chatbot"    # change this to the index name you chose
NAMESPACE = "abc-policy-v1"      # a labelled section inside the index (handy for versions)


def get_client():
    """Create the Pinecone client using the key from the .env file."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Check your .env file.")
    return Pinecone(api_key=api_key)


def get_index():
    """Open the existing index. Used by the chatbot (it only reads)."""
    pc = get_client()
    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(f"Index '{INDEX_NAME}' does not exist yet. Run ingest.py first.")
    return pc.Index(INDEX_NAME)