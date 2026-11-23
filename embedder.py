from sentence_transformers import SentenceTransformer, util # type: ignore
from loader import load_all_documents
import os
from chunker import chunk_documents

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_DIR = os.path.join(BASE_DIR, "docs")

    docs = load_all_documents(DOCS_DIR)
    chunks = chunk_documents(docs)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    print("Shape of vectors:", vectors.shape)
    print("First 5 numbers of chunk 0:", vectors[0][:5])

    sentences = [
        "How long does delivery take?",
        "How many days until my parcel arrives?",
        "What is the return policy?",
    ]
    v = embed_texts(sentences)
    print()
    print("delivery vs parcel :", float(util.cos_sim(v[0], v[1])))
    print("delivery vs return :", float(util.cos_sim(v[0], v[2])))