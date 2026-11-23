import os
import re
from loader import load_all_documents


def split_into_chunks(text, max_chars=1400, overlap=150):
    """Split text into chunks of about max_chars, where each chunk
    repeats the last `overlap` characters of the previous one."""
    text = " ".join(text.split())   
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + max_chars // 2:
                end = sentence_end + 1
            else:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step back by `overlap` so the next chunk repeats a little text
        start = max(end - overlap, start + 1)
        next_space = text.find(" ", start, end)   # begin at a whole word
        if next_space != -1:
            start = next_space + 1

    return chunks


def make_slug(label):
    """Turn 'Annual Leave Policy' into 'annual-leave-policy' (safe for use in an id)."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def chunk_documents(documents, max_chars=1400, overlap=150):
    """Turn sections/pages into chunks. Each chunk gets a stable id and keeps its source."""
    all_chunks = []
    for doc in documents:
        heading = doc.get("heading")
        label = heading or f"page-{doc.get('page')}"
        pieces = split_into_chunks(doc["text"], max_chars, overlap)
        for i, piece in enumerate(pieces):
            # Put the section title at the start so the chunk always says what it is about
            text = f"{heading}: {piece}" if heading else piece
            all_chunks.append({
                "id": f"{make_slug(doc['source'])}-{make_slug(label)}-c{i}",
                "source": doc["source"],
                "page": doc.get("page"),
                "heading": heading,
                "text": text,
            })
    return all_chunks


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    docs = load_all_documents(os.path.join(BASE_DIR, "docs"))
    chunks = chunk_documents(docs)

    lengths = [len(c["text"]) for c in chunks]
    print(f"Sections/pages: {len(docs)} | Chunks: {len(chunks)}")
    print(f"Chunk length: min {min(lengths)}, average {sum(lengths) // len(lengths)}, max {max(lengths)}\n")

    for chunk in chunks[:3]:
        print(f"--- {chunk['id']} ---")
        print(chunk["text"])
        print()