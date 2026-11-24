import os
import re
from collections import Counter

import pymupdf
from langchain_community.document_loaders import TextLoader


def read_pdf_lines(file_path):

    lines = []
    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = " ".join("".join(span["text"] for span in line["spans"]).split())
                    if text:
                        size = round(max(span["size"] for span in line["spans"]), 1)
                        lines.append({"text": text, "size": size, "page": page_number})
    return lines


def load_pdf(file_path):
    try:
        lines = read_pdf_lines(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    if not lines:
        print(f"Warning: no text found in {file_path} (is it a scanned image?)")
        return []

    filename = os.path.basename(file_path)

    # Body size = the font size that covers the most characters
    chars_per_size = Counter()
    for line in lines:
        chars_per_size[line["size"]] += len(line["text"])
    body_size = chars_per_size.most_common(1)[0][0]
    is_heading = lambda line: line["size"] >= body_size * 1.2

    # Walk through the lines and cut a new section at each heading
    sections = []
    heading, start_page, body, last_was_heading = "Introduction", 1, [], False

    def save_section():
        text = " ".join(body).strip()
        if text:
            sections.append({"source": filename, "page": start_page, "heading": heading, "text": text})

    for line in lines:
        if is_heading(line):
            if last_was_heading:
                heading = heading + " " + line["text"]    # a heading wrapped onto a second line
            else:
                save_section()
                heading, start_page, body = line["text"], line["page"], []
            last_was_heading = True
        else:
            body.append(line["text"])
            last_was_heading = False
    save_section()

    if len([s for s in sections if s["heading"] != "Introduction"]) < 2:
        print(f"Note: no clear headings in {filename}, using one section per page instead.")
        by_page = {}
        for line in lines:
            by_page.setdefault(line["page"], []).append(line["text"])
        return [
            {"source": filename, "page": p, "heading": None, "text": " ".join(texts)}
            for p, texts in by_page.items()
        ]

    for s in sections:
        s["heading"] = re.sub(r"^\d+[.)]\s*", "", s["heading"])   
    return sections


def load_text_file(file_path):
    """Read a .txt file with LangChain."""
    try:
        lc_docs = TextLoader(file_path, encoding="utf-8").load()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return [
        {"source": os.path.basename(file_path), "page": 1, "heading": None,
         "text": doc.page_content.strip()}
        for doc in lc_docs
        if doc.page_content.strip()
    ]


def load_all_documents(folder):
    """Load every .pdf and .txt file in a folder."""
    documents = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        name = filename.lower()
        if name.endswith(".pdf"):
            documents.extend(load_pdf(path))
        elif name.endswith(".txt"):
            documents.extend(load_text_file(path))
    return documents


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    docs = load_all_documents(os.path.join(BASE_DIR, "docs"))
    print(f"Loaded {len(docs)} sections\n")
    for doc in docs:
        label = doc["heading"] or f"page {doc['page']}"
        print(f"{doc['source']} | page {doc['page']} | {label}: {len(doc['text'])} characters")
    if docs:
        print("\n--- Preview of the first section ---")
        print(docs[0]["text"][:400])