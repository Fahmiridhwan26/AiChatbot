import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pinecone.exceptions import PineconeException

from retriever import retrieve
from store import get_index

load_dotenv()

model = "gpt-4o-mini"
min_score = 0.3

INSTRUCTION = """You are a helpful assistant that answers questions based on the provided context.
If the context does not contain the answer, respond with "I don't know." """


def build_prompt(question, results):
    excerpts = ""
    for i, r in enumerate(results, start=1):
        title = r["heading"] or r["source"]
        excerpts += f"[Excerpt {i}: {title}]\n{r['text']}\n\n"
    return f"Policy excerpts:\n\n{excerpts}Question: {question}\n\nAnswer:"


def answer_question(question, index):
    """Return (answer_text, sources_used)."""
    try:
        results = retrieve(question, index, top_k=2)
    except PineconeException as e:
        return f"Error searching the policy: {e}", []

    good = [r for r in results if r["score"] >= min_score]
    if not good:
        return "I don't know.", []

    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OPENAI_API_KEY is not set in the environment variables.", []

    try:
        client = OpenAI()
        prompt = build_prompt(question, good)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
        return answer, good
    except OpenAIError as e:
        return f"Error: {e}", []


if __name__ == "__main__":
    index = get_index()
    print("Ask a question (type 'quit' to stop)\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break

        answer, sources = answer_question(question, index)
        print(f"\nAnswer: {answer}\n")
        if sources:
            print("Sources:")
            for s in sources:
                page = f", page {int(s['page'])}" if s["page"] is not None else ""
                print(f"- {s['heading'] or s['source']} ({s['source']}{page}) (score: {s['score']:.2f})")
        print()