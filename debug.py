import sys
from openai import OpenAI

from llm import build_prompt, min_score, INSTRUCTION, model
from retriever import retrieve
from store import get_index

question = " ".join(sys.argv[1:]) or "how many days of annual leave do I get"
index = get_index()

print(f"QUESTION: {question}")
print(f"SETTINGS: min_score={min_score}, bot uses the top 2 results\n")

print("=== STAGE 1: what Pinecone returns (top 5) ===")
results = retrieve(question, index, top_k=5)
for rank, r in enumerate(results, start=1):
    verdict = "passes cutoff" if r["score"] >= min_score else "BELOW cutoff"
    print(f"{rank}. score {r['score']:.2f} | {verdict} | {r['heading']} (page {r['page']})")
    print(f"   {r['text'][:200]}...")

print("\n=== STAGE 2: what the bot would actually use ===")
used = [r for r in results[:2] if r["score"] >= min_score]
if not used:
    print("Nothing passed the cutoff, so the CODE answers \"I don't know.\" The LLM is never called.")
    sys.exit()
print("Using:", [r["heading"] for r in used])

print("\n=== STAGE 3: the exact prompt sent to the LLM ===")
prompt = build_prompt(question, used)
print(prompt)

print("=== STAGE 4: the LLM's answer ===")
response = OpenAI().chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": prompt},
    ],
    temperature=0.2,
    max_tokens=300,
)
print(response.choices[0].message.content.strip())