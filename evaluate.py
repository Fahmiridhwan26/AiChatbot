import os
import json
from retriever import build_index, retrieve
from llm import min_score   

TOP_K = 2


def load_eval_set(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found -> {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: {path} is not valid JSON -> {e}")
        return []


def check_question(item, chunks, vectors):
    results = retrieve(item["question"], chunks, vectors, top_k=TOP_K)
    good = [r for r in results if r["score"] >= min_score]
    top_score = results[0]["score"] if results else 0.0

    if item["should_refuse"]:
        return {
            "question": item["question"],
            "passed": len(good) == 0,
            "rank": None,
            "top_score": top_score,
            "should_refuse": True,
        }

    rank = None
    for position, r in enumerate(good, start=1):
        if item["must_contain"] in r["text"]:
            rank = position
            break

    return {
        "question": item["question"],
        "passed": rank is not None,
        "rank": rank,
        "top_score": top_score,
        "should_refuse": False,
    }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    eval_set = load_eval_set(os.path.join(BASE_DIR, "eval_set.json"))
    if not eval_set:
        raise SystemExit("No test questions to run.")

    chunks, vectors = build_index(os.path.join(BASE_DIR, "docs"))
    print(f"Settings: top_k={TOP_K}, min_score={min_score}, chunks={len(chunks)}\n")

    outcomes = [check_question(item, chunks, vectors) for item in eval_set]

    for o in outcomes:
        status = "PASS" if o["passed"] else "FAIL"
        if o["should_refuse"]:
            detail = "should refuse"
        else:
            detail = f"correct chunk rank: {o['rank']}"
        print(f"{status} | score {o['top_score']:.2f} | {detail} | {o['question']}")

    answerable = [o for o in outcomes if not o["should_refuse"]]
    off_topic = [o for o in outcomes if o["should_refuse"]]

    print("\n--- Summary ---")
    print(f"Answerable questions found : {sum(o['passed'] for o in answerable)}/{len(answerable)}")
    print(f"Correct chunk ranked first : {sum(o['rank'] == 1 for o in answerable)}/{len(answerable)}")
    print(f"Off-topic correctly refused: {sum(not o['passed'] for o in off_topic)}/{len(off_topic)}") 