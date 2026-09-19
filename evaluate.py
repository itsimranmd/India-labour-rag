"""Retrieval evaluation: Recall@k and MRR against a hand-audited answer key."""
import json, re

ANCHOR_LENGTH = 120


def normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def build_anchors(questions, chunks_by_id, length=ANCHOR_LENGTH):
    """Match on CONTENT, not chunk id.

    A gold id like social_security_p26_c2 means "the third 1000-character window
    on page 26" - it only has meaning under one chunking scheme. Change the
    chunking and every id breaks, so every new strategy would score zero.

    Instead, take a distinctive snippet from the middle of each gold passage. A
    retrieval counts as a hit if the returned text contains that snippet, which
    holds across any chunking scheme.
    """
    for q in questions:
        anchors = []
        for gold_id in q.get("gold", []):
            chunk = chunks_by_id.get(gold_id)
            if not chunk:
                continue
            text = normalise(chunk["text"])
            if len(text) < length:
                continue
            middle = len(text) // 2
            anchors.append(text[max(0, middle - length // 2): middle + length // 2])
        q["anchors"] = anchors
    return questions


def evaluate(search_fn, questions, ks=(1, 4, 8)):
    """Recall@k is the ceiling on the system: if the passage isn't retrieved,
    the answer cannot be right. MRR adds position - recall alone cannot tell
    rank 1 from rank 4.
    """
    scored = [q for q in questions if q.get("anchors")]
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []
    by_difficulty = {}

    for q in scored:
        matched = [
            any(a in normalise(p["text"]) for a in q["anchors"])
            for p in search_fn(q["q"], k=max_k)
        ]
        for k in ks:
            if any(matched[:k]):
                hits[k] += 1
        rank = next((i + 1 for i, m in enumerate(matched) if m), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        by_difficulty.setdefault(q["difficulty"], []).append(any(matched[:max_k]))

    n = len(scored)
    return {
        "n": n,
        "mrr": sum(reciprocal_ranks) / n,
        **{f"recall@{k}": hits[k] / n for k in ks},
        "by_difficulty": {d: sum(v) / len(v) for d, v in by_difficulty.items()},
    }


if __name__ == "__main__":
    from retrieve import Retriever

    retriever = Retriever()
    questions = json.load(open("eval/questions.json"))
    questions = build_anchors(questions, retriever.by_id)
    results = evaluate(retriever.search, questions)
    print(json.dumps(results, indent=2))
