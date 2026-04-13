from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import ndcg_at_k, precision_at_k
from src.agent import Agent, PaperSearchResult


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def keyword_gain(text: str, expected_keywords: list[str]) -> float:
    low = (text or "").lower()
    if not expected_keywords:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in low)
    return hits / len(expected_keywords)


def evaluate_query(agent: Agent, query: str, expected_keywords: list[str], min_results: int) -> dict:
    response = agent.chatAgent(query)

    if not isinstance(response, PaperSearchResult) or not response.papers:
        return {
            "query": query,
            "returned": 0,
            "precision_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "min_results_pass": False,
            "status": getattr(response, "status", "non_paper"),
        }

    gains: list[float] = []
    flags: list[int] = []
    for paper in response.papers:
        combined = " ".join([
            paper.title or "",
            paper.journal or "",
            paper.relevance or "",
        ])
        gain = keyword_gain(combined, expected_keywords)
        gains.append(gain)
        flags.append(1 if gain > 0 else 0)

    return {
        "query": query,
        "returned": len(response.papers),
        "precision_at_5": precision_at_k(flags, k=5),
        "ndcg_at_10": ndcg_at_k(gains, k=10),
        "min_results_pass": len(response.papers) >= min_results,
        "status": response.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/ranking quality with heuristic labels.")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_eval.jsonl")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--output", default="evals/results/retrieval_eval.jsonl")
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    agent = Agent(session="retrieval-eval", backend=args.backend)

    rows: list[dict] = []
    for item in dataset:
        rows.append(
            evaluate_query(
                agent=agent,
                query=item["query"],
                expected_keywords=item.get("expected_keywords", []),
                min_results=int(item.get("min_results", 1)),
            )
        )

    out_path = Path(args.output)
    out_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    p5 = sum(row["precision_at_5"] for row in rows) / len(rows) if rows else 0.0
    ndcg = sum(row["ndcg_at_10"] for row in rows) / len(rows) if rows else 0.0
    coverage = sum(1 for row in rows if row["min_results_pass"]) / len(rows) if rows else 0.0

    print(f"mean_precision_at_5={p5:.3f}")
    print(f"mean_ndcg_at_10={ndcg:.3f}")
    print(f"min_results_coverage={coverage:.3f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
