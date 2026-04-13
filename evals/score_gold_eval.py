from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evals.metrics import ndcg_at_k, precision_at_k


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def norm_title(title: str) -> str:
    text = (title or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_gold_map(gold_rows: list[dict]) -> dict[str, dict[str, int]]:
    gold_map: dict[str, dict[str, int]] = {}
    for row in gold_rows:
        query = row.get("query", "")
        labels = row.get("labels", [])
        label_map: dict[str, int] = {}
        for label in labels:
            label_map[norm_title(label.get("title", ""))] = int(label.get("grade", 0))
        gold_map[query] = label_map
    return gold_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval output against human-labeled gold titles.")
    parser.add_argument("--predictions", default="evals/results/retrieval_eval.jsonl")
    parser.add_argument("--gold", default="evals/datasets/retrieval_gold_labels_template.jsonl")
    args = parser.parse_args()

    pred_rows = load_jsonl(Path(args.predictions))
    gold_map = build_gold_map(load_jsonl(Path(args.gold)))

    scored = []
    for row in pred_rows:
        query = row.get("query", "")
        papers = row.get("papers") or []
        label_map = gold_map.get(query, {})

        gains = []
        flags = []
        for paper in papers:
            grade = label_map.get(norm_title(paper.get("title", "")), 0)
            gains.append(float(grade))
            flags.append(1 if grade > 0 else 0)

        scored.append(
            {
                "query": query,
                "precision_at_5": precision_at_k(flags, k=5),
                "ndcg_at_10": ndcg_at_k(gains, k=10),
                "judged_count": len(label_map),
            }
        )

    if not scored:
        print("No predictions to score.")
        return

    avg_p5 = sum(item["precision_at_5"] for item in scored) / len(scored)
    avg_ndcg = sum(item["ndcg_at_10"] for item in scored) / len(scored)

    print(f"gold_precision_at_5={avg_p5:.3f}")
    print(f"gold_ndcg_at_10={avg_ndcg:.3f}")


if __name__ == "__main__":
    main()
