from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import accuracy
from src.agent import Agent


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intent routing accuracy.")
    parser.add_argument("--dataset", default="evals/datasets/intent_classification.jsonl")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--output", default="evals/results/intent_eval.jsonl")
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    agent = Agent(session="intent-eval", backend=args.backend)

    predictions: list[str] = []
    labels: list[str] = []
    rows: list[str] = []

    for item in dataset:
        query = item["query"]
        label = item["expected_intent"].strip().upper()
        decision = agent._classify_intent(query)
        pred = decision.intent.strip().upper()

        labels.append(label)
        predictions.append(pred)

        rows.append(json.dumps({
            "query": query,
            "expected_intent": label,
            "predicted_intent": pred,
            "confidence": decision.confidence,
            "ambiguous": decision.ambiguous,
            "needs_clarification": decision.needs_clarification,
            "correct": pred == label,
        }))

    score = accuracy(predictions, labels)
    Path(args.output).write_text("\n".join(rows) + "\n")
    print(f"intent_accuracy={score:.3f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
