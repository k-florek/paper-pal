# Evaluation Scripts

## Intent Routing

Run:

```bash
python -m evals.run_intent_eval --dataset evals/datasets/intent_classification.jsonl --backend ollama --output evals/results/intent_eval.jsonl
```

Outputs per-example predictions and prints aggregate accuracy.

## Retrieval And Ranking Heuristic Eval

Run:

```bash
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend ollama --output evals/results/retrieval_eval.jsonl
```

This script computes coarse proxy metrics per query:
- `precision_at_5` from keyword overlap flags
- `ndcg_at_10` from keyword-overlap gains
- `min_results_pass` coverage check

## Notes

- `run_intent_eval.py` currently calls the agent's routing method directly to isolate classifier quality.
- `run_retrieval_eval.py` is a lightweight proxy and should be complemented by human-labeled relevance judgments.
