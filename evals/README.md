# Evaluation Scripts

## Intent Routing

Run:

```bash
python -m evals.run_intent_eval --dataset evals/datasets/intent_classification.jsonl --backend ollama --output evals/results/intent_eval.jsonl
```

Outputs per-example predictions and prints aggregate accuracy.

## Notes

- `run_intent_eval.py` currently calls the agent's routing method directly to isolate classifier quality.
- Add retrieval and ranking evaluations next using fixed query sets and human labels.
