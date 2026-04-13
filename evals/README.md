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

Run across all search modes:

```bash
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend ollama --modes all --output evals/results/retrieval_eval_modes.jsonl
```

To include paper details for human judging:

```bash
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend ollama --include-papers --output evals/results/retrieval_eval.jsonl
```

To measure in-session feedback impact (no persistence across sessions):

```bash
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend ollama --simulate-feedback --feedback-k 5 --output evals/results/retrieval_eval_feedback.jsonl
```

This runs each query twice in the same session:
- pass 1: baseline ranking
- pass 2: ranking after simulated paper-level feedback on top-k results

The script reports `mean_delta_precision_at_5` and `mean_delta_ndcg_at_10`.

This script computes coarse proxy metrics per query:
- `precision_at_5` from keyword overlap flags
- `ndcg_at_10` from keyword-overlap gains
- `min_results_pass` coverage check
- `mode_summaries` aggregate output when one or multiple modes are evaluated

## Gold-Labeled Evaluation

1. Create human labels in [evals/datasets/retrieval_gold_labels_template.jsonl](evals/datasets/retrieval_gold_labels_template.jsonl) using title-level grades:
	- `0` = irrelevant
	- `1` = partially relevant
	- `2` = directly relevant

2. Score predictions against labels:

```bash
python -m evals.score_gold_eval --predictions evals/results/retrieval_eval.jsonl --gold evals/datasets/retrieval_gold_labels_template.jsonl
```

This prints `gold_precision_at_5` and `gold_ndcg_at_10`.

## Failure Checks

Run lightweight failure-mode checks:

```bash
python -m evals.run_failure_checks
```

Release gate checklist is provided in [evals/release_checklist.md](evals/release_checklist.md).

## One-Command A/B Compare

Run baseline vs upgraded (feedback-adaptive) evaluation and generate a markdown report:

```bash
python -m evals.run_ab_compare --dataset evals/datasets/retrieval_eval.jsonl --backend ollama --modes all --feedback-k 5 --out-dir evals/results --prefix ab
```

For Bedrock robustness testing, force the research pipeline during benchmark runs:

```bash
python -m evals.run_ab_compare --dataset evals/datasets/retrieval_eval.jsonl --backend aws_bedrock --modes all --feedback-k 5 --force-research --out-dir evals/results --prefix ab_bedrock
```

To compare sanitizer behavior explicitly:

```bash
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend aws_bedrock --modes all --force-research --sanitizer-mode strict --output evals/results/bedrock_strict.jsonl
python -m evals.run_retrieval_eval --dataset evals/datasets/retrieval_eval.jsonl --backend aws_bedrock --modes all --force-research --sanitizer-mode tolerant --output evals/results/bedrock_tolerant.jsonl
```

This produces:
- `evals/results/ab_baseline.jsonl`
- `evals/results/ab_upgraded.jsonl`
- `evals/results/ab_report.md`

## Notes

- `run_intent_eval.py` currently calls the agent's routing method directly to isolate classifier quality.
- `run_retrieval_eval.py` is a lightweight proxy and should be complemented by human-labeled relevance judgments.
- During interactive app usage, paper-level feedback is captured via `POST /api/feedback` and fed into deterministic pre-ranking for the active session.
- Feedback adaptation is session-scoped only in the current implementation.
