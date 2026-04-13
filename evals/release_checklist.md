# Release Checklist

## Metrics Gate

- [ ] Baseline snapshot exists for intent and retrieval benchmarks.
- [ ] Mean Precision@5 meets target threshold.
- [ ] Mean NDCG@10 meets target threshold.
- [ ] Session feedback delta metrics are non-negative on benchmark runs.

## Reliability Gate

- [ ] Failure checks pass (`python -m evals.run_failure_checks`).
- [ ] No unhandled exceptions for malformed search outputs.
- [ ] Tool fallback behavior verified for sparse and broad queries.

## Safety Gate

- [ ] Malformed/injection-like metadata is rejected by sanitizer.
- [ ] Returned URLs are PubMed URLs only.
- [ ] Ranker output remains field-locked to source metadata.

## Product Gate

- [ ] Search mode presets validated (`balanced`, `clinical`, `mechanism`, `latest`, `reviews`).
- [ ] Confidence and evidence fields visible in paper cards.
- [ ] Relevance feedback buttons update session ranking behavior.

## Sign-off

- [ ] Evaluation artifacts stored in `evals/results/`.
- [ ] Changes reviewed and approved.
