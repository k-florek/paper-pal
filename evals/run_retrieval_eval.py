from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import ndcg_at_k, precision_at_k
from src.agent import Agent, PaperSearchResult

VALID_MODES = ["balanced", "clinical", "mechanism", "latest", "reviews"]
MODE_FEEDBACK_CONFIG = {
    "balanced": {"min_conf": 0.55, "min_gain": 0.34, "pos_weight": 1.0, "neg_weight": 0.7},
    "clinical": {"min_conf": 0.72, "min_gain": 0.50, "pos_weight": 0.55, "neg_weight": 0.35},
    "mechanism": {"min_conf": 0.55, "min_gain": 0.34, "pos_weight": 1.1, "neg_weight": 0.7},
    "latest": {"min_conf": 0.75, "min_gain": 0.50, "pos_weight": 0.5, "neg_weight": 0.25},
    "reviews": {"min_conf": 0.60, "min_gain": 0.34, "pos_weight": 0.95, "neg_weight": 0.6},
}


def load_backend_config(config_path: Path, backend: str) -> dict:
    """Load backend defaults from config.json for direct evaluator runs."""
    if not config_path.exists():
        return {}
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return {}
    backend_cfg = cfg.get(backend, {})
    return backend_cfg if isinstance(backend_cfg, dict) else {}


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


def score_response(response: PaperSearchResult, expected_keywords: list[str]) -> tuple[float, float, int]:
    """Return (precision@5, ndcg@10, returned_count) for a ranked response."""
    gains: list[float] = []
    flags: list[int] = []
    for paper in response.papers:
        combined = " ".join([
            paper.title or "",
            paper.journal or "",
            paper.relevance or "",
            paper.evidence or "",
        ])
        gain = keyword_gain(combined, expected_keywords)
        gains.append(gain)
        flags.append(1 if gain > 0 else 0)
    return precision_at_k(flags, k=5), ndcg_at_k(gains, k=10), len(response.papers)


def apply_simulated_feedback(
    agent: Agent,
    query: str,
    response: PaperSearchResult,
    expected_keywords: list[str],
    feedback_k: int,
    search_mode: str,
) -> int:
    """Apply heuristic positive/negative feedback from top-k returned papers."""
    applied = 0
    cfg = MODE_FEEDBACK_CONFIG.get(search_mode, MODE_FEEDBACK_CONFIG["balanced"])
    for paper in response.papers[:feedback_k]:
        combined = " ".join([
            paper.title or "",
            paper.journal or "",
            paper.relevance or "",
            paper.evidence or "",
        ])
        gain = keyword_gain(combined, expected_keywords)
        confidence = float(getattr(paper, "confidence", 0.5) or 0.5)
        if confidence < cfg["min_conf"]:
            continue

        relevant = gain >= cfg["min_gain"]
        score_weight = cfg["pos_weight"] if relevant else cfg["neg_weight"]

        agent.record_feedback(
            paper_url=paper.url or "",
            paper_title=paper.title or "",
            query=query,
            relevant=relevant,
            note="simulated_eval_feedback",
            search_mode=search_mode,
            confidence=confidence,
            score_weight=score_weight,
        )
        applied += 1
    return applied


def evaluate_query(
    agent: Agent,
    query: str,
    expected_keywords: list[str],
    min_results: int,
    include_papers: bool,
    simulate_feedback: bool,
    feedback_k: int,
    search_mode: str,
    force_research: bool,
) -> dict:
    baseline = agent.chatAgent(query, search_mode, force_research=force_research)

    if not isinstance(baseline, PaperSearchResult) or not baseline.papers:
        return {
            "query": query,
            "search_mode": search_mode,
            "returned": 0,
            "precision_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "min_results_pass": False,
            "status": getattr(baseline, "status", "non_paper"),
            "before": {"returned": 0, "precision_at_5": 0.0, "ndcg_at_10": 0.0},
            "after": {"returned": 0, "precision_at_5": 0.0, "ndcg_at_10": 0.0},
            "delta": {"precision_at_5": 0.0, "ndcg_at_10": 0.0, "returned": 0},
            "feedback_events": 0,
            "papers": [] if include_papers else None,
        }

    p5_before, ndcg_before, returned_before = score_response(baseline, expected_keywords)
    status = baseline.status
    after = baseline
    feedback_events = 0

    if simulate_feedback:
        feedback_events = apply_simulated_feedback(
            agent=agent,
            query=query,
            response=baseline,
            expected_keywords=expected_keywords,
            feedback_k=feedback_k,
            search_mode=search_mode,
        )
        second_pass = agent.chatAgent(query, search_mode, force_research=force_research)
        if isinstance(second_pass, PaperSearchResult) and second_pass.papers:
            after = second_pass
            status = second_pass.status

    p5_after, ndcg_after, returned_after = score_response(after, expected_keywords)

    result = {
        "query": query,
        "search_mode": search_mode,
        "returned": returned_after,
        "precision_at_5": p5_after,
        "ndcg_at_10": ndcg_after,
        "min_results_pass": returned_after >= min_results,
        "status": status,
        "before": {
            "returned": returned_before,
            "precision_at_5": p5_before,
            "ndcg_at_10": ndcg_before,
        },
        "after": {
            "returned": returned_after,
            "precision_at_5": p5_after,
            "ndcg_at_10": ndcg_after,
        },
        "delta": {
            "precision_at_5": p5_after - p5_before,
            "ndcg_at_10": ndcg_after - ndcg_before,
            "returned": returned_after - returned_before,
        },
        "feedback_events": feedback_events,
    }

    if include_papers:
        result["papers"] = [
            {
                "title": paper.title,
                "url": paper.url,
                "relevance": paper.relevance,
                "confidence": paper.confidence,
                "evidence": paper.evidence,
            }
            for paper in after.papers
        ]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/ranking quality with heuristic labels.")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_eval.jsonl")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="evals/results/retrieval_eval.jsonl")
    parser.add_argument("--include-papers", action="store_true")
    parser.add_argument("--simulate-feedback", action="store_true")
    parser.add_argument("--feedback-k", type=int, default=5)
    parser.add_argument("--modes", default="balanced", help="Comma-separated modes or 'all'.")
    parser.add_argument("--force-research", action="store_true", help="Bypass intent routing and force research pipeline.")
    parser.add_argument("--sanitizer-mode", choices=["strict", "tolerant"], default="tolerant")
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    backend_config = load_backend_config(Path(args.config), args.backend)
    backend_config = {**backend_config, "sanitizer_mode": args.sanitizer_mode}
    if args.backend == "aws_bedrock":
        backend_config = {**backend_config, "ranker_strict_json": True}
    modes = VALID_MODES if args.modes.strip().lower() == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid_modes = [m for m in modes if m not in VALID_MODES]
    if invalid_modes:
        raise ValueError(f"Invalid mode(s): {invalid_modes}. Valid modes: {VALID_MODES}")

    rows: list[dict] = []
    mode_summaries: dict[str, dict[str, float]] = {}
    for mode in modes:
        agent = Agent(
            session=f"retrieval-eval-{mode}",
            backend=args.backend,
            backend_config=backend_config,
        )
        mode_rows: list[dict] = []
        for item in dataset:
            mode_rows.append(
                evaluate_query(
                    agent=agent,
                    query=item["query"],
                    expected_keywords=item.get("expected_keywords", []),
                    min_results=int(item.get("min_results", 1)),
                    include_papers=args.include_papers,
                    simulate_feedback=args.simulate_feedback,
                    feedback_k=max(1, args.feedback_k),
                    search_mode=mode,
                    force_research=args.force_research,
                )
            )
        rows.extend(mode_rows)

        mode_p5 = sum(row["precision_at_5"] for row in mode_rows) / len(mode_rows) if mode_rows else 0.0
        mode_ndcg = sum(row["ndcg_at_10"] for row in mode_rows) / len(mode_rows) if mode_rows else 0.0
        mode_cov = sum(1 for row in mode_rows if row["min_results_pass"]) / len(mode_rows) if mode_rows else 0.0
        mode_summaries[mode] = {
            "mean_precision_at_5": mode_p5,
            "mean_ndcg_at_10": mode_ndcg,
            "min_results_coverage": mode_cov,
        }

        if args.simulate_feedback and mode_rows:
            mode_summaries[mode]["mean_delta_precision_at_5"] = (
                sum(row["delta"]["precision_at_5"] for row in mode_rows) / len(mode_rows)
            )
            mode_summaries[mode]["mean_delta_ndcg_at_10"] = (
                sum(row["delta"]["ndcg_at_10"] for row in mode_rows) / len(mode_rows)
            )

    out_path = Path(args.output)
    out_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    p5 = sum(row["precision_at_5"] for row in rows) / len(rows) if rows else 0.0
    ndcg = sum(row["ndcg_at_10"] for row in rows) / len(rows) if rows else 0.0
    coverage = sum(1 for row in rows if row["min_results_pass"]) / len(rows) if rows else 0.0

    print(f"mean_precision_at_5={p5:.3f}")
    print(f"mean_ndcg_at_10={ndcg:.3f}")
    print(f"min_results_coverage={coverage:.3f}")
    print(f"sanitizer_mode={args.sanitizer_mode}")
    if args.simulate_feedback and rows:
        mean_dp5 = sum(row["delta"]["precision_at_5"] for row in rows) / len(rows)
        mean_dndcg = sum(row["delta"]["ndcg_at_10"] for row in rows) / len(rows)
        print(f"mean_delta_precision_at_5={mean_dp5:.3f}")
        print(f"mean_delta_ndcg_at_10={mean_dndcg:.3f}")
    print("mode_summaries=" + json.dumps(mode_summaries, sort_keys=True))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
