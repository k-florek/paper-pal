from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "mean_precision_at_5": 0.0,
            "mean_ndcg_at_10": 0.0,
            "min_results_coverage": 0.0,
            "by_mode": {},
        }

    def agg(items: list[dict]) -> dict:
        return {
            "count": len(items),
            "mean_precision_at_5": sum(r.get("precision_at_5", 0.0) for r in items) / len(items),
            "mean_ndcg_at_10": sum(r.get("ndcg_at_10", 0.0) for r in items) / len(items),
            "min_results_coverage": sum(1 for r in items if r.get("min_results_pass")) / len(items),
        }

    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        mode = row.get("search_mode", "balanced")
        by_mode.setdefault(mode, []).append(row)

    return {
        **agg(rows),
        "by_mode": {mode: agg(mode_rows) for mode, mode_rows in by_mode.items()},
    }


def fmt(v: float) -> str:
    return f"{v:.3f}"


def build_markdown(baseline: dict, upgraded: dict, baseline_path: str, upgraded_path: str) -> str:
    bp5 = baseline["mean_precision_at_5"]
    bpndcg = baseline["mean_ndcg_at_10"]
    bcov = baseline["min_results_coverage"]

    up5 = upgraded["mean_precision_at_5"]
    upndcg = upgraded["mean_ndcg_at_10"]
    ucov = upgraded["min_results_coverage"]

    lines = [
        "# A/B Retrieval Evaluation Report",
        "",
        "## Inputs",
        f"- Baseline file: {baseline_path}",
        f"- Upgraded file: {upgraded_path}",
        "",
        "## Overall Summary",
        "",
        "| Metric | Baseline | Upgraded | Delta |",
        "|---|---:|---:|---:|",
        f"| Mean Precision@5 | {fmt(bp5)} | {fmt(up5)} | {fmt(up5 - bp5)} |",
        f"| Mean NDCG@10 | {fmt(bpndcg)} | {fmt(upndcg)} | {fmt(upndcg - bpndcg)} |",
        f"| Min Results Coverage | {fmt(bcov)} | {fmt(ucov)} | {fmt(ucov - bcov)} |",
        "",
        "## Per-Mode Summary",
        "",
        "| Mode | Metric | Baseline | Upgraded | Delta |",
        "|---|---|---:|---:|---:|",
    ]

    modes = sorted(set(baseline["by_mode"].keys()) | set(upgraded["by_mode"].keys()))
    for mode in modes:
        b = baseline["by_mode"].get(mode, {})
        u = upgraded["by_mode"].get(mode, {})

        bp5m = float(b.get("mean_precision_at_5", 0.0))
        up5m = float(u.get("mean_precision_at_5", 0.0))
        bndcgm = float(b.get("mean_ndcg_at_10", 0.0))
        undcgm = float(u.get("mean_ndcg_at_10", 0.0))
        bcovm = float(b.get("min_results_coverage", 0.0))
        ucovm = float(u.get("min_results_coverage", 0.0))

        lines.append(f"| {mode} | Precision@5 | {fmt(bp5m)} | {fmt(up5m)} | {fmt(up5m - bp5m)} |")
        lines.append(f"| {mode} | NDCG@10 | {fmt(bndcgm)} | {fmt(undcgm)} | {fmt(undcgm - bndcgm)} |")
        lines.append(f"| {mode} | Coverage | {fmt(bcovm)} | {fmt(ucovm)} | {fmt(ucovm - bcovm)} |")

    lines.extend([
        "",
        "## Recommendation",
        "- Promote the upgraded pipeline if deltas are non-negative for Precision@5 and NDCG@10 overall and for the primary production mode.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown A/B report from two eval JSONL files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--upgraded", required=True)
    parser.add_argument("--output", default="evals/results/ab_report.md")
    args = parser.parse_args()

    baseline_rows = load_jsonl(Path(args.baseline))
    upgraded_rows = load_jsonl(Path(args.upgraded))

    baseline_summary = summarize(baseline_rows)
    upgraded_summary = summarize(upgraded_rows)
    report = build_markdown(
        baseline=baseline_summary,
        upgraded=upgraded_summary,
        baseline_path=args.baseline,
        upgraded_path=args.upgraded,
    )

    out_path = Path(args.output)
    out_path.write_text(report)

    print(f"wrote_report={args.output}")


if __name__ == "__main__":
    main()
