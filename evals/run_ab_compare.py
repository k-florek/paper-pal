from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs upgraded retrieval eval and generate report.")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_eval.jsonl")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--modes", default="all")
    parser.add_argument("--feedback-k", type=int, default=5)
    parser.add_argument("--out-dir", default="evals/results")
    parser.add_argument("--prefix", default="ab")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / f"{args.prefix}_baseline.jsonl"
    upgraded_path = out_dir / f"{args.prefix}_upgraded.jsonl"
    report_path = out_dir / f"{args.prefix}_report.md"

    baseline_cmd = [
        sys.executable,
        "-m",
        "evals.run_retrieval_eval",
        "--dataset",
        args.dataset,
        "--backend",
        args.backend,
        "--modes",
        args.modes,
        "--output",
        str(baseline_path),
    ]

    upgraded_cmd = [
        sys.executable,
        "-m",
        "evals.run_retrieval_eval",
        "--dataset",
        args.dataset,
        "--backend",
        args.backend,
        "--modes",
        args.modes,
        "--simulate-feedback",
        "--feedback-k",
        str(max(1, args.feedback_k)),
        "--output",
        str(upgraded_path),
    ]

    report_cmd = [
        sys.executable,
        "-m",
        "evals.generate_ab_report",
        "--baseline",
        str(baseline_path),
        "--upgraded",
        str(upgraded_path),
        "--output",
        str(report_path),
    ]

    run_cmd(baseline_cmd)
    run_cmd(upgraded_cmd)
    run_cmd(report_cmd)

    print(f"baseline={baseline_path}")
    print(f"upgraded={upgraded_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
