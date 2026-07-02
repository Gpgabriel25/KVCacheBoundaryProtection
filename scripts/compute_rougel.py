#!/usr/bin/env python3
"""Compute ROUGE-L vs token-F1 on bundled per-item JSONL result files.

Scans ``results/**/*.jsonl`` under the repository root (or ``--results-dir``)
and reports mean ROUGE-L F1 and mean token-F1 per file, plus the Pearson
correlation across conditions — matching the metric-robustness check in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _mean_token_f1(path: Path) -> tuple[float, int]:
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        scores.append(float(json.loads(line).get("token_f1", 0.0)))
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def _mean_rouge_l(path: Path, scorer) -> tuple[float, int]:
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        ref = item.get("reference", "")
        pred = item.get("prediction", "")
        if not ref:
            continue
        scores.append(float(scorer.score(ref, pred)["rougeL"].fmeasure))
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root directory to scan for *.jsonl (default: <repo>/results)",
    )
    args = parser.parse_args()

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Missing dependency: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    repo = Path(__file__).resolve().parent.parent
    results_dir = args.results_dir or (repo / "results")
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows: list[tuple[str, float, float, int]] = []

    for fpath in sorted(results_dir.rglob("*.jsonl")):
        rouge, n_r = _mean_rouge_l(fpath, scorer)
        f1, n_f = _mean_token_f1(fpath)
        n = min(n_r, n_f) if n_r and n_f else max(n_r, n_f)
        if n <= 0:
            continue
        rel = fpath.relative_to(results_dir)
        rows.append((str(rel), rouge, f1, n))

    if not rows:
        print(f"No JSONL result files under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'Condition':<52} {'ROUGE-L':>8} {'Token-F1':>8} {'N':>5}")
    print("-" * 76)
    for label, rouge, f1, n in rows:
        print(f"{label:<52} {rouge:>8.4f} {f1:>8.4f} {n:>5}")

    if len(rows) >= 2:
        import numpy as np

        rouge_vals = np.array([r[1] for r in rows])
        f1_vals = np.array([r[2] for r in rows])
        pearson = float(np.corrcoef(rouge_vals, f1_vals)[0, 1])
        print(
            f"\nPearson r (ROUGE-L vs token-F1) across {len(rows)} bundled conditions: "
            f"{pearson:.3f}"
        )


if __name__ == "__main__":
    main()
