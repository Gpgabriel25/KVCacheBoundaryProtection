#!/usr/bin/env python3
"""Plot 64K NIAH position-breakdown bar chart (Early/Middle/Late F1 by policy)."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "64k-qwen35"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "figures"


def load_results(results_dir: Path, fname: str):
    fpath = results_dir / fname
    if not fpath.exists():
        return None
    with fpath.open() as f:
        return [json.loads(l) for l in f]


def position_means(items):
    """Compute Early/Middle/Late mean F1 using fixed 20/20/20 boundaries.
    If fewer than 60 items, uses available data per position (may be 0)."""
    n = len(items)
    early = [x["token_f1"] for x in items[:min(n, 20)]]
    mid_items = items[20:min(n, 40)]
    late_items = items[40:min(n, 60)]
    mid = [x["token_f1"] for x in mid_items] if mid_items else None
    late = [x["token_f1"] for x in late_items] if late_items else None
    return (
        statistics.mean(early) if early else 0,
        statistics.mean(mid) if mid else None,
        statistics.mean(late) if late else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix", default="q35-64k")
    parser.add_argument("--title", default="64K NIAH: Position-Dependent Retrieval by Policy (Qwen3.5-27B)")
    parser.add_argument("--output-stem", default="fig_64k_position")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--preset",
        choices=("qwen35", "q3-4b-2507"),
        default="qwen35",
        help="Result bundle preset (default: hybrid Qwen3.5-27B panel)",
    )
    args = parser.parse_args()

    if args.preset == "q3-4b-2507":
        args.results_dir = Path("results/q3-4b-2507-niah-64k")
        args.prefix = "q3-4b-2507-64k"
        args.output_stem = "fig_64k_position_q3_4b"
        args.title = "64K NIAH: Position-Dependent Retrieval (Qwen3-4B-Instruct-2507)"

    if args.preset == "q3-4b-2507":
        conditions = [
            ("Full cache", f"{args.prefix}-fullcache-lru-noprot.jsonl", "#2ca02c"),
            ("LRU+prot C=4096", f"{args.prefix}-c4096-lru-prot.jsonl", "#ff7f0e"),
            ("LRU (no prot) C=4096", f"{args.prefix}-c4096-lru-noprot.jsonl", "#8c564b"),
            ("H2O+prot C=4096", f"{args.prefix}-c4096-h2o-prot.jsonl", "#d62728"),
            ("H2O (no prot) C=4096", f"{args.prefix}-c4096-h2o-noprot.jsonl", "#bcbd22"),
            ("Random+prot C=4096", f"{args.prefix}-c4096-random-prot.jsonl", "#9467bd"),
        ]
    else:
        conditions = [
            ("Full cache", f"{args.prefix}-fullcache-lru-noprot.jsonl", "#2ca02c"),
            ("LRU+prot C=4096", f"{args.prefix}-c4096-lru-prot.jsonl", "#ff7f0e"),
            ("LRU (no prot) C=4096", f"{args.prefix}-c4096-lru-noprot.jsonl", "#8c564b"),
            ("H2O+prot C=8192", f"{args.prefix}-c8192-h2o-prot.jsonl", "#d62728"),
            ("Random+prot C=8192", f"{args.prefix}-c8192-random-prot.jsonl", "#9467bd"),
            ("LRU+prot C=8192", f"{args.prefix}-c8192-lru-prot.jsonl", "#1f77b4"),
            ("LRU+abs256 C=4096", f"{args.prefix}-c4096-lru-absprot256.jsonl", "#17becf"),
        ]

    labels = []
    early_vals = []
    mid_vals = []
    late_vals = []
    colors = []

    for name, fname, color in conditions:
        items = load_results(args.results_dir, fname)
        if items is None:
            print(f"Skipping {name}: file not found")
            continue
        n = len(items)
        if n < 60:
            print(f"Warning: {name} has only {n}/60 items (using partial data)")
        if n < 3:
            print(f"Skipping {name}: too few items")
            continue

        means = position_means(items)
        labels.append(name)
        early_vals.append(means[0])
        mid_vals.append(means[1] if means[1] is not None else 0)
        late_vals.append(means[2] if means[2] is not None else 0)
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.2, 4.6))

    bars_e = ax.bar(x - width, early_vals, width, label="Early", color=[c for c in colors], alpha=0.9, edgecolor="white", linewidth=0.5)
    bars_m = ax.bar(x, mid_vals, width, label="Middle", color=[c for c in colors], alpha=0.6, edgecolor="white", linewidth=0.5, hatch="//")
    bars_l = ax.bar(x + width, late_vals, width, label="Late", color=[c for c in colors], alpha=0.35, edgecolor="white", linewidth=0.5, hatch="xx")

    for bars in [bars_e, bars_m, bars_l]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Token F1", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=10, ha="right")
    ymax = max(early_vals + mid_vals + late_vals) if labels else 1.0
    ax.set_ylim(0, max(0.40, ymax * 1.30))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="Early"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", label="Middle"),
        Patch(facecolor="gray", alpha=0.35, hatch="xx", label="Late"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    ax.set_title(args.title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out_dir / f"{args.output_stem}.pdf"
    out_png = args.out_dir / f"{args.output_stem}.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
