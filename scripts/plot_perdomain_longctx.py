#!/usr/bin/env python3
"""Per-domain protection effect at 11K context (Qwen2.5-3B, C=1024)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "longctx"
OUT = REPO / "paper" / "figures" / "fig_perdomain_longctx.pdf"

# longbench-longctx.jsonl: 8 items per subtask in this order
DOMAINS = [
    "2WikiMQA",
    "MuSiQue",
    "MultifieldQA",
    "HotPotQA",
    "NarrativeQA",
    "Qasper",
]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def by_domain(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {d: [] for d in DOMAINS}
    for row in rows:
        domain = DOMAINS[row["idx"] // 8]
        buckets[domain].append(float(row["token_f1"]))
    return {d: float(np.mean(v)) for d, v in buckets.items()}


def main() -> None:
    full = by_domain(load(RESULTS / "q3b-longctx-fullcache.jsonl"))
    noprot = by_domain(load(RESULTS / "q3b-longctx-c1024-noprot.jsonl"))
    prot = by_domain(load(RESULTS / "q3b-longctx-c1024-prot.jsonl"))

    x = np.arange(len(DOMAINS))
    width = 0.26
    labels = [d.replace("MultifieldQA", "MultiFieldQA") for d in DOMAINS]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(
        x - width,
        [full[d] for d in DOMAINS],
        width,
        label="Full cache",
        color="#4C72B0",
    )
    ax.bar(
        x,
        [noprot[d] for d in DOMAINS],
        width,
        label="LRU, no prot.",
        color="#DD8452",
    )
    ax.bar(
        x + width,
        [prot[d] for d in DOMAINS],
        width,
        label="LRU + prot.",
        color="#55A868",
    )

    hotpot_i = DOMAINS.index("HotPotQA")
    pct = prot["HotPotQA"] / full["HotPotQA"] * 100
    ax.annotate(
        f"{pct:.0f}% of ceiling\n(prot. > full cache)",
        xy=(hotpot_i + width, prot["HotPotQA"]),
        xytext=(hotpot_i + 0.35, min(0.78, prot["HotPotQA"] + 0.10)),
        fontsize=11,
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )

    ax.set_ylabel("Mean token-F1", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylim(0, 0.82)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
