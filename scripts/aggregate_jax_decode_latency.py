#!/usr/bin/env python3
"""Aggregate per-row JAX decode-step latency fields from run_v3_jax JSONL outputs.

Each JSON row should include decode_step_mean_ms, decode_step_p50_ms, decode_step_p95_ms,
decode_step_p99_ms (from KVCoupledQwen35Generator).  We summarize each policy with the
median of per-item means and per-item p99 (robust across benchmark items).

Optional --min-idx excludes the first k benchmark rows (e.g. idx 0) from aggregation so
first-sequence compile/sync tails do not dominate median p99.

Usage:
  python3 scripts/aggregate_jax_decode_latency.py results/some_dir
  python3 scripts/aggregate_jax_decode_latency.py results/some_dir --min-idx 1 --latex paper/jax_decode_latency_table.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _policy_from_name(name: str) -> str | None:
    n = name.lower()
    if "adakv_faithful" in n or "adakv-faithful" in n:
        return "Ada-KV faithful + prot"
    if "quest_faithful" in n or "quest-faithful" in n:
        return "QUEST faithful + prot"
    if "probe-lru" in n or "-lru-" in n or "-lru-prot-" in n or "lru-prot-jit" in n:
        if "faithful" not in n:
            return "LRU + prot"
    return None


def _collect(dir_path: Path, min_idx: int) -> dict[str, list[dict]]:
    by_pol: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(dir_path.glob("*.jsonl")):
        pol = _policy_from_name(path.name)
        if pol is None:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row.get("idx", -1))
            if idx < min_idx:
                continue
            by_pol[pol].append(row)
    return by_pol


def _summarize(rows: list[dict]) -> dict[str, float]:
    def ok(r: dict) -> bool:
        return int(r.get("decode_step_n", 0) or 0) > 0

    rows = [r for r in rows if ok(r)]
    if not rows:
        return {
            "n": 0.0,
            "med_mean": float("nan"),
            "med_p50": float("nan"),
            "med_p95": float("nan"),
            "med_p99": float("nan"),
        }
    return {
        "n": float(len(rows)),
        "med_mean": float(np.median([float(r["decode_step_mean_ms"]) for r in rows])),
        "med_p50": float(np.median([float(r["decode_step_p50_ms"]) for r in rows])),
        "med_p95": float(np.median([float(r["decode_step_p95_ms"]) for r in rows])),
        "med_p99": float(np.median([float(r["decode_step_p99_ms"]) for r in rows])),
    }


def _latex_row(label: str, s: dict[str, float]) -> str:
    """Emit one tabular row (always end with \\\\; \\bottomrule lives in the fragment footer)."""
    if s["n"] <= 0:
        return f"{label} & {{--}} & {{--}} & {{--}} & {{--}} \\\\\n"
    return (
        f"{label} & {s['med_mean']:.2f} & {s['med_p50']:.2f} & {s['med_p95']:.2f} & {s['med_p99']:.2f} \\\\\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path, nargs="?", default=Path("results"))
    ap.add_argument(
        "--min-idx",
        type=int,
        default=0,
        help="Drop JSONL rows with idx < this (default 0). Use 1 to omit first benchmark item.",
    )
    ap.add_argument("--latex", type=Path, help="Write LaTeX tabular body rows here")
    args = ap.parse_args()

    d = args.directory
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        sys.exit(1)

    by_pol = _collect(d, min_idx=args.min_idx)
    order = ["LRU + prot", "Ada-KV faithful + prot", "QUEST faithful + prot"]
    stats = {pol: _summarize(by_pol.get(pol, [])) for pol in order}

    print(
        "Per-policy (median of per-item stats, items with decode_step_n > 0"
        + (f", idx>={args.min_idx}" if args.min_idx else "")
        + "):"
    )
    best_pol = None
    best_p99 = float("inf")
    for pol in order:
        s = stats[pol]
        print(f"  {pol}: n_items={int(s['n'])} med_mean_ms={s['med_mean']:.4g} med_p99_ms={s['med_p99']:.4g}")
        if s["n"] > 0 and not np.isnan(s["med_p99"]) and s["med_p99"] < best_p99:
            best_p99 = s["med_p99"]
            best_pol = pol
    lru = stats["LRU + prot"]
    if lru["n"] > 0 and not np.isnan(lru["med_p99"]) and lru["med_p99"] > 0:
        print("Median p99 vs LRU (positive % = slower tail than LRU):")
        for pol in order:
            s = stats[pol]
            if pol == "LRU + prot" or s["n"] <= 0 or np.isnan(s["med_p99"]):
                continue
            pct = 100.0 * (s["med_p99"] / lru["med_p99"] - 1.0)
            print(f"  {pol}: {pct:+.2f}%")
    if best_pol is not None:
        print(f"Lowest median p99 decode-step (this aggregate): {best_pol} ({best_p99:.4g} ms)")

    if args.latex:
        lines = []
        for pol in order:
            s = stats[pol]
            pl = pol.replace(" + prot", " {+} prot")
            lines.append(_latex_row(pl, s))
        body = "".join(lines) + "\\bottomrule\n"
        header = (
            "% Regenerate: python3 scripts/aggregate_jax_decode_latency.py "
            f"{d} --min-idx {args.min_idx} --latex {args.latex}\n"
        )
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(header + body, encoding="utf-8")
        print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
