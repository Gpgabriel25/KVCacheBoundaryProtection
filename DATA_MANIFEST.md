# Data manifest (public-facing)

Full per-item **JSONL** outputs for the paper are **not** vendored in this public repository (they are large and live in the private `CounterFactKVEviction` checkout under `results/`).

## Figure → internal evidence (high level)

| Artifact | Typical internal path (examples) |
|----------|-----------------------------------|
| Fig. 2 capacity / sensitivity / lift | `results/phi35-multiarch/`, `results/q7b-multimodel/`, hard-coded baselines in `scripts/generate_figures.py` |
| F1 histogram | `results/ablation/q25-c256-lru.jsonl`, `results/confound-fix/q25-c256-lru-protected.jsonl` |
| Context scaling | `results/confound-fix/`, `results/longctx/`, `results/32k*/`, `results/q3-4b-2507-niah-64k/` |
| 64K position (Qwen3-4B) | `results/q3-4b-2507-niah-64k/*.jsonl` |
| Per-domain longctx | `results/longctx/q3b-longctx-*.jsonl` |
| JAX latency table (paper) | Regenerated via `scripts/aggregate_jax_decode_latency.py` (see comment in `paper/jax_decode_latency_table.tex`) |

If you need a **redistributable** sample bundle, open an issue; the authors can attach a minimal tar with e.g. 10 anonymized rows per condition for reproducibility review.
