# Data and results manifest

This public repository ships **only** artifacts referenced in
`paper/main.tex` (NeurIPS 2026 / arXiv preprint:
*Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction*).

**Maintainers:** rebuild this tree from the private `CounterFactKVEviction`
checkout with:

```bash
bash scripts/build_public_repo.sh   # run from private repo root, not here
```

Clonees should treat this GitHub repository as the source of truth.

## Benchmark data (`data/`)

| File | Paper use | Items |
|------|-----------|------:|
| `longbench-balanced.jsonl` | Primary LongBench QA panel | 162 |
| `longbench-longctx.jsonl` | 11K per-domain long-context probe | 48 |
| `longbench-summarization.jsonl` | Multi-News summarization appendix | 40 |
| `niah-benchmark.jsonl` | Short NIAH appendix (Phi-3.5, Table `tab:niah`) | 63 |
| `niah-32k.jsonl` | 32K NIAH regime-transfer pilot | 66 |
| `niah-64k.jsonl` | 64K NIAH retrieval stress test | 60 |
| `jit-probe-bench.jsonl` | JAX decode-step latency micro-benchmark | 6 |

Regenerate synthetic NIAH sets with `scripts/generate_niah_dataset.py` and
`scripts/generate_niah_64k.py`. LongBench JSONL files are derived from the
public [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) corpus
using the task counts listed in the paper (§ Benchmark).

## Main-paper figures (`figures/`)

| PDF | Regeneration command |
|-----|----------------------|
| `f1_histogram.pdf` | `python3 scripts/plot_f1_histogram.py` |
| `fig_capacity_curve.pdf` | `python3 scripts/generate_figures.py` |
| `fig_sensitivity.pdf` | `python3 scripts/generate_figures.py` |
| `fig_perdomain_longctx.pdf` | `python3 scripts/plot_perdomain_longctx.py` |
| `fig_context_scaling.pdf` | `python3 scripts/plot_context_scaling.py` |
| `fig_64k_position_q3_4b.pdf` | `python3 scripts/plot_64k_position.py --preset q3-4b-2507` |

## Per-item JSONL bundled under `results/`

| Directory / files | Paper use |
|-------------------|-----------|
| `results/ablation/`, `results/confound-fix/`, `results/confound/`, `results/sweep-20260309-202648/` | Fig. `f1hist`, context-scaling 1.9K panel |
| `results/longctx/` | Fig. per-domain longctx + context-scaling 11K panel |
| `results/32k/`, `results/32k-q3b-final/` | Context-scaling 32K panels |
| `results/q3-4b-2507-niah-64k/` | Fig. 64K position + context-scaling 64K panel |
| `results/phi35-multiarch/`, `results/q7b-multimodel/` | Capacity-curve dynamic loads (Qwen-7B / Phi-3.5 panels) |
| `results/jit-latency-probe/` | Appendix `tab:p99_jax` via `aggregate_jax_decode_latency.py` |
| `results/niah/` | Appendix `tab:niah` |
| `results/credit-v2/`, `results/protection-sweep/` | Appendix `tab:credit`, Fig. `fig_sensitivity` (5/10/15/20% fractions) |
| `results/q3-4b-2507-robustness/` | Appendix `tab:sampling` |
| `results/n481/` | Table `tab:cross_model_n481` (full 481-item pool) |
| `results/summarization-probe/*.summary.json` | Appendix `tab:summarization` aggregates only (per-item JSONL not retained) |

The bundled JSONL set covers all six main figures and the appendix tables listed
above. It does **not** include every condition in the full 427-condition
ROUGE-L robustness sweep cited in the paper.

## Code map

| Path | Role |
|------|------|
| `scripts/run_v3_jax.py` | Primary TPU/JAX experiment runner (globally capped decode-time harness) |
| `src/counterfact_kv_eviction/policies.py` | LRU, H2O, SnapKV, SLW, Ada-KV, QUEST, Random |
| `src/counterfact_kv_eviction/kv_coupled_generator.py` | KV-coupled generation + structural protection |
| `src/counterfact_kv_eviction/wrappers.py` | Protection wrapper + online-credit gating |
| `src/counterfact_kv_eviction/estimator.py` | Online counterfactual credit estimator (Appendix `tab:credit`) |
| `scripts/aggregate_n481.py` | Scale-up table aggregation |
| `scripts/attention_mass_analysis_v2.py` | Attention-mass pilot (§ mechanistic foundation; requires PyTorch + GPU) |
| `scripts/compute_rougel.py` | ROUGE-L vs token-F1 correlation on bundled `results/**/*.jsonl` |

## Not included (out of paper scope)

Internal TPU launchers, exploratory sweeps, GSM8K probes, production EasyDeL
serving paths, and non-paper figure scripts remain in the private
`CounterFactKVEviction` engineering checkout.
