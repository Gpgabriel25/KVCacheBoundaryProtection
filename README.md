# KVCacheBoundaryProtection

Code and data for **Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction** ([arXiv:2605.18053](https://arxiv.org/abs/2605.18053)).

Gabriel Garcia (Independent Researcher).

Canonical URL referenced in the paper:
<https://github.com/gpgabriel25/KVCacheBoundaryProtection>

## What is here

This repository contains the paper-scoped slice of the CFKVE project:

- JAX inference + KV eviction policies under a globally capped decode-time harness
- Structural prefix/suffix protection wrapper
- Online counterfactual credit estimator (negative result in Appendix)
- Benchmark JSONL files used in the paper
- Per-item result JSONL needed to regenerate all six main figures and key appendix tables
- Matplotlib scripts that reproduce those figures

See [`DATA_MANIFEST.md`](DATA_MANIFEST.md) for the complete file map.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-release.txt
pip install -e .
```

For figure-only regeneration (no JAX/TPU):

```bash
pip install -r requirements-figures.txt
```

## Regenerate all main-paper figures

From the repository root:

```bash
python3 scripts/generate_figures.py
python3 scripts/plot_f1_histogram.py
python3 scripts/plot_context_scaling.py
python3 scripts/plot_perdomain_longctx.py
python3 scripts/plot_64k_position.py --preset q3-4b-2507
```

PDFs are written to `figures/`.

## Run an experiment (TPU / JAX)

```bash
python3 scripts/run_v3_jax.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --data-path data/longbench-balanced.jsonl \
  --policy lru \
  --capacity 256 \
  --output results/example/q25-c256-lru-prot.jsonl \
  --max-cache-len 2048 \
  --max-new-tokens 128 \
  --protect-prefix-suffix \
  --protect-frac 0.10
```

Policy names, capacities, and protection flags match the paper setup (§ Experimental Setup).
Requires Cloud TPU + JAX with Hugging Face model weights.

## Appendix utilities

```bash
# JAX decode-step latency table (Appendix tab:p99_jax)
python3 scripts/aggregate_jax_decode_latency.py results/jit-latency-probe --min-idx 1 \
  --latex paper/jax_decode_latency_table.tex

# N=481 scale-up table (Table tab:cross_model_n481)
python3 scripts/aggregate_n481.py results/n481 reports/n481-aggregate

# ROUGE-L vs token-F1 on all bundled result JSONL
python3 scripts/compute_rougel.py
```

`scripts/attention_mass_analysis_v2.py` reproduces the attention-mass pilot but
requires a separate PyTorch + GPU environment (`pip install torch transformers`).

## License

Code is released under the [MIT License](LICENSE). Benchmark JSONL under
`data/` is derived from public corpora; model checkpoints remain under their
respective Hugging Face licenses (see paper Table `tab:models`).

## Citation

If you use this code or build on these results, please cite:

```bibtex
@misc{garcia2026protectionnearlyneedstructural,
      title={Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction}, 
      author={Gabriel Garcia},
      year={2026},
      eprint={2605.18053},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.18053}, 
}
```
