# KVCacheBoundaryProtection

Public artifacts for the paper **“Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction.”**

This repository is meant to be **fork-friendly**: figures, plotting scripts, and a thin slice of the inference/policy code used in the study. The full private research checkout (large JSONL bundles, TPU runbooks, and experiment orchestration) stays internal.

## What is here

| Path | Purpose |
|------|---------|
| `figures/` | PDFs of every main-paper figure (same files shipped to arXiv). |
| `scripts/` | Matplotlib scripts that regenerate those PDFs from summarized metrics or checked-in result files. |
| `configs/` | YAML policy/matrix configs mirrored from the internal `release/` bundle. |
| `src/counterfact_kv_eviction/` | Core Python modules for policies and JAX-side plumbing (subset of the internal package). |
| `DATA_MANIFEST.md` | Where full per-item JSONL lives internally and how to regenerate plots if you have access. |

## Quick start (figures only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-figures.txt
python3 scripts/generate_figures.py
python3 scripts/plot_context_scaling.py figures/
python3 scripts/plot_f1_histogram.py
python3 scripts/plot_64k_position.py --preset q3-4b-2507
python3 scripts/plot_perdomain_longctx.py
```

Outputs land under `figures/` when you run commands from this repository root (paths were normalized for the public tree).

## Ethics / scope

- Checkpoints are public Hugging Face models; see the paper’s model table for IDs and licenses.
- This repo **does not** ship large raw JSONL dumps by default (bandwidth + privacy of aggregate-only release). Use `DATA_MANIFEST.md` if you are collaborating with the authors and need the exact artifact paths.

## Citation

Use the citation block from the arXiv page once the paper is live, or cite the NeurIPS preprint PDF metadata from `ArxivMetadata.md` in the private paper repository.
