#!/usr/bin/env python3
"""Generate publication figures for the protection universality paper.

Supports 4-model cross-architecture analysis:
  Qwen2.5-3B (GQA), Qwen2.5-1.5B (GQA), Phi-3.5-mini (MHA), Qwen2.5-7B (GQA)
"""

import json
import re
from pathlib import Path
import numpy as np

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")

# ── Data ──────────────────────────────────────────────────────────────────────

# Qwen-3B results (from aggregate scripts)
Q3B = {
    'fullcache': 0.315,
    'lru_noprot':   {64: 0.019, 96: 0.019, 128: 0.019, 256: 0.011, 512: 0.010},
    'lru_prot':     {64: 0.153, 96: 0.214, 128: 0.229, 256: 0.282, 512: 0.285},
    'h2o_noprot':   {128: 0.030, 256: 0.038, 512: 0.038},
    'h2o_prot':     {128: 0.230, 256: 0.290, 512: 0.298},
    'snapkv_noprot':{128: 0.030, 256: 0.038, 512: 0.037},
    'snapkv_prot':  {128: 0.230, 256: 0.290, 512: 0.298},
    'streamllm_prot': {128: 0.124, 256: 0.184, 512: 0.198},
}

# Qwen-1.5B results
Q15B = {
    'fullcache': 0.304,
    'lru_noprot': {128: 0.013, 256: 0.012},
    'lru_prot':   {128: 0.214, 256: 0.241, 512: 0.251},
    'h2o_prot':   {256: 0.231},
    'snapkv_prot': {256: 0.231},
}

# Phi-3.5-mini results (MHA 32Q/32KV — different architecture family)
# NOTE: Updated from final experiment results. Placeholder values from early partials.
PHI35 = {
    'fullcache': None,  # Will be filled from results/phi35-multiarch/
    'lru_noprot': {},
    'lru_prot': {},
    'h2o_prot': {},
    'snapkv_prot': {},
}

def _load_phi35_data():
    """Load Phi-3.5 data from results files if available."""
    phidir = Path('results/phi35-multiarch')
    if not phidir.exists():
        return
    mapping = {
        'phi35-fullcache.jsonl': ('fullcache', None, None),
        'phi35-c128-lru-noprot.jsonl': ('lru_noprot', 128, None),
        'phi35-c128-lru-prot.jsonl': ('lru_prot', 128, None),
        'phi35-c128-h2o-prot.jsonl': ('h2o_prot', 128, None),
        'phi35-c128-snapkv-prot.jsonl': ('snapkv_prot', 128, None),
        'phi35-c256-lru-noprot.jsonl': ('lru_noprot', 256, None),
        'phi35-c256-lru-prot.jsonl': ('lru_prot', 256, None),
        'phi35-c256-h2o-prot.jsonl': ('h2o_prot', 256, None),
        'phi35-c256-snapkv-prot.jsonl': ('snapkv_prot', 256, None),
        'phi35-c512-lru-prot.jsonl': ('lru_prot', 512, None),
        'phi35-c512-h2o-prot.jsonl': ('h2o_prot', 512, None),
        'phi35-c512-snapkv-prot.jsonl': ('snapkv_prot', 512, None),
    }
    for fname, (key, cap, _) in mapping.items():
        fpath = phidir / fname
        if not fpath.exists():
            continue
        scores = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    scores.append(json.loads(line)['token_f1'])
        if len(scores) >= 50:  # Only use if enough data
            mean_f1 = float(np.mean(scores))
            if key == 'fullcache':
                PHI35['fullcache'] = mean_f1
            else:
                PHI35[key][cap] = mean_f1
            print(f"  Phi-3.5 {fname}: n={len(scores)}, F1={mean_f1:.4f}")

# Try loading Phi-3.5 data
_load_phi35_data()

# Qwen-7B results (loaded dynamically from results)
Q7B = {
    'fullcache': None,
    'lru_noprot': {},
    'lru_prot': {},
    'h2o_prot': {},
    'snapkv_prot': {},
}

def _load_q7b_data():
    """Load Qwen-7B data from results files if available."""
    q7bdir = Path('results/q7b-multimodel')
    if not q7bdir.exists():
        return
    mapping = {
        'q7b-fullcache.jsonl': ('fullcache', None),
        'q7b-c128-lru-noprot.jsonl': ('lru_noprot', 128),
        'q7b-c128-lru-prot.jsonl': ('lru_prot', 128),
        'q7b-c128-h2o-prot.jsonl': ('h2o_prot', 128),
        'q7b-c128-snapkv-prot.jsonl': ('snapkv_prot', 128),
        'q7b-c256-lru-noprot.jsonl': ('lru_noprot', 256),
        'q7b-c256-lru-prot.jsonl': ('lru_prot', 256),
        'q7b-c256-h2o-prot.jsonl': ('h2o_prot', 256),
        'q7b-c256-snapkv-prot.jsonl': ('snapkv_prot', 256),
        'q7b-c512-lru-noprot.jsonl': ('lru_noprot', 512),
        'q7b-c512-lru-prot.jsonl': ('lru_prot', 512),
        'q7b-c512-h2o-prot.jsonl': ('h2o_prot', 512),
        'q7b-c512-snapkv-prot.jsonl': ('snapkv_prot', 512),
    }
    for fname, (key, cap) in mapping.items():
        fpath = q7bdir / fname
        if not fpath.exists():
            continue
        scores = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    scores.append(json.loads(line)['token_f1'])
        if len(scores) >= 50:
            mean_f1 = float(np.mean(scores))
            if key == 'fullcache':
                Q7B['fullcache'] = mean_f1
            else:
                Q7B[key][cap] = mean_f1
            print(f"  Qwen-7B {fname}: n={len(scores)}, F1={mean_f1:.4f}")

_load_q7b_data()

# Protection sensitivity (Qwen-3B, c256, LRU)
SENSITIVITY = {0: 0.011, 5: 0.247, 10: 0.282, 15: 0.285, 20: 0.283}

# ── Figure 1: Capacity curve (F1 vs cache size) ──────────────────────────────

def _plot_model_panel(ax, model_data, title, show_ylabel=True, caps=None):
    """Plot one model panel for the capacity curve figure."""
    if caps is None:
        caps = sorted(set(c for key in model_data if isinstance(model_data[key], dict)
                         for c in model_data[key].keys()))
    
    # Unprotected (dashed + open markers)
    for key, label, color, marker in [
        ('lru_noprot', 'LRU', '#d62728', 'o'),
        ('h2o_noprot', 'H2O', '#ff7f0e', 's'),
        ('snapkv_noprot', 'SnapKV', '#2ca02c', '^'),
    ]:
        data = model_data.get(key, {})
        if not data:
            continue
        x = sorted(data.keys())
        y = [data[c] for c in x]
        ax.plot(x, y, linestyle='--', marker=marker, color=color, alpha=0.75,
                linewidth=1.6, markersize=6, markerfacecolor='white', markeredgewidth=1.2,
                label=f'{label} (no prot)')
    
    # Protected (solid + filled markers)
    for key, label, color, marker in [
        ('lru_prot', 'LRU+prot', '#d62728', 'o'),
        ('h2o_prot', 'H2O+prot', '#ff7f0e', 's'),
        ('snapkv_prot', 'SnapKV+prot', '#2ca02c', '^'),
        ('streamllm_prot', 'SLW+prot', '#9467bd', 'D'),
    ]:
        data = model_data.get(key, {})
        if not data:
            continue
        x = sorted(data.keys())
        y = [data[c] for c in x]
        ax.plot(x, y, linestyle='-', marker=marker, color=color, linewidth=2.1,
                markersize=6, label=label)
    
    fc = model_data.get('fullcache')
    if fc is not None:
        ax.axhline(fc, color='gray', ls=':', alpha=0.7, label='Full cache')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Cache capacity $C$', fontsize=14)
    if show_ylabel:
        ax.set_ylabel('Token F1', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    if caps:
        ax.set_xticks(caps)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_ylim(-0.02, 0.36)
    ax.grid(True, alpha=0.3)


def fig_capacity_curve(outdir: Path):
    """F1 vs cache capacity for protected vs unprotected policies — up to 4 models.

    Arranged in a 2×2 grid so the figure scales well at text width.
    """
    if not HAS_MPL:
        return

    has_phi = PHI35['fullcache'] is not None and len(PHI35['lru_prot']) > 0
    has_q7b = Q7B['fullcache'] is not None and len(Q7B['lru_prot']) > 0

    # Always use 2×2 grid; fill empty cells if fewer than 4 models
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), sharey=True)

    panels = [
        (axes[0, 0], Q3B,  'Qwen2.5-3B (GQA 16Q/2KV)',  True,  [64, 128, 256, 512]),
        (axes[0, 1], Q15B, 'Qwen2.5-1.5B (GQA 12Q/2KV)', False, [128, 256, 512]),
        (axes[1, 0], PHI35 if has_phi else None, 'Phi-3.5-mini (MHA 32Q/32KV)', True,  [128, 256, 512]),
        (axes[1, 1], Q7B   if has_q7b  else None, 'Qwen2.5-7B (GQA 28Q/4KV)',   False, [128, 256, 512]),
    ]

    for ax, data, title, show_y, caps in panels:
        if data is None or data.get('fullcache') is None:
            ax.set_visible(False)
            continue
        _plot_model_panel(ax, data, title, show_ylabel=show_y, caps=caps)
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)

    fig.tight_layout()
    out = outdir / 'fig_capacity_curve.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    n_active = sum(1 for _, data, _, _, _ in panels if data is not None)
    print(f"Saved: {out} (2x2 grid, {n_active} panels)")


def fig_protection_sensitivity(outdir: Path):
    """Protection fraction vs F1 (diminishing returns curve)."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    fracs = sorted(SENSITIVITY.keys())
    f1s = [SENSITIVITY[f] for f in fracs]
    ceiling = Q3B['fullcache']
    
    ax.plot(fracs, f1s, '-o', color='#1f77b4', linewidth=2.5, markersize=8, zorder=5)
    ax.axhline(ceiling, color='gray', ls=':', alpha=0.7, label=f'Full cache ({ceiling:.3f})')
    
    # Annotate key points
    ax.annotate(f'{f1s[0]:.3f}\n(3.6% ceil)',
                xy=(fracs[0], f1s[0]), xytext=(2, 0.06),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')
    ax.annotate(f'{f1s[1]:.3f}\n(78%)',
                xy=(fracs[1], f1s[1]), xytext=(5, 0.20),
                fontsize=9, ha='center', color='#1f77b4')
    ax.annotate(f'{f1s[2]:.3f}\n(89%)',
                xy=(fracs[2], f1s[2]), xytext=(12, 0.32),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center', fontweight='bold')
    
    # Shade "optimal" region
    ax.axvspan(8, 12, alpha=0.1, color='green', label='Optimal range')
    
    ax.set_xlabel('Protection fraction (% each side)', fontsize=12)
    ax.set_ylabel('Token F1 (LRU, $C{=}256$)', fontsize=12)
    ax.set_xticks(fracs)
    ax.set_xticklabels([f'{f}%' for f in fracs])
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out = outdir / 'fig_sensitivity.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_protection_lift_bar(outdir: Path):
    """Bar chart showing protection lift across models and policies."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    
    # Data: (label, no_prot, with_prot) at c256
    groups = [
        ('Q-3B\nLRU', 0.011, 0.282),
        ('Q-3B\nH2O', 0.038, 0.290),
        ('Q-3B\nSnapKV', 0.038, 0.290),
        ('Q-3B\nStreamLLM', 0.027, 0.184),
        ('Q-1.5B\nLRU', 0.012, 0.241),
        ('Q-1.5B\nH2O', None, 0.231),
        ('Q-1.5B\nSnapKV', None, 0.231),
    ]
    
    # Add Phi-3.5 groups if data available
    if PHI35.get('lru_prot', {}).get(256) is not None:
        phi_lru_np = PHI35.get('lru_noprot', {}).get(256)
        phi_lru_p = PHI35['lru_prot'][256]
        groups.append(('Phi-3.5\nLRU', phi_lru_np, phi_lru_p))
        if PHI35.get('h2o_prot', {}).get(256) is not None:
            groups.append(('Phi-3.5\nH2O', None, PHI35['h2o_prot'][256]))
        if PHI35.get('snapkv_prot', {}).get(256) is not None:
            groups.append(('Phi-3.5\nSnapKV', None, PHI35['snapkv_prot'][256]))

    # Add Qwen-7B groups if data available
    if Q7B.get('lru_prot', {}).get(256) is not None:
        q7b_lru_np = Q7B.get('lru_noprot', {}).get(256)
        q7b_lru_p = Q7B['lru_prot'][256]
        groups.append(('Q-7B\nLRU', q7b_lru_np, q7b_lru_p))
        if Q7B.get('h2o_prot', {}).get(256) is not None:
            groups.append(('Q-7B\nH2O', None, Q7B['h2o_prot'][256]))
        if Q7B.get('snapkv_prot', {}).get(256) is not None:
            groups.append(('Q-7B\nSnapKV', None, Q7B['snapkv_prot'][256]))
    
    x = np.arange(len(groups))
    width = 0.35
    
    noprot_vals = [g[1] if g[1] is not None else 0 for g in groups]
    prot_vals = [g[2] for g in groups]
    labels = [g[0] for g in groups]
    
    bars1 = ax.bar(x - width/2, noprot_vals, width, label='No protection', 
                   color='#d62728', alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, prot_vals, width, label='With protection (10%)',
                   color='#2ca02c', alpha=0.7, edgecolor='white')
    
    # Add full-cache lines with model boundaries
    q3b_end = 3.5 / len(groups)
    q15_end = 6.5 / len(groups)
    phi_end = 1.0
    
    ax.axhline(Q3B['fullcache'], color='#1f77b4', ls=':', alpha=0.5, xmin=0, xmax=q3b_end)
    ax.axhline(Q15B['fullcache'], color='#1f77b4', ls=':', alpha=0.5, xmin=q3b_end, xmax=q15_end)
    ax.text(1.5, Q3B['fullcache'] + 0.005, 'Q-3B ceil', fontsize=9, color='#1f77b4', alpha=0.7)
    ax.text(5.0, Q15B['fullcache'] + 0.005, 'Q-1.5B ceil', fontsize=9, color='#1f77b4', alpha=0.7)
    
    if PHI35['fullcache'] is not None and len(groups) > 7:
        ax.axhline(PHI35['fullcache'], color='#1f77b4', ls=':', alpha=0.5, xmin=q15_end, xmax=phi_end)
        ax.text(len(groups) - 1.5, PHI35['fullcache'] + 0.005, 'Phi-3.5 ceil', fontsize=9, color='#1f77b4', alpha=0.7)
    
    # Add improvement labels
    for i, (label, np_val, p_val) in enumerate(groups):
        if np_val is not None and np_val > 0:
            ratio = p_val / np_val
            ax.text(i + width/2, p_val + 0.005, f'{ratio:.0f}\u00d7', 
                    ha='center', fontsize=9, fontweight='bold', color='#2ca02c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Token F1 ($C{=}256$)', fontsize=12)
    ax.set_ylim(0, 0.36)
    ax.legend(fontsize=10.5, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Vertical separators between models
    ax.axvline(3.5, color='gray', ls='-', alpha=0.3)
    ax.axvline(6.5, color='gray', ls='-', alpha=0.3)
    
    fig.tight_layout()
    out = outdir / 'fig_protection_lift.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    outdir = Path('figures')
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig_capacity_curve(outdir)
    fig_protection_sensitivity(outdir)
    fig_protection_lift_bar(outdir)
    
    print(f"\nAll figures saved to {outdir}/")


if __name__ == '__main__':
    main()
