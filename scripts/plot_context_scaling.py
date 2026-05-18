#!/usr/bin/env python3
"""Generate context-length scaling figure: protection lift across 1.9K, 11K, 32K.

Shows that protection dominance is context-length-invariant (or strengthens).

Usage:
    python scripts/plot_context_scaling.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def load_f1s(path):
    f1s = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            d = json.loads(line)
            f1s.append(d.get('token_f1', 0.0))
    return np.array(f1s)


def bootstrap_ci(scores, n_boot=10000, ci=0.95):
    arr = np.array(scores)
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


def wilcoxon_p(a, b):
    n = min(len(a), len(b))
    diffs = a[:n] - b[:n]
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 10:
        return float('nan')
    _, p = wilcoxon(nonzero)
    return p


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'figures/'

    # 1.9K context: (Qwen-3B, C=256, ~13% retention — closest to 12.5% at 32K)
    short_ctx = {
        'prot': 'results/confound-fix/q25-c256-lru-protected.jsonl',
        'noprot': 'results/sweep-20260309-202648/q25-c256-lru.jsonl',
        'fullcache': 'results/confound/q25-c2048-fullcache.jsonl',
    }

    # 11K context: from longctx experiments (Qwen-3B, C=1024)
    mid_ctx = {
        'prot': 'results/longctx/q3b-longctx-c1024-prot.jsonl',
        'noprot': 'results/longctx/q3b-longctx-c1024-noprot.jsonl',
        'fullcache': 'results/longctx/q3b-longctx-fullcache.jsonl',
    }

    # 32K context: from current experiments (Qwen-7B, C=4096)
    long_ctx_dir = 'results/32k-partial-v4-r6/'
    if os.path.isdir('results/32k/'):
        # Use final results if available
        test_path = os.path.join('results/32k/', 'q7b-32k-c4096-lru-prot.jsonl')
        if os.path.exists(test_path):
            long_ctx_dir = 'results/32k/'

    long_ctx = {
        'prot': os.path.join(long_ctx_dir, 'q7b-32k-c4096-lru-prot.jsonl'),
        'noprot': os.path.join(long_ctx_dir, 'q7b-32k-c4096-lru-noprot.jsonl'),
        'fullcache': os.path.join(long_ctx_dir, 'q7b-32k-c32768-lru-noprot.jsonl'),
    }

    # 32K context: Q3B cross-model replication
    long_ctx_q3b = {
        'prot': 'results/32k-q3b-final/q3b-32k-c4096-lru-prot.jsonl',
        'noprot': 'results/32k-q3b-final/q3b-32k-c4096-lru-noprot.jsonl',
        'fullcache': 'results/32k-q3b-final/q3b-32k-c32768-lru-noprot.jsonl',
    }

    # 64K context: Qwen3-4B-Instruct-2507 NIAH (C=4096, ~6.3% retention)
    ultra_ctx = {
        'prot': 'results/q3-4b-2507-niah-64k/q3-4b-2507-64k-c4096-lru-prot.jsonl',
        'noprot': 'results/q3-4b-2507-niah-64k/q3-4b-2507-64k-c4096-lru-noprot.jsonl',
        'fullcache': 'results/q3-4b-2507-niah-64k/q3-4b-2507-64k-fullcache-lru-noprot.jsonl',
    }

    # 64K context: Random+prot for divergence comparison
    ultra_ctx_random = {
        'prot': 'results/64k-qwen35/q35-64k-c8192-random-prot.jsonl',
        'noprot': 'results/64k-qwen35/q35-64k-c8192-random-noprot.jsonl',
        'fullcache': 'results/64k-qwen35/q35-64k-fullcache-lru-noprot.jsonl',
    }

    # Load and compute metrics for each context length
    # Auto-update Q7B label when final data is available
    q7b_label = '32K (Q7B, N=46*)'
    if long_ctx_dir == 'results/32k/':
        q7b_label = '32K (Q7B, N=58)'

    contexts = [
        ('1.9K (Q3B, N=162)', short_ctx),
        ('11K (Q3B, N=48)', mid_ctx),
        ('32K (Q3B, N=58)', long_ctx_q3b),
        (q7b_label, long_ctx),
        ('64K (Q3-4B, N=60)', ultra_ctx),
    ]

    data_points = []
    for label, paths in contexts:
        available = all(os.path.exists(p) for p in paths.values())
        if not available:
            missing = [k for k, p in paths.items() if not os.path.exists(p)]
            print(f"  SKIP {label}: missing {missing}")
            continue

        fc = load_f1s(paths['fullcache'])
        prot = load_f1s(paths['prot'])
        noprot = load_f1s(paths['noprot'])

        fc_mean, fc_lo, fc_hi = bootstrap_ci(fc)
        prot_mean, prot_lo, prot_hi = bootstrap_ci(prot)
        noprot_mean, noprot_lo, noprot_hi = bootstrap_ci(noprot)

        prot_pct = prot_mean / fc_mean * 100 if fc_mean > 0 else 0
        noprot_pct = noprot_mean / fc_mean * 100 if fc_mean > 0 else 0

        p_lift = wilcoxon_p(prot, noprot)

        data_points.append({
            'label': label,
            'fc': (fc_mean, fc_lo, fc_hi),
            'prot': (prot_mean, prot_lo, prot_hi),
            'noprot': (noprot_mean, noprot_lo, noprot_hi),
            'prot_pct': prot_pct,
            'noprot_pct': noprot_pct,
            'p_lift': p_lift,
            'n': min(len(prot), len(noprot)),
        })

    if len(data_points) < 2:
        print("Need at least 2 context lengths to plot scaling figure.")
        sys.exit(1)

    # Print summary
    print("\nContext-Length Scaling Summary:")
    print(f"{'Context':<12} {'Ceiling':>8} {'Prot':>8} {'NoProt':>8} {'%Ceil(P)':>10} {'%Ceil(N)':>10} {'p':>8}")
    for d in data_points:
        print(f"{d['label'].replace(chr(10),' '):<12} "
              f"{d['fc'][0]:>8.4f} "
              f"{d['prot'][0]:>8.4f} "
              f"{d['noprot'][0]:>8.4f} "
              f"{d['prot_pct']:>9.1f}% "
              f"{d['noprot_pct']:>9.1f}% "
              f"{d['p_lift']:>8.4f}")

    def short_label(label: str) -> str:
        """First line only (e.g. '1.9K' from '1.9K\\n(Q3B, N=162)')."""
        return label.split('\n')[0].strip()

    # Figure: % of ceiling across context lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: % of ceiling recovery
    x = np.arange(len(data_points))
    width = 0.3

    ax1.bar(x - width/2, [d['prot_pct'] for d in data_points], width,
            color='#4ca84c', alpha=0.85, label='LRU + prot', edgecolor='white', hatch='')
    ax1.bar(x + width/2, [d['noprot_pct'] for d in data_points], width,
            color='#d94f4f', alpha=0.85, label='LRU (no prot)', edgecolor='white', hatch='//')

    ax1.axhline(y=100, color='#4a86c8', linestyle='--', linewidth=1, alpha=0.5,
                label='Full-cache ceiling')

    # Add p-value annotations
    for i, d in enumerate(data_points):
        p = d['p_lift']
        if np.isnan(p):
            annot = 'n.s.'
        else:
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            annot = f'p={p:.3f}{stars}'
        y_top = max(d['prot_pct'], d['noprot_pct']) + 8
        ax1.annotate(annot,
                    xy=(i, y_top), ha='center', fontsize=10, color='#333')

    ax1.set_xticks(x)
    ax1.set_xticklabels([d['label'] for d in data_points], fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
    ax1.set_ylabel('% of Full-Cache Ceiling', fontsize=12)
    ax1.set_title('(a) Cache Recovery vs. Context Length', fontsize=12, pad=8)
    ax1.set_ylim(0, 160)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Protection lift (absolute F1 delta)
    lifts = [d['prot'][0] - d['noprot'][0] for d in data_points]
    colors = ['#4ca84c' if l > 0 else '#d94f4f' for l in lifts]
    ax2.bar(x, lifts, 0.5, color=colors, alpha=0.85, edgecolor='white')

    for i, (l, d) in enumerate(zip(lifts, data_points)):
        ax2.text(i, l + 0.005, f'+{l:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#333')

    ax2.set_xticks(x)
    ax2.set_xticklabels([short_label(d['label']) for d in data_points], fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')
    ax2.set_ylabel('Protection Lift (F1)', fontsize=12)
    ax2.set_title('(b) Absolute Protection Lift', fontsize=12, pad=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        outpath = os.path.join(out_dir, f'fig_context_scaling.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close()


if __name__ == '__main__':
    main()
