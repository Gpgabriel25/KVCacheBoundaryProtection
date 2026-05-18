#!/usr/bin/env python3
"""Generate 32K context protection effect figure.

Creates a bar chart showing F1 scores at 32K context for:
- Full cache (ceiling)
- LRU (no protection)
- LRU + protection
- Random + protection
- [Optional wave 2: H2O+prot, SnapKV+prot]

With horizontal line at full-cache ceiling and bootstrap 95% CIs.

Usage:
    python scripts/plot_32k_protection.py results/32k/
    python scripts/plot_32k_protection.py results/32k-partial-v4/  # partial data
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'figures/'

    # Two models side by side
    models = [
        {
            'name': 'Qwen-3B (n=58)',
            'dir': 'results/32k-q3b-final/',
            'prefix': 'q3b-32k',
        },
        {
            'name': 'Qwen-7B (n=46)',
            'dir': 'results/32k-partial-v4-r6/',
            'prefix': 'q7b-32k',
        },
    ]
    # Override Q7B with final results if available
    if os.path.isdir('results/32k/') and os.path.exists('results/32k/q7b-32k-c4096-lru-prot.jsonl'):
        models[1]['dir'] = 'results/32k/'
        models[1]['name'] = 'Qwen-7B (n=58)'

    conditions = [
        ('c32768-lru-noprot', 'Full cache', '#4a86c8'),
        ('c4096-lru-noprot', 'LRU (no prot)', '#d94f4f'),
        ('c4096-lru-prot', 'LRU + prot', '#4ca84c'),
        ('c4096-random-prot', 'Random + prot', '#8b6bb5'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=False)

    for ax, model in zip(axes, models):
        bars = []
        for suffix, label, color in conditions:
            fname = f"{model['prefix']}-{suffix}.jsonl"
            path = os.path.join(model['dir'], fname)
            if os.path.exists(path):
                f1s = load_f1s(path)
                if len(f1s) > 0:
                    mean, lo, hi = bootstrap_ci(f1s)
                    bars.append((label, mean, lo, hi, color, len(f1s)))

        if len(bars) < 2:
            ax.set_title(f"{model['name']}: insufficient data")
            continue

        ceiling = bars[0][1] if bars[0][0].startswith('Full') else None

        x = np.arange(len(bars))
        width = 0.6

        for i, (label, mean, lo, hi, color, n) in enumerate(bars):
            yerr_lo = mean - lo
            yerr_hi = hi - mean
            ax.bar(i, mean, width, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
            ax.errorbar(i, mean, yerr=[[yerr_lo], [yerr_hi]],
                        fmt='none', color='#333', capsize=4, linewidth=1.2)
            pct = mean / ceiling * 100 if ceiling and ceiling > 0 else 0
            ax.text(i, mean + yerr_hi + 0.002, f'{pct:.0f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')

        if ceiling is not None:
            ax.axhline(y=ceiling, color='#4a86c8', linestyle='--', linewidth=1,
                       alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=10)
        ax.set_ylabel('Token F1', fontsize=11)
        ax.set_title(model['name'], fontsize=11, pad=8)
        ax.set_ylim(0, max(b[3] for b in bars) * 1.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('32K Context: Protection Effect (C=4096, 12.5% retention)', fontsize=12, y=1.02)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        outpath = os.path.join(out_dir, f'fig_32k_protection.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
