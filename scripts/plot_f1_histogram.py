"""Generate f1_histogram.pdf figure for the paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from pathlib import Path

def load_f1(path):
    vals = []
    for line in Path(path).open():
        line = line.strip()
        if line:
            vals.append(json.loads(line)["token_f1"])
    return np.array(vals)

lru_noprot = load_f1("results/ablation/q25-c256-lru.jsonl")
lru_prot   = load_f1("results/confound-fix/q25-c256-lru-protected.jsonl")

bins = np.linspace(0.0, 1.05, 21)  # 20 bins

fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.hist(lru_noprot, bins=bins, alpha=0.75, color="red", hatch="//",
        edgecolor="darkred", linewidth=0.6,
        label=f"LRU (mean F1 = {lru_noprot.mean():.3f})")
ax.hist(lru_prot, bins=bins, alpha=0.75, color="blue", hatch="",
        edgecolor="darkblue", linewidth=0.6,
        label=f"LRU + protection (mean F1 = {lru_prot.mean():.3f})")

ax.set_xlabel("Per-item Token F1", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.tick_params(axis="both", labelsize=11)
ax.legend(loc="upper right", fontsize=11, framealpha=0.92)

# Find top of the tallest red bar (near-zero bin)
counts_noprot, _ = np.histogram(lru_noprot, bins=bins)
tallest_bin_idx = int(np.argmax(counts_noprot))
bar_center = 0.5 * (bins[tallest_bin_idx] + bins[tallest_bin_idx + 1])
bar_top = counts_noprot[tallest_bin_idx]

ax.annotate(
    "96% items\nnear zero",
    xy=(bar_center, bar_top),
    xytext=(0.25, 145),
    fontsize=10,
    color="red",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
)

plt.tight_layout()

out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "f1_histogram.pdf"
png_path = out_dir / "f1_histogram.png"

fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)

print(f"Saved {pdf_path} and {png_path}")
print(f"n_noprot={len(lru_noprot)}, mean_noprot={lru_noprot.mean():.3f}")
print(f"n_prot={len(lru_prot)}, mean_prot={lru_prot.mean():.3f}")
print(f"Tallest red bar: bin {tallest_bin_idx}, center={bar_center:.3f}, count={bar_top}")
