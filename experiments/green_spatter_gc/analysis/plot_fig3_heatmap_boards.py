"""Fig.3 heatmaps, board-parametric.

Generates the 4x4 (size x reuse) co-run efficiency heatmap for every board
directly from results/<board>/spatter_dense_grid.csv, so all three boards'
heatmaps are reproducible from repo data alone (no GPU, no stray CSV).

Outputs paper_figures/fig3_heatmap_<board>.png/.pdf, plus a
fig3_heatmap_4x4.png alias for AGX (orin-b) to match existing references.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]      # .../green_spatter_gc
RESULTS = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "paper_figures"

BOARDS = ["orin-b", "orin-nx", "orin-nano"]
ALIAS = {"orin-b": "fig3_heatmap_4x4"}           # keep AGX's canonical name too

PATTERNS = [("uniform", "(a) Uniform"), ("strided", "(b) Strided"), ("scatter", "(c) Scatter")]
SIZES = [1, 4, 16, 64]
REUSES = [32, 16, 4, 1]


def load_board(board):
    path = RESULTS / board / "spatter_dense_grid.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["pattern"], int(float(row["size_mb"])), int(float(row["iters"])))
            out[key] = float(row["aggregate_ratio_p50_pct"])
    return out


def render(board, data):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_blue", ["#FFFFFF", "#D8EFF9", "#8BD0EF", "#4DB3E6", "#14476B"])

    fig = plt.figure(figsize=(7.15, 2.45), dpi=240)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = None
    for ax, (pattern, title) in zip(axes, PATTERNS):
        matrix = np.array([[data[(pattern, s, r)] for s in SIZES] for r in REUSES])
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        for i, _ in enumerate(REUSES):
            for j, _ in enumerate(SIZES):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.0,
                        fontweight="bold", color="white" if v >= 62 else "#14476B")
                if v >= 90:
                    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor="#14476B", linewidth=1.0))
        ax.set_title(title, pad=4, fontweight="bold")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.set_yticks(range(len(REUSES)))
        ax.set_yticklabels([str(r) for r in REUSES] if ax is axes[0] else [])
        ax.set_xlabel("Working-set size (MB)", labelpad=3)
        if ax is axes[0]:
            ax.set_ylabel("Reuse count", labelpad=4)
        ax.set_xticks(np.arange(-0.5, len(SIZES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(REUSES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, cax=cax, ticks=[0, 25, 50, 75, 100])
    cbar.set_label("Co-run efficiency (%)", labelpad=5)
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.22, top=0.90)

    names = [f"fig3_heatmap_{board}"]
    if board in ALIAS:
        names.append(ALIAS[board])
    for name in names:
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"{name}.{ext}"
            fig.savefig(out, bbox_inches="tight", pad_inches=0.035)
            print(out)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    for board in BOARDS:
        render(board, load_board(board))


if __name__ == "__main__":
    main()
