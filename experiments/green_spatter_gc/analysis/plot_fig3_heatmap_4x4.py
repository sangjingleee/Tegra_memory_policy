from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "dense_efficiency.csv",
    ROOT / "remote_archive_20260720" / "green_ctx_0522" / "dense_efficiency.csv",
]
DATA = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
OUT_DIR = ROOT / "paper_figures"

PATTERNS = [("uniform", "(a) Uniform"), ("strided", "(b) Strided"), ("scatter", "(c) Scatter")]
SIZES = [1, 4, 16, 64]
REUSES = [32, 16, 4, 1]


def load_rows():
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = (row["pattern"], int(float(row["size_mb"])), int(float(row["reuse"])))
        out[key] = float(row["efficiency_pct"])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_rows()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_blue",
        ["#FFFFFF", "#D8EFF9", "#8BD0EF", "#4DB3E6", "#14476B"],
    )

    fig = plt.figure(figsize=(7.15, 2.45), dpi=240)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = None
    for ax, (pattern, title) in zip(axes, PATTERNS):
        matrix = np.array([[data[(pattern, size, reuse)] for size in SIZES] for reuse in REUSES])
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        for i, reuse in enumerate(REUSES):
            for j, size in enumerate(SIZES):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="white" if value >= 62 else "#14476B",
                )
                if value >= 90:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#14476B",
                            linewidth=1.0,
                        )
                    )

        ax.set_title(title, pad=4, fontweight="bold")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([str(size) for size in SIZES])
        ax.set_yticks(range(len(REUSES)))
        ax.set_yticklabels([str(reuse) for reuse in REUSES] if ax is axes[0] else [])
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

    out_png = OUT_DIR / "fig3_heatmap_4x4.png"
    out_pdf = OUT_DIR / "fig3_heatmap_4x4.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
