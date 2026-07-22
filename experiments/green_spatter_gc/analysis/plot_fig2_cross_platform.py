"""Fig.2 (cross-platform variant): co-run stacked throughput per board.

Same representative x-axis as the AGX detailed Fig.2, but instead of the
solo-vs-corun pair it draws three bars per case (AGX / NX / Nano). Each bar is
the co-run throughput split into device + ZC; a faint outlined bar behind it
marks the solo-sum ceiling, and the preservation ratio is printed on top.

Reads each board's spatter_dense_grid.csv directly -- no GPU needed.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]          # .../green_spatter_gc
RESULTS = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "paper_figures"

BOARDS = [
    ("orin-b", "AGX Orin\n8/8 SM"),
    ("orin-nx", "Orin NX\n4/4 SM"),
    ("orin-nano", "Orin Nano\n4/4 SM"),
]

# (pattern, size_mb, iters)
SELECTED = [
    ("uniform", 1, 32),
    ("strided", 1, 32),
    ("uniform", 4, 32),
    ("uniform", 64, 1),
    ("strided", 64, 1),
    ("scatter", 64, 1),
]
LABELS = [
    "Uniform\n1MB / R32",
    "Strided\n1MB / R32",
    "Uniform\n4MB / R32",
    "Uniform\n64MB / R1",
    "Strided\n64MB / R1",
    "Scatter\n64MB / R1",
]


def load_board(board):
    path = RESULTS / board / "spatter_dense_grid.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["pattern"], int(float(row["size_mb"])), int(float(row["iters"])))
            out[key] = {
                "dev_co": float(row["dev_co_p50_gbps"]),
                "zc_co": float(row["zc_co_p50_gbps"]),
                "solo_sum": float(row["solo_sum_p50_gbps"]),
                "ratio": float(row["aggregate_ratio_p50_pct"]),
            }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {b: load_board(b) for b, _ in BOARDS}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    dev_color = "#1f5d7a"   # dark device (matches fig2_throughput_pressure solo palette)
    zc_color = "#6f9db1"    # ZC (lighter, on top)

    n_cases = len(SELECTED)
    n_boards = len(BOARDS)
    w = 0.26
    offsets = np.linspace(-w, w, n_boards)
    x = np.arange(n_cases)

    fig, ax = plt.subplots(figsize=(7.15, 2.9), dpi=240)

    dev_handle = zc_handle = None
    for bi, (board, _) in enumerate(BOARDS):
        xs = x + offsets[bi]
        dev = np.array([data[board][c]["dev_co"] for c in SELECTED])
        zc = np.array([data[board][c]["zc_co"] for c in SELECTED])
        total = dev + zc
        ratio = [data[board][c]["ratio"] for c in SELECTED]

        # co-run device + ZC
        dev_handle = ax.bar(xs, dev, width=w, color=dev_color,
                           edgecolor="white", linewidth=0.4, zorder=3,
                           label="Co-run device")
        zc_handle = ax.bar(xs, zc, width=w, bottom=dev, color=zc_color,
                          edgecolor="white", linewidth=0.4, zorder=3,
                          label="Co-run ZC")
        # preservation % on top of the co-run bar
        for xi, ti, r in zip(xs, total, ratio):
            ax.text(xi, ti + 4, f"{r:.0f}", ha="center", va="bottom",
                    fontsize=5.6, color="#333333", rotation=0)
        # board initial under each sub-bar
        for xi in xs:
            ax.text(xi, -0.055, ["B", "NX", "NN"][bi], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.4, color="#666666")

    ax.axvline(2.5, color="#7a7a7a", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(1.0, ax.get_ylim()[1], "", ha="center")
    ax.set_ylabel("Co-run read throughput (GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.tick_params(axis="x", pad=12)   # room for board initials
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(1.0, ax.get_ylim()[1] * 0.97, "regular + high reuse\npreserved (~95-99%)",
            ha="center", va="top", fontsize=6.6, color="#365f2c")
    ax.text(4.5, ax.get_ylim()[1] * 0.97, "pattern-dependent\ncollapse",
            ha="center", va="top", fontsize=6.6, color="#7b2d2f")

    handles = [dev_handle, zc_handle]
    labels = ["Co-run device", "Co-run ZC"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=2, frameon=False)

    # numbers under bars: which board is which
    ax.text(0.005, -0.24, "B=AGX  NX=Orin NX  NN=Orin Nano   (number on top = preservation %)",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color="#666666")

    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])
    out_png = OUT_DIR / "fig2_cross_platform_throughput.png"
    out_pdf = OUT_DIR / "fig2_cross_platform_throughput.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
