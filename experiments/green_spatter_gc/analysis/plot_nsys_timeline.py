"""Kernel timeline for the single-stream device//device vs device//zero-copy runs.

Draws the actual nsys-recorded kernel intervals so the two things the wall-clock
numbers only imply become visible: the kernels never overlap (one stream is
ordered), and the device kernel itself shrinks when its partner is zero-copy.

Usage: plot_nsys_timeline.py <dir with mix_only2.sqlite and mix_only3.sqlite>
"""
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nsys_overlap")
OUT_DIR = Path(__file__).resolve().parent / "paper_figures"

DEV = "#1f5d7a"
OTHER_DEV = "#6f9db1"
ZC = "#c44e52"
WARMUP_PAIRS = 5
SHOW_PAIRS = 6


def load(only):
    rows = sqlite3.connect(str(SRC / f"mix_only{only}.sqlite")).execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    return rows[WARMUP_PAIRS * 2:][: SHOW_PAIRS * 2]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    cases = [(2, "device + device", OTHER_DEV), (3, "device + zero-copy", ZC)]
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 2.1), dpi=240, sharex=True)

    for ax, (only, title, partner_color) in zip(axes, cases):
        rows = load(only)
        t0 = rows[0][0]
        for i, (s, e) in enumerate(rows):
            first = i % 2 == 0
            ax.broken_barh([((s - t0) / 1000, (e - s) / 1000)], (0.15, 0.7),
                           facecolor=DEV if first else partner_color,
                           edgecolor="white", linewidth=0.6)
            ax.text((s - t0) / 1000 + (e - s) / 2000, 0.5, f"{(e-s)/1000:.0f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if first else "#222222")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, loc="left", pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time since first kernel (us)   -- bar labels are kernel duration in us")

    handles = [mpatches.Patch(color=DEV, label="device-path kernel"),
               mpatches.Patch(color=OTHER_DEV, label="partner: device"),
               mpatches.Patch(color=ZC, label="partner: zero-copy")]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.42),
                   ncol=3, frameon=False, fontsize=7)

    fig.suptitle("One stream, kernels never overlap; the device kernel shrinks "
                 "when its partner is zero-copy", fontsize=8.5, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "nsys_single_stream_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
