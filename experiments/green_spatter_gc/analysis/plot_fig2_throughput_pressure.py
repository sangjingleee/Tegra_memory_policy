from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "results" / "clocked_spatter_20260626",
    ROOT / "remote_archive_20260720" / "green_ctx_0522" / "results" / "clocked_spatter_20260626",
]
DATA = next((path for path in DATA_CANDIDATES if (path / "campaign_summary.csv").exists()), DATA_CANDIDATES[0])
OUT_DIR = ROOT / "paper_figures"


SELECTED = [
    "front_uniform_1mb_r32",
    "front_strided_1mb_r32",
    "front_uniform_4mb_r32",
    "back_uniform_64mb_r1",
    "back_strided_64mb_r1",
    "back_scatter_64mb_r1",
]

LABELS = [
    "Uniform\n1MB / R32",
    "Strided\n1MB / R32",
    "Uniform\n4MB / R32",
    "Uniform\n64MB / R1",
    "Strided\n64MB / R1",
    "Scatter\n64MB / R1",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    campaign = {row["label"]: row for row in read_csv(DATA / "campaign_summary.csv")}
    actmon = {row["label"]: row for row in read_csv(DATA / "actmon_summary.csv")}

    rows = []
    for label in SELECTED:
        row = dict(campaign[label])
        row["mc_avg_activity_p50"] = actmon[label]["mc_avg_activity_p50"]
        for key in [
            "size_mb",
            "iters",
            "dev_solo_p50_gbps",
            "zc_solo_p50_gbps",
            "dev_co_p50_gbps",
            "zc_co_p50_gbps",
            "solo_sum_p50_gbps",
            "aggregate_common_p50_gbps",
            "aggregate_ratio_p50_pct",
            "mc_avg_activity_p50",
        ]:
            row[key] = float(row[key])
        row["preservation_pct"] = row["aggregate_ratio_p50_pct"]
        rows.append(row)

    max_mc = max(row["mc_avg_activity_p50"] for row in rows)
    for row in rows:
        row["mc_activity_norm_pct"] = row["mc_avg_activity_p50"] / max_mc * 100.0

    x = np.arange(len(rows))
    w = 0.24

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.15, 2.65), dpi=240)
    ax2 = ax.twinx()

    solo_dev_color = "#1f5d7a"
    solo_zc_color = "#6f9db1"
    corun_dev_color = "#49a9d8"
    corun_zc_color = "#a8d8ef"
    pressure_color = "#c44e52"

    solo_dev = ax.bar(
        x - w,
        [row["dev_solo_p50_gbps"] for row in rows],
        width=w,
        label="Solo device",
        color=solo_dev_color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    solo_zc = ax.bar(
        x - w,
        [row["zc_solo_p50_gbps"] for row in rows],
        bottom=[row["dev_solo_p50_gbps"] for row in rows],
        width=w,
        label="Solo ZC",
        color=solo_zc_color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    corun_dev = ax.bar(
        x,
        [row["dev_co_p50_gbps"] for row in rows],
        width=w,
        label="Co-run device",
        color=corun_dev_color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    corun_zc = ax.bar(
        x,
        [row["zc_co_p50_gbps"] for row in rows],
        bottom=[row["dev_co_p50_gbps"] for row in rows],
        width=w,
        label="Co-run ZC",
        color=corun_zc_color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    pressure = ax2.bar(
        x + w,
        [row["mc_activity_norm_pct"] for row in rows],
        width=w,
        label="Memory-controller pressure",
        color=pressure_color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.78,
        zorder=2,
    )

    max_solo = max(row["solo_sum_p50_gbps"] for row in rows)
    for i, row in enumerate(rows):
        pct = row["preservation_pct"]
        y = row["aggregate_common_p50_gbps"]
        ax.text(
            i,
            y + max_solo * 0.022,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color="#222222",
        )

    ax.axvline(2.5, color="#7a7a7a", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.text(1.0, 333, "regular + high reuse\npreserved", ha="center", va="top", fontsize=7.0, color="#365f2c")
    ax.text(5.0, 333, "pattern-dependent\ncollapse", ha="center", va="top", fontsize=7.0, color="#7b2d2f")

    ax.set_ylabel("Effective read throughput (GB/s)")
    ax2.set_ylabel("Memory-controller pressure")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 350)
    ax2.set_ylim(0, 110)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    handles = [solo_dev, solo_zc, corun_dev, corun_zc, pressure]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    out_png = OUT_DIR / "fig2_throughput_pressure.png"
    out_pdf = OUT_DIR / "fig2_throughput_pressure.pdf"
    out_csv = OUT_DIR / "fig2_throughput_pressure_data.csv"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "pattern",
            "size_mb",
            "iters",
            "dev_solo_p50_gbps",
            "zc_solo_p50_gbps",
            "dev_co_p50_gbps",
            "zc_co_p50_gbps",
            "solo_sum_p50_gbps",
            "aggregate_common_p50_gbps",
            "preservation_pct",
            "mc_avg_activity_p50",
            "mc_activity_norm_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    print(out_png)
    print(out_pdf)
    print(out_csv)


if __name__ == "__main__":
    main()
