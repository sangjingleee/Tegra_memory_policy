#!/usr/bin/env python3
"""Summarize the clock-locked Spatter/Green-Context campaign without overclaiming cache topology."""
from __future__ import annotations

import sys
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(outdir: str) -> None:
    root = Path(outdir)
    bw = pd.read_csv(root / "bandwidth_runs.csv")
    actmon = pd.read_csv(root / "actmon_summary.csv")
    df = bw.merge(actmon[["label", "mc_avg_activity_p50", "mc_avg_activity_p95", "mc_last_activity_p50"]], on="label", how="left")
    df.to_csv(root / "campaign_summary.csv", index=False)

    dram_note = "No separate streaming-DRAM calibration file was found."
    dram_read_peak = None
    calibration = root / "dram_peak.txt"
    if calibration.exists():
        text = calibration.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"stream READ peak\s*:\s*([0-9.]+) GB/s", text)
        if match:
            dram_read_peak = float(match.group(1))
            dram_note = f"Separate 512MB stream-read calibration: {dram_read_peak:.1f} GB/s."

    colors = {"uniform": "#1f77b4", "strided": "#2ca02c", "scatter": "#d62728"}
    x = range(len(df))
    labels = [f"{r.pattern}\n{r.size_mb}MB,r{r.iters}" for r in df.itertuples()]
    ratio = df["aggregate_ratio_p50_pct"]
    lower = ratio - df["aggregate_ratio_p05_pct"]
    upper = df["aggregate_ratio_p95_pct"] - ratio

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(x, ratio, yerr=[lower, upper], fmt="none", ecolor="#444444", capsize=3, zorder=1)
    ax.scatter(x, ratio, s=72, c=[colors[p] for p in df["pattern"]], zorder=2)
    ax.axhline(100, color="#222222", linestyle="--", linewidth=1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("co-run aggregate / solo-sum (%)")
    ax.set_xticks(list(x), labels, rotation=35, ha="right")
    ax.set_title("Green Context 8SM:8SM co-run efficiency\n(error bars: within-run p05-p95)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    width = 0.36
    ax.bar([i - width / 2 for i in x], df["aggregate_common_p50_gbps"], width,
           color=[colors[p] for p in df["pattern"]], label="co-run aggregate")
    ax.bar([i + width / 2 for i in x], df["solo_sum_p50_gbps"], width,
           color="#c7c7c7", edgecolor="#555555", label="sum of solo paths")
    if dram_read_peak is not None:
        ax.axhline(dram_read_peak, color="#222222", linestyle="--", linewidth=1.2,
                   label=f"512MB stream-read DRAM baseline ({dram_read_peak:.1f} GB/s)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_xticks(list(x), labels, rotation=35, ha="right")
    ax.set_title("Effective throughput versus measured DRAM baseline\n(cache service can exceed the DRAM-only baseline)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Clock-locked device + mapped-zero-copy co-run characterization", fontsize=13)
    fig.savefig(root / "clocked_spatter_efficiency_emc.png", dpi=180)

    lines = [
        "# Clock-locked Spatter + Green Context campaign",
        "",
        "- Platform clock state is captured in `clock_state.txt`.",
        "- Green Context requests 8 SM for device and 8 SM for mapped zero-copy; actual counts are in the CSV.",
        "- Aggregate is computed from total bytes over the common host wall interval, not by adding unrelated best trials.",
        "- MC_ALL ACTMON activity is a shared memory-controller hardware signal in raw units, not a per-path DRAM GB/s counter.",
        "- ACTMON is sampled over the full benchmark process (solo calibration plus co-run); it is retained as a raw diagnostic, not used for per-phase attribution.",
        "- This measures performance behavior; it does not directly prove SLC residency or physical cache topology.",
        f"- {dram_note}",
        "",
        "## Results",
        "",
        "```csv",
        df[["label", "aggregate_ratio_p50_pct", "aggregate_ratio_p05_pct", "aggregate_ratio_p95_pct", "mc_avg_activity_p50", "overlap_p50_ratio"]].to_csv(index=False, float_format="%.2f").strip(),
        "```",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_clocked_spatter_campaign.py <output-dir>")
    main(sys.argv[1])
