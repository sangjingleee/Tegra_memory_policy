#!/usr/bin/env python3
# Visualize ZC-vs-device co-run isolation: does a zero-copy background protect
# the device (critical) path's bandwidth? (spatter gather, 8:8 SM split)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_zc_isolation.csv")
PAT_C = {"uniform": "#1565c0", "strided": "#2e7d32", "scatter": "#c62828"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for col, ax, title in [
    ("device_preserve_ratio_pct", axes[0],
     "(a) Critical (device-path) bandwidth preserved\nwith a zero-copy background co-running"),
    ("aggregate_ratio_pct", axes[1],
     "(b) Aggregate BW / sum-of-solo\n100% = device & ZC paths fully independent"),
]:
    for pat, c in PAT_C.items():
        for it, ls, mk in [(32, "-", "o"), (1, "--", "x")]:
            sub = df[(df["pattern"] == pat) & (df["iters"] == it)].sort_values("size_mb")
            ax.plot(sub["size_mb"], sub[col], ls + mk, color=c,
                    label=f"{pat}, reuse={it}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("working-set size (MB)")
    ax.set_ylabel(col.replace("_pct", " (%)").replace("_", " "))
    ax.axhline(100, ls=":", color="k", lw=0.8)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)

fig.suptitle("Zero-copy as memory-path isolation on Jetson Orin (Green Context 8:8)\n"
             "high-reuse uniform/strided: ZC background barely touches the critical device path",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("zc_isolation.png", dpi=130)
print("wrote zc_isolation.png")
