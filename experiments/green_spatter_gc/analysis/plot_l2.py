#!/usr/bin/env python3
# L2 cache-partition: mechanism control (works when hot set fits L2) vs
# real model (ACT+ResNet) where it doesn't help (weights >> 4MB L2).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ctl = pd.read_csv("results_l2_control.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) control: preserve vs crit working-set, pin off vs on
ax = axes[0]
for pin, ls, mk, lab in [(0, "--", "x", "no pin"), (1, "-", "o", "L2 pin (persisting)")]:
    sub = ctl[ctl["pin"] == pin].sort_values("crit_mb")
    ax.plot(sub["crit_mb"], sub["preserve_pct"], ls + mk, lw=2,
            color=("#c62828" if pin == 0 else "#2e7d32"), label=lab)
ax.axvspan(0.8, 2.75, color="#2e7d32", alpha=0.08)
ax.text(1.0, 30, "fits L2\nset-aside\n(<=2.75MB)", fontsize=8, color="#2e7d32")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("critical reuse working-set (MB)")
ax.set_ylabel("critical bandwidth preserved (%)")
ax.set_title("(a) CONTROL: mechanism works when hot set fits L2\n"
             "pin lifts preserve for small buffers; no effect once > L2", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) real model ACT+ResNet: p99 vs L2 set-aside (within noise)
ax = axes[1]
off = [0.575168, 0.585888, 0.583680]
on  = [0.556736, 0.587392, 0.582272]
import numpy as np
ax.scatter([0]*3, [v*1000 for v in off], color="#c62828", s=60, label="L2 carveout off")
ax.scatter([1]*3, [v*1000 for v in on],  color="#2e7d32", s=60, label="L2 carveout 2MB")
ax.plot([-0.3, 0.3], [np.mean(off)*1000]*2, color="#c62828", lw=2)
ax.plot([0.7, 1.3], [np.mean(on)*1000]*2,  color="#2e7d32", lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["off", "2 MB"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("ACT critical p99 (us)")
ax.set_xlabel("L2 persisting set-aside")
ax.set_title("(b) REAL MODEL ACT (weights ~25MB) + ResNet50\n"
             "carveout gives no reliable change (within run-to-run noise)", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("L2 cache partition on Jetson Orin (L2 = 4MB, max set-aside = 2.75MB)\n"
             "works only when the critical hot set fits L2 -> ineffective for real large-weight models",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("l2_partition.png", dpi=130)
print("wrote l2_partition.png")
