"""device(copy+read) vs zero-copy(read-only) crossover.

The steady-state read benchmark never times the H2D copy that a cudaMalloc
pipeline must pay before its kernel can read, so zero-copy is charged its
slower reads while the device path gets its copy for free. This joins the
measured per-point solo read throughputs with the measured H2D copy cost and
reports, for each (pattern, size, reuse), which path is actually cheaper end
to end -- and the reuse count at which they break even.

Model for one produce-once/consume pass over an N-byte working set read R
times:
    T_device = copy(N) + N*R / BW_device_read
    T_zc     =           N*R / BW_zc_read
Break-even reuse:
    R* = copy(N) / ( N * (1/BW_zc - 1/BW_device) )
(if BW_zc >= BW_device the zero-copy path wins at every reuse)
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "orin-b" / "spatter_dense_grid.csv"
COPY = Path(__file__).resolve().parent / "data" / "h2d_copy_cost.csv"

GB = 1e9
MB = 1024 * 1024


def load_copy():
    out = {}
    with COPY.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["size_mb"])] = float(row["copy_ms_p50"]) / 1e3  # seconds
    return out


def main():
    copy_s = load_copy()
    rows = []
    with GRID.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mb = int(float(row["size_mb"]))
            if mb not in copy_s:
                continue
            rows.append({
                "pattern": row["pattern"],
                "size_mb": mb,
                "reuse": int(float(row["iters"])),
                "bw_dev": float(row["dev_solo_p50_gbps"]),
                "bw_zc": float(row["zc_solo_p50_gbps"]),
            })

    print(f"{'pattern':8} {'MB':>3} {'R':>3} {'BWdev':>7} {'BWzc':>7} "
          f"{'Tdev_ms':>8} {'Tzc_ms':>8} {'winner':>7} {'ZCgain%':>8} {'R*':>7}")
    print("-" * 78)
    summary = {}
    for r in rows:
        n_bytes = r["size_mb"] * MB
        logical = n_bytes * r["reuse"]
        t_dev = copy_s[r["size_mb"]] + logical / (r["bw_dev"] * GB)
        t_zc = logical / (r["bw_zc"] * GB)
        gain = (t_dev - t_zc) / t_dev * 100.0

        inv = 1.0 / r["bw_zc"] - 1.0 / r["bw_dev"]
        if inv <= 0:
            rstar = float("inf")
        else:
            rstar = copy_s[r["size_mb"]] / (n_bytes / GB * inv)

        winner = "ZC" if t_zc < t_dev else "device"
        summary.setdefault((r["pattern"], r["size_mb"]), rstar)
        print(f"{r['pattern']:8} {r['size_mb']:3d} {r['reuse']:3d} "
              f"{r['bw_dev']:7.1f} {r['bw_zc']:7.1f} "
              f"{t_dev*1e3:8.3f} {t_zc*1e3:8.3f} {winner:>7} {gain:8.1f} "
              f"{rstar:7.1f}")

    print()
    print("Break-even reuse R* per (pattern, size)  [ZC wins below R*]")
    for (pat, mb), rstar in sorted(summary.items()):
        print(f"  {pat:8} {mb:3d} MB -> R* = {rstar:.1f}")


if __name__ == "__main__":
    main()
