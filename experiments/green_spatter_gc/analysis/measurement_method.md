# Bandwidth-aggregation measurement — how it works

Platform: Jetson **AGX Orin** (64GB, LPDDR5, DRAM peak ≈ **204.8 GB/s**), 16 SM,
GPU L2 = 4 MB. Tool: `green_spatter_split.cpp`.

## 1. Two memory paths, one kernel
The same gather kernel (`gather_reuse`, `spatter_gather.ptx`) is run over two buffers:
- **device path** — `cuMemAlloc` (GPU device memory → served by GPU **L2** when resident).
- **zero-copy path** — `cuMemHostAlloc(..., DEVICEMAP)` + `cuMemHostGetDevicePointer`
  (host-mapped memory the GPU reads directly, no copy → served via **SLC**/DRAM).

Kernel body (per thread): `acc += in[ idx[(k+r) & (m-1)] ]`, looped `iters` times over
`m` elements. So **total reads = m × iters** elements.

Access pattern is set by the `idx` array:
`uniform` (idx[i]=i), `strided` (idx=i·stride mod m), `scatter` (hashed/random).
`iters` = **reuse** (high iters → data re-read many times → cache-resident).

## 2. Effective bandwidth (the metric)
```
bytes_read = m × 4B × iters
time       = CUDA-event elapsed time bracketing the kernel on its own stream
bandwidth  = bytes_read / time          (GB/s)
```
This is *achieved* throughput (bytes moved ÷ wall time), not a HW counter.

## 3. Solo vs co-run
- **solo**: each path measured alone → `dev_solo`, `zc_solo`.
- **co**: both kernels launched **concurrently on two Green Contexts** that own
  **disjoint SM partitions** (8 SM each). Because the SMs are disjoint, the *only*
  shared resource is the **memory hierarchy** (L2 / SLC / DRAM). Each kernel is timed
  by its own event pair → `dev_co`, `zc_co`.
- ZC reuse is scaled (`zc_it = IT·zc_solo/dev_solo`) so both kernels take ~equal time
  and actually overlap during the co-run window.

## 4. What we report
```
aggregate      = dev_co + zc_co
solo_sum       = dev_solo + zc_solo
aggregate_ratio= aggregate / solo_sum × 100%      # ~100% ⇒ paths independent
device_preserve= dev_co / dev_solo × 100%         # how much the critical path keeps
zc_preserve    = zc_co  / zc_solo  × 100%
```
- ratio ≈ 100% **and** aggregate **> DRAM peak (204.8 GB/s)** ⇒ the two paths are
  served by *different* bandwidth resources (L2 + SLC), not one shared DRAM channel.
- Sanity: combined working set **16 MB > L2 4 MB** still aggregates at 98.8% ⇒ not a
  "both fit in one L2" artifact.

## 5. Robustness fixes (important)
- **Locked clocks — REQUIRED.** `sudo jetson_clocks` pins GPU=1300 MHz, EMC=3199 MHz.
  Without it the GPU idles at 306 MHz / EMC 665 MHz and DVFS ramps the clock up under
  the *heavier* co-run, so `dev_co > dev_solo` (ratio > 100%, even 240%) — a pure clock
  artifact. With clocks locked this disappears and results are reproducible.
- **MEDIAN of 31 trials** (was: max/best, which is optimistically biased), + 3 warmup
  iterations discarded.

## 6. Headline result (locked + median, `results_zc_isolation_locked.csv`)
| pattern (reuse=32) | 1MB | 4MB | 16MB | 64MB |
|---|---|---|---|---|
| uniform aggregate (GB/s) | 284 | 294 | 298 | 298 |
| uniform ratio | 96.5% | 98.8% | 98.8% | 98.4% |

All uniform-high-reuse aggregates (284–298 GB/s) exceed DRAM peak (204.8). strided
high-reuse aggregates ~94% (device/critical preserved 96–99%, ZC absorbs the loss).
scatter / low-reuse / large-streaming degrade (down to ~12%) — they fall through to
shared DRAM.

## 7. Still open (to make it air-tight for ESL)
- NCU L2 + SLC counters → turn the ">DRAM peak" inference into a **direct** SLC measurement.
- A dedicated peak-DRAM-BW micro-baseline on this exact board (vs the theoretical 204.8).
