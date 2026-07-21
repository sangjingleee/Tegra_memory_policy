# Green Context + Spatter Memory-Path Benchmark

This package runs the controlled single-process Green Context microbenchmark used for
the Tegra memory-path study.

## What It Measures

- `device` path: `cuMemAlloc` buffers.
- `zero-copy` path: `cuMemHostAlloc(..., CU_MEMHOSTALLOC_DEVICEMAP)` mapped host buffers.
- Green Context split: two contexts using a platform-specific 50:50 SM split.
- Patterns: `uniform`, `strided`, `scatter`.
- Main strided setting: `stride=17`.
- Main sweep:
  - sizes: `1,2,4,8,16,32,64` MB
  - reuse: `1,2,4,8,16,32`
  - trials: `101`

The main metric is preservation:

```text
preservation = co-run aggregate throughput / solo-sum throughput
```

`solo-sum` is the sum of each path measured alone. `co-run aggregate` is the
common-interval throughput when device and mapped zero-copy kernels run
concurrently on disjoint Green Context SM partitions.

## Run On Each Jetson

```bash
git pull
cd experiments/green_spatter_gc
./scripts/run_full_sweep.sh
```

The script creates:

```text
results/<hostname>/platform.txt
results/<hostname>/clock_state.txt
results/<hostname>/spatter_dense_grid.csv
```

Then push only source files and compact CSV/text results:

```bash
git add experiments/green_spatter_gc
git commit -m "Add $(hostname) Green Context Spatter results"
git push
```

Do not commit raw Nsight reports or large logs.

## Optional cuBLAS Sanity Check

```bash
./scripts/run_cublas_gemm.sh
```

This is only an AI-primitive sanity check. The Spatter benchmark is the main
mechanism experiment because it controls access pattern, reuse, and working-set
size.

