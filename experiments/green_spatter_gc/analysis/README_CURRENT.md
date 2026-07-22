# Current Orin-B experiment set (2026-07-20)

Server directory: `/home/sangjin/codex/green_ctx_0522`

## 1. Main controlled Spatter + Green Context experiment

- `green_spatter_split.cpp`: single process, two Green Contexts, device and mapped zero-copy paths.
- `spatter_gather.cu`: common Spatter-style gather kernel for uniform, strided, and scatter patterns.
- `run_clocked_spatter_campaign.sh`: clock-locked campaign runner.
- `summarize_clocked_spatter_campaign.py`: campaign summarizer.
- `dram_peak.cu`: large streaming-read calibration.
- `results/clocked_spatter_20260626/`: primary clock-locked raw and summarized results.
- `results_dense_grid.csv`: dense size/reuse/pattern matrix.
- `results_heatmap_grid.csv`, `dense_efficiency.csv`: heatmap-ready summaries.
- `spatter_split_lowreuse_sweep.csv`: low-reuse split-ratio sweep.

Primary metric: effective logical read throughput. It is not physical DRAM bandwidth.

## 2. Green Context validation and multi-process experiment

- `green_smid_check.cpp` + `smid_probe.cu`: verifies disjoint physical SM use for two Green Contexts in one process.
- `results/gc_same_process_verify.nsys-rep`: existing same-process Nsight Systems trace.
- `gc_probe_one.cpp`: creates one N-SM Green Context in one process and reports physical SM IDs used.
- `spatter_one.cpp`: runs one device or mapped zero-copy path in one process, optionally inside one Green Context.

Use `gc_probe_one` and `spatter_one` as two concurrent OS processes to test whether the current single-process result generalizes to multi-process execution.

## 3. Cache and memory-system diagnostics

- `l2_micro.cpp`, `query_l2*.cu`: isolation-level L2 probes.
- `results_zc_isolation*.csv`, `results_l2_*.csv`: isolation results.
- `results/emc_pressure_smpart_0526.csv`: EMC diagnostic summary.
- `results/scf_pmu_smpart_0526.csv`: SCF PMU attempt; not suitable as direct GPU SLC proof.

## Terminology

The main microbenchmark compares:

- device allocation path: `cuMemAlloc`
- mapped zero-copy path: `cuMemHostAlloc(...DEVICEMAP)` + `cuMemHostGetDevicePointer`

It does not directly test a pageable host-staging path (`malloc` + `cudaMemcpy`).

## Recovery

The complete pre-cleanup directory is backed up locally under:

`experiments/green_ctx_0601/remote_archive_20260720/green_ctx_0522`
