# Verification notes for professor questions (2026-06-28)

## 1. What does the DRAM number mean?

`dram_peak.cu` measures a direct streaming read/copy baseline on a 512 MB device
buffer.

- `stream READ peak = bytes / kernel_time`
- `bytes = 512 MB`
- kernel reads each `float4` element once
- result on Orin-B with clocks locked: `177.1 GB/s`

This number is an actual DRAM-oriented streaming read baseline for this board.

The Spatter/Green Context result is different:

- `bytes_read = elements * 4B * reuse_iters * batches`
- this is logical bytes requested by the gather kernel
- if data is served from cache, this effective GB/s can exceed the DRAM stream
  baseline

Therefore:

- `dram_peak` = physical DRAM streaming anchor
- `green_spatter_split` GB/s = effective logical read throughput

Do not present Spatter GB/s as direct DRAM bytes/s.

## 2. Are the thread/kernel launch conditions the same?

Yes for the compared solo/co-run path measurements.

In `green_spatter_split.cpp`:

- block size is fixed: `block = 256`
- grid is fixed by working-set size: `grid = (m + block - 1) / block`
- both device and zero-copy launch the same `gather_reuse` kernel
- both use the same `m`, `iters`, `grid`, and `block`
- Green Context changes the SM partition, not the number of logical CUDA threads

The code records actual SM counts from CUDA:

- `cuGreenCtxGetDevResource(gc0, ..., CU_DEV_RESOURCE_TYPE_SM)`
- `cuGreenCtxGetDevResource(gc1, ..., CU_DEV_RESOURCE_TYPE_SM)`

The clock-locked campaign CSV reports `dev_sms=8`, `zc_sms=8` for all main rows.

Important nuance:

- solo measurements are also run inside the corresponding 8-SM Green Context
- co-run measurements use the same 8-SM Green Contexts concurrently
- some runs use different `dev_batches`/`zc_batches` to align runtime overlap, but
  this changes how many repeated launches are included in the timing window, not
  the kernel's thread geometry

## 3. Does Green Context work only across different processes?

No. The CUDA Driver API supports creating and using green contexts inside one
application/process.

Our main experiment is a single-process setup:

1. create two Green Contexts: `gc0`, `gc1`
2. query actual SM resources for both
3. convert each green context to a CUDA context with `cuCtxFromGreenCtx`
4. create a stream for each green context with `cuGreenCtxStreamCreate`
5. launch the two streams concurrently

Smoke verification on Orin-B:

```text
./green_spatter_split --dev-sms=8 --zc-sms=8 --mb=1 --iters=32 \
  --batches=4 --trials=7 --pat=uniform

reported: dev_sms=8, zc_sms=8, overlap_p50_ratio=0.993976
```

Nsight Systems verification:

```text
results/gc_same_process_verify.nsys-rep
```

This report was generated from a single-process run of `green_spatter_split`.

## 4. Safe interpretation

What is directly verified:

- clocks were pinned (`jetson_clocks`)
- device and zero-copy paths use the same gather kernel and launch geometry
- actual Green Context SM counts are 8 and 8
- the two Green Context kernels overlap in time
- 512 MB streaming read baseline is about `177.1 GB/s`
- Spatter effective throughput is logical read throughput, not direct DRAM bytes/s

What should remain hedged:

- SLC residency is not directly proven by a hardware SLC counter
- L2/SLC physical independence is inferred from performance, not directly profiled
- `~100% aggregate / solo_sum` means co-run preserves the effective throughput
  under that access pattern; it should not be worded as a direct SLC-counter proof

