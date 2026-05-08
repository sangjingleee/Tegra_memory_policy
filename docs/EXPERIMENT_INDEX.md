# Experiment Index

This page lists which files matter and what each experiment answers.

## 1. PyTorch Allocator Experiments

Files:

```text
experiments/pytorch_allocator/custom_alloc_contention.cu
experiments/pytorch_allocator/orin_contention_worker.py
experiments/pytorch_allocator/policy_search_experiments.py
experiments/pytorch_allocator/confirm_policy_candidates.py
```

Question:

```text
If PyTorch CUDA tensor allocation is intercepted, which internal activation
memory placement policy works best?
```

Important baselines:

```text
PyTorch default
  Real PyTorch CUDA caching allocator.

custom_device
  CUDAPluggableAllocator path, but returns cudaMalloc/device memory.
  This controls for custom allocator overhead.
```

Important policies:

```text
all_managed
  All selected allocations use cudaMallocManaged.

all_zc
  All selected allocations use mapped zero-copy.

mixed_managed
  Only tensors above threshold use cudaMallocManaged.

mixed_zc
  Only tensors above threshold use mapped zero-copy.
```

Current conclusion:

```text
Do not claim that all-managed or all-zero-copy is best.
The current evidence supports selective placement based on model role and tensor size.
```

## 2. Input Boundary Experiments

Files:

```text
experiments/input_boundary/actual_model_input_policy.py
experiments/input_boundary/input_boundary_bench.cu
experiments/input_boundary/sensor_boundary_bench.cu
```

Question:

```text
For CPU/sensor/image/token input entering a GPU model, which memory path avoids
unnecessary copy overhead?
```

Compared modes:

```text
pageable input
  malloc host + cudaMemcpy H2D.

pinned input
  cudaHostAlloc host + cudaMemcpy H2D.

mapped zero-copy input
  cudaHostAllocMapped + cudaHostGetDevicePointer.

managed input
  cudaMallocManaged.

device preloaded input
  cudaMalloc device memory already prepared before timing.
```

Current conclusion:

```text
Large image/camera-like inputs are where input boundary memory policy matters.
Tiny token inputs are too small for input placement to dominate latency.
```

## 3. Model-Model Contention Experiments

Files:

```text
experiments/contention/run_model_model_matrix.py
experiments/contention/run_orin_contention_matrix.py
experiments/contention/run_paper_flow_experiments.py
```

Question:

```text
If two models run simultaneously, can changing the attacker's activation memory
placement reduce the victim model's latency?
```

Strongest confirmed case:

```text
victim   = GPT2
attacker = MobileNetV2@640
policy   = attacker mixed_managed threshold=4MB
```

Result:

```text
GPT2 seq512:
  66.371 ms -> 54.464 ms

GPT2 seq1024:
  166.172 ms -> 134.936 ms
```

Interpretation:

```text
The policy should consider model role.
Changing the victim is not always the right knob.
For this case, changing the moderate CNN attacker reduced Transformer victim latency.
```

## 4. Profiling / Evidence Files

Files:

```text
experiments/profiling/summarize_nsys_cuda_api.py
experiments/profiling/cuda_memory_attrs.cu
experiments/profiling/memory_policy_trace_bench.py
```

Purpose:

```text
Verify which CUDA APIs are actually called and whether allocation/copy occurs
inside the measured loop.
```

Key evidence needed for future runs:

```text
1. Which CUDA API is used?
2. Does cudaMalloc/cudaFree occur during inference loop?
3. Does cudaMemcpy occur during inference loop?
4. What are the pointer attributes for device, managed, and mapped memory?
```

## 5. Results To Look At First

Orin-A:

```text
results/orin_a/policy_confirm_internal_summary.csv
results/orin_a/policy_confirm_contention_summary.csv
results/orin_a/policy_search_contention_winners.csv
results/orin_a/orin_5x5_gpt2victim_mobilenetattacker_summary.csv
```

Orin Nano:

```text
results/orin_nano/nano_model_model_full_5trial_summary.csv
results/orin_nano/reverse_nano_full_5trial_summary.csv
```

SVG heatmaps:

```text
results/orin_a/*heatmap.svg
results/orin_nano/*heatmap.svg
```
