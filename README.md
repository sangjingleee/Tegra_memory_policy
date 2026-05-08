# Tegra Memory Policy Experiments

This repository snapshot contains experiments for selective memory placement on NVIDIA Tegra / Jetson systems.

The main research question is:

```text
Can we reduce model latency under contention by selectively placing input buffers
and intermediate activation tensors in device, managed, or mapped zero-copy memory?
```

## Current Scope

The project separates memory placement into two layers.

1. Input boundary
   - CPU/sensor/image/token data entering a GPU model.
   - Compares pageable copy, pinned copy, mapped zero-copy, managed memory, and preloaded device input.

2. Internal activation allocation
   - Intermediate tensors created during PyTorch model execution.
   - Uses `CUDAPluggableAllocator` to intercept PyTorch CUDA tensor allocation.
   - Compares PyTorch default, custom device baseline, managed, zero-copy, and mixed policies.

## Key Terms

```text
PyTorch default
  PyTorch CUDA caching allocator. This is the real user baseline.

custom_device
  Experimental control baseline. PyTorch allocation is routed through
  CUDAPluggableAllocator, but custom_malloc returns normal CUDA device memory.

managed / unified
  Implemented with cudaMallocManaged(..., cudaMemAttachGlobal).

zero-copy / mapped
  Implemented with cudaHostAlloc(..., cudaHostAllocMapped) and
  cudaHostGetDevicePointer(...).

mixed policy
  Uses tensor size threshold to place only selected tensors in managed or
  zero-copy memory, while keeping the rest in device memory.
```

## Repository Layout

```text
experiments/
  pytorch_allocator/
    custom_alloc_contention.cu      # custom CUDA allocator implementation
    orin_contention_worker.py       # PyTorch worker using CUDAPluggableAllocator
    policy_search_experiments.py    # broad policy sweep
    confirm_policy_candidates.py    # 5-trial confirmation

  input_boundary/
    actual_model_input_policy.py    # real model input boundary policies
    input_boundary_bench.cu         # synthetic input boundary benchmark
    sensor_boundary_bench.cu        # sensor-like boundary benchmark

  contention/
    run_model_model_matrix.py
    run_orin_contention_matrix.py
    run_paper_flow_experiments.py
    run_sensor_boundary_contention.py

  profiling/
    plot_*.py
    summarize_nsys_cuda_api.py
    cuda_memory_attrs.cu
    memory_policy_trace_bench.py

results/
  orin_a/
    CSV and SVG results from Orin-A.

  orin_nano/
    CSV and SVG results from Orin Nano.

docs/
  EXPERIMENT_INDEX.md
  SHARING_GUIDE.md
```

## Most Important Implementation Points

Unified memory for internal activation allocation is implemented in:

```text
experiments/pytorch_allocator/custom_alloc_contention.cu
```

Core call:

```cpp
cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
```

Mapped zero-copy for internal activation allocation is implemented with:

```cpp
cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
```

Input boundary managed tensors are implemented in:

```text
experiments/input_boundary/actual_model_input_policy.py
```

using a C++ extension that wraps `cudaMallocManaged` pointers with `torch::from_blob`.

## Current Confirmed Result

The strongest confirmed result so far:

```text
victim  = GPT2
attacker = MobileNetV2@640

Policy:
  keep GPT2 victim on default/device allocation
  move attacker activations >= 4MB to managed memory
```

5-trial confirmation:

```text
GPT2 seq512 victim:
  default attacker       66.371 ms
  mixed_managed attacker 54.464 ms

GPT2 seq1024 victim:
  default attacker       166.172 ms
  mixed_managed attacker 134.936 ms
```

Interpretation:

```text
This does not prove that all-managed is best.
It suggests that selected large activations from a moderate CNN attacker can be
moved to managed memory to reduce contention seen by a Transformer victim.
```

## Recommended GitHub Workflow

Use a private GitHub repository.

```text
main
  stable code and results that can be shown to the professor.

feature/<name>-<experiment>
  one branch per experiment.
```

Examples:

```text
feature/gpt2-mobilenet-contention
feature/input-boundary-policy
feature/nano-replication
feature/unified-allocator-cleanup
```

Do not commit server passwords, SSH keys, raw SSH config, or personal paths.
