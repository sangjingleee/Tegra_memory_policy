# Sharing Guide

Use this folder as the first private GitHub repository snapshot.

## Recommended GitHub Setup

1. Create a private repository, for example:

```text
tegra-memory-policy
```

2. Upload the contents of this folder.

3. Add collaborators.

4. Use branches for experiments.

```text
main
  stable code and confirmed results

feature/<person>-<experiment>
  active experiment work
```

## What To Share With A Collaborator

If a collaborator asks how unified memory was implemented, point them to:

```text
experiments/pytorch_allocator/custom_alloc_contention.cu
experiments/input_boundary/actual_model_input_policy.py
```

Short explanation:

```text
Unified memory was implemented with cudaMallocManaged.

For internal activations, PyTorch allocation is intercepted through
CUDAPluggableAllocator. The custom allocator returns cudaMallocManaged memory
for tensors above a threshold.

For input boundary tests, cudaMallocManaged pointers are wrapped as PyTorch CUDA
tensors using torch::from_blob.
```

## Files Not To Commit

Do not commit:

```text
SSH private keys
passwords
server IP notes if private
raw shell history
large model weights
TensorRT engine binaries
__pycache__
nsys report binaries unless explicitly needed
```

## Commit Message Style

Use simple messages:

```text
Add mixed managed allocator experiment
Add Orin-A contention confirmation results
Add input boundary benchmark
Update policy sweep plots
```

## Suggested First Issues

```text
1. Add PyTorch default baseline to all future allocator sweeps.
2. Verify custom_device overhead against PyTorch default.
3. Repeat GPT2 victim + CNN attacker with ResNet/MobileViT.
4. Add Nsight evidence for cudaMalloc/cudaMemcpy inside measured loop.
5. Clean zero-copy operator compatibility notes.
```
