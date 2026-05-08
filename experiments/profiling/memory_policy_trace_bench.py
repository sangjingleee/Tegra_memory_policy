#!/usr/bin/env python3
import argparse
import time

import torch

import orin_contention_worker as worker


def cuda_profiler_start():
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass


def cuda_profiler_stop():
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass


def run_model_policy(args):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    lib = worker.install_custom_allocator(args.policy, args.threshold_mb)
    model, input_obj = worker.build_model_for_name(args.model, args.image_size, args.seq_len)
    if lib is not None:
        lib.set_phase(1)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = worker.forward(model, input_obj, args.model)
    torch.cuda.synchronize()

    cuda_profiler_start()
    with torch.no_grad():
        start = time.time()
        for _ in range(args.repeats):
            _ = worker.forward(model, input_obj, args.model)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    cuda_profiler_stop()
    print(
        "TRACE_RESULT "
        f"kind=model policy={args.policy} model={args.model} "
        f"repeats={args.repeats} elapsed_ms={elapsed * 1000.0:.3f}",
        flush=True,
    )


def run_staging_policy(args):
    shape = (args.elements,)
    src_gpu = torch.randn(shape, device="cuda", dtype=torch.float32)
    dst_gpu = torch.empty_like(src_gpu)
    if args.policy == "pageable_staging":
        host = torch.empty(shape, dtype=torch.float32, pin_memory=False)
    elif args.policy == "pinned_staging":
        host = torch.empty(shape, dtype=torch.float32, pin_memory=True)
    else:
        raise ValueError(args.policy)

    for _ in range(args.warmup):
        host.copy_(src_gpu, non_blocking=False)
        dst_gpu.copy_(host, non_blocking=False)
    torch.cuda.synchronize()

    cuda_profiler_start()
    start = time.time()
    for _ in range(args.repeats):
        host.copy_(src_gpu, non_blocking=False)
        dst_gpu.copy_(host, non_blocking=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    cuda_profiler_stop()
    print(
        "TRACE_RESULT "
        f"kind=staging policy={args.policy} elements={args.elements} "
        f"repeats={args.repeats} elapsed_ms={elapsed * 1000.0:.3f}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        choices=[
            "default",
            "custom_device",
            "all_managed",
            "all_zc",
            "pageable_staging",
            "pinned_staging",
        ],
        required=True,
    )
    p.add_argument("--model", choices=["mobilenetv2", "gpt2"], default="mobilenetv2")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--elements", type=int, default=1_048_576)
    args = p.parse_args()

    if args.policy in ("pageable_staging", "pinned_staging"):
        run_staging_policy(args)
    else:
        run_model_policy(args)


if __name__ == "__main__":
    main()
