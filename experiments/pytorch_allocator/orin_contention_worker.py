#!/usr/bin/env python3
import argparse
import csv
import ctypes
import os
import signal
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.environ.get(
    "ALLOC_SO_PATH", os.path.join(SCRIPT_DIR, "custom_alloc_contention.so")
)


def run_cpu_attacker(size_mb: int) -> None:
    import numpy as np

    n = size_mb * 1024 * 1024
    src = np.ones(n, dtype=np.uint8)
    dst = np.zeros(n, dtype=np.uint8)
    iters = 0
    start = time.time()
    while True:
        np.copyto(dst, src)
        np.copyto(src, dst)
        iters += 2
        if iters % 256 == 0:
            elapsed = max(time.time() - start, 1e-9)
            mb = iters * size_mb
            print(f"ATTACKER cpu_memcpy mbps={mb / elapsed:.1f}", flush=True)


def run_gpu_attacker(size: int) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    c = torch.empty((size, size), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    iters = 0
    start = time.time()
    while True:
        torch.matmul(a, b, out=c)
        iters += 1
        if iters % 100 == 0:
            torch.cuda.synchronize()
            elapsed = max(time.time() - start, 1e-9)
            print(f"ATTACKER gpu_gemm iters_per_s={iters / elapsed:.1f}", flush=True)


def run_model_attacker(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    lib = install_custom_allocator(args.attacker_policy, args.threshold_mb)
    model, input_obj = build_model_for_name(
        args.attacker_model, args.attacker_image_size, args.attacker_seq_len
    )
    if lib is not None:
        lib.set_phase(1)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = forward(model, input_obj, args.attacker_model)
    torch.cuda.synchronize()
    iters = 0
    start = time.time()
    with torch.no_grad():
        while True:
            _ = forward(model, input_obj, args.attacker_model)
            iters += 1
            if iters % 100 == 0:
                torch.cuda.synchronize()
                elapsed = max(time.time() - start, 1e-9)
                print(f"ATTACKER model={args.attacker_model} ips={iters / elapsed:.1f}", flush=True)


def start_attackers(args):
    procs = []
    base_cmd = [sys.executable, os.path.abspath(__file__)]
    if args.contention in ("cpu_memcpy", "cpu_gpu"):
        procs.append(
            subprocess.Popen(
                base_cmd + ["--attacker", "cpu", "--cpu-mb", str(args.cpu_mb)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    if args.contention in ("gpu_gemm", "cpu_gpu"):
        procs.append(
            subprocess.Popen(
                base_cmd + ["--attacker", "gpu", "--gemm-size", str(args.gemm_size)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    if args.contention == "model":
        procs.append(
            subprocess.Popen(
                base_cmd
                + [
                    "--attacker",
                    "model",
                    "--attacker-model",
                    args.attacker_model,
                    "--attacker-policy",
                    args.attacker_policy,
                    "--attacker-image-size",
                    str(args.attacker_image_size),
                    "--attacker-seq-len",
                    str(args.attacker_seq_len),
                    "--threshold-mb",
                    str(args.threshold_mb),
                    "--warmup",
                    str(args.warmup),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    if procs:
        time.sleep(args.attacker_warmup_s)
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f"attacker exited early: returncode={p.returncode}")
    return procs


def stop_attackers(procs) -> None:
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass


def install_custom_allocator(policy: str, threshold_mb: int):
    if policy == "default":
        return None

    import torch.cuda.memory as tmem

    if policy == "custom_device":
        strategy = 0
        threshold = threshold_mb * 1024 * 1024
    elif policy == "all_zc":
        strategy = 1
        threshold = 0
    elif policy == "all_managed":
        strategy = 4
        threshold = 0
    elif policy == "mixed_managed":
        strategy = 4
        threshold = threshold_mb * 1024 * 1024
    elif policy == "mixed_zc":
        strategy = 2
        threshold = threshold_mb * 1024 * 1024
    else:
        raise ValueError(f"unknown policy: {policy}")

    os.environ["ALLOC_STRATEGY"] = str(strategy)
    lib = ctypes.CDLL(SO_PATH)
    lib.set_phase.argtypes = [ctypes.c_int]
    lib.set_phase.restype = None
    lib.set_threshold.argtypes = [ctypes.c_size_t]
    lib.set_threshold.restype = None
    lib.set_threshold(threshold)
    lib.set_phase(0)

    alloc = tmem.CUDAPluggableAllocator(SO_PATH, "custom_malloc", "custom_free")
    tmem.change_current_allocator(alloc)
    return lib


def build_model_for_name(model_name: str, image_size: int, seq_len: int):
    import torch
    import torch.nn as nn

    if model_name == "mobilenetv2":
        import torchvision.models as models

        model = models.mobilenet_v2(weights=None)
        input_obj = torch.zeros((1, 3, image_size, image_size), device="cuda")
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU)):
                m.inplace = False
        return model.cuda().eval(), input_obj

    if model_name == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=50257,
            n_positions=max(1024, seq_len),
            n_embd=768,
            n_layer=12,
            n_head=12,
            use_cache=True,
        )
        model = GPT2LMHeadModel(config)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long, device="cuda")
        return model.cuda().eval(), input_ids

    raise ValueError(f"unknown model: {model_name}")


def build_model(args):
    return build_model_for_name(args.model, args.image_size, args.seq_len)


def forward(model, input_obj, model_name: str):
    if model_name == "gpt2":
        return model(input_ids=input_obj, use_cache=True)
    return model(input_obj)


def run_victim(args) -> int:
    procs = start_attackers(args)
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(0)

        lib = install_custom_allocator(args.policy, args.threshold_mb)
        model, input_obj = build_model(args)
        if lib is not None:
            lib.set_phase(1)

        with torch.no_grad():
            for _ in range(args.warmup):
                _ = forward(model, input_obj, args.model)
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_evt.record()
            for _ in range(args.repeats):
                _ = forward(model, input_obj, args.model)
            end_evt.record()
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt) / args.repeats

        try:
            allocated = f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}"
            reserved = f"{torch.cuda.max_memory_reserved() / 1024**2:.1f}"
        except RuntimeError:
            allocated = "NA"
            reserved = "NA"
        row = {
            "model": args.model,
            "image_size": args.image_size if args.model == "mobilenetv2" else "",
            "seq_len": args.seq_len if args.model == "gpt2" else "",
            "policy": args.policy,
            "contention": args.contention,
            "attacker_model": args.attacker_model if args.contention == "model" else "",
            "attacker_policy": args.attacker_policy if args.contention == "model" else "",
            "threshold_mb": args.threshold_mb,
            "latency_ms": f"{latency_ms:.4f}",
            "max_allocated_mb": allocated,
            "max_reserved_mb": reserved,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }
        print("RESULT " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        if args.alloc_log and lib is not None:
            lib.dump_alloc_summary.argtypes = [ctypes.c_char_p]
            lib.dump_alloc_summary.restype = None
            lib.dump_alloc_summary(args.alloc_log.encode("utf-8"))

        if args.csv:
            exists = os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        return 0
    finally:
        stop_attackers(procs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attacker", choices=["cpu", "gpu", "model"])
    p.add_argument("--cpu-mb", type=int, default=256)
    p.add_argument("--gemm-size", type=int, default=1536)
    p.add_argument("--attacker-model", choices=["mobilenetv2", "gpt2"], default="mobilenetv2")
    p.add_argument(
        "--attacker-policy",
        choices=["default", "custom_device", "all_managed", "mixed_managed", "mixed_zc", "all_zc"],
        default="default",
    )
    p.add_argument("--attacker-image-size", type=int, default=224)
    p.add_argument("--attacker-seq-len", type=int, default=128)
    p.add_argument("--model", choices=["mobilenetv2", "gpt2"], default="mobilenetv2")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument(
        "--policy",
        choices=["default", "custom_device", "all_managed", "mixed_managed", "mixed_zc", "all_zc"],
        default="default",
    )
    p.add_argument(
        "--contention",
        choices=["none", "cpu_memcpy", "gpu_gemm", "cpu_gpu", "model"],
        default="none",
    )
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--attacker-warmup-s", type=float, default=2.0)
    p.add_argument("--csv")
    p.add_argument("--alloc-log")
    return p.parse_args()


def main():
    args = parse_args()
    if args.attacker == "cpu":
        run_cpu_attacker(args.cpu_mb)
        return 0
    if args.attacker == "gpu":
        run_gpu_attacker(args.gemm_size)
        return 0
    if args.attacker == "model":
        run_model_attacker(args)
        return 0
    return run_victim(args)


if __name__ == "__main__":
    raise SystemExit(main())
