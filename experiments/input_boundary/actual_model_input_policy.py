#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, "orin_contention_worker.py")


EXT_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor make_mapped_float(std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t v : shape) n *= v;
    void* host_ptr = nullptr;
    void* dev_ptr = nullptr;
    check(cudaHostAlloc(&host_ptr, n * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAllocMapped float");
    check(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0), "cudaHostGetDevicePointer float");
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return torch::from_blob(dev_ptr, shape, [host_ptr](void*) { cudaFreeHost(host_ptr); }, opts);
}

torch::Tensor make_managed_float(std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t v : shape) n *= v;
    void* ptr = nullptr;
    check(cudaMallocManaged(&ptr, n * sizeof(float), cudaMemAttachGlobal), "cudaMallocManaged float");
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return torch::from_blob(ptr, shape, [](void* p) { cudaFree(p); }, opts);
}

torch::Tensor make_mapped_long(std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t v : shape) n *= v;
    void* host_ptr = nullptr;
    void* dev_ptr = nullptr;
    check(cudaHostAlloc(&host_ptr, n * sizeof(int64_t), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAllocMapped long");
    check(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0), "cudaHostGetDevicePointer long");
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    return torch::from_blob(dev_ptr, shape, [host_ptr](void*) { cudaFreeHost(host_ptr); }, opts);
}

torch::Tensor make_managed_long(std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t v : shape) n *= v;
    void* ptr = nullptr;
    check(cudaMallocManaged(&ptr, n * sizeof(int64_t), cudaMemAttachGlobal), "cudaMallocManaged long");
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    return torch::from_blob(ptr, shape, [](void* p) { cudaFree(p); }, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_mapped_float", &make_mapped_float);
    m.def("make_managed_float", &make_managed_float);
    m.def("make_mapped_long", &make_mapped_long);
    m.def("make_managed_long", &make_managed_long);
}
"""


def load_ext():
    from torch.utils.cpp_extension import load

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda-11.4"
    src_dir = os.path.join(tempfile.gettempdir(), "actual_input_ext")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "actual_input_ext.cpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(EXT_SRC)
    return load(
        name="actual_input_ext",
        sources=[src],
        extra_include_paths=[os.path.join(cuda_home, "include")],
        extra_cflags=["-O3"],
        extra_ldflags=[f"-L{os.path.join(cuda_home, 'lib64')}", "-lcudart"],
        verbose=False,
    )


def start_attacker(args):
    if args.attacker_model == "none":
        return None
    cmd = [
        sys.executable,
        WORKER,
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
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(args.attacker_warmup_s)
    if proc.poll() is not None:
        raise RuntimeError(f"attacker exited early: rc={proc.returncode}")
    return proc


def stop_proc(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def build_model(args):
    import torch
    import torch.nn as nn

    if args.victim_model == "mobilenetv2":
        import torchvision.models as models

        model = models.mobilenet_v2(weights=None)
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU)):
                m.inplace = False
        return model.cuda().eval()

    if args.victim_model == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=50257,
            n_positions=max(1024, args.victim_seq_len),
            n_embd=768,
            n_layer=12,
            n_head=12,
            use_cache=True,
        )
        return GPT2LMHeadModel(config).cuda().eval()

    raise ValueError(args.victim_model)


def make_inputs(args, ext):
    import torch

    if args.victim_model == "mobilenetv2":
        shape = (1, 3, args.victim_image_size, args.victim_image_size)
        if args.input_mode == "pageable":
            host = torch.zeros(shape, dtype=torch.float32, pin_memory=False)
            dev = torch.empty(shape, dtype=torch.float32, device="cuda")
            return host, dev
        if args.input_mode == "pinned":
            host = torch.zeros(shape, dtype=torch.float32, pin_memory=True)
            dev = torch.empty(shape, dtype=torch.float32, device="cuda")
            return host, dev
        if args.input_mode == "mapped_zc":
            dev = ext.make_mapped_float(list(shape))
            dev.zero_()
            torch.cuda.synchronize()
            return None, dev
        if args.input_mode == "managed":
            dev = ext.make_managed_float(list(shape))
            dev.zero_()
            torch.cuda.synchronize()
            return None, dev
        if args.input_mode == "device_preloaded":
            dev = torch.zeros(shape, dtype=torch.float32, device="cuda")
            return None, dev

    if args.victim_model == "gpt2":
        shape = (1, args.victim_seq_len)
        if args.input_mode == "pageable":
            host = torch.zeros(shape, dtype=torch.long, pin_memory=False)
            dev = torch.empty(shape, dtype=torch.long, device="cuda")
            return host, dev
        if args.input_mode == "pinned":
            host = torch.zeros(shape, dtype=torch.long, pin_memory=True)
            dev = torch.empty(shape, dtype=torch.long, device="cuda")
            return host, dev
        if args.input_mode == "mapped_zc":
            dev = ext.make_mapped_long(list(shape))
            dev.zero_()
            torch.cuda.synchronize()
            return None, dev
        if args.input_mode == "managed":
            dev = ext.make_managed_long(list(shape))
            dev.zero_()
            torch.cuda.synchronize()
            return None, dev
        if args.input_mode == "device_preloaded":
            dev = torch.zeros(shape, dtype=torch.long, device="cuda")
            return None, dev

    raise ValueError(args.input_mode)


def forward(args, model, x):
    if args.victim_model == "gpt2":
        return model(input_ids=x, use_cache=True)
    return model(x)


def run(args):
    import torch

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    ext = load_ext()
    model = build_model(args)
    host, dev = make_inputs(args, ext)
    proc = start_attacker(args)
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                if host is not None:
                    dev.copy_(host, non_blocking=(args.input_mode == "pinned"))
                _ = forward(args, model, dev)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            for _ in range(args.repeats):
                if host is not None:
                    dev.copy_(host, non_blocking=(args.input_mode == "pinned"))
                _ = forward(args, model, dev)
            end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / args.repeats
        row = {
            "victim_model": args.victim_model,
            "victim_image_size": args.victim_image_size if args.victim_model == "mobilenetv2" else "",
            "victim_seq_len": args.victim_seq_len if args.victim_model == "gpt2" else "",
            "input_mode": args.input_mode,
            "attacker_model": args.attacker_model,
            "attacker_policy": args.attacker_policy if args.attacker_model != "none" else "",
            "latency_ms": f"{latency:.4f}",
            "repeats": args.repeats,
            "warmup": args.warmup,
        }
        print("RESULT " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        if args.csv:
            exists = os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(row)
    finally:
        stop_proc(proc)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--victim-model", choices=["mobilenetv2", "gpt2"], required=True)
    p.add_argument("--victim-image-size", type=int, default=640)
    p.add_argument("--victim-seq-len", type=int, default=512)
    p.add_argument(
        "--input-mode",
        choices=["pageable", "pinned", "mapped_zc", "managed", "device_preloaded"],
        required=True,
    )
    p.add_argument("--attacker-model", choices=["none", "mobilenetv2", "gpt2"], default="none")
    p.add_argument("--attacker-policy", choices=["default", "custom_device", "all_managed", "mixed_managed", "mixed_zc", "all_zc"], default="default")
    p.add_argument("--attacker-image-size", type=int, default=640)
    p.add_argument("--attacker-seq-len", type=int, default=512)
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--attacker-warmup-s", type=float, default=3.0)
    p.add_argument("--csv")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
