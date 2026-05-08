#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import subprocess
import sys
import time


def parse_result(line):
    if not line.startswith("RESULT "):
        return None
    row = {}
    for part in line[len("RESULT ") :].split():
        if "=" in part:
            k, v = part.split("=", 1)
            row[k] = v
    return row


def start_attacker(args, mode):
    if mode == "none":
        return None
    base = [sys.executable, args.worker]
    if mode == "cpu_memcpy":
        cmd = base + ["--attacker", "cpu", "--cpu-mb", str(args.cpu_mb)]
    elif mode == "gpu_gemm":
        cmd = base + ["--attacker", "gpu", "--gemm-size", str(args.gemm_size)]
    elif mode.startswith("model_"):
        parts = mode.split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"model contention must be model_<model>_<policy>, got {mode}")
        _, model, policy = parts
        image_size = args.mobilenet_image_size if model == "mobilenetv2" else args.gpt2_image_size
        seq_len = args.gpt2_seq_len if model == "gpt2" else args.mobilenet_seq_len
        cmd = base + [
            "--attacker",
            "model",
            "--attacker-model",
            model,
            "--attacker-policy",
            policy,
            "--attacker-image-size",
            str(image_size),
            "--attacker-seq-len",
            str(seq_len),
            "--threshold-mb",
            str(args.threshold_mb),
            "--warmup",
            str(args.attacker_warmup),
        ]
    else:
        raise ValueError(mode)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(args.attacker_warmup_s)
    if proc.poll() is not None:
        raise RuntimeError(f"attacker exited early: {mode} rc={proc.returncode}")
    return proc


def stop_attacker(proc):
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


def run_one(args, size_bytes, contention):
    proc = start_attacker(args, contention)
    try:
        cmd = [args.bench, str(size_bytes), str(args.repeats)]
        out = subprocess.check_output(cmd, text=True)
    finally:
        stop_attacker(proc)
    rows = []
    for line in out.splitlines():
        print(f"[{contention} {size_bytes}] {line}", flush=True)
        row = parse_result(line)
        if row:
            row["contention"] = contention
            row["size_mb"] = f"{size_bytes / 1024**2:.1f}"
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="./sensor_boundary_bench")
    p.add_argument("--worker", default="./orin_contention_worker.py")
    p.add_argument("--out", default="sensor_boundary_contention_orin.csv")
    p.add_argument("--sizes-mb", nargs="+", type=float, default=[4, 16])
    p.add_argument("--contentions", nargs="+", default=["none", "cpu_memcpy", "gpu_gemm"])
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--cpu-mb", type=int, default=256)
    p.add_argument("--gemm-size", type=int, default=1536)
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--attacker-warmup", type=int, default=20)
    p.add_argument("--attacker-warmup-s", type=float, default=2.0)
    p.add_argument("--mobilenet-image-size", type=int, default=640)
    p.add_argument("--mobilenet-seq-len", type=int, default=128)
    p.add_argument("--gpt2-image-size", type=int, default=224)
    p.add_argument("--gpt2-seq-len", type=int, default=512)
    args = p.parse_args()

    rows = []
    for mb in args.sizes_mb:
        size_bytes = int(mb * 1024 * 1024)
        for contention in args.contentions:
            rows.extend(run_one(args, size_bytes, contention))

    fieldnames = [
        "contention",
        "size_mb",
        "mode",
        "bytes",
        "repeats",
        "h2d_ms",
        "kernel_ms",
        "d2h_ms",
        "total_ms",
        "memcpy_count_per_iter",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
