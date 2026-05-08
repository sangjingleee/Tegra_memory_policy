#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time


POLICIES = ["default", "custom_device", "all_managed", "mixed_managed"]
CONTENTION = ["none", "cpu_memcpy", "gpu_gemm", "cpu_gpu"]


def parse_result(stdout: str):
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        row = {}
        for part in line[len("RESULT ") :].split():
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            row[k] = v
        return row
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home2/sangjin/codex/orin_contention_results.csv")
    p.add_argument("--worker", default="/home2/sangjin/codex/orin_contention_worker.py")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mobilenet-size", type=int, default=224)
    p.add_argument("--gpt2-seq-len", type=int, default=128)
    args = p.parse_args()

    models = [
        ("mobilenetv2", {"--image-size": str(args.mobilenet_size)}),
        ("gpt2", {"--seq-len": str(args.gpt2_seq_len)}),
    ]
    if args.quick:
        models = [("mobilenetv2", {"--image-size": "224"})]
        policies = ["default", "custom_device"]
        contentions = ["none", "cpu_memcpy"]
    else:
        policies = POLICIES
        contentions = CONTENTION

    if os.path.exists(args.out):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.rename(args.out, args.out + f".bak_{stamp}")

    rows = []
    total = len(models) * len(policies) * len(contentions)
    idx = 0
    for model, model_args in models:
        for policy in policies:
            for contention in contentions:
                idx += 1
                cmd = [
                    sys.executable,
                    args.worker,
                    "--model",
                    model,
                    "--policy",
                    policy,
                    "--contention",
                    contention,
                    "--threshold-mb",
                    str(args.threshold_mb),
                    "--repeats",
                    str(args.repeats),
                    "--warmup",
                    str(args.warmup),
                    "--csv",
                    args.out,
                ]
                for k, v in model_args.items():
                    cmd += [k, v]
                print(f"[{idx}/{total}] {' '.join(cmd)}", flush=True)
                proc = subprocess.run(cmd, text=True, capture_output=True)
                if proc.stdout:
                    print(proc.stdout, end="")
                if proc.stderr:
                    print(proc.stderr, end="", file=sys.stderr)
                if proc.returncode != 0:
                    print(f"FAILED returncode={proc.returncode}", file=sys.stderr)
                    continue
                row = parse_result(proc.stdout)
                if row:
                    rows.append(row)

    print(f"\nSaved CSV: {args.out}")
    if not rows and os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f))
    if rows:
        print("\nSummary:")
        for r in rows:
            target = r["image_size"] or ("seq" + r["seq_len"])
            print(
                f"{r['model']:12s} {target:>5s} {r['policy']:14s} "
                f"{r['contention']:10s} {float(r['latency_ms']):8.3f} ms"
            )


if __name__ == "__main__":
    main()
