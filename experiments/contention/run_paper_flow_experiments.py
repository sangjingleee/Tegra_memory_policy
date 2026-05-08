#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("RUN", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(map(str, cmd))}")
    return proc.stdout


def append_row(path, row, fieldnames):
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def parse_result(stdout):
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        row = {}
        for part in line[len("RESULT ") :].split():
            if "=" in part:
                k, v = part.split("=", 1)
                row[k] = v
        return row
    return {}


def actual_input(args):
    out = ROOT / "paper_flow_input_sweep.csv"
    if out.exists():
        out.rename(out.with_suffix(".csv.bak_" + time.strftime("%Y%m%d_%H%M%S")))
    modes = ["pageable", "pinned", "mapped_zc", "managed", "device_preloaded"]
    fieldnames = [
        "experiment",
        "trial",
        "victim_model",
        "victim_image_size",
        "victim_seq_len",
        "input_mode",
        "attacker_model",
        "attacker_policy",
        "latency_ms",
        "repeats",
        "warmup",
    ]
    for trial in range(1, args.trials + 1):
        for image_size in [224, 640, 1024]:
            for attacker in ["none", "gpt2"]:
                for mode in modes:
                    cmd = [
                        sys.executable,
                        str(ROOT / "actual_model_input_policy.py"),
                        "--victim-model",
                        "mobilenetv2",
                        "--victim-image-size",
                        str(image_size),
                        "--input-mode",
                        mode,
                        "--attacker-model",
                        attacker,
                        "--repeats",
                        str(args.input_repeats),
                        "--warmup",
                        str(args.warmup),
                    ]
                    if attacker == "gpt2":
                        cmd += [
                            "--attacker-policy",
                            "default",
                            "--attacker-seq-len",
                            "512",
                            "--attacker-warmup-s",
                            str(args.attacker_warmup_s),
                        ]
                    row = parse_result(run(cmd))
                    row["experiment"] = "input_boundary_sweep"
                    row["trial"] = trial
                    append_row(out, row, fieldnames)
    print(out)


def internal_policy(args):
    out = ROOT / "paper_flow_internal_policy_sweep.csv"
    if out.exists():
        out.rename(out.with_suffix(".csv.bak_" + time.strftime("%Y%m%d_%H%M%S")))
    policies = ["default", "custom_device", "mixed_managed", "mixed_zc", "all_managed", "all_zc"]
    fieldnames = [
        "experiment",
        "trial",
        "model",
        "image_size",
        "seq_len",
        "policy",
        "contention",
        "attacker_model",
        "attacker_policy",
        "threshold_mb",
        "latency_ms",
        "max_allocated_mb",
        "max_reserved_mb",
        "repeats",
        "warmup",
    ]
    cases = []
    for image_size in [224, 640, 1024]:
        for policy in policies:
            cases.append(("mobilenetv2", image_size, 128, policy))
    for seq_len in [128, 512, 1024]:
        for policy in policies:
            cases.append(("gpt2", 224, seq_len, policy))
    for trial in range(1, args.trials + 1):
        for model, image_size, seq_len, policy in cases:
            cmd = [
                sys.executable,
                str(ROOT / "orin_contention_worker.py"),
                "--model",
                model,
                "--image-size",
                str(image_size),
                "--seq-len",
                str(seq_len),
                "--policy",
                policy,
                "--contention",
                "none",
                "--threshold-mb",
                "4",
                "--repeats",
                str(args.internal_repeats),
                "--warmup",
                str(args.warmup),
            ]
            row = parse_result(run(cmd))
            row["experiment"] = "internal_policy_sweep"
            row["trial"] = trial
            append_row(out, row, fieldnames)
    print(out)


def contention(args):
    out = ROOT / "paper_flow_contention_core.csv"
    if out.exists():
        out.rename(out.with_suffix(".csv.bak_" + time.strftime("%Y%m%d_%H%M%S")))
    fieldnames = [
        "experiment",
        "trial",
        "model",
        "image_size",
        "seq_len",
        "policy",
        "contention",
        "attacker_model",
        "attacker_policy",
        "threshold_mb",
        "latency_ms",
        "max_allocated_mb",
        "max_reserved_mb",
        "repeats",
        "warmup",
    ]
    cases = [
        ("gpt2", 224, 512, "default", "mobilenetv2", 640, 128, "default"),
        ("gpt2", 224, 512, "default", "mobilenetv2", 640, 128, "mixed_managed"),
        ("gpt2", 224, 512, "default", "mobilenetv2", 640, 128, "mixed_zc"),
        ("gpt2", 224, 1024, "default", "mobilenetv2", 640, 128, "default"),
        ("gpt2", 224, 1024, "default", "mobilenetv2", 640, 128, "mixed_managed"),
        ("mobilenetv2", 640, 128, "default", "gpt2", 224, 512, "default"),
        ("mobilenetv2", 640, 128, "default", "gpt2", 224, 512, "mixed_managed"),
        ("mobilenetv2", 1024, 128, "default", "gpt2", 224, 512, "default"),
        ("mobilenetv2", 1024, 128, "default", "gpt2", 224, 512, "mixed_managed"),
    ]
    for trial in range(1, args.trials + 1):
        for model, image_size, seq_len, policy, attacker_model, attacker_image, attacker_seq, attacker_policy in cases:
            cmd = [
                sys.executable,
                str(ROOT / "orin_contention_worker.py"),
                "--model",
                model,
                "--image-size",
                str(image_size),
                "--seq-len",
                str(seq_len),
                "--policy",
                policy,
                "--contention",
                "model",
                "--attacker-model",
                attacker_model,
                "--attacker-image-size",
                str(attacker_image),
                "--attacker-seq-len",
                str(attacker_seq),
                "--attacker-policy",
                attacker_policy,
                "--threshold-mb",
                "4",
                "--repeats",
                str(args.contention_repeats),
                "--warmup",
                str(args.warmup),
                "--attacker-warmup-s",
                str(args.attacker_warmup_s),
            ]
            row = parse_result(run(cmd))
            row["experiment"] = "contention_core"
            row["trial"] = trial
            append_row(out, row, fieldnames)
    print(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--input-repeats", type=int, default=30)
    p.add_argument("--internal-repeats", type=int, default=30)
    p.add_argument("--contention-repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--attacker-warmup-s", type=float, default=3.0)
    p.add_argument("--stage", choices=["all", "input", "internal", "contention"], default="all")
    args = p.parse_args()

    if args.stage in ("all", "input"):
        actual_input(args)
    if args.stage in ("all", "internal"):
        internal_policy(args)
    if args.stage in ("all", "contention"):
        contention(args)


if __name__ == "__main__":
    main()
