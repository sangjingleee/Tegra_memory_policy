#!/usr/bin/env python3
import argparse
import csv
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


def append_row(path, row, fields):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def backup(path):
    if path.exists():
        path.rename(path.with_suffix(path.suffix + ".bak_" + time.strftime("%Y%m%d_%H%M%S")))


def run_internal(args):
    out = ROOT / "policy_confirm_internal.csv"
    backup(out)
    fields = [
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
        ("gpt2_seq1024_default", "gpt2", 640, 1024, "default", 4),
        ("gpt2_seq1024_mixed_zc_16mb", "gpt2", 640, 1024, "mixed_zc", 16),
        ("gpt2_seq1024_mixed_managed_4mb", "gpt2", 640, 1024, "mixed_managed", 4),
    ]
    for trial in range(1, args.trials + 1):
        for label, model, image_size, seq_len, policy, threshold in cases:
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
                str(threshold),
                "--repeats",
                str(args.internal_repeats),
                "--warmup",
                str(args.warmup),
            ]
            row = parse_result(run(cmd))
            row.update({"experiment": label, "trial": trial})
            append_row(out, row, fields)
    print(out)


def run_contention(args):
    out = ROOT / "policy_confirm_contention.csv"
    backup(out)
    fields = [
        "experiment",
        "trial",
        "model",
        "image_size",
        "seq_len",
        "policy",
        "contention",
        "attacker_model",
        "attacker_policy",
        "attacker_image_size",
        "attacker_seq_len",
        "threshold_mb",
        "latency_ms",
        "max_allocated_mb",
        "max_reserved_mb",
        "repeats",
        "warmup",
    ]
    cases = [
        (
            "gpt2seq512_victim_mobilenet640_attacker",
            "gpt2",
            640,
            512,
            "default",
            "mobilenetv2",
            640,
            512,
            "default",
            4,
        ),
        (
            "gpt2seq512_victim_mobilenet640_attacker",
            "gpt2",
            640,
            512,
            "default",
            "mobilenetv2",
            640,
            512,
            "mixed_managed",
            4,
        ),
        (
            "gpt2seq1024_victim_mobilenet640_attacker",
            "gpt2",
            640,
            1024,
            "default",
            "mobilenetv2",
            640,
            512,
            "default",
            4,
        ),
        (
            "gpt2seq1024_victim_mobilenet640_attacker",
            "gpt2",
            640,
            1024,
            "default",
            "mobilenetv2",
            640,
            512,
            "mixed_managed",
            4,
        ),
        (
            "mobilenet1024_victim_gpt2seq512_attacker",
            "mobilenetv2",
            1024,
            512,
            "default",
            "gpt2",
            640,
            512,
            "default",
            4,
        ),
        (
            "mobilenet1024_victim_gpt2seq512_attacker",
            "mobilenetv2",
            1024,
            512,
            "default",
            "gpt2",
            640,
            512,
            "mixed_managed",
            4,
        ),
        (
            "gpt2seq512_victim_mobilenet1024_attacker",
            "gpt2",
            640,
            512,
            "default",
            "mobilenetv2",
            1024,
            512,
            "default",
            4,
        ),
        (
            "gpt2seq512_victim_mobilenet1024_attacker",
            "gpt2",
            640,
            512,
            "default",
            "mobilenetv2",
            1024,
            512,
            "mixed_managed",
            16,
        ),
        (
            "gpt2seq1024_victim_mobilenet1024_attacker",
            "gpt2",
            640,
            1024,
            "default",
            "mobilenetv2",
            1024,
            512,
            "default",
            4,
        ),
        (
            "gpt2seq1024_victim_mobilenet1024_attacker",
            "gpt2",
            640,
            1024,
            "default",
            "mobilenetv2",
            1024,
            512,
            "mixed_managed",
            16,
        ),
    ]
    for trial in range(1, args.trials + 1):
        for label, model, image_size, seq_len, policy, attacker_model, attacker_image, attacker_seq, attacker_policy, threshold in cases:
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
                str(threshold),
                "--repeats",
                str(args.contention_repeats),
                "--warmup",
                str(args.warmup),
                "--attacker-warmup-s",
                str(args.attacker_warmup_s),
            ]
            row = parse_result(run(cmd))
            row.update(
                {
                    "experiment": label,
                    "trial": trial,
                    "attacker_image_size": attacker_image if attacker_model == "mobilenetv2" else "",
                    "attacker_seq_len": attacker_seq if attacker_model == "gpt2" else "",
                }
            )
            append_row(out, row, fields)
    print(out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "internal", "contention"], default="all")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--internal-repeats", type=int, default=40)
    parser.add_argument("--contention-repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--attacker-warmup-s", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in ("all", "internal"):
        run_internal(args)
    if args.stage in ("all", "contention"):
        run_contention(args)


if __name__ == "__main__":
    main()
