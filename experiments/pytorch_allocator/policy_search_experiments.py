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


def append_row(path, row, fieldnames):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_backup(path):
    if path.exists():
        path.rename(path.with_suffix(path.suffix + ".bak_" + time.strftime("%Y%m%d_%H%M%S")))


def input_boundary(args):
    out = ROOT / "policy_search_input_boundary.csv"
    maybe_backup(out)
    fieldnames = [
        "experiment",
        "trial",
        "victim_model",
        "victim_image_size",
        "victim_seq_len",
        "input_mode",
        "attacker_model",
        "attacker_policy",
        "attacker_image_size",
        "attacker_seq_len",
        "threshold_mb",
        "latency_ms",
        "repeats",
        "warmup",
    ]
    modes = ["device_preloaded", "pageable", "pinned", "mapped_zc", "managed"]
    cases = []
    for image_size in args.image_sizes:
        for attacker_model, attacker_policy, attacker_image, attacker_seq in [
            ("none", "default", 640, 512),
            ("gpt2", "default", 640, 512),
            ("gpt2", "mixed_managed", 640, 512),
        ]:
            for mode in modes:
                cases.append(("mobilenetv2", image_size, 512, mode, attacker_model, attacker_policy, attacker_image, attacker_seq))
    for seq_len in args.seq_lens:
        for attacker_model, attacker_policy, attacker_image, attacker_seq in [
            ("none", "default", 640, 512),
            ("mobilenetv2", "default", 640, 512),
            ("mobilenetv2", "mixed_managed", 640, 512),
        ]:
            for mode in modes:
                cases.append(("gpt2", 640, seq_len, mode, attacker_model, attacker_policy, attacker_image, attacker_seq))

    for trial in range(1, args.trials + 1):
        for victim_model, image_size, seq_len, mode, attacker_model, attacker_policy, attacker_image, attacker_seq in cases:
            cmd = [
                sys.executable,
                str(ROOT / "actual_model_input_policy.py"),
                "--victim-model",
                victim_model,
                "--victim-image-size",
                str(image_size),
                "--victim-seq-len",
                str(seq_len),
                "--input-mode",
                mode,
                "--attacker-model",
                attacker_model,
                "--attacker-policy",
                attacker_policy,
                "--attacker-image-size",
                str(attacker_image),
                "--attacker-seq-len",
                str(attacker_seq),
                "--threshold-mb",
                str(args.threshold_mb),
                "--repeats",
                str(args.input_repeats),
                "--warmup",
                str(args.warmup),
                "--attacker-warmup-s",
                str(args.attacker_warmup_s),
            ]
            row = parse_result(run(cmd))
            row.update(
                {
                    "experiment": "input_boundary_policy_search",
                    "trial": trial,
                    "attacker_image_size": attacker_image if attacker_model == "mobilenetv2" else "",
                    "attacker_seq_len": attacker_seq if attacker_model == "gpt2" else "",
                    "threshold_mb": args.threshold_mb,
                }
            )
            append_row(out, row, fieldnames)
    print(out)


def internal_threshold(args):
    out = ROOT / "policy_search_internal_threshold.csv"
    maybe_backup(out)
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
    for image_size in args.image_sizes:
        for policy in ["default", "custom_device", "all_managed", "all_zc"]:
            cases.append(("mobilenetv2", image_size, 512, policy, 4))
        for threshold in args.thresholds:
            for policy in ["mixed_managed", "mixed_zc"]:
                cases.append(("mobilenetv2", image_size, 512, policy, threshold))
    for seq_len in args.seq_lens:
        for policy in ["default", "custom_device", "all_managed", "all_zc"]:
            cases.append(("gpt2", 640, seq_len, policy, 4))
        for threshold in args.thresholds:
            for policy in ["mixed_managed", "mixed_zc"]:
                cases.append(("gpt2", 640, seq_len, policy, threshold))

    for trial in range(1, args.trials + 1):
        for model, image_size, seq_len, policy, threshold in cases:
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
            row.update({"experiment": "internal_threshold_policy_search", "trial": trial})
            append_row(out, row, fieldnames)
    print(out)


def contention_search(args):
    out = ROOT / "policy_search_contention.csv"
    maybe_backup(out)
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
        "attacker_image_size",
        "attacker_seq_len",
        "threshold_mb",
        "latency_ms",
        "max_allocated_mb",
        "max_reserved_mb",
        "repeats",
        "warmup",
    ]
    cases = []
    attacker_policies = [("default", 4), ("all_managed", 4), ("mixed_managed", 1), ("mixed_managed", 4), ("mixed_managed", 16), ("mixed_zc", 4)]
    victim_policies = ["default", "mixed_managed"]

    for seq_len in [512, 1024]:
        for image_size in [640, 1024]:
            for victim_policy in victim_policies:
                for attacker_policy, threshold in attacker_policies:
                    cases.append(("gpt2", 640, seq_len, victim_policy, "mobilenetv2", image_size, 512, attacker_policy, threshold))
                    cases.append(("mobilenetv2", image_size, 512, victim_policy, "gpt2", 640, seq_len, attacker_policy, threshold))

    for trial in range(1, args.trials + 1):
        for model, image_size, seq_len, policy, attacker_model, attacker_image, attacker_seq, attacker_policy, threshold in cases:
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
                    "experiment": "contention_policy_search",
                    "trial": trial,
                    "attacker_image_size": attacker_image if attacker_model == "mobilenetv2" else "",
                    "attacker_seq_len": attacker_seq if attacker_model == "gpt2" else "",
                }
            )
            append_row(out, row, fieldnames)
    print(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "input", "internal", "contention"], default="all")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--image-sizes", type=int, nargs="+", default=[224, 640, 1024])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024])
    p.add_argument("--thresholds", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--input-repeats", type=int, default=25)
    p.add_argument("--internal-repeats", type=int, default=25)
    p.add_argument("--contention-repeats", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--attacker-warmup-s", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage in ("all", "input"):
        input_boundary(args)
    if args.stage in ("all", "internal"):
        internal_threshold(args)
    if args.stage in ("all", "contention"):
        contention_search(args)


if __name__ == "__main__":
    main()
