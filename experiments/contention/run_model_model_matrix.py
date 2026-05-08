#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time


POLICIES = ["default", "custom_device", "all_managed", "all_zc", "mixed_managed", "mixed_zc"]
CORE_CASES = [
    ("default", "default"),
    ("default", "all_zc"),
    ("default", "mixed_zc"),
    ("all_zc", "default"),
    ("all_managed", "all_managed"),
    ("mixed_zc", "mixed_zc"),
]


def parse_result(stdout: str):
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            row = {}
            for part in line[len("RESULT ") :].split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    row[k] = v
            return row
    return None


def run_case(args, victim_policy, attacker_policy, extra_label=""):
    cmd = [
        sys.executable,
        args.worker,
        "--model",
        args.victim_model,
        "--image-size",
        str(args.victim_image_size),
        "--seq-len",
        str(args.victim_seq_len),
        "--policy",
        victim_policy,
        "--contention",
        "model",
        "--attacker-model",
        args.attacker_model,
        "--attacker-image-size",
        str(args.attacker_image_size),
        "--attacker-seq-len",
        str(args.attacker_seq_len),
        "--attacker-policy",
        attacker_policy,
        "--threshold-mb",
        str(args.threshold_mb),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--attacker-warmup-s",
        str(args.attacker_warmup_s),
    ]
    print(f"{extra_label}{' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr and proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)
    row = parse_result(proc.stdout)
    if row is None:
        row = {
            "model": args.victim_model,
            "image_size": str(args.victim_image_size) if args.victim_model == "mobilenetv2" else "",
            "seq_len": str(args.victim_seq_len) if args.victim_model == "gpt2" else "",
            "policy": victim_policy,
            "contention": "model",
            "attacker_model": args.attacker_model,
            "attacker_policy": attacker_policy,
            "threshold_mb": str(args.threshold_mb),
            "latency_ms": "FAIL",
            "max_allocated_mb": "NA",
            "max_reserved_mb": "NA",
            "repeats": str(args.repeats),
            "warmup": str(args.warmup),
        }
    row["returncode"] = str(proc.returncode)
    return row


def run_alone(args, victim_policy):
    cmd = [
        sys.executable,
        args.worker,
        "--model",
        args.victim_model,
        "--image-size",
        str(args.victim_image_size),
        "--seq-len",
        str(args.victim_seq_len),
        "--policy",
        victim_policy,
        "--contention",
        "none",
        "--threshold-mb",
        str(args.threshold_mb),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
    ]
    print(f"[alone] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr and proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)
    row = parse_result(proc.stdout)
    if row:
        return row.get("latency_ms")
    return ""


def main():
    p = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--worker", default=os.path.join(script_dir, "orin_contention_worker.py"))
    p.add_argument(
        "--out",
        default=os.path.join(
            script_dir, "model_model_gpt2_victim_mobilenet_attacker.csv"
        ),
    )
    p.add_argument("--victim-model", choices=["mobilenetv2", "gpt2"], default="gpt2")
    p.add_argument("--attacker-model", choices=["mobilenetv2", "gpt2"], default="mobilenetv2")
    p.add_argument("--victim-image-size", type=int, default=224)
    p.add_argument("--victim-seq-len", type=int, default=128)
    p.add_argument("--attacker-image-size", type=int, default=224)
    p.add_argument("--attacker-seq-len", type=int, default=128)
    p.add_argument("--threshold-mb", type=int, default=4)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--attacker-warmup-s", type=float, default=3.0)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--core", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out):
        os.rename(args.out, args.out + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))

    case_pairs = CORE_CASES if args.core else [(vp, ap) for vp in POLICIES for ap in POLICIES]
    victim_policies = sorted({vp for vp, _ in case_pairs}, key=POLICIES.index)

    alone = {}
    for vp in victim_policies:
        alone[vp] = run_alone(args, vp)

    rows = []
    fieldnames = None
    total = len(case_pairs) * args.trials
    idx = 0
    for trial in range(1, args.trials + 1):
        for vp, ap in case_pairs:
            idx += 1
            row = run_case(args, vp, ap, f"[{idx}/{total}] trial={trial} ")
            row["trial"] = str(trial)
            row["victim_alone_ms"] = alone.get(vp, "")
            try:
                row["slowdown_vs_alone"] = f"{float(row['latency_ms']) / float(row['victim_alone_ms']):.4f}"
            except Exception:
                row["slowdown_vs_alone"] = "NA"
            rows.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())
                with open(args.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerow(row)
            else:
                with open(args.out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writerow(row)

    print(f"\nSaved CSV: {args.out}")
    print("\nSummary:")
    for r in rows:
        print(
            f"victim={r['policy']:13s} attacker={r['attacker_policy']:13s} "
            f"lat={r['latency_ms']:>8s} slowdown={r['slowdown_vs_alone']:>6s} rc={r['returncode']}"
        )


if __name__ == "__main__":
    main()
