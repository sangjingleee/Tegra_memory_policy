#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


API_NAMES = [
    "cudaMalloc",
    "cudaFree",
    "cudaMallocManaged",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cudaHostGetDevicePointer",
    "cudaFreeHost",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaMemset",
    "cudaMemsetAsync",
]

API_PATTERNS = {
    name: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for name in API_NAMES
}


def run_stats(report: Path) -> str:
    proc = subprocess.run(
        ["nsys", "stats", "--report", "cuda_api_sum", str(report)],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout + proc.stderr


def parse_counts(text: str):
    counts = {name: 0 for name in API_NAMES}
    for line in text.splitlines():
        normalized = line.replace(",", "")
        for name in API_NAMES:
            match = API_PATTERNS[name].search(normalized)
            if not match:
                continue
            # cuda_api_sum prints Num Calls immediately before the Avg column,
            # and before the API Name column in text output. Taking the third
            # numeric field before the exact API name is stable for Nsight's
            # default text table: Time %, Total Time, Num Calls, ...
            before = normalized[: match.start()]
            before_numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", before)
            if len(before_numbers) >= 3:
                counts[name] += int(float(before_numbers[2]))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for rep in args.reports:
        report = Path(rep)
        text = run_stats(report)
        counts = parse_counts(text)
        row = {"report": report.name, **counts}
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        fieldnames = ["report"] + API_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
