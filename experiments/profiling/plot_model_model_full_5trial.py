#!/usr/bin/env python3
import csv
import math
import statistics as st
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent

POLICIES = ["default", "custom_device", "all_managed", "all_zc", "mixed_managed", "mixed_zc"]
LABELS = ["default", "device", "managed", "all-zc", "mixed-mgd", "mixed-zc"]


def read_rows(csv_path):
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows, out_summary):
    grouped = {(vp, ap): [] for vp in POLICIES for ap in POLICIES}
    for r in rows:
        key = (r["policy"], r["attacker_policy"])
        if key in grouped:
            grouped[key].append(r)

    summary = []
    for vp in POLICIES:
        for ap in POLICIES:
            rs = grouped[(vp, ap)]
            if not rs:
                summary.append(
                    {
                        "victim_policy": vp,
                        "attacker_policy": ap,
                        "latency_mean_ms": "nan",
                        "latency_std_ms": "nan",
                        "slowdown_mean": "nan",
                        "slowdown_std": "nan",
                        "n": "0",
                    }
                )
                continue
            lat = [float(r["latency_ms"]) for r in rs]
            slow = [float(r["slowdown_vs_alone"]) for r in rs]
            summary.append(
                {
                    "victim_policy": vp,
                    "attacker_policy": ap,
                    "latency_mean_ms": f"{st.mean(lat):.4f}",
                    "latency_std_ms": f"{st.stdev(lat) if len(lat) > 1 else 0.0:.4f}",
                    "slowdown_mean": f"{st.mean(slow):.4f}",
                    "slowdown_std": f"{st.stdev(slow) if len(slow) > 1 else 0.0:.4f}",
                    "n": str(len(rs)),
                }
            )
    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return summary


def matrix(summary, key):
    mat = [[math.nan for _ in POLICIES] for _ in POLICIES]
    for r in summary:
        i = POLICIES.index(r["victim_policy"])
        j = POLICIES.index(r["attacker_policy"])
        mat[i][j] = float(r[key])
    return mat


def lerp(a, b, t):
    return int(a + (b - a) * t)


def color_scale(val, vmin, vmax):
    if not math.isfinite(val) or vmax <= vmin:
        return "#cccccc"
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if t < 0.5:
        u = t / 0.5
        r, g, b = lerp(92, 255, u), lerp(184, 221, u), lerp(92, 80, u)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = lerp(255, 215, u), lerp(221, 48, u), lerp(80, 39, u)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_svg(mat, out, title, unit_fmt, x_axis_label, y_axis_label, trial_label):
    vals = [v for row in mat for v in row if math.isfinite(v)]
    vmin, vmax = min(vals), max(vals)
    cell = 118
    left = 155
    top = 105
    n = len(POLICIES)
    width = left + cell * n + 40
    height = top + cell * n + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="23" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left + cell*n/2}" y="{top - 35}" text-anchor="middle" font-size="16" font-family="Arial">{x_axis_label}</text>',
        f'<text x="24" y="{top + cell*n/2}" transform="rotate(-90 24 {top + cell*n/2})" text-anchor="middle" font-size="16" font-family="Arial">{y_axis_label}</text>',
    ]
    for j, label in enumerate(LABELS):
        x = left + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top - 10}" text-anchor="middle" font-size="13" font-family="Arial">{label}</text>')
    for i, label in enumerate(LABELS):
        y = top + i * cell + cell / 2 + 5
        parts.append(f'<text x="{left - 15}" y="{y}" text-anchor="end" font-size="13" font-family="Arial">{label}</text>')
    for i in range(n):
        for j in range(n):
            val = mat[i][j]
            x = left + j * cell
            y = top + i * cell
            color = color_scale(val, vmin, vmax)
            text = "FAIL" if not math.isfinite(val) else format(val, unit_fmt)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="3"/>')
            parts.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 6}" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">{text}</text>')
    parts.append(f'<text x="{left}" y="{height - 25}" font-size="12" font-family="Arial" fill="#555">Mean over {trial_label}. Range: {vmin:.4f} to {vmax:.4f}. Lower is better.</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=str(ROOT / "model_model_full_stable_5trial_gpt2seq512_mobilenet640.csv"),
    )
    p.add_argument("--prefix", default="model_model_full_5trial")
    p.add_argument("--title-prefix", default="GPT2 seq512")
    p.add_argument("--x-axis-label", default="Attacker MobileNetV2@640 policy")
    p.add_argument("--y-axis-label", default="Victim GPT2 seq512 policy")
    p.add_argument("--trial-label", default="5 trials")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_lat = ROOT / f"{args.prefix}_latency_mean_heatmap.svg"
    out_slow = ROOT / f"{args.prefix}_slowdown_mean_heatmap.svg"
    out_std = ROOT / f"{args.prefix}_latency_std_heatmap.svg"
    out_summary = ROOT / f"{args.prefix}_summary.csv"

    rows = read_rows(csv_path)
    summary = aggregate(rows, out_summary)
    draw_svg(
        matrix(summary, "latency_mean_ms"),
        out_lat,
        f"{args.title_prefix} Latency Mean",
        ".2f",
        args.x_axis_label,
        args.y_axis_label,
        args.trial_label,
    )
    draw_svg(
        matrix(summary, "slowdown_mean"),
        out_slow,
        f"{args.title_prefix} Slowdown Mean",
        ".3f",
        args.x_axis_label,
        args.y_axis_label,
        args.trial_label,
    )
    draw_svg(
        matrix(summary, "latency_std_ms"),
        out_std,
        f"{args.title_prefix} Latency Std",
        ".2f",
        args.x_axis_label,
        args.y_axis_label,
        args.trial_label,
    )
    print(out_summary)


if __name__ == "__main__":
    main()
