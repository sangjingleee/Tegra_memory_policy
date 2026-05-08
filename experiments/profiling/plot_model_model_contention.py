#!/usr/bin/env python3
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "model_model_gpt2_victim_mobilenet_attacker.csv"
OUT_LAT = ROOT / "model_model_gpt2_latency_heatmap_clean.svg"
OUT_SLOW = ROOT / "model_model_gpt2_slowdown_heatmap_clean.svg"

POLICIES = ["default", "custom_device", "all_managed", "all_zc"]
LABELS = ["default", "device", "managed", "zero-copy"]


def read_rows():
    with CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def matrix(rows, key):
    mat = [[math.nan for _ in POLICIES] for _ in POLICIES]
    for r in rows:
        vp = r["policy"]
        ap = r["attacker_policy"]
        if vp not in POLICIES or ap not in POLICIES:
            continue
        try:
            val = float(r[key])
        except Exception:
            val = math.nan
        mat[POLICIES.index(vp)][POLICIES.index(ap)] = val
    return mat


def lerp(a, b, t):
    return int(a + (b - a) * t)


def color_scale(val, vmin, vmax, invert=False):
    if not math.isfinite(val) or vmax <= vmin:
        return "#cccccc"
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if invert:
        t = 1.0 - t
    # green -> yellow -> red
    if t < 0.5:
        u = t / 0.5
        r, g, b = lerp(92, 255, u), lerp(184, 221, u), lerp(92, 80, u)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = lerp(255, 215, u), lerp(221, 48, u), lerp(80, 39, u)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_svg(mat, out, title, unit_fmt, invert=False):
    vals = [v for row in mat for v in row if math.isfinite(v)]
    vmin, vmax = min(vals), max(vals)
    cell = 118
    left = 155
    top = 105
    width = left + cell * 4 + 40
    height = top + cell * 4 + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left + cell*2}" y="{top - 35}" text-anchor="middle" font-size="16" font-family="Arial">Attacker MobileNetV2 policy</text>',
        f'<text x="24" y="{top + cell*2}" transform="rotate(-90 24 {top + cell*2})" text-anchor="middle" font-size="16" font-family="Arial">Victim GPT2 policy</text>',
    ]
    for j, label in enumerate(LABELS):
        x = left + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top - 10}" text-anchor="middle" font-size="13" font-family="Arial">{label}</text>')
    for i, label in enumerate(LABELS):
        y = top + i * cell + cell / 2 + 5
        parts.append(f'<text x="{left - 15}" y="{y}" text-anchor="end" font-size="13" font-family="Arial">{label}</text>')
    for i in range(4):
        for j in range(4):
            val = mat[i][j]
            x = left + j * cell
            y = top + i * cell
            color = color_scale(val, vmin, vmax, invert=invert)
            text = "FAIL" if not math.isfinite(val) else format(val, unit_fmt)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="3"/>')
            parts.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 6}" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">{text}</text>')
    parts.append(f'<text x="{left}" y="{height - 25}" font-size="12" font-family="Arial" fill="#555">Range: {vmin:.4f} to {vmax:.4f}. Lower is better.</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


def main():
    rows = read_rows()
    lat = matrix(rows, "latency_ms")
    slow = matrix(rows, "slowdown_vs_alone")
    draw_svg(
        lat,
        OUT_LAT,
        "GPT2 Victim Latency Under MobileNetV2 Attacker",
        ".2f",
    )
    draw_svg(
        slow,
        OUT_SLOW,
        "GPT2 Slowdown vs Alone",
        ".3f",
    )


if __name__ == "__main__":
    main()
