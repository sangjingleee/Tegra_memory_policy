#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


MODES = [
    "pageable_input",
    "pinned_input",
    "mapped_zc_input",
    "managed_input",
    "device_preloaded_input",
]
COLORS = {
    "pageable_input": "#d94841",
    "pinned_input": "#f0a33a",
    "mapped_zc_input": "#3c9d5a",
    "managed_input": "#3b82c4",
    "device_preloaded_input": "#555555",
}


def read_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["total_ms"] = float(r["total_ms"])
        r["h2d_ms"] = float(r["h2d_ms"])
        r["kernel_ms"] = float(r["kernel_ms"])
    return rows


def short_label(cont):
    return (
        cont.replace("model_", "")
        .replace("mobilenetv2", "mbv2")
        .replace("mixed_managed", "mixed")
        .replace("default", "def")
    )


def draw_grouped(rows, out, title, size_mb):
    subset = [r for r in rows if r["size_mb"] == str(float(size_mb))]
    contentions = []
    for r in subset:
        if r["contention"] not in contentions:
            contentions.append(r["contention"])
    w, h = max(1080, 245 * len(contentions)), 560
    left, top, bottom = 90, 70, 88
    plot_w, plot_h = w - left - 35, h - top - bottom
    vmax = max(r["total_ms"] for r in subset) * 1.18
    group_w = plot_w / len(contentions)
    bar_w = group_w / (len(MODES) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-size="15" font-family="Arial">Input latency (ms)</text>',
    ]
    for i in range(6):
        yv = vmax * i / 5
        y = top + plot_h - (yv / vmax) * plot_h
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial">{yv:.1f}</text>')
    for gi, cont in enumerate(contentions):
        x0 = left + gi * group_w
        parts.append(f'<text x="{x0 + group_w/2}" y="{h - 45}" text-anchor="middle" font-size="12" font-family="Arial">{short_label(cont)}</text>')
        for mi, mode in enumerate(MODES):
            r = next((r for r in subset if r["contention"] == cont and r["mode"] == mode), None)
            if r is None:
                continue
            bh = (r["total_ms"] / vmax) * plot_h
            x = x0 + bar_w * (mi + 0.5)
            y = top + plot_h - bh
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w*0.82}" height="{bh}" fill="{COLORS[mode]}"/>')
            parts.append(f'<text x="{x + bar_w*0.41}" y="{y - 5}" text-anchor="middle" font-size="10" font-family="Arial">{r["total_ms"]:.2f}</text>')
    lx = left
    ly = h - 18
    for mode in MODES:
        label = mode.replace("_input", "").replace("device_preloaded", "device")
        parts.append(f'<rect x="{lx}" y="{ly-11}" width="12" height="12" fill="{COLORS[mode]}"/>')
        parts.append(f'<text x="{lx+17}" y="{ly}" font-size="12" font-family="Arial">{label}</text>')
        lx += 165
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="input_boundary_model_contention_orin.csv")
    p.add_argument("--prefix", default="input_boundary_model_orin")
    args = p.parse_args()
    rows = read_rows(Path(args.csv))
    sizes = sorted({float(r["size_mb"]) for r in rows})
    for size in sizes:
        label = f"{int(size)}" if size.is_integer() else f"{size:g}"
        draw_grouped(
            rows,
            Path(f"{args.prefix}_{label}mb_total.svg"),
            f"Orin-A Input-Only Boundary Latency ({label}MB)",
            size,
        )


if __name__ == "__main__":
    main()
