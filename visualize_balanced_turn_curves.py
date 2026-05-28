#!/usr/bin/env python3
"""Plot balanced turn sweep curves from a trial manifest."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("trials/balanced_turn_sweep_70.json")
DEFAULT_OUTPUT = Path("trials/balanced_turn_sweep_70_curves.png")
DEFAULT_BIN_MM = 10.0
TURN_ORDER = ("FORWARD+LEFT", "BACK+RIGHT", "FORWARD+RIGHT", "BACK+LEFT")
TURN_COLORS = {
    "FORWARD+LEFT": "#2563eb",
    "BACK+RIGHT": "#dc2626",
    "FORWARD+RIGHT": "#16a34a",
    "BACK+LEFT": "#9333ea",
}


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _completed_rows(manifest: dict) -> list[dict]:
    rows = []
    for row in list(manifest.get("trials") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() != "completed":
            continue
        if row.get("usable") is not True:
            continue
        start_dist = _coerce_float(row.get("startDist"))
        x_gap_closed = _coerce_float(row.get("xGapClosed"))
        curve_pair = str(row.get("curvePair") or "").strip().upper()
        if start_dist is None or x_gap_closed is None or not curve_pair:
            continue
        rows.append(
            {
                "curve_pair": curve_pair,
                "start_dist": float(start_dist),
                "x_gap_closed": float(x_gap_closed),
                "trial": int(row.get("trial") or 0),
            }
        )
    return rows


def _bin_start(value: float, bin_mm: float) -> float:
    return float(int(float(value) // float(bin_mm)) * float(bin_mm))


def _binned_medians(rows: list[dict], *, bin_mm: float) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_bin_start(row["start_dist"], bin_mm)].append(row)
    medians = []
    for bin_start in sorted(grouped):
        bucket = grouped[bin_start]
        medians.append(
            {
                "bin_start": float(bin_start),
                "bin_mid": float(bin_start) + (float(bin_mm) / 2.0),
                "count": int(len(bucket)),
                "median_start_dist": float(statistics.median(row["start_dist"] for row in bucket)),
                "median_x_gap_closed": float(statistics.median(row["x_gap_closed"] for row in bucket)),
            }
        )
    return medians


def plot_balanced_turn_curves(
    *,
    input_path: Path,
    output_path: Path,
    bin_mm: float = DEFAULT_BIN_MM,
) -> Path:
    manifest = _load_manifest(input_path)
    rows = _completed_rows(manifest)
    if not rows:
        raise ValueError(f"No complete usable rows found in {input_path}")

    curve = dict(manifest.get("curve") or {})
    duration_ms = curve.get("measured_phase_duration_ms")
    score_pct = curve.get("score_pct")
    title_bits = ["Balanced Turn Sweep"]
    if duration_ms:
        title_bits.append(f"{int(float(duration_ms))}ms")
    if score_pct:
        title_bits.append(f"score {int(float(score_pct))}%")

    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#ffffff")

    present_order = [name for name in TURN_ORDER if any(row["curve_pair"] == name for row in rows)]
    for curve_pair in present_order:
        curve_rows = [row for row in rows if row["curve_pair"] == curve_pair]
        color = TURN_COLORS.get(curve_pair, "#111827")
        ax.scatter(
            [row["start_dist"] for row in curve_rows],
            [row["x_gap_closed"] for row in curve_rows],
            s=34,
            alpha=0.26,
            color=color,
            edgecolors="none",
        )
        medians = _binned_medians(curve_rows, bin_mm=float(bin_mm))
        ax.plot(
            [row["bin_mid"] for row in medians],
            [row["median_x_gap_closed"] for row in medians],
            marker="o",
            linewidth=2.6,
            markersize=5.6,
            color=color,
            label=f"{curve_pair} ({len(curve_rows)} trials)",
        )

    ax.axhline(0.0, color="#6b7280", linewidth=1.2)
    ax.grid(True, which="major", color="#d8d8d2", linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlabel("Starting brick distance (mm)", fontsize=12, labelpad=10)
    ax.set_ylabel("X gap closed per act (mm)", fontsize=12, labelpad=10)
    ax.set_title(" - ".join(title_bits), fontsize=17, fontweight="bold", pad=15)
    subtitle = f"Raw points with {float(bin_mm):.0f}mm-bin median curves. Source: {input_path}"
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        fontsize=10.5,
        color="#4b5563",
    )
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#d1d5db")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bin-mm", type=float, default=DEFAULT_BIN_MM)
    args = parser.parse_args()

    output_path = plot_balanced_turn_curves(
        input_path=Path(args.input),
        output_path=Path(args.output),
        bin_mm=float(args.bin_mm),
    )
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
