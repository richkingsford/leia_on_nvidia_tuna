#!/usr/bin/env python3
"""Plot turn-drive x travel versus act duration from trial manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUTS = (Path("trials/balanced_turn_sweep_70.json"),)
DEFAULT_OUTPUT = Path("trials/balanced_turn_sweep_70_x_traveled_by_duration.png")
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


def _duration_from_row(row: dict, *, prefer_setup: bool = False):
    keys = (
        ("setupDurationMs", "duration", "duration_ms")
        if prefer_setup
        else ("measuredDurationMs", "measured_duration", "duration", "duration_ms", "setupDurationMs")
    )
    for key in keys:
        value = _coerce_float(row.get(key))
        if value is not None and value > 0:
            return float(value)
    return None


def _phase_from_moves(row: dict) -> str:
    moves = row.get("moves")
    if isinstance(moves, list) and moves:
        move = moves[-1] if isinstance(moves[-1], dict) else {}
        label = str(move.get("label") or "").strip().upper()
        if label:
            return label
    return str(row.get("curvePair") or row.get("plannedPhase") or "").strip().upper()


def _phase_from_manifest_curve(manifest: dict, *, measured: bool = True) -> str:
    curve = dict(manifest.get("curve") or {})
    phase = curve.get("measured_phase" if measured else "setup_phase")
    if isinstance(phase, dict):
        label = str(phase.get("label") or "").strip().upper()
        if label:
            return label
    label = str(curve.get("label") or "").strip().upper()
    return label


def _rows_from_manifest(manifest: dict, *, source: Path) -> list[dict]:
    rows = []
    direct_rows = [row for row in list(manifest.get("trials") or []) if isinstance(row, dict)]
    if direct_rows:
        for row in direct_rows:
            if str(row.get("status") or "").strip().lower() != "completed":
                continue
            if row.get("usable") is not True:
                continue
            duration = _duration_from_row(row)
            x_delta = _coerce_float(row.get("xDelta"))
            phase = _phase_from_moves(row)
            if duration is None or x_delta is None or phase not in TURN_ORDER:
                continue
            rows.append(
                {
                    "phase": phase,
                    "duration_ms": float(duration),
                    "x_traveled_mm": abs(float(x_delta)),
                    "trial": int(row.get("trial") or 0),
                    "source": str(source),
                }
            )

    measured_phase = _phase_from_manifest_curve(manifest, measured=True)
    for row in list(manifest.get("trials_backwards") or []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        if status != "completed":
            continue
        if row.get("usable") is not True:
            continue
        duration = _duration_from_row(row)
        x_delta = _coerce_float(row.get("xDelta"))
        if duration is None or x_delta is None or measured_phase not in TURN_ORDER:
            continue
        rows.append(
            {
                "phase": measured_phase,
                "duration_ms": float(duration),
                "x_traveled_mm": abs(float(x_delta)),
                "trial": int(row.get("trial") or 0),
                "source": str(source),
            }
        )
    return rows


def _load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if isinstance(manifest, dict):
            rows.extend(_rows_from_manifest(manifest, source=path))
    return rows


def _median_points(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(round(float(row["duration_ms"])))].append(row)
    points = []
    for duration in sorted(grouped):
        bucket = grouped[duration]
        points.append(
            {
                "duration_ms": int(duration),
                "median_x_traveled_mm": float(statistics.median(row["x_traveled_mm"] for row in bucket)),
                "count": int(len(bucket)),
            }
        )
    return points


def plot_turn_duration_curves(*, input_paths: list[Path], output_path: Path) -> Path:
    rows = _load_rows(input_paths)
    if not rows:
        raise ValueError("No complete usable turn-duration rows found.")

    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#ffffff")

    for phase in TURN_ORDER:
        phase_rows = [row for row in rows if row["phase"] == phase]
        if not phase_rows:
            continue
        color = TURN_COLORS.get(phase, "#111827")
        ax.scatter(
            [row["duration_ms"] for row in phase_rows],
            [row["x_traveled_mm"] for row in phase_rows],
            s=38,
            alpha=0.28,
            color=color,
            edgecolors="none",
        )
        medians = _median_points(phase_rows)
        linestyle = "-" if len(medians) > 1 else "None"
        ax.plot(
            [row["duration_ms"] for row in medians],
            [row["median_x_traveled_mm"] for row in medians],
            marker="o",
            linewidth=2.8,
            markersize=6.2,
            linestyle=linestyle,
            color=color,
            label=f"{phase} ({len(phase_rows)} trials, {len(medians)} durations)",
        )

    ax.axhline(0.0, color="#6b7280", linewidth=1.2)
    ax.grid(True, which="major", color="#d8d8d2", linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlabel("Act duration (ms)", fontsize=12, labelpad=10)
    ax.set_ylabel("X traveled per act, absolute mm", fontsize=12, labelpad=10)
    ax.set_title("Turn X Travel vs Act Duration", fontsize=17, fontweight="bold", pad=15)
    subtitle = "Raw points plus same-duration medians. Use multi-duration lines as true curves; single-duration series are calibration anchors."
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=10.5, color="#4b5563")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#d1d5db")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_paths = list(args.input or DEFAULT_INPUTS)
    output_path = plot_turn_duration_curves(
        input_paths=[Path(path) for path in input_paths],
        output_path=Path(args.output),
    )
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
