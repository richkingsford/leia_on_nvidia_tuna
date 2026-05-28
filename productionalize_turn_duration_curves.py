#!/usr/bin/env python3
"""Build provisional production turn-duration curves from balanced sweep manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("trials/balanced_turn_sweep_70_range.json")
DEFAULT_OUTPUT = Path("trials/turn_duration_production_curves.json")
DEFAULT_PRODUCTION_PHASES = ("forward_left", "back_right", "forward_right", "back_left")
PHASE_LABELS = {
    "forward_left": "FORWARD+LEFT",
    "back_right": "BACK+RIGHT",
    "forward_right": "FORWARD+RIGHT",
    "back_left": "BACK+LEFT",
}


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase_key(value) -> str:
    key = str(value or "").strip().lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    key = key.replace("backward_", "back_").replace("bwd_", "back_").replace("fwd_", "forward_")
    return key if key in PHASE_LABELS else ""


def _phase_list(value) -> set[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value or [])
    return {phase for phase in (_phase_key(item) for item in raw_items) if phase}


def _manifest_rows(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    rows = []
    if not isinstance(payload, dict):
        return rows
    for row in list(payload.get("trials") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() != "completed":
            continue
        if row.get("usable") is not True:
            continue
        phase = _phase_key(row.get("plannedPhase"))
        duration_ms = _coerce_float(row.get("measuredDurationMs"))
        x_mm = _coerce_float(row.get("xDelta"))
        if not phase or duration_ms is None or x_mm is None:
            continue
        move = {}
        moves = row.get("moves") if isinstance(row.get("moves"), list) else []
        if moves and isinstance(moves[0], dict):
            move = dict(moves[0])
        motor_pair = move.get("motor_pair") if isinstance(move.get("motor_pair"), dict) else {}
        rows.append(
            {
                "phase": phase,
                "duration_ms": int(round(float(duration_ms))),
                "x_traveled_mm": abs(float(x_mm)),
                "source": str(path),
                "trial": int(row.get("trial") or 0),
                "cmd": str(move.get("cmd") or "").strip().lower(),
                "drive_mode": str(move.get("drive_mode") or "").strip().lower(),
                "profile_name": str(row.get("measuredProfile") or move.get("profile_name") or "").strip(),
                "score_pct": int(round(_coerce_float(row.get("measuredScore")) or _coerce_float(move.get("score_pct")) or 1)),
                "action_note": str(move.get("action_note") or "").strip(),
                "left_motor_pwm": _coerce_float(motor_pair.get("left_motor_pwm")),
                "left_motor_action": str(motor_pair.get("left_motor_action") or "").strip().lower(),
                "right_motor_pwm": _coerce_float(motor_pair.get("right_motor_pwm")),
                "right_motor_action": str(motor_pair.get("right_motor_action") or "").strip().lower(),
            }
        )
    return rows


def _phase_motion_details(points: list[dict]) -> dict:
    rows = [dict(row) for row in points if isinstance(row, dict)]
    cmd_values = [str(row.get("cmd") or "").strip().lower() for row in rows if str(row.get("cmd") or "").strip()]
    drive_values = [str(row.get("drive_mode") or "").strip().lower() for row in rows if str(row.get("drive_mode") or "").strip()]
    profile_values = [str(row.get("profile_name") or "").strip() for row in rows if str(row.get("profile_name") or "").strip()]
    score_values = [_coerce_float(row.get("score_pct")) for row in rows]
    score_values = [float(value) for value in score_values if value is not None and float(value) > 0.0]
    action_values = [str(row.get("action_note") or "").strip() for row in rows if str(row.get("action_note") or "").strip()]
    left_pwm_values = [_coerce_float(row.get("left_motor_pwm")) for row in rows]
    right_pwm_values = [_coerce_float(row.get("right_motor_pwm")) for row in rows]
    left_pwm_values = [float(value) for value in left_pwm_values if value is not None and float(value) >= 0.0]
    right_pwm_values = [float(value) for value in right_pwm_values if value is not None and float(value) >= 0.0]
    left_pwm = statistics.median(left_pwm_values) if left_pwm_values else None
    right_pwm = statistics.median(right_pwm_values) if right_pwm_values else None
    outer_pwm = max(left_pwm, right_pwm) if left_pwm is not None and right_pwm is not None else None
    inner_pwm = min(left_pwm, right_pwm) if left_pwm is not None and right_pwm is not None else None
    inner_ratio = None
    if outer_pwm is not None and float(outer_pwm) > 0.0 and inner_pwm is not None:
        inner_ratio = float(inner_pwm) / float(outer_pwm)
    return {
        "cmd": cmd_values[0] if cmd_values else "",
        "drive_mode": drive_values[0] if drive_values else "",
        "profile_name": profile_values[0] if profile_values else "",
        "score_pct": int(round(statistics.median(score_values))) if score_values else 1,
        "pwm_override": int(round(float(outer_pwm))) if outer_pwm is not None else None,
        "profile_override": {
            "profile_name": profile_values[0] if profile_values else "",
            "drive_mode": drive_values[0] if drive_values else "",
            "inner_ratio": round(float(inner_ratio), 6) if inner_ratio is not None else 0.7,
            "outer_ratio": 1.0,
            "duration_mode": "max_turn_drive",
            "action_note": action_values[0] if action_values else "",
        },
        "motor_pair": {
            "left_motor_pwm": int(round(float(left_pwm))) if left_pwm is not None else None,
            "right_motor_pwm": int(round(float(right_pwm))) if right_pwm is not None else None,
        },
    }


def _linear_fit(points: list[dict]) -> dict:
    if not points:
        return {"slope_mm_per_ms": None, "intercept_mm": None, "r_squared": None}
    xs = [float(row["duration_ms"]) for row in points]
    ys = [float(row["x_traveled_mm"]) for row in points]
    if len(points) == 1 or max(xs) == min(xs):
        return {
            "slope_mm_per_ms": 0.0,
            "intercept_mm": round(float(statistics.median(ys)), 6),
            "r_squared": None,
        }
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - (slope * x_mean)
    predictions = [(slope * x) + intercept for x in xs]
    ss_res = sum((y - pred) ** 2 for y, pred in zip(ys, predictions))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = None if ss_tot <= 1e-9 else 1.0 - (ss_res / ss_tot)
    return {
        "slope_mm_per_ms": round(float(slope), 6),
        "intercept_mm": round(float(intercept), 6),
        "r_squared": None if r_squared is None else round(float(r_squared), 6),
    }


def _fit_outlier_filtered(points: list[dict]) -> tuple[list[dict], list[dict], dict]:
    if len(points) < 6:
        return list(points), [], _linear_fit(points)
    first_fit = _linear_fit(points)
    slope = _coerce_float(first_fit.get("slope_mm_per_ms")) or 0.0
    intercept = _coerce_float(first_fit.get("intercept_mm")) or 0.0
    residuals = [
        abs(float(row["x_traveled_mm"]) - ((float(row["duration_ms"]) * slope) + intercept))
        for row in points
    ]
    median_residual = statistics.median(residuals)
    mad = statistics.median(abs(residual - median_residual) for residual in residuals)
    if mad <= 1e-9:
        return list(points), [], first_fit
    threshold = max(0.5, 3.0 * 1.4826 * float(mad))
    kept = []
    outliers = []
    for row, residual in zip(points, residuals):
        out = dict(row)
        out["fit_residual_mm"] = round(float(residual), 3)
        if float(residual) > float(threshold):
            outliers.append(out)
        else:
            kept.append(out)
    if len(kept) < 4:
        return list(points), [], first_fit
    return kept, outliers, _linear_fit(kept)


def _median_points(points: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in points:
        grouped[int(row["duration_ms"])].append(float(row["x_traveled_mm"]))
    out = []
    for duration_ms in sorted(grouped):
        values = grouped[duration_ms]
        out.append(
            {
                "duration_ms": int(duration_ms),
                "median_x_traveled_mm": round(float(statistics.median(values)), 3),
                "samples": int(len(values)),
            }
        )
    return out


def _monotonic_score(points: list[dict]) -> float | None:
    medians = _median_points(points)
    if len(medians) < 2:
        return None
    ok = 0
    total = 0
    last = float(medians[0]["median_x_traveled_mm"])
    for row in medians[1:]:
        current = float(row["median_x_traveled_mm"])
        total += 1
        if current + 0.2 >= last:
            ok += 1
        last = current
    return round((float(ok) / float(total)) * 100.0, 3) if total else None


def build_curves(
    *,
    input_paths: list[Path],
    production_phases: set[str],
    min_used_samples: int = 8,
    min_duration_count: int = 5,
    production_note: str | None = None,
) -> dict:
    rows = []
    for path in input_paths:
        rows.extend(_manifest_rows(Path(path)))
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(row)

    phase_curves = {}
    for phase in PHASE_LABELS:
        phase_rows = sorted(by_phase.get(phase, []), key=lambda row: (int(row["duration_ms"]), int(row["trial"])))
        kept, outliers, fit = _fit_outlier_filtered(phase_rows)
        median_points = _median_points(kept)
        production_ready = (
            phase in production_phases
            and len(kept) >= int(min_used_samples)
            and len(median_points) >= int(min_duration_count)
        )
        motion_details = _phase_motion_details(kept or phase_rows)
        phase_curves[phase] = {
            "label": PHASE_LABELS[phase],
            "status": "provisional_production" if production_ready else "needs_more_trials",
            "production_ready": bool(production_ready),
            "cmd": motion_details["cmd"],
            "drive_mode": motion_details["drive_mode"],
            "profile_name": motion_details["profile_name"],
            "score_pct": motion_details["score_pct"],
            "pwm_override": motion_details["pwm_override"],
            "profile_override": motion_details["profile_override"],
            "motor_pair": motion_details["motor_pair"],
            "sample_count": int(len(phase_rows)),
            "used_sample_count": int(len(kept)),
            "outlier_count": int(len(outliers)),
            "duration_count": int(len(median_points)),
            "duration_range_ms": {
                "min": int(min((row["duration_ms"] for row in kept), default=0)),
                "max": int(max((row["duration_ms"] for row in kept), default=0)),
            },
            "monotonic_score_pct": _monotonic_score(kept),
            "fit": fit,
            "points": median_points,
            "outliers": outliers,
        }
    return {
        "file_type": "turn_drive_duration_curves",
        "schema_version": 1,
        "production": True,
        "production_note": (
            str(production_note)
            if production_note
            else "All selected phase curves are enabled for follow-game production by operator request."
        ),
        "production_min_used_samples": int(min_used_samples),
        "production_min_duration_count": int(min_duration_count),
        "production_phases": sorted(production_phases),
        "inputs": [str(path) for path in input_paths],
        "phase_curves": phase_curves,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--production-phases",
        type=str,
        default=",".join(DEFAULT_PRODUCTION_PHASES),
        help="Comma-separated phase names to mark as provisional production.",
    )
    parser.add_argument(
        "--min-used-samples",
        type=int,
        default=8,
        help="Minimum non-outlier samples required before a selected phase is production_ready.",
    )
    parser.add_argument(
        "--min-duration-count",
        type=int,
        default=5,
        help="Minimum distinct tested durations required before a selected phase is production_ready.",
    )
    parser.add_argument(
        "--production-note",
        type=str,
        default="",
        help="Optional note to write into the production curve payload.",
    )
    args = parser.parse_args()
    payload = build_curves(
        input_paths=[Path(path) for path in list(args.input or [DEFAULT_INPUT])],
        production_phases=_phase_list(args.production_phases),
        min_used_samples=max(1, int(args.min_used_samples)),
        min_duration_count=max(1, int(args.min_duration_count)),
        production_note=str(args.production_note or ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
