#!/usr/bin/env python3
"""Run repeated yMotorCal sweeps and summarize the mast duration curve."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from helper_robot_control import Robot


PULSE_MS = (500, 650, 800, 900)
DEFAULT_RUNS = 10
DEFAULT_TARGET_Y_MM = 9.0
DEFAULT_TARGET_BAND_MM = 8.0
DEFAULT_PWM = 255
DEFAULT_PRE_RAISE_MS = 200


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else None


def _iqr(values: list[float]) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if len(clean) < 4:
        return None
    qs = statistics.quantiles(clean, n=4, method="inclusive")
    return float(qs[2] - qs[0])


def _send_mast(cmd: str, pwm: int, duration_ms: int) -> None:
    robot = Robot(exit_on_failure=False)
    try:
        robot.stop()
        robot.send_command_pwm(cmd, int(pwm), duration_ms=int(duration_ms))
        time.sleep(float(duration_ms) / 1000.0)
        robot.stop()
    finally:
        try:
            robot.close()
        except Exception:
            pass


def _safe_pre_raise(duration_ms: int, pwm: int) -> None:
    if duration_ms <= 0:
        return
    print(f"[Y-CURVE] pre-raise mast: U {int(duration_ms)}ms pwm={int(pwm)}", flush=True)
    _send_mast("u", int(pwm), int(duration_ms))
    time.sleep(0.45)


def _run_once(args: argparse.Namespace, run_idx: int, out_dir: Path) -> dict:
    json_path = out_dir / f"y_curve_run_{run_idx:02d}.json"
    profiles: list[str] = []
    for pulse_ms in PULSE_MS:
        profiles.extend(["--profile", f"p{pulse_ms}:255:{pulse_ms}:1"])
    cmd = [
        sys.executable,
        "-u",
        "yMotorCal.py",
        "--source",
        str(args.source),
        "--target-y-mm",
        str(float(args.target_y_mm)),
        "--target-y-band-mm",
        str(float(args.target_y_band_mm)),
        "--no-require-target-zone",
        "--centering-pulse-ms",
        str(int(args.centering_pulse_ms)),
        "--centering-max-attempts",
        str(int(args.centering_max_attempts)),
        "--recovery-pulse-ms",
        str(int(args.recovery_pulse_ms)),
        "--recovery-pwm",
        str(int(args.recovery_pwm)),
        "--recovery-max-attempts",
        str(int(args.recovery_max_attempts)),
        "--settle-s",
        str(float(args.settle_s)),
        "--samples",
        str(int(args.samples)),
        "--sample-interval-s",
        str(float(args.sample_interval_s)),
        "--min-confidence-pct",
        str(float(args.min_confidence_pct)),
        "--max-y-excursion-mm",
        str(float(args.max_y_excursion_mm)),
        "--max-dist-drift-mm",
        str(float(args.max_dist_drift_mm)),
        "--max-x-drift-mm",
        str(float(args.max_x_drift_mm)),
        "--output-json",
        str(json_path),
        "--measure-order",
        str(args.measure_order),
        *profiles,
    ]
    print(f"[Y-CURVE] run {run_idx}/{args.runs}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, check=False)
    if not json_path.exists():
        return {"ok": False, "reason": "missing_output_json", "returncode": proc.returncode, "cycles": []}
    result = json.loads(json_path.read_text(encoding="utf-8"))
    result["returncode"] = int(proc.returncode)
    return result


def _extract_points(results: list[dict]) -> dict[int, dict[str, object]]:
    by_pulse: dict[int, dict[str, object]] = {
        int(ms): {"down_closure_mm": [], "up_closure_mm": []} for ms in PULSE_MS
    }
    for result in results:
        for row in result.get("cycles") or []:
            if not row.get("quality_ok"):
                continue
            pulse_ms = int(row.get("pulse_ms") or 0)
            if pulse_ms not in by_pulse:
                continue
            up_delta = row.get("up_delta_mm")
            down_delta = row.get("down_delta_mm")
            if up_delta is not None:
                by_pulse[pulse_ms]["up_closure_mm"].append(float(up_delta))
            if down_delta is not None:
                by_pulse[pulse_ms]["down_closure_mm"].append(abs(float(down_delta)))
    for pulse_ms, row in by_pulse.items():
        down_values = list(row["down_closure_mm"])
        up_values = list(row["up_closure_mm"])
        row["samples"] = len(down_values)
        row["median_down_closure_mm"] = _median(down_values)
        row["median_up_closure_mm"] = _median(up_values)
        row["down_iqr_mm"] = _iqr(down_values)
        row["up_iqr_mm"] = _iqr(up_values)
        if row["median_down_closure_mm"] is not None:
            row["down_mm_per_100ms"] = float(row["median_down_closure_mm"]) * 100.0 / float(pulse_ms)
        else:
            row["down_mm_per_100ms"] = None
    return by_pulse


def _recommend_world_model(points: dict[int, dict[str, object]]) -> dict:
    rates = [
        float(row["down_mm_per_100ms"])
        for row in points.values()
        if row.get("down_mm_per_100ms") is not None and int(row.get("samples") or 0) >= 2
    ]
    median_rate = _median(rates) or 1.1
    return {
        "mast_min_pulse_ms": 170,
        "mast_max_pulse_ms": 350,
        "finish_mast_min_pulse_ms": 170,
        "finish_mast_max_pulse_ms": 350,
        "mast_down_mm_per_100ms": round(float(median_rate), 3),
        "mast_min_effect_mm": round(float(median_rate) * 1.70, 3),
        "mast_overshoot_guard_margin_mm": 0.2,
    }


def _write_summary(out_dir: Path, results: list[dict]) -> dict:
    points = _extract_points(results)
    recommendation = _recommend_world_model(points)
    summary = {
        "pulse_ms": list(PULSE_MS),
        "runs": len(results),
        "ok_runs": sum(1 for r in results if bool(r.get("ok"))),
        "points": points,
        "recommendation": recommendation,
    }
    summary_path = out_dir / "y_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[Y-CURVE] summary:", flush=True)
    for pulse_ms in PULSE_MS:
        row = points[pulse_ms]
        med = row.get("median_down_closure_mm")
        spread = row.get("down_iqr_mm")
        print(
            f"[Y-CURVE]   {pulse_ms}ms down closes "
            f"{'N/A' if med is None else f'{float(med):.2f}mm'} "
            f"(n={row.get('samples')}, iqr={'N/A' if spread is None else f'{float(spread):.2f}mm'})",
            flush=True,
        )
    print(f"[Y-CURVE] recommendation {json.dumps(recommendation, sort_keys=True)}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated paired y calibration sweeps.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--source", choices=("auto", "stream", "camera"), default="camera")
    parser.add_argument("--measure-order", choices=("up-first", "down-first"), default="down-first")
    parser.add_argument("--target-y-mm", type=float, default=DEFAULT_TARGET_Y_MM)
    parser.add_argument("--target-y-band-mm", type=float, default=DEFAULT_TARGET_BAND_MM)
    parser.add_argument("--pre-raise-ms", type=int, default=DEFAULT_PRE_RAISE_MS)
    parser.add_argument("--pre-raise-pwm", type=int, default=DEFAULT_PWM)
    parser.add_argument("--centering-pulse-ms", type=int, default=170)
    parser.add_argument("--centering-max-attempts", type=int, default=4)
    parser.add_argument("--recovery-pulse-ms", type=int, default=170)
    parser.add_argument("--recovery-pwm", type=int, default=DEFAULT_PWM)
    parser.add_argument("--recovery-max-attempts", type=int, default=6)
    parser.add_argument("--settle-s", type=float, default=0.45)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-interval-s", type=float, default=0.06)
    parser.add_argument("--min-confidence-pct", type=float, default=40.0)
    parser.add_argument("--max-y-excursion-mm", type=float, default=12.0)
    parser.add_argument("--max-dist-drift-mm", type=float, default=45.0)
    parser.add_argument("--max-x-drift-mm", type=float, default=45.0)
    parser.add_argument("--out-dir", default="runs/y_curve_calibration")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _safe_pre_raise(int(args.pre_raise_ms), int(args.pre_raise_pwm))
    results: list[dict] = []
    for run_idx in range(1, max(1, int(args.runs)) + 1):
        result = _run_once(args, run_idx, out_dir)
        results.append(result)
        _write_summary(out_dir, results)
        if not bool(result.get("ok")):
            print(
                f"[Y-CURVE] run {run_idx} stopped early: {result.get('reason', 'unknown')}. "
                "Pausing after summary so the next run can recover from a known state.",
                flush=True,
            )
        time.sleep(0.5)
    _write_summary(out_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
