#!/usr/bin/env python3
"""Discover the breakaway PWM for a named duty-cycle curve."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from a_follow_the_brick import (
    CROWN_PROFILE_TUNING,
    BrickDetector as FollowBrickDetector,
    _read_brick_measurement,
    _reset_follow_reading_history,
    _set_game_profile,
    _warmup,
)
from helper_manual_drive_breakaway_test import (
    DEFAULT_VISION_MODE,
    _build_vision,
    _collect_pose_samples,
    _median_pose_or_none,
)
from helper_robot_control import Robot
from run_speed_duty_curves import DEFAULT_DIRECTION_ORDER, DIRECTIONS, INTERVAL_MS, STRENGTHS, _actions_for_curve
from telemetry_robot import StepState, WorldModel, pwm_to_power


DEFAULT_LABEL = "fr_gentle"
DEFAULT_CANDIDATE_PWMS = (104, 112, 120, 128, 136, 145, 153, 160, 176, 196)
DEFAULT_TICKS = 4
DEFAULT_MOVEMENT_THRESHOLD_MM = 3.0
DEFAULT_LOG_DIR = Path("logs/speed_tests")


def _parse_label(label: str) -> tuple[str, str]:
    raw = str(label or "").strip().lower()
    if "_" not in raw:
        raise ValueError(f"Curve label must look like fr_gentle; got {label!r}.")
    direction_key, strength = raw.split("_", 1)
    if direction_key not in DIRECTIONS:
        raise ValueError(f"Unsupported direction {direction_key!r}; expected one of {sorted(DIRECTIONS)}.")
    if strength not in STRENGTHS:
        raise ValueError(f"Unsupported strength {strength!r}; expected one of {sorted(STRENGTHS)}.")
    return direction_key, strength


def _labels_for_strength(strength: str) -> tuple[str, ...]:
    strength_key = str(strength or "").strip().lower()
    if strength_key not in STRENGTHS:
        raise ValueError(f"Unsupported strength {strength!r}; expected one of {sorted(STRENGTHS)}.")
    return tuple(f"{direction_key}_{strength_key}" for direction_key in DEFAULT_DIRECTION_ORDER)


def _pose(vision, world, *, samples: int, timeout_s: float, use_follow_reading: bool) -> dict | None:
    if bool(use_follow_reading):
        values = []
        started = time.monotonic()
        requested = max(1, int(samples))
        while len(values) < requested and (time.monotonic() - started) < float(timeout_s):
            reading = _read_brick_measurement(vision, jump_guard=True)
            if bool(reading.get("confident")):
                try:
                    x_value = reading.get("x_axis_mm")
                    if x_value is None:
                        x_value = reading.get("x_mm")
                    values.append(
                        {
                            "dist_mm": float(reading.get("dist_mm")),
                            "x_axis_mm": float(x_value),
                        }
                    )
                except (TypeError, ValueError):
                    pass
            time.sleep(0.04)
        return _median_pose_or_none(values)
    return _median_pose_or_none(
        _collect_pose_samples(
            vision,
            world,
            samples=max(1, int(samples)),
            timeout_s=max(0.1, float(timeout_s)),
        )
    )


def _fmt_pose(pose: dict | None) -> str:
    if not isinstance(pose, dict):
        return "dist=? x=?"
    return f"dist={float(pose.get('dist_mm')):.1f} x={float(pose.get('x_axis_mm')):+.1f}"


def _movement_delta(pre_pose: dict, post_pose: dict) -> dict:
    dist_delta = float(post_pose["dist_mm"]) - float(pre_pose["dist_mm"])
    x_delta = float(post_pose["x_axis_mm"]) - float(pre_pose["x_axis_mm"])
    return {
        "dist_delta_mm": float(dist_delta),
        "x_delta_mm": float(x_delta),
        "vector_delta_mm": float(math.hypot(float(dist_delta), float(x_delta))),
    }


def _stable_pose(
    vision,
    world,
    *,
    samples: int,
    timeout_s: float,
    settle_s: float,
    max_drift_mm: float,
    attempts: int,
    label: str,
    use_follow_reading: bool,
) -> tuple[dict | None, dict]:
    history = []
    for attempt in range(1, max(1, int(attempts)) + 1):
        first = _pose(
            vision,
            world,
            samples=samples,
            timeout_s=timeout_s,
            use_follow_reading=bool(use_follow_reading),
        )
        time.sleep(max(0.0, float(settle_s)))
        second = _pose(
            vision,
            world,
            samples=samples,
            timeout_s=timeout_s,
            use_follow_reading=bool(use_follow_reading),
        )
        if isinstance(first, dict) and isinstance(second, dict):
            drift = _movement_delta(first, second)
            stable = bool(float(drift["vector_delta_mm"]) <= float(max_drift_mm))
            history.append(
                {
                    "attempt": int(attempt),
                    "first_pose": first,
                    "second_pose": second,
                    **drift,
                    "stable": bool(stable),
                }
            )
            if stable:
                return second, {"ok": True, "label": str(label), "attempts": int(attempt), "history": history}
        else:
            history.append(
                {
                    "attempt": int(attempt),
                    "first_pose": first,
                    "second_pose": second,
                    "stable": False,
                    "error": "pose_unavailable",
                }
            )
    return None, {"ok": False, "label": str(label), "attempts": int(attempts), "history": history}


def _run_constant_pwm_curve(
    robot: Robot,
    *,
    direction_key: str,
    strength: str,
    pwm: int,
    ticks: int,
    interval_ms: int,
) -> list[dict]:
    direction = DIRECTIONS[direction_key]
    strength_cfg = STRENGTHS[strength]
    records: list[dict] = []
    start_s = time.monotonic()
    for idx in range(max(1, int(ticks))):
        actions, skipped, left_pwm, right_pwm = _actions_for_curve(
            str(direction["cmd"]),
            pwm=int(pwm),
            step_zero_based=int(idx),
            skip_target=str(direction["skip_target"]),
            every_n=int(strength_cfg["every_n"]),
            phase_offset=int(strength_cfg["phase_offset"]),
            skip_pwm=int(strength_cfg["skip_pwm"]),
            mode=str(strength_cfg["mode"]),
        )
        send_result = robot.send_custom_actions_pwm(
            str(direction["cmd"]),
            actions,
            duration_ms=int(interval_ms),
        )
        records.append(
            {
                "tick": int(idx + 1),
                "t_ms": int(idx) * int(interval_ms),
                "duration_ms": int(interval_ms),
                "pwm": int(pwm),
                "left_pwm": int(left_pwm),
                "right_pwm": int(right_pwm),
                "skipped_inside_tread": bool(skipped),
                "wire_text": (send_result or {}).get("wire_text"),
                "send_result": send_result,
            }
        )
        next_start_s = float(start_s) + (float(idx + 1) * float(interval_ms) / 1000.0)
        remaining_s = float(next_start_s) - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(float(remaining_s))
    robot.stop()
    return records


def discover_curve_breakaway(
    *,
    label: str,
    candidate_pwms: tuple[int, ...],
    ticks: int,
    interval_ms: int,
    movement_threshold_mm: float,
    vision_mode: str,
    log_dir: Path,
    samples: int,
    observe_timeout_s: float,
    settle_s: float,
    min_safe_dist_mm: float,
    pre_stability_drift_mm: float,
    pre_stability_attempts: int,
    use_follow_vision: bool,
    max_wrong_direction_dist_delta_mm: float,
) -> dict:
    direction_key, strength = _parse_label(label)
    direction = DIRECTIONS[direction_key]
    strength_cfg = STRENGTHS[strength]
    robot = Robot()
    if bool(use_follow_vision):
        _set_game_profile("empty")
        vision = FollowBrickDetector(debug=False)
        set_tuning = getattr(vision, "set_runtime_tuning", None)
        if callable(set_tuning):
            set_tuning(**dict(CROWN_PROFILE_TUNING))
        _warmup(vision)
        _reset_follow_reading_history(vision, allow_large_dist_jump=True)
    else:
        vision = _build_vision(vision_mode)
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK
    rows: list[dict] = []
    try:
        print(
            f"[DUTY BREAKAWAY] {label}: {direction['summary']}; "
            f"{strength_cfg['mode']} every {strength_cfg['every_n']} tick(s), "
            f"{int(ticks)} x {int(interval_ms)}ms per candidate.",
            flush=True,
        )
        robot.stop()
        time.sleep(max(0.0, float(settle_s)))
        baseline, baseline_stability = _stable_pose(
            vision,
            world,
            samples=samples,
            timeout_s=observe_timeout_s,
            settle_s=float(settle_s),
            max_drift_mm=float(pre_stability_drift_mm),
            attempts=int(pre_stability_attempts),
            label="baseline",
            use_follow_reading=bool(use_follow_vision),
        )
        print(f"[DUTY BREAKAWAY] Baseline {_fmt_pose(baseline)}.", flush=True)
        if isinstance(baseline, dict) and float(baseline.get("dist_mm")) < float(min_safe_dist_mm):
            return {
                "ok": False,
                "label": str(label),
                "error": "too_close_for_forward_probe",
                "baseline": baseline,
                "baseline_stability": baseline_stability,
                "min_safe_dist_mm": float(min_safe_dist_mm),
                "rows": rows,
            }
        if baseline is None:
            return {
                "ok": False,
                "label": str(label),
                "error": "unstable_baseline_pose",
                "baseline_stability": baseline_stability,
                "rows": rows,
            }

        recommendation = None
        for pwm in candidate_pwms:
            pre_pose, pre_stability = _stable_pose(
                vision,
                world,
                samples=samples,
                timeout_s=observe_timeout_s,
                settle_s=float(settle_s),
                max_drift_mm=float(pre_stability_drift_mm),
                attempts=int(pre_stability_attempts),
                label=f"pwm_{int(pwm)}_pre",
                use_follow_reading=bool(use_follow_vision),
            )
            if pre_pose is None:
                rows.append(
                    {
                        "ok": False,
                        "pwm": int(pwm),
                        "error": "unstable_pre_pose",
                        "pre_stability": pre_stability,
                    }
                )
                print(
                    f"[DUTY BREAKAWAY] pwm={int(pwm)} pre pose unstable; refusing to move.",
                    flush=True,
                )
                break
            if float(pre_pose.get("dist_mm")) < float(min_safe_dist_mm):
                rows.append(
                    {
                        "ok": False,
                        "pwm": int(pwm),
                        "error": "too_close_for_forward_probe",
                        "pre_pose": pre_pose,
                        "pre_stability": pre_stability,
                    }
                )
                print(
                    f"[DUTY BREAKAWAY] pwm={int(pwm)} abort: too close before probe ({_fmt_pose(pre_pose)}).",
                    flush=True,
                )
                break

            tick_records = _run_constant_pwm_curve(
                robot,
                direction_key=direction_key,
                strength=strength,
                pwm=int(pwm),
                ticks=int(ticks),
                interval_ms=int(interval_ms),
            )
            time.sleep(max(0.0, float(settle_s)))
            if bool(use_follow_vision):
                _reset_follow_reading_history(vision, allow_large_dist_jump=False)
            post_pose, post_stability = _stable_pose(
                vision,
                world,
                samples=samples,
                timeout_s=observe_timeout_s,
                settle_s=float(settle_s),
                max_drift_mm=float(pre_stability_drift_mm),
                attempts=int(pre_stability_attempts),
                label=f"pwm_{int(pwm)}_post",
                use_follow_reading=bool(use_follow_vision),
            )
            if post_pose is None:
                row = {
                    "ok": False,
                    "pwm": int(pwm),
                    "power": float(pwm_to_power(int(pwm)) or 0.0),
                    "error": "unstable_post_pose",
                    "pre_pose": pre_pose,
                    "pre_stability": pre_stability,
                    "post_stability": post_stability,
                    "ticks": tick_records,
                }
                rows.append(row)
                print(
                    f"[DUTY BREAKAWAY] pwm={int(pwm)} post pose unstable; refusing to score.",
                    flush=True,
                )
                break

            delta = _movement_delta(pre_pose, post_pose)
            dist_delta = float(delta["dist_delta_mm"])
            max_wrong_sign = max(0.0, float(max_wrong_direction_dist_delta_mm))
            wrong_direction = (
                str(direction["cmd"]) == "f" and dist_delta > max_wrong_sign
            ) or (
                str(direction["cmd"]) == "b" and dist_delta < -max_wrong_sign
            )
            moved = bool(float(delta["vector_delta_mm"]) >= float(movement_threshold_mm))
            row = {
                "ok": not bool(wrong_direction),
                "label": str(label),
                "pwm": int(pwm),
                "power": float(pwm_to_power(int(pwm)) or 0.0),
                "ticks": tick_records,
                "pre_pose": pre_pose,
                "post_pose": post_pose,
                "pre_stability": pre_stability,
                "post_stability": post_stability,
                **delta,
                "movement_threshold_mm": float(movement_threshold_mm),
                "moved": bool(moved),
                "wrong_direction_or_model_switch": bool(wrong_direction),
            }
            rows.append(row)
            print(
                f"[DUTY BREAKAWAY] pwm={int(pwm):3d} {_fmt_pose(pre_pose)} -> {_fmt_pose(post_pose)} "
                f"delta dist={float(delta['dist_delta_mm']):+.1f} x={float(delta['x_delta_mm']):+.1f} "
                f"vector={float(delta['vector_delta_mm']):.1f} moved={bool(moved)}"
                f"{' WRONG-SIGN' if bool(wrong_direction) else ''}.",
                flush=True,
            )
            if wrong_direction:
                row["error"] = "wrong_direction_or_model_switch"
                break
            if moved:
                recommendation = row
                break
            time.sleep(0.25)

        result = {
            "ok": bool(recommendation is not None),
            "label": str(label),
            "direction_key": str(direction_key),
            "strength": str(strength),
            "candidate_pwms": [int(item) for item in candidate_pwms],
            "movement_threshold_mm": float(movement_threshold_mm),
            "ticks_per_candidate": int(ticks),
            "interval_ms": int(interval_ms),
            "recommended_pwm": None if recommendation is None else int(recommendation["pwm"]),
            "recommended_power": None if recommendation is None else float(recommendation["power"]),
            "baseline": baseline,
            "baseline_stability": baseline_stability,
            "use_follow_vision": bool(use_follow_vision),
            "rows": rows,
        }
        return result
    finally:
        try:
            robot.stop()
        finally:
            try:
                robot.close()
            except Exception:
                pass


def _candidate_pwms(value: str) -> tuple[int, ...]:
    items = []
    for chunk in str(value or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        items.append(max(0, min(255, int(round(float(token))))))
    return tuple(items) or tuple(DEFAULT_CANDIDATE_PWMS)


def _requested_labels(args: argparse.Namespace) -> tuple[str, ...]:
    if str(getattr(args, "labels", "") or "").strip():
        labels = tuple(
            item
            for item in (str(chunk).strip().lower() for chunk in str(args.labels).split(","))
            if item
        )
    elif str(getattr(args, "strength", "") or "").strip():
        labels = _labels_for_strength(str(args.strength))
    else:
        labels = (str(args.label or DEFAULT_LABEL).strip().lower(),)
    for label in labels:
        _parse_label(label)
    return labels


def _write_result_aliases(result: dict, *, label: str, log_dir: Path) -> tuple[Path, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = log_dir / f"discover_{str(label).strip().lower()}_{stamp}.json"
    alias_path = log_dir / f"discover_{str(label).strip().lower()}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    alias_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return out_path, alias_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None, help="Single curve label, e.g. fr_gentle.")
    parser.add_argument("--labels", default="", help="Comma-separated curve labels, e.g. fr_gentle,bl_gentle.")
    parser.add_argument("--strength", choices=tuple(STRENGTHS), help="Run all four direction permutations for one strength.")
    parser.add_argument("--candidate-pwms", default=",".join(str(item) for item in DEFAULT_CANDIDATE_PWMS))
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--interval-ms", type=int, default=INTERVAL_MS)
    parser.add_argument("--movement-threshold-mm", type=float, default=DEFAULT_MOVEMENT_THRESHOLD_MM)
    parser.add_argument("--vision", default=str(DEFAULT_VISION_MODE), choices=("cyan", "yolo", "leia", "aruco"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--observe-timeout-s", type=float, default=1.4)
    parser.add_argument("--settle-s", type=float, default=0.35)
    parser.add_argument("--min-safe-dist-mm", type=float, default=100.0)
    parser.add_argument("--pre-stability-drift-mm", type=float, default=25.0)
    parser.add_argument("--pre-stability-attempts", type=int, default=3)
    parser.add_argument("--raw-breakaway-vision", action="store_true", help="Use the old breakaway-helper vision path instead of the follow-game read path.")
    parser.add_argument("--max-wrong-direction-dist-delta-mm", type=float, default=10.0)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    labels = _requested_labels(args)
    results = []
    for index, label in enumerate(labels, start=1):
        print(f"[DUTY BREAKAWAY] === {index}/{len(labels)} {label} ===", flush=True)
        result = discover_curve_breakaway(
            label=str(label),
            candidate_pwms=_candidate_pwms(str(args.candidate_pwms)),
            ticks=int(args.ticks),
            interval_ms=int(args.interval_ms),
            movement_threshold_mm=float(args.movement_threshold_mm),
            vision_mode=str(args.vision),
            log_dir=Path(args.log_dir),
            samples=int(args.samples),
            observe_timeout_s=float(args.observe_timeout_s),
            settle_s=float(args.settle_s),
            min_safe_dist_mm=float(args.min_safe_dist_mm),
            pre_stability_drift_mm=float(args.pre_stability_drift_mm),
            pre_stability_attempts=int(args.pre_stability_attempts),
            use_follow_vision=not bool(args.raw_breakaway_vision),
            max_wrong_direction_dist_delta_mm=float(args.max_wrong_direction_dist_delta_mm),
        )
        results.append(result)
        out_path, alias_path = _write_result_aliases(result, label=str(label), log_dir=Path(args.log_dir))
        print(f"[DUTY BREAKAWAY] log saved: {out_path}", flush=True)
        print(f"[DUTY BREAKAWAY] stable alias: {alias_path}", flush=True)
        if result.get("recommended_pwm") is not None:
            print(
                f"[DUTY BREAKAWAY] recommendation: {label} breakaway pwm={int(result['recommended_pwm'])} "
                f"power={float(result['recommended_power']):.3f}.",
                flush=True,
            )
        else:
            print(f"[DUTY BREAKAWAY] no breakaway found for {label}.", flush=True)
        time.sleep(0.5)

    if len(labels) > 1:
        combined_label = str(args.strength or "multi").strip().lower()
        combined = {
            "ok": all(bool(item.get("ok")) for item in results),
            "labels": list(labels),
            "results": results,
        }
        out_path, alias_path = _write_result_aliases(
            combined,
            label=f"{combined_label}_all",
            log_dir=Path(args.log_dir),
        )
        print(f"[DUTY BREAKAWAY] combined log saved: {out_path}", flush=True)
        print(f"[DUTY BREAKAWAY] combined stable alias: {alias_path}", flush=True)

    return 0 if all(result.get("recommended_pwm") is not None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
