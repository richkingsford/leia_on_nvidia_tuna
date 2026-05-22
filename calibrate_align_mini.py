#!/usr/bin/env python3
"""Compatibility mini-align calibration helpers for focused unit tests."""

from __future__ import annotations

from collections import deque
import statistics
import time

from helper_calibrate_telemetry import (
    telemetry_average_smoothed_frames,
    telemetry_latest_unique_smoothed_frames,
)
from telemetry_process import lite_gate_unique_frames, send_robot_command_pwm, update_world_from_vision
from telemetry_robot import speed_power_pwm_for_motion_intensity


OBSERVE_SLEEP_S = 0.02
OBSERVE_SAMPLES = 3
POST_ACT_SETTLE_S = 0.08
POST_ACT_MAX_WAIT_S = 1.0

Y_AXIS_MOTION_PROFILE_FIXED_PWM_DURATION = "fixed_pwm_duration"
Y_AXIS_LARGE_OBSERVE_MODE = Y_AXIS_MOTION_PROFILE_FIXED_PWM_DURATION
Y_AXIS_PROGRESS_FAMILIES_DEFAULT = ("large", "medium", "small")


def _num(value, fallback: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _median(values) -> float | None:
    cleaned = [_num(value) for value in values]
    cleaned = [float(value) for value in cleaned if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def _build_trial_gap_families(*, min_err_mm: float, max_err_mm: float, target_tol_mm: float) -> list[dict]:
    min_err = max(0.0, float(min_err_mm))
    max_err = max(min_err, float(max_err_mm))
    small_max = min(max_err, max(min_err, float(target_tol_mm) * 2.5))
    medium_max = min(max_err, max(small_max, small_max * 1.5))
    ranges = (
        ("large", medium_max, max_err),
        ("medium", small_max, medium_max),
        ("small", min_err, small_max),
    )
    out = []
    for family, lo, hi in ranges:
        out.append(
            {
                "family": family,
                "range_min_mm": float(lo),
                "range_max_mm": float(hi),
                "target_abs_mm": (float(lo) + float(hi)) / 2.0,
            }
        )
    return out


def _classify_gap_family(abs_gap_mm: float, families: list[dict]) -> dict | None:
    gap = abs(float(abs_gap_mm))
    for row in families or []:
        if gap >= float(row.get("range_min_mm", 0.0)) and gap <= float(row.get("range_max_mm", 0.0)):
            return dict(row)
    return None


def _trial_hit_tolerance_mm(*, axis: str, axis_tol_mm: float, family: str) -> float:
    if str(axis).strip().lower() == "y" and str(family).strip().lower() in {"medium", "large"}:
        return 4.0
    if str(axis).strip().lower() == "y":
        return max(2.0, float(axis_tol_mm))
    return float(axis_tol_mm)


def _large_family_observation_profile(*, axis: str, family: str) -> dict | None:
    if str(axis).strip().lower() != "y" or str(family).strip().lower() != "large":
        return None
    return {
        "mode": Y_AXIS_LARGE_OBSERVE_MODE,
        "motion_intensity_pct": 100.0,
        "duration_override_ms": 250,
        "duration_model_ms": 250,
    }


def _results_trial_summary_line(family: str, row: dict) -> str:
    trials = int(row.get("trials", 0) or 0)
    success_count = int(row.get("success_count", 0) or 0)
    success_total = int(row.get("success_total", trials) or 0)
    hit_count = int(row.get("trial_hit_count", 0) or 0)
    hit_total = int(row.get("trial_hit_total", trials) or 0)
    success_pct = 0.0 if success_total <= 0 else (100.0 * success_count / success_total)
    hit_pct = 0.0 if hit_total <= 0 else (100.0 * hit_count / hit_total)
    return (
        f"{family}: success={success_pct:.1f}% ({success_count}/{success_total}) "
        f"one_shot_hit={hit_pct:.1f}% ({hit_count}/{hit_total}) trials={trials} "
        f"median_miss={float(row.get('median_trial_target_miss_mm', 0.0)):.2f}mm "
        f"median_cmd_delta={float(row.get('median_trial_cmd_delta_mm', 0.0)):.2f}mm"
    )


def _parse_progress_families(raw) -> tuple[str, ...]:
    text = str(raw or "").strip().lower()
    if text in {"", "default"}:
        return Y_AXIS_PROGRESS_FAMILIES_DEFAULT
    if text in {"off", "none", "false", "0"}:
        return ()
    values = tuple(part.strip() for part in text.split(",") if part.strip() in {"large", "medium", "small"})
    return values or Y_AXIS_PROGRESS_FAMILIES_DEFAULT


def _y_cmd_delta_mm(cmd: str, before_y_mm: float, after_y_mm: float) -> float:
    if str(cmd).strip().lower() == "u":
        return float(before_y_mm) - float(after_y_mm)
    return float(after_y_mm) - float(before_y_mm)


def _send_axis_command(**kwargs):
    robot = kwargs.get("robot")
    cmd = kwargs.get("cmd")
    intensity = kwargs.get("intensity", kwargs.get("curve_intensity_pct", 1.0))
    _power, pwm, _score, duration_ms, _eff = speed_power_pwm_for_motion_intensity(cmd, intensity)
    if robot is not None:
        send_robot_command_pwm(robot, cmd, pwm, duration_ms=duration_ms)
    return {"cmd": cmd, "pwm": pwm, "duration_ms": duration_ms}


def _send_fixed_score_axis_command(**kwargs):
    return _send_axis_command(**kwargs)


def _pose_axis_value(pose: dict, axis: str) -> float | None:
    key = "offset_y" if str(axis).strip().lower() == "y" else "offset_x"
    return _num((pose or {}).get(key))


def _position_y_into_target_families(
    *,
    vision,
    world,
    robot,
    step_state,
    current_pose: dict,
    axis_target_mm: float,
    axis_sign: float,
    families: list[dict],
    target_families,
    max_acts: int,
    allowed_y_cmds=None,
):
    current = dict(current_pose or {})
    start_gap = abs(float(_pose_axis_value(current, "y") or 0.0) - float(axis_target_mm))
    start_family = _classify_gap_family(start_gap, families)
    target_set = {str(item).strip().lower() for item in (target_families or ())}
    allowed = None if allowed_y_cmds is None else {str(cmd).strip().lower() for cmd in allowed_y_cmds}
    meta = {
        "status": "not_positioned",
        "acts": 0,
        "start_family": None if start_family is None else start_family.get("family"),
        "end_family": None,
        "steps": [],
    }
    for _idx in range(max(0, int(max_acts))):
        y_now = float(_pose_axis_value(current, "y") or 0.0)
        err = (y_now - float(axis_target_mm)) * float(axis_sign)
        cmd = "u" if err > 0.0 else "d"
        if allowed is not None and cmd not in allowed:
            break
        sent = _send_fixed_score_axis_command(
            vision=vision,
            world=world,
            robot=robot,
            step_state=step_state,
            axis="y",
            cmd=cmd,
            duration_override_ms=250,
        )
        meta["acts"] += 1
        _observe_after_action(float((sent or {}).get("duration_ms", 0) or 0) / 1000.0)
        pose = _read_pose(vision, world)
        if not isinstance(pose, dict):
            break
        current = pose
        gap = abs(float(_pose_axis_value(current, "y") or 0.0) - float(axis_target_mm))
        family = _classify_gap_family(gap, families)
        family_name = None if family is None else str(family.get("family"))
        meta["steps"].append({"cmd": cmd, "post_family": family_name})
        meta["end_family"] = family_name
        if family_name in target_set:
            meta["status"] = "positioned"
            return current, meta
    return current, meta


def _y_confirmed_window_metrics(rows: list[dict], *, hit_tol_mm: float) -> dict | None:
    if not rows:
        return None
    median_pre = _median(row.get("pre_abs_err_mm") for row in rows)
    median_delta = _median(row.get("trial_cmd_delta_mm") for row in rows)
    hit_rate = sum(1 for row in rows if bool(row.get("trial_hit_success"))) / float(len(rows))
    return {
        "median_pre_abs_err_mm": float(median_pre or 0.0),
        "median_trial_cmd_delta_mm": float(median_delta or 0.0),
        "coverage_ratio": 0.0 if not median_pre else float(median_delta or 0.0) / float(median_pre),
        "hit_rate": float(hit_rate),
        "hit_tol_mm": float(hit_tol_mm),
    }


def _tune_y_family_scale(current_scale: float, family: str, metrics: dict) -> tuple[float, str]:
    coverage = float(metrics.get("coverage_ratio", 0.0) or 0.0)
    hit_rate = float(metrics.get("hit_rate", 0.0) or 0.0)
    if str(family).strip().lower() == "large" and (coverage < 0.25 or hit_rate < 0.25):
        return float(current_scale) * 1.35, "confirmed_underpowered_hard"
    if coverage < 0.5 or hit_rate < 0.5:
        return float(current_scale) * 1.15, "confirmed_underpowered"
    return float(current_scale), "confirmed_ok"


def _resolve_axis_motion_profile(
    *,
    axis: str,
    cmd: str,
    curve_intensity_pct: float,
    y_motion_profile: str | None = None,
    y_fixed_motion_intensity_pct: float | None = None,
) -> dict:
    _power, pwm, score, duration_ms, effective = speed_power_pwm_for_motion_intensity(cmd, curve_intensity_pct)
    motion_intensity = float(effective)
    if str(axis).strip().lower() == "y" and y_motion_profile == Y_AXIS_MOTION_PROFILE_FIXED_PWM_DURATION:
        fixed_intensity = float(y_fixed_motion_intensity_pct)
        _fixed_power, pwm, score, _fixed_duration, fixed_effective = speed_power_pwm_for_motion_intensity(cmd, fixed_intensity)
        motion_intensity = float(fixed_effective)
    return {
        "mode": y_motion_profile or "curve",
        "cmd": cmd,
        "pwm": int(pwm),
        "score": int(score),
        "curve_intensity_pct": float(curve_intensity_pct),
        "motion_intensity_pct": float(motion_intensity),
        "duration_override_ms": int(duration_ms),
        "duration_model_ms": int(duration_ms),
    }


def _tuple_pose(result) -> dict | None:
    if not (isinstance(result, tuple) and len(result) >= 6 and bool(result[0])):
        return None
    return {
        "visible": True,
        "angle": _num(result[1], 0.0),
        "dist": _num(result[2], 0.0),
        "offset_x": _num(result[3], 0.0),
        "x_axis": _num(result[3], 0.0),
        "confidence": _num(result[4], 0.0),
        "offset_y": _num(result[5], 0.0),
        "y_axis": _num(result[5], 0.0),
    }


def _read_pose(vision, world, *, samples: int = OBSERVE_SAMPLES, timeout_s: float = 1.0) -> dict | None:
    if hasattr(world, "brick") and hasattr(world, "process_rules"):
        try:
            update_world_from_vision(world, vision)
            if lite_gate_unique_frames(world) > 0:
                frames = telemetry_latest_unique_smoothed_frames(world, max(1, int(samples)))
                averaged = telemetry_average_smoothed_frames(frames)
                if isinstance(averaged, dict) and bool(averaged.get("visible")):
                    pose = dict(averaged)
                    pose.setdefault("offset_x", pose.get("x_axis"))
                    pose.setdefault("offset_y", pose.get("y_axis"))
                    pose["pose_source"] = "lite_smoothed"
                    return pose
        except Exception:
            pass

    deadline = time.time() + max(0.0, float(timeout_s))
    rows = []
    while len(rows) < max(1, int(samples)) and time.time() <= deadline:
        pose = _tuple_pose(vision.read())
        if pose is not None:
            rows.append(pose)
        if len(rows) < max(1, int(samples)):
            time.sleep(float(OBSERVE_SLEEP_S))
    if not rows:
        return None
    pose = dict(rows[-1])
    pose["offset_x"] = float(_median(row.get("offset_x") for row in rows) or 0.0)
    pose["x_axis"] = pose["offset_x"]
    pose["offset_y"] = float(_median(row.get("offset_y") for row in rows) or 0.0)
    pose["y_axis"] = pose["offset_y"]
    updater = getattr(world, "update_vision", None)
    if callable(updater):
        updater(pose.get("visible"), pose.get("angle"), pose.get("dist"), pose.get("offset_x"), pose.get("confidence"))
    return pose


def _observe_after_action(duration_s: float) -> float:
    now = time.time()
    return now + min(float(POST_ACT_MAX_WAIT_S), max(float(POST_ACT_SETTLE_S), float(duration_s)))


def _maybe_confirm_y_post_pose(
    *,
    vision,
    world,
    axis: str,
    axis_target_mm: float,
    axis_sign: float,
    pre_abs_err: float,
    pose_after: dict,
) -> tuple[dict, bool]:
    time.sleep(float(OBSERVE_SLEEP_S))
    confirmed = _read_pose(vision, world)
    if not isinstance(confirmed, dict):
        return pose_after, False
    confirmed_err = abs((float(_pose_axis_value(confirmed, axis) or 0.0) - float(axis_target_mm)) * float(axis_sign))
    initial_err = abs((float(_pose_axis_value(pose_after, axis) or 0.0) - float(axis_target_mm)) * float(axis_sign))
    if confirmed_err <= min(float(pre_abs_err), float(initial_err)):
        return confirmed, True
    return pose_after, False


def _axis_sign_evidence(
    *,
    axis: str,
    cmd: str,
    axis_before_mm: float,
    axis_after_mm: float,
    axis_target_mm: float,
    candidate_sign: float,
) -> int:
    before_err = abs((float(axis_before_mm) * float(candidate_sign)) - float(axis_target_mm))
    after_err = abs((float(axis_after_mm) * float(candidate_sign)) - float(axis_target_mm))
    actual_delta = float(axis_after_mm) - float(axis_before_mm)
    expected_delta_sign = 1.0 if str(cmd).strip().lower() == "d" else -1.0
    expected_delta_sign *= 1.0 if float(candidate_sign) >= 0.0 else -1.0
    return 1 if after_err < before_err and (actual_delta * expected_delta_sign) > 0.0 else -1


def _probe_axis_sign(
    *,
    vision,
    world,
    robot,
    step_state,
    axis: str,
    axis_target_mm: float,
    axis_sign: float,
    intensity: float,
) -> tuple[float, str]:
    before = _read_pose(vision, world, samples=1)
    if not isinstance(before, dict):
        return float(axis_sign), "probe_no_before"
    cmd = "d" if (float(_pose_axis_value(before, axis) or 0.0) - float(axis_target_mm)) < 0.0 else "u"
    sent = _send_axis_command(
        vision=vision,
        world=world,
        robot=robot,
        step_state=step_state,
        axis=axis,
        cmd=cmd,
        intensity=float(intensity),
    )
    _observe_after_action(float((sent or {}).get("duration_ms", 0) or 0) / 1000.0)
    after = _read_pose(vision, world, samples=1)
    if not isinstance(after, dict):
        return float(axis_sign), "probe_no_after"
    current = _axis_sign_evidence(
        axis=axis,
        cmd=cmd,
        axis_before_mm=float(_pose_axis_value(before, axis) or 0.0),
        axis_after_mm=float(_pose_axis_value(after, axis) or 0.0),
        axis_target_mm=float(axis_target_mm),
        candidate_sign=float(axis_sign),
    )
    alternate_sign = -float(axis_sign)
    alternate = _axis_sign_evidence(
        axis=axis,
        cmd=cmd,
        axis_before_mm=float(_pose_axis_value(before, axis) or 0.0),
        axis_after_mm=float(_pose_axis_value(after, axis) or 0.0),
        axis_target_mm=float(axis_target_mm),
        candidate_sign=alternate_sign,
    )
    return (alternate_sign, "probe_match") if alternate > current else (float(axis_sign), "probe_match")


def _symmetrize_curve_if_needed(candidate: dict, samples: list[dict], *, axis: str) -> dict:
    if str(axis).strip().lower() == "y":
        return candidate
    return candidate


def _recover_visibility(**kwargs):
    return None


def _attempt_recovery(*, vision, world, robot, step_state, recent_acts: deque):
    pose = _read_pose(vision, world)
    if isinstance(pose, dict):
        return pose
    pose = _recover_visibility(
        vision=vision,
        world=world,
        robot=robot,
        step_state=step_state,
        recent_acts=recent_acts,
    )
    if isinstance(pose, dict):
        return pose
    return _scan_recover_visibility(
        vision=vision,
        world=world,
        robot=robot,
        step_state=step_state,
        recent_acts=recent_acts,
    )


def _scan_recover_visibility(*, vision, world, robot, step_state, recent_acts: deque, max_acts: int = 3):
    last = recent_acts[-1] if recent_acts else {}
    last_cmd = str(last.get("cmd") or "d").strip().lower()
    cmd = "u" if last_cmd == "d" else "d"
    intensity = max(1.0, float(last.get("turn_intensity_pct", last.get("intensity", 20.0)) or 20.0) * 0.85)
    for _idx in range(max(0, int(max_acts))):
        sent = _send_axis_command(
            vision=vision,
            world=world,
            robot=robot,
            step_state=step_state,
            axis="y",
            cmd=cmd,
            intensity=float(intensity),
        )
        _observe_after_action(float((sent or {}).get("duration_ms", 0) or 0) / 1000.0)
        pose = _read_pose(vision, world)
        if isinstance(pose, dict):
            return pose
    return None
