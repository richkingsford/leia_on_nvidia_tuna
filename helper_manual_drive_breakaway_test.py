#!/usr/bin/env python3
"""Manual low-speed motion test for drive and turn hotkeys."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from helper_manual_drive_assist import execute_manual_drive_assist_plan, load_manual_drive_assist_config
from helper_vision_config import (
    VISION_MODE_ARUCO,
    VISION_MODE_CYAN,
    active_vision_mode as world_model_active_vision_mode,
    normalize_vision_mode,
)
from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from helper_vision_leia import LeiaVision
from telemetry_process import send_robot_command
from telemetry_robot import StepState, WorldModel

try:
    from helper_brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_drive_breakaway_test.json"
TARGET_HOTKEYS_DRIVE_DEFAULT = ("r", "f")
TARGET_HOTKEYS_TURN_DEFAULT = ("q", "e")
TARGET_HOTKEYS_ALL_DEFAULT = TARGET_HOTKEYS_DRIVE_DEFAULT + TARGET_HOTKEYS_TURN_DEFAULT
HOTKEY_SPECS = {
    "r": {
        "cmd": "f",
        "metric_key": "dist_mm",
        "metric_label": "dist",
        "expected_effect": "decrease",
        "display_label": "R/FWD",
    },
    "f": {
        "cmd": "b",
        "metric_key": "dist_mm",
        "metric_label": "dist",
        "expected_effect": "increase",
        "display_label": "F/BACK",
    },
    "q": {
        "cmd": "l",
        "metric_key": "x_axis_mm",
        "metric_label": "x_axis",
        "expected_effect": "increase",
        "display_label": "Q/LEFT",
    },
    "e": {
        "cmd": "r",
        "metric_key": "x_axis_mm",
        "metric_label": "x_axis",
        "expected_effect": "decrease",
        "display_label": "E/RIGHT",
    },
}
DEFAULT_START_SCORE = 1
DEFAULT_MAX_SCORE = 6
DEFAULT_REPEATS_PER_SCORE = 5
DEFAULT_MOVEMENT_THRESHOLD_MM = 0.10
DEFAULT_REQUIRED_SUCCESS_RATIO = 0.80
DEFAULT_PAUSE_BETWEEN_PULSES_MS = 500
DEFAULT_USE_MANUAL_ASSIST = True
OBSERVE_SAMPLE_FRAMES = 3
OBSERVE_TIMEOUT_S = 1.5
OBSERVE_SLEEP_S = 0.02
POST_ACT_SETTLE_S = 0.12
RECENTER_TOLERANCE_DIST_MM = 5.0
RECENTER_TOLERANCE_X_AXIS_MM = 5.0
RECENTER_TIMEOUT_S = 20.0
RECENTER_OBSERVE_SAMPLES = 3
RECENTER_OBSERVE_TIMEOUT_S = 1.2
DEFAULT_VISION_MODE = normalize_vision_mode(
    world_model_active_vision_mode(),
    fallback=VISION_MODE_CYAN,
)


def _clamp_score(value, default: int) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = int(default)
    return int(max(1, min(100, int(score))))


def _clamp_positive_int(value, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(default)
    return int(max(int(minimum), int(parsed)))


def _collect_pose_samples(vision, world, *, samples=OBSERVE_SAMPLE_FRAMES, timeout_s=OBSERVE_TIMEOUT_S) -> list[dict]:
    values = []
    started = time.time()
    requested = max(1, int(samples))
    while len(values) < requested and (time.time() - started) < float(timeout_s):
        found, angle, dist, offset_x, conf, cam_h, above, below = vision.read()
        world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
        if found:
            try:
                values.append(
                    {
                        "dist_mm": float(dist),
                        "x_axis_mm": float(offset_x),
                    }
                )
            except (TypeError, ValueError):
                pass
        time.sleep(float(OBSERVE_SLEEP_S))
    return values


def _median_pose_or_none(values: list[dict]) -> dict | None:
    if not values:
        return None
    dist_values = [float(item.get("dist_mm")) for item in values if isinstance(item, dict) and item.get("dist_mm") is not None]
    x_values = [float(item.get("x_axis_mm")) for item in values if isinstance(item, dict) and item.get("x_axis_mm") is not None]
    if not dist_values or not x_values:
        return None
    return {
        "dist_mm": float(statistics.median(dist_values)),
        "x_axis_mm": float(statistics.median(x_values)),
    }


def _format_pose_text(pose: dict | None) -> str:
    if not isinstance(pose, dict):
        return "dist=unknown x_axis=unknown"
    try:
        dist_text = f"{float(pose.get('dist_mm')):.2f}mm"
    except (TypeError, ValueError):
        dist_text = "unknown"
    try:
        x_text = f"{float(pose.get('x_axis_mm')):+.2f}mm"
    except (TypeError, ValueError):
        x_text = "unknown"
    return f"dist={dist_text} x_axis={x_text}"


def _recenter_status(
    pose: dict | None,
    baseline_pose: dict | None,
    *,
    tolerance_dist_mm: float,
    tolerance_x_axis_mm: float,
) -> dict:
    if not isinstance(pose, dict) or not isinstance(baseline_pose, dict):
        return {
            "ok": False,
            "dist_error_mm": None,
            "x_axis_error_mm": None,
        }
    try:
        dist_error_mm = float(pose.get("dist_mm")) - float(baseline_pose.get("dist_mm"))
        x_axis_error_mm = float(pose.get("x_axis_mm")) - float(baseline_pose.get("x_axis_mm"))
    except (TypeError, ValueError):
        return {
            "ok": False,
            "dist_error_mm": None,
            "x_axis_error_mm": None,
        }
    return {
        "ok": bool(
            abs(float(dist_error_mm)) <= float(tolerance_dist_mm)
            and abs(float(x_axis_error_mm)) <= float(tolerance_x_axis_mm)
        ),
        "dist_error_mm": float(dist_error_mm),
        "x_axis_error_mm": float(x_axis_error_mm),
    }


def _hotkey_spec(hotkey: str | None) -> dict | None:
    key = str(hotkey or "").strip().lower()
    raw = HOTKEY_SPECS.get(key)
    if not isinstance(raw, dict):
        return None
    spec = dict(raw)
    spec["hotkey"] = key
    return spec


def _resolve_requested_hotkeys(selection: str | None) -> tuple[str, ...]:
    token = str(selection or "").strip().lower()
    if not token or token in {"all", "both", "4", "four"}:
        return tuple(TARGET_HOTKEYS_ALL_DEFAULT)
    if token in {"drive", "d", "fb", "f/b"}:
        return tuple(TARGET_HOTKEYS_DRIVE_DEFAULT)
    if token in {"turn", "t", "lr", "l/r"}:
        return tuple(TARGET_HOTKEYS_TURN_DEFAULT)
    if token in {"r", "forward", "fwd"}:
        return ("r",)
    if token in {"f", "backward", "back"}:
        return ("f",)
    if token in {"q", "left", "l"}:
        return ("q",)
    if token in {"e", "right"}:
        return ("e",)
    parsed = tuple(
        item
        for item in (str(chunk).strip().lower() for chunk in str(token).split(","))
        if item in HOTKEY_SPECS
    )
    if parsed:
        return parsed
    return tuple(TARGET_HOTKEYS_ALL_DEFAULT)


def _movement_effect_from_raw_delta(raw_delta_mm: float, *, threshold_mm: float) -> str:
    raw_delta = float(raw_delta_mm)
    if abs(raw_delta) < max(0.0, float(threshold_mm)):
        return "no_change"
    return "increase" if raw_delta > 0.0 else "decrease"


def _movement_flags_for_hotkey(
    hotkey: str,
    raw_delta_mm: float,
    *,
    threshold_mm: float,
) -> dict:
    spec = _hotkey_spec(hotkey) or {}
    observed_effect = _movement_effect_from_raw_delta(float(raw_delta_mm), threshold_mm=float(threshold_mm))
    moved = bool(observed_effect != "no_change")
    expected_effect = str(spec.get("expected_effect") or "")
    direction_ok = bool(moved and observed_effect == expected_effect)
    return {
        "observed_effect": str(observed_effect),
        "expected_effect": expected_effect,
        "moved": bool(moved),
        "direction_ok": bool(direction_ok),
        "passes": bool(direction_ok),
    }


def _default_duration_ms_for_hotkey(hotkey: str) -> int:
    row = telemetry_robot_module.HOTKEY_SPEED_SCORES.get(str(hotkey).strip().lower())
    if isinstance(row, dict):
        try:
            duration_ms = int(round(float(row.get("duration_ms"))))
        except (TypeError, ValueError):
            duration_ms = None
        if duration_ms is not None and duration_ms > 0:
            return int(duration_ms)
    spec = _hotkey_spec(hotkey) or {}
    cmd = str(spec.get("cmd") or "f")
    _power, _pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, 1)
    return int(max(1, int(duration_ms or 250)))


def _run_recenter_checkpoint(
    *,
    vision,
    world,
    baseline_pose: dict | None,
    next_hotkey: str,
    timeout_s: float = RECENTER_TIMEOUT_S,
    tolerance_dist_mm: float = RECENTER_TOLERANCE_DIST_MM,
    tolerance_x_axis_mm: float = RECENTER_TOLERANCE_X_AXIS_MM,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    spec = _hotkey_spec(next_hotkey) or {}
    display_label = str(spec.get("display_label") or str(next_hotkey or "").upper())
    if not isinstance(baseline_pose, dict):
        logger(f"[BREAKAWAY TEST] Recenter checkpoint before {display_label}: skipped (baseline unavailable).")
        return {
            "ok": False,
            "skipped": True,
            "reason": "baseline_unavailable",
        }

    logger(
        "[BREAKAWAY TEST] Recenter checkpoint: "
        f"return near {_format_pose_text(baseline_pose)} before {display_label}. "
        f"Window: +/-{float(tolerance_dist_mm):.1f}mm dist, +/-{float(tolerance_x_axis_mm):.1f}mm x_axis."
    )
    started = time.time()
    last_pose = None
    last_status = None
    while (time.time() - started) < float(timeout_s):
        pose_samples = _collect_pose_samples(
            vision,
            world,
            samples=int(RECENTER_OBSERVE_SAMPLES),
            timeout_s=float(RECENTER_OBSERVE_TIMEOUT_S),
        )
        pose = _median_pose_or_none(pose_samples)
        status = _recenter_status(
            pose,
            baseline_pose,
            tolerance_dist_mm=float(tolerance_dist_mm),
            tolerance_x_axis_mm=float(tolerance_x_axis_mm),
        )
        last_pose = pose
        last_status = status
        if bool(status.get("ok")):
            logger(
                "[BREAKAWAY TEST] Recenter checkpoint passed: "
                f"{_format_pose_text(pose)} "
                f"(dist_err={float(status.get('dist_error_mm') or 0.0):+.2f}mm, "
                f"x_err={float(status.get('x_axis_error_mm') or 0.0):+.2f}mm)."
            )
            return {
                "ok": True,
                "skipped": False,
                "pose": dict(pose or {}),
                "dist_error_mm": float(status.get("dist_error_mm") or 0.0),
                "x_axis_error_mm": float(status.get("x_axis_error_mm") or 0.0),
                "seconds": float(max(0.0, time.time() - started)),
            }

    logger(
        "[BREAKAWAY TEST] Recenter checkpoint timed out: "
        f"current {_format_pose_text(last_pose)} "
        f"(dist_err={float((last_status or {}).get('dist_error_mm') or 0.0):+.2f}mm, "
        f"x_err={float((last_status or {}).get('x_axis_error_mm') or 0.0):+.2f}mm). "
        "Continuing anyway."
    )
    return {
        "ok": False,
        "skipped": False,
        "timed_out": True,
        "pose": dict(last_pose or {}),
        "dist_error_mm": None if not isinstance(last_status, dict) else last_status.get("dist_error_mm"),
        "x_axis_error_mm": None if not isinstance(last_status, dict) else last_status.get("x_axis_error_mm"),
        "seconds": float(max(0.0, time.time() - started)),
    }


def _raw_trial(
    *,
    robot,
    vision,
    world,
    hotkey: str,
    score: int,
    duration_ms: int | None,
    movement_threshold_mm: float,
    use_manual_assist: bool,
) -> dict:
    spec = _hotkey_spec(hotkey)
    if not isinstance(spec, dict):
        return {
            "ok": False,
            "error": "unsupported_hotkey",
        }

    metric_key = str(spec.get("metric_key") or "dist_mm")
    cmd = str(spec.get("cmd") or "")
    pre_vals = _collect_pose_samples(vision, world)
    pre_pose = _median_pose_or_none(pre_vals)
    if pre_pose is None:
        return {
            "ok": False,
            "error": "pre_pose_unavailable",
        }
    pre_metric_mm = pre_pose.get(metric_key)
    if pre_metric_mm is None:
        return {
            "ok": False,
            "error": "pre_metric_unavailable",
        }

    if duration_ms is None:
        duration_ms = _default_duration_ms_for_hotkey(hotkey)
    try:
        duration_used_ms = max(1, int(round(float(duration_ms))))
    except (TypeError, ValueError):
        duration_used_ms = _default_duration_ms_for_hotkey(hotkey)

    send_result = None
    if bool(use_manual_assist):
        send_result = execute_manual_drive_assist_plan(
            robot=robot,
            world=world,
            step_state=StepState.ALIGN_BRICK,
            hotkey=hotkey,
            cmd=cmd,
            score=int(score),
            hold_duration_ms=int(duration_used_ms),
        )
    if not isinstance(send_result, dict):
        send_result = send_robot_command(
            robot,
            world,
            StepState.ALIGN_BRICK,
            cmd,
            speed=0.0,
            speed_score=int(score),
            duration_override_ms=int(duration_used_ms),
            auto_mode=False,
            ease_in_out_enabled=False,
        )
    try:
        send_duration_ms = int(round(float((send_result or {}).get("duration_ms") or duration_used_ms)))
    except (TypeError, ValueError):
        send_duration_ms = int(duration_used_ms)
    wait_s = max(float(POST_ACT_SETTLE_S), (float(send_duration_ms) / 1000.0) + 0.02)
    time.sleep(float(wait_s))

    post_vals = _collect_pose_samples(vision, world)
    post_pose = _median_pose_or_none(post_vals)
    if post_pose is None:
        return {
            "ok": False,
            "error": "post_pose_unavailable",
            "pre_metric_mm": float(pre_metric_mm),
        }
    post_metric_mm = post_pose.get(metric_key)
    if post_metric_mm is None:
        return {
            "ok": False,
            "error": "post_metric_unavailable",
            "pre_metric_mm": float(pre_metric_mm),
        }

    raw_delta_mm = float(post_metric_mm) - float(pre_metric_mm)
    abs_delta_mm = abs(float(raw_delta_mm))
    flags = _movement_flags_for_hotkey(
        str(hotkey),
        float(raw_delta_mm),
        threshold_mm=float(movement_threshold_mm),
    )
    return {
        "ok": True,
        "hotkey": str(hotkey),
        "cmd": str(cmd),
        "score": int(score),
        "duration_ms": int(send_duration_ms),
        "cmd_sent": (send_result or {}).get("cmd_sent"),
        "pwm": (send_result or {}).get("pwm"),
        "power": (send_result or {}).get("power"),
        "assist_applied": bool((send_result or {}).get("manual_drive_assist")),
        "manual_drive_assist": (send_result or {}).get("manual_drive_assist"),
        "segments": (send_result or {}).get("segments"),
        "metric_key": str(metric_key),
        "metric_label": str(spec.get("metric_label") or metric_key),
        "display_label": str(spec.get("display_label") or str(hotkey).upper()),
        "expected_effect": str(flags["expected_effect"]),
        "observed_effect": str(flags["observed_effect"]),
        "pre_metric_mm": float(pre_metric_mm),
        "post_metric_mm": float(post_metric_mm),
        "pre_dist_mm": float(pre_pose.get("dist_mm")),
        "post_dist_mm": float(post_pose.get("dist_mm")),
        "pre_x_axis_mm": float(pre_pose.get("x_axis_mm")),
        "post_x_axis_mm": float(post_pose.get("x_axis_mm")),
        "raw_delta_mm": float(raw_delta_mm),
        "abs_delta_mm": float(abs_delta_mm),
        "moved": bool(flags["moved"]),
        "direction_ok": bool(flags["direction_ok"]),
        "passes": bool(flags["passes"]),
    }


def summarize_breakaway_results(
    rows: list[dict],
    *,
    repeats_per_score: int,
    required_success_ratio: float,
) -> dict:
    grouped: dict[str, dict[int, list[dict]]] = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        hotkey = str(row.get("hotkey") or "").strip().lower()
        try:
            score = int(row.get("score"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(hotkey, {}).setdefault(int(score), []).append(row)

    by_hotkey = {}
    required_count = max(1, int(round(float(repeats_per_score) * float(required_success_ratio))))
    for hotkey, by_score in grouped.items():
        score_summaries = []
        recommended_score = None
        for score in sorted(by_score.keys()):
            trials = list(by_score.get(score) or [])
            moved_count = sum(1 for row in trials if bool(row.get("moved")))
            direction_ok_count = sum(
                1 for row in trials if bool(row.get("direction_ok", row.get("passes", False)))
            )
            success_count = sum(
                1 for row in trials if bool(row.get("passes", row.get("moved", False)))
            )
            abs_deltas = [float(row.get("abs_delta_mm") or 0.0) for row in trials if row.get("ok")]
            spec = next((row for row in trials if isinstance(row, dict)), {}) or {}
            median_abs_delta = float(statistics.median(abs_deltas)) if abs_deltas else None
            summary = {
                "score": int(score),
                "trial_count": len(trials),
                "metric_label": str(spec.get("metric_label") or ""),
                "expected_effect": str(spec.get("expected_effect") or ""),
                "moved_count": int(moved_count),
                "direction_ok_count": int(direction_ok_count),
                "success_count": int(success_count),
                "required_success_count": int(required_count),
                "passes": bool(int(success_count) >= int(required_count)),
                "median_abs_delta_mm": median_abs_delta,
                "max_abs_delta_mm": max(abs_deltas) if abs_deltas else None,
            }
            score_summaries.append(summary)
            if recommended_score is None and bool(summary["passes"]):
                recommended_score = int(score)
        by_hotkey[hotkey] = {
            "recommended_score": recommended_score,
            "scores": score_summaries,
        }

    return {
        "required_success_ratio": float(required_success_ratio),
        "required_success_count": int(required_count),
        "by_hotkey": by_hotkey,
    }


def run_drive_breakaway_test(
    *,
    robot,
    vision,
    world,
    hotkeys: tuple[str, ...],
    start_score: int,
    max_score: int,
    repeats_per_score: int,
    duration_ms: int,
    movement_threshold_mm: float,
    required_success_ratio: float,
    pause_between_pulses_ms: int,
    use_manual_assist: bool = DEFAULT_USE_MANUAL_ASSIST,
    log_path: Path = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    rows = []
    recenter_checkpoints = []
    hotkeys_norm = tuple(str(item).strip().lower() for item in hotkeys if str(item).strip())
    start_score = _clamp_score(start_score, DEFAULT_START_SCORE)
    max_score = _clamp_score(max_score, DEFAULT_MAX_SCORE)
    if max_score < start_score:
        max_score = int(start_score)
    repeats_per_score = _clamp_positive_int(repeats_per_score, DEFAULT_REPEATS_PER_SCORE)
    try:
        duration_ms = int(round(float(duration_ms)))
    except (TypeError, ValueError):
        duration_ms = None
    if duration_ms is not None and duration_ms <= 0:
        duration_ms = None
    pause_between_pulses_ms = _clamp_positive_int(
        pause_between_pulses_ms,
        DEFAULT_PAUSE_BETWEEN_PULSES_MS,
        minimum=0,
    )
    movement_threshold_mm = max(0.01, float(movement_threshold_mm))
    required_success_ratio = max(0.1, min(1.0, float(required_success_ratio)))
    assist_config = load_manual_drive_assist_config() if bool(use_manual_assist) else {}
    assist_enabled = bool(use_manual_assist) and bool((assist_config or {}).get("enabled"))

    logger("[BREAKAWAY TEST] Goal: verify that low-speed hotkeys move reliably in the correct direction.")
    logger("[BREAKAWAY TEST] Keep the robot on its normal surface, keep the brick visible, and do not touch it during the run.")
    logger(
        f"[BREAKAWAY TEST] Hotkeys={tuple(h.upper() for h in hotkeys_norm)} "
        f"scores={int(start_score)}..{int(max_score)} repeats={int(repeats_per_score)} "
        f"duration={'per-hotkey model' if duration_ms is None else f'{int(duration_ms)}ms'} "
        f"threshold={float(movement_threshold_mm):.2f}mm."
    )
    logger(
        "[BREAKAWAY TEST] Assist mode: "
        + ("manual hotkey assist enabled." if bool(assist_enabled) else "raw pulses only.")
    )
    logger("[BREAKAWAY TEST] Drive mode uses dist changes; turn mode uses x_axis changes.")
    baseline_pose = _median_pose_or_none(
        _collect_pose_samples(
            vision,
            world,
            samples=int(RECENTER_OBSERVE_SAMPLES),
            timeout_s=float(RECENTER_OBSERVE_TIMEOUT_S),
        )
    )
    if isinstance(baseline_pose, dict):
        logger(f"[BREAKAWAY TEST] Baseline pose: {_format_pose_text(baseline_pose)}.")
    else:
        logger("[BREAKAWAY TEST] Baseline pose unavailable; recenter checkpoints will be best-effort.")

    for hotkey_index, hotkey in enumerate(hotkeys_norm):
        spec = _hotkey_spec(hotkey)
        if not isinstance(spec, dict):
            continue
        if hotkey_index > 0:
            checkpoint = _run_recenter_checkpoint(
                vision=vision,
                world=world,
                baseline_pose=baseline_pose,
                next_hotkey=str(hotkey),
                log_fn=logger,
            )
            checkpoint["next_hotkey"] = str(hotkey)
            recenter_checkpoints.append(checkpoint)
        cmd = str(spec.get("cmd") or "")
        logger(
            f"[BREAKAWAY TEST] Testing {str(spec.get('display_label') or hotkey.upper())}: "
            f"cmd={cmd.upper()} metric={str(spec.get('metric_label') or '')} "
            f"expected={str(spec.get('expected_effect') or '')}."
        )
        for score in range(int(start_score), int(max_score) + 1):
            for trial_idx in range(1, int(repeats_per_score) + 1):
                row = _raw_trial(
                    robot=robot,
                    vision=vision,
                    world=world,
                    hotkey=str(hotkey),
                    score=int(score),
                    duration_ms=None if duration_ms is None else int(duration_ms),
                    movement_threshold_mm=float(movement_threshold_mm),
                    use_manual_assist=bool(assist_enabled),
                )
                row["trial_index"] = int(trial_idx)
                rows.append(row)
                if bool(row.get("ok")):
                    logger(
                        f"[BREAKAWAY TEST] {str(row.get('display_label') or hotkey.upper())} "
                        f"score={int(score)}% trial={int(trial_idx)}/{int(repeats_per_score)} "
                        f"{str(row.get('metric_label') or 'metric')} raw_delta={float(row.get('raw_delta_mm') or 0.0):+.3f}mm "
                        f"observed={str(row.get('observed_effect') or 'unknown')} "
                        f"moved={bool(row.get('moved'))} direction_ok={bool(row.get('direction_ok'))} "
                        f"pass={bool(row.get('passes'))}."
                    )
                else:
                    logger(
                        f"[BREAKAWAY TEST] {hotkey.upper()} score={int(score)}% trial={int(trial_idx)}/{int(repeats_per_score)} "
                        f"failed ({row.get('error')})."
                    )
                if int(pause_between_pulses_ms) > 0:
                    time.sleep(float(pause_between_pulses_ms) / 1000.0)

    summary = summarize_breakaway_results(
        rows,
        repeats_per_score=int(repeats_per_score),
        required_success_ratio=float(required_success_ratio),
    )
    for hotkey in hotkeys_norm:
        hotkey_summary = (summary.get("by_hotkey") or {}).get(hotkey, {})
        recommended = hotkey_summary.get("recommended_score")
        spec = _hotkey_spec(hotkey) or {}
        display_label = str(spec.get("display_label") or hotkey.upper())
        if recommended is not None:
            logger(
                f"[BREAKAWAY TEST] Recommendation for {display_label}: first reliable correct-direction score is {int(recommended)}%."
            )
        else:
            logger(
                f"[BREAKAWAY TEST] Recommendation for {display_label}: no reliable correct-direction score found in {int(start_score)}..{int(max_score)}%."
            )

    result = {
        "ok": True,
        "seconds": float(max(0.0, time.time() - started)),
        "settings": {
            "hotkeys": list(hotkeys_norm),
            "start_score": int(start_score),
            "max_score": int(max_score),
            "repeats_per_score": int(repeats_per_score),
            "duration_ms": None if duration_ms is None else int(duration_ms),
            "movement_threshold_mm": float(movement_threshold_mm),
            "required_success_ratio": float(required_success_ratio),
            "pause_between_pulses_ms": int(pause_between_pulses_ms),
            "use_manual_assist": bool(assist_enabled),
            "recenter_tolerance_dist_mm": float(RECENTER_TOLERANCE_DIST_MM),
            "recenter_tolerance_x_axis_mm": float(RECENTER_TOLERANCE_X_AXIS_MM),
            "recenter_timeout_s": float(RECENTER_TIMEOUT_S),
        },
        "baseline_pose": baseline_pose,
        "recenter_checkpoints": recenter_checkpoints,
        "rows": rows,
        "summary": summary,
    }
    try:
        Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
    except OSError as exc:
        result["write_error"] = str(exc)
    return result


def _prompt_with_default(prompt_fn, label: str, default: str) -> str:
    raw = str(prompt_fn(f"{str(label).rstrip()} [{default}]: ") or "").strip()
    return raw if raw else str(default)


def run_interactive_drive_breakaway_test(
    *,
    robot,
    vision,
    world,
    prompt_fn=input,
    log_fn=None,
    log_path: Path = RUN_LOG_FILE_DEFAULT,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    logger("[BREAKAWAY TEST] Command mode helper for low-speed drive/turn hotkeys.")
    logger("[BREAKAWAY TEST] drive = hotkeys R/F -> cmds F/B, turn = hotkeys Q/E -> cmds L/R.")
    logger("[BREAKAWAY TEST] This uses the current hotkey assist behavior when configured.")
    logger(
        "[BREAKAWAY TEST] Between hotkey groups, the test will wait for the robot to be recentered "
        f"near the baseline pose within about +/-{float(RECENTER_TOLERANCE_DIST_MM):.1f}mm dist "
        f"and +/-{float(RECENTER_TOLERANCE_X_AXIS_MM):.1f}mm x_axis."
    )
    selection = _prompt_with_default(
        prompt_fn,
        "  Commands to test (drive/turn/all/f/b/l/r)",
        "all",
    ).strip().lower()
    hotkeys = _resolve_requested_hotkeys(selection)
    duration_default = max(_default_duration_ms_for_hotkey(hotkey) for hotkey in hotkeys)
    start_score = _clamp_score(
        _prompt_with_default(prompt_fn, "  Start score %", str(DEFAULT_START_SCORE)),
        DEFAULT_START_SCORE,
    )
    max_score = _clamp_score(
        _prompt_with_default(prompt_fn, "  Max score %", str(DEFAULT_MAX_SCORE)),
        DEFAULT_MAX_SCORE,
    )
    repeats_per_score = _clamp_positive_int(
        _prompt_with_default(prompt_fn, "  Repeats per score", str(DEFAULT_REPEATS_PER_SCORE)),
        DEFAULT_REPEATS_PER_SCORE,
    )
    duration_ms = _clamp_positive_int(
        _prompt_with_default(prompt_fn, "  Pulse duration ms", str(duration_default)),
        duration_default,
    )
    movement_threshold_mm = max(
        0.01,
        float(_prompt_with_default(prompt_fn, "  Movement threshold mm", f"{DEFAULT_MOVEMENT_THRESHOLD_MM:.2f}")),
    )
    return run_drive_breakaway_test(
        robot=robot,
        vision=vision,
        world=world,
        hotkeys=hotkeys,
        start_score=int(start_score),
        max_score=int(max_score),
        repeats_per_score=int(repeats_per_score),
        duration_ms=int(duration_ms),
        movement_threshold_mm=float(movement_threshold_mm),
        required_success_ratio=float(DEFAULT_REQUIRED_SUCCESS_RATIO),
        pause_between_pulses_ms=int(DEFAULT_PAUSE_BETWEEN_PULSES_MS),
        use_manual_assist=bool(DEFAULT_USE_MANUAL_ASSIST),
        log_path=Path(log_path),
        log_fn=logger,
    )


def _parse_hotkeys_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(
        item
        for item in (str(chunk).strip().lower() for chunk in str(value).split(","))
        if item in HOTKEY_SPECS
    )
    return parsed or tuple(TARGET_HOTKEYS_ALL_DEFAULT)


def _build_vision(vision_mode: str):
    mode = normalize_vision_mode(vision_mode, fallback=DEFAULT_VISION_MODE)
    if mode == VISION_MODE_ARUCO:
        return ArucoBrickVision(debug=False)
    if YoloBrickDetector is not None:
        return YoloBrickDetector(debug=False)
    return LeiaVision(debug=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a low-speed drive/turn hotkey movement test.")
    parser.add_argument("--hotkeys", type=str, default="r,f,q,e", help="Comma-separated hotkeys to test. Supports r,f,q,e.")
    parser.add_argument("--start-score", type=int, default=DEFAULT_START_SCORE)
    parser.add_argument("--max-score", type=int, default=DEFAULT_MAX_SCORE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS_PER_SCORE)
    parser.add_argument("--duration-ms", type=int, default=0, help="Pulse duration override. Use 0 to keep each hotkey's model duration.")
    parser.add_argument("--movement-threshold-mm", type=float, default=DEFAULT_MOVEMENT_THRESHOLD_MM)
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    parser.add_argument("--raw-only", action="store_true", help="Bypass manual hotkey assist and send raw score pulses only.")
    parser.add_argument(
        "--vision",
        type=str,
        choices=("cyan", "yolo", "leia", "aruco"),
        default=str(DEFAULT_VISION_MODE),
        help="Vision mode for the standalone breakaway test. Defaults to the active repo vision mode.",
    )
    args = parser.parse_args()

    robot = Robot()
    vision = _build_vision(str(args.vision))
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK
    try:
        result = run_drive_breakaway_test(
            robot=robot,
            vision=vision,
            world=world,
            hotkeys=_parse_hotkeys_csv(args.hotkeys),
            start_score=args.start_score,
            max_score=args.max_score,
            repeats_per_score=args.repeats,
            duration_ms=None if int(args.duration_ms) <= 0 else int(args.duration_ms),
            movement_threshold_mm=args.movement_threshold_mm,
            required_success_ratio=DEFAULT_REQUIRED_SUCCESS_RATIO,
            pause_between_pulses_ms=DEFAULT_PAUSE_BETWEEN_PULSES_MS,
            use_manual_assist=not bool(args.raw_only),
            log_path=Path(args.log),
        )
        print(json.dumps(result, indent=2))
        return 0 if bool(result.get("ok")) else 1
    except KeyboardInterrupt:
        print("\n[BREAKAWAY TEST] Interrupted.")
        return 130
    finally:
        try:
            vision.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
