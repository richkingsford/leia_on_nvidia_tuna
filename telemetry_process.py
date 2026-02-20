#!/usr/bin/env python3
import argparse
import json
import collections
import math
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_step_label
from helper_manual_config import load_manual_training_config
from helper_vision_aruco import ArucoBrickVision
import helper_gate_utils as gate_utils
from helper_robot_control import Robot
from telemetry_robot import MotionEvent, StepState, WorldModel, draw_telemetry_overlay
import telemetry_brick
import helper_next as next_module
import telemetry_robot as telemetry_robot_module
import telemetry_wall
import helper_learning

USE_LEARNED_POLICY = False
GLOBAL_POLICY = None


DEMO_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
GATE_CHECKER_MODEL_FILE = gate_utils.GATE_CHECKER_MODEL_FILE

_MANUAL_CONFIG = load_manual_training_config()
STREAM_HOST = _MANUAL_CONFIG.get("stream_host", "127.0.0.1")
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))

DEFAULT_AUTOBUILD_CONFIG = {
    "control_hz": 20.0,
    "gate_stability_frames": 5,
    "success_visible_false_grace_s": 0.2,
    "failure_tighten_low_pct": 0.1,
    "failure_tighten_high_pct": 0.9,
    "start_gate_timeout_s": 8.0,
    "success_settle_s": 1.0,
    "success_tail_window_s": 1.5,
    "success_tail_count": 3,
    "max_phase_attempts": 1,
    "smooth_step_s": 1.0,
    "fail_pause_s": 0.35,
    "demo_action_pause_frames": 2,
    "post_act_pause_frames": 3,
    "find_brick_slow_factor": 4.0,
    "visibility_lost_hold_s": 0.5,
    "learned_policy_confidence_threshold": 0.4,
    # Continuous replay tuning: cycle between move (brief) and gate-check (longer)
    # so auto-running steps keep moving instead of "act then pause".
    "auto_cycle_move_score": 6,
    "auto_cycle_move_s": 0.5,
    "auto_cycle_check_score": 2,
    "auto_cycle_check_s": 1.5,
}

DEFAULT_LITE_GATE_CHECK_CONFIG = {
    "default_unique_smoothed_frames": 3,
    "steps": {},
}

DEFAULT_SPEED_SCORE = telemetry_robot_module.SPEED_SCORE_DEFAULT
MAX_SPEED_SCORE = telemetry_robot_module.SPEED_SCORE_MAX

ALIGN_RECOVERY_REVERSE_MAX_ACTS = 6
ALIGN_RECOVERY_REVERSE_OBSERVE_S = 0.8
ALIGN_RECOVERY_SCAN_MAX_ATTEMPTS = 8
ALIGN_RECOVERY_SCAN_SCORE = 8
ALIGN_RECOVERY_SCAN_BURST_ACTS = 2
ALIGN_RECOVERY_SCAN_OBSERVE_S = 0.6
ALIGN_RECOVERY_LOST_VISIBLE_FRAMES = 3
ALIGN_RECOVERY_LOST_VISIBLE_MIN_S = 0.20
ALIGN_RECOVERY_PREMOVE_RECHECK_ROUNDS = 3

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}
SUCCESS_GATE_EXCLUDE_ANGLE_STEPS = {
    "ALIGN_BRICK",
    "POSITION_BRICK",
}
RETIRED_PROCESS_STEPS = {
    "RETREAT",
}
AUTO_DIAG_OFFSET_PRIORITY = ("xAxis_offset_abs", "dist")
AUTO_DIAG_SINGLE_OFFSET_PRECHECK_STEPS = {
    "ALIGN_BRICK",
    "POSITION_BRICK",
}
AUTO_SPEED_SCORE_HARD_MAX = 25
AUTO_DEMO_SPEED_STEPS = {
    "FIND_WALL",
    "EXIT_WALL",
    "FIND_BRICK",
    "FIND_WALL2",
}
FIND_BRICK_TURN_MOVE_MAX_SCORE = 2
FIND_BRICK_TURN_CHECK_MAX_SCORE = 1
MIN_MM_TOL = 1.5
X_AXIS_OFFSET_ABS_TOL_MM = 1.4



def _as_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _frames_from_seconds(seconds, fallback_frames):
    sec = _as_float(seconds, None)
    if sec is None:
        return max(0, int(fallback_frames))
    return max(0, int(round(sec * max(1, int(STREAM_FPS)))))


def default_speed_for_cmd(cmd, score=DEFAULT_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def max_speed_for_cmd(cmd, score=MAX_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def apply_autobuild_config(cfg):
    global CONTROL_HZ
    global CONTROL_DT
    global GATE_STABILITY_FRAMES
    global SUCCESS_VISIBLE_FALSE_GRACE_S
    global FAILURE_TIGHTEN_LOW_PCT
    global FAILURE_TIGHTEN_HIGH_PCT
    global START_GATE_TIMEOUT_S
    global SUCCESS_SETTLE_S
    global SUCCESS_TAIL_WINDOW_S
    global SUCCESS_TAIL_COUNT
    global MAX_PHASE_ATTEMPTS
    global SMOOTH_STEP_S
    global FAIL_PAUSE_S
    global DEMO_ACTION_PAUSE_FRAMES
    global POST_ACT_PAUSE_FRAMES
    global FIND_BRICK_SLOW_FACTOR
    global VISIBILITY_LOST_HOLD_S
    global LEARNED_POLICY_CONFIDENCE_THRESHOLD
    global AUTO_CYCLE_MOVE_SCORE
    global AUTO_CYCLE_MOVE_S
    global AUTO_CYCLE_CHECK_SCORE
    global AUTO_CYCLE_CHECK_S

    CONTROL_HZ = _as_float(cfg.get("control_hz"), DEFAULT_AUTOBUILD_CONFIG["control_hz"])
    CONTROL_DT = 1.0 / max(CONTROL_HZ, 1e-6)
    GATE_STABILITY_FRAMES = _as_int(cfg.get("gate_stability_frames"), DEFAULT_AUTOBUILD_CONFIG["gate_stability_frames"])
    SUCCESS_VISIBLE_FALSE_GRACE_S = _as_float(
        cfg.get("success_visible_false_grace_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_visible_false_grace_s"],
    )
    FAILURE_TIGHTEN_LOW_PCT = _as_float(
        cfg.get("failure_tighten_low_pct"),
        DEFAULT_AUTOBUILD_CONFIG["failure_tighten_low_pct"],
    )
    FAILURE_TIGHTEN_HIGH_PCT = _as_float(
        cfg.get("failure_tighten_high_pct"),
        DEFAULT_AUTOBUILD_CONFIG["failure_tighten_high_pct"],
    )
    START_GATE_TIMEOUT_S = _as_float(
        cfg.get("start_gate_timeout_s"),
        DEFAULT_AUTOBUILD_CONFIG["start_gate_timeout_s"],
    )
    SUCCESS_SETTLE_S = _as_float(
        cfg.get("success_settle_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_settle_s"],
    )
    SUCCESS_TAIL_WINDOW_S = _as_float(
        cfg.get("success_tail_window_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_tail_window_s"],
    )
    SUCCESS_TAIL_COUNT = _as_int(
        cfg.get("success_tail_count"),
        DEFAULT_AUTOBUILD_CONFIG["success_tail_count"],
    )
    MAX_PHASE_ATTEMPTS = _as_int(
        cfg.get("max_phase_attempts"),
        DEFAULT_AUTOBUILD_CONFIG["max_phase_attempts"],
    )
    SMOOTH_STEP_S = _as_float(
        cfg.get("smooth_step_s"),
        DEFAULT_AUTOBUILD_CONFIG["smooth_step_s"],
    )
    FAIL_PAUSE_S = _as_float(
        cfg.get("fail_pause_s"),
        DEFAULT_AUTOBUILD_CONFIG["fail_pause_s"],
    )
    if cfg.get("demo_action_pause_frames") is None and cfg.get("demo_action_pause_s") is not None:
        DEMO_ACTION_PAUSE_FRAMES = _frames_from_seconds(
            cfg.get("demo_action_pause_s"),
            DEFAULT_AUTOBUILD_CONFIG["demo_action_pause_frames"],
        )
    else:
        DEMO_ACTION_PAUSE_FRAMES = _as_int(
            cfg.get("demo_action_pause_frames"),
            DEFAULT_AUTOBUILD_CONFIG["demo_action_pause_frames"],
        )
    if cfg.get("post_act_pause_frames") is None and cfg.get("post_act_pause_s") is not None:
        POST_ACT_PAUSE_FRAMES = _frames_from_seconds(
            cfg.get("post_act_pause_s"),
            DEFAULT_AUTOBUILD_CONFIG["post_act_pause_frames"],
        )
    else:
        POST_ACT_PAUSE_FRAMES = _as_int(
            cfg.get("post_act_pause_frames"),
            DEFAULT_AUTOBUILD_CONFIG["post_act_pause_frames"],
        )
    FIND_BRICK_SLOW_FACTOR = _as_float(
        cfg.get("find_brick_slow_factor"),
        DEFAULT_AUTOBUILD_CONFIG["find_brick_slow_factor"],
    )
    VISIBILITY_LOST_HOLD_S = _as_float(
        cfg.get("visibility_lost_hold_s"),
        DEFAULT_AUTOBUILD_CONFIG["visibility_lost_hold_s"],
    )
    LEARNED_POLICY_CONFIDENCE_THRESHOLD = _as_float(
        cfg.get("learned_policy_confidence_threshold"),
        DEFAULT_AUTOBUILD_CONFIG["learned_policy_confidence_threshold"],
    )
    AUTO_CYCLE_MOVE_SCORE = telemetry_robot_module.normalize_speed_score(
        cfg.get("auto_cycle_move_score"),
        DEFAULT_AUTOBUILD_CONFIG["auto_cycle_move_score"],
    )
    AUTO_CYCLE_MOVE_S = max(
        0.0,
        _as_float(
            cfg.get("auto_cycle_move_s"),
            DEFAULT_AUTOBUILD_CONFIG["auto_cycle_move_s"],
        ),
    )
    AUTO_CYCLE_CHECK_SCORE = telemetry_robot_module.normalize_speed_score(
        cfg.get("auto_cycle_check_score"),
        DEFAULT_AUTOBUILD_CONFIG["auto_cycle_check_score"],
    )
    AUTO_CYCLE_CHECK_S = max(
        0.0,
        _as_float(
            cfg.get("auto_cycle_check_s"),
            DEFAULT_AUTOBUILD_CONFIG["auto_cycle_check_s"],
        ),
    )


apply_autobuild_config(DEFAULT_AUTOBUILD_CONFIG)


def apply_lite_gate_check_config(cfg):
    global LITE_GATE_DEFAULT_UNIQUE_FRAMES
    global LITE_GATE_STEP_UNIQUE_FRAMES

    if not isinstance(cfg, dict):
        cfg = {}
    default_cfg = DEFAULT_LITE_GATE_CHECK_CONFIG
    LITE_GATE_DEFAULT_UNIQUE_FRAMES = max(
        1,
        _as_int(
            cfg.get("default_unique_smoothed_frames"),
            default_cfg["default_unique_smoothed_frames"],
        ),
    )
    steps_raw = cfg.get("steps") if isinstance(cfg, dict) else {}
    parsed = {}
    if isinstance(steps_raw, list):
        for step_name in steps_raw:
            step_key = normalize_step_label(step_name)
            if not step_key:
                continue
            parsed[step_key] = LITE_GATE_DEFAULT_UNIQUE_FRAMES
    elif isinstance(steps_raw, dict):
        for step_name, step_cfg in steps_raw.items():
            step_key = normalize_step_label(step_name)
            if not step_key:
                continue
            if isinstance(step_cfg, dict):
                enabled = step_cfg.get("enabled", True)
                if enabled is False:
                    continue
                unique_frames = _as_int(
                    step_cfg.get("unique_smoothed_frames"),
                    LITE_GATE_DEFAULT_UNIQUE_FRAMES,
                )
            elif isinstance(step_cfg, bool):
                if not step_cfg:
                    continue
                unique_frames = LITE_GATE_DEFAULT_UNIQUE_FRAMES
            else:
                unique_frames = _as_int(step_cfg, LITE_GATE_DEFAULT_UNIQUE_FRAMES)
            parsed[step_key] = max(1, int(unique_frames))
    LITE_GATE_STEP_UNIQUE_FRAMES = parsed


apply_lite_gate_check_config(DEFAULT_LITE_GATE_CHECK_CONFIG)


def lite_gate_unique_frames(step):
    step_key = normalize_step_label(step)
    if not step_key:
        return None
    frames = LITE_GATE_STEP_UNIQUE_FRAMES.get(step_key)
    if frames is None:
        return None
    return max(1, int(frames))


def apply_gate_checker_config(cfg):
    global GATECHECK_CONSECUTIVE_REQUIRED
    global GATECHECK_MAJORITY_WINDOW
    global GATECHECK_MAJORITY_REQUIRED
    cfg = cfg or {}
    default_cfg = gate_utils.DEFAULT_GATE_CHECKER_CONFIG
    GATECHECK_CONSECUTIVE_REQUIRED = _as_int(
        cfg.get("consecutive_required"),
        default_cfg["consecutive_required"],
    )
    GATECHECK_MAJORITY_WINDOW = _as_int(
        cfg.get("majority_window"),
        default_cfg["majority_window"],
    )
    GATECHECK_MAJORITY_REQUIRED = _as_int(
        cfg.get("majority_required"),
        default_cfg["majority_required"],
    )
    GATECHECK_CONSECUTIVE_REQUIRED = max(1, GATECHECK_CONSECUTIVE_REQUIRED)
    GATECHECK_MAJORITY_WINDOW = max(1, GATECHECK_MAJORITY_WINDOW)
    GATECHECK_MAJORITY_REQUIRED = max(
        1,
        min(GATECHECK_MAJORITY_REQUIRED, GATECHECK_MAJORITY_WINDOW),
    )


def refresh_gate_checker_config(path=GATE_CHECKER_MODEL_FILE):
    cfg = gate_utils.load_gate_checker_config(path)
    apply_gate_checker_config(cfg)
    return cfg


refresh_gate_checker_config()


COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_WHITE = "\033[37m"
COLOR_GRAY = "\033[90m"
COLOR_CYAN = "\033[36m"
COLOR_YELLOW = "\033[33m"
COLOR_ORANGE_BRIGHT = "\033[38;5;208m"
ACTION_DISPLAY_TTL_S = 3.0


ACTION_CMD_MAP = {
    "forward": "f",
    "backward": "b",
    "left_turn": "l",
    "right_turn": "r",
    "mast_up": "u",
    "mast_down": "d",
}

ACTION_CMD_DESC = {
    "f": "moving forward",
    "b": "moving backward",
    "l": "turning left",
    "r": "turning right",
    "u": "lifting mast",
    "d": "lowering mast",
}


def cmd_to_motion_type(cmd):
    return {
        "f": "forward",
        "b": "backward",
        "l": "left_turn",
        "r": "right_turn",
        "u": "mast_up",
        "d": "mast_down",
    }.get(cmd, "wait")


def format_headline(headline, color, details=""):
    if details is None:
        details = ""
    return f"{color}{headline}{COLOR_RESET}{details}"


@dataclass
class MotionStep:
    cmd: str
    speed: float
    duration_s: float
    label: str
    speed_score: int = None

def select_tail_states(states, window_s=SUCCESS_TAIL_WINDOW_S, max_count=SUCCESS_TAIL_COUNT):
    if not states:
        return []
    if max_count is not None and max_count > 0:
        states = states[-max_count:]
    last_ts = None
    for state in reversed(states):
        ts = state.get("timestamp")
        if ts is not None:
            last_ts = ts
            break
    if last_ts is None:
        return states
    cutoff = last_ts - window_s
    tail = [state for state in states if state.get("timestamp") is not None and state.get("timestamp") >= cutoff]
    return tail or states


def select_head_states(states, window_s=SUCCESS_TAIL_WINDOW_S, max_count=SUCCESS_TAIL_COUNT):
    if not states:
        return []
    if max_count is not None and max_count > 0:
        states = states[:max_count]
    first_ts = None
    for state in states:
        ts = state.get("timestamp")
        if ts is not None:
            first_ts = ts
            break
    if first_ts is None:
        return states
    cutoff = first_ts + window_s
    head = [state for state in states if state.get("timestamp") is not None and state.get("timestamp") <= cutoff]
    return head or states


def failure_based_scale(success_count, fail_count):
    return (success_count + 1.0) / (fail_count + 1.0)


def derive_success_gate_scales(segments_by_obj, step_rules=None):
    scales = {}
    for obj, segs in segments_by_obj.items():
        success_count = len(segs.get("SUCCESS", []))
        fail_count = len(segs.get("FAIL", []))
        if success_count <= 0 and fail_count <= 0:
            continue
        scales[obj] = failure_based_scale(success_count, fail_count)
    if isinstance(step_rules, dict):
        for obj, rules in step_rules.items():
            if not isinstance(rules, dict):
                continue
            scale = rules.get("success_gate_scale")
            if scale is None:
                continue
            try:
                scale_val = float(scale)
            except (TypeError, ValueError):
                continue
            if scale_val <= 0:
                continue
            obj_key = normalize_step_label(obj)
            if not obj_key:
                continue
            current = scales.get(obj_key, 1.0)
            scales[obj_key] = current * scale_val
    return scales


def success_frames_required(step):
    lite_frames = lite_gate_unique_frames(step)
    if lite_frames is not None:
        return int(lite_frames)
    return int(GATECHECK_CONSECUTIVE_REQUIRED)


def new_success_tracker(step, process_rules=None):
    lite_frames = lite_gate_unique_frames(step)
    if lite_frames is not None:
        tracker = gate_utils.SuccessGateTracker(1, 1, 1)
        tracker.lite_unique_frames = int(lite_frames)
        return tracker
    consecutive_required = int(GATECHECK_CONSECUTIVE_REQUIRED)
    majority_window = int(GATECHECK_MAJORITY_WINDOW)
    majority_required = int(GATECHECK_MAJORITY_REQUIRED)

    # Visible-only gates can be noisy; confirm success ("positive") for ~2x as long as
    # we confirm not-success ("negative"). With the default 8-frame positive, this
    # becomes a 12-frame majority window requiring 8 passes (8 pass / 4 fail).
    if isinstance(process_rules, dict) and next_module.success_gates_visible_only(process_rules, step):
        negative_frames = max(1, consecutive_required // 2)
        majority_window = max(1, consecutive_required + negative_frames)
        majority_required = max(1, min(consecutive_required, majority_window))

    return gate_utils.SuccessGateTracker(
        consecutive_required,
        majority_window,
        majority_required,
    )


def round_value(value, decimals=2):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, decimals)
    return value


def _fmt_gate_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return str(value)


def format_gate_metrics(gates):
    if not gates:
        return "none"
    parts = []
    for metric, stats in gates.items():
        if not isinstance(stats, dict):
            continue
        if metric == "visible":
            if "min" in stats:
                parts.append(f"{metric}={_fmt_gate_value(stats.get('min'))}")
            elif "max" in stats:
                parts.append(f"{metric}={_fmt_gate_value(stats.get('max'))}")
            continue
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            parts.append(f"{metric}~{_fmt_gate_value(target)}+/-{_fmt_gate_value(tol)}")
            continue
        mu = stats.get("mu")
        sigma = stats.get("sigma")
        if mu is not None and sigma is not None:
            parts.append(f"{metric}~{_fmt_gate_value(mu)}+/-{_fmt_gate_value(sigma)}")
            continue
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and max_val is not None:
            parts.append(f"{_fmt_gate_value(min_val)}<={metric}<={_fmt_gate_value(max_val)}")
        elif min_val is not None:
            parts.append(f"{metric}>={_fmt_gate_value(min_val)}")
        elif max_val is not None:
            parts.append(f"{metric}<={_fmt_gate_value(max_val)}")
    return ", ".join(parts) if parts else "none"


def format_gate_lines(cfg):
    return (
        format_gate_metrics(cfg.get("start_gates") or {}),
        format_gate_metrics(cfg.get("success_gates") or {}),
    )


def _visibility_grace_s(world, step):
    if success_visible_target(world, step) is False:
        return SUCCESS_VISIBLE_FALSE_GRACE_S
    return None


def _success_gate_entries(world, step):
    process_rules = world.process_rules or {}
    learned_rules = world.learned_rules or {}
    visibility_grace_s = _visibility_grace_s(world, step)
    return telemetry_brick.success_gate_entries(
        world,
        step,
        learned_rules,
        process_rules=process_rules,
        visibility_grace_s=visibility_grace_s,
    )


def _format_success_gate_entry(world, entry):
    metric = entry.get("metric")
    stats = entry.get("stats") or {}
    value = entry.get("value")
    actual_str = _fmt_gate_value(value) if value is not None else "n/a"
    gate_bits = []
    if isinstance(stats, dict):
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            gate_bits.append(f"{_fmt_gate_value(target)} +/- {_fmt_gate_value(tol)}")
        else:
            if "min" in stats:
                gate_bits.append(f">={_fmt_gate_value(stats.get('min'))}")
            if "max" in stats:
                gate_bits.append(f"<={_fmt_gate_value(stats.get('max'))}")
        if "mu" in stats:
            gate_bits.append(f"mu={_fmt_gate_value(stats.get('mu'))}")
        if "sigma" in stats:
            gate_bits.append(f"sigma={_fmt_gate_value(stats.get('sigma'))}")

    if metric == "visible":
        raw = entry.get("raw_visible")
        effective = entry.get("effective_visible")
        grace_s = entry.get("visible_grace_s")
        if effective is not None:
            eff_str = _fmt_gate_value(bool(effective))
            raw_str = _fmt_gate_value(bool(raw)) if raw is not None else "n/a"
            if raw is not None and bool(raw) != bool(effective):
                if grace_s is not None:
                    actual_str = f"{eff_str} (raw={raw_str}, grace={grace_s:.2f}s)"
                else:
                    actual_str = f"{eff_str} (raw={raw_str})"
            elif grace_s is not None and grace_s > 0:
                actual_str = f"{eff_str} (grace={grace_s:.2f}s)"
            else:
                actual_str = eff_str
        last_seen = getattr(world, "last_visible_time", None)
        if last_seen is not None:
            age_s = time.time() - last_seen
            gate_bits.append(f"last_seen={age_s:.1f}s")

    gate_desc = " ".join(gate_bits) if gate_bits else "gate"
    return metric, actual_str, gate_desc


def format_success_event_lines(world, step):
    entries = _success_gate_entries(world, step)
    return [f"{metric}={actual} ({gate})" for metric, actual, gate in (
        _format_success_gate_entry(world, entry) for entry in entries
    )]


def format_success_details(world, step):
    lines = format_success_event_lines(world, step)
    if not lines:
        return "[SUCCESS DETAILS] no success gates"
    return "[SUCCESS DETAILS] " + "; ".join(lines)


def print_success_events(world, step):
    lines = format_success_event_lines(world, step)
    if not lines:
        print(format_headline("[SUCCESS EVENTS] no success gates", COLOR_WHITE))
        return
    print(format_headline("[SUCCESS EVENTS]", COLOR_WHITE))
    for line in lines:
        print(f"  - {line}")


def format_brick_state_line(world):
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    parts = [f"visible={'true' if visible else 'false'}"]
    if visible:
        dist = brick.get("dist")
        angle = brick.get("angle")
        offset = brick.get("x_axis")
        if offset is None:
            offset = brick.get("offset_x")
        conf = brick.get("confidence")
        above = brick.get("brickAbove")
        below = brick.get("brickBelow")
        if dist is not None:
            parts.append(f"dist={dist:.1f}mm")
        if angle is not None:
            parts.append(f"angle={angle:.2f}deg")
        if offset is not None:
            parts.append(f"x_axis={offset:.2f}mm")
        if conf is not None:
            parts.append(f"conf={conf:.0f}")
        if above is not None:
            parts.append(f"above={str(bool(above)).lower()}")
        if below is not None:
            parts.append(f"below={str(bool(below)).lower()}")
    else:
        last_seen = getattr(world, "last_visible_time", None)
        if last_seen is not None:
            parts.append(f"last_seen={time.time() - last_seen:.2f}s")
    return "[BRICK] " + " ".join(parts)


def format_action_line(step, target_visible, reason=None):
    action = ACTION_CMD_DESC.get(step.cmd, "moving")
    power = f"{step.speed:.2f}"
    if target_visible is None:
        suffix = ""
    else:
        if target_visible:
            suffix = "until brick becomes visible"
        else:
            suffix = "until brick is no longer visible"
    reason = str(reason).strip() if reason else ""
    # Make reason gray
    reason_suffix = f" {COLOR_GRAY}({reason}){COLOR_RESET}" if reason else ""
    return f"[ACT] {action} at {power} power {suffix}{reason_suffix}"


def format_control_action_line(cmd, speed, reason=None):
    if not cmd:
        reason_str = str(reason).strip() if reason else ""
        suffix = f" ({reason_str})" if reason_str else ""
        return f"[ACT] holding position{suffix}"

    display_cmd = _remap_cmd_for_display(cmd)

    # Action Description
    action_map = {
        "f": "move forward",
        "b": "move backward",
        "l": "turn left",
        "r": "turn right",
        "u": "lift mast",
        "d": "lower mast",
    }
    action = action_map.get(display_cmd, "move")
    
    # Power Level
    level = "Major" if speed >= 0.24 else "Minor"
    
    # Parse Reason
    reason_str = str(reason) if reason else ""
    parts = reason_str.split("|")
    gap_info = parts[0] if parts else ""
    delta_info = parts[1] if len(parts) > 1 else ""
    
    speed_str = f"{speed:.2f}p"
    # Format Line 1
    if ":" in gap_info:
        # e.g. "angle:19.18deg" -> "close the 19.18deg gap"
        _, gap_val = gap_info.split(":", 1)
        line1 = f"[ACT] {level} {action} {speed_str} to close the {gap_val} gap."
    else:
        line1 = f"[ACT] {level} {action} {speed_str} ({reason_str})"
        
    # Format Line 2 (Progress)
    line2 = ""
    if delta_info and ":" in delta_info:
        # e.g. "closer:2.13deg" -> "🟢 The last act got us 2.13deg closer to perfect alignment."
        direction, delta_val = delta_info.split(":", 1)
        emoji = "🟢" if direction == "closer" else "🔴"
        word = "closer to" if direction == "closer" else "further from"
        line2 = f"\n{emoji} The last act got us {delta_val} {word} perfect alignment."
        
    return f"{line1}{line2}"


def action_display_text(cmd, speed_score=None):
    if not cmd:
        return "HOLD"
    score_suffix = f" {int(speed_score)}%" if speed_score is not None else ""
    return f"{str(cmd).upper()}{score_suffix}"

def action_sent_display_text(cmd, speed_score=None, *, cmd_sent=None, pwm=None, power=None, duration_ms=None):
    if not cmd:
        return "HOLD"
    cmd_display = str(cmd)
    score_suffix = f" {int(speed_score)}%" if speed_score is not None else ""
    parts = [f"{cmd_display.upper()}"]
    if cmd_sent is not None:
        sent_cmd = str(cmd_sent).strip()
        if sent_cmd and sent_cmd.lower() != str(cmd).strip().lower():
            parts.append(f"sent={sent_cmd.upper()}")
    if pwm is not None:
        try:
            pwm_val = int(round(float(pwm)))
        except (TypeError, ValueError):
            pwm_val = None
        if pwm_val is not None:
            parts.append(f"pwm={pwm_val}")
    if power is not None:
        try:
            power_val = float(power)
        except (TypeError, ValueError):
            power_val = None
        if power_val is not None:
            parts.append(f"power={power_val:.3f}")
    if duration_ms is not None:
        try:
            ms_val = int(round(float(duration_ms)))
        except (TypeError, ValueError):
            ms_val = None
        if ms_val is not None:
            parts.append(f"{ms_val}ms")
    details = ""
    if len(parts) > 1:
        details = " (" + ", ".join(parts[1:]) + ")"
    return f"{parts[0]}{score_suffix}{details}"


def auto_action_detail_text(cmd, speed_score=None, *, action_meta=None):
    if isinstance(action_meta, dict):
        score_effective = action_meta.get("score_effective")
        if score_effective is None:
            score_effective = action_meta.get("score_model", speed_score)
        duration_ms = action_meta.get("duration_ms")
        if duration_ms is None:
            duration_ms = action_meta.get("duration_model_ms")
        return action_sent_display_text(
            cmd,
            score_effective,
            cmd_sent=action_meta.get("cmd_sent"),
            pwm=action_meta.get("pwm"),
            power=action_meta.get("power"),
            duration_ms=duration_ms,
        )
    return action_display_text(cmd, speed_score)


def record_action_display(
    world,
    step,
    cmd,
    speed,
    speed_score=None,
    *,
    cmd_sent=None,
    pwm=None,
    duration_ms=None,
    score_model=None,
):
    if world is None or not cmd:
        return
    obj_key = normalize_step_label(step)
    display_cmd = cmd_sent if cmd_sent is not None else cmd
    world._last_action_obj = obj_key
    world._last_action_time = time.time()
    world._last_action_cmd = cmd
    world._last_action_speed = speed
    world._last_action_score = speed_score
    world._last_action_score_model = score_model
    world._last_action_cmd_sent = cmd_sent
    world._last_action_pwm = pwm
    world._last_action_duration_ms = duration_ms
    world._last_action_display = action_display_text(display_cmd, speed_score)
    world._last_action_sent_display = action_sent_display_text(
        cmd,
        speed_score,
        cmd_sent=cmd_sent,
        pwm=pwm,
        power=speed,
        duration_ms=duration_ms,
    )


def record_hold_display(world, step, reason=None, *, duration_ms=0):
    if world is None:
        return
    obj_key = normalize_step_label(step)
    world._last_action_obj = obj_key
    world._last_action_time = time.time()
    world._last_action_cmd = None
    world._last_action_speed = 0.0
    world._last_action_score = None
    world._last_action_score_model = None
    world._last_action_cmd_sent = None
    world._last_action_pwm = 0
    try:
        world._last_action_duration_ms = int(round(float(duration_ms)))
    except (TypeError, ValueError):
        world._last_action_duration_ms = 0
    hold = "HOLD"
    if reason:
        hold = f"HOLD ({reason})"
    world._last_action_display = hold
    world._last_action_sent_display = hold


def _capture_sent_action_snapshot(world):
    if world is None:
        return None
    snapshot = {}
    sent_display = getattr(world, "_last_action_sent_display", None)
    if sent_display:
        snapshot["sent_display"] = str(sent_display)
    sent_step = normalize_step_label(getattr(world, "_last_action_obj", None))
    if sent_step:
        snapshot["sent_step"] = sent_step
    wire_text = getattr(world, "_last_action_wire", None)
    if wire_text:
        snapshot["wire_text"] = str(wire_text)
    wire_step = normalize_step_label(getattr(world, "_last_action_wire_step", None))
    if wire_step:
        snapshot["wire_step"] = wire_step
    sent_time = getattr(world, "_last_action_time", None)
    wire_time = getattr(world, "_last_action_wire_time", None)
    if sent_time is not None:
        try:
            snapshot["sent_time"] = float(sent_time)
        except (TypeError, ValueError):
            pass
    if wire_time is not None:
        try:
            snapshot["wire_time"] = float(wire_time)
        except (TypeError, ValueError):
            pass
    return snapshot if snapshot else None


def _snapshot_event_time(snapshot):
    if not isinstance(snapshot, dict):
        return None
    candidates = []
    for key in ("sent_time", "wire_time"):
        value = snapshot.get(key)
        if value is None:
            continue
        try:
            candidates.append(float(value))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None
    return max(candidates)


def _format_sent_snapshot_value(snapshot, obj_name):
    sent_value = "(none)"
    if isinstance(snapshot, dict):
        sent_display = snapshot.get("sent_display")
        sent_step = normalize_step_label(snapshot.get("sent_step"))
        wire_text = snapshot.get("wire_text")
        wire_step = normalize_step_label(snapshot.get("wire_step"))
        if sent_display:
            sent_value = str(sent_display)
            if sent_step and sent_step != obj_name:
                sent_value = f"{sent_value} [{sent_step}]"
        if wire_text:
            wire_value = str(wire_text)
            if wire_step and wire_step != obj_name:
                wire_value = f"{wire_value} [{wire_step}]"
            if sent_value == "(none)":
                sent_value = f"(none) [wire: {wire_value}]"
            else:
                sent_value = f"{sent_value} [wire: {wire_value}]"
    return sent_value


def _remap_cmd_for_display(cmd):
    if not cmd:
        return cmd
    cmd_remap = getattr(telemetry_robot_module, "COMMAND_REMAP", None)
    if isinstance(cmd_remap, dict):
        remapped = cmd_remap.get(cmd)
        if remapped:
            return remapped
    return cmd


def _duration_used_ms_for_cmd(robot, cmd, duration_ms, *, auto_mode=False):
    try:
        duration_used_ms = int(duration_ms)
    except (TypeError, ValueError):
        duration_used_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 0) or 0)
    if cmd in ("l", "r"):
        last_turn_cmd = getattr(robot, "_last_turn_cmd", None)
        if auto_mode:
            eff_scale = telemetry_robot_module.turn_duration_scale(cmd)
            duration_used_ms = max(1, int(round(duration_used_ms * eff_scale)))
            if last_turn_cmd != cmd:
                duration_used_ms = max(1, int(round(duration_used_ms * 0.4)))
            # Never allow autonomous micro-turns to be shorter than the configured
            # minimum movement (1% score). Short bursts can be too weak to overcome
            # static friction and appear much slower than the manual hotkeys.
            try:
                _, _, _, floor_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                    cmd,
                    getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1),
                )
                floor_ms = int(round(float(floor_ms)))
            except Exception:
                floor_ms = None
            if floor_ms is not None and floor_ms > 0:
                duration_used_ms = max(int(floor_ms), int(duration_used_ms))
        else:
            # Manual hotkeys: first turn pulse should be half the sequential pulse.
            if last_turn_cmd != cmd:
                duration_used_ms = max(1, int(round(duration_used_ms * 0.5)))
        try:
            robot._last_turn_cmd = cmd
        except Exception:
            pass
    else:
        try:
            robot._last_turn_cmd = None
        except Exception:
            pass
    return int(duration_used_ms)


def _cap_auto_speed_score(score):
    try:
        score_val = int(round(float(score)))
    except (TypeError, ValueError):
        score_val = int(DEFAULT_SPEED_SCORE)
    score_val = telemetry_robot_module.normalize_speed_score(score_val)
    return min(int(score_val), int(AUTO_SPEED_SCORE_HARD_MAX))


def _apply_find_brick_turn_speed_policy(step, cmd, score, *, phase=None):
    if score is None:
        return None
    step_key = normalize_step_label(step)
    if step_key != "FIND_BRICK" or cmd not in ("l", "r"):
        return score
    try:
        score_val = telemetry_robot_module.normalize_speed_score(score)
    except Exception:
        return score
    try:
        min_score = telemetry_robot_module.normalize_speed_score(
            getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1)
        )
    except Exception:
        min_score = 1
    move_cap = telemetry_robot_module.normalize_speed_score(FIND_BRICK_TURN_MOVE_MAX_SCORE)
    check_cap = telemetry_robot_module.normalize_speed_score(FIND_BRICK_TURN_CHECK_MAX_SCORE)
    phase_key = str(phase or "").strip().lower()
    cap = check_cap if phase_key == "check" else move_cap
    cap = max(int(min_score), int(cap))
    return min(int(score_val), int(cap))


def _auto_uses_demo_speed(step):
    step_key = normalize_step_label(step)
    return step_key in AUTO_DEMO_SPEED_STEPS


def send_robot_command_pwm(robot, world, step, cmd, power, pwm, duration_ms, speed_score=None, auto_mode=False):
    if robot is None or cmd is None:
        return None
    try:
        pwm_val = int(round(pwm))
    except (TypeError, ValueError):
        pwm_val = 0
    try:
        pwm_cap = int(round(float(getattr(telemetry_robot_module, "MAX_PWM", 255) or 255)))
    except (TypeError, ValueError):
        pwm_cap = 255
    pwm_cap = max(0, min(255, pwm_cap))
    pwm_val = max(0, min(pwm_cap, pwm_val))
    try:
        power_val = float(power)
    except (TypeError, ValueError):
        power_val = 0.0
    power_val = max(0.0, min(1.0, power_val))

    duration_used_ms = _duration_used_ms_for_cmd(robot, cmd, duration_ms, auto_mode=auto_mode)
    cmd_remap = getattr(telemetry_robot_module, "COMMAND_REMAP", None)
    if not isinstance(cmd_remap, dict):
        cmd_remap = {}
    cmd_sent = cmd_remap.get(cmd, cmd)

    requested_score = None
    if speed_score is not None:
        try:
            requested_score = telemetry_robot_module.normalize_speed_score(speed_score)
        except Exception:
            requested_score = None

    def _floor_pwm_for_cmd(cmd_key, score_key):
        try:
            _, floor_pwm, _, _ = telemetry_robot_module.speed_power_pwm_for_cmd(
                cmd_key,
                score_key,
            )
            return int(round(float(floor_pwm)))
        except Exception:
            return None

    # Enforce a minimum non-zero PWM based on score floors for both the logical
    # command and the actual wire command (after remap).
    if cmd in ("f", "b", "l", "r") and pwm_val > 0:
        floor_scores = [getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1)]
        if requested_score is not None:
            floor_scores.append(int(requested_score))
        candidate_cmds = []
        for cmd_key in (cmd, cmd_sent):
            if cmd_key in ("f", "b", "l", "r") and cmd_key not in candidate_cmds:
                candidate_cmds.append(cmd_key)
        for cmd_key in candidate_cmds:
            for floor_score in floor_scores:
                floor_pwm = _floor_pwm_for_cmd(cmd_key, floor_score)
                if floor_pwm is not None and floor_pwm > 0:
                    pwm_val = max(int(floor_pwm), int(pwm_val))

    if auto_mode and cmd in ("l", "r"):
        floor_scores = [getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1)]
        if requested_score is not None:
            floor_scores.append(int(requested_score))
        candidate_cmds = []
        for cmd_key in (cmd, cmd_sent):
            if cmd_key in ("l", "r") and cmd_key not in candidate_cmds:
                candidate_cmds.append(cmd_key)
        for cmd_key in candidate_cmds:
            for floor_score in floor_scores:
                try:
                    _, _, _, floor_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd_key, floor_score)
                    floor_ms = int(round(float(floor_ms)))
                except Exception:
                    floor_ms = None
                if floor_ms is not None and floor_ms > 0:
                    duration_used_ms = max(int(duration_used_ms), int(floor_ms))

    send_result = None
    if hasattr(robot, "send_command_pwm"):
        send_result = robot.send_command_pwm(cmd, pwm_val, duration_ms=duration_used_ms)
    else:
        send_result = robot.send_command(cmd, power_val, duration_ms=duration_used_ms)
    if isinstance(send_result, dict):
        cmd_sent = send_result.get("cmd_sent") or cmd_sent
        try:
            pwm_val = int(round(float(send_result.get("pwm"))))
        except (TypeError, ValueError):
            pass
        try:
            duration_used_ms = int(round(float(send_result.get("duration_ms"))))
        except (TypeError, ValueError):
            pass

    if world is not None:
        wire_text = getattr(robot, "last_command", None)
        if not wire_text:
            wire_text = f"{cmd_sent} {int(pwm_val)} {int(duration_used_ms)}"
        world._last_action_wire = str(wire_text)
        world._last_action_wire_step = normalize_step_label(step)
        world._last_action_wire_time = time.time()

    power_from_pwm = telemetry_robot_module.pwm_to_power(pwm_val)
    if power_from_pwm is not None:
        power_val = max(0.0, min(1.0, float(power_from_pwm)))
    _, score_effective = telemetry_robot_module.quantize_speed(cmd_sent, speed=power_val)
    if score_effective is None:
        score_effective = speed_score
    
    # For precision mode (very low speeds), quantize_speed may round up incorrectly.
    # If the quantized score is significantly higher than the intended speed_score,
    # use the intended score instead (prevents 2% when we intended 1%)
    if speed_score is not None and score_effective is not None:
        try:
            effective = int(round(float(score_effective)))
            intended = int(round(float(speed_score)))
            # If quantized score is just 1 above intended (common rounding error), use intended
            if effective > intended and effective - intended == 1 and intended < 10:
                score_effective = speed_score
        except (TypeError, ValueError):
            pass
    record_action_display(
        world,
        step,
        cmd,
        power_val,
        speed_score=score_effective,
        cmd_sent=cmd_sent,
        pwm=pwm_val,
        duration_ms=duration_used_ms,
        score_model=speed_score,
    )

    try:
        duration_model_ms = int(duration_ms)
    except (TypeError, ValueError):
        duration_model_ms = duration_used_ms
    return {
        "cmd_sent": cmd_sent,
        "power": power_val,
        "pwm": pwm_val,
        "duration_model_ms": int(duration_model_ms),
        "duration_ms": int(duration_used_ms),
        "score_model": speed_score,
        "score_effective": score_effective,
    }


def send_robot_command(robot, world, step, cmd, speed, speed_score=None, auto_mode=False):
    if robot is None or cmd is None:
        return None

    score_used = speed_score
    if score_used is None:
        if speed is None or speed <= 0:
            return None
        _, score_used = telemetry_robot_module.quantize_speed(cmd, speed=speed)
    if score_used is None:
        return None
    if auto_mode:
        score_used = telemetry_robot_module.normalize_speed_score(score_used)
        score_used = _cap_auto_speed_score(score_used)
    power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score_used)

    return send_robot_command_pwm(
        robot,
        world,
        step,
        cmd,
        power,
        pwm,
        duration_ms,
        speed_score=score_used,
        auto_mode=auto_mode,
    )


def step_uses_alignment_control(step, process_rules):
    obj_key = normalize_step_label(step)
    rules = (process_rules or {}).get(obj_key, {})
    controller = rules.get("controller")
    if controller == "replay":
        return False
    if controller == "align":
        return True
    alignment_metrics = rules.get("alignment_metrics")
    if isinstance(alignment_metrics, list):
        metrics = [metric for metric in alignment_metrics if metric]
        if not metrics:
            return False
        if set(metrics).issubset({"visible"}):
            return False
        return True
    success_gates = rules.get("success_gates") or {}
    gate_metrics = {metric for metric in success_gates.keys() if metric}
    if gate_metrics and gate_metrics.issubset({"visible"}):
        return False
    return bool(gate_metrics)


def step_is_nominal_only(step, process_rules):
    obj_key = normalize_step_label(step)
    rules = (process_rules or {}).get(obj_key, {})
    return bool(rules.get("nominalDemosOnly"))


def step_min_acts(step, process_rules):
    obj_key = normalize_step_label(step)
    rules = (process_rules or {}).get(obj_key, {})
    value = rules.get("minActs")
    try:
        min_acts = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min_acts)


def derive_action_speeds(steps, fallback_score=DEFAULT_SPEED_SCORE):
    def avg(cmds):
        values = [step.speed for step in steps if step.cmd in cmds]
        if not values:
            return None
        return sum(values) / len(values)

    turn = avg({"l", "r"})
    forward = avg({"f"})
    backward = avg({"b"})
    if backward is None:
        backward = forward

    def pick(value, cmd):
        if value is None:
            value = default_speed_for_cmd(cmd, fallback_score)
        cap = max_speed_for_cmd(cmd)
        return min(value, cap)

    turn = pick(turn, "l")
    forward = pick(forward, "f")
    backward = pick(backward, "b")
    return {
        "turn": turn,
        "scan": turn,
        "forward": forward,
        "backward": backward,
    }


def alignment_command(world, step, gate_bounds, speeds, preview=False):
    analytics = next_module.compute_alignment_decision(
        world=world,
        step=step,
        process_rules=world.process_rules or {},
        learned_rules=world.learned_rules or {},
        duration_s=CONTROL_DT,
    )
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") or 0.0
    reason = analytics.get("worst_metric") or "align"
    return cmd, speed, reason


def adjust_speed_for_find_brick(world, step, speed):
    if speed is None or speed <= 0:
        return speed
    obj_key = normalize_step_label(step)
    if obj_key != "FIND_BRICK":
        return speed
    brick = world.brick or {}
    if not brick.get("visible"):
        return speed
    return speed / FIND_BRICK_SLOW_FACTOR


def auto_cycle_phase(elapsed_s, *, force_check=False):
    """
    Auto-run replay speed cycle.

    Phase order: move (brief) -> check (longer) -> repeat.
    """
    if force_check:
        return "check"
    try:
        elapsed = float(elapsed_s or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    elapsed = max(0.0, elapsed)
    move_s = max(0.0, float(AUTO_CYCLE_MOVE_S))
    check_s = max(0.0, float(AUTO_CYCLE_CHECK_S))
    period = move_s + check_s
    if period <= 1e-6:
        return "check" if check_s > 0.0 else "move"
    t = elapsed % period
    return "move" if t < move_s else "check"


def success_visible_target(world, step):
    step_key = normalize_step_label(step)
    rules = world.process_rules.get(step_key, {}) if world.process_rules else {}
    visible_gate = (rules.get("success_gates") or {}).get("visible", {})
    if "min" in visible_gate:
        return bool(visible_gate.get("min"))
    if "max" in visible_gate:
        return bool(visible_gate.get("max"))
    return None




def step_elapsed_s(world):
    start_time = getattr(world, "_step_start_time", None)
    if start_time is None:
        return None
    return max(0.0, time.time() - start_time)


def step_confidence(success_ok, world, step):
    if not success_ok:
        world._success_confirm_frames = 0
        world._success_confirm_progress = None
        world._success_confirm_logged = False
        return 0.0
    world._success_confirm_frames = min(
        getattr(world, "_success_confirm_frames", 0) + 1,
        success_frames_required(step),
    )
    world._success_confirm_progress = None
    world._success_confirm_logged = False
    return 1.0


def apply_confidence_speed(speed, success_ok, confidence, world=None):
    if speed is None:
        return speed
    return speed


def apply_pursuit_speed(speed):
    if speed is None:
        return speed
    return min(max(0.0, speed), 1.0)


def log_confidence(world, confidence, step):
    if isinstance(step, MotionStep):
        return
    if confidence is None:
        return
    if normalize_step_label(step) == "EXIT_WALL":
        return
    if confidence <= 0.0 or confidence >= 1.0:
        return
    pct = int(round(confidence * 100))
    print(format_headline(f"[CONF] {step} {pct}% confidence", COLOR_WHITE))


def print_gate_summary_line(world, tracker=None):
    if tracker is not None:
        gate_utils.store_gate_summary(world, tracker)
    summary = gate_utils.consume_gate_summary(world)
    if not isinstance(summary, dict):
        return
    smooth_frames = int(getattr(telemetry_brick, "BRICK_SMOOTH_FRAMES", 1) or 1)
    text = gate_utils.format_gate_summary_line(summary, smooth_frames=smooth_frames)
    if not text:
        return
    print(
        format_headline(
            text,
            COLOR_WHITE,
        )
    )


def pause_after_fail(robot):
    if robot:
        robot.stop()
    time.sleep(FAIL_PAUSE_S)


def load_process_model(path=PROCESS_MODEL_FILE):
    if not path.exists():
        return {"steps": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"steps": {}}


def refresh_autobuild_config(path=PROCESS_MODEL_FILE, gate_checker_path=GATE_CHECKER_MODEL_FILE):
    cfg = dict(DEFAULT_AUTOBUILD_CONFIG)
    model = load_process_model(path)
    model_cfg = model.get("autobuild") if isinstance(model, dict) else None
    if isinstance(model_cfg, dict):
        cfg.update(model_cfg)
    lite_cfg = model.get("lite_gate_check") if isinstance(model, dict) else None
    apply_autobuild_config(cfg)
    apply_lite_gate_check_config(lite_cfg)
    refresh_gate_checker_config(gate_checker_path)
    return cfg


def write_process_model(model, path=PROCESS_MODEL_FILE):
    path.write_text(json.dumps(model, indent=4))


def step_metrics_map():
    metrics = {}
    for obj, items in telemetry_brick.METRICS_BY_STEP.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    for obj, items in telemetry_robot_module.METRICS_BY_STEP.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    return metrics


def _find_or_exit_step(step):
    step_key = normalize_step_label(step)
    step_lower = step_key.lower()
    return "find" in step_lower or "exit" in step_lower


def _no_start_gates_step(step):
    return _find_or_exit_step(step)


def step_requires_start_gates(step, process_rules=None):
    step_key = normalize_step_label(step)
    if _no_start_gates_step(step_key):
        return False
    cfg = (process_rules or {}).get(step_key, {})
    if isinstance(cfg, dict) and cfg.get("require_start_gates") is False:
        return False
    return True


def success_gate_metrics_for_step(metrics, step, step_rules=None):
    obj_key = normalize_step_label(step)
    rules = (step_rules or {}).get(obj_key, {})
    metrics_out = list(metrics or [])
    alignment_metrics = rules.get("alignment_metrics")
    if isinstance(alignment_metrics, list):
        metrics_out = [metric for metric in metrics_out if metric in alignment_metrics]
    allowed = rules.get("success_metrics")
    if isinstance(allowed, list):
        metrics_out = [metric for metric in metrics_out if metric in allowed]
    if obj_key in SUCCESS_GATE_EXCLUDE_ANGLE_STEPS:
        metrics_out = [metric for metric in metrics_out if metric != "angle_abs"]
    return metrics_out


def metric_value_from_state(state, metric):
    brick = state.get("brick") or {}
    if metric == "angle_abs":
        val = brick.get("angle")
        return abs(val) if val is not None else None
    if metric == "xAxis_offset_abs":
        val = brick.get("x_axis")
        if val is None:
            val = brick.get("offset_x")
        return float(val) if val is not None else None
    if metric == "dist":
        val = brick.get("dist")
        if val is None or val <= 0:
            return None
        return float(val)
    if metric == "visible":
        return bool(brick.get("visible"))
    if metric == "confidence":
        return brick.get("confidence")
    if metric == "lift_height":
        return state.get("lift_height")
    return None


def metric_direction(metric, step):
    if metric in next_module.METRIC_DIRECTIONS:
        return next_module.metric_direction_for_step(metric, step)
    return telemetry_robot_module.METRIC_DIRECTIONS.get(metric)


def collect_metric_values(segments, metrics):
    metric_values = {metric: [] for metric in metrics}
    for seg in segments:
        for state in select_tail_states(seg.get("states") or []):
            for metric in metrics:
                value = metric_value_from_state(state, metric)
                if value is None:
                    continue
                metric_values[metric].append(value)
    return metric_values


def collect_segments(logs):
    segments_by_obj = {}
    attempt_types = {}
    for _, data in logs:
        for seg in extract_attempt_segments(data):
            obj = normalize_step_label(seg.get("step"))
            seg_type = seg.get("type")
            if not obj or not seg_type:
                continue
            attempt_types.setdefault(obj, set()).add(seg_type)
            segments_by_obj.setdefault(obj, {}).setdefault(seg_type, []).append(seg)
    return segments_by_obj, attempt_types


def derive_start_gates(success_segments):
    start_gates = {}
    for obj, segs in success_segments.items():
        total_segs = 0
        visible_counts = []
        for seg in segs:
            states = select_head_states(seg.get("states") or [])
            if not states:
                continue
            total_segs += 1
            visible_ratio = sum(1 for s in states if (s.get("brick") or {}).get("visible")) / len(states)
            visible_counts.append(visible_ratio >= 0.5)

        if total_segs <= 0:
            continue
        consensus_ratio = sum(1 for v in visible_counts if v) / total_segs
        if consensus_ratio >= 0.5:
            start_gates[obj] = {"visible": {"min": True}}
        else:
            start_gates[obj] = {"visible": {"min": False}}
    return start_gates


def derive_success_gates(success_segments, scale_by_step=None, step_rules=None):
    metrics_by_obj = step_metrics_map()
    scale_by_step = scale_by_step or {}

    success_gates = {}
    for obj, segs in success_segments.items():
        metrics = success_gate_metrics_for_step(
            metrics_by_obj.get(obj, []),
            obj,
            step_rules,
        )
        if not metrics:
            continue
        metric_values = collect_metric_values(segs, metrics)

        visible_gate = None
        if metric_values.get("visible"):
            visible_ratio = sum(1 for v in metric_values["visible"] if v) / len(metric_values["visible"])
            visible_gate = {"min": visible_ratio >= 0.5}

        obj_gates = {}
        if visible_gate is not None:
            obj_gates["visible"] = visible_gate
        for metric, values in metric_values.items():
            if not values or metric == "visible":
                continue

            p10 = telemetry_brick.percentile(values, 0.1)
            p50 = telemetry_brick.percentile(values, 0.5)
            p90 = telemetry_brick.percentile(values, 0.9)
            if p50 is None:
                continue
            tol = None
            if p10 is not None and p90 is not None:
                tol = max(abs(p50 - p10), abs(p90 - p50))
            scale = scale_by_step.get(obj, 1.0)
            if tol is not None:
                tol *= scale
            if metric in MM_METRICS:
                if tol is None:
                    tol = MIN_MM_TOL
                else:
                    tol = max(tol, MIN_MM_TOL)

            stats = {"target": round_value(p50)}
            if tol is not None:
                stats["tol"] = round_value(tol)
            obj_gates[metric] = stats

        if obj_gates:
            success_gates[obj] = obj_gates
    return success_gates


def derive_success_durations(success_segments):
    durations = {}
    for obj, segs in success_segments.items():
        values = []
        for seg in segs:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                timestamps = [
                    state.get("timestamp")
                    for state in (seg.get("states") or [])
                    if state.get("timestamp") is not None
                ]
                if not timestamps:
                    timestamps = [
                        evt.get("timestamp")
                        for evt in (seg.get("events") or [])
                        if evt.get("timestamp") is not None
                    ]
                if timestamps:
                    start = min(timestamps)
                    end = max(timestamps)
            if start is None or end is None:
                continue
            duration = end - start
            if duration is None or duration <= 0:
                continue
            values.append(duration)
        if not values:
            continue
        p10 = telemetry_brick.percentile(values, 0.1)
        p50 = telemetry_brick.percentile(values, 0.5)
        p90 = telemetry_brick.percentile(values, 0.9)
        if p50 is None:
            continue
        tol = None
        if p10 is not None and p90 is not None:
            tol = max(abs(p50 - p10), abs(p90 - p50))
        stats = {"target": round_value(p50)}
        if tol is not None:
            stats["tol"] = round_value(tol)
        durations[obj] = stats
    return durations


def refine_success_gates_with_failures(success_gates, fail_segments, step_rules=None):
    if not success_gates or not fail_segments:
        return success_gates
    metrics_by_obj = step_metrics_map()
    for obj, gates in success_gates.items():
        if not isinstance(gates, dict):
            continue
        # Tighten success tolerances when failure samples overlap the current success window.
        metrics = success_gate_metrics_for_step(
            metrics_by_obj.get(obj, []),
            obj,
            step_rules,
        )
        if not metrics:
            continue
        fail_values = collect_metric_values(fail_segments.get(obj, []), metrics)
        for metric, stats in gates.items():
            if not isinstance(stats, dict):
                continue
            if metric == "visible":
                continue
            values = fail_values.get(metric)
            if not values:
                continue

            direction = metric_direction(metric, obj)
            if direction not in ("low", "high"):
                continue

            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                if direction == "low":
                    failure_floor = telemetry_brick.percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                    if failure_floor is None:
                        continue
                    success_max = target + tol
                    if failure_floor <= target or failure_floor >= success_max:
                        continue
                    stats["tol"] = round_value(failure_floor - target)
                else:
                    failure_ceiling = telemetry_brick.percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
                    if failure_ceiling is None:
                        continue
                    success_min = target - tol
                    if failure_ceiling >= target or failure_ceiling <= success_min:
                        continue
                    stats["tol"] = round_value(target - failure_ceiling)
                continue

            if direction == "low":
                failure_floor = telemetry_brick.percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                if failure_floor is None:
                    continue
                max_val = stats.get("max")
                if max_val is None or failure_floor >= max_val:
                    continue
                stats["max"] = round_value(failure_floor)
            else:
                failure_ceiling = telemetry_brick.percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
                if failure_ceiling is None:
                    continue
                min_val = stats.get("min")
                if min_val is None or failure_ceiling <= min_val:
                    continue
                stats["min"] = round_value(failure_ceiling)
    return success_gates


def tighten_x_axis_success_tolerance(success_gates, *, tol_mm=X_AXIS_OFFSET_ABS_TOL_MM):
    if not isinstance(success_gates, dict):
        return success_gates
    try:
        tol_val = abs(float(tol_mm))
    except (TypeError, ValueError):
        return success_gates
    if tol_val <= 0:
        return success_gates
    for obj, gates in success_gates.items():
        if not isinstance(gates, dict):
            continue
        stats = gates.get("xAxis_offset_abs")
        if not isinstance(stats, dict):
            continue
        stats["tol"] = round_value(tol_val)
        stats.pop("min", None)
        stats.pop("max", None)
    return success_gates


def update_process_model_from_demos(logs, path=PROCESS_MODEL_FILE):
    model = load_process_model(path)
    steps = model.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        model["steps"] = steps
    for name in list(steps.keys()):
        if normalize_step_label(name) in RETIRED_PROCESS_STEPS:
            steps.pop(name, None)

    segments_by_obj, attempt_types = collect_segments(logs)
    success_segments = {
        obj: segs.get("SUCCESS", [])
        for obj, segs in segments_by_obj.items()
        if segs.get("SUCCESS")
    }
    fail_segments = {
        obj: segs.get("FAIL", [])
        for obj, segs in segments_by_obj.items()
        if segs.get("FAIL")
    }

    start_gates = derive_start_gates(success_segments)
    wall_step_rules = telemetry_wall.load_wall_step_rules()
    wall_visible_only_steps = set()
    if isinstance(wall_step_rules, dict):
        for step_name, step_cfg in wall_step_rules.items():
            if not isinstance(step_cfg, dict):
                continue
            metrics = step_cfg.get("success_metrics")
            if not isinstance(metrics, list):
                continue
            metric_keys = {str(metric).strip() for metric in metrics if str(metric).strip()}
            if metric_keys == {"visible"}:
                wall_visible_only_steps.add(normalize_step_label(step_name))
    step_rules = {}
    if isinstance(wall_step_rules, dict):
        step_rules.update(wall_step_rules)
    if isinstance(steps, dict):
        step_rules.update(steps)
    for step_name in wall_visible_only_steps:
        cfg = step_rules.get(step_name)
        if not isinstance(cfg, dict):
            cfg = {}
            step_rules[step_name] = cfg
        # Wall telemetry marks these steps as visibility-only targets.
        cfg["success_metrics"] = ["visible"]
    success_gate_scales = derive_success_gate_scales(segments_by_obj, step_rules)

    # Train Policy if requested
    if USE_LEARNED_POLICY:
        global GLOBAL_POLICY
        print(format_headline("[LEARNING REPLAY] Training Policy from Demos...", COLOR_GREEN))
        GLOBAL_POLICY = helper_learning.BehavioralCloningPolicy()
        GLOBAL_POLICY.train(segments_by_obj)

    success_gates = derive_success_gates(
        success_segments,
        success_gate_scales,
        step_rules,
    )
    success_durations = derive_success_durations(success_segments)
    success_gates = refine_success_gates_with_failures(
        success_gates,
        fail_segments,
        step_rules,
    )
    tighten_x_axis_success_tolerance(success_gates)
    for obj, stats in []:
        if obj not in success_gates:
            continue

    all_steps = set(steps.keys())
    all_steps.update(start_gates.keys())
    all_steps.update(success_gates.keys())
    all_steps.update(attempt_types.keys())
    all_steps.update(wall_visible_only_steps)
    all_steps = {
        name
        for name in all_steps
        if normalize_step_label(name) not in RETIRED_PROCESS_STEPS
    }

    for obj in all_steps:
        cfg = steps.setdefault(obj, {})

        lock_all_gates = bool(cfg.get("lock_gates")) if isinstance(cfg, dict) else False
        lock_start_gates = lock_all_gates or (bool(cfg.get("lock_start_gates")) if isinstance(cfg, dict) else False)
        lock_success_gates = lock_all_gates or (bool(cfg.get("lock_success_gates")) if isinstance(cfg, dict) else False)

        if obj in start_gates and not _no_start_gates_step(obj):
            if not lock_start_gates or not isinstance(cfg.get("start_gates"), dict) or not cfg.get("start_gates"):
                cfg["start_gates"] = start_gates[obj]
        else:
            if not lock_start_gates:
                cfg.pop("start_gates", None)

        if obj in success_gates:
            if not lock_success_gates or not isinstance(cfg.get("success_gates"), dict) or not cfg.get("success_gates"):
                cfg["success_gates"] = success_gates[obj]
        else:
            if not lock_success_gates:
                cfg.pop("success_gates", None)
        if obj in SUCCESS_GATE_EXCLUDE_ANGLE_STEPS and isinstance(cfg.get("success_gates"), dict):
            cfg["success_gates"].pop("angle_abs", None)
        obj_key = normalize_step_label(obj)
        if obj_key in wall_visible_only_steps:
            gates = cfg.get("success_gates")
            if not isinstance(gates, dict):
                cfg["success_gates"] = {"visible": {"min": True}}
            else:
                visible_gate = gates.get("visible")
                if not isinstance(visible_gate, dict):
                    visible_gate = {"min": True}
                cfg["success_gates"] = {"visible": visible_gate}
        cfg.pop("fail_gates", None)
        visible_gate = cfg.get("success_gates", {}).get("visible", {})

        obj_types = attempt_types.get(obj, set())
        if obj_types == {"NOMINAL"}:
            cfg["nominalDemosOnly"] = True
        else:
            cfg.pop("nominalDemosOnly", None)

        for key in ("start_gates", "success_gates"):
            if key in cfg and not cfg[key]:
                cfg.pop(key)

    ordered = {}
    for name in steps.keys():
        ordered[name] = steps[name]
    for name in sorted(all_steps):
        if name not in ordered:
            ordered[name] = steps.get(name, {})
    model["steps"] = ordered

    write_process_model(model, path)
    return model


def build_motion_sequence(events):
    steps = []
    default_duration_s = max(0.001, telemetry_robot_module.ACT_DURATION_MS / 1000.0)
    for evt in events:
        if evt.get("type") == "action":
            cmd_name = evt.get("command")
            speed_score = evt.get("speedScore")
            power = evt.get("power", 0)
        elif evt.get("type") == "event":
            payload = evt.get("event") or {}
            cmd_name = payload.get("type")
            speed_score = payload.get("speedScore")
            power = payload.get("power", 0)
        else:
            continue

        cmd = ACTION_CMD_MAP.get(cmd_name)
        if not cmd:
            continue
        if speed_score is not None:
            power_val, _, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, speed_score)
            speed = max(0.0, min(1.0, float(power_val or 0)))
            duration_s = max(0.001, (duration_ms or 0) / 1000.0)
        else:
            speed = max(0.0, min(1.0, float(power or 0) / 255.0))
            duration_s = default_duration_s
        if speed <= 0:
            continue
        score_val = None
        if speed_score is not None:
            try:
                score_val = int(speed_score)
            except (TypeError, ValueError):
                score_val = None
        steps.append(MotionStep(cmd, speed, duration_s, cmd_name, speed_score=score_val))
    return steps


def merge_motion_steps(steps, speed_tol=0.02):
    if not steps:
        return []
    merged = [steps[0]]
    for step in steps[1:]:
        last = merged[-1]
        if (
            step.cmd == last.cmd
            and step.speed_score == last.speed_score
            and abs(step.speed - last.speed) <= speed_tol
        ):
            last.duration_s += step.duration_s
        else:
            merged.append(step)
    return merged


def nominal_actions_from_events(events):
    actions = []
    default_duration_s = max(0.001, telemetry_robot_module.ACT_DURATION_MS / 1000.0)
    for evt in events:
        if evt.get("type") == "action":
            cmd_name = evt.get("command")
            speed_score = evt.get("speedScore")
            power = evt.get("power", 0)
        elif evt.get("type") == "event":
            payload = evt.get("event") or {}
            cmd_name = payload.get("type")
            speed_score = payload.get("speedScore")
            power = payload.get("power", 0)
        else:
            continue
        cmd = ACTION_CMD_MAP.get(cmd_name)
        if not cmd:
            continue
        score = None
        if speed_score is not None:
            try:
                score = int(speed_score)
            except (TypeError, ValueError):
                score = None
        if score is not None:
            _, _, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
            duration_s = max(0.001, (duration_ms or 0) / 1000.0)
        else:
            duration_s = default_duration_s
        actions.append(
            {
                "cmd": cmd,
                "speed_score": score,
                "power": power,
                "duration_s": duration_s,
            }
        )
    return actions


def min_acts_prefix_duration(events, min_acts):
    if min_acts <= 0:
        return 0.0
    actions = nominal_actions_from_events(events)
    if not actions:
        return 0.0
    cap = min(int(min_acts), len(actions))
    total = 0.0
    for action in actions[:cap]:
        total += max(0.0, float(action.get("duration_s") or 0.0))
    return total


def smooth_motion_steps(steps, speed_score=DEFAULT_SPEED_SCORE, step_s=SMOOTH_STEP_S):
    if not steps:
        return []
    smooth = []
    for step in steps:
        total_duration = step.duration_s
        chunks = max(1, int(math.ceil(total_duration / step_s)))
        for idx in range(chunks):
            duration = min(step_s, total_duration - (idx * step_s))
            if duration <= 0:
                continue
            base_speed = default_speed_for_cmd(step.cmd, speed_score)
            capped_speed = min(base_speed, max_speed_for_cmd(step.cmd))
            smooth.append(
                MotionStep(
                    step.cmd,
                    capped_speed,
                    duration,
                    step.label,
                    speed_score=step.speed_score,
                )
            )
    return smooth


def select_demo_segment(segments_by_obj, step, nominal_only):
    obj_key = normalize_step_label(step)
    if not obj_key:
        return None, None

    if nominal_only:
        nominal_segs = segments_by_obj.get(obj_key, {}).get("NOMINAL", [])
        if len(nominal_segs) > 1:
            print(
                format_headline(
                    f"[FAIL] {obj_key}: {len(nominal_segs)} nominal demos found; expected exactly 1.",
                    COLOR_RED,
                )
            )
            return None, None

    prefer = ["NOMINAL", "SUCCESS"] if nominal_only else ["SUCCESS", "NOMINAL"]
    candidates = []
    chosen_type = None
    for attempt_type in prefer:
        candidates = segments_by_obj.get(obj_key, {}).get(attempt_type, [])
        if candidates:
            chosen_type = attempt_type
            break

    if not candidates:
        return None, None

    def score(seg):
        events = seg.get("events") or []
        duration = 0.0
        if seg.get("start") is not None and seg.get("end") is not None:
            duration = seg["end"] - seg["start"]
        return (len(events), duration)

    candidates.sort(key=score, reverse=True)
    return candidates[0], chosen_type


def _record_smoothed_frame_snapshot(world):
    if world is None:
        return
    frame_id = int(getattr(world, "_frame_id", 0) or 0)
    if frame_id <= 0:
        return
    history = getattr(world, "_smoothed_frame_history", None)
    if history is None:
        history = []
        world._smoothed_frame_history = history
    if history and int(history[-1].get("frame_id", 0) or 0) == frame_id:
        return
    brick = world.brick or {}
    x_axis = brick.get("x_axis")
    if x_axis is None:
        x_axis = brick.get("offset_x")
    history.append(
        {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "visible": bool(brick.get("visible")),
            "dist": float(brick.get("dist", 0.0) or 0.0),
            "angle": float(brick.get("angle", 0.0) or 0.0),
            "x_axis": float(x_axis or 0.0),
            "offset_x": float(brick.get("offset_x", x_axis if x_axis is not None else 0.0) or 0.0),
            "confidence": float(brick.get("confidence", 0.0) or 0.0),
        }
    )
    if len(history) > 120:
        del history[:-120]


def update_world_from_vision(world, vision, log=True):
    now = time.time()
    sec = int(now)
    ms = int((now - sec) * 1000)
    cam_times = getattr(world, "_camera_frame_times", [])
    stamp = (sec, ms)
    duplicate_timestamp = stamp in cam_times
    cam_times.append((sec, ms))
    if len(cam_times) > 10:
        cam_times.pop(0)
    world._camera_frame_times = cam_times
    world._camera_dupe_ms = duplicate_timestamp
    if duplicate_timestamp:
        world._camera_dupe_count = int(getattr(world, "_camera_dupe_count", 0)) + 1
    fps = None
    if len(cam_times) >= 2:
        first = cam_times[0][0] + cam_times[0][1] / 1000.0
        last = cam_times[-1][0] + cam_times[-1][1] / 1000.0
        span = last - first
        if span > 1e-3:
            fps = (len(cam_times) - 1) / span
    world._camera_fps = fps

    found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
    if dist == -1:
        found = False
        angle = 0.0
        dist = 0.0
        offset_x = 0.0
        conf = 0.0
        cam_h = 0.0
        brick_above = False
        brick_below = False
    frame = {
        "found": bool(found),
        "dist": float(dist),
        "angle": float(angle),
        "offset_x": float(offset_x),
        "conf": float(conf),
        "cam_h": float(cam_h),
        "brick_above": bool(brick_above),
        "brick_below": bool(brick_below),
        "stamp": stamp,
    }
    buffer = getattr(world, "_brick_frame_buffer", None)
    if buffer is None:
        buffer = []
        setattr(world, "_brick_frame_buffer", buffer)
    fresh_input_frame = False
    if not duplicate_timestamp and all(entry.get("stamp") != stamp for entry in buffer):
        buffer.append(frame)
        fresh_input_frame = True
    if fresh_input_frame:
        # Frame counter tracks fresh camera inputs (not just smoothed gate frames).
        world._frame_id = getattr(world, "_frame_id", 0) + 1
    if len(buffer) > telemetry_brick.BRICK_SMOOTH_FRAMES:
        buffer.pop(0)
    avg = None
    obs_note = None
    if len(buffer) >= telemetry_brick.BRICK_SMOOTH_FRAMES:
        avg, should_reset, obs_note = telemetry_brick._filtered_brick_frame_average(buffer)
        if should_reset and avg is None:
            buffer.clear()
    if obs_note:
        world._last_obs_note = obs_note
    if avg is not None:
        world._brick_inconsistent = False
        world.update_vision(
            avg["found"],
            avg["dist"],
            avg["angle"],
            avg["conf"],
            avg["offset_x"],
            avg["cam_h"],
            avg["brick_above"],
            avg["brick_below"],
        )
        if fresh_input_frame:
            _record_smoothed_frame_snapshot(world)
        if log:
            current_obj = getattr(world, "step_state", None)
            current_name = getattr(current_obj, "value", None)
            if normalize_step_label(current_name) == "EXIT_WALL":
                pass
            elif not getattr(world, "suppress_brick_state_log", False):
                pass
    elif len(buffer) >= telemetry_brick.BRICK_SMOOTH_FRAMES:
        # Mark inconsistent snapshot: force confidence to 0 and visible false until we stabilize.
        world._brick_inconsistent = True
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.brick["dist"] = 0.0
        world.brick["angle"] = 0.0
        world.brick["offset_x"] = 0.0
        world.brick["x_axis"] = 0.0
        world.brick["brickAbove"] = False
        world.brick["brickBelow"] = False
    else:
        world._brick_inconsistent = True
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.brick["dist"] = 0.0
        world.brick["angle"] = 0.0
        world.brick["offset_x"] = 0.0
        world.brick["x_axis"] = 0.0
        world.brick["brickAbove"] = False
        world.brick["brickBelow"] = False
    visible_now = bool(world.brick.get("visible"))
    lost_frames = getattr(world, "_visibility_lost_frames", 0)
    if visible_now:
        world._visibility_lost_frames = 0
        world._vision_lost_reason = None
    else:
        world._visibility_lost_frames = min(lost_frames + 1, 1_000_000)
        reasons = []
        if getattr(world, "_camera_dupe_ms", False):
            reasons.append("frame timestamp repeated")
        if obs_note:
            reasons.append(obs_note)
        if not reasons:
            reasons.append("no brick detected")
        world._vision_lost_reason = "; ".join(reasons)
    update_stream_frame(world, vision)


def _gate_value_line(metric, value, stats):
    if value is None:
        val_str = "n/a"
    elif isinstance(value, bool):
        val_str = "true" if value else "false"
    else:
        val_str = f"{float(value):.1f}"
    gate_desc = ""
    if isinstance(stats, dict):
        target = stats.get("target")
        tol = stats.get("tol")
        min_val = stats.get("min")
        max_val = stats.get("max")
        # Round values for display (1 decimal max)
        if target is not None:
            target = round(float(target), 1)
        if tol is not None:
            tol = round(float(tol), 1)
        if min_val is not None:
            min_val = round(float(min_val), 1)
        if max_val is not None:
            max_val = round(float(max_val), 1)
        if target is not None and tol is not None:
            gate_desc = f"{target:.1f} +/- {tol:.1f}"
        elif min_val is not None and max_val is not None:
            gate_desc = f"{min_val:.1f}-{max_val:.1f}"
        elif min_val is not None:
            gate_desc = f">={min_val:.1f}"
        elif max_val is not None:
            gate_desc = f"<={max_val:.1f}"
    suffix = f" ({gate_desc})" if gate_desc else ""
    return f"{metric}: {val_str}{suffix}"


def _hold_reason(world, step, analytics):
    reasons = []
    align = (analytics or {}).get("align") or {}
    brick = world.brick or {}

    if getattr(world, "_brick_inconsistent", False) and brick.get("visible"):
        obs_note = getattr(world, "_last_obs_note", None)
        reasons.append(obs_note or "inconsistent brick frames")

    if not brick.get("visible"):
        lost_reason = getattr(world, "_vision_lost_reason", None)
        if lost_reason:
            filtered = []
            for part in (p.strip() for p in lost_reason.split(";")):
                if not part:
                    continue
                lower = part.lower()
                if "inconsistent" in lower or "unique frame" in lower:
                    continue
                filtered.append(part)
            if filtered:
                reasons.append("; ".join(filtered))
            else:
                reasons.append("brick not visible")
        else:
            reasons.append("brick not visible")

    brick_start = telemetry_brick.evaluate_start_gates(world, step, {}, world.process_rules)
    wall_start = telemetry_wall.evaluate_start_gates(world, step, world.wall_envelope)
    robot_start = telemetry_robot_module.evaluate_start_gates(world, step, {}, world.process_rules)
    if not (brick_start.ok and wall_start.ok and robot_start.ok):
        reasons.extend(brick_start.reasons or [])
        reasons.extend(wall_start.reasons or [])
        reasons.extend(robot_start.reasons or [])

    if align and align.get("cmd") is None:
        worst = align.get("worst_metric")
        if worst:
            reasons.append(f"no cmd for {worst}")
        else:
            progress = align.get("progress")
            if progress is not None and progress >= 0.99:
                reasons.append("within alignment gates")

    if not reasons:
        return "alignment controller returned no action"

    deduped = []
    for reason in reasons:
        if not reason:
            continue
        if reason not in deduped:
            deduped.append(reason)
    if not deduped:
        return "alignment controller returned no action"
    if len(deduped) > 2:
        return "; ".join(deduped[:2]) + "; ..."
    return "; ".join(deduped)


STREAM_GREEN = (0, 255, 0)
STREAM_RED = (0, 0, 255)
STREAM_WHITE = (255, 255, 255)
STREAM_ORANGE = (0, 165, 255)
STREAM_GRAY = (180, 180, 180)


def _gate_requirement_text(stats):
    if not isinstance(stats, dict):
        return ""
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        op = ">=" if min_val else "<="
        return f"{op}{_fmt_gate_value(min_val)}"
    if isinstance(max_val, bool):
        op = "<=" if max_val is False else ">="
        return f"{op}{_fmt_gate_value(max_val)}"
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        return f"={_fmt_gate_value(target)}+/-{_fmt_gate_value(tol)}"
    if min_val is not None and max_val is not None:
        return f"={_fmt_gate_value(min_val)}..{_fmt_gate_value(max_val)}"
    if min_val is not None:
        return f">={_fmt_gate_value(min_val)}"
    if max_val is not None:
        return f"<={_fmt_gate_value(max_val)}"
    return ""


def _auto_gate_snapshot_text(world, step, *, include_requirements):
    entries = _success_gate_entries(world, step)
    if not entries:
        return "none"
    parts = []
    for entry in entries:
        metric = entry.get("metric")
        if not metric:
            continue
        if metric == "visible":
            value = entry.get("effective_visible")
            if value is None:
                value = entry.get("raw_visible")
            if value is None:
                value = entry.get("value")
            value = bool(value) if value is not None else None
        else:
            value = entry.get("value")
        value_text = "n/a" if value is None else _fmt_gate_value(value)
        item = f"{metric}={value_text}"
        if include_requirements:
            requirement = _gate_requirement_text(entry.get("stats") or {})
            if requirement:
                item += f" ({requirement})"
        parts.append(item)
    return ", ".join(parts) if parts else "none"


def _metric_display_unit(metric):
    if metric in MM_METRICS or metric == "xAxis_offset_abs":
        return "mm"
    if metric == "angle_abs":
        return "deg"
    return ""


def _fmt_metric_value_with_unit(metric, value):
    if value is None:
        return "n/a"
    if metric == "visible":
        return _fmt_gate_value(bool(value))
    value_text = _fmt_gate_value(value)
    unit = _metric_display_unit(metric)
    if unit:
        return f"{value_text}{unit}"
    return str(value_text)


def _auto_diag_value_from_entry(entry):
    metric = entry.get("metric")
    if metric == "visible":
        value = entry.get("effective_visible")
        if value is None:
            value = entry.get("raw_visible")
        if value is None:
            value = entry.get("value")
        return bool(value) if value is not None else None
    return entry.get("value")


def _gate_distance_to_success(metric, value, stats, direction):
    stats = stats or {}
    if metric == "visible":
        if value is None:
            return 1.0
        bool_val = bool(value)
        min_val = stats.get("min")
        max_val = stats.get("max")
        if isinstance(min_val, bool):
            return 0.0 if bool_val == min_val else 1.0
        if isinstance(max_val, bool):
            return 0.0 if bool_val == max_val else 1.0
        numeric = 1.0 if bool_val else 0.0
        try:
            min_num = float(min_val) if min_val is not None else None
        except (TypeError, ValueError):
            min_num = None
        try:
            max_num = float(max_val) if max_val is not None else None
        except (TypeError, ValueError):
            max_num = None
        distance = 0.0
        if min_num is not None and numeric < min_num:
            distance = max(distance, min_num - numeric)
        if max_num is not None and numeric > max_num:
            distance = max(distance, numeric - max_num)
        return distance

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        try:
            target_num = float(target)
            tol_num = abs(float(tol))
        except (TypeError, ValueError):
            target_num = None
            tol_num = None
        if target_num is not None and tol_num is not None:
            if direction == "high":
                return max(0.0, (target_num - tol_num) - numeric)
            if direction == "low":
                return max(0.0, numeric - (target_num + tol_num))
            return max(0.0, abs(numeric - target_num) - tol_num)

    try:
        min_num = float(stats.get("min")) if stats.get("min") is not None else None
    except (TypeError, ValueError):
        min_num = None
    try:
        max_num = float(stats.get("max")) if stats.get("max") is not None else None
    except (TypeError, ValueError):
        max_num = None
    if min_num is not None and numeric < min_num:
        return min_num - numeric
    if max_num is not None and numeric > max_num:
        return numeric - max_num
    return 0.0


def _auto_diag_metric_rank(metric):
    if metric in AUTO_DIAG_OFFSET_PRIORITY:
        return AUTO_DIAG_OFFSET_PRIORITY.index(metric)
    if metric == "angle_abs":
        return len(AUTO_DIAG_OFFSET_PRIORITY) + 1
    if metric == "visible":
        return len(AUTO_DIAG_OFFSET_PRIORITY) + 10
    return len(AUTO_DIAG_OFFSET_PRIORITY) + 5


def _auto_diag_focus_from_entry(entry, step):
    if not isinstance(entry, dict):
        return None
    metric = entry.get("metric")
    if not metric:
        return None
    stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
    value = _auto_diag_value_from_entry(entry)
    direction = metric_direction(metric, step)
    matched = _gate_entry_matches(metric, value, stats, direction)
    distance = _gate_distance_to_success(metric, value, stats, direction)
    return {
        "metric": metric,
        "value": value,
        "stats": dict(stats),
        "direction": direction,
        "matched": bool(matched),
        "distance": distance,
    }


def _pick_most_relevant_focus(candidates):
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -(float(item.get("distance")) if item.get("distance") is not None else -1.0),
            _auto_diag_metric_rank(item.get("metric")),
        ),
    )[0]


def _select_auto_diag_focus(entries, step):
    focuses = []
    for entry in entries or []:
        focus = _auto_diag_focus_from_entry(entry, step)
        if focus:
            focuses.append(focus)
    if not focuses:
        return None

    failing = [focus for focus in focuses if not focus.get("matched")]
    failing_offsets = [
        focus
        for focus in failing
        if focus.get("metric") in AUTO_DIAG_OFFSET_PRIORITY and focus.get("distance") is not None
    ]
    if failing_offsets:
        return _pick_most_relevant_focus(failing_offsets)

    failing_numeric = [
        focus
        for focus in failing
        if focus.get("metric") != "visible" and focus.get("distance") is not None
    ]
    if failing_numeric:
        return _pick_most_relevant_focus(failing_numeric)

    failing_visible = [focus for focus in failing if focus.get("metric") == "visible"]
    if failing_visible:
        return _pick_most_relevant_focus(failing_visible)

    numeric = [
        focus
        for focus in focuses
        if focus.get("metric") in AUTO_DIAG_OFFSET_PRIORITY and focus.get("distance") is not None
    ]
    if numeric:
        return _pick_most_relevant_focus(numeric)
    return _pick_most_relevant_focus(focuses)


def _capture_auto_diag_focus(world, step):
    return _select_auto_diag_focus(_success_gate_entries(world, step), step)


def _find_entry_by_metric(entries, metric):
    for entry in entries or []:
        if entry.get("metric") == metric:
            return entry
    return None


def _is_single_boolean_success_gate(entries):
    if not isinstance(entries, list) or len(entries) != 1:
        return False
    entry = entries[0] if entries else None
    if not isinstance(entry, dict):
        return False
    if entry.get("metric") != "visible":
        return False
    stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
    min_val = stats.get("min")
    max_val = stats.get("max")
    return isinstance(min_val, bool) or isinstance(max_val, bool)


def _auto_diag_focus_observation(focus):
    if not isinstance(focus, dict):
        return "none"
    metric = focus.get("metric")
    if not metric:
        return "none"
    value_text = _fmt_metric_value_with_unit(metric, focus.get("value"))
    return f"{metric}={value_text}"


def _auto_diag_focus_snapshot(focus, *, include_requirement=True):
    if not isinstance(focus, dict):
        return "none"
    metric = focus.get("metric")
    if not metric:
        return "none"
    value_text = _fmt_metric_value_with_unit(metric, focus.get("value"))
    snapshot = f"{metric}={value_text}"
    if include_requirement:
        requirement = _gate_requirement_text(focus.get("stats") or {})
        if requirement:
            snapshot += f" ({requirement})"
    return snapshot


def _auto_diag_delta_phrase(metric, pre_distance, post_distance):
    if pre_distance is None or post_distance is None:
        plain = "unknown change to the success gates"
        return plain, plain, "unknown"
    try:
        delta = float(pre_distance) - float(post_distance)
    except (TypeError, ValueError):
        plain = "unknown change to the success gates"
        return plain, plain, "unknown"

    abs_delta = abs(delta)
    unit = _metric_display_unit(metric)
    if not unit and metric == "visible":
        unit = " gate"
    magnitude_text = f"{abs_delta:.1f}{unit}"

    if delta > 1e-6:
        return (
            f"{magnitude_text} closer to the success gates",
            f"{COLOR_GREEN}{magnitude_text}{COLOR_RESET} closer to the success gates",
            "closer",
        )
    if delta < -1e-6:
        return (
            f"{magnitude_text} further from the success gates",
            f"{COLOR_RED}{magnitude_text}{COLOR_RESET} further from the success gates",
            "backward",
        )
    return (
        f"{magnitude_text} unchanged vs the success gates",
        f"{COLOR_WHITE}{magnitude_text}{COLOR_RESET} unchanged vs the success gates",
        "unchanged",
    )


def _ansi_to_stream_segments(colored_text):
    text = str(colored_text or "")
    if not text:
        return []
    color_map = {
        COLOR_GREEN: STREAM_GREEN,
        COLOR_RED: STREAM_RED,
        COLOR_WHITE: STREAM_WHITE,
        COLOR_ORANGE_BRIGHT: STREAM_ORANGE,
        COLOR_GRAY: STREAM_GRAY,
    }
    current = STREAM_WHITE
    segments = []
    chunk = []
    idx = 0
    while idx < len(text):
        if text[idx] == "\033" and text.startswith("\033[", idx):
            end = text.find("m", idx)
            if end == -1:
                chunk.append(text[idx:])
                break
            if chunk:
                segments.append(("".join(chunk), current))
                chunk = []
            code = text[idx : end + 1]
            if code == COLOR_RESET:
                current = STREAM_WHITE
            elif code in color_map:
                current = color_map[code]
            idx = end + 1
            continue
        chunk.append(text[idx])
        idx += 1
    if chunk:
        segments.append(("".join(chunk), current))
    if not segments:
        return [(text, STREAM_WHITE)]
    return [seg for seg in segments if seg and seg[0]]


def _strip_auto_prefix_segments(segments):
    if not isinstance(segments, list):
        return []
    cleaned = []
    trimmed = False
    for seg in segments:
        if not isinstance(seg, (tuple, list)) or not seg:
            continue
        txt = str(seg[0])
        color = seg[1] if len(seg) > 1 else STREAM_WHITE
        if not trimmed:
            if txt.startswith("[AUTO] "):
                txt = txt[len("[AUTO] "):]
                trimmed = True
            elif txt.startswith("AUTO: "):
                txt = txt[len("AUTO: "):]
                trimmed = True
            elif txt.startswith("[AUTO]"):
                txt = txt[len("[AUTO]") :].lstrip()
                trimmed = True
        if txt:
            cleaned.append((txt, color))
    return cleaned


def _strip_known_ansi(text):
    if text is None:
        return ""
    out = str(text)
    for code in (
        COLOR_RESET,
        COLOR_GREEN,
        COLOR_RED,
        COLOR_WHITE,
        COLOR_GRAY,
        COLOR_CYAN,
        COLOR_YELLOW,
        COLOR_ORANGE_BRIGHT,
    ):
        out = out.replace(str(code), "")
    return out


def _auto_action_cmd_score(action_text):
    text = str(action_text or "").strip()
    upper = text.upper()
    cmd = "?"
    for candidate in ("F", "B", "L", "R", "U", "D"):
        if upper.startswith(candidate):
            cmd = candidate
            break
    match = re.search(r"(\d+)%", text)
    score = int(match.group(1)) if match else 0
    return cmd, score


def _build_auto_step_diagnostic(
    world,
    step,
    action_text,
    pre_gate_text,
    *,
    pre_focus=None,
    success_override=None,
):
    step_key = normalize_step_label(step)
    post_entries = _success_gate_entries(world, step)
    selected_pre = pre_focus if isinstance(pre_focus, dict) else None
    if selected_pre is None:
        selected_pre = _select_auto_diag_focus(post_entries, step)

    metric = selected_pre.get("metric") if isinstance(selected_pre, dict) else None
    if metric:
        post_entry = _find_entry_by_metric(post_entries, metric)
        post_focus = _auto_diag_focus_from_entry(post_entry, step) if post_entry else None
    else:
        post_focus = None
    if post_focus is None:
        post_focus = _select_auto_diag_focus(post_entries, step)

    pre_obs = _auto_diag_focus_observation(selected_pre)
    if pre_obs == "none":
        pre_obs = str(pre_gate_text or "none")
    pre_snapshot = str(pre_gate_text or "none")
    if not pre_snapshot or pre_snapshot.strip().lower() == "none":
        pre_snapshot = pre_obs
    if (
        step_key in AUTO_DIAG_SINGLE_OFFSET_PRECHECK_STEPS
        and isinstance(selected_pre, dict)
        and selected_pre.get("metric") in AUTO_DIAG_OFFSET_PRIORITY
    ):
        pre_snapshot = _auto_diag_focus_snapshot(selected_pre, include_requirement=True)
    post_obs = _auto_diag_focus_observation(post_focus)

    pre_distance = selected_pre.get("distance") if isinstance(selected_pre, dict) else None
    post_distance = post_focus.get("distance") if isinstance(post_focus, dict) else None
    delta_plain, delta_colored, delta_class = _auto_diag_delta_phrase(metric, pre_distance, post_distance)

    action_plain = str(action_text)
    post_tail = f" ({post_obs})" if post_obs and post_obs != "none" else ""
    action_cmd, action_score = _auto_action_cmd_score(action_plain)
    pre_obs_text = pre_snapshot

    success_hit = False
    result_delta_obj = None

    if _is_single_boolean_success_gate(post_entries):
        if isinstance(success_override, bool):
            success_hit = bool(success_override)
        else:
            success_hit = bool(post_focus.get("matched")) if isinstance(post_focus, dict) else False
        pre_matched = selected_pre.get("matched") if isinstance(selected_pre, dict) else None
        post_matched = post_focus.get("matched") if isinstance(post_focus, dict) else None
        if isinstance(pre_matched, bool) and isinstance(post_matched, bool):
            if (not pre_matched) and post_matched:
                delta_class = "closer"
            elif pre_matched and (not post_matched):
                delta_class = "backward"
            else:
                delta_class = "unchanged"
        elif success_hit:
            delta_class = "closer"
        else:
            delta_class = "unknown"
        if success_hit:
            result_delta_obj = {
                "delta_class": "closer",
                "delta_text": f"success gates met{post_tail}",
            }
        else:
            result_delta_obj = {
                "delta_class": delta_class,
                "delta_text": f"not meeting success gates{post_tail}",
            }
    else:
        result_delta_obj = {
            "delta_class": delta_class,
            "delta_text": f"{delta_plain}{post_tail}",
        }

    trial_num = int(getattr(world, "loop_id", 0) or 0)
    phase_upper = str(step_key or str(step) or "?").upper()
    action_text_colored = f"{COLOR_WHITE}{action_cmd.lower()} {int(action_score)}%{COLOR_RESET}"
    motor_suffix = ""
    action_plain_strip = str(action_plain).strip()
    if "(" in action_plain_strip and action_plain_strip.endswith(")"):
        try:
            motor_suffix_raw = action_plain_strip[action_plain_strip.index("("):]
        except ValueError:
            motor_suffix_raw = ""
        if motor_suffix_raw:
            motor_suffix = f" {COLOR_GRAY}{motor_suffix_raw}{COLOR_RESET}"
    delta_class = (result_delta_obj or {}).get("delta_class", "unknown")
    delta_text = str((result_delta_obj or {}).get("delta_text", "") or "")
    if delta_class == "closer":
        result_color = COLOR_GREEN
    elif delta_class == "backward":
        result_color = COLOR_RED
    else:
        result_color = COLOR_YELLOW
    colored = (
        f"{COLOR_CYAN}[T{trial_num} {phase_upper}]{COLOR_RESET} "
        f"{COLOR_WHITE}{pre_obs_text}{COLOR_RESET} → "
        f"{action_text_colored}{motor_suffix}"
    )
    if delta_text:
        colored += f" {result_color}{delta_text}{COLOR_RESET}"
    plain = _strip_known_ansi(colored)
    return {
        "plain": plain,
        "colored": colored,
        "success_hit": bool(success_hit),
        "delta_class": (result_delta_obj or {}).get("delta_class"),
        "segments": _ansi_to_stream_segments(colored),
    }


def auto_diag_focus_text(world, step):
    focus = _capture_auto_diag_focus(world, step)
    if not isinstance(focus, dict):
        return "none"
    metric = focus.get("metric")
    if not metric:
        return "none"
    value_text = _fmt_metric_value_with_unit(metric, focus.get("value"))
    requirement = _gate_requirement_text(focus.get("stats") or {})
    if requirement:
        return f"{metric}={value_text} ({requirement})"
    return f"{metric}={value_text}"


def _current_success_ok(world, step):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    lite_brick_ok = _evaluate_lite_gate_brick_success(world, step, process_rules)
    if lite_brick_ok is None:
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 0
        world._gatecheck_lite_collected = 0
        visibility_grace_s = _visibility_grace_s(world, step)
        brick_success = telemetry_brick.evaluate_success_gates(
            world,
            step,
            telemetry_rules,
            process_rules,
            visibility_grace_s=visibility_grace_s,
        )
        brick_ok = bool(brick_success.ok)
    else:
        brick_ok = bool(lite_brick_ok)
    wall_ok = bool(telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope).ok)
    robot_ok = bool(telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules).ok)
    return brick_ok and wall_ok and robot_ok


def _store_auto_step_diagnostic(world, step, line, sent_snapshot=None, *, colored_line=None, segments=None):
    if world is None:
        return
    world._last_auto_step_diag_line = str(line)
    world._last_auto_step_diag_colored = str(colored_line) if colored_line is not None else None
    if isinstance(segments, list):
        world._last_auto_step_diag_segments = [tuple(seg) for seg in segments if isinstance(seg, (tuple, list))]
    else:
        world._last_auto_step_diag_segments = None
    world._last_auto_step_diag_step = normalize_step_label(step)
    world._last_auto_step_diag_time = time.time()
    world._last_auto_step_diag_sent = sent_snapshot if isinstance(sent_snapshot, dict) else None


def clear_pending_auto_step_diagnostic(world):
    if world is None:
        return
    world._pending_auto_step_diag = None


def reset_auto_step_action_stats(world, step):
    if world is None:
        return
    world._auto_step_action_stats = {
        "step": normalize_step_label(step),
        "total": 0,
        "closer": 0,
        "backward": 0,
        "unchanged": 0,
        "unknown": 0,
    }


def consume_auto_step_action_stats(world, step=None):
    if world is None:
        return {
            "step": normalize_step_label(step) if step is not None else None,
            "total": 0,
            "closer": 0,
            "backward": 0,
            "unchanged": 0,
            "unknown": 0,
        }
    stats = getattr(world, "_auto_step_action_stats", None)
    step_key = normalize_step_label(step) if step is not None else None
    if not isinstance(stats, dict):
        return {
            "step": step_key,
            "total": 0,
            "closer": 0,
            "backward": 0,
            "unchanged": 0,
            "unknown": 0,
        }
    stats_step = normalize_step_label(stats.get("step"))
    if step_key is not None and stats_step != step_key:
        return {
            "step": step_key,
            "total": 0,
            "closer": 0,
            "backward": 0,
            "unchanged": 0,
            "unknown": 0,
        }
    return {
        "step": stats_step,
        "total": int(stats.get("total") or 0),
        "closer": int(stats.get("closer") or 0),
        "backward": int(stats.get("backward") or 0),
        "unchanged": int(stats.get("unchanged") or 0),
        "unknown": int(stats.get("unknown") or 0),
    }


def _record_auto_step_action_stats(world, step, delta_class):
    if world is None:
        return
    step_key = normalize_step_label(step)
    stats = getattr(world, "_auto_step_action_stats", None)
    if not isinstance(stats, dict) or normalize_step_label(stats.get("step")) != step_key:
        stats = {
            "step": step_key,
            "total": 0,
            "closer": 0,
            "backward": 0,
            "unchanged": 0,
            "unknown": 0,
        }
        world._auto_step_action_stats = stats
    stats["total"] = int(stats.get("total") or 0) + 1
    if delta_class == "closer":
        stats["closer"] = int(stats.get("closer") or 0) + 1
    elif delta_class == "backward":
        stats["backward"] = int(stats.get("backward") or 0) + 1
    elif delta_class == "unchanged":
        stats["unchanged"] = int(stats.get("unchanged") or 0) + 1
    else:
        stats["unknown"] = int(stats.get("unknown") or 0) + 1


def queue_auto_step_diagnostic(
    world,
    step,
    action_text,
    pre_gate_text,
    *,
    pre_focus=None,
    sent_snapshot=None,
):
    if world is None:
        return
    world._pending_auto_step_diag = {
        "step": normalize_step_label(step),
        "action_text": str(action_text),
        "pre_gate_text": str(pre_gate_text or "none"),
        "pre_focus": dict(pre_focus) if isinstance(pre_focus, dict) else None,
        "frame_id": int(getattr(world, "_frame_id", 0) or 0),
        "timestamp": time.time(),
        "sent_snapshot": sent_snapshot if isinstance(sent_snapshot, dict) else None,
    }


def flush_auto_step_diagnostic(world, step=None, *, force=False, emit=True):
    pending = getattr(world, "_pending_auto_step_diag", None)
    if not isinstance(pending, dict):
        return None
    current_frame_id = int(getattr(world, "_frame_id", 0) or 0)
    pending_frame_id = int(pending.get("frame_id", 0) or 0)
    if not force and current_frame_id <= pending_frame_id:
        return None
    pending_step = pending.get("step") or normalize_step_label(step) or step
    action_text = pending.get("action_text") or "HOLD"
    pre_gate_text = pending.get("pre_gate_text") or "none"
    pre_focus = pending.get("pre_focus")
    sent_snapshot = pending.get("sent_snapshot")
    diag = _build_auto_step_diagnostic(
        world,
        pending_step,
        action_text,
        pre_gate_text,
        pre_focus=pre_focus,
    )
    _record_auto_step_action_stats(world, pending_step, diag.get("delta_class"))
    line = diag["plain"]
    # Emit only at completion boundaries (force=True), not every check tick.
    should_emit = bool(emit) and bool(force) and not bool(diag.get("success_hit", False))
    if should_emit:
        print(diag["colored"])
    _store_auto_step_diagnostic(
        world,
        pending_step,
        line,
        sent_snapshot=sent_snapshot,
        colored_line=diag.get("colored"),
        segments=diag.get("segments"),
    )
    clear_pending_auto_step_diagnostic(world)
    return line


def emit_auto_step_diagnostic(
    world,
    step,
    action_text,
    pre_gate_text,
    *,
    emit=True,
    pre_focus=None,
    success_override=None,
    sent_snapshot=None,
):
    diag = _build_auto_step_diagnostic(
        world,
        step,
        action_text,
        pre_gate_text,
        pre_focus=pre_focus,
        success_override=success_override,
    )
    _record_auto_step_action_stats(world, step, diag.get("delta_class"))
    line = diag["plain"]
    should_emit = bool(emit) and not bool(diag.get("success_hit", False))
    if should_emit:
        print(diag["colored"])
    _store_auto_step_diagnostic(
        world,
        step,
        line,
        sent_snapshot=sent_snapshot,
        colored_line=diag.get("colored"),
        segments=diag.get("segments"),
    )
    return line


def _gate_entry_matches(metric, value, stats, direction):
    if metric == "visible":
        if value is None:
            return False
        bool_val = bool(value)
        min_val = stats.get("min") if isinstance(stats, dict) else None
        max_val = stats.get("max") if isinstance(stats, dict) else None
        if isinstance(min_val, bool):
            return bool_val == min_val
        if isinstance(max_val, bool):
            return bool_val == max_val
        numeric = 1.0 if bool_val else 0.0
        try:
            min_num = float(min_val) if min_val is not None else None
        except (TypeError, ValueError):
            min_num = None
        try:
            max_num = float(max_val) if max_val is not None else None
        except (TypeError, ValueError):
            max_num = None
        if min_num is not None and numeric < min_num:
            return False
        if max_num is not None and numeric > max_num:
            return False
        return True

    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    target = stats.get("target") if isinstance(stats, dict) else None
    tol = stats.get("tol") if isinstance(stats, dict) else None
    if target is not None and tol is not None:
        try:
            target_num = float(target)
            tol_num = float(tol)
        except (TypeError, ValueError):
            target_num = None
            tol_num = None
        if target_num is not None and tol_num is not None:
            if direction == "high":
                return numeric >= (target_num - tol_num)
            if direction == "low":
                return numeric <= (target_num + tol_num)
            return abs(numeric - target_num) <= tol_num

    min_val = stats.get("min") if isinstance(stats, dict) else None
    max_val = stats.get("max") if isinstance(stats, dict) else None
    try:
        min_num = float(min_val) if min_val is not None else None
    except (TypeError, ValueError):
        min_num = None
    try:
        max_num = float(max_val) if max_val is not None else None
    except (TypeError, ValueError):
        max_num = None
    if min_num is not None and numeric < min_num:
        return False
    if max_num is not None and numeric > max_num:
        return False
    return True


def _stream_success_gate_line(step, entry):
    metric = entry.get("metric")
    stats = entry.get("stats") or {}
    value = entry.get("value")
    requirement = _gate_requirement_text(stats)
    if metric == "visible":
        effective = entry.get("effective_visible")
        raw = entry.get("raw_visible")
        current = _fmt_gate_value(bool(effective))
        if raw is not None and bool(raw) != bool(effective):
            current = f"{current} raw={_fmt_gate_value(bool(raw))}"
    else:
        current = _fmt_gate_value(value) if value is not None else "n/a"
    direction = metric_direction(metric, step)
    matched = _gate_entry_matches(metric, value, stats, direction)
    return {
        "segments": [
            (f"{metric}{requirement} ", STREAM_WHITE),
            (f"({current})", STREAM_GREEN if matched else STREAM_RED),
        ]
    }


def compute_stream_gate_summary(world, step, active=True):
    if not active:
        return [], None
    obj_name = normalize_step_label(step)
    if not obj_name:
        obj_name = str(step)
    nominal_only = step_is_nominal_only(obj_name, world.process_rules)
    analytics = telemetry_brick.compute_brick_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        obj_name,
        duration_s=CONTROL_DT,
    )
    align = (analytics or {}).get("align") if isinstance(analytics, dict) else None
    if not isinstance(align, dict):
        align = {}
    metric_lines = []
    for entry in _success_gate_entries(world, obj_name):
        metric_lines.append(_stream_success_gate_line(obj_name, entry))

    suggested_cmd = None
    suggested_score = None
    cmd = align.get("cmd")
    speed = align.get("speed") if cmd else None
    if (not nominal_only) and cmd and speed is not None:
        success_ok, confidence = evaluate_gate_status(world, obj_name)
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        speed_score = align.get("speed_score")
        speed, speed_score = telemetry_robot_module.quantize_speed(
            cmd,
            speed=speed,
            score=speed_score,
        )
        suggested_cmd = cmd
        suggested_score = speed_score

    if suggested_cmd is None:
        scan_dir = (world.process_rules or {}).get(obj_name, {}).get("scan_direction")
        if scan_dir in ("l", "r"):
            suggested_cmd = scan_dir
            suggested_score = DEFAULT_SPEED_SCORE

    if suggested_cmd is None:
        suggested = "HOLD"
        if not nominal_only:
            reason = _hold_reason(world, obj_name, analytics)
            if reason:
                suggested = f"HOLD ({reason})"
    else:
        suggested_score = _cap_auto_speed_score(suggested_score)
        suggested_cmd_sent = _remap_cmd_for_display(suggested_cmd)
        _, pwm, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
            suggested_cmd,
            suggested_score,
        )
        suggested = action_sent_display_text(
            suggested_cmd,
            suggested_score,
            cmd_sent=suggested_cmd_sent,
            pwm=pwm,
            duration_ms=duration_ms,
        )
    summary = [f"STEP: {obj_name}"]
    summary.extend(metric_lines)
    diag_line = getattr(world, "_last_auto_step_diag_line", None)
    diag_step = normalize_step_label(getattr(world, "_last_auto_step_diag_step", None))
    diag_time = getattr(world, "_last_auto_step_diag_time", None)
    diag_sent_snapshot = getattr(world, "_last_auto_step_diag_sent", None)
    diag_fresh = False
    if diag_line and diag_step == obj_name:
        try:
            age_s = float(time.time() - float(diag_time))
        except (TypeError, ValueError):
            age_s = None
        diag_fresh = age_s is None or age_s <= 8.0
    sent_snapshot = _capture_sent_action_snapshot(world)
    if (
        diag_line
        and diag_step == obj_name
        and diag_fresh
        and isinstance(diag_sent_snapshot, dict)
    ):
        current_event_time = _snapshot_event_time(sent_snapshot)
        diag_event_time = _snapshot_event_time(diag_sent_snapshot)
        if (
            current_event_time is None
            or diag_event_time is None
            or current_event_time <= (diag_event_time + 1e-6)
        ):
            sent_snapshot = dict(diag_sent_snapshot)
    sent_value = _format_sent_snapshot_value(sent_snapshot, obj_name)

    auto_line_text = None
    auto_line_segments = None
    if diag_line and diag_step == obj_name and diag_fresh:
        line_text = str(diag_line)
        if line_text.startswith("[AUTO] "):
            line_text = line_text[len("[AUTO] "):]
        auto_line_text = line_text
        diag_segments = getattr(world, "_last_auto_step_diag_segments", None)
        if isinstance(diag_segments, list) and diag_segments:
            cleaned = _strip_auto_prefix_segments(diag_segments)
            if cleaned:
                auto_line_segments = cleaned
    elif nominal_only:
        act_text = sent_value if sent_value != "(none)" else suggested
        auto_line_text = f"ACT ONLY (nominal): {act_text}"
        auto_line_segments = [
            ("ACT ONLY (nominal): ", STREAM_WHITE),
            (str(act_text), STREAM_ORANGE),
        ]
    else:
        fallback_action = suggested
        if suggested_cmd:
            fallback_action = action_display_text(suggested_cmd, suggested_score)
        fallback_diag_payload = _build_auto_step_diagnostic(
            world,
            obj_name,
            fallback_action,
            _auto_gate_snapshot_text(world, obj_name, include_requirements=False),
            pre_focus=_capture_auto_diag_focus(world, obj_name),
            success_override=_current_success_ok(world, obj_name),
        )
        fallback_diag = fallback_diag_payload["plain"]
        if fallback_diag.startswith("[AUTO] "):
            fallback_diag = fallback_diag[len("[AUTO] "):]
        auto_line_text = fallback_diag
        fallback_segments = fallback_diag_payload.get("segments")
        if isinstance(fallback_segments, list) and fallback_segments:
            cleaned = _strip_auto_prefix_segments(fallback_segments)
            if cleaned:
                auto_line_segments = cleaned
    if auto_line_segments:
        summary.append(
            {
                "segments": [("AUTO: ", STREAM_WHITE)] + list(auto_line_segments),
            }
        )
    else:
        summary.append(f"AUTO: {auto_line_text}")
    return summary, analytics


def _gate_progress_pct(world, step):
    obj_name = normalize_step_label(step)
    if not obj_name:
        obj_name = str(step)
    analytics = telemetry_brick.compute_brick_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        obj_name,
        duration_s=CONTROL_DT,
    )
    pct = None
    for name, val in analytics.get("gate_progress") or []:
        if normalize_step_label(name) == obj_name:
            pct = val
            break
    if pct is None:
        return None
    return int(max(0.0, min(100.0, pct)))


def update_stream_frame(world, vision):
    stream_state = getattr(world, "_stream_state", None)
    if stream_state is None or vision.current_frame is None:
        return
    if stream_state.get("skip_telemetry_process"):
        return
    gate_summary, analytics = compute_stream_gate_summary(
        world,
        world.step_state.value,
        active=True,
    )
    world._gate_summary = gate_summary
    world._gate_checker_summary = gate_utils.format_gatecheck_stream_lines(
        world,
        world.step_state.value,
    )
    world._highlight_metric = (analytics or {}).get("highlight_metric")
    frame = vision.current_frame.copy()
    draw_telemetry_overlay(
        frame,
        world,
        show_prompt=False,
        gate_status=None,
        gate_progress=None,
        step_suggestions=None,
        highlight_metric=getattr(world, "_highlight_metric", None),
        loop_id=getattr(world, "loop_id", None),
        gate_summary=getattr(world, "_gate_summary", []),
        gate_checker_summary=getattr(world, "_gate_checker_summary", []),
        show_center_line=bool(stream_state.get("show_center_line", True)),
    )
    with stream_state["lock"]:
        stream_state["frame"] = frame


def refresh_world_after_action(world, vision, log=True, attempts=8):
    start_frame_id = getattr(world, "_frame_id", 0)
    last_frame = None
    for _ in range(attempts):
        update_world_from_vision(world, vision, log=log)
        frame_id = getattr(world, "_frame_id", 0)
        if log and getattr(world, "log_fresh_frames", False):
            print(
                format_headline(
                    f"[FRAME] action refresh frame_id={frame_id} (start={start_frame_id})",
                    COLOR_WHITE,
                )
            )
        if world.brick:
            last_frame = {
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
                "frame_id": frame_id,
            }
        # Prefer a new observed frame over fixed sleeps.
        if frame_id > start_frame_id:
            if log and getattr(world, "log_fresh_frames", False):
                print(
                    format_headline(
                        f"[FRAME] fresh image observed (frame_id {frame_id} > {start_frame_id})",
                        COLOR_GREEN,
                    )
                )
            return last_frame
        time.sleep(CONTROL_DT * 0.5)
    return last_frame


def wait_for_frame_settle(world, vision, required_frames, log=False):
    frames = max(0, int(required_frames or 0))
    if frames <= 0:
        return
    gate_utils.wait_for_fresh_frames(
        world,
        lambda: update_world_from_vision(world, vision, log=log),
        required_new_frames=frames,
        max_cycles=max(frames * 12, 12),
        sleep_s=0.0,
    )


def post_act_settle_pause(world, vision):
    if POST_ACT_PAUSE_FRAMES <= 0:
        return
    # Push an immediate frame before waiting for additional fresh frames.
    update_stream_frame(world, vision)
    wait_for_frame_settle(world, vision, POST_ACT_PAUSE_FRAMES, log=False)


def post_act_analysis(world, vision, step=None, log=True, *, include_pause=True):
    last_frame = refresh_world_after_action(world, vision, log=log)
    if include_pause:
        post_act_settle_pause(world, vision)
    return last_frame


def _latest_unique_smoothed_frames(world, required_frames):
    required = max(1, int(required_frames or 1))
    history = getattr(world, "_smoothed_frame_history", None)
    if not history:
        return []
    selected = []
    seen_frame_ids = set()
    for entry in reversed(history):
        frame_id = int(entry.get("frame_id", 0) or 0)
        if frame_id <= 0 or frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)
        selected.append(entry)
        if len(selected) >= required:
            break
    selected.reverse()
    return selected


def _average_smoothed_frames(frames):
    if not frames:
        return None

    def mean(key):
        values = [float(frame.get(key, 0.0) or 0.0) for frame in frames]
        return sum(values) / len(values) if values else 0.0

    def majority(key):
        values = [bool(frame.get(key)) for frame in frames]
        return sum(1 for value in values if value) >= (len(values) / 2.0)

    x_axis = mean("x_axis")
    return {
        "visible": majority("visible"),
        "dist": mean("dist"),
        "angle": mean("angle"),
        "x_axis": x_axis,
        "offset_x": mean("offset_x"),
        "confidence": mean("confidence"),
    }


def _evaluate_lite_gate_brick_success(world, step, process_rules):
    required_frames = lite_gate_unique_frames(step)
    if required_frames is None:
        return None
    step_key = normalize_step_label(step)
    step_cfg = (process_rules or {}).get(step_key, {}) if isinstance(process_rules, dict) else {}
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    if not isinstance(success_gates, dict) or not success_gates:
        return None

    frames = _latest_unique_smoothed_frames(world, required_frames)
    world._gatecheck_mode = "lite"
    world._gatecheck_lite_required = int(required_frames)
    world._gatecheck_lite_collected = int(len(frames))
    if len(frames) < required_frames:
        return False

    measurement = _average_smoothed_frames(frames)
    if measurement is None:
        return False
    world._lite_gate_measurement = measurement
    world._lite_gate_frame_ids = [int(frame.get("frame_id", 0) or 0) for frame in frames]
    return gate_utils.gate_satisfied(measurement, success_gates)


def evaluate_gate_status(world, step):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    lite_brick_ok = _evaluate_lite_gate_brick_success(world, step, process_rules)
    if lite_brick_ok is None:
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 0
        world._gatecheck_lite_collected = 0
        visibility_grace_s = _visibility_grace_s(world, step)
        brick_success = telemetry_brick.evaluate_success_gates(
            world,
            step,
            telemetry_rules,
            process_rules,
            visibility_grace_s=visibility_grace_s,
        )
        brick_ok = bool(brick_success.ok)
    else:
        brick_ok = bool(lite_brick_ok)
    wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules)

    success_ok = brick_ok and wall_success.ok and robot_success.ok
    confidence = step_confidence(success_ok, world, step)
    return success_ok, confidence


def gatecheck_after_move(world, step, tracker, phase, log=True):
    if tracker is None:
        return False
    success_ok, confidence = evaluate_gate_status(world, step)
    gate_utils.record_success_gate_entry(world, step, success_ok)
    if log:
        log_confidence(world, confidence, step)
    return gate_utils.update_gatecheck(
        world,
        step,
        tracker,
        success_ok,
        phase=phase,
    )


def run_full_gatecheck_after_act(
    world,
    vision,
    step,
    tracker,
    phase,
    log=True,
    observer=None,
):
    if tracker is None:
        return False
    step_key = normalize_step_label(step)
    # Keep observe→act cadence fast for brick-alignment loops; full streak/majority
    # confirmation still happens across loop iterations via tracker state.
    if step_key in {"ALIGN_BRICK", "POSITION_BRICK"}:
        checks_per_act = 1
    else:
        checks_per_act = max(
            1,
            int(getattr(tracker, "consecutive_required", 1)),
            int(getattr(tracker, "majority_window", 1)),
        )
    lite_frames = lite_gate_unique_frames(step)
    if lite_frames is not None:
        checks_per_act = max(checks_per_act, int(lite_frames))
    truth_hit = False
    truth_by = None
    for _ in range(checks_per_act):
        gate_utils.wait_for_fresh_frames(
            world,
            lambda: update_world_from_vision(world, vision, log=log),
            required_new_frames=1,
            max_cycles=24,
            sleep_s=0.0,
        )
        if observer:
            observer("frame", world, vision, None, None, None)
        success_met = gatecheck_after_move(world, step, tracker, phase=phase, log=log)
        status = getattr(world, "_gatecheck_status", None)
        if success_met or (isinstance(status, dict) and bool(status.get("truth_ok", False))):
            truth_hit = True
            if isinstance(status, dict) and status.get("truth_by"):
                truth_by = status.get("truth_by")
    status = getattr(world, "_gatecheck_status", None)
    if truth_hit and isinstance(status, dict):
        status["truth_ok"] = True
        if truth_by:
            status["truth_by"] = truth_by
    return truth_hit


def wait_for_start_gates(
    world,
    vision,
    step,
    robot=None,
    cmd=None,
    speed=None,
    log=True,
    observer=None,
    allow_success=True,
):
    step_key = normalize_step_label(step)
    # Reset turn-history state at step start so the first autonomous L/R act
    # in this step uses the "first turn" duration scaling, without sending any
    # stop/motion command before observations.
    if robot is not None:
        try:
            robot._last_turn_cmd = None
        except Exception:
            pass
    if not step_requires_start_gates(step_key, getattr(world, "process_rules", None)):
        # Start gates are skipped for find/exit steps, but still run a short
        # success precheck so auto never moves before evaluating current gates.
        if world is not None and vision is not None:
            success_tracker = (
                new_success_tracker(step, world.process_rules)
                if allow_success
                else None
            )
            precheck_frames = 1
            if success_tracker is not None:
                precheck_frames = max(
                    int(getattr(success_tracker, "majority_window", 1)),
                    int(getattr(success_tracker, "required_consecutive", 1)),
                    1,
                )
            for _ in range(precheck_frames):
                update_world_from_vision(world, vision, log=log)
                if observer:
                    observer("frame", world, vision, None, None, None)
                if not allow_success or success_tracker is None:
                    continue
                success_ok, confidence = evaluate_gate_status(world, step)
                gate_utils.record_success_gate_entry(world, step, success_ok)
                if log:
                    log_confidence(world, confidence, step)
                success_met = gate_utils.update_gatecheck(
                    world,
                    step,
                    success_tracker,
                    success_ok,
                    phase="start",
                )
                if success_met:
                    gate_utils.store_gate_summary(world, success_tracker)
                    if robot:
                        robot.stop()
                    if log:
                        print(format_headline(f"[START] {step_key} success gates already met", COLOR_GREEN))
                    return "success"
        if log:
            print(format_headline(f"[START] {step_key} start gates skipped", COLOR_WHITE))
        return "start"
    start_time = time.time()
    stable = 0
    success_tracker = new_success_tracker(step, world.process_rules)
    success_seen = False
    last_cmd = None
    last_speed = None
    last_reason = None
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision, log=log)
        if observer:
            observer("frame", world, vision, None, None, None)
        brick_check = telemetry_brick.evaluate_start_gates(world, step, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, step, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, step, {}, world.process_rules)
        if not (brick_check.ok and wall_check.ok and robot_check.ok):
            reasons = []
            reasons.extend(brick_check.reasons or [])
            reasons.extend(wall_check.reasons or [])
            reasons.extend(robot_check.reasons or [])
            hold_reason = "start_gate: " + ", ".join(reasons) if reasons else "start_gate: awaiting confidence"
            if log and (hold_reason != last_reason or last_cmd is not None or last_speed not in (None, 0.0)):
                print(format_headline(format_control_action_line(None, 0.0, hold_reason), COLOR_WHITE))
            if robot:
                robot.stop()
            if observer:
                observer("action", world, vision, None, 0.0, "start_gate")
            last_cmd = None
            last_reason = hold_reason
            last_speed = 0.0
            time.sleep(CONTROL_DT)
            continue
        if allow_success:
            success_ok, confidence = evaluate_gate_status(world, step)
            gate_utils.record_success_gate_entry(world, step, success_ok)
            if log:
                log_confidence(world, confidence, step)
            success_met = gate_utils.update_gatecheck(
                world,
                step,
                success_tracker,
                success_ok,
                phase="start",
            )
            if success_ok and not success_seen and robot:
                robot.stop()
                success_seen = True
            if success_met:
                gate_utils.store_gate_summary(world, success_tracker)
                if robot:
                    robot.stop()
                return "success"
            if not success_ok:
                success_seen = False
            if gate_utils.should_hold_for_success_confirmation(
                next_module.success_gates_visible_only(world.process_rules or {}, step),
                success_tracker,
                success_met,
            ):
                if robot:
                    robot.stop()
                if observer:
                    observer("action", world, vision, None, 0.0, "gate confirm hold")
                time.sleep(CONTROL_DT)
                continue

        brick_check = telemetry_brick.evaluate_start_gates(world, step, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, step, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, step, {}, world.process_rules)
        if brick_check.ok and wall_check.ok and robot_check.ok:
            stable += 1
            if stable >= GATE_STABILITY_FRAMES:
                if log:
                    print(format_headline(f"[START] {step} start gates met", COLOR_GREEN))
                return "start"
        else:
            stable = 0
            if robot and cmd and speed:
                if log and (cmd != last_cmd or speed != last_speed):
                    print(format_headline(format_control_action_line(cmd, speed, "start_gate"), COLOR_WHITE))
                    last_cmd = cmd
                    last_speed = speed
                send_robot_command(robot, world, step, cmd, speed, auto_mode=True)
                evt = MotionEvent(
                    cmd_to_motion_type(cmd),
                    int(speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
                acted_success = run_full_gatecheck_after_act(
                    world,
                    vision,
                    step,
                    success_tracker,
                    phase="start",
                    log=log,
                    observer=observer,
                )
                if allow_success and acted_success:
                    gate_utils.store_gate_summary(world, success_tracker)
                    if robot:
                        robot.stop()
                    return "success"
            elif log and (cmd != last_cmd or speed != last_speed):
                hold_reason = "start_gate: missing cmd/speed"
                print(format_headline(format_control_action_line(None, 0.0, hold_reason), COLOR_WHITE))
                last_cmd = cmd
                last_speed = speed
        time.sleep(CONTROL_DT)
    return "timeout"


def run_alignment_segment(
    segment,
    step,
    robot,
    vision,
    world,
    steps,
    raw_steps,
    observer=None,
    analysis_pause_s=0.0,
    confirm_callback=None,
    align_silent=False,
):
    def _classify_align_delta_class(
        *,
        correction_type,
        local_gate_before_action,
        prev_log_correction_type,
        prev_log_x_err,
        prev_log_dist,
    ):
        ctype = str(correction_type or "").strip().lower()
        if ctype == "distance":
            dist_target = local_gate_before_action.get("dist_target")
            dist_now = local_gate_before_action.get("dist")
            try:
                curr = abs(float(dist_now) - float(dist_target))
            except (TypeError, ValueError):
                return "unknown"
            try:
                if str(prev_log_correction_type or "").strip().lower() == "distance" and prev_log_dist is not None:
                    prev = abs(float(prev_log_dist) - float(dist_target))
                else:
                    prev = None
            except (TypeError, ValueError):
                prev = None
        else:
            x_now = local_gate_before_action.get("x_err")
            try:
                curr = abs(float(x_now))
            except (TypeError, ValueError):
                return "unknown"
            try:
                if str(prev_log_correction_type or "").strip().lower() == "x_axis" and prev_log_x_err is not None:
                    prev = abs(float(prev_log_x_err))
                else:
                    prev = None
            except (TypeError, ValueError):
                prev = None

        if prev is None:
            return "unknown"
        delta = float(prev) - float(curr)
        if abs(delta) < 0.05:
            return "unchanged"
        if delta > 0.0:
            return "closer"
        return "backward"

    def _emit_align_shorthand_action_line(
        *,
        local_gate_before_action,
        prev_log_correction_type,
        prev_log_x_err,
        prev_log_dist,
        cmd,
        score_effective,
        action_meta,
        action_index,
        settle=False,
        force_gap_switch=False,
    ):
        cmd_key = str(cmd).strip().lower()
        if cmd_key in ("f", "b"):
            correction_type = "distance"
        elif cmd_key in ("l", "r"):
            correction_type = "x_axis"
        else:
            correction_type = str(prev_log_correction_type or "").strip().lower()
            if correction_type not in {"distance", "x_axis"}:
                correction_type = "x_axis"

        x_err_now = local_gate_before_action.get("x_err")
        x_target = local_gate_before_action.get("x_target")
        x_tol = local_gate_before_action.get("x_tol")
        dist_now = local_gate_before_action.get("dist")
        dist_target = local_gate_before_action.get("dist_target")
        dist_tol = local_gate_before_action.get("dist_tol")

        metric_color = COLOR_YELLOW
        metric_text = ""
        metric_name = "x_err"
        switch_message = None
        prev_type = str(prev_log_correction_type or "").strip().lower()
        switched_gap_type = bool(force_gap_switch) or (
            prev_type in {"distance", "x_axis"} and prev_type != correction_type
        )

        if correction_type == "distance":
            metric_name = "dist_err"
            try:
                curr_err = float(dist_now) - float(dist_target)
                curr_metric = abs(float(curr_err))
                curr_abs_val = float(dist_target) + float(curr_err)
            except (TypeError, ValueError):
                curr_err = None
                curr_metric = None
                curr_abs_val = None
            prev_metric = None
            prev_err = None
            prev_err_any = None
            try:
                if prev_log_dist is not None:
                    prev_err_any = float(prev_log_dist) - float(dist_target)
                if prev_log_correction_type == "distance" and prev_log_dist is not None:
                    prev_err = float(prev_log_dist) - float(dist_target)
                    prev_metric = abs(float(prev_err))
            except (TypeError, ValueError):
                prev_metric = None
                prev_err = None
                prev_err_any = None

            if curr_metric is None:
                metric_text = "unknown"
                metric_color = COLOR_WHITE
            else:
                if prev_metric is None or prev_err is None:
                    metric_color = COLOR_YELLOW
                    delta_text = "(na)"
                else:
                    delta = float(prev_metric) - float(curr_metric)
                    if abs(delta) < 0.05:
                        metric_color = COLOR_YELLOW
                        delta_text = f"(~unchanged; previous {metric_name}={float(prev_err):+.2f}mm)"
                    elif delta > 0:
                        metric_color = COLOR_GREEN
                        delta_text = f"({delta:+.2f}mm better than previous {metric_name}={float(prev_err):+.2f}mm)"
                    else:
                        metric_color = COLOR_RED
                        delta_text = f"({delta:+.2f}mm worse than previous {metric_name}={float(prev_err):+.2f}mm)"
                abs_text = f"{metric_color}{float(curr_abs_val):.2f}mm{COLOR_RESET}"
                metric_text = (
                    f"target ({float(dist_target):.2f}mm ±{float(dist_tol):.2f}mm) "
                    f"{float(curr_err):+.2f}mm={abs_text} {delta_text}"
                )
                if switched_gap_type:
                    if prev_err_any is None:
                        switch_delta_text = "(na)"
                    else:
                        delta_any = abs(float(prev_err_any)) - abs(float(curr_err))
                        if abs(delta_any) < 0.05:
                            switch_delta_text = (
                                f"(~unchanged; previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                        elif delta_any > 0:
                            switch_delta_text = (
                                f"({delta_any:+.2f}mm better than previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                        else:
                            switch_delta_text = (
                                f"({delta_any:+.2f}mm worse than previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                    switch_message = (
                        f"Switched gap type to {metric_name} "
                        f"(the last act resulted in {metric_name}=target ({float(dist_target):.2f}mm ±{float(dist_tol):.2f}mm) "
                        f"{float(curr_err):+.2f}mm={float(curr_abs_val):.2f}mm {switch_delta_text})"
                    )
        else:
            try:
                curr_err = float(x_err_now)
                curr_metric = abs(float(curr_err))
                curr_abs_val = float(x_target) + float(curr_err)
            except (TypeError, ValueError):
                curr_err = None
                curr_metric = None
                curr_abs_val = None
            prev_metric = None
            prev_err = None
            prev_err_any = None
            try:
                if prev_log_x_err is not None:
                    prev_err_any = float(prev_log_x_err)
                if prev_log_correction_type == "x_axis" and prev_log_x_err is not None:
                    prev_err = float(prev_log_x_err)
                    prev_metric = abs(float(prev_err))
            except (TypeError, ValueError):
                prev_metric = None
                prev_err = None
                prev_err_any = None

            if curr_metric is None:
                metric_text = "unknown"
                metric_color = COLOR_WHITE
            else:
                overshot = False
                if prev_err is not None:
                    overshot = (
                        (float(prev_err) * float(curr_err)) < 0.0
                        and abs(float(prev_err)) > 0.30
                        and abs(float(curr_err)) > 0.30
                    )
                if prev_metric is None or prev_err is None:
                    metric_color = COLOR_YELLOW
                    delta_text = "(na)"
                elif overshot:
                    metric_color = COLOR_RED
                    delta_text = f"(overshot target; previous {metric_name}={float(prev_err):+.2f}mm)"
                else:
                    delta = abs(float(prev_err)) - abs(float(curr_err))
                    if abs(delta) < 0.05:
                        metric_color = COLOR_YELLOW
                        delta_text = f"(~unchanged; previous {metric_name}={float(prev_err):+.2f}mm)"
                    elif delta > 0:
                        metric_color = COLOR_GREEN
                        delta_text = f"({delta:+.2f}mm better than previous {metric_name}={float(prev_err):+.2f}mm)"
                    else:
                        metric_color = COLOR_RED
                        delta_text = f"({delta:+.2f}mm worse than previous {metric_name}={float(prev_err):+.2f}mm)"
                abs_text = f"{metric_color}{float(curr_abs_val):+.2f}mm{COLOR_RESET}"
                metric_text = (
                    f"target ({float(x_target):+.2f}mm ±{float(x_tol):.2f}mm) "
                    f"{float(curr_err):+.2f}mm={abs_text} {delta_text}"
                )
                if switched_gap_type:
                    if prev_err_any is None:
                        switch_delta_text = "(na)"
                    else:
                        delta_any = abs(float(prev_err_any)) - abs(float(curr_err))
                        if abs(delta_any) < 0.05:
                            switch_delta_text = (
                                f"(~unchanged; previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                        elif delta_any > 0:
                            switch_delta_text = (
                                f"({delta_any:+.2f}mm better than previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                        else:
                            switch_delta_text = (
                                f"({delta_any:+.2f}mm worse than previous {metric_name}={float(prev_err_any):+.2f}mm)"
                            )
                    switch_message = (
                        f"Switched gap type to {metric_name} "
                        f"(the last act resulted in {metric_name}=target ({float(x_target):+.2f}mm ±{float(x_tol):.2f}mm) "
                        f"{float(curr_err):+.2f}mm={float(curr_abs_val):+.2f}mm {switch_delta_text})"
                    )

        try:
            score_display = int(round(float(score_effective))) if score_effective is not None else 0
        except (TypeError, ValueError):
            score_display = 0
        motor_power = action_meta.get("power") if isinstance(action_meta, dict) else None
        motor_pwm = action_meta.get("pwm") if isinstance(action_meta, dict) else None
        motor_duration_ms = None
        if isinstance(action_meta, dict):
            motor_duration_ms = action_meta.get("duration_model_ms")
            if motor_duration_ms is None:
                motor_duration_ms = action_meta.get("duration_ms")
        motor_details = ""
        parts = []
        if motor_pwm is not None:
            parts.append(f"pwm={int(motor_pwm)}")
        if motor_power is not None:
            parts.append(f"pwr={float(motor_power):.3f}")
        if motor_duration_ms is not None:
            parts.append(f"t={int(motor_duration_ms)}ms")
        if parts:
            motor_details = f" {COLOR_GRAY}({', '.join(parts)}){COLOR_RESET}"

        trial_num = int(getattr(world, "loop_id", 0) or 0)
        phase_label = "ALIGN_SETTLE" if bool(settle) else "ALIGN"
        if switch_message:
            print(
                f"{COLOR_WHITE}[T{trial_num}.{int(action_index)} {phase_label}] "
                f"{switch_message}{COLOR_RESET}"
            )
        print(
            f"{COLOR_CYAN}[T{trial_num}.{int(action_index)} {phase_label}]{COLOR_RESET} "
            f"{metric_name}={metric_text} → "
            f"{COLOR_ORANGE_BRIGHT}{str(cmd).lower()} {int(score_display)}%{COLOR_RESET}{motor_details}"
        )

    def _dominant_turn_from_steps(motion_steps):
        if not motion_steps:
            return None
        left_s = 0.0
        right_s = 0.0
        for step_entry in motion_steps:
            cmd_val = getattr(step_entry, "cmd", None)
            try:
                dur_s = float(getattr(step_entry, "duration_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                dur_s = 0.0
            if cmd_val == "l":
                left_s += max(0.0, dur_s)
            elif cmd_val == "r":
                right_s += max(0.0, dur_s)
        if left_s <= 0.0 and right_s <= 0.0:
            return None
        if left_s > right_s:
            return "l"
        return "r"

    def _observe_visible(timeout_s):
        deadline = float(time.time()) + max(0.05, float(timeout_s))
        while time.time() < deadline:
            update_world_from_vision(world, vision, log=False)
            if bool((getattr(world, "brick", None) or {}).get("visible")):
                return True
            time.sleep(float(CONTROL_DT))
        return False

    def _inverse_cmd(cmd):
        return {
            "l": "r",
            "r": "l",
            "f": "b",
            "b": "f",
            "u": "d",
            "d": "u",
        }.get(str(cmd))

    action_speeds = derive_action_speeds(raw_steps)
    gate_bounds = telemetry_brick.success_gate_bounds(
        world.process_rules or {},
        world.learned_rules or {},
        step,
    )
    demo_cmds = [getattr(step_entry, "cmd", None) for step_entry in (raw_steps or [])]
    demo_cmds = [cmd for cmd in demo_cmds if cmd in ("f", "b", "l", "r")]
    demo_turn_cmds = {cmd for cmd in demo_cmds if cmd in ("l", "r")}
    demo_has_drive = any(cmd in ("f", "b") for cmd in demo_cmds)
    demo_turn_only = bool(demo_turn_cmds) and not demo_has_drive
    demo_single_turn_cmd = next(iter(demo_turn_cmds)) if len(demo_turn_cmds) == 1 else None
    dominant_demo_turn = _dominant_turn_from_steps(raw_steps)
    suppress_auto_diag = normalize_step_label(step) in {"ALIGN_BRICK", "POSITION_BRICK"}

    scan_cmd = telemetry_robot_module.resolve_scan_direction(
        world.process_rules,
        step,
        fallback=demo_single_turn_cmd or dominant_demo_turn,
    )
    if scan_cmd not in ("l", "r") and demo_single_turn_cmd in ("l", "r"):
        scan_cmd = demo_single_turn_cmd

    def _apply_turn_only_demo_policy(cmd, speed, speed_score, cmd_reason):
        if normalize_step_label(step) == "ALIGN_BRICK":
            return cmd, speed, speed_score, cmd_reason
        if not demo_turn_only:
            return cmd, speed, speed_score, cmd_reason
        if scan_cmd not in ("l", "r"):
            return cmd, speed, speed_score, cmd_reason

        cmd_out = cmd
        speed_out = speed
        score_out = speed_score
        reason_out = cmd_reason
        forced = False

        if cmd_out in ("f", "b"):
            cmd_out = scan_cmd
            forced = True
        elif cmd_out is None:
            cmd_out = scan_cmd
            forced = True
        elif demo_single_turn_cmd in ("l", "r") and cmd_out in ("l", "r") and cmd_out != demo_single_turn_cmd:
            cmd_out = demo_single_turn_cmd
            forced = True

        if forced:
            speed_out = action_speeds["scan"]
            _, q_score = telemetry_robot_module.quantize_speed(cmd_out, speed=speed_out)
            score_out = q_score
            reason_out = "turn-only demo policy"
        return cmd_out, speed_out, score_out, reason_out

    start_cmd = scan_cmd
    start_speed = action_speeds["scan"]
    step_key = normalize_step_label(step)
    skip_settle_pause = step_key == "ALIGN_BRICK"
    # ALIGN_BRICK demos are often state-only, so derived scan speed can fall back to 50%.
    # Use explicit score-1 turn speed so start-gate scan matches "1%" tuning from world_model_robot.
    if step_key == "ALIGN_BRICK" and start_cmd in ("l", "r"):
        start_speed = telemetry_robot_module.manual_speed_for_cmd(
            start_cmd,
            telemetry_robot_module.SPEED_SCORE_MIN,
        )
    # For ALIGN_BRICK, avoid any robot movement during start-gate checks. We only
    # allow motion after the per-loop gate check determines movement is needed.
    start_cmd_for_wait = start_cmd
    start_speed_for_wait = start_speed
    if step_key == "ALIGN_BRICK":
        start_cmd_for_wait = None
        start_speed_for_wait = None
    start_status = wait_for_start_gates(
        world,
        vision,
        step,
        robot=robot,
        cmd=start_cmd_for_wait,
        speed=start_speed_for_wait,
        log=not align_silent,
        observer=observer,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        if not align_silent:
            print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
            print_gate_summary_line(world)
            print_success_events(world, step)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    if next_module.success_gates_visible_only(world.process_rules or {}, step):
        world._visible_speed_cycle = 0

    success_tracker = new_success_tracker(step, world.process_rules)
    clear_pending_auto_step_diagnostic(world)
    last_cmd = None
    last_reason = None
    last_speed = None
    last_action_frame = None
    prev_log_correction_type = None
    prev_log_x_err = None
    prev_log_dist = None
    align_action_idx = 0
    stale_pre_obs_streak = 0
    stale_frame_streak = 0
    not_visible_streak = 0
    last_visible_ts = time.time()
    force_gap_switch_after_recovery = False
    last_align_action = None
    last_align_signature = None
    recent_align_acts = collections.deque(maxlen=32)
    loop_id = 0

    def _format_last_align_action_for_recovery():
        if not isinstance(last_align_action, dict):
            return "none"
        cmd_val = str(last_align_action.get("cmd") or "?").upper()
        score_val = last_align_action.get("score")
        pwm_val = last_align_action.get("pwm")
        pwr_val = last_align_action.get("power")
        t_val = last_align_action.get("duration_ms")
        parts = [f"cmd={cmd_val}"]
        if score_val is not None:
            try:
                parts.append(f"score={int(round(float(score_val)))}%")
            except (TypeError, ValueError):
                pass
        if pwm_val is not None:
            try:
                parts.append(f"pwm={int(round(float(pwm_val)))}")
            except (TypeError, ValueError):
                pass
        if pwr_val is not None:
            try:
                parts.append(f"pwr={float(pwr_val):.3f}")
            except (TypeError, ValueError):
                pass
        if t_val is not None:
            try:
                parts.append(f"t={int(round(float(t_val)))}ms")
            except (TypeError, ValueError):
                pass
        return ", ".join(parts)

    def _recover_visibility(source="unknown", reason=None):
        source_key = str(source or "").strip().lower()
        if not align_silent:
            reason_text = str(reason or "confident visibility loss").strip()
            print(format_headline(f"[RECOVERY] trigger: {reason_text}", COLOR_ORANGE_BRIGHT))
            print(
                format_headline(
                    f"[RECOVERY] last act: {_format_last_align_action_for_recovery()}",
                    COLOR_WHITE,
                )
            )
            print(format_headline(f"[RECOVERY] {source_key}: reversing recent acts...", COLOR_WHITE))

        plan = []
        for row in reversed(list(recent_align_acts)[-int(ALIGN_RECOVERY_REVERSE_MAX_ACTS):]):
            inv = _inverse_cmd(row.get("cmd"))
            if inv is None:
                continue
            score = row.get("score")
            try:
                score = int(round(float(score))) if score is not None else int(ALIGN_RECOVERY_SCAN_SCORE)
            except (TypeError, ValueError):
                score = int(ALIGN_RECOVERY_SCAN_SCORE)
            score = max(int(telemetry_robot_module.SPEED_SCORE_MIN), min(int(score), int(MAX_SPEED_SCORE)))
            plan.append({"cmd": str(inv), "score": int(score)})

        for idx, row in enumerate(plan, start=1):
            if not align_silent:
                print(
                    format_headline(
                        f"[RECOVERY] reverse {idx}/{len(plan)}: {row['cmd'].upper()} {int(row['score'])}%",
                        COLOR_WHITE,
                    )
                )
            send_robot_command(
                robot,
                world,
                step,
                row["cmd"],
                0.0,
                speed_score=row["score"],
                auto_mode=True,
            )
            if _observe_visible(float(ALIGN_RECOVERY_REVERSE_OBSERVE_S)):
                if not align_silent:
                    print(format_headline("[RECOVERY] Visibility restored by reverse acts.", COLOR_GREEN))
                return True

        if not align_silent:
            print(format_headline("[RECOVERY] Reverse phase did not restore visibility; switching to scan phase.", COLOR_ORANGE_BRIGHT))
            print(format_headline("[RECOVERY] Scanning for brick...", COLOR_WHITE))
        for i in range(1, int(ALIGN_RECOVERY_SCAN_MAX_ATTEMPTS) + 1):
            block = int((i - 1) // int(max(1, ALIGN_RECOVERY_SCAN_BURST_ACTS)))
            scan_cmd_local = "l" if (block % 2 == 0) else "r"
            scan_score_local = int(ALIGN_RECOVERY_SCAN_SCORE)
            if not align_silent:
                print(
                    format_headline(
                        f"[RECOVERY] scan {i}/{int(ALIGN_RECOVERY_SCAN_MAX_ATTEMPTS)}: {scan_cmd_local.upper()} {scan_score_local}%",
                        COLOR_WHITE,
                    )
                )
            send_robot_command(
                robot,
                world,
                step,
                scan_cmd_local,
                0.0,
                speed_score=scan_score_local,
                auto_mode=True,
            )
            if _observe_visible(float(ALIGN_RECOVERY_SCAN_OBSERVE_S)):
                if not align_silent:
                    print(format_headline("[RECOVERY] Visibility restored by scan.", COLOR_GREEN))
                return True

        if not align_silent:
            print(format_headline("[RECOVERY] Failed to restore visibility.", COLOR_RED))
        return False

    def _hold_and_recheck_visibility(rounds=ALIGN_RECOVERY_PREMOVE_RECHECK_ROUNDS):
        try:
            rounds_local = max(1, int(rounds))
        except (TypeError, ValueError):
            rounds_local = int(ALIGN_RECOVERY_PREMOVE_RECHECK_ROUNDS)
        hold_window_s = max(float(CONTROL_DT), float(VISIBILITY_LOST_HOLD_S))
        if robot:
            robot.stop()
        record_hold_display(world, step, f"visibility recheck x{int(rounds_local)}")
        for idx in range(1, int(rounds_local) + 1):
            if robot:
                robot.stop()
            if not align_silent:
                print(
                    format_headline(
                        f"[RECOVERY] pre-move check {idx}/{int(rounds_local)}: hold still and re-check visibility ({float(hold_window_s):.2f}s)",
                        COLOR_WHITE,
                    )
                )
            if _observe_visible(float(hold_window_s)):
                if not align_silent:
                    print(
                        format_headline(
                            f"[RECOVERY] pre-move recheck result: restored on round {idx}/{int(rounds_local)}.",
                            COLOR_GREEN,
                        )
                    )
                return {
                    "visible": True,
                    "round_hit": int(idx),
                    "rounds_total": int(rounds_local),
                }
        if not align_silent:
            print(
                format_headline(
                    f"[RECOVERY] pre-move recheck result: visibility still lost after {int(rounds_local)}/{int(rounds_local)} rounds.",
                    COLOR_ORANGE_BRIGHT,
                )
            )
        return {
            "visible": False,
            "round_hit": None,
            "rounds_total": int(rounds_local),
        }

    while True:
        loop_id += 1
        world.loop_id = loop_id
        step_norm = normalize_step_label(step)
        update_world_from_vision(world, vision, log=not align_silent)
        visible_now = bool((getattr(world, "brick", None) or {}).get("visible"))
        if visible_now:
            not_visible_streak = 0
            last_visible_ts = time.time()
        else:
            not_visible_streak = int(not_visible_streak) + 1
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            world._last_obs_note = None
        # pause1 removed per request
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        local_gate = next_module.align_local_gate_status(
            world,
            step,
            process_rules=world.process_rules or {},
        )
        success_truth_ok = bool(success_ok and local_gate.get("ok", True))
        gate_utils.record_success_gate_entry(world, step, success_truth_ok)
        if not align_silent:
            log_confidence(world, confidence, step)
        success_met = gate_utils.update_gatecheck(
            world,
            step,
            success_tracker,
            success_truth_ok,
            phase="align",
        )
        if success_truth_ok:
            if robot:
                robot.stop()
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print_gate_summary_line(world, success_tracker)
                print_success_events(world, step)
            return True, "success gate"
        if success_met:
            if robot:
                robot.stop()
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print_gate_summary_line(world, success_tracker)
                print_success_events(world, step)
            return True, "success gate"

        if step_norm == "ALIGN_BRICK" and not visible_now:
            lost_for_s = max(0.0, float(time.time()) - float(last_visible_ts))
            confident_loss = bool(
                int(not_visible_streak) >= int(ALIGN_RECOVERY_LOST_VISIBLE_FRAMES)
                and float(lost_for_s) >= float(ALIGN_RECOVERY_LOST_VISIBLE_MIN_S)
            )
            if not confident_loss:
                time.sleep(CONTROL_DT)
                continue
            reason = (
                "brick invisible "
                f"for {int(not_visible_streak)} consecutive frames and {float(lost_for_s):.2f}s "
                f"(threshold {int(ALIGN_RECOVERY_LOST_VISIBLE_FRAMES)} frames / {float(ALIGN_RECOVERY_LOST_VISIBLE_MIN_S):.2f}s)"
            )
            premove_recheck = _hold_and_recheck_visibility()
            if bool((premove_recheck or {}).get("visible")):
                stale_pre_obs_streak = 0
                stale_frame_streak = 0
                not_visible_streak = 0
                last_visible_ts = time.time()
                prev_log_correction_type = None
                prev_log_x_err = None
                prev_log_dist = None
                force_gap_switch_after_recovery = True
                last_align_signature = None
                continue
            if _recover_visibility(source="align_observe", reason=reason):
                stale_pre_obs_streak = 0
                stale_frame_streak = 0
                not_visible_streak = 0
                last_visible_ts = time.time()
                prev_log_correction_type = None
                prev_log_x_err = None
                prev_log_dist = None
                force_gap_switch_after_recovery = True
                last_align_signature = None
                continue
            return False, "lost_vision"

        if step_norm == "ALIGN_BRICK":
            brick = getattr(world, "brick", {}) or {}
            act_plan = next_module.select_align_brick_next_act(
                process_rules=world.process_rules or {},
                x_axis_mm=brick.get("x_axis", brick.get("offset_x")),
                dist_mm=brick.get("dist"),
                visible=bool(brick.get("visible")),
                angle_deg=brick.get("angle", 0.0),
                duration_s=CONTROL_DT,
            )
            cmd = act_plan.get("cmd")
            cmd_reason = act_plan.get("reason") or "align"
            speed_score = act_plan.get("score")
            speed = telemetry_robot_module.manual_speed_for_cmd(cmd, speed_score) if cmd and speed_score is not None else 0.0
        else:
            analytics = next_module.compute_alignment_decision(
                world=world,
                step=step,
                process_rules=world.process_rules or {},
                learned_rules=world.learned_rules or {},
                duration_s=CONTROL_DT,
            )
            cmd = analytics.get("cmd")
            speed = analytics.get("speed") or 0.0
            cmd_reason = analytics.get("worst_metric") or "align"
            speed_score = analytics.get("speed_score")
        cmd, speed, speed_score, cmd_reason = _apply_turn_only_demo_policy(
            cmd,
            speed,
            speed_score,
            cmd_reason,
        )
        if step_norm != "ALIGN_BRICK":
            speed = apply_pursuit_speed(speed)
            speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            if observer:
                observer("analysis", world, vision, cmd, speed, cmd_reason)
            if analysis_pause_s:
                time.sleep(analysis_pause_s)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            local_gate_before_action = next_module.align_local_gate_status(
                world,
                step,
                process_rules=world.process_rules or {},
            )
            pre_frame_id = getattr(world, "_frame_id", 0)
            pre_gate_text = None
            pre_focus = None
            if not suppress_auto_diag:
                pre_gate_text = _auto_gate_snapshot_text(world, step, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            action_meta = send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=speed_score,
                auto_mode=True,
            )
            cmd_sent = cmd
            score_effective = speed_score
            if isinstance(action_meta, dict):
                cmd_sent = action_meta.get("cmd_sent") or cmd_sent
                score_effective = action_meta.get("score_effective")
                if score_effective is None:
                    score_effective = action_meta.get("score_model", speed_score)

            if step_norm == "ALIGN_BRICK":
                try:
                    x_sig = round(float(local_gate_before_action.get("x_err")), 2)
                except (TypeError, ValueError):
                    x_sig = None
                try:
                    d_sig = round(float(local_gate_before_action.get("dist")), 1)
                except (TypeError, ValueError):
                    d_sig = None
                try:
                    s_sig = int(round(float(score_effective))) if score_effective is not None else None
                except (TypeError, ValueError):
                    s_sig = None
                sig = (x_sig, d_sig, str(cmd), s_sig)
                if sig == last_align_signature:
                    stale_pre_obs_streak = int(stale_pre_obs_streak) + 1
                else:
                    stale_pre_obs_streak = 0
                last_align_signature = sig

                if stale_pre_obs_streak >= 8:
                    if robot:
                        robot.stop()
                    print(
                        format_headline(
                            "[WARN] ALIGN_BRICK stale observation for 8 consecutive acts; treating as lost visibility.",
                            COLOR_RED,
                        )
                    )
                    reason = (
                        "stale observation signature repeated for 8 acts "
                        f"(signature={str(sig)})"
                    )
                    if _recover_visibility(source="stale_observation", reason=reason):
                        stale_pre_obs_streak = 0
                        stale_frame_streak = 0
                        not_visible_streak = 0
                        last_visible_ts = time.time()
                        prev_log_correction_type = None
                        prev_log_x_err = None
                        prev_log_dist = None
                        force_gap_switch_after_recovery = True
                        last_align_signature = None
                        continue
                    return False, "lost_vision_stale_observation"
            if not align_silent:
                align_action_idx = int(align_action_idx) + 1
                _emit_align_shorthand_action_line(
                    local_gate_before_action=local_gate_before_action,
                    prev_log_correction_type=prev_log_correction_type,
                    prev_log_x_err=prev_log_x_err,
                    prev_log_dist=prev_log_dist,
                    cmd=cmd,
                    score_effective=score_effective,
                    action_meta=action_meta,
                    action_index=align_action_idx,
                    settle=False,
                    force_gap_switch=force_gap_switch_after_recovery,
                )
                force_gap_switch_after_recovery = False
            if suppress_auto_diag:
                correction_type_for_stats = "distance" if cmd in ("f", "b") else "x_axis" if cmd in ("l", "r") else None
                delta_class = _classify_align_delta_class(
                    correction_type=correction_type_for_stats,
                    local_gate_before_action=local_gate_before_action,
                    prev_log_correction_type=prev_log_correction_type,
                    prev_log_x_err=prev_log_x_err,
                    prev_log_dist=prev_log_dist,
                )
                _record_auto_step_action_stats(world, step, delta_class)
            action_detail = None
            if not suppress_auto_diag:
                action_detail = auto_action_detail_text(
                    cmd,
                    score_effective,
                    action_meta=action_meta,
                )
            prev_log_x_err = local_gate_before_action.get("x_err")
            prev_log_dist = local_gate_before_action.get("dist")
            if cmd in ("f", "b"):
                prev_log_correction_type = "distance"
            elif cmd in ("l", "r"):
                prev_log_correction_type = "x_axis"
            else:
                prev_log_correction_type = None
            sent_snapshot = _capture_sent_action_snapshot(world) if not suppress_auto_diag else None
            segments = []
            if isinstance(action_meta, dict) and isinstance(action_meta.get("segments"), list):
                segments = [seg for seg in action_meta.get("segments") if isinstance(seg, dict)]
            elif isinstance(action_meta, dict):
                segments = [action_meta]
            for seg in segments:
                duration_ms = seg.get("duration_model_ms") or seg.get("duration_ms") or int(CONTROL_DT * 1000)
                power_used = seg.get("power")
                if power_used is None:
                    power_used = speed
                evt = MotionEvent(
                    cmd_to_motion_type(cmd),
                    int(float(power_used) * 255),
                    int(duration_ms),
                    speed_score=seg.get("score_effective") or seg.get("score_model"),
                )
                world.update_from_motion(evt)
            if next_module.success_gates_visible_only(world.process_rules or {}, step):
                world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0)) + 1
            if step_norm == "ALIGN_BRICK" and cmd in ("f", "b", "l", "r"):
                try:
                    hist_score = int(round(float(score_effective))) if score_effective is not None else None
                except (TypeError, ValueError):
                    hist_score = None
                if hist_score is None:
                    try:
                        hist_score = int(round(float(speed_score))) if speed_score is not None else int(DEFAULT_SPEED_SCORE)
                    except (TypeError, ValueError):
                        hist_score = int(DEFAULT_SPEED_SCORE)
                hist_cmd = str(cmd_sent or cmd).strip().lower()
                if hist_cmd not in ("f", "b", "l", "r"):
                    hist_cmd = str(cmd).strip().lower()
                recent_align_acts.append({"cmd": str(hist_cmd), "score": int(hist_score)})
                if isinstance(action_meta, dict):
                    last_align_action = {
                        "cmd": str(action_meta.get("cmd_sent") or cmd),
                        "score": hist_score,
                        "pwm": action_meta.get("pwm"),
                        "power": action_meta.get("power"),
                        "duration_ms": action_meta.get("duration_ms"),
                    }
                else:
                    last_align_action = {"cmd": str(cmd), "score": hist_score}
            if confirm_callback and robot:
                robot.stop()
            last_action_frame = post_act_analysis(
                world,
                vision,
                step=step,
                log=not align_silent,
                include_pause=False,
            )
            success_hit = run_full_gatecheck_after_act(
                world,
                vision,
                step,
                success_tracker,
                phase="align",
                log=not align_silent,
                observer=observer,
            )
            if not suppress_auto_diag:
                emit_auto_step_diagnostic(
                    world,
                    step,
                    action_detail,
                    pre_gate_text,
                    emit=not align_silent,
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                )
            if success_hit:
                if robot:
                    robot.stop()
                if not align_silent:
                    print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                    print_gate_summary_line(world, success_tracker)
                    print_success_events(world, step)
                return True, "success gate"
            if not skip_settle_pause:
                post_act_settle_pause(world, vision)
            post_frame_id = getattr(world, "_frame_id", 0)
            if post_frame_id <= pre_frame_id:
                stale_frame_streak = int(stale_frame_streak) + 1
                print(format_headline("[WARN] No new frame observed after action", COLOR_RED))
                if step_norm == "ALIGN_BRICK" and stale_frame_streak >= 3:
                    if robot:
                        robot.stop()
                    print(
                        format_headline(
                            "[WARN] ALIGN_BRICK frame stream appears stale; treating as lost visibility.",
                            COLOR_RED,
                        )
                    )
                    reason = (
                        f"no new frame id for {int(stale_frame_streak)} consecutive actions "
                        f"(pre={int(pre_frame_id)}, post={int(post_frame_id)})"
                    )
                    if _recover_visibility(source="stale_frames", reason=reason):
                        stale_pre_obs_streak = 0
                        stale_frame_streak = 0
                        not_visible_streak = 0
                        last_visible_ts = time.time()
                        prev_log_correction_type = None
                        prev_log_x_err = None
                        prev_log_dist = None
                        force_gap_switch_after_recovery = True
                        last_align_signature = None
                        continue
                    return False, "lost_vision_stale_frames"
            else:
                stale_frame_streak = 0
        else:
            if robot:
                robot.stop()
        if observer:
            observer("action", world, vision, cmd, speed, cmd_reason)
        time.sleep(CONTROL_DT)
        if last_action_frame:
            current_frame = {
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
            }
            if current_frame == last_action_frame:
                print(format_headline("[ERROR] Frame unchanged before/after action", COLOR_RED))

    if robot:
        robot.stop()
    if not align_silent:
        loops = getattr(world, "loop_id", 0)
        print(
            format_headline(
                f"[FAIL] Gave up after {loops} loops.",
                COLOR_RED,
            )
        )
    settle_deadline = time.time() + SUCCESS_SETTLE_S
    settle_tracker = new_success_tracker(step, world.process_rules)
    settle_action_idx = 0
    loop_id = 0
    while time.time() < settle_deadline:
        loop_id += 1
        world.loop_id = loop_id
        update_world_from_vision(world, vision, log=not align_silent)
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            world._last_obs_note = None
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        local_gate = next_module.align_local_gate_status(
            world,
            step,
            process_rules=world.process_rules or {},
        )
        success_truth_ok = bool(success_ok and local_gate.get("ok", True))
        gate_utils.record_success_gate_entry(world, step, success_truth_ok)
        if not align_silent:
            log_confidence(world, confidence, step)
        if success_truth_ok:
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print_gate_summary_line(world, settle_tracker)
                print_success_events(world, step)
            return True, "success gate"

        step_norm = normalize_step_label(step)
        if step_norm == "ALIGN_BRICK":
            brick = getattr(world, "brick", {}) or {}
            act_plan = next_module.select_align_brick_next_act(
                process_rules=world.process_rules or {},
                x_axis_mm=brick.get("x_axis", brick.get("offset_x")),
                dist_mm=brick.get("dist"),
                visible=bool(brick.get("visible")),
                angle_deg=brick.get("angle", 0.0),
                duration_s=CONTROL_DT,
            )
            cmd = act_plan.get("cmd")
            cmd_reason = act_plan.get("reason") or "align"
            speed_score = act_plan.get("score")
            speed = telemetry_robot_module.manual_speed_for_cmd(cmd, speed_score) if cmd and speed_score is not None else 0.0
        else:
            analytics = next_module.compute_alignment_decision(
                world=world,
                step=step,
                process_rules=world.process_rules or {},
                learned_rules=world.learned_rules or {},
                duration_s=CONTROL_DT,
            )
            cmd = analytics.get("cmd")
            speed = analytics.get("speed") or 0.0
            cmd_reason = analytics.get("worst_metric") or "align"
            speed_score = analytics.get("speed_score")
        cmd, speed, speed_score, cmd_reason = _apply_turn_only_demo_policy(
            cmd,
            speed,
            speed_score,
            cmd_reason,
        )
        if step_norm != "ALIGN_BRICK":
            speed = apply_pursuit_speed(speed)
            speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            local_gate_before_action = next_module.align_local_gate_status(
                world,
                step,
                process_rules=world.process_rules or {},
            )
            pre_gate_text = None
            pre_focus = None
            if not suppress_auto_diag:
                pre_gate_text = _auto_gate_snapshot_text(world, step, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            action_meta = send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=speed_score,
                auto_mode=True,
            )
            cmd_sent = cmd
            score_effective = speed_score
            if isinstance(action_meta, dict):
                cmd_sent = action_meta.get("cmd_sent") or cmd_sent
                score_effective = action_meta.get("score_effective")
                if score_effective is None:
                    score_effective = action_meta.get("score_model", speed_score)
            if not align_silent:
                settle_action_idx = int(settle_action_idx) + 1
                _emit_align_shorthand_action_line(
                    local_gate_before_action=local_gate_before_action,
                    prev_log_correction_type=prev_log_correction_type,
                    prev_log_x_err=prev_log_x_err,
                    prev_log_dist=prev_log_dist,
                    cmd=cmd,
                    score_effective=score_effective,
                    action_meta=action_meta,
                    action_index=settle_action_idx,
                    settle=True,
                )
            action_detail = None
            if not suppress_auto_diag:
                action_detail = auto_action_detail_text(
                    cmd,
                    score_effective,
                    action_meta=action_meta,
                )
            prev_log_x_err = local_gate_before_action.get("x_err")
            prev_log_dist = local_gate_before_action.get("dist")
            if cmd in ("f", "b"):
                prev_log_correction_type = "distance"
            elif cmd in ("l", "r"):
                prev_log_correction_type = "x_axis"
            else:
                prev_log_correction_type = None
            sent_snapshot = _capture_sent_action_snapshot(world) if not suppress_auto_diag else None
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if next_module.success_gates_visible_only(world.process_rules or {}, step):
                world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0)) + 1
            if confirm_callback and robot:
                robot.stop()
            post_act_analysis(
                world,
                vision,
                step=step,
                log=not align_silent,
                include_pause=False,
            )
            success_hit = run_full_gatecheck_after_act(
                world,
                vision,
                step,
                settle_tracker,
                phase="settle",
                log=not align_silent,
                observer=observer,
            )
            if not suppress_auto_diag:
                emit_auto_step_diagnostic(
                    world,
                    step,
                    action_detail,
                    pre_gate_text,
                    emit=not align_silent,
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                )
            if success_hit:
                if not align_silent:
                    print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                    print_gate_summary_line(world, settle_tracker)
                    print_success_events(world, step)
                return True, "success gate"
            if not skip_settle_pause:
                post_act_settle_pause(world, vision)
        else:
            if robot:
                if (
                    normalize_step_label(step) == "ALIGN_BRICK"
                    and not bool((getattr(world, "brick", None) or {}).get("visible"))
                ):
                    record_hold_display(world, step, _hold_reason(world, step, {"align": analytics}))
                robot.stop()
            time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"


def replay_segment(
    segment,
    step,
    robot,
    vision,
    world,
    observer=None,
    analysis_pause_s=0.0,
    confirm_callback=None,
    align_silent=False,
):
    step_key = normalize_step_label(step)
    events = segment.get("events") or []
    base_steps = build_motion_sequence(events)
    raw_steps = merge_motion_steps(base_steps)
    cfg = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        cfg = world.process_rules.get(step_key, {}) or {}
    loop_demo_acts = bool(isinstance(cfg, dict) and cfg.get("loop_demo_acts"))
    max_speed_score = None
    if isinstance(cfg, dict) and cfg.get("max_speed_score") is not None:
        try:
            max_speed_score = int(cfg.get("max_speed_score"))
        except (TypeError, ValueError):
            max_speed_score = None
    prefer_demo_speed = _auto_uses_demo_speed(step_key)
    if isinstance(cfg, dict) and cfg.get("prefer_demo_speed") is not None:
        prefer_demo_speed = bool(cfg.get("prefer_demo_speed"))
    demo_actions = nominal_actions_from_events(events)
    if prefer_demo_speed and demo_actions:
        loop_demo_acts = True
    steps = raw_steps if loop_demo_acts else smooth_motion_steps(raw_steps)
    if step_uses_alignment_control(step, world.process_rules):
        return run_alignment_segment(
            segment,
            step,
            robot,
            vision,
            world,
            steps,
            raw_steps,
            observer=observer,
            analysis_pause_s=analysis_pause_s,
            confirm_callback=confirm_callback,
            align_silent=align_silent,
        )
    if step_is_nominal_only(step, world.process_rules):
        actions = nominal_actions_from_events(events)
        if not actions:
            actions = [{"cmd": None, "speed_score": None, "power": 0.0, "duration_s": 0.1}]
        total_actions = len(actions)
        # Nominal replay is a strict script playback path; do not run gatecheck loops.
        world._gatecheck_status = None
        world._last_gate_summary = None
        prior_suppress = getattr(world, "suppress_brick_state_log", False)
        world.suppress_brick_state_log = True
        try:
            for idx, action in enumerate(actions, start=1):
                cmd = action.get("cmd")
                score = action.get("speed_score")
                pwm = None
                duration_ms = int(telemetry_robot_module.ACT_DURATION_MS)
                if cmd:
                    if score is None:
                        print(
                            format_headline(
                                f"[FAIL] {step}: nominal action missing speedScore for {cmd.upper()}",
                                COLOR_RED,
                            )
                        )
                        if robot:
                            robot.stop()
                        return False, "missing speedScore"
                    if not telemetry_robot_module.is_valid_speed_score(score):
                        print(
                            format_headline(
                                f"[FAIL] {step}: speedScore {score} out of range (expected 1-100)",
                                COLOR_RED,
                            )
                        )
                        if robot:
                            robot.stop()
                        return False, "invalid speedScore"
                    speed, pwm, score, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
                else:
                    speed = 0.0
                action_duration_s = max(0.001, (duration_ms or 0) / 1000.0)
                if cmd:
                    action_meta = send_robot_command(
                        robot,
                        world,
                        step,
                        cmd,
                        speed,
                        speed_score=score,
                        auto_mode=True,
                    )
                    cmd_sent = cmd
                    score_effective = score
                    if isinstance(action_meta, dict):
                        cmd_sent = action_meta.get("cmd_sent") or cmd_sent
                        if action_meta.get("score_effective") is not None:
                            score_effective = action_meta.get("score_effective")
                    world._last_action_obj = normalize_step_label(step)
                    world._last_action_time = time.time()
                    world._last_action_display = (
                        f"DEMO {idx}/{total_actions}: "
                        f"{action_display_text(cmd_sent, score_effective)}"
                    )
                    evt = MotionEvent(
                        cmd_to_motion_type(cmd),
                        int(speed * 255),
                        int(action_duration_s * 1000),
                    )
                    world.update_from_motion(evt)
                    if observer:
                        observer("action", world, vision, cmd, speed, "nominal replay")
                    time.sleep(action_duration_s)
                    post_act_analysis(world, vision, step=step, log=True)
                else:
                    if robot:
                        robot.stop()
                if robot:
                    robot.stop()
                if DEMO_ACTION_PAUSE_FRAMES > 0 and idx < total_actions:
                    wait_for_frame_settle(world, vision, DEMO_ACTION_PAUSE_FRAMES, log=False)
        finally:
            world.suppress_brick_state_log = prior_suppress
        if robot:
            robot.stop()
        return True, "nominal demo replay"
    if not steps:
        # Static segment (e.g. wait/observe). Insert a dummy nop step to allow wait_for_start_gates.
        steps = [MotionStep(None, 0.0, 0.1, "wait")]

    default_step = steps[0]
    required_acts_for_success = step_min_acts(step_key, world.process_rules)
    required_action_prefix_s = min_acts_prefix_duration(events, required_acts_for_success)
    if not step_is_nominal_only(step, world.process_rules):
        required_action_prefix_s = 0.0
    success_checks_enabled = required_action_prefix_s <= 0.0
    replay_action_start_time = None
    target_visible = success_visible_target(world, step_key)
    move_score = int(AUTO_CYCLE_MOVE_SCORE)
    check_score = int(AUTO_CYCLE_CHECK_SCORE)
    move_score = _apply_find_brick_turn_speed_policy(step_key, default_step.cmd, move_score, phase="move")
    check_score = _apply_find_brick_turn_speed_policy(step_key, default_step.cmd, check_score, phase="check")
    force_check_until = 0.0
    last_gate_frame_id = None
    start_speed = default_step.speed
    if default_step.cmd:
        start_score = _apply_find_brick_turn_speed_policy(
            step_key,
            default_step.cmd,
            move_score,
            phase="move",
        )
        start_speed = default_speed_for_cmd(default_step.cmd, start_score)
    start_status = wait_for_start_gates(
        world,
        vision,
        step_key,
        robot=robot,
        cmd=default_step.cmd,
        speed=start_speed,
        allow_success=success_checks_enabled,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
        print_gate_summary_line(world)
        print_success_events(world, step_key)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    allow_early_exit = True

    success_tracker = new_success_tracker(step, world.process_rules)
    clear_pending_auto_step_diagnostic(world)
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = default_step.speed

    if loop_demo_acts and demo_actions:
        while True:
            for action in demo_actions:
                cmd = action.get("cmd")
                if not cmd:
                    if robot:
                        robot.stop()
                    time.sleep(CONTROL_DT)
                    continue
                score = action.get("speed_score")
                if score is None:
                    if prefer_demo_speed and cmd:
                        try:
                            action_power = float(action.get("power", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            action_power = 0.0
                        action_speed = max(0.0, min(1.0, action_power / 255.0))
                        if action_speed > 0.0:
                            _, score = telemetry_robot_module.quantize_speed(cmd, speed=action_speed)
                    if score is None:
                        score = move_score
                if score is None:
                    score = move_score
                if max_speed_score is not None:
                    score = min(int(max_speed_score), int(score))
                score = _apply_find_brick_turn_speed_policy(step_key, cmd, score, phase="demo")
                action_duration_s = max(0.001, float(action.get("duration_s") or CONTROL_DT))
                pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step_key)
                action_meta = send_robot_command(
                    robot,
                    world,
                    step_key,
                    cmd,
                    0.0,
                    speed_score=score,
                    auto_mode=True,
                )
                pwm = None
                duration_ms = int(action_duration_s * 1000)
                cmd_sent = None
                if isinstance(action_meta, dict):
                    active_speed = float(action_meta.get("power", 0.0) or 0.0)
                    score_effective = action_meta.get("score_effective")
                    if score_effective is None:
                        score_effective = action_meta.get("score_model", score)
                    pwm = action_meta.get("pwm")
                    cmd_sent = action_meta.get("cmd_sent")
                    duration_ms = int(
                        action_meta.get("duration_model_ms")
                        or action_meta.get("duration_ms")
                        or duration_ms
                    )
                else:
                    active_speed = 0.0
                    score_effective = score
                record_action_display(
                    world,
                    step_key,
                    cmd,
                    active_speed,
                    speed_score=score_effective,
                    cmd_sent=cmd_sent,
                    pwm=pwm,
                    duration_ms=duration_ms,
                    score_model=score,
                )
                sent_snapshot = _capture_sent_action_snapshot(world)
                world._last_action_line = format_control_action_line(
                    cmd,
                    active_speed,
                    f"demo {int(score_effective)}%",
                )
                evt = MotionEvent(
                    cmd_to_motion_type(cmd),
                    int(active_speed * 255),
                    int(action_duration_s * 1000),
                )
                world.update_from_motion(evt)
                if observer:
                    observer("action", world, vision, cmd, active_speed, "demo replay")
                time.sleep(action_duration_s)
                post_act_analysis(world, vision, step=step_key, log=True, include_pause=False)
                success_hit = run_full_gatecheck_after_act(
                    world,
                    vision,
                    step_key,
                    success_tracker,
                    phase="replay",
                    log=True,
                    observer=observer,
                )
                detail = auto_action_detail_text(
                    cmd,
                    score_effective,
                    action_meta=action_meta,
                )
                emit_auto_step_diagnostic(
                    world,
                    step_key,
                    detail,
                    pre_gate_text,
                    emit=True,
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                )
                if success_hit:
                    if robot:
                        robot.stop()
                    print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                    print_gate_summary_line(world, success_tracker)
                    print_success_events(world, step_key)
                    return True, "success gate"
                post_act_settle_pause(world, vision)
            if not loop_demo_acts:
                break

    while True:
        last_action = None
        for motion_step in steps:
            if motion_step.label != last_action:
                update_world_from_vision(world, vision)
                flush_auto_step_diagnostic(world, step_key, emit=True)
                if observer:
                    observer("frame", world, vision, None, None, None)
                world._last_action_line = format_control_action_line(
                    motion_step.cmd,
                    motion_step.speed,
                    "replay",
                )
                last_action = motion_step.label
            last_cmd = motion_step.cmd
            last_speed_base = motion_step.speed
            step_start = time.time()
            step_success_met = False
            while time.time() - step_start < motion_step.duration_s:
                now = time.time()
                if replay_action_start_time is None:
                    replay_action_start_time = now
                update_world_from_vision(world, vision)
                flush_auto_step_diagnostic(world, step_key, emit=True)
                if observer:
                    observer("frame", world, vision, None, None, None)

                replay_action_elapsed_s = now - replay_action_start_time
                can_check_success = replay_action_elapsed_s >= (required_action_prefix_s - 1e-6)

                frame_id = int(getattr(world, "_frame_id", 0) or 0)
                fresh_frame = last_gate_frame_id is None or frame_id != last_gate_frame_id
                if fresh_frame:
                    last_gate_frame_id = frame_id

                visible_only = next_module.success_gates_visible_only(world.process_rules or {}, step_key)
                forcing_check = now < float(force_check_until or 0.0)
                phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
                desired_score = move_score if phase == "move" else check_score
                if motion_step.speed_score is not None:
                    desired_score = motion_step.speed_score
                    phase = "demo"
                elif prefer_demo_speed and motion_step.cmd:
                    try:
                        base_speed = float(motion_step.speed or 0.0)
                    except (TypeError, ValueError):
                        base_speed = 0.0
                    if base_speed > 0.0:
                        _, demo_score = telemetry_robot_module.quantize_speed(motion_step.cmd, speed=base_speed)
                        if demo_score is not None:
                            desired_score = demo_score
                            phase = "demo"
                if max_speed_score is not None:
                    desired_score = min(int(max_speed_score), int(desired_score))
                desired_score = _apply_find_brick_turn_speed_policy(
                    step_key,
                    motion_step.cmd,
                    desired_score,
                    phase=phase,
                )

                # If we're seeing positive evidence, stay in slow/check mode while we confirm/refute.
                if allow_early_exit and can_check_success and fresh_frame and phase == "check":
                    success_ok, confidence = evaluate_gate_status(world, step_key)
                    gate_utils.record_success_gate_entry(world, step_key, success_ok)
                    log_confidence(world, confidence, step_key)
                    success_met = gate_utils.update_gatecheck(
                        world,
                        step_key,
                        success_tracker,
                        success_ok,
                        phase="replay",
                    )
                    if success_met:
                        step_success_met = True
                        break
                    if gate_utils.should_hold_for_success_confirmation(visible_only, success_tracker, success_met):
                        force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                        phase = "check"
                        desired_score = check_score

                cmd = motion_step.cmd
                if cmd:
                    pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                    pre_focus = _capture_auto_diag_focus(world, step_key)
                    action_meta = send_robot_command(
                        robot,
                        world,
                        step_key,
                        cmd,
                        0.0,
                        speed_score=desired_score,
                        auto_mode=True,
                    )
                    if isinstance(action_meta, dict):
                        active_speed = float(action_meta.get("power", 0.0) or 0.0)
                        score_effective = action_meta.get("score_effective")
                        if score_effective is None:
                            score_effective = action_meta.get("score_model", desired_score)
                        cmd_sent = action_meta.get("cmd_sent") or cmd
                    else:
                        active_speed = 0.0
                        score_effective = desired_score
                        cmd_sent = cmd
                    sent_snapshot = _capture_sent_action_snapshot(world)
                    world._last_action_line = format_control_action_line(
                        cmd,
                        active_speed,
                        f"replay {phase} {int(score_effective)}%",
                    )
                    evt = MotionEvent(
                        cmd_to_motion_type(cmd),
                        int(active_speed * 255),
                        int(CONTROL_DT * 1000),
                    )
                    world.update_from_motion(evt)
                    if observer:
                        observer("action", world, vision, cmd, active_speed, "replay")
                    queue_auto_step_diagnostic(
                        world,
                        step_key,
                        auto_action_detail_text(cmd, score_effective, action_meta=action_meta),
                        pre_gate_text,
                        pre_focus=pre_focus,
                        sent_snapshot=sent_snapshot,
                    )
                else:
                    if robot:
                        robot.stop()
                time.sleep(CONTROL_DT)
            if step_success_met:
                flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                if robot:
                    robot.stop()
                print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                print_gate_summary_line(world, success_tracker)
                print_success_events(world, step_key)
                return True, "success gate"

        if not loop_demo_acts:
            break

    # Once replayed acts complete, always enable success checking to avoid indefinite acting loops.
    success_checks_enabled = True
    settle_deadline = time.time() + SUCCESS_SETTLE_S
    settle_tracker = new_success_tracker(step, world.process_rules)
    while time.time() < settle_deadline:
        now = time.time()
        if replay_action_start_time is None:
            replay_action_start_time = now
        update_world_from_vision(world, vision)
        flush_auto_step_diagnostic(world, step_key, emit=True)
        if observer:
            observer("frame", world, vision, None, None, None)
        replay_action_elapsed_s = now - replay_action_start_time
        frame_id = int(getattr(world, "_frame_id", 0) or 0)
        fresh_frame = last_gate_frame_id is None or frame_id != last_gate_frame_id
        if fresh_frame:
            last_gate_frame_id = frame_id

        visible_only = next_module.success_gates_visible_only(world.process_rules or {}, step_key)
        forcing_check = now < float(force_check_until or 0.0)
        phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
        desired_score = move_score if phase == "move" else check_score
        if prefer_demo_speed and last_cmd:
            try:
                last_base = float(last_speed_base or 0.0)
            except (TypeError, ValueError):
                last_base = 0.0
            if last_base > 0.0:
                _, demo_score = telemetry_robot_module.quantize_speed(last_cmd, speed=last_base)
                if demo_score is not None:
                    desired_score = demo_score
                    phase = "demo"
        desired_score = _apply_find_brick_turn_speed_policy(
            step_key,
            last_cmd,
            desired_score,
            phase=phase,
        )

        if success_checks_enabled and fresh_frame and phase == "check":
            success_ok, confidence = evaluate_gate_status(world, step_key)
            gate_utils.record_success_gate_entry(world, step_key, success_ok)
            log_confidence(world, confidence, step_key)
            success_met = gate_utils.update_gatecheck(
                world,
                step_key,
                settle_tracker,
                success_ok,
                phase="settle",
            )
            if success_met:
                flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                print_gate_summary_line(world, settle_tracker)
                print_success_events(world, step_key)
                return True, "success gate"
            if gate_utils.should_hold_for_success_confirmation(visible_only, settle_tracker, success_met):
                force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                phase = "check"
                desired_score = check_score

        if last_cmd:
            pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
            pre_focus = _capture_auto_diag_focus(world, step_key)
            action_meta = send_robot_command(
                robot,
                world,
                step_key,
                last_cmd,
                0.0,
                speed_score=desired_score,
                auto_mode=True,
            )
            if isinstance(action_meta, dict):
                active_speed = float(action_meta.get("power", 0.0) or 0.0)
                score_effective = action_meta.get("score_effective")
                if score_effective is None:
                    score_effective = action_meta.get("score_model", desired_score)
                cmd_sent = action_meta.get("cmd_sent") or last_cmd
            else:
                active_speed = 0.0
                score_effective = desired_score
                cmd_sent = last_cmd
            sent_snapshot = _capture_sent_action_snapshot(world)
            world._last_action_line = format_control_action_line(
                last_cmd,
                active_speed,
                f"settle {phase} {int(score_effective)}%",
            )
            evt = MotionEvent(
                cmd_to_motion_type(last_cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if observer:
                observer("action", world, vision, last_cmd, active_speed, "replay")
            queue_auto_step_diagnostic(
                world,
                step_key,
                auto_action_detail_text(last_cmd, score_effective, action_meta=action_meta),
                pre_gate_text,
                pre_focus=pre_focus,
                sent_snapshot=sent_snapshot,
            )
        else:
            if robot:
                robot.stop()
        time.sleep(CONTROL_DT)

    if True:
        tail_tracker = new_success_tracker(step, world.process_rules)
        while True:
            now = time.time()
            if replay_action_start_time is None:
                replay_action_start_time = now
            update_world_from_vision(world, vision)
            flush_auto_step_diagnostic(world, step_key, emit=True)
            if observer:
                observer("frame", world, vision, None, None, None)

            replay_action_elapsed_s = now - replay_action_start_time
            frame_id = int(getattr(world, "_frame_id", 0) or 0)
            fresh_frame = last_gate_frame_id is None or frame_id != last_gate_frame_id
            if fresh_frame:
                last_gate_frame_id = frame_id

            visible_only = next_module.success_gates_visible_only(world.process_rules or {}, step_key)
            forcing_check = now < float(force_check_until or 0.0)
            phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
            desired_score = move_score if phase == "move" else check_score
            if prefer_demo_speed and last_cmd:
                try:
                    last_base = float(last_speed_base or 0.0)
                except (TypeError, ValueError):
                    last_base = 0.0
                if last_base > 0.0:
                    _, demo_score = telemetry_robot_module.quantize_speed(last_cmd, speed=last_base)
                    if demo_score is not None:
                        desired_score = demo_score
                        phase = "demo"
            desired_score = _apply_find_brick_turn_speed_policy(
                step_key,
                last_cmd,
                desired_score,
                phase=phase,
            )

            if success_checks_enabled and fresh_frame and phase == "check":
                success_ok, confidence = evaluate_gate_status(world, step_key)
                gate_utils.record_success_gate_entry(world, step_key, success_ok)
                log_confidence(world, confidence, step_key)
                success_met = gate_utils.update_gatecheck(
                    world,
                    step_key,
                    tail_tracker,
                    success_ok,
                    phase="tail",
                )
                if success_met:
                    flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                    if robot:
                        robot.stop()
                    print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                    print_gate_summary_line(world, tail_tracker)
                    print_success_events(world, step_key)
                    return True, "success gate"
                if gate_utils.should_hold_for_success_confirmation(visible_only, tail_tracker, success_met):
                    force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                    phase = "check"
                    desired_score = check_score

            if last_cmd:
                pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step_key)
                action_meta = send_robot_command(
                    robot,
                    world,
                    step_key,
                    last_cmd,
                    0.0,
                    speed_score=desired_score,
                    auto_mode=True,
                )
                if isinstance(action_meta, dict):
                    active_speed = float(action_meta.get("power", 0.0) or 0.0)
                    score_effective = action_meta.get("score_effective")
                    if score_effective is None:
                        score_effective = action_meta.get("score_model", desired_score)
                    cmd_sent = action_meta.get("cmd_sent") or last_cmd
                else:
                    active_speed = 0.0
                    score_effective = desired_score
                    cmd_sent = last_cmd
                sent_snapshot = _capture_sent_action_snapshot(world)
                world._last_action_line = format_control_action_line(
                    last_cmd,
                    active_speed,
                    f"tail {phase} {int(score_effective)}%",
                )
                evt = MotionEvent(
                    cmd_to_motion_type(last_cmd),
                    int(active_speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
                if observer:
                    observer("action", world, vision, last_cmd, active_speed, "replay")
                queue_auto_step_diagnostic(
                    world,
                    step_key,
                    auto_action_detail_text(last_cmd, score_effective, action_meta=action_meta),
                    pre_gate_text,
                    pre_focus=pre_focus,
                    sent_snapshot=sent_snapshot,
                )
            else:
                if robot:
                    robot.stop()
            time.sleep(CONTROL_DT)

    flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
    pause_after_fail(robot)
    return False, "success gate not reached"
