#!/usr/bin/env python3
import argparse
import json
import collections
import math
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

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}
MIN_MM_TOL = 1.5



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

    # Action Description
    action_map = {
        "f": "move forward",
        "b": "move backward",
        "l": "turn left",
        "r": "turn right",
        "u": "lift mast",
        "d": "lower mast",
    }
    action = action_map.get(cmd, "move")
    
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


def record_action_display(world, step, cmd, speed, speed_score=None):
    if world is None or not cmd:
        return
    obj_key = normalize_step_label(step)
    world._last_action_obj = obj_key
    world._last_action_time = time.time()
    world._last_action_cmd = cmd
    world._last_action_speed = speed
    world._last_action_score = speed_score
    world._last_action_display = action_display_text(cmd, speed_score)


def _duration_used_ms_for_cmd(robot, cmd, duration_ms):
    try:
        duration_used_ms = int(duration_ms)
    except (TypeError, ValueError):
        duration_used_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 0) or 0)
    if cmd in ("l", "r"):
        eff_scale = telemetry_robot_module.turn_duration_scale(cmd)
        duration_used_ms = max(1, int(round(duration_used_ms * eff_scale)))
        last_turn_cmd = getattr(robot, "_last_turn_cmd", None)
        if last_turn_cmd != cmd:
            duration_used_ms = max(1, int(round(duration_used_ms * 0.4)))
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


def send_robot_command_pwm(robot, world, step, cmd, power, pwm, duration_ms, speed_score=None):
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

    _, score_effective = telemetry_robot_module.quantize_speed(cmd, speed=power_val)
    if score_effective is None:
        score_effective = speed_score
    record_action_display(world, step, cmd, power_val, speed_score=score_effective)

    duration_used_ms = _duration_used_ms_for_cmd(robot, cmd, duration_ms)
    if hasattr(robot, "send_command_pwm"):
        robot.send_command_pwm(cmd, pwm_val, duration_ms=duration_used_ms)
    else:
        robot.send_command(cmd, power_val, duration_ms=duration_used_ms)

    try:
        duration_model_ms = int(duration_ms)
    except (TypeError, ValueError):
        duration_model_ms = duration_used_ms
    return {
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
    step_key = normalize_step_label(step)

    def _coerce_micro_ms(value, fallback):
        try:
            ms = int(round(float(value)))
        except (TypeError, ValueError):
            ms = int(fallback)
        return max(1, min(5000, ms))

    def _smoothstep01(value):
        try:
            x = float(value)
        except (TypeError, ValueError):
            return 0.0
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - (2.0 * x))

    score_used = speed_score
    if score_used is None:
        if speed is None or speed <= 0:
            return None
        _, score_used = telemetry_robot_module.quantize_speed(cmd, speed=speed)
    if score_used is None:
        return None
    power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score_used)

    if auto_mode and step_key == "ALIGN_BRICK" and cmd == "f":
        micro = getattr(telemetry_robot_module, "AUTO_MICRO_ADJUSTMENTS", {}) or {}
        step_micro = micro.get(step_key) if isinstance(micro, dict) else None
        forward_micro = step_micro.get("forward") if isinstance(step_micro, dict) else None
        if isinstance(forward_micro, dict):
            creep_ms = _coerce_micro_ms(
                forward_micro.get("creep_ms", 20),
                20,
            )
            segment_ms = _coerce_micro_ms(
                forward_micro.get("segment_ms", creep_ms),
                creep_ms,
            )
        else:
            creep_ms = None
            segment_ms = None
        if creep_ms is None or segment_ms is None:
            forward_micro = None

        thresholds = next_module.align_brick_micro_forward_profile(
            getattr(world, "process_rules", None) if world is not None else None,
            step_key,
        )
        dist_mm = None
        if forward_micro is not None:
            try:
                dist_mm = float((world.brick or {}).get("dist"))
            except (TypeError, ValueError, AttributeError):
                dist_mm = None

        creep_score = telemetry_robot_module.SPEED_SCORE_MIN
        target_score = telemetry_robot_module.normalize_speed_score(score_used)
        mid_score = telemetry_robot_module.normalize_speed_score(
            int(round(creep_score + (target_score - creep_score) * 0.4))
        )
        segments = [(int(creep_score), int(creep_ms or 0))]
        if forward_micro is not None:
            dist_val = dist_mm if dist_mm is not None else float(thresholds["far_mm"])
            very_close = float(thresholds["very_close_mm"])
            close = float(thresholds["close_mm"])
            somewhat_close = float(thresholds["somewhat_close_mm"])
            far = float(thresholds["far_mm"])

            seg2_factor = (dist_val - very_close) / max(1e-6, close - very_close)
            # Use the same near-range ramp window for the final segment so the
            # target score (shown in `SUG`) is actually reached even when we're
            # still inside the "somewhat_close" band.
            seg3_factor = (dist_val - very_close) / max(1e-6, close - very_close)
            seg2_ms = int(round(float(segment_ms) * _smoothstep01(seg2_factor)))
            seg3_ms = int(round(float(segment_ms) * _smoothstep01(seg3_factor)))
            if seg2_ms > 0:
                segments.append((int(mid_score), int(seg2_ms)))
            if seg3_ms > 0:
                segments.append((int(target_score), int(seg3_ms)))

        sent_segments = []
        if forward_micro is not None:
            for seg_score, seg_ms in segments:
                seg_ms = _coerce_micro_ms(seg_ms, creep_ms)
                seg_power, seg_pwm, seg_score, _ = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, seg_score)
                meta = send_robot_command_pwm(
                    robot,
                    world,
                    step,
                    cmd,
                    seg_power,
                    seg_pwm,
                    seg_ms,
                    speed_score=int(seg_score),
                )
                if isinstance(meta, dict):
                    sent_segments.append(meta)
                time.sleep(float(seg_ms) / 1000.0)

        if forward_micro is not None:
            if not sent_segments:
                return None

            total_model_ms = sum(int(seg.get("duration_model_ms") or 0) for seg in sent_segments)
            total_ms = sum(int(seg.get("duration_ms") or 0) for seg in sent_segments)
            merged = dict(sent_segments[-1])
            merged["segments"] = sent_segments
            merged["duration_model_ms"] = int(total_model_ms)
            merged["duration_ms"] = int(total_ms)
            return merged

    if auto_mode and cmd in ("l", "r") and score_used == telemetry_robot_module.SPEED_SCORE_MIN:
        boost_pct = float(getattr(telemetry_robot_module, "AUTO_TURN_SPEED_BOOST_PCT", 0.0) or 0.0)
        if boost_pct > 0:
            boost_scale = 1.0 + (boost_pct / 100.0)
            try:
                pwm_cap = int(round(float(getattr(telemetry_robot_module, "MAX_PWM", 255) or 255)))
            except (TypeError, ValueError):
                pwm_cap = 255
            pwm_cap = max(0, min(255, pwm_cap))
            pwm = max(0, min(pwm_cap, int(round(pwm * boost_scale))))
            power = max(0.0, min(1.0, float(power) * boost_scale))
    return send_robot_command_pwm(robot, world, step, cmd, power, pwm, duration_ms, speed_score=score_used)


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
    analytics = next_module.compute_alignment_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        step,
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


def success_gate_metrics_for_step(metrics, step, step_rules=None):
    obj_key = normalize_step_label(step)
    rules = (step_rules or {}).get(obj_key, {})
    alignment_metrics = rules.get("alignment_metrics")
    if isinstance(alignment_metrics, list):
        return [metric for metric in metrics if metric in alignment_metrics]
    allowed = rules.get("success_metrics")
    if isinstance(allowed, list):
        return [metric for metric in metrics if metric in allowed]
    return metrics


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


def update_process_model_from_demos(logs, path=PROCESS_MODEL_FILE):
    model = load_process_model(path)
    steps = model.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        model["steps"] = steps

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
    step_rules = {}
    if isinstance(wall_step_rules, dict):
        step_rules.update(wall_step_rules)
    if isinstance(steps, dict):
        step_rules.update(steps)
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
    for obj, stats in []:
        if obj not in success_gates:
            continue

    all_steps = set(steps.keys())
    all_steps.update(start_gates.keys())
    all_steps.update(success_gates.keys())
    all_steps.update(attempt_types.keys())

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
        steps.append(MotionStep(cmd, speed, duration_s, cmd_name))
    return steps


def merge_motion_steps(steps, speed_tol=0.02):
    if not steps:
        return []
    merged = [steps[0]]
    for step in steps[1:]:
        last = merged[-1]
        if step.cmd == last.cmd and abs(step.speed - last.speed) <= speed_tol:
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
            smooth.append(MotionStep(step.cmd, capped_speed, duration, step.label))
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


def _gate_requirement_text(stats):
    if not isinstance(stats, dict):
        return ""
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        return f"={_fmt_gate_value(min_val)}"
    if isinstance(max_val, bool):
        return f"={_fmt_gate_value(max_val)}"
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
        success_check = telemetry_brick.evaluate_success_gates(world, obj_name, world.learned_rules or {}, world.process_rules)
        pct = 100.0 if success_check.ok else 0.0
    pct_display = int(max(0.0, min(100.0, pct)))
    metric_lines = []
    for entry in _success_gate_entries(world, obj_name):
        metric_lines.append(_stream_success_gate_line(obj_name, entry))

    suggested = analytics.get("suggestion") or "HOLD"
    last_display = getattr(world, "_last_action_display", None)
    last_obj = getattr(world, "_last_action_obj", None)
    last_time = getattr(world, "_last_action_time", None)
    robot_display = None
    if last_display and last_obj == obj_name:
        if last_time is None or (time.time() - last_time) <= ACTION_DISPLAY_TTL_S:
            robot_display = last_display
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") if cmd else None
    if cmd and speed is not None:
        success_ok, confidence = evaluate_gate_status(world, obj_name)
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        speed_score = analytics.get("speed_score")
        speed, speed_score = telemetry_robot_module.quantize_speed(
            cmd,
            speed=speed,
            score=speed_score,
        )
        suggested = action_display_text(cmd, speed_score)
    if suggested == "HOLD":
        scan_dir = (world.process_rules or {}).get(obj_name, {}).get("scan_direction")
        if scan_dir in ("l", "r"):
            suggested = action_display_text(scan_dir, DEFAULT_SPEED_SCORE)
    if suggested == "HOLD":
        reason = _hold_reason(world, obj_name, analytics)
        if reason:
            suggested = f"HOLD ({reason})"
    summary = [f"STEP: {obj_name}"]
    summary.extend(metric_lines)
    if robot_display:
        summary.append(f"ACT: {robot_display}")
        if suggested and suggested != robot_display:
            summary.append(f"SUG: {suggested}")
    else:
        summary.append(f"ACT: {suggested}")
    return summary, analytics


def update_stream_frame(world, vision):
    stream_state = getattr(world, "_stream_state", None)
    if stream_state is None or vision.current_frame is None:
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


def post_act_analysis(world, vision, step=None, log=True):
    last_frame = refresh_world_after_action(world, vision, log=log)
    if POST_ACT_PAUSE_FRAMES > 0:
        # Push an immediate frame before waiting for additional fresh frames.
        update_stream_frame(world, vision)
        wait_for_frame_settle(world, vision, POST_ACT_PAUSE_FRAMES, log=False)
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
    start_time = time.time()
    stable = 0
    success_tracker = new_success_tracker(step, world.process_rules)
    success_seen = False
    last_cmd = None
    last_speed = None
    last_reason = None
    if robot:
        robot.stop()
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
    action_speeds = derive_action_speeds(raw_steps)
    gate_bounds = telemetry_brick.success_gate_bounds(
        world.process_rules or {},
        world.learned_rules or {},
        step,
    )
    scan_cmd = telemetry_robot_module.resolve_scan_direction(world.process_rules, step)
    start_cmd = scan_cmd
    start_speed = action_speeds["scan"]
    step_key = normalize_step_label(step)
    # ALIGN_BRICK demos are often state-only, so derived scan speed can fall back to 50%.
    # Use explicit score-1 turn speed so start-gate scan matches "1%" tuning from world_model_robot.
    if step_key == "ALIGN_BRICK" and start_cmd in ("l", "r"):
        start_speed = telemetry_robot_module.manual_speed_for_cmd(
            start_cmd,
            telemetry_robot_module.SPEED_SCORE_MIN,
        )
    start_status = wait_for_start_gates(
        world,
        vision,
        step,
        robot=robot,
        cmd=start_cmd,
        speed=start_speed,
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
    last_cmd = None
    last_reason = None
    last_speed = None
    last_action_frame = None
    loop_id = 0

    while True:
        loop_id += 1
        world.loop_id = loop_id
        update_world_from_vision(world, vision, log=not align_silent)
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            world._last_obs_note = None
        # pause1 removed per request
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        gate_utils.record_success_gate_entry(world, step, success_ok)
        if not align_silent:
            log_confidence(world, confidence, step)
        success_met = gate_utils.update_gatecheck(
            world,
            step,
            success_tracker,
            success_ok,
            phase="align",
        )
        if success_met:
            if robot:
                robot.stop()
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print_gate_summary_line(world, success_tracker)
                print_success_events(world, step)
            return True, "success gate"

        analytics = next_module.compute_alignment_analytics(
            world,
            world.process_rules or {},
            world.learned_rules or {},
            step,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed_score = analytics.get("speed_score")
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    step,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, step, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            if not align_silent:
                pass
            if observer:
                observer("analysis", world, vision, cmd, speed, cmd_reason)
            if analysis_pause_s:
                time.sleep(analysis_pause_s)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            pre_frame_id = getattr(world, "_frame_id", 0)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            action_meta = send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=analytics.get("speed_score"),
                auto_mode=True,
            )
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
            if confirm_callback and robot:
                robot.stop()
            last_action_frame = post_act_analysis(world, vision, step=step, log=not align_silent)
            if run_full_gatecheck_after_act(
                world,
                vision,
                step,
                success_tracker,
                phase="align",
                log=not align_silent,
                observer=observer,
            ):
                if robot:
                    robot.stop()
                if not align_silent:
                    print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                    print_gate_summary_line(world, success_tracker)
                    print_success_events(world, step)
                return True, "success gate"
            post_frame_id = getattr(world, "_frame_id", 0)
            if post_frame_id <= pre_frame_id:
                print(format_headline("[WARN] No new frame observed after action", COLOR_RED))
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
        gate_utils.record_success_gate_entry(world, step, success_ok)
        if not align_silent:
            log_confidence(world, confidence, step)
        if success_ok:
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print_gate_summary_line(world, settle_tracker)
                print_success_events(world, step)
            return True, "success gate"
        if success_ok:
            if robot:
                robot.stop()
            time.sleep(CONTROL_DT)
            continue

        analytics = next_module.compute_alignment_analytics(
            world,
            world.process_rules or {},
            world.learned_rules or {},
            step,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed_score = analytics.get("speed_score")
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    step,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, step, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=analytics.get("speed_score"),
                auto_mode=True,
            )
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
            post_act_analysis(world, vision, step=step, log=not align_silent)
            if run_full_gatecheck_after_act(
                world,
                vision,
                step,
                settle_tracker,
                phase="settle",
                log=not align_silent,
                observer=observer,
            ):
                if not align_silent:
                    print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                    print_gate_summary_line(world, settle_tracker)
                    print_success_events(world, step)
                return True, "success gate"
        else:
            if robot:
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
    events = segment.get("events") or []
    base_steps = build_motion_sequence(events)
    raw_steps = merge_motion_steps(base_steps)
    steps = smooth_motion_steps(raw_steps)
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
                    score_effective = score
                    if isinstance(action_meta, dict) and action_meta.get("score_effective") is not None:
                        score_effective = action_meta.get("score_effective")
                    world._last_action_obj = normalize_step_label(step)
                    world._last_action_time = time.time()
                    world._last_action_display = (
                        f"DEMO {idx}/{total_actions}: "
                        f"{action_display_text(cmd, score_effective)}"
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
    step_key = step
    required_acts_for_success = step_min_acts(step_key, world.process_rules)
    required_action_prefix_s = min_acts_prefix_duration(events, required_acts_for_success)
    success_checks_enabled = required_action_prefix_s <= 0.0
    replay_action_start_time = None
    target_visible = success_visible_target(world, step_key)
    move_score = int(AUTO_CYCLE_MOVE_SCORE)
    check_score = int(AUTO_CYCLE_CHECK_SCORE)
    force_check_until = 0.0
    last_gate_frame_id = None
    start_speed = default_step.speed
    if default_step.cmd:
        start_speed = default_speed_for_cmd(default_step.cmd, move_score)
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

    cfg = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        cfg = world.process_rules.get(normalize_step_label(step_key), {}) or {}
    loop_demo_acts = bool(isinstance(cfg, dict) and cfg.get("loop_demo_acts"))

    success_tracker = new_success_tracker(step, world.process_rules)
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = default_step.speed

    while True:
        last_action = None
        for motion_step in steps:
            if motion_step.label != last_action:
                update_world_from_vision(world, vision)
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
                    else:
                        active_speed = 0.0
                        score_effective = desired_score
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
                else:
                    if robot:
                        robot.stop()
                time.sleep(CONTROL_DT)
            if step_success_met:
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
                print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                print_gate_summary_line(world, settle_tracker)
                print_success_events(world, step_key)
                return True, "success gate"
            if gate_utils.should_hold_for_success_confirmation(visible_only, settle_tracker, success_met):
                force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                phase = "check"
                desired_score = check_score

        if last_cmd:
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
            else:
                active_speed = 0.0
                score_effective = desired_score
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
                else:
                    active_speed = 0.0
                    score_effective = desired_score
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
            else:
                if robot:
                    robot.stop()
            time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"
