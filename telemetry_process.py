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
from helper_vision_config import (
    VISION_MODE_ARUCO,
    VISION_MODE_CYAN,
    active_vision_mode as _world_model_active_vision_mode,
    demos_dir_for_mode,
)
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


DEMO_DIR = demos_dir_for_mode()
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
    "start_gate_timeout_s": 30.0,
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
    "default_unique_smoothed_frames": 1,
    "steps": {},
}
GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = bool(
    gate_utils.DEFAULT_GATE_CHECKER_CONFIG.get("lite_only_aruco_experiment", False)
)
GATECHECK_ARUCO_FULL_PASS_SCALE = float(
    gate_utils.DEFAULT_GATE_CHECKER_CONFIG.get("aruco_full_gatecheck_pass_scale", 1.0)
)

# Steps that should keep an observe->act cadence without extra post-act settle waits.
# Full confirmation still runs through the gatecheck tracker/lite precheck rules.
FAST_POST_ACT_GATECHECK_STEPS = {
    "ALIGN_BRICK",
    "POSITION_BRICK",
    "BRICK_LOCK",
    "BRICK_LOCK_WALL",
    "SEAT_BRICK",
    "SEAT_BRICK2",
    "EXIT_WALL",
}

PROCESS_SUCCESS_METRICS_BY_STEP = {
    "FIND_WALL": ("visible", "angle_abs"),
    "EXIT_WALL": ("visible",),
    "FIND_BRICK": ("visible", "xAxis_offset_abs", "yAxis_offset_abs"),
    "APPROACH_VECTOR_BRICK_SUPPLY": ("visible", "angle_abs", "x_axis", "y_axis"),
    "FIND_TOPMOST_BRICK": ("inCrosshairs",),
    "FIND_TOPMOST_BRICK_WALL": ("inCrosshairs",),
    "BRICK_LOCK": ("xAxis_offset_abs", "dist"),
    "BRICK_LOCK_WALL": ("xAxis_offset_abs", "yAxis_offset_abs", "dist"),
    "ALIGN_BRICK": ("visible", "xAxis_offset_abs", "yAxis_offset_abs", "dist"),
    "SEAT_BRICK": ("visible", "dist"),
    "SEAT_BRICK2": ("visible", "dist", "x_axis", "y_axis"),
    "ELEVATE_BRICK": ("visible",),
    "FIND_WALL2": ("visible", "xAxis_offset_abs", "y_axis"),
    "APPROACH_VECTOR_WALL": ("visible", "angle_abs", "xAxis_offset_abs", "yAxis_offset_abs"),
    "POSITION_BRICK": ("visible", "xAxis_offset_abs", "yAxis_offset_abs", "dist", "brick_above"),
    "RETREAT": ("visible", "dist"),
}

PROCESS_START_METRICS_BY_STEP = {
    "BRICK_LOCK": ("visible",),
    "BRICK_LOCK_WALL": ("visible",),
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
ALIGN_RECOVERY_PREMOVE_RECHECK_HOLD_MULTIPLIER = 3.0
ALIGN_RECOVERY_PREMOVE_RECHECK_CYAN_HOLD_MULTIPLIER = 3.0
CYAN_RAW_FALLBACK_MIN_CONFIDENCE_PCT = 10.0
ARUCO_RAW_FALLBACK_MIN_CONFIDENCE_PCT = 50.0
ALIGN_BRICK_TOPMOST_LIFT_SCORE = 1
ALIGN_BRICK_TOPMOST_MAX_ACTS = 70
ALIGN_BRICK_TOPMOST_TIMEOUT_S = 30.0
ALIGN_BRICK_TOPMOST_GATECHECK_MAX_TRIES = 6
ALIGN_BRICK_SECOND_UPPERMOST_DESCENT_HOTKEY = "l"
ALIGN_BRICK_SECOND_UPPERMOST_DESCENT_PULSES = 3
POST_ACTION_OBSERVE_DELAY_S = 0.5

MM_METRICS = {
    "xAxis_offset_abs",
    "yAxis_offset_abs",
    "x_axis",
    "y_axis",
    "dist",
    "distance",
    "lift_height",
}
SUCCESS_GATE_EXCLUDE_ANGLE_STEPS = {
    "ALIGN_BRICK",
    "POSITION_BRICK",
}
RETIRED_PROCESS_STEPS = {
    "PLACE",
}
AUTO_DIAG_OFFSET_PRIORITY = ("xAxis_offset_abs", "dist")
AUTO_DIAG_SINGLE_OFFSET_PRECHECK_STEPS = {
    "ALIGN_BRICK",
    "POSITION_BRICK",
}
AUTO_DIAG_FOCUS_PRECHECK_STEPS = {
    "FIND_TOPMOST_BRICK",
    "FIND_TOPMOST_BRICK_WALL",
    "BRICK_LOCK",
    "BRICK_LOCK_WALL",
}
AUTO_DIAG_COMPACT_BOOLEAN_DELTA_STEPS = {
    "FIND_TOPMOST_BRICK",
    "FIND_TOPMOST_BRICK_WALL",
    "BRICK_LOCK",
    "BRICK_LOCK_WALL",
}
GAP_SWITCH_REASSURE_STEPS = {
    "POSITION_BRICK",
}
AUTO_SPEED_SCORE_HARD_MAX = 25
REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP = 20
REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT = 8
AUTO_DEMO_SPEED_STEPS = {
    "FIND_WALL",
    "EXIT_WALL",
    "FIND_BRICK",
    "FIND_WALL2",
}
TOPMOST_GROUND_UP_LEVEL2_STEPS = {
    "FIND_TOPMOST_BRICK",
    "FIND_TOPMOST_BRICK_WALL",
}
_STEP_NUMBER_BY_LABEL_CACHE = None
FIND_BRICK_TURN_MOVE_MAX_SCORE = 2
FIND_BRICK_TURN_CHECK_MAX_SCORE = 1
MIN_MM_TOL = 1.5
X_AXIS_OFFSET_ABS_TOL_MM = 1.4
try:
    ARUCO_GATE_MIN_CONFIDENCE = float(
        _MANUAL_CONFIG.get("aruco_gate_min_confidence", 50.0)
    )
except (TypeError, ValueError):
    ARUCO_GATE_MIN_CONFIDENCE = 50.0
ARUCO_GATE_MIN_CONFIDENCE = max(0.0, min(100.0, float(ARUCO_GATE_MIN_CONFIDENCE)))

LOCKED_SUCCESS_GATE_MODE_ALIASES = {
    "aruco": VISION_MODE_ARUCO,
    "marker": VISION_MODE_ARUCO,
    "markers": VISION_MODE_ARUCO,
    "cyan": VISION_MODE_CYAN,
    "yolo": VISION_MODE_CYAN,
    "markerless": VISION_MODE_CYAN,
}



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


def _step_allows_topmost_ground_up_level2(step):
    step_key = normalize_step_label(step)
    return bool(step_key in TOPMOST_GROUND_UP_LEVEL2_STEPS)


def _step_number_for_label(step):
    global _STEP_NUMBER_BY_LABEL_CACHE
    step_key = normalize_step_label(step)
    if not step_key:
        return None
    cache = _STEP_NUMBER_BY_LABEL_CACHE
    if not isinstance(cache, dict):
        cache = {}
        try:
            sequence = telemetry_robot_module.step_sequence()
        except Exception:
            sequence = []
        for idx, item in enumerate(sequence, start=1):
            item_key = normalize_step_label(getattr(item, "value", item))
            if item_key and item_key not in cache:
                cache[item_key] = int(idx)
        _STEP_NUMBER_BY_LABEL_CACHE = cache
    try:
        return int(cache.get(step_key))
    except (TypeError, ValueError):
        return None


def default_speed_for_cmd(cmd, score=DEFAULT_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def max_speed_for_cmd(cmd, score=MAX_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def _vision_backend_name(vision):
    if isinstance(vision, ArucoBrickVision):
        return "aruco"
    return "cyan"


def _raw_visibility_fallback_min_confidence_pct(vision):
    backend = _vision_backend_name(vision)
    if backend == "aruco":
        return max(1.0, float(ARUCO_RAW_FALLBACK_MIN_CONFIDENCE_PCT))
    # Cyan can be noisier than ArUco; use a relaxed confidence floor when
    # deciding whether raw detections are good enough to keep telemetry visible.
    floor = float(CYAN_RAW_FALLBACK_MIN_CONFIDENCE_PCT)
    try:
        conf_threshold = float(getattr(vision, "conf_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf_threshold = 0.0
    if conf_threshold > 0.0:
        floor = min(floor, conf_threshold * 100.0)
    return max(1.0, float(floor))


def _allow_raw_visibility_fallback(vision, frame):
    backend = _vision_backend_name(vision)
    if backend not in ("cyan", "aruco"):
        return False
    if not bool((frame or {}).get("found")):
        return False
    try:
        conf = float((frame or {}).get("conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return conf >= _raw_visibility_fallback_min_confidence_pct(vision)


def _visibility_recheck_mode_hold_multiplier(vision):
    # Runtime uses exactly two backends today: ArUco markers and cyan/YOLO.
    # Keep ArUco timing unchanged; extend hold window only for cyan backend.
    if vision is None or isinstance(vision, ArucoBrickVision):
        return 1.0
    return max(1.0, float(ALIGN_RECOVERY_PREMOVE_RECHECK_CYAN_HOLD_MULTIPLIER))


def visibility_recheck_hold_seconds(vision, *, base_hold_s=None, control_dt=None):
    if base_hold_s is None:
        base_hold_s = VISIBILITY_LOST_HOLD_S
    if control_dt is None:
        control_dt = CONTROL_DT
    hold_window_s = max(float(control_dt), float(base_hold_s))
    hold_window_s *= max(1.0, float(ALIGN_RECOVERY_PREMOVE_RECHECK_HOLD_MULTIPLIER))
    hold_window_s *= _visibility_recheck_mode_hold_multiplier(vision)
    return float(hold_window_s)


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
        return max(1, int(LITE_GATE_DEFAULT_UNIQUE_FRAMES))
    return max(1, int(frames))


def apply_gate_checker_config(cfg):
    global GATECHECK_CONSECUTIVE_REQUIRED
    global GATECHECK_MAJORITY_WINDOW
    global GATECHECK_MAJORITY_REQUIRED
    global GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT
    global GATECHECK_ARUCO_FULL_PASS_SCALE
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
    GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = bool(
        cfg.get(
            "lite_only_aruco_experiment",
            default_cfg.get("lite_only_aruco_experiment", False),
        )
    )
    GATECHECK_ARUCO_FULL_PASS_SCALE = max(
        0.05,
        min(
            1.0,
            _as_float(
                cfg.get(
                    "aruco_full_gatecheck_pass_scale",
                    default_cfg.get("aruco_full_gatecheck_pass_scale", 1.0),
                ),
                default_cfg.get("aruco_full_gatecheck_pass_scale", 1.0),
            ),
        ),
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
COLOR_BLUE_BRIGHT = "\033[94m"
COLOR_YELLOW = "\033[33m"
COLOR_ORANGE_BRIGHT = "\033[38;5;208m"
COLOR_PINK = "\033[38;5;213m"
COLOR_MAGENTA_BRIGHT = "\033[95m"
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


def _log_dist_priority_cheat_act(step, act_plan, *, align_silent=False, settle=False):
    if align_silent or not isinstance(act_plan, dict):
        return
    if not bool(act_plan.get("cheat_dist_priority")):
        return
    step_key = normalize_step_label(step) or str(step or "UNKNOWN")
    phase_label = "SETTLE" if bool(settle) else "ALIGN"
    cmd = str(act_plan.get("cmd") or "").strip().upper() or "NONE"
    try:
        score = int(round(float(act_plan.get("score"))))
    except (TypeError, ValueError):
        score = None
    ctx = act_plan.get("cheat_dist_priority_context")
    if not isinstance(ctx, dict):
        ctx = {}

    def _fmt(name, precision=2):
        try:
            return f"{float(ctx.get(name)):.{int(precision)}f}"
        except (TypeError, ValueError):
            return "N/A"

    score_text = str(int(score)) if score is not None else "N/A"
    print(
        format_headline(
            f"[DIST PRIORITY CHEAT ACT] {step_key} {phase_label}: FORCING DISTANCE GAP CORRECTION",
            COLOR_MAGENTA_BRIGHT,
        )
    )
    print(
        format_headline(
            (
                f"[DIST PRIORITY CHEAT ACT] {step_key} {phase_label}: "
                f"CMD={cmd} SCORE={score_text} "
                f"DIST_OUTSIDE_MM={_fmt('dist_outside_mm')} DIST_RATIO={_fmt('dist_ratio')} "
                f"X_OUTSIDE_MM={_fmt('x_outside_mm')} X_RATIO={_fmt('x_ratio')} "
                f"Y_OUTSIDE_MM={_fmt('y_outside_mm')} Y_RATIO={_fmt('y_ratio')}"
            ),
            COLOR_MAGENTA_BRIGHT,
        )
    )
    print(
        format_headline(
            (
                f"[DIST PRIORITY CHEAT ACT] {step_key} {phase_label}: "
                f"THRESHOLDS MIN_RATIO={_fmt('min_ratio')} MIN_OUTSIDE_MM={_fmt('min_outside_mm')}"
            ),
            COLOR_MAGENTA_BRIGHT,
        )
    )


def _log_gap_rotation_tech_debt_act(world, step, act_plan, *, align_silent=False, settle=False):
    if align_silent or not isinstance(act_plan, dict):
        return
    notes = act_plan.get("exception_tech_debt_notes")
    if not isinstance(notes, (list, tuple)):
        notes = []
    notes_clean = [str(note).strip() for note in notes if str(note).strip()]
    gap_rotation_active = bool(act_plan.get("gap_rotation_active"))
    chunk_switch = bool(act_plan.get("gap_rotation_chunk_switch"))
    non_repeat_override = bool(act_plan.get("gap_rotation_non_repeat_override"))
    if not notes_clean and not chunk_switch and not non_repeat_override:
        return
    step_key = normalize_step_label(step) or str(step or "UNKNOWN")
    phase_label = "SETTLE" if bool(settle) else "ALIGN"
    corr_type = str(act_plan.get("correction_type") or "none").strip().lower() or "none"
    try:
        chunk_prog = float(act_plan.get("gap_rotation_chunk_progress_mm"))
    except (TypeError, ValueError):
        chunk_prog = None
    try:
        chunk_goal = float(act_plan.get("gap_rotation_chunk_target_mm"))
    except (TypeError, ValueError):
        chunk_goal = None
    sig = (
        str(step_key),
        str(phase_label),
        tuple(notes_clean),
        bool(chunk_switch),
        bool(non_repeat_override),
        str(corr_type),
        None if chunk_prog is None else round(float(chunk_prog), 2),
        None if chunk_goal is None else round(float(chunk_goal), 2),
    )
    sig_map = getattr(world, "_gap_rotation_tech_debt_log_sig", None)
    if not isinstance(sig_map, dict):
        sig_map = {}
        try:
            world._gap_rotation_tech_debt_log_sig = sig_map
        except Exception:
            sig_map = {}
    last_sig = sig_map.get(step_key)
    force_emit = bool(chunk_switch or non_repeat_override)
    if not force_emit and last_sig == sig:
        return
    sig_map[step_key] = sig
    for note in notes_clean:
        print(format_headline("[EXCEPTION]", COLOR_MAGENTA_BRIGHT, f" [{step_key}] {note}"))
    if gap_rotation_active:
        chunk_txt = "N/A" if (chunk_prog is None or chunk_goal is None) else f"{float(chunk_prog):.1f}/{float(chunk_goal):.1f}mm"
        status_tags = []
        if chunk_switch:
            status_tags.append("chunk-switch")
        if non_repeat_override:
            status_tags.append("recovery-nonrepeat")
        status_text = ", ".join(status_tags) if status_tags else "tracking"
        print(
            format_headline(
                "[EXCEPTION]",
                COLOR_MAGENTA_BRIGHT,
                (
                    f" [{step_key}] {phase_label} gap-rotation status: "
                    f"correction={str(corr_type)} chunk={chunk_txt} ({status_text})."
                ),
            )
        )


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
    consecutive_required = int(GATECHECK_CONSECUTIVE_REQUIRED)
    majority_window = int(GATECHECK_MAJORITY_WINDOW)
    majority_required = int(GATECHECK_MAJORITY_REQUIRED)
    visible_only = isinstance(process_rules, dict) and next_module.success_gates_visible_only(process_rules, step)

    tracker = gate_utils.SuccessGateTracker(
        consecutive_required,
        majority_window,
        majority_required,
    )
    if _active_success_gate_mode() == VISION_MODE_ARUCO:
        pass_scale = max(0.05, min(1.0, float(GATECHECK_ARUCO_FULL_PASS_SCALE)))
        if pass_scale < 1.0:
            base_consec_pass = max(
                1,
                int(getattr(tracker, "consecutive_pass_required", consecutive_required)),
            )
            base_majority_pass = max(
                1,
                int(getattr(tracker, "majority_pass_required", majority_required)),
            )
            tracker.consecutive_pass_required = max(
                1,
                min(
                    int(consecutive_required),
                    int(math.ceil(float(base_consec_pass) * float(pass_scale))),
                ),
            )
            tracker.majority_pass_required = max(
                1,
                min(
                    int(majority_required),
                    int(math.ceil(float(base_majority_pass) * float(pass_scale))),
                ),
            )
    if visible_only:
        # Do not relax full gatecheck thresholds for visible-only steps.
        # Keep strict tracker semantics consistent with configured windows.
        tracker.consecutive_pass_required = int(consecutive_required)
        tracker.majority_pass_required = int(majority_required)
    # Full gatecheck hardening: if tracker windows require multi-frame evidence,
    # never allow pass thresholds to collapse to a single-frame confirmation.
    if max(int(consecutive_required), int(majority_required)) > 1:
        full_floor = 2
        tracker.consecutive_pass_required = min(
            int(consecutive_required),
            max(
                int(full_floor),
                int(getattr(tracker, "consecutive_pass_required", consecutive_required)),
            ),
        )
        tracker.majority_pass_required = min(
            int(majority_required),
            max(
                int(full_floor),
                int(getattr(tracker, "majority_pass_required", majority_required)),
            ),
        )
    if lite_frames is not None:
        # Lite check is a precheck only; full gate tracking still uses
        # traditional consecutive/majority confirmation.
        tracker.lite_unique_frames = int(lite_frames)
    return tracker


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
    if world is None:
        return []
    process_rules = getattr(world, "process_rules", None) or {}
    learned_rules = getattr(world, "learned_rules", None) or {}
    visibility_grace_s = _visibility_grace_s(world, step)
    try:
        return telemetry_brick.success_gate_entries(
            world,
            step,
            learned_rules,
            process_rules=process_rules,
            visibility_grace_s=visibility_grace_s,
        )
    except Exception:
        return []


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
                min_val = stats.get("min")
                if isinstance(min_val, bool):
                    gate_bits.append(f"={_fmt_gate_value(min_val)}")
                else:
                    gate_bits.append(f">={_fmt_gate_value(min_val)}")
            if "max" in stats:
                max_val = stats.get("max")
                if isinstance(max_val, bool):
                    gate_bits.append(f"={_fmt_gate_value(max_val)}")
                else:
                    gate_bits.append(f"<={_fmt_gate_value(max_val)}")
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


def _success_entry_matches(step, entry, process_rules=None):
    metric = entry.get("metric")
    stats = entry.get("stats") or {}
    value = entry.get("value")
    direction = metric_direction(metric, step, process_rules=process_rules)
    matched = _gate_entry_matches(metric, value, stats, direction)
    if metric == "visible" and bool(entry.get("confident_visible_recent")) and gate_utils.bool_gate_target(stats) is False:
        return False
    return bool(matched)


def _success_event_state_and_suffix(entry, actual):
    actual_text = str(actual if actual is not None else "n/a")
    value = entry.get("value")
    state_text = _fmt_gate_value(value) if value is not None else "n/a"
    if state_text and actual_text.startswith(state_text):
        return state_text, actual_text[len(state_text):]
    if " (" in actual_text:
        head, tail = actual_text.split(" (", 1)
        if head:
            return head, f" ({tail}"
    return actual_text, ""


def format_success_event_lines(world, step, *, colored=False):
    entries = _success_gate_entries(world, step)
    lines = []
    process_rules = getattr(world, "process_rules", None) if world is not None else None
    for entry in entries:
        metric, actual, gate = _format_success_gate_entry(world, entry)
        if not colored:
            lines.append(f"{metric}={actual} ({gate})")
            continue
        match_ok = _success_entry_matches(step, entry, process_rules=process_rules)
        color = COLOR_GREEN if match_ok else COLOR_RED
        state_text, suffix = _success_event_state_and_suffix(entry, actual)
        lines.append(f"{metric}={color}{state_text}{COLOR_RESET}{suffix} ({gate})")
    return lines


def format_success_details(world, step):
    lines = format_success_event_lines(world, step)
    if not lines:
        return "[SUCCESS DETAILS] no success gates"
    return "[SUCCESS DETAILS] " + "; ".join(lines)


def format_success_gate_requirements(world, step):
    entries = _success_gate_entries(world, step)
    if not entries:
        return "none"
    parts = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("metric") or "").strip()
        if not metric:
            continue
        stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
        requirement = _gate_requirement_text(stats)
        parts.append(f"{metric}{requirement}")
    return ", ".join(parts) if parts else "none"


def print_success_events(world, step):
    lines = format_success_event_lines(world, step, colored=True)
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
    action = action_map.get(str(cmd).strip().lower(), "move")
    
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

def action_sent_display_text(
    cmd,
    speed_score=None,
    *,
    cmd_sent=None,
    pwm=None,
    power=None,
    duration_ms=None,
    ease_note=None,
    anti_alias_note=None,
):
    # Operator-facing displays should report semantic motion intent (`cmd`).
    # Raw transport remaps (`cmd_sent`) belong in explicit wire/debug logs.
    cmd_display = str(cmd).strip()
    if not cmd_display:
        return "HOLD"
    score_suffix = f" {int(speed_score)}%" if speed_score is not None else ""
    parts = [f"{cmd_display.upper()}"]
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
    note_value = ease_note if ease_note else anti_alias_note
    if note_value:
        parts.append(str(note_value))
    details = ""
    if len(parts) > 1:
        details = " (" + ", ".join(parts[1:]) + ")"
    return f"{parts[0]}{score_suffix}{details}"


def _join_action_notes(*notes):
    merged = []
    seen = set()
    for note in notes:
        text = str(note or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        merged.append(text)
        seen.add(text)
    if not merged:
        return None
    return " | ".join(merged)


def auto_action_detail_text(cmd, speed_score=None, *, action_meta=None, context_note=None):
    if isinstance(action_meta, dict):
        score_effective = action_meta.get("score_effective")
        if score_effective is None:
            score_effective = action_meta.get("score_model", speed_score)
        duration_ms = action_meta.get("duration_ms")
        if duration_ms is None:
            duration_ms = action_meta.get("duration_model_ms")
        note_value = _join_action_notes(
            action_meta.get("ease_in_out_note"),
            action_meta.get("anti_alias_note"),
            context_note,
        )
        return action_sent_display_text(
            cmd,
            score_effective,
            cmd_sent=action_meta.get("cmd_sent"),
            pwm=action_meta.get("pwm"),
            power=action_meta.get("power"),
            duration_ms=duration_ms,
            ease_note=note_value,
            anti_alias_note=note_value,
        )
    text = action_display_text(cmd, speed_score)
    note_text = str(context_note or "").strip()
    if not note_text:
        return text
    return f"{text} ({note_text})"


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
    ease_note=None,
    score_model=None,
    anti_alias_note=None,
):
    if world is None or not cmd:
        return
    obj_key = normalize_step_label(step)
    world._last_action_obj = obj_key
    world._last_action_time = time.time()
    world._last_action_cmd = cmd
    world._last_action_speed = speed
    world._last_action_score = speed_score
    world._last_action_score_model = score_model
    world._last_action_cmd_sent = cmd_sent
    world._last_action_pwm = pwm
    world._last_action_duration_ms = duration_ms
    note_value = ease_note if ease_note else anti_alias_note
    world._last_action_ease_in_out_note = note_value
    world._last_action_anti_alias_note = note_value
    world._last_action_display = action_display_text(cmd, speed_score)
    world._last_action_sent_display = action_sent_display_text(
        cmd,
        speed_score,
        cmd_sent=cmd_sent,
        pwm=pwm,
        power=speed,
        duration_ms=duration_ms,
        ease_note=note_value,
        anti_alias_note=anti_alias_note,
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


def _duration_used_ms_for_cmd(robot, cmd, duration_ms, *, auto_mode=False, half_first_turn_pulse=True):
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
            if bool(half_first_turn_pulse) and last_turn_cmd != cmd:
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


def _align_correction_type_for_cmd(cmd, *, fallback=None):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key in ("f", "b"):
        return "distance"
    if cmd_key in ("l", "r"):
        return "x_axis"
    if cmd_key in ("u", "d"):
        return "y_axis"
    fallback_key = str(fallback or "").strip().lower()
    if fallback_key in {"distance", "x_axis", "y_axis"}:
        return fallback_key
    return "x_axis"


def _align_error_signature_for_correction(correction_type, local_gate_before_action):
    ctype = str(correction_type or "").strip().lower()
    state = local_gate_before_action if isinstance(local_gate_before_action, dict) else {}
    try:
        if ctype == "distance":
            dist_now = state.get("dist")
            dist_target = state.get("dist_target")
            return round(float(dist_now) - float(dist_target), 2)
        if ctype == "y_axis":
            return round(float(state.get("y_err")), 2)
        return round(float(state.get("x_err")), 2)
    except (TypeError, ValueError):
        return None


def _align_local_gate_status_single_source(world, step):
    entries = _success_gate_entries(world, step)
    if not isinstance(entries, list) or not entries:
        return next_module.align_local_gate_status(
            world,
            step,
            process_rules=getattr(world, "process_rules", None) or {},
        )

    by_metric = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metric_key = str(entry.get("metric") or "").strip()
        if metric_key:
            by_metric[metric_key] = entry

    def _entry(metric_key):
        return by_metric.get(str(metric_key) or "")

    def _entry_alias(*metric_keys):
        for metric_key in metric_keys:
            entry = _entry(metric_key)
            if isinstance(entry, dict):
                return entry
        return None

    def _stats(metric_key):
        entry = _entry(metric_key)
        raw = entry.get("stats") if isinstance(entry, dict) else None
        return raw if isinstance(raw, dict) else {}

    def _stats_alias(*metric_keys):
        for metric_key in metric_keys:
            stats = _stats(metric_key)
            if stats:
                return stats
        return {}

    def _num(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    visible_entry = _entry("visible")
    visible_val = None
    if isinstance(visible_entry, dict):
        visible_val = visible_entry.get("effective_visible")
        if visible_val is None:
            visible_val = visible_entry.get("raw_visible")
        if visible_val is None:
            visible_val = visible_entry.get("value")
    if visible_val is None:
        visible_val = bool((getattr(world, "brick", None) or {}).get("visible"))
    visible = bool(visible_val)

    x_entry = _entry_alias("x_axis", "xAxis_offset_abs", "xAxis_offset")
    x_stats = _stats_alias("x_axis", "xAxis_offset_abs", "xAxis_offset")
    x_axis = _num(_auto_diag_value_from_entry(x_entry)) if isinstance(x_entry, dict) else None
    x_required = bool(x_stats)
    x_target = _num(x_stats.get("target"))
    if x_target is None:
        x_target = 0.0
    x_tol = abs(_num(x_stats.get("tol")) or 0.0)
    x_err = None if x_axis is None else float(x_axis - x_target)
    x_abs_err = None if x_err is None else abs(float(x_err))
    x_within_tol = (not x_required) or bool(x_abs_err is not None and x_abs_err <= x_tol)

    y_entry = _entry_alias("y_axis", "yAxis_offset_abs", "yAxis_offset")
    y_stats = _stats_alias("y_axis", "yAxis_offset_abs", "yAxis_offset")
    y_axis = _num(_auto_diag_value_from_entry(y_entry)) if isinstance(y_entry, dict) else None
    y_required = bool(y_stats)
    y_target = _num(y_stats.get("target"))
    if y_target is None:
        y_target = 0.0
    y_tol = _num(y_stats.get("tol"))
    if y_tol is None:
        y_tol = x_tol
    y_tol = abs(float(y_tol) or 0.0)
    y_err = None if y_axis is None else float(y_axis - y_target)
    y_abs_err = None if y_err is None else abs(float(y_err))
    y_within_tol = (not y_required) or bool(y_abs_err is not None and y_abs_err <= y_tol)

    d_entry = _entry("dist")
    d_stats = _stats("dist")
    dist_val = _num(_auto_diag_value_from_entry(d_entry)) if isinstance(d_entry, dict) else None
    d_required = bool(d_stats)
    d_target = _num(d_stats.get("target"))
    if d_target is None:
        d_target = 0.0
    d_tol = abs(_num(d_stats.get("tol")) or 0.0)
    dist_err = None if dist_val is None else abs(float(dist_val - d_target))
    d_direction = metric_direction("dist", step, process_rules=(getattr(world, "process_rules", None) or {}))
    dist_match = None
    if d_required and dist_val is not None:
        dist_match = _gate_entry_matches("dist", float(dist_val), d_stats, d_direction)
    dist_within_tol = (not d_required) or bool(dist_match)

    ok = bool(
        visible
        and x_within_tol
        and dist_within_tol
        and (y_within_tol if y_required else True)
    )
    return {
        "ok": ok,
        "visible": bool(visible),
        "x_axis": x_axis,
        "x_target": x_target,
        "x_tol": x_tol,
        "x_err": x_err,
        "x_abs_err": x_abs_err,
        "x_within_tol": x_within_tol,
        "x_required": x_required,
        "y_axis": y_axis,
        "y_target": y_target,
        "y_tol": y_tol,
        "y_err": y_err,
        "y_abs_err": y_abs_err,
        "y_within_tol": y_within_tol,
        "y_required": y_required,
        "dist": dist_val,
        "dist_target": d_target,
        "dist_tol": d_tol,
        "dist_err": dist_err,
        "dist_within_tol": dist_within_tol,
        "dist_required": d_required,
    }


def _maybe_apply_repeated_unchanged_speed_bump(
    *,
    cmd,
    score,
    delta_class,
    correction_type,
    error_signature,
    previous_cmd,
    state=None,
):
    cmd_key = str(cmd or "").strip().lower()
    prev_cmd_key = str(previous_cmd or "").strip().lower()
    reset_state = {
        "streak": 0,
        "cmd": None,
        "correction_type": None,
        "error_signature": None,
        "bump_count": 0,
        "last_boosted_score": None,
        "exhausted": False,
    }
    base_state = {
        "streak": 0,
        "cmd": None,
        "correction_type": None,
        "error_signature": None,
        "bump_count": 0,
        "last_boosted_score": None,
        "exhausted": False,
    }
    if isinstance(state, dict):
        base_state.update(
            {
                "streak": int(state.get("streak", 0) or 0),
                "cmd": state.get("cmd"),
                "correction_type": state.get("correction_type"),
                "error_signature": state.get("error_signature"),
                "bump_count": int(state.get("bump_count", 0) or 0),
                "last_boosted_score": state.get("last_boosted_score"),
                "exhausted": bool(state.get("exhausted", False)),
            }
        )

    ctype = str(correction_type or "").strip().lower() or "x_axis"
    should_track = (
        cmd_key in ("f", "b", "l", "r", "u", "d")
        and prev_cmd_key == cmd_key
        and str(delta_class or "").strip().lower() == "unchanged"
    )
    if not should_track:
        return score, reset_state, False

    same_signature = (
        str(base_state.get("cmd") or "").strip().lower() == cmd_key
        and str(base_state.get("correction_type") or "").strip().lower() == ctype
        and base_state.get("error_signature") == error_signature
    )
    if same_signature:
        streak = int(base_state.get("streak", 0) or 0) + 1
        bump_count = int(base_state.get("bump_count", 0) or 0)
        last_boosted_score = base_state.get("last_boosted_score")
    else:
        streak = 1
        bump_count = 0
        last_boosted_score = None
    next_state = {
        "streak": int(streak),
        "cmd": cmd_key,
        "correction_type": ctype,
        "error_signature": error_signature,
        "bump_count": int(bump_count),
        "last_boosted_score": last_boosted_score,
        "exhausted": False,
    }

    try:
        score_now = _cap_auto_speed_score(score)
    except Exception:
        score_now = None
    if score_now is None:
        return score, next_state, False

    score_now = min(int(score_now), int(REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP))
    if last_boosted_score is not None:
        try:
            last_boosted_score = int(round(float(last_boosted_score)))
        except (TypeError, ValueError):
            last_boosted_score = None
    if last_boosted_score is not None:
        score_now = max(
            int(score_now),
            min(int(last_boosted_score), int(REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP)),
        )

    if int(bump_count) >= int(REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT):
        next_state["last_boosted_score"] = int(score_now)
        next_state["exhausted"] = True
        return int(score_now), next_state, False

    score_boosted = min(
        int(score_now) + 1,
        int(REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP),
    )
    if int(score_boosted) <= int(score_now):
        next_state["last_boosted_score"] = int(score_now)
        return int(score_now), next_state, False

    bump_count = int(bump_count) + 1
    next_state["bump_count"] = int(bump_count)
    next_state["last_boosted_score"] = int(score_boosted)
    if int(bump_count) >= int(REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT):
        next_state["exhausted"] = True
    return int(score_boosted), next_state, True


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
    # Honor recorded demo turn scores for FIND_BRICK; caps apply to synthetic
    # auto-cycle move/check phases only.
    if phase_key in {"demo", "search"}:
        return int(score_val)
    cap = check_cap if phase_key == "check" else move_cap
    cap = max(int(min_score), int(cap))
    return min(int(score_val), int(cap))


def _auto_cycle_speed_label(score):
    if score is None:
        return None
    try:
        score_val = int(telemetry_robot_module.normalize_speed_score(score))
    except Exception:
        return None
    try:
        normal_score = int(telemetry_robot_module.normalize_speed_score(AUTO_CYCLE_CHECK_SCORE))
    except Exception:
        normal_score = int(telemetry_robot_module.SPEED_SCORE_MIN)
    try:
        standard_score = int(telemetry_robot_module.normalize_speed_score(AUTO_CYCLE_MOVE_SCORE))
    except Exception:
        standard_score = int(normal_score)
    try:
        fast_score = int(telemetry_robot_module.normalize_speed_score(AUTO_SPEED_SCORE_HARD_MAX))
    except Exception:
        fast_score = int(standard_score)
    if score_val >= int(fast_score):
        return "fast"
    if score_val >= int(standard_score):
        return "standard"
    if score_val >= int(normal_score):
        return "normal"
    return "normal"


def _visible_only_replay_speed_tier(world, step, *, step_cfg=None, cmd=None):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in ("f", "b", "l", "r", "u", "d"):
        return None
    process_rules = getattr(world, "process_rules", None) or {}
    if not next_module.success_gates_visible_only(process_rules, step):
        return None
    cfg = step_cfg if isinstance(step_cfg, dict) else {}
    if not cfg:
        obj_key = normalize_step_label(step)
        cfg = (process_rules.get(obj_key, {}) if isinstance(process_rules, dict) else {}) or {}
    tier_cfg = cfg.get("visible_only_speed_tiers") if isinstance(cfg, dict) else None
    enabled = False
    if isinstance(tier_cfg, bool):
        enabled = bool(tier_cfg)
        tier_cfg = {}
    elif isinstance(tier_cfg, dict):
        enabled = bool(tier_cfg.get("enabled", True))
    if not enabled:
        return None
    if not isinstance(tier_cfg, dict):
        tier_cfg = {}

    normal_score = _cap_auto_speed_score(tier_cfg.get("normal", AUTO_CYCLE_CHECK_SCORE))
    standard_score = _cap_auto_speed_score(tier_cfg.get("standard", AUTO_CYCLE_MOVE_SCORE))
    fast_score = _cap_auto_speed_score(tier_cfg.get("fast", AUTO_SPEED_SCORE_HARD_MAX))
    standard_score = max(int(normal_score), int(standard_score))
    fast_score = max(int(standard_score), int(fast_score))
    tiers = (
        ("normal", int(normal_score)),
        ("standard", int(standard_score)),
        ("fast", int(fast_score)),
    )
    try:
        cycle_idx = int(getattr(world, "_visible_speed_cycle", 0) or 0)
    except Exception:
        cycle_idx = 0
    cycle_idx = max(0, min(int(cycle_idx), len(tiers) - 1))
    tier_label, tier_score = tiers[cycle_idx]
    return {
        "label": str(tier_label),
        "score": int(tier_score),
    }


def _auto_uses_demo_speed(step):
    step_key = normalize_step_label(step)
    return step_key in AUTO_DEMO_SPEED_STEPS


def _height_intel_brick_height_mm():
    try:
        return float(getattr(telemetry_brick, "BRICK_HEIGHT_MM", 44.0))
    except Exception:
        return 44.0


def _height_intel_step_cfg(world, step):
    step_key = normalize_step_label(step)
    process_rules = getattr(world, "process_rules", None) or {}
    if not isinstance(process_rules, dict):
        return step_key, {}
    step_cfg = process_rules.get(step_key, {})
    if not isinstance(step_cfg, dict):
        return step_key, {}
    return step_key, step_cfg


def height_intel_values(world):
    values = {
        "brick_height_mm": float(_height_intel_brick_height_mm()),
        "wall_height_mm": None,
        "wall_height_bricks": None,
        "brick_supply_height_mm": None,
        "brick_supply_height_bricks": None,
    }
    if world is None:
        return values

    def _num(value):
        return _as_float(value, None)

    def _whole(value):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    wall_mm = _num(getattr(world, "wall_height_mm", None))
    wall_bricks = _whole(getattr(world, "wall_height_bricks", None))
    supply_mm = _num(getattr(world, "brick_supply_height_mm", None))
    supply_bricks = _whole(getattr(world, "brick_supply_height_bricks", None))

    if wall_bricks is None and wall_mm is not None:
        try:
            wall_bricks = telemetry_brick.height_mm_to_brick_count(wall_mm, minimum=0)
        except Exception:
            wall_bricks = None
    if wall_mm is None and wall_bricks is not None:
        try:
            wall_mm = telemetry_brick.brick_count_to_height_mm(wall_bricks)
        except Exception:
            wall_mm = None

    if supply_bricks is None and supply_mm is not None:
        try:
            supply_bricks = telemetry_brick.height_mm_to_brick_count(supply_mm, minimum=0)
        except Exception:
            supply_bricks = None
    if supply_mm is None and supply_bricks is not None:
        try:
            supply_mm = telemetry_brick.brick_count_to_height_mm(supply_bricks)
        except Exception:
            supply_mm = None

    values["wall_height_mm"] = wall_mm
    values["wall_height_bricks"] = wall_bricks
    values["brick_supply_height_mm"] = supply_mm
    values["brick_supply_height_bricks"] = supply_bricks
    return values


def _height_intel_value_text(name, bricks, mm):
    if bricks is None and mm is None:
        return f"{name}=unknown"
    if bricks is None:
        return f"{name}={float(mm):.1f}mm"
    if mm is None:
        return f"{name}={int(bricks)} bricks"
    return f"{name}={int(bricks)} bricks ({float(mm):.1f}mm)"


def height_intel_values_text(world):
    values = height_intel_values(world)
    return (
        f"{_height_intel_value_text('wall_height', values.get('wall_height_bricks'), values.get('wall_height_mm'))}, "
        f"{_height_intel_value_text('brick_supply_height', values.get('brick_supply_height_bricks'), values.get('brick_supply_height_mm'))}, "
        f"brick_height={float(values.get('brick_height_mm') or 44.0):.1f}mm"
    )


def height_intel_step_begin_line(world, step):
    step_key = normalize_step_label(step) or str(step or "UNKNOWN")
    return f"[HEIGHT INTEL] {step_key} start: {height_intel_values_text(world)}"


def apply_height_snapshot_from_step(world, step, *, log=True):
    step_key, step_cfg = _height_intel_step_cfg(world, step)
    cfg = step_cfg.get("height_snapshot") if isinstance(step_cfg, dict) else {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {"applied": False, "reason": "disabled", "step": step_key}

    target = str(cfg.get("target") or "").strip().lower()
    if target not in {"wall_height", "brick_supply_height"}:
        return {"applied": False, "reason": "invalid_target", "step": step_key}

    source = str(cfg.get("source") or "lift_height").strip().lower()
    measured_mm = None
    if source == "lift_height":
        measured_mm = _as_float(getattr(world, "lift_height", None), None)
    elif source in {"wall_height", "brick_supply_height"}:
        source_values = height_intel_values(world)
        measured_mm = _as_float(source_values.get(f"{source}_mm"), None)
    else:
        measured_mm = _as_float(getattr(world, source, None), None)

    if measured_mm is None:
        return {"applied": False, "reason": "missing_measurement", "step": step_key}

    try:
        min_bricks = max(0, int(_as_int(cfg.get("min_bricks"), 1)))
    except Exception:
        min_bricks = 1
    max_bricks = _as_int(cfg.get("max_bricks"), None)
    if max_bricks is not None:
        max_bricks = max(int(min_bricks), int(max_bricks))
    try:
        bricks = telemetry_brick.height_mm_to_brick_count(
            measured_mm,
            minimum=int(min_bricks),
            maximum=max_bricks,
        )
    except Exception:
        bricks = None
    if bricks is None:
        return {"applied": False, "reason": "conversion_failed", "step": step_key}
    try:
        quantized_mm = telemetry_brick.brick_count_to_height_mm(bricks)
    except Exception:
        quantized_mm = float(bricks) * float(_height_intel_brick_height_mm())

    if target == "wall_height":
        setattr(world, "wall_height_bricks", int(bricks))
        setattr(world, "wall_height_mm", float(quantized_mm))
    else:
        setattr(world, "brick_supply_height_bricks", int(bricks))
        setattr(world, "brick_supply_height_mm", float(quantized_mm))

    if log:
        print(
            format_headline(
                (
                    f"[HEIGHT INTEL] {step_key} snapshot -> {target}={int(bricks)} bricks "
                    f"({float(quantized_mm):.1f}mm) from {source}={float(measured_mm):.1f}mm; "
                    f"{height_intel_values_text(world)}"
                ),
                COLOR_WHITE,
            )
        )

    return {
        "applied": True,
        "step": step_key,
        "target": str(target),
        "source": str(source),
        "source_mm": float(measured_mm),
        "bricks": int(bricks),
        "height_mm": float(quantized_mm),
    }


def _apply_height_intel_replay_override(world, step, cmd, score, *, phase=None, log=True):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in ("f", "b", "l", "r", "u", "d"):
        return cmd, score, None, False
    step_key, step_cfg = _height_intel_step_cfg(world, step)
    cfg = step_cfg.get("height_intelligence") if isinstance(step_cfg, dict) else {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return cmd, score, None, False

    apply_when_visible_false = bool(cfg.get("apply_when_visible_false", True))
    brick = getattr(world, "brick", None) or {}
    if apply_when_visible_false and bool(brick.get("visible")):
        return cmd, score, None, False

    allowed_cmds = cfg.get("apply_on_cmds")
    if not isinstance(allowed_cmds, (list, tuple, set)) or not allowed_cmds:
        allowed_cmds = ("f", "b", "l", "r")
    allowed = {str(item).strip().lower() for item in allowed_cmds if item is not None}
    if cmd_key not in allowed:
        return cmd, score, None, False

    values = height_intel_values(world)
    source_key = str(cfg.get("source") or "").strip().lower()
    source_bricks = None
    if source_key == "wall_height":
        source_bricks = values.get("wall_height_bricks")
    elif source_key == "brick_supply_height":
        source_bricks = values.get("brick_supply_height_bricks")
    if source_bricks is None:
        return cmd, score, None, False

    low_cmd = str(cfg.get("low_cmd") or "d").strip().lower()
    high_cmd = str(cfg.get("high_cmd") or "u").strip().lower()
    if low_cmd not in ("u", "d"):
        low_cmd = "d"
    if high_cmd not in ("u", "d"):
        high_cmd = "u"

    low_max = _as_int(cfg.get("low_bricks_max"), 2)
    high_min = _as_int(cfg.get("high_bricks_min"), 4)
    chosen_cmd = None
    reason_key = None
    if low_max is not None and int(source_bricks) <= int(low_max):
        chosen_cmd = str(low_cmd)
        reason_key = "low"
    elif high_min is not None and int(source_bricks) >= int(high_min):
        chosen_cmd = str(high_cmd)
        reason_key = "high"
    if chosen_cmd is None:
        return cmd, score, None, False

    cooldown_s = max(0.0, float(_as_float(cfg.get("cooldown_s"), 0.6)))
    now_s = float(time.time())
    cooldown_by_step = getattr(world, "_height_intel_last_override_ts", None)
    if not isinstance(cooldown_by_step, dict):
        cooldown_by_step = {}
    last_ts = _as_float(cooldown_by_step.get(step_key), None)
    if last_ts is not None and cooldown_s > 0.0 and (now_s - float(last_ts)) < float(cooldown_s):
        return cmd, score, None, False
    cooldown_by_step[step_key] = float(now_s)
    setattr(world, "_height_intel_last_override_ts", cooldown_by_step)

    score_out = _cap_auto_speed_score(cfg.get("score", telemetry_robot_module.SPEED_SCORE_MIN))
    context_note = f"height intel ({source_key} {reason_key})"
    if log:
        phase_key = str(phase or "replay").strip().lower() or "replay"
        print(
            format_headline(
                (
                    f"[HEIGHT INTEL] {step_key} {phase_key}: {source_key}={int(source_bricks)} bricks -> "
                    f"{str(chosen_cmd).upper()} {int(score_out)}%; {height_intel_values_text(world)}"
                ),
                COLOR_MAGENTA_BRIGHT,
            )
        )
    return str(chosen_cmd), int(score_out), context_note, True


def send_robot_command_pwm(
    robot,
    world,
    step,
    cmd,
    power,
    pwm,
    duration_ms,
    speed_score=None,
    auto_mode=False,
    turn_intensity_requested=None,
    turn_intensity_effective=None,
    half_first_turn_pulse=True,
    ease_in_out_enabled=None,
):
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

    duration_used_ms = _duration_used_ms_for_cmd(
        robot,
        cmd,
        duration_ms,
        auto_mode=auto_mode,
        half_first_turn_pulse=half_first_turn_pulse,
    )
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

    def _apply_precision_score_fix(score_eff, score_model_local):
        if score_model_local is None or score_eff is None:
            return score_eff
        try:
            effective = int(round(float(score_eff)))
            intended = int(round(float(score_model_local)))
            if effective > intended and effective - intended == 1 and intended < 10:
                return score_model_local
        except (TypeError, ValueError):
            return score_eff
        return score_eff

    def _score_effective_for_cmd(cmd_eff, power_eff, score_model_local):
        _, score_eff = telemetry_robot_module.quantize_speed(cmd_eff, speed=power_eff)
        if score_eff is None:
            score_eff = score_model_local
        return _apply_precision_score_fix(score_eff, score_model_local)

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

    ease_profile = None
    ease_note = None
    # Easing can run in two modes:
    # 1) Queued transport/firmware: send timed segments back-to-back.
    # 2) Manual non-queued transport: block between segment writes to preserve
    #    sequence fidelity for hotkey presses.
    # Auto non-queued remains disabled to avoid adding long sleeps into the
    # autonomous control loop.
    supports_timed_command_queue = bool(
        getattr(robot, "supports_timed_command_queue", False)
        or getattr(robot, "supports_timed_segment_queue", False)
    )
    if ease_in_out_enabled is None:
        ease_allowed = True
    else:
        ease_allowed = bool(ease_in_out_enabled)
    allow_sequenced_ease = bool(ease_allowed and (supports_timed_command_queue or not bool(auto_mode)))
    if allow_sequenced_ease:
        ease_builder = getattr(telemetry_robot_module, "drive_ease_in_out_segments", None)
        if not callable(ease_builder):
            ease_builder = getattr(telemetry_robot_module, "drive_anti_alias_segments", None)
        if callable(ease_builder):
            try:
                ease_profile = ease_builder(
                    cmd,
                    speed_score=requested_score if requested_score is not None else speed_score,
                    power=power_val,
                    pwm=pwm_val,
                    duration_ms=duration_used_ms,
                )
            except Exception:
                ease_profile = None

    if isinstance(ease_profile, dict):
        seg_defs = ease_profile.get("segments")
        if isinstance(seg_defs, list) and seg_defs:
            valid_seg_defs = [seg for seg in seg_defs if isinstance(seg, dict)]
            sent_segments = []
            wire_parts = []
            total_model_ms = 0
            total_used_ms = 0
            final_cmd_sent = cmd_sent
            peak_pwm = int(pwm_val)
            peak_power = float(power_val)
            for seg_idx, seg in enumerate(valid_seg_defs):
                try:
                    seg_pwm = int(round(float(seg.get("pwm") or 0)))
                except (TypeError, ValueError):
                    seg_pwm = 0
                seg_pwm = max(0, min(pwm_cap, seg_pwm))
                try:
                    seg_power = float(seg.get("power"))
                except (TypeError, ValueError):
                    seg_power = telemetry_robot_module.pwm_to_power(seg_pwm) or 0.0
                seg_power = max(0.0, min(1.0, float(seg_power or 0.0)))
                try:
                    seg_duration_model_ms = max(1, int(round(float(seg.get("duration_ms") or 0))))
                except (TypeError, ValueError):
                    seg_duration_model_ms = max(1, int(duration_used_ms))
                seg_score_model = seg.get("score_model")
                try:
                    if seg_score_model is not None:
                        seg_score_model = telemetry_robot_module.normalize_speed_score(seg_score_model)
                except Exception:
                    seg_score_model = None
                seg_cmd_model = str(seg.get("cmd") or cmd).strip().lower() or str(cmd)

                seg_send_result = None
                if hasattr(robot, "send_command_pwm"):
                    seg_send_result = robot.send_command_pwm(
                        seg_cmd_model,
                        seg_pwm,
                        duration_ms=seg_duration_model_ms,
                    )
                else:
                    seg_send_result = robot.send_command(
                        seg_cmd_model,
                        seg_power,
                        duration_ms=seg_duration_model_ms,
                    )

                seg_cmd_sent = cmd_sent
                seg_pwm_actual = int(seg_pwm)
                seg_duration_used_ms = int(seg_duration_model_ms)
                if isinstance(seg_send_result, dict):
                    seg_cmd_sent = seg_send_result.get("cmd_sent") or seg_cmd_sent
                    try:
                        seg_pwm_actual = int(round(float(seg_send_result.get("pwm"))))
                    except (TypeError, ValueError):
                        pass
                    try:
                        seg_duration_used_ms = int(round(float(seg_send_result.get("duration_ms"))))
                    except (TypeError, ValueError):
                        pass
                seg_power_from_pwm = telemetry_robot_module.pwm_to_power(seg_pwm_actual)
                if seg_power_from_pwm is not None:
                    seg_power = max(0.0, min(1.0, float(seg_power_from_pwm)))
                seg_score_effective = _score_effective_for_cmd(seg_cmd_sent, seg_power, seg_score_model)

                total_model_ms += int(seg_duration_model_ms)
                total_used_ms += int(seg_duration_used_ms)
                peak_pwm = max(int(peak_pwm), int(seg_pwm_actual))
                peak_power = max(float(peak_power), float(seg_power))
                final_cmd_sent = seg_cmd_sent or final_cmd_sent
                wire_parts.append(f"{seg_cmd_sent} {int(seg_pwm_actual)} {int(seg_duration_used_ms)}")
                sent_segments.append(
                    {
                        "cmd_model": seg_cmd_model,
                        "cmd_sent": seg_cmd_sent,
                        "power": float(seg_power),
                        "pwm": int(seg_pwm_actual),
                        "duration_model_ms": int(seg_duration_model_ms),
                        "duration_ms": int(seg_duration_used_ms),
                        "score_model": seg_score_model,
                        "score_effective": seg_score_effective,
                    }
                )
                if not supports_timed_command_queue and seg_idx < (len(valid_seg_defs) - 1):
                    try:
                        wait_s = max(0.0, float(seg_duration_used_ms) / 1000.0)
                    except (TypeError, ValueError):
                        wait_s = 0.0
                    if wait_s > 0.0:
                        time.sleep(wait_s)

            if sent_segments:
                cmd_sent = final_cmd_sent
                pwm_val = int(peak_pwm)
                power_val = float(peak_power)
                duration_model_ms = int(total_model_ms or duration_used_ms)
                duration_used_ms = int(total_used_ms or duration_used_ms)
                top_score_effective = _score_effective_for_cmd(cmd_sent, power_val, speed_score)
                try:
                    ramp_ms = int(round(float(ease_profile.get("ramp_ms") or 0)))
                except (TypeError, ValueError):
                    ramp_ms = 0
                try:
                    ramp_ms_eff = int(round(float(ease_profile.get("ramp_ms_effective") or 0)))
                except (TypeError, ValueError):
                    ramp_ms_eff = 0
                ease_note = (
                    f"EASE({len(sent_segments)} seg, in={max(0, int(ramp_ms))}ms, "
                    f"out={max(0, int(ramp_ms))}ms, eff={max(0, int(ramp_ms_eff))}ms)"
                )

                if world is not None:
                    wire_text = " | ".join([str(p) for p in wire_parts if p])
                    if len(wire_text) > 180:
                        wire_text = wire_text[:177] + "..."
                    world._last_action_wire = str(wire_text)
                    world._last_action_wire_step = normalize_step_label(step)
                    world._last_action_wire_time = time.time()

                record_action_display(
                    world,
                    step,
                    cmd,
                    power_val,
                    speed_score=top_score_effective,
                    cmd_sent=cmd_sent,
                    pwm=pwm_val,
                    duration_ms=duration_used_ms,
                    ease_note=ease_note,
                    score_model=speed_score,
                    anti_alias_note=ease_note,
                )
                return {
                    "cmd_sent": cmd_sent,
                    "power": power_val,
                    "pwm": pwm_val,
                    "duration_model_ms": int(duration_model_ms),
                    "duration_ms": int(duration_used_ms),
                    "score_model": speed_score,
                    "score_effective": top_score_effective,
                    "turn_intensity_requested": turn_intensity_requested,
                    "turn_intensity_effective": turn_intensity_effective,
                    "segments": sent_segments,
                    "ease_in_out_note": ease_note,
                    "anti_alias_note": ease_note,
                    "ease_in_out": {
                        "applied": True,
                        "kind": "ease_in_out",
                        "ramp_ms": ease_profile.get("ramp_ms"),
                        "ramp_ms_effective": ease_profile.get("ramp_ms_effective"),
                        "ramp_steps": ease_profile.get("ramp_steps"),
                        "slice_ms": ease_profile.get("slice_ms"),
                        "target_score": ease_profile.get("target_score"),
                        "threshold_score": ease_profile.get("threshold_score"),
                    },
                    "anti_aliasing": {
                        "applied": True,
                        "kind": "ease_in_out",
                        "ramp_ms": ease_profile.get("ramp_ms"),
                        "ramp_ms_effective": ease_profile.get("ramp_ms_effective"),
                        "ramp_steps": ease_profile.get("ramp_steps"),
                        "slice_ms": ease_profile.get("slice_ms"),
                        "target_score": ease_profile.get("target_score"),
                        "threshold_score": ease_profile.get("threshold_score"),
                    },
                }

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
    score_effective = _score_effective_for_cmd(cmd_sent, power_val, speed_score)
    record_action_display(
        world,
        step,
        cmd,
        power_val,
        speed_score=score_effective,
        cmd_sent=cmd_sent,
        pwm=pwm_val,
        duration_ms=duration_used_ms,
        ease_note=ease_note,
        score_model=speed_score,
        anti_alias_note=ease_note,
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
        "turn_intensity_requested": turn_intensity_requested,
        "turn_intensity_effective": turn_intensity_effective,
        "ease_in_out_note": ease_note,
        "anti_alias_note": ease_note,
    }


def send_robot_command(
    robot,
    world,
    step,
    cmd,
    speed,
    speed_score=None,
    auto_mode=False,
    turn_intensity=None,
    duration_override_ms=None,
    half_first_turn_pulse=True,
    ease_in_out_enabled=None,
):
    if robot is None or cmd is None:
        return None

    turn_intensity_requested = None
    turn_intensity_effective = None

    if cmd in ("l", "r") and turn_intensity is not None:
        try:
            turn_intensity_requested = float(turn_intensity)
        except (TypeError, ValueError):
            turn_intensity_requested = None
        if turn_intensity_requested is not None:
            power, pwm, score_used, duration_ms, turn_intensity_effective = telemetry_robot_module.speed_power_pwm_for_turn_intensity(
                cmd,
                turn_intensity_requested,
            )
            if auto_mode:
                score_used = telemetry_robot_module.normalize_speed_score(score_used)
                score_used = _cap_auto_speed_score(score_used)
                power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score_used)
                turn_intensity_effective = None
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
                turn_intensity_requested=turn_intensity_requested,
                turn_intensity_effective=turn_intensity_effective,
                half_first_turn_pulse=half_first_turn_pulse,
                ease_in_out_enabled=ease_in_out_enabled,
            )

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
    try:
        duration_override_val = int(round(float(duration_override_ms)))
    except (TypeError, ValueError):
        duration_override_val = None
    if duration_override_val is not None and duration_override_val > 0:
        duration_ms = int(duration_override_val)

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
        turn_intensity_requested=turn_intensity_requested,
        turn_intensity_effective=turn_intensity_effective,
        half_first_turn_pulse=half_first_turn_pulse,
        ease_in_out_enabled=ease_in_out_enabled,
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
        non_alignment_metrics = {"visible", "confidence"}
        if set(metrics).issubset(non_alignment_metrics):
            return False
        return True
    success_gates = rules.get("success_gates") or {}
    gate_metrics = {metric for metric in success_gates.keys() if metric}
    non_alignment_metrics = {"visible", "confidence"}
    if gate_metrics and gate_metrics.issubset(non_alignment_metrics):
        return False
    return bool(gate_metrics)


def step_is_nominal_only(step, process_rules):
    rules_map = process_rules or {}
    obj_key = normalize_step_label(step)
    rules = rules_map.get(obj_key, {})
    if not isinstance(rules, dict):
        rules = {}
    if not rules and step is not None:
        raw_key = str(step).strip().upper()
        if raw_key and raw_key != obj_key:
            fallback = rules_map.get(raw_key, {})
            if isinstance(fallback, dict):
                rules = fallback
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
    brick = getattr(world, "brick", {}) or {}
    act_plan = next_module.select_alignment_next_act(
        process_rules=world.process_rules or {},
        learned_rules=world.learned_rules or {},
        step=step,
        x_axis_mm=brick.get("x_axis", brick.get("offset_x")),
        y_axis_mm=brick.get("y_axis", brick.get("offset_y")),
        dist_mm=brick.get("dist"),
        visible=bool(brick.get("visible")),
        angle_deg=brick.get("angle", 0.0),
        duration_s=CONTROL_DT,
    )
    cmd = act_plan.get("cmd")
    speed = act_plan.get("speed") or 0.0
    reason = act_plan.get("reason") or "align"
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


def _apply_post_action_observe_delay(world):
    # Global observe-after-act guardrail: require a fixed settle delay after any
    # sent motion command before we sample fresh vision again.
    if world is None:
        return
    cmd = str(getattr(world, "_last_action_cmd", "") or "").strip().lower()
    if cmd not in {"f", "b", "l", "r", "u", "d"}:
        return
    sent_time = _as_float(getattr(world, "_last_action_time", None), None)
    if sent_time is None:
        return
    delay_s = _as_float(
        getattr(world, "_post_action_observe_delay_s", POST_ACTION_OBSERVE_DELAY_S),
        POST_ACTION_OBSERVE_DELAY_S,
    )
    delay_s = max(0.0, float(delay_s))
    if delay_s <= 0.0:
        return
    elapsed = float(time.time()) - float(sent_time)
    remaining = float(delay_s) - float(elapsed)
    if remaining > 0.0:
        time.sleep(float(remaining))




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


def print_gatecheck_progress_line(world, step=None, *, log=True, force=False):
    if not log or world is None:
        return
    status = getattr(world, "_gatecheck_status", None)
    if not isinstance(status, dict):
        return

    step_key = normalize_step_label(step) or normalize_step_label(status.get("step")) or "UNKNOWN"
    phase = str(status.get("phase") or "run").strip().lower()
    mode = str(status.get("mode") or "traditional").strip().lower()
    checks = int(status.get("checks", 0) or 0)
    truth_ok = bool(status.get("truth_ok", False))

    if mode == "lite":
        lite_collected = max(0, int(status.get("lite_collected", 0) or 0))
        lite_required = max(1, int(status.get("lite_required", 1) or 1))
        state = "pass" if truth_ok else "wait"
        text = (
            f"[GATECHECK] {step_key} {phase}: "
            f"lite {lite_collected}/{lite_required} {state} "
            f"(checks={checks})"
        )
        # Lite checks can run frequently; dedupe by progress state and emit a
        # periodic heartbeat bucket so operator logs still show forward motion.
        heartbeat_bucket = max(0, int(checks) // 20)
        sig = (
            step_key,
            phase,
            mode,
            lite_collected,
            lite_required,
            truth_ok,
            heartbeat_bucket,
        )
    else:
        lite_required = int(status.get("lite_required", 0) or 0)
        lite_passed = getattr(world, "_gatecheck_lite_passed", None)
        if not force and lite_required > 0 and lite_passed is False:
            return
        streak = int(status.get("streak", 0) or 0)
        need = max(1, int(status.get("need", 1) or 1))
        window_pass = int(status.get("window_pass", 0) or 0)
        window_total = max(1, int(status.get("window_total", 1) or 1))
        state = "pass" if truth_ok else "wait"
        text = (
            f"[GATECHECK] {step_key} {phase}: "
            f"consec {streak}/{need}, majority {window_pass}/{window_total} {state} "
            f"(checks={checks})"
        )
        sig = (
            step_key,
            phase,
            mode,
            checks,
            streak,
            need,
            window_pass,
            window_total,
            truth_ok,
        )

    last_sig = getattr(world, "_last_gatecheck_progress_sig", None)
    if not force and sig == last_sig:
        return
    world._last_gatecheck_progress_sig = sig
    print(format_headline(text, COLOR_WHITE))
    if mode == "lite":
        detail = _result_lite_gate_detail(world, step_key)
        if isinstance(detail, dict):
            detail_line = detail.get("colored") or detail.get("plain")
            if detail_line:
                print(format_headline(f"[GATECHECK-DETAIL] {step_key} {phase}: {detail_line}", COLOR_WHITE))


def print_gatecheck_entry_proof_line(world, step, *, phase=None, success_ok=None, log=True):
    if not log or world is None:
        return
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    phase_key = str(phase or "run").strip().lower() or "run"
    lite_required = int(getattr(world, "_gatecheck_lite_required", 0) or 0)
    lite_collected = int(getattr(world, "_gatecheck_lite_collected", 0) or 0)
    lite_text = f"lite {lite_collected}/{lite_required}" if lite_required > 0 else "lite n/a"
    sample_pass = bool(success_ok)
    sample_text = f"{COLOR_GREEN}PASS{COLOR_RESET}" if sample_pass else f"{COLOR_RED}FAIL{COLOR_RESET}"
    entries = _success_gate_entries(world, step)
    gates_text = _auto_diag_entries_snapshot_colored(entries, step, include_requirement=True)
    print(
        f"{COLOR_WHITE}[GATECHECK-ENTER] {step_key} {phase_key}: "
        f"{lite_text}; effective sample={sample_text}; gates: {COLOR_RESET}{gates_text}"
    )


def _gatecheck_failure_detail(world, step):
    rows = _success_gate_state_rows(world, step)
    fail_parts = []
    for row in rows:
        if str(row.get("status") or "") != "FAIL":
            continue
        fail_parts.append(
            f"{row.get('metric')} state={row.get('value_text')} gate={row.get('gate_text')}"
        )
    if fail_parts:
        return "; ".join(fail_parts)
    status = getattr(world, "_gatecheck_status", None)
    if isinstance(status, dict):
        streak = int(status.get("streak", 0) or 0)
        need = max(1, int(status.get("need", 1) or 1))
        window_pass = int(status.get("window_pass", 0) or 0)
        window_total = max(1, int(status.get("window_total", 1) or 1))
        return (
            "effective sample failed"
            f" (consec {streak}/{need}, majority {window_pass}/{window_total})"
        )
    return "effective sample failed"


def _log_gatecheck_failure(world, step, phase, *, log=True):
    if not log:
        return
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    phase_key = str(phase or "run").strip().lower() or "run"
    detail = _gatecheck_failure_detail(world, step)
    print(format_headline(f"[GATECHECK-FAIL] {step_key} {phase_key}: {detail}", COLOR_RED))
    lite_detail = _result_lite_gate_detail(world, step)
    if isinstance(lite_detail, dict):
        lite_line = lite_detail.get("colored") or lite_detail.get("plain")
        if lite_line:
            print(format_headline(f"[GATECHECK-DETAIL] {step_key} {phase_key}: {lite_line}", COLOR_WHITE))


def _gatecheck_next_action_text(cmd, speed_score=None, reason=None):
    cmd_key = str(cmd or "").strip().lower()
    reason_text = str(reason or "").strip()
    if cmd_key:
        act_text = action_display_text(cmd_key, speed_score)
        if reason_text:
            return f"continuing with next act {act_text} ({reason_text})"
        return f"continuing with next act {act_text}"
    if reason_text:
        return f"continuing with adjustment acts ({reason_text})"
    return "continuing with adjustment acts"


def _align_brick_lite_fail_fallback_action(world, step):
    step_key_local = normalize_step_label(step)
    if step_key_local not in {
        "ALIGN_BRICK",
        "POSITION_BRICK",
        "BRICK_LOCK",
        "BRICK_LOCK_WALL",
    }:
        return None
    process_rules = getattr(world, "process_rules", None) or {}
    step_cfg = process_rules.get(step_key_local, {}) if isinstance(process_rules, dict) else {}
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    if not isinstance(success_gates, dict) or not success_gates:
        return None

    measurement, _lite_meta = _lite_gate_measurement_for_step(world, step_key_local)
    source = measurement if isinstance(measurement, dict) else (getattr(world, "brick", None) or {})

    best = None
    for metric, stats in success_gates.items():
        metric_key = str(metric or "").strip()
        if metric_key not in ("dist", "xAxis_offset_abs"):
            continue
        value = gate_utils.metric_value_from_measurement(source, metric_key)
        if value is None:
            continue
        direction = metric_direction(metric_key, step_key_local, process_rules=process_rules)
        if _gate_entry_matches(metric_key, value, stats, direction):
            continue
        distance = _gate_distance_to_success(metric_key, value, stats, direction)
        try:
            distance_val = float(distance if distance is not None else 0.0)
        except (TypeError, ValueError):
            distance_val = 0.0
        try:
            value_num = float(value)
        except (TypeError, ValueError):
            continue
        target = stats.get("target") if isinstance(stats, dict) else None
        try:
            target_num = float(target) if target is not None else 0.0
        except (TypeError, ValueError):
            target_num = 0.0
        candidate = {
            "metric": metric_key,
            "distance": max(0.0, distance_val),
            "value": value_num,
            "target": target_num,
        }
        if best is None or candidate["distance"] > float(best.get("distance", 0.0)):
            best = candidate

    if not isinstance(best, dict):
        return None

    metric_key = str(best.get("metric") or "")
    value_num = float(best.get("value", 0.0))
    target_num = float(best.get("target", 0.0))
    if metric_key == "dist":
        cmd_local = "b" if value_num < target_num else "f"
        score_local = 5
        reason_local = "lite_fail_fallback_dist"
    else:
        cmd_local = "r" if value_num < target_num else "l"
        score_local = 5
        reason_local = "lite_fail_fallback_x_axis"
    speed_local = telemetry_robot_module.manual_speed_for_cmd(cmd_local, score_local)
    return {
        "cmd": cmd_local,
        "speed": float(speed_local or 0.0),
        "score": int(score_local),
        "reason": reason_local,
    }


def _log_gatecheck_next_action(step, phase, cmd, speed_score=None, reason=None, *, log=True):
    if not log:
        return
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    phase_key = str(phase or "run").strip().lower() or "run"
    text = _gatecheck_next_action_text(cmd, speed_score=speed_score, reason=reason)
    color = COLOR_WHITE if str(cmd or "").strip() else COLOR_YELLOW
    print(format_headline(f"[GATECHECK-NEXT] {step_key} {phase_key}: {text}", color))


def _exception_lift_estimate_text(world, previous_lift_mm=None):
    current_lift_mm = _as_float(getattr(world, "lift_height", None), None)
    source = str(getattr(world, "lift_height_source", "unknown") or "unknown").strip()
    quality = _as_float(getattr(world, "lift_height_quality", None), None)
    if current_lift_mm is None:
        lift_text = "lift=n/a"
    else:
        lift_text = f"lift={float(current_lift_mm):.1f}mm"
    if quality is None:
        lift_text = f"{lift_text} (source={source})"
    else:
        lift_text = f"{lift_text} (source={source}, q={float(quality):.2f})"
    if current_lift_mm is not None and previous_lift_mm is not None:
        lift_text = f"{lift_text}, Δ={float(current_lift_mm) - float(previous_lift_mm):+0.1f}mm"
    return current_lift_mm, lift_text


def _run_start_ground_reset_exception(
    world,
    vision,
    step,
    *,
    robot=None,
    observer=None,
    confirm_callback=None,
    align_silent=False,
):
    step_key = normalize_step_label(step)
    step_cfg = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        step_cfg = world.process_rules.get(step_key, {}) or {}
    cfg = step_cfg.get("start_ground_reset_exception") if isinstance(step_cfg, dict) else {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {"enabled": False, "success": True, "reason": "disabled"}

    cmd = str(cfg.get("command") or "d").strip().lower()
    if cmd not in ("d", "u"):
        cmd = "d"
    score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(cfg.get("score"), 100))),
    )
    max_acts = max(1, int(_as_int(cfg.get("max_acts"), _as_int(cfg.get("acts"), 12))))
    min_acts = max(0, int(_as_int(cfg.get("min_acts"), 0)))
    min_acts = min(int(min_acts), int(max_acts))
    goal_lift_mm = _as_float(cfg.get("ground_level_lift_mm_goal"), 50.0)
    if goal_lift_mm is None:
        goal_lift_mm = 50.0
    require_tiny_roi_floor = bool(cfg.get("require_tiny_roi_floor", False))
    fail_on_unconfirmed = bool(cfg.get("fail_on_unconfirmed", False))
    operator_reminder = str(cfg.get("operator_log_reminder") or "").strip()

    speed = telemetry_robot_module.manual_speed_for_cmd(cmd, score)
    prev_lift_mm = _as_float(getattr(world, "lift_height", None), None)
    acts_sent = 0

    if not align_silent:
        if operator_reminder:
            print(format_headline("[EXCEPTION]", COLOR_MAGENTA_BRIGHT, f" [{step_key}] {operator_reminder}"))
        tiny_req_text = "required" if require_tiny_roi_floor else "preferred"
        print(
            format_headline(
                "[EXCEPTION]",
                COLOR_MAGENTA_BRIGHT,
                (
                    f" [{step_key}] Start ground reset: {str(cmd).upper()} {int(score)}%, "
                    f"goal=lift<={float(goal_lift_mm):.1f}mm (tiny_roi_floor {tiny_req_text}), "
                    f"min={int(min_acts)}, max={int(max_acts)}."
                ),
            )
        )

    def _status_words():
        lift_mm, lift_text = _exception_lift_estimate_text(world, prev_lift_mm)
        source_key = str(getattr(world, "lift_height_source", "") or "").strip().lower()
        source_ok = (not require_tiny_roi_floor) or source_key.startswith("tiny_roi_floor")
        lift_ok = bool(lift_mm is not None and float(lift_mm) <= float(goal_lift_mm))
        gate_ok = bool(source_ok and lift_ok and int(acts_sent) >= int(min_acts))
        src_word = f"{COLOR_GREEN}PASS{COLOR_RESET}" if bool(source_ok) else f"{COLOR_RED}FAIL{COLOR_RESET}"
        gate_word = f"{COLOR_GREEN}PASS{COLOR_RESET}" if bool(gate_ok) else f"{COLOR_RED}FAIL{COLOR_RESET}"
        return {
            "lift_mm": lift_mm,
            "lift_text": lift_text,
            "source_ok": bool(source_ok),
            "lift_ok": bool(lift_ok),
            "gate_ok": bool(gate_ok),
            "src_word": src_word,
            "gate_word": gate_word,
        }

    update_world_from_vision(world, vision, log=not align_silent)
    if observer:
        observer("frame", world, vision, None, None, None)
    status = _status_words()
    if bool(status.get("gate_ok")):
        if robot:
            robot.stop()
        if not align_silent:
            print(
                format_headline(
                    "[EXCEPTION]",
                    COLOR_MAGENTA_BRIGHT,
                    (
                        f" [{step_key}] Start ground reset complete: "
                        f"source={status.get('src_word')}, gate={status.get('gate_word')}, "
                        f"{status.get('lift_text')}."
                    ),
                )
            )
        return {"enabled": True, "success": True, "reason": "ground reset pass", "acts": int(acts_sent)}

    for idx in range(1, int(max_acts) + 1):
        if confirm_callback and not confirm_callback(world, vision):
            return {"enabled": True, "success": False, "reason": "confirm cancelled", "acts": int(acts_sent)}
        if observer:
            observer("analysis", world, vision, cmd, speed, "start_ground_reset_exception")
        send_robot_command(
            robot,
            world,
            step_key,
            cmd,
            speed,
            speed_score=score,
            auto_mode=True,
        )
        acts_sent += 1
        post_act_analysis(world, vision, step=step_key, log=not align_silent, include_pause=False)
        update_world_from_vision(world, vision, log=not align_silent)
        if observer:
            observer("frame", world, vision, None, None, None)
            observer("action", world, vision, cmd, speed, "start_ground_reset_exception")
        status = _status_words()
        if not align_silent:
            print(
                format_headline(
                    "[EXCEPTION]",
                    COLOR_MAGENTA_BRIGHT,
                    (
                        f" [{step_key}] Start ground reset pulse {int(idx)}/{int(max_acts)}: "
                        f"{str(cmd).upper()} {int(score)}%, source={status.get('src_word')}, "
                        f"{status.get('lift_text')}, gate={status.get('gate_word')}."
                    ),
                )
            )
        if bool(status.get("lift_mm") is not None):
            prev_lift_mm = _as_float(status.get("lift_mm"), prev_lift_mm)
        if bool(status.get("gate_ok")):
            if robot:
                robot.stop()
            if not align_silent:
                print(
                    format_headline(
                        "[EXCEPTION]",
                        COLOR_MAGENTA_BRIGHT,
                        (
                            f" [{step_key}] Start ground reset complete: "
                            f"source={status.get('src_word')}, gate={status.get('gate_word')}, "
                            f"{status.get('lift_text')}."
                        ),
                    )
                )
            return {"enabled": True, "success": True, "reason": "ground reset pass", "acts": int(acts_sent)}
        time.sleep(CONTROL_DT)

    if robot:
        robot.stop()
    if not align_silent:
        fail_word = f"{COLOR_RED}FAIL{COLOR_RESET}"
        print(
            format_headline(
                "[EXCEPTION]",
                COLOR_MAGENTA_BRIGHT,
                (
                    f" [{step_key}] Start ground reset complete: gate={fail_word}. "
                    f"acts={int(acts_sent)}/{int(max_acts)}."
                ),
            )
        )
    if fail_on_unconfirmed:
        return {"enabled": True, "success": False, "reason": "ground reset unconfirmed", "acts": int(acts_sent)}
    return {"enabled": True, "success": True, "reason": "ground reset unconfirmed (continuing)", "acts": int(acts_sent)}


def _run_ground_up_level2_exception(
    world,
    vision,
    step,
    *,
    robot=None,
    observer=None,
    confirm_callback=None,
    align_silent=False,
):
    step_key = normalize_step_label(step)
    if not _step_allows_topmost_ground_up_level2(step_key):
        return {"enabled": False, "handled": False}
    step_cfg = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        step_cfg = world.process_rules.get(step_key, {}) or {}
    if not isinstance(step_cfg, dict):
        return {"enabled": False, "handled": False}

    topmost_cfg = step_cfg.get("topmost_crosshair_exception")
    if not isinstance(topmost_cfg, dict):
        return {"enabled": False, "handled": False}
    if not bool(topmost_cfg.get("enabled")):
        return {"enabled": False, "handled": False}
    if not bool(topmost_cfg.get("ground_up_level2_enabled")):
        return {"enabled": False, "handled": False}

    success_gates_cfg = step_cfg.get("success_gates")
    if not isinstance(success_gates_cfg, dict):
        success_gates_cfg = {}
    above_gate_cfg = success_gates_cfg.get("brick_above")
    if above_gate_cfg is None:
        above_gate_cfg = success_gates_cfg.get("brickAbove")
    below_gate_cfg = success_gates_cfg.get("brick_below")
    if below_gate_cfg is None:
        below_gate_cfg = success_gates_cfg.get("brickBelow")
    require_brick_above_false = gate_utils.bool_gate_target(above_gate_cfg) is False
    require_brick_below_false = gate_utils.bool_gate_target(below_gate_cfg) is False
    require_stack_false_gate = bool(require_brick_above_false or require_brick_below_false)

    reminder_msg = str(topmost_cfg.get("operator_log_reminder") or "").strip()
    ground_reset_enabled = bool(topmost_cfg.get("ground_reset_enabled", True))
    ground_cmd = str(topmost_cfg.get("ground_reset_command") or "d").strip().lower()
    if ground_cmd not in ("d", "u"):
        ground_cmd = "d"
    ground_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(topmost_cfg.get("ground_reset_score"), 100))),
    )
    ground_max_acts = max(1, int(_as_int(topmost_cfg.get("ground_reset_max_acts"), 12)))
    ground_min_acts = max(0, int(_as_int(topmost_cfg.get("ground_reset_min_acts"), 0)))
    ground_min_acts = min(int(ground_min_acts), int(ground_max_acts))
    ground_y_max = float(_as_float(topmost_cfg.get("ground_level_y_axis_max"), 50.0))

    mast_cmd = str(topmost_cfg.get("mast_up_command") or "u").strip().lower()
    if mast_cmd not in ("u", "d"):
        mast_cmd = "u"
    mast_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(topmost_cfg.get("mast_up_score"), 100))),
    )
    mast_max_acts = max(1, int(_as_int(topmost_cfg.get("mast_up_max_acts"), 3)))
    no_streak_required = max(
        1,
        int(
            _as_int(
                topmost_cfg.get("consecutive_no_required"),
                _as_int(topmost_cfg.get("false_up_acts_required"), 3),
            )
        ),
    )
    level2_reset_on_skipped_observation = bool(
        topmost_cfg.get("level2_reset_on_skipped_observation", True)
    )
    level2_use_full_gatecheck = bool(topmost_cfg.get("level2_use_full_gatecheck", True))
    level2_false_confirm_frames = max(
        1,
        int(_as_int(topmost_cfg.get("false_confirm_frames"), 1)),
    )
    level2_confirm_frames = 1 if bool(level2_use_full_gatecheck) else int(level2_false_confirm_frames)
    level2_require_visible_for_confirm = bool(
        topmost_cfg.get("level2_require_visible_for_confirm", True)
    )
    level2_fail_on_skipped_observation = bool(
        topmost_cfg.get("level2_fail_on_skipped_observation", False)
    )
    observation_confidence_min = _as_float(
        topmost_cfg.get("observation_confidence_min"),
        ARUCO_GATE_MIN_CONFIDENCE,
    )
    if observation_confidence_min is not None:
        observation_confidence_min = max(0.0, min(100.0, float(observation_confidence_min)))
    post_ground_reset_handoff_cfg = topmost_cfg.get("post_ground_reset_handoff")
    if not isinstance(post_ground_reset_handoff_cfg, dict):
        post_ground_reset_handoff_cfg = {}
    post_ground_reset_handoff_step = normalize_step_label(post_ground_reset_handoff_cfg.get("step"))
    post_ground_reset_handoff_acts = max(1, int(_as_int(post_ground_reset_handoff_cfg.get("acts"), 1)))
    if not post_ground_reset_handoff_step:
        if step_key == "FIND_TOPMOST_BRICK":
            post_ground_reset_handoff_step = "BRICK_LOCK"
        elif step_key == "FIND_TOPMOST_BRICK_WALL":
            post_ground_reset_handoff_step = "BRICK_LOCK_WALL"
    if "enabled" in post_ground_reset_handoff_cfg:
        post_ground_reset_handoff_enabled = bool(post_ground_reset_handoff_cfg.get("enabled"))
    else:
        # Explicit opt-in only: implicit step injection is too fragile.
        post_ground_reset_handoff_enabled = False
    if not bool(ground_reset_enabled):
        post_ground_reset_handoff_enabled = False
    post_ground_reset_handoff_min_acts = max(
        0,
        int(_as_int(post_ground_reset_handoff_cfg.get("min_acts"), 1 if post_ground_reset_handoff_enabled else 0)),
    )
    post_ground_reset_handoff_min_acts = min(
        int(post_ground_reset_handoff_min_acts),
        int(post_ground_reset_handoff_acts),
    )
    success_tracker = new_success_tracker(step, world.process_rules)

    def _crosshair_obs_from_world_snapshot(brick_local):
        if not isinstance(brick_local, dict):
            brick_local = {}
        raw_crosshair = brick_local.get("inCrosshairs")
        if raw_crosshair is None:
            raw_crosshair = brick_local.get("in_crosshairs")
        raw_crosshair_val = None if raw_crosshair is None else bool(raw_crosshair)

        raw_visible = brick_local.get("visible")
        # Legacy test doubles may omit visible; treat as visible rather than
        # forcing synthetic false negatives in the level2 exception logic.
        visible_val = True if raw_visible is None else bool(raw_visible)

        x_axis_local = brick_local.get("x_axis")
        if x_axis_local is None:
            x_axis_local = brick_local.get("offset_x")
        y_axis_local = brick_local.get("y_axis")
        if y_axis_local is None:
            y_axis_local = brick_local.get("offset_y")
        x_axis_num = _as_float(x_axis_local, None)
        y_axis_num = _as_float(y_axis_local, None)

        calc_crosshair_val = None
        if x_axis_num is not None and y_axis_num is not None:
            try:
                calc_crosshair_val = bool(
                    telemetry_brick.compute_in_crosshairs_for_step(
                        world,
                        visible_val,
                        x_axis_num,
                        y_axis_num,
                        step=step,
                        process_rules=getattr(world, "process_rules", None),
                    )
                )
            except Exception:
                calc_crosshair_val = None

        # Conservative merge: if either source says YES, treat as YES. This
        # prevents false NO streak accumulation from source disagreement.
        if raw_crosshair_val is None:
            merged_crosshair = calc_crosshair_val
        elif calc_crosshair_val is None:
            merged_crosshair = raw_crosshair_val
        else:
            merged_crosshair = bool(raw_crosshair_val or calc_crosshair_val)

        mismatch = bool(
            raw_crosshair_val is not None
            and calc_crosshair_val is not None
            and bool(raw_crosshair_val) != bool(calc_crosshair_val)
        )
        return {
            "crosshair": merged_crosshair,
            "crosshair_raw": raw_crosshair_val,
            "crosshair_calc": calc_crosshair_val,
            "crosshair_mismatch": mismatch,
            "visible": bool(visible_val),
            "x_axis": x_axis_num,
            "y_axis": y_axis_num,
        }

    def _update_and_read_crosshair():
        update_world_from_vision(world, vision, log=not align_silent)
        if observer:
            observer("frame", world, vision, None, None, None)
        brick_local = getattr(world, "brick", {}) or {}
        crosshair_obs = _crosshair_obs_from_world_snapshot(brick_local)
        crosshair_val = crosshair_obs.get("crosshair")
        y_axis_local = crosshair_obs.get("y_axis")
        conf_now = _as_float(brick_local.get("confidence"), None)
        raw_above = brick_local.get("brickAbove")
        if raw_above is None:
            raw_above = brick_local.get("brick_above")
        brick_above_now = None if raw_above is None else bool(raw_above)
        raw_below = brick_local.get("brickBelow")
        if raw_below is None:
            raw_below = brick_local.get("brick_below")
        brick_below_now = None if raw_below is None else bool(raw_below)
        obs_confident = True
        if conf_now is not None and observation_confidence_min is not None:
            obs_confident = float(conf_now) >= float(observation_confidence_min)
        frame_id_now = int(getattr(world, "_frame_id", 0) or 0)
        return (
            crosshair_val,
            _as_float(y_axis_local, None),
            bool(crosshair_obs.get("visible", True)),
            bool(obs_confident),
            conf_now,
            brick_above_now,
            brick_below_now,
            frame_id_now,
            crosshair_obs.get("crosshair_raw"),
            crosshair_obs.get("crosshair_calc"),
            bool(crosshair_obs.get("crosshair_mismatch", False)),
        )

    def _crosshair_samples_consistent(samples, *, require_non_wait=True):
        if not isinstance(samples, (list, tuple)) or not samples:
            return False
        first = samples[0]
        if require_non_wait and first is None:
            return False
        return all(value == first for value in samples)

    def _crosshair_sample_text(samples):
        if not isinstance(samples, (list, tuple)):
            return ""
        parts = []
        for value in samples:
            parts.append("WAIT" if value is None else ("YES" if bool(value) else "NO"))
        return ",".join(parts)

    def _observe_level2_crosshair_confident(*, confirm_frames=1):
        required = max(1, int(confirm_frames or 1))
        samples = []
        latest_snapshot = None
        for idx in range(required):
            if idx > 0:
                wait_for_frame_settle(world, vision, 1, log=not align_silent)
            latest_snapshot = _update_and_read_crosshair()
            samples.append(latest_snapshot[0] if isinstance(latest_snapshot, tuple) else None)

        if not isinstance(latest_snapshot, tuple):
            latest_snapshot = (
                None,
                None,
                False,
                False,
                None,
                None,
                None,
                int(getattr(world, "_frame_id", 0) or 0),
                None,
                None,
                False,
            )
        (
            _crosshair_latest,
            _y_now,
            visible_now,
            obs_confident,
            obs_confidence,
            brick_above_now,
            brick_below_now,
            frame_id_now,
            raw_crosshair_now,
            calc_crosshair_now,
            crosshair_mismatch_now,
        ) = latest_snapshot

        if required <= 1:
            crosshair_now = samples[0] if samples else None
            samples_consistent = bool(crosshair_now is not None)
        else:
            samples_consistent = _crosshair_samples_consistent(samples, require_non_wait=True)
            crosshair_now = samples[0] if bool(samples_consistent) else None
        return (
            crosshair_now,
            tuple(samples),
            bool(samples_consistent),
            bool(visible_now),
            bool(obs_confident),
            obs_confidence,
            brick_above_now,
            brick_below_now,
            int(frame_id_now),
            raw_crosshair_now,
            calc_crosshair_now,
            bool(crosshair_mismatch_now),
        )

    if not align_silent:
        reminder = reminder_msg or "Exception active: mast-up 100% + level2 crosshair confidence confirmation."
        print(format_headline(f"[EXCEPTION] [{step_key}] {reminder}", COLOR_MAGENTA_BRIGHT))
        if ground_reset_enabled:
            print(
                format_headline(
                    (
                        f"[EXCEPTION] [{step_key}] Phase 1 (first): ground reset "
                        f"{str(ground_cmd).upper()} {int(ground_score)}% until ground gate passes "
                        f"(y_axis<={float(ground_y_max):.1f}) and min acts {int(ground_min_acts)}."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        else:
            print(
                format_headline(
                    (
                        f"[EXCEPTION] [{step_key}] Phase 1 (first): mast-only confidence sweep "
                        f"({str(mast_cmd).upper()} {int(mast_score)}%) while tracking inCrosshairs NO streak."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )

    exc_runtime_color = COLOR_WHITE
    ground_speed = telemetry_robot_module.manual_speed_for_cmd(ground_cmd, ground_score)
    if not ground_reset_enabled:
        ground_max_acts = 0
        ground_min_acts = 0
    ground_reached = not bool(ground_reset_enabled)
    ground_pulses_sent = 0
    prev_ground_lift_mm = _as_float(getattr(world, "lift_height", None), None)
    for idx in range(1, int(ground_max_acts) + 1):
        crosshair_now, y_now, _visible_now, _obs_ok, _obs_conf, _above_now, _below_now, _frame_id, _raw_crosshair, _calc_crosshair, _mismatch = _update_and_read_crosshair()
        if (
            y_now is not None
            and float(y_now) <= float(ground_y_max)
            and int(ground_pulses_sent) >= int(ground_min_acts)
        ):
            ground_reached = True
            break
        if confirm_callback and not confirm_callback(world, vision):
            return {"enabled": True, "handled": True, "success": False, "reason": "confirm cancelled"}
        if observer:
            observer("analysis", world, vision, ground_cmd, ground_speed, "ground_up_level2_ground_reset")
        send_robot_command(
            robot,
            world,
            step,
            ground_cmd,
            ground_speed,
            speed_score=ground_score,
            auto_mode=True,
        )
        ground_pulses_sent = int(ground_pulses_sent) + 1
        post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
        if observer:
            observer("action", world, vision, ground_cmd, ground_speed, "ground_up_level2_ground_reset")
        if not align_silent:
            lift_now_mm, lift_text = _exception_lift_estimate_text(world, prev_ground_lift_mm)
            if lift_now_mm is not None:
                prev_ground_lift_mm = float(lift_now_mm)
            print(
                format_headline(
                    (
                        f"[EXCEPTION] [{step_key}] Ground reset pulse {int(ground_pulses_sent)}/{int(ground_max_acts)}: "
                        f"{lift_text}."
                    ),
                    exc_runtime_color,
                )
            )
        time.sleep(CONTROL_DT)

    if not ground_reached:
        _cross, y_now, _visible_now, _obs_ok, _obs_conf, _above_now, _below_now, _frame_id, _raw_crosshair, _calc_crosshair, _mismatch = _update_and_read_crosshair()
        if (
            y_now is not None
            and float(y_now) <= float(ground_y_max)
            and int(ground_pulses_sent) >= int(ground_min_acts)
        ):
            ground_reached = True
    if not ground_reached:
        if robot:
            robot.stop()
        return {
            "enabled": True,
            "handled": True,
            "success": False,
            "reason": (
                f"ground reset not reached (need y_axis<={float(ground_y_max):.2f} and min acts {int(ground_min_acts)}; "
                f"sent={int(ground_pulses_sent)}/{int(ground_max_acts)}; "
                f"lift={_as_float(getattr(world, 'lift_height', None), None)})"
            ),
        }

    if ground_reset_enabled:
        start_after_ground = wait_for_start_gates(
            world,
            vision,
            step,
            robot=robot,
            cmd=None,
            speed=None,
            log=not align_silent,
            observer=observer,
            allow_success=False,
            force_require_start_gates=True,
        )
        if start_after_ground != "start":
            if robot:
                robot.stop()
            return {
                "enabled": True,
                "handled": True,
                "success": False,
                "reason": f"start gates not met after ground reset ({str(start_after_ground)})",
            }
        if not align_silent:
            print(
                format_headline(
                    f"[EXCEPTION] [{step_key}] Start gates confirmed after ground reset; entering mast-up phase.",
                    exc_runtime_color,
                )
            )

    if post_ground_reset_handoff_enabled and post_ground_reset_handoff_step:
        if robot:
            robot.stop()
        handoff_step_cfg = None
        if isinstance(getattr(world, "process_rules", None), dict):
            cfg_candidate = world.process_rules.get(post_ground_reset_handoff_step)
            if isinstance(cfg_candidate, dict):
                handoff_step_cfg = cfg_candidate
        if not align_silent:
            print(
                format_headline(
                    f"[EXCEPTION] [{step_key}] Post-ground-reset phase-2 cheat: injecting {post_ground_reset_handoff_step} behavior inside {step_key}, then continuing {step_key} phases.",
                    exc_runtime_color,
                )
            )
        if not isinstance(handoff_step_cfg, dict):
            if not align_silent:
                print(
                        format_headline(
                            f"[EXCEPTION] [{step_key}] Handoff phase skipped: {post_ground_reset_handoff_step} rules not found; continuing {step_key}.",
                            exc_runtime_color,
                        )
                    )
        else:
            handoff_tracker = new_success_tracker(post_ground_reset_handoff_step, world.process_rules)
            handoff_prev_correction_type = None
            handoff_acts_sent = 0
            handoff_completion_logged = False
            for handoff_idx in range(1, int(post_ground_reset_handoff_acts) + 1):
                if confirm_callback and not confirm_callback(world, vision):
                    return {"enabled": True, "handled": True, "success": False, "reason": "confirm cancelled"}
                update_world_from_vision(world, vision, log=not align_silent)
                if observer:
                    observer("frame", world, vision, None, None, None)
                pre_handoff_obs = pre_action_success_observation(
                    world,
                    vision,
                    post_ground_reset_handoff_step,
                    handoff_tracker,
                    phase="ground_up_handoff",
                    robot=robot,
                    observer=observer,
                    log=not align_silent,
                    success_log=False,
                )
                if bool((pre_handoff_obs or {}).get("success_met")):
                    if int(handoff_acts_sent) < int(post_ground_reset_handoff_min_acts):
                        if not align_silent:
                            print(
                                format_headline(
                                    (
                                        f"[EXCEPTION] [{step_key}] Handoff phase precheck already passing, "
                                        f"but min acts not met ({int(handoff_acts_sent)}/{int(post_ground_reset_handoff_min_acts)}); "
                                        f"executing bridge act."
                                    ),
                                    exc_runtime_color,
                                )
                            )
                    else:
                        if not align_silent:
                            print(
                                format_headline(
                                    f"[EXCEPTION] [{step_key}] Handoff phase {post_ground_reset_handoff_step} already satisfies success gates; continuing {step_key}.",
                                    exc_runtime_color,
                                )
                            )
                        break
                if bool((pre_handoff_obs or {}).get("hold_for_confirm")):
                    if int(handoff_acts_sent) < int(post_ground_reset_handoff_min_acts):
                        if not align_silent:
                            print(
                                format_headline(
                                    (
                                        f"[EXCEPTION] [{step_key}] Handoff confirm-hold suppressed until min acts "
                                        f"{int(post_ground_reset_handoff_min_acts)} are sent "
                                        f"({int(handoff_acts_sent)}/{int(post_ground_reset_handoff_min_acts)})."
                                    ),
                                    exc_runtime_color,
                                )
                            )
                    else:
                        time.sleep(CONTROL_DT)
                        continue

                brick_local = getattr(world, "brick", {}) or {}
                handoff_plan = next_module.select_alignment_next_act(
                    process_rules=world.process_rules or {},
                    learned_rules=world.learned_rules or {},
                    step=post_ground_reset_handoff_step,
                    x_axis_mm=brick_local.get("x_axis", brick_local.get("offset_x")),
                    y_axis_mm=brick_local.get("y_axis", brick_local.get("offset_y")),
                    dist_mm=brick_local.get("dist"),
                    visible=bool(brick_local.get("visible")),
                    angle_deg=brick_local.get("angle", 0.0),
                    duration_s=CONTROL_DT,
                    previous_correction_type=handoff_prev_correction_type,
                    avoid_correction_type=None,
                )
                handoff_cmd = handoff_plan.get("cmd") if isinstance(handoff_plan, dict) else None
                handoff_score = handoff_plan.get("score") if isinstance(handoff_plan, dict) else None
                handoff_speed = handoff_plan.get("speed") if isinstance(handoff_plan, dict) else None
                if isinstance(handoff_plan, dict):
                    handoff_prev_correction_type = (
                        str(handoff_plan.get("correction_type") or "").strip().lower() or None
                    )
                if handoff_cmd is None:
                    handoff_fallback = _align_brick_lite_fail_fallback_action(world, post_ground_reset_handoff_step)
                    if isinstance(handoff_fallback, dict):
                        handoff_cmd = handoff_fallback.get("cmd")
                        handoff_score = handoff_fallback.get("score")
                        handoff_speed = handoff_fallback.get("speed")
                if handoff_cmd is None and int(handoff_acts_sent) < int(post_ground_reset_handoff_min_acts):
                    handoff_cmd = telemetry_robot_module.resolve_scan_direction(
                        world.process_rules or {},
                        post_ground_reset_handoff_step,
                        fallback="l",
                    )
                    if handoff_cmd not in ("l", "r"):
                        handoff_cmd = "l"
                    if handoff_score is None:
                        handoff_score = int(getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1))
                if handoff_cmd is None:
                    if not align_silent:
                        print(
                            format_headline(
                                f"[EXCEPTION] [{step_key}] Handoff pulse {int(handoff_idx)}/{int(post_ground_reset_handoff_acts)}: no actionable {post_ground_reset_handoff_step} command; continuing.",
                                exc_runtime_color,
                            )
                        )
                    time.sleep(CONTROL_DT)
                    continue

                if handoff_score is None:
                    _q_speed, q_score = telemetry_robot_module.quantize_speed(handoff_cmd, speed=handoff_speed)
                    handoff_score = q_score
                handoff_score = max(
                    int(telemetry_robot_module.SPEED_SCORE_MIN),
                    min(int(MAX_SPEED_SCORE), int(_as_int(handoff_score, int(telemetry_robot_module.SPEED_SCORE_MIN)))),
                )
                if handoff_speed is None or float(handoff_speed) <= 0.0:
                    handoff_speed = telemetry_robot_module.manual_speed_for_cmd(handoff_cmd, handoff_score)

                if observer:
                    observer("analysis", world, vision, handoff_cmd, handoff_speed, "ground_up_level2_handoff")
                send_robot_command(
                    robot,
                    world,
                    step,
                    handoff_cmd,
                    handoff_speed,
                    speed_score=handoff_score,
                    auto_mode=True,
                )
                handoff_acts_sent = int(handoff_acts_sent) + 1
                post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
                if observer:
                    observer("action", world, vision, handoff_cmd, handoff_speed, "ground_up_level2_handoff")
                if not align_silent:
                    print(
                        format_headline(
                            f"[EXCEPTION] [{step_key}] Handoff pulse {int(handoff_idx)}/{int(post_ground_reset_handoff_acts)}: {str(handoff_cmd).upper()} {int(handoff_score)}% using {post_ground_reset_handoff_step} planner.",
                            exc_runtime_color,
                        )
                    )
                    print(
                        format_headline(
                            (
                                f"[EXCEPTION] [{step_key}] phase2_min_acts sent="
                                f"{int(handoff_acts_sent)}/{int(post_ground_reset_handoff_min_acts)} "
                                f"(max={int(post_ground_reset_handoff_acts)})."
                            ),
                            exc_runtime_color,
                        )
                    )
                handoff_success = run_full_gatecheck_after_act(
                    world,
                    vision,
                    post_ground_reset_handoff_step,
                    handoff_tracker,
                    phase="ground_up_handoff",
                    log=not align_silent,
                    observer=observer,
                )
                if bool(handoff_success):
                    if not align_silent:
                        print(
                            format_headline(
                                f"[EXCEPTION] [{step_key}] Handoff phase reached {post_ground_reset_handoff_step} success gates; continuing {step_key}.",
                                exc_runtime_color,
                            )
                        )
                        print(
                            format_headline(
                                (
                                    f"[EXCEPTION] [{step_key}] phase2 complete: sent "
                                    f"{int(handoff_acts_sent)} act(s) "
                                    f"(min={int(post_ground_reset_handoff_min_acts)}, max={int(post_ground_reset_handoff_acts)})."
                                ),
                                exc_runtime_color,
                            )
                        )
                        handoff_completion_logged = True
                    break
                time.sleep(CONTROL_DT)

            if not align_silent and int(handoff_acts_sent) > 0 and not bool(handoff_completion_logged):
                print(
                    format_headline(
                        (
                            f"[EXCEPTION] [{step_key}] phase2 complete: sent "
                            f"{int(handoff_acts_sent)} act(s) "
                            f"(min={int(post_ground_reset_handoff_min_acts)}, max={int(post_ground_reset_handoff_acts)})."
                        ),
                        exc_runtime_color,
                    )
                )

    mast_speed = telemetry_robot_module.manual_speed_for_cmd(mast_cmd, mast_score)
    confident_no_streak = 0
    mast_pulses_sent = 0
    last_crosshair = None
    last_brick_above = None
    last_brick_below = None
    last_success_hit = False
    last_obs_confident = True
    last_obs_confidence = None
    last_obs_fresh = True
    stack_gate_armed = False

    def _read_crosshair_from_world_snapshot():
        brick_local = getattr(world, "brick", {}) or {}
        crosshair_obs = _crosshair_obs_from_world_snapshot(brick_local)
        crosshair_local = crosshair_obs.get("crosshair")
        raw_above_local = brick_local.get("brickAbove")
        if raw_above_local is None:
            raw_above_local = brick_local.get("brick_above")
        above_local = None if raw_above_local is None else bool(raw_above_local)
        raw_below_local = brick_local.get("brickBelow")
        if raw_below_local is None:
            raw_below_local = brick_local.get("brick_below")
        below_local = None if raw_below_local is None else bool(raw_below_local)
        conf_local = _as_float(brick_local.get("confidence"), None)
        frame_id_local = int(getattr(world, "_frame_id", 0) or 0)
        return crosshair_local, above_local, below_local, conf_local, frame_id_local

    prev_mast_lift_mm = _as_float(getattr(world, "lift_height", None), None)
    for idx in range(1, int(mast_max_acts) + 1):
        mast_pulses_sent = int(idx)
        if confirm_callback and not confirm_callback(world, vision):
            return {"enabled": True, "handled": True, "success": False, "reason": "confirm cancelled"}
        pre_crosshair_now, pre_above_now, pre_below_now, pre_conf_now, pre_obs_frame_id = _read_crosshair_from_world_snapshot()
        pre_crosshair_text = "WAIT" if pre_crosshair_now is None else ("YES" if pre_crosshair_now else "NO")
        pre_above_text = "WAIT" if pre_above_now is None else ("YES" if pre_above_now else "NO")
        pre_below_text = "WAIT" if pre_below_now is None else ("YES" if pre_below_now else "NO")
        if pre_conf_now is None:
            pre_conf_text = "n/a"
        else:
            pre_conf_text = f"{float(pre_conf_now):.1f}%"
        pre_frame_id = int(getattr(world, "_frame_id", 0) or 0)
        if observer:
            observer("analysis", world, vision, mast_cmd, mast_speed, "ground_up_level2_mast_up")
        send_robot_command(
            robot,
            world,
            step,
            mast_cmd,
            mast_speed,
            speed_score=mast_score,
            auto_mode=True,
        )
        post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
        if level2_use_full_gatecheck:
            success_hit = run_full_gatecheck_after_act(
                world,
                vision,
                step,
                success_tracker,
                phase="ground_up_level2",
                log=not align_silent,
                observer=observer,
            )
        else:
            success_hit = False
        (
            crosshair_now,
            crosshair_samples,
            crosshair_samples_consistent,
            visible_now,
            obs_confident,
            obs_confidence,
            brick_above_now,
            brick_below_now,
            frame_id_now,
            raw_crosshair_now,
            calc_crosshair_now,
            crosshair_mismatch_now,
        ) = _observe_level2_crosshair_confident(confirm_frames=level2_confirm_frames)
        last_crosshair = crosshair_now
        last_brick_above = brick_above_now
        last_brick_below = brick_below_now
        last_success_hit = bool(success_hit)
        last_obs_confident = bool(obs_confident)
        last_obs_confidence = obs_confidence
        fresh_after_mast = int(frame_id_now) > int(pre_frame_id)
        last_obs_fresh = bool(fresh_after_mast)
        visible_ok_for_level2 = bool(visible_now) or (not bool(level2_require_visible_for_confirm))
        base_usable = (
            bool(visible_ok_for_level2)
            and bool(obs_confident)
            and bool(fresh_after_mast)
            and (crosshair_now is not None)
        )
        stack_known = True
        stack_clear = True
        if require_brick_above_false:
            if brick_above_now is None:
                stack_known = False
            elif bool(brick_above_now):
                stack_clear = False
        if require_brick_below_false:
            if brick_below_now is None:
                stack_known = False
            elif bool(brick_below_now):
                stack_clear = False
        skip_reason = None
        if not bool(fresh_after_mast):
            skip_reason = "stale frame"
        elif bool(level2_require_visible_for_confirm) and not bool(visible_now):
            skip_reason = "visible=NO"
        elif not bool(obs_confident):
            if obs_confidence is None:
                skip_reason = "low confidence"
            else:
                skip_reason = (
                    f"conf={float(obs_confidence):.1f}% < "
                    f"{float(observation_confidence_min or 0.0):.1f}%"
                )
        elif require_brick_above_false and brick_above_now is None:
            skip_reason = "brickAbove=WAIT"
        elif require_brick_below_false and brick_below_now is None:
            skip_reason = "brickBelow=WAIT"
        elif require_brick_above_false and bool(brick_above_now):
            skip_reason = "brickAbove=YES"
        elif require_brick_below_false and bool(brick_below_now):
            skip_reason = "brickBelow=YES"
        elif crosshair_now is None:
            if int(level2_confirm_frames) > 1 and not bool(crosshair_samples_consistent):
                skip_reason = "inCrosshairs=UNSTABLE"
            else:
                skip_reason = "inCrosshairs=WAIT"
        usable_for_gate = bool(base_usable and stack_known)
        stack_gate_armed = bool(stack_clear) if bool(require_stack_false_gate) else True
        used_observation = bool(usable_for_gate) and bool(stack_gate_armed)
        if (not bool(used_observation)) and bool(level2_fail_on_skipped_observation):
            fail_reason = (
                f"level2 observation skipped at cycle {int(idx)}/{int(mast_max_acts)}: "
                f"{str(skip_reason or 'unqualified observation')}"
            )
            if not align_silent:
                print(
                    format_headline(
                        "[ERROR]",
                        COLOR_RED,
                        f" [{step_key}] {fail_reason}; fail-fast policy enabled.",
                    )
                )
            if robot:
                robot.stop()
            return {"enabled": True, "handled": True, "success": False, "reason": fail_reason}
        condition_hit = bool(
            used_observation
            and (crosshair_now is False)
        )
        # Level2 streak policy:
        # - PASS cycle increments.
        # - Contradicting usable observation resets.
        # - Skipped observation can hold or reset based on step config.
        prev_confident_no_streak = int(confident_no_streak)
        if bool(condition_hit):
            confident_no_streak = int(confident_no_streak) + 1
            streak_update_state = "counted"
        elif bool(used_observation):
            confident_no_streak = 0
            streak_update_state = "reset"
        elif bool(level2_reset_on_skipped_observation):
            confident_no_streak = 0
            streak_update_state = "reset_skip"
        else:
            confident_no_streak = int(confident_no_streak)
            streak_update_state = "hold"
        level2_ok = int(confident_no_streak) >= int(no_streak_required)
        if observer:
            observer("action", world, vision, mast_cmd, mast_speed, "ground_up_level2_mast_up")
        if not align_silent:
            c_text = "WAIT" if crosshair_now is None else ("YES" if crosshair_now else "NO")
            if brick_above_now is None:
                b_text = "WAIT"
            else:
                b_text = "YES" if bool(brick_above_now) else "NO"
            if brick_below_now is None:
                bb_text = "WAIT"
            else:
                bb_text = "YES" if bool(brick_below_now) else "NO"
            # `cycle_condition` is per-cycle truth only.
            if bool(condition_hit):
                cond_word = f"{COLOR_GREEN}PASS{COLOR_RESET}"
                streak_word = f"{COLOR_GREEN}PASS{COLOR_RESET}"
            elif bool(used_observation):
                cond_word = f"{COLOR_RED}FAIL{COLOR_RESET}"
                streak_word = f"{COLOR_RED}FAIL{COLOR_RESET}"
            else:
                if bool(level2_reset_on_skipped_observation):
                    cond_word = f"{COLOR_RED}FAIL{COLOR_RESET}"
                    streak_word = f"{COLOR_RED}FAIL{COLOR_RESET}"
                else:
                    cond_word = f"{COLOR_ORANGE_BRIGHT}SKIP{COLOR_RESET}"
                    streak_word = f"{COLOR_ORANGE_BRIGHT}SKIP{COLOR_RESET}"
            if bool(level2_ok):
                progress_word = f"{COLOR_GREEN}COMPLETE{COLOR_RESET}"
            elif bool(condition_hit):
                progress_word = f"{COLOR_ORANGE_BRIGHT}COUNTED{COLOR_RESET}"
            elif streak_update_state in {"reset", "reset_skip"} and int(prev_confident_no_streak) > 0:
                progress_word = f"{COLOR_RED}RESET{COLOR_RESET}"
            elif streak_update_state == "hold" and int(prev_confident_no_streak) > 0:
                progress_word = f"{COLOR_ORANGE_BRIGHT}HOLD{COLOR_RESET}"
            else:
                progress_word = f"{COLOR_WHITE}WAIT{COLOR_RESET}"
            armed_word = f"{COLOR_GREEN}YES{COLOR_RESET}" if bool(stack_gate_armed) else f"{COLOR_RED}NO{COLOR_RESET}"
            if skip_reason is not None:
                if bool(level2_reset_on_skipped_observation):
                    obs_text = f"FAIL({str(skip_reason)})"
                else:
                    obs_text = f"SKIP({str(skip_reason)})"
            else:
                if (not bool(visible_now)) and (not bool(level2_require_visible_for_confirm)):
                    obs_text = "USED(visible=NO allowed)"
                else:
                    obs_text = "USED"
            raw_crosshair_text = "WAIT" if raw_crosshair_now is None else ("YES" if bool(raw_crosshair_now) else "NO")
            calc_crosshair_text = "WAIT" if calc_crosshair_now is None else ("YES" if bool(calc_crosshair_now) else "NO")
            crosshair_samples_note = ""
            if int(level2_confirm_frames) > 1:
                crosshair_samples_note = (
                    f"obs{int(level2_confirm_frames)}=[{_crosshair_sample_text(crosshair_samples)}], "
                )
            mismatch_text = (
                f"crosshair-src(raw={raw_crosshair_text}, calc={calc_crosshair_text}), "
                if bool(crosshair_mismatch_now)
                else ""
            )
            lift_now_mm, lift_text = _exception_lift_estimate_text(world, prev_mast_lift_mm)
            if lift_now_mm is not None:
                prev_mast_lift_mm = float(lift_now_mm)
            print(
                format_headline(
                    "[EXCEPTION]",
                    COLOR_MAGENTA_BRIGHT,
                    f" {exc_runtime_color}[{step_key}] Cycle {int(idx)}/{int(mast_max_acts)}: "
                    f"observe(pre)=inCrosshairs={pre_crosshair_text}, "
                    f"brickAbove={pre_above_text}, "
                    + (
                        f"brickBelow={pre_below_text}, "
                        if bool(require_brick_below_false)
                        else ""
                    )
                    + f"conf={pre_conf_text}, frame={int(pre_obs_frame_id)} "
                    + f"-> act={COLOR_ORANGE_BRIGHT}{('MAST_UP' if str(mast_cmd) == 'u' else 'MAST_DOWN' if str(mast_cmd) == 'd' else str(mast_cmd).upper())} {int(mast_score)}%{exc_runtime_color} "
                    + f"-> confirm: inCrosshairs={c_text}, "
                    + f"{crosshair_samples_note}"
                    + f"brickAbove={b_text}, "
                    + (
                        f"brickBelow={bb_text}, "
                        if bool(require_brick_below_false)
                        else ""
                    )
                    + (
                        f"gate-armed={armed_word}, "
                        if bool(require_stack_false_gate)
                        else ""
                    )
                    + (
                        f"{mismatch_text}"
                        f"observation={obs_text}, "
                        f"cycle_condition={cond_word}, "
                        f"level2 success gate check={int(confident_no_streak)}/{int(no_streak_required)} ({streak_word}), "
                        f"streak_state={progress_word}, "
                        + (
                            f"gatecheck={('PASS' if bool(success_hit) else 'FAIL')}, "
                            if bool(level2_use_full_gatecheck)
                            else ""
                        )
                        + f"{lift_text}.{COLOR_RESET}"
                    ),
                )
            )
        time.sleep(CONTROL_DT)
        if level2_ok:
            if robot:
                robot.stop()
            if not align_silent:
                print(
                    format_headline(
                        "[EXCEPTION]",
                        COLOR_MAGENTA_BRIGHT,
                        (
                            f" {exc_runtime_color}[{step_key}] Transition: "
                            f"inCrosshairs=NO observed for {int(no_streak_required)} "
                            f"consecutive confident mast-up observations"
                            + (
                                (
                                    " with brickAbove=NO and brickBelow=NO"
                                    if bool(require_brick_above_false and require_brick_below_false)
                                    else (
                                        " with brickAbove=NO"
                                        if bool(require_brick_above_false)
                                        else (
                                            " with brickBelow=NO"
                                            if bool(require_brick_below_false)
                                            else ""
                                        )
                                    )
                                )
                            )
                            + f"; completing step.{COLOR_RESET}"
                        ),
                    )
                )
                print(format_headline(f"[SUCCESS] {step_key} criteria met", COLOR_GREEN))
                if level2_use_full_gatecheck:
                    print_gate_summary_line(world, success_tracker)
                print_success_events(world, step)
            return {"enabled": True, "handled": True, "success": True, "reason": "success gate + level2"}

    if robot:
        robot.stop()
    last_crosshair_text = "WAIT" if last_crosshair is None else ("YES" if bool(last_crosshair) else "NO")
    if not bool(last_obs_confident):
        if last_obs_confidence is None:
            last_conf_text = "low/unknown"
        else:
            last_conf_text = f"{float(last_obs_confidence):.1f}%<{float(observation_confidence_min or 0.0):.1f}%"
    else:
        last_conf_text = (
            "unknown"
            if last_obs_confidence is None
            else f"{float(last_obs_confidence):.1f}%"
        )
    return {
        "enabled": True,
        "handled": True,
        "success": False,
        "reason": (
            "level2 no-streak condition not satisfied within mast-up limit "
            f"({int(mast_pulses_sent)}/{int(mast_max_acts)} pulses; "
            f"last_inCrosshairs={last_crosshair_text}; "
            + (
                f"last_brickAbove={('WAIT' if last_brick_above is None else ('YES' if bool(last_brick_above) else 'NO'))}; "
                if bool(require_brick_above_false)
                else ""
            )
            + (
                f"last_brickBelow={('WAIT' if last_brick_below is None else ('YES' if bool(last_brick_below) else 'NO'))}; "
                if bool(require_brick_below_false)
                else ""
            )
            + (
                f"gate_armed={('YES' if bool(stack_gate_armed) else 'NO')}; "
                if bool(require_stack_false_gate)
                else ""
            )
            + (
                f"last_obs_conf={last_conf_text}; "
                f"last_obs_fresh={('YES' if bool(last_obs_fresh) else 'NO')}; "
            )
            + (
                f"last_gatecheck={('PASS' if bool(last_success_hit) else 'FAIL')}; "
                if bool(level2_use_full_gatecheck)
                else ""
            )
            + f"no_streak={int(confident_no_streak)}/{int(no_streak_required)}; "
            + "requires consecutive PASS cycles)"
        ),
    }


def pause_after_fail(robot):
    if robot:
        robot.stop()
    time.sleep(FAIL_PAUSE_S)


def _visible_success_target_false(gates):
    if not isinstance(gates, dict):
        return False
    visible_cfg = gates.get("visible")
    if not isinstance(visible_cfg, dict):
        return False
    return gate_utils.bool_gate_target(visible_cfg) is False


def _inject_aruco_confidence_gate(gates):
    if not isinstance(gates, dict):
        return
    gates.pop("confidence", None)


def _remove_aruco_confidence_gate(gates):
    if not isinstance(gates, dict):
        return
    gates.pop("confidence", None)


def _aruco_confidence_gate_enabled(step_cfg):
    if not isinstance(step_cfg, dict):
        return False
    setting = step_cfg.get("aruco_confidence_gate")
    if setting is None:
        return False
    return bool(setting)


def _apply_aruco_confidence_policy(model):
    if not isinstance(model, dict):
        return model
    steps = model.get("steps")
    if not isinstance(steps, dict):
        return model
    for _, cfg in steps.items():
        if not isinstance(cfg, dict):
            continue
        success_gates = cfg.get("success_gates")
        if isinstance(success_gates, dict):
            _remove_aruco_confidence_gate(success_gates)
        locked = cfg.get("locked_success_gates_by_mode")
        if isinstance(locked, dict):
            for _mode, mode_gates in locked.items():
                if isinstance(mode_gates, dict):
                    _remove_aruco_confidence_gate(mode_gates)
    return model


def load_process_model(path=PROCESS_MODEL_FILE):
    if not path.exists():
        return _apply_aruco_confidence_policy(_canonicalize_process_model({"steps": {}}))
    try:
        model = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        model = {"steps": {}}
    return _apply_aruco_confidence_policy(_canonicalize_process_model(model))


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


def _canonicalize_process_model(model):
    if not isinstance(model, dict):
        return {"steps": {}}
    lite_cfg = model.get("lite_gate_check")
    if isinstance(lite_cfg, dict):
        lite_cfg.pop("steps", None)
    steps = model.get("steps")
    if isinstance(steps, dict):
        for _, cfg in steps.items():
            if isinstance(cfg, dict):
                cfg.pop("success_metrics", None)
    return model


def write_process_model(model, path=PROCESS_MODEL_FILE):
    model = _canonicalize_process_model(model)
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
    def _metric_alias_group(metric_name):
        metric_key = str(metric_name or "").strip()
        if metric_key in {"xAxis_offset_abs", "xAxis_offset", "x_axis"}:
            return "x_axis"
        if metric_key in {"yAxis_offset_abs", "yAxis_offset", "y_axis"}:
            return "y_axis"
        return metric_key

    obj_key = normalize_step_label(step)
    rules = (step_rules or {}).get(obj_key, {})
    metrics_out = list(metrics or [])
    alignment_metrics = rules.get("alignment_metrics")
    if isinstance(alignment_metrics, list):
        metrics_out = [metric for metric in metrics_out if metric in alignment_metrics]
    allowed = PROCESS_SUCCESS_METRICS_BY_STEP.get(obj_key)
    if not isinstance(allowed, (list, tuple)):
        allowed = rules.get("success_metrics")
    if isinstance(allowed, (list, tuple)):
        allowed_metrics = [str(metric).strip() for metric in allowed if str(metric).strip()]
        allowed_groups = {_metric_alias_group(metric) for metric in allowed_metrics}
        preferred_metric_by_group = {}
        for metric in allowed_metrics:
            group = _metric_alias_group(metric)
            if group not in preferred_metric_by_group:
                preferred_metric_by_group[group] = metric
        filtered = []
        seen = set()
        for metric in metrics_out:
            metric_key = str(metric).strip()
            group = _metric_alias_group(metric_key)
            if group not in allowed_groups:
                continue
            preferred_metric = preferred_metric_by_group.get(group, metric_key)
            if preferred_metric in seen:
                continue
            filtered.append(preferred_metric)
            seen.add(preferred_metric)
        metrics_out = filtered
    if obj_key in SUCCESS_GATE_EXCLUDE_ANGLE_STEPS:
        metrics_out = [metric for metric in metrics_out if metric != "angle_abs"]
    return metrics_out


def _visible_only_success_gate_steps(step_rules=None):
    out = set()
    for step_name, metric_list in PROCESS_SUCCESS_METRICS_BY_STEP.items():
        if not isinstance(metric_list, (list, tuple)):
            continue
        metric_keys = {str(metric).strip() for metric in metric_list if str(metric).strip()}
        if metric_keys == {"visible"}:
            norm = normalize_step_label(step_name)
            if norm:
                out.add(norm)
    if isinstance(step_rules, dict):
        for step_name, step_cfg in step_rules.items():
            if not isinstance(step_cfg, dict):
                continue
            metrics = step_cfg.get("success_metrics")
            if not isinstance(metrics, list):
                continue
            metric_keys = {str(metric).strip() for metric in metrics if str(metric).strip()}
            if metric_keys == {"visible"}:
                norm = normalize_step_label(step_name)
                if norm:
                    out.add(norm)
    return out


def metric_value_from_state(state, metric):
    brick = state.get("brick") or {}
    if metric == "angle_abs":
        val = brick.get("angle")
        return abs(val) if val is not None else None
    if metric in ("xAxis_offset_abs", "xAxis_offset", "x_axis"):
        val = brick.get("x_axis")
        if val is None:
            val = brick.get("offset_x")
        return float(val) if val is not None else None
    if metric in ("yAxis_offset_abs", "yAxis_offset", "y_axis"):
        val = brick.get("y_axis")
        if val is None:
            val = brick.get("offset_y")
        return float(val) if val is not None else None
    if metric == "dist":
        val = brick.get("dist")
        if val is None or val <= 0:
            return None
        return float(val)
    if metric in ("brick_above", "brickAbove"):
        if "brickAbove" in brick:
            val = brick.get("brickAbove")
            return None if val is None else bool(val)
        if "brick_above" in brick:
            val = brick.get("brick_above")
            return None if val is None else bool(val)
        return None
    if metric in ("brick_below", "brickBelow"):
        if "brickBelow" in brick:
            val = brick.get("brickBelow")
            return None if val is None else bool(val)
        if "brick_below" in brick:
            val = brick.get("brick_below")
            return None if val is None else bool(val)
        return None
    if metric in ("inCrosshairs", "in_crosshairs"):
        if "inCrosshairs" in brick:
            val = brick.get("inCrosshairs")
            return None if val is None else bool(val)
        if "in_crosshairs" in brick:
            val = brick.get("in_crosshairs")
            return None if val is None else bool(val)
        return None
    if metric == "visible":
        return bool(brick.get("visible"))
    if metric == "confidence":
        return brick.get("confidence")
    if metric == "lift_height":
        return state.get("lift_height")
    return None


def metric_direction(metric, step, process_rules=None):
    if metric in next_module.METRIC_DIRECTIONS:
        return next_module.metric_direction_for_step(metric, step, process_rules=process_rules)
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
        step_key = normalize_step_label(obj)
        metrics = PROCESS_START_METRICS_BY_STEP.get(step_key, ("visible",))
        metric_votes = {str(metric): [] for metric in (metrics or ("visible",))}
        for seg in segs:
            states = select_head_states(seg.get("states") or [])
            if not states:
                continue
            for metric in list(metric_votes.keys()):
                per_state_vals = []
                for state in states:
                    value = metric_value_from_state(state, metric)
                    if value is None:
                        continue
                    per_state_vals.append(bool(value))
                if not per_state_vals:
                    continue
                true_ratio = sum(1 for v in per_state_vals if v) / len(per_state_vals)
                metric_votes[metric].append(true_ratio >= 0.5)

        obj_gates = {}
        for metric, votes in metric_votes.items():
            if not votes:
                continue
            consensus_true = (sum(1 for v in votes if v) / len(votes)) >= 0.5
            if consensus_true:
                obj_gates[metric] = {"min": True}
            else:
                obj_gates[metric] = {"max": False}
        if not obj_gates:
            continue
        start_gates[obj] = obj_gates
    return start_gates


def _bool_start_gate_from_success_stats(stats):
    if not isinstance(stats, dict):
        return None
    target = stats.get("target")
    if isinstance(target, bool):
        return {"min": True} if bool(target) else {"max": False}
    min_val = stats.get("min")
    if isinstance(min_val, bool):
        return {"min": bool(min_val)} if bool(min_val) else {"max": False}
    max_val = stats.get("max")
    if isinstance(max_val, bool):
        return {"min": True} if bool(max_val) else {"max": False}
    return None


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

            direction = metric_direction(metric, obj, process_rules=step_rules)
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


def _clone_gate_config(gates):
    if not isinstance(gates, dict):
        return {}
    cloned = {}
    for metric, stats in gates.items():
        key = str(metric or "").strip()
        if not key:
            continue
        if isinstance(stats, dict):
            cloned[key] = dict(stats)
        else:
            cloned[key] = stats
    return cloned


def _merge_existing_tolerance_overrides(derived_gates, existing_gates):
    """Keep explicit per-metric tolerance overrides from existing gate config."""
    if not isinstance(derived_gates, dict):
        return derived_gates
    merged = _clone_gate_config(derived_gates)
    if not isinstance(existing_gates, dict):
        return merged
    for metric, derived_stats in list(merged.items()):
        if not isinstance(derived_stats, dict):
            continue
        existing_stats = existing_gates.get(metric)
        if not isinstance(existing_stats, dict):
            continue
        if "tol" not in existing_stats:
            continue
        try:
            tol_val = abs(float(existing_stats.get("tol")))
        except (TypeError, ValueError):
            continue
        if tol_val <= 0.0:
            continue
        derived_stats["tol"] = round_value(tol_val)
    return merged


def _success_gate_metric_alias_keys(metric):
    metric_key = str(metric or "").strip()
    if metric_key in {"x_axis", "xAxis_offset_abs", "xAxis_offset"}:
        return ("x_axis", "xAxis_offset_abs", "xAxis_offset")
    if metric_key in {"y_axis", "yAxis_offset_abs", "yAxis_offset"}:
        return ("y_axis", "yAxis_offset_abs", "yAxis_offset")
    return (metric_key,)


def _apply_success_gate_tolerance_additions(derived_gates, step_cfg):
    if not isinstance(derived_gates, dict):
        return derived_gates
    if not isinstance(step_cfg, dict):
        return derived_gates
    tol_add_cfg = step_cfg.get("success_gate_tol_add")
    if not isinstance(tol_add_cfg, dict) or not tol_add_cfg:
        return derived_gates

    adjusted = _clone_gate_config(derived_gates)
    for metric_name, tol_add in tol_add_cfg.items():
        try:
            tol_add_num = float(tol_add)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(tol_add_num) or abs(float(tol_add_num)) <= 0.0:
            continue

        stats_key = None
        for alias_key in _success_gate_metric_alias_keys(metric_name):
            if isinstance(adjusted.get(alias_key), dict):
                stats_key = alias_key
                break
        if not stats_key:
            continue

        stats = adjusted.get(stats_key)
        if not isinstance(stats, dict):
            continue
        try:
            tol_base = abs(float(stats.get("tol")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(tol_base):
            continue
        stats["tol"] = round_value(max(0.0, float(tol_base) + float(tol_add_num)))
    return adjusted


def _normalize_locked_success_gates_by_mode(raw):
    if not isinstance(raw, dict):
        return {}
    normalized = {}
    for mode_key, gates in raw.items():
        mode = LOCKED_SUCCESS_GATE_MODE_ALIASES.get(str(mode_key or "").strip().lower())
        if mode not in (VISION_MODE_ARUCO, VISION_MODE_CYAN):
            continue
        cloned = _clone_gate_config(gates)
        if cloned:
            normalized[mode] = cloned
    return normalized


def _active_success_gate_mode():
    try:
        mode = str(_world_model_active_vision_mode() or "").strip().lower()
    except Exception:
        mode = ""
    mode = LOCKED_SUCCESS_GATE_MODE_ALIASES.get(mode)
    if mode in (VISION_MODE_ARUCO, VISION_MODE_CYAN):
        return mode
    return VISION_MODE_ARUCO


def _lite_only_aruco_experiment_active():
    if not bool(GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT):
        return False
    return _active_success_gate_mode() == VISION_MODE_ARUCO


def _log_lite_only_aruco_gatecheck_pass(world, step, phase):
    if world is None:
        return
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    phase_key = str(phase or "run").strip().upper() or "RUN"
    frame_id = int(getattr(world, "_frame_id", 0) or 0)
    sig = (step_key, phase_key, frame_id)
    if getattr(world, "_last_lite_only_aruco_pass_log_sig", None) == sig:
        return
    world._last_lite_only_aruco_pass_log_sig = sig
    print(
        format_headline(
            (
                "[TEMPORARY ARUCO LITE-ONLY EXPERIMENT] "
                f"{step_key} {phase_key}: LITE GATE PASSED - SKIPPING FULL GATE CHECK"
            ),
            COLOR_MAGENTA_BRIGHT,
        )
    )


def update_process_model_from_demos(
    logs,
    path=PROCESS_MODEL_FILE,
    *,
    force_rederive_success_gates=False,
):
    """
    Rebuild process gates from demo logs.

    By default, success-gate lock flags are respected. When
    `force_rederive_success_gates=True`, demo-derived success gates overwrite
    locked success gates for the active mode so startup always refreshes numeric
    gates from the latest demos.
    """
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
    wall_visible_only_steps = _visible_only_success_gate_steps(wall_step_rules)
    step_rules = {}
    if isinstance(wall_step_rules, dict):
        step_rules.update(wall_step_rules)
    if isinstance(steps, dict):
        step_rules.update(steps)
    visible_only_success_steps = set(wall_visible_only_steps)
    visible_only_success_steps.update(_visible_only_success_gate_steps(step_rules))
    for step_name in visible_only_success_steps:
        cfg = step_rules.get(step_name)
        if not isinstance(cfg, dict):
            cfg = {}
            step_rules[step_name] = cfg
        # Visibility-only steps must retain only `visible` in success gates.
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
    all_steps.update(visible_only_success_steps)
    all_steps = {
        name
        for name in all_steps
        if normalize_step_label(name) not in RETIRED_PROCESS_STEPS
    }
    active_gate_mode = _active_success_gate_mode()
    force_success_rederive = bool(force_rederive_success_gates)

    for obj in all_steps:
        cfg = steps.setdefault(obj, {})

        lock_all_gates = bool(cfg.get("lock_gates")) if isinstance(cfg, dict) else False
        lock_start_gates = lock_all_gates or (bool(cfg.get("lock_start_gates")) if isinstance(cfg, dict) else False)
        lock_success_gates = lock_all_gates or (bool(cfg.get("lock_success_gates")) if isinstance(cfg, dict) else False)
        effective_lock_success_gates = bool(lock_success_gates and not force_success_rederive)
        locked_success_by_mode = _normalize_locked_success_gates_by_mode(
            cfg.get("locked_success_gates_by_mode")
        )
        if locked_success_by_mode:
            cfg["locked_success_gates_by_mode"] = locked_success_by_mode
        else:
            cfg.pop("locked_success_gates_by_mode", None)

        if obj in start_gates and not _no_start_gates_step(obj):
            if not lock_start_gates or not isinstance(cfg.get("start_gates"), dict) or not cfg.get("start_gates"):
                cfg["start_gates"] = start_gates[obj]
        else:
            if not lock_start_gates:
                cfg.pop("start_gates", None)

        derived_success = success_gates.get(obj)
        if isinstance(derived_success, dict) and force_success_rederive and lock_success_gates:
            existing_success = cfg.get("success_gates")
            derived_success = _merge_existing_tolerance_overrides(derived_success, existing_success)
            if locked_success_by_mode:
                active_locked_existing = locked_success_by_mode.get(active_gate_mode)
                derived_success = _merge_existing_tolerance_overrides(
                    derived_success,
                    active_locked_existing,
                )
        if isinstance(derived_success, dict):
            derived_success = _apply_success_gate_tolerance_additions(derived_success, cfg)
        if locked_success_by_mode:
            active_locked = locked_success_by_mode.get(active_gate_mode)
            if isinstance(derived_success, dict):
                if (
                    force_success_rederive
                    or not effective_lock_success_gates
                    or not isinstance(active_locked, dict)
                    or not active_locked
                ):
                    locked_success_by_mode[active_gate_mode] = _clone_gate_config(derived_success)
                    active_locked = locked_success_by_mode.get(active_gate_mode)
            elif not effective_lock_success_gates and not force_success_rederive:
                locked_success_by_mode.pop(active_gate_mode, None)
                active_locked = locked_success_by_mode.get(active_gate_mode)
            if locked_success_by_mode:
                cfg["locked_success_gates_by_mode"] = locked_success_by_mode
            else:
                cfg.pop("locked_success_gates_by_mode", None)
            if isinstance(active_locked, dict) and active_locked:
                cfg["success_gates"] = _clone_gate_config(active_locked)
            elif not effective_lock_success_gates and not force_success_rederive:
                cfg.pop("success_gates", None)
        else:
            if isinstance(derived_success, dict):
                if (
                    force_success_rederive
                    or not effective_lock_success_gates
                    or not isinstance(cfg.get("success_gates"), dict)
                    or not cfg.get("success_gates")
                ):
                    cfg["success_gates"] = derived_success
            else:
                if not effective_lock_success_gates and not force_success_rederive:
                    cfg.pop("success_gates", None)
        if obj in SUCCESS_GATE_EXCLUDE_ANGLE_STEPS and isinstance(cfg.get("success_gates"), dict):
            cfg["success_gates"].pop("angle_abs", None)
        obj_key = normalize_step_label(obj)
        if not lock_start_gates:
            required_start_metrics = PROCESS_START_METRICS_BY_STEP.get(obj_key)
            if isinstance(required_start_metrics, (list, tuple)) and required_start_metrics:
                current_start = cfg.get("start_gates")
                if not isinstance(current_start, dict):
                    current_start = {}
                success_cfg = cfg.get("success_gates")
                if not isinstance(success_cfg, dict):
                    success_cfg = {}
                for metric in required_start_metrics:
                    metric_key = str(metric)
                    if metric_key in current_start and isinstance(current_start.get(metric_key), dict):
                        continue
                    fallback_gate = _bool_start_gate_from_success_stats(success_cfg.get(metric_key))
                    if isinstance(fallback_gate, dict):
                        current_start[metric_key] = fallback_gate
                if current_start:
                    cfg["start_gates"] = current_start
        if obj_key in visible_only_success_steps:
            derived_visible_gate = None
            derived_cfg = success_gates.get(obj)
            if isinstance(derived_cfg, dict):
                candidate = derived_cfg.get("visible")
                if isinstance(candidate, dict):
                    derived_visible_gate = dict(candidate)
            if not isinstance(derived_visible_gate, dict):
                gates = cfg.get("success_gates")
                if isinstance(gates, dict):
                    candidate = gates.get("visible")
                    if isinstance(candidate, dict):
                        derived_visible_gate = dict(candidate)
            if not isinstance(derived_visible_gate, dict):
                derived_visible_gate = {"min": True}
            cfg["success_gates"] = {"visible": derived_visible_gate}

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
    y_axis = brick.get("y_axis")
    if y_axis is None:
        y_axis = brick.get("offset_y")
    history.append(
        {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "visible": bool(brick.get("visible")),
            "dist": float(brick.get("dist", 0.0) or 0.0),
            "angle": float(brick.get("angle", 0.0) or 0.0),
            "x_axis": float(x_axis or 0.0),
            "offset_x": float(brick.get("offset_x", x_axis if x_axis is not None else 0.0) or 0.0),
            "y_axis": float(y_axis or 0.0),
            "offset_y": float(brick.get("offset_y", y_axis if y_axis is not None else 0.0) or 0.0),
            "confidence": float(brick.get("confidence", 0.0) or 0.0),
            "brick_above": brick.get("brickAbove"),
            "brick_below": brick.get("brickBelow"),
        }
    )
    if len(history) > 120:
        del history[:-120]


def update_world_from_vision(world, vision, log=True):
    _apply_post_action_observe_delay(world)
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

    vision_backend = _vision_backend_name(vision)
    world._vision_backend = vision_backend
    world._raw_visibility_fallback_confidence_min = _raw_visibility_fallback_min_confidence_pct(vision)
    if vision_backend == "cyan":
        world._cyan_raw_fallback_confidence_min = world._raw_visibility_fallback_confidence_min
    else:
        world._cyan_raw_fallback_confidence_min = None

    found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
    floor_lift_mm = getattr(vision, "floor_lift_mm", None)
    floor_lift_quality = getattr(vision, "floor_lift_quality", 0.0)
    try:
        floor_lift_mm = float(floor_lift_mm) if floor_lift_mm is not None else None
    except (TypeError, ValueError):
        floor_lift_mm = None
    try:
        floor_lift_quality = float(floor_lift_quality)
    except (TypeError, ValueError):
        floor_lift_quality = 0.0

    def _update_world_vision_safe(
        found_val,
        dist_val,
        angle_val,
        conf_val,
        offset_x_val,
        cam_h_val,
        brick_above_val,
        brick_below_val,
    ):
        try:
            world.update_vision(
                found_val,
                dist_val,
                angle_val,
                conf_val,
                offset_x_val,
                cam_h_val,
                brick_above_val,
                brick_below_val,
                floor_lift_mm=floor_lift_mm,
                floor_lift_quality=floor_lift_quality,
            )
        except TypeError:
            world.update_vision(
                found_val,
                dist_val,
                angle_val,
                conf_val,
                offset_x_val,
                cam_h_val,
                brick_above_val,
                brick_below_val,
            )
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
        "offset_y": float(cam_h),
        "y_axis": float(cam_h),
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
        raw_history = getattr(world, "_raw_brick_visibility_history", None)
        if raw_history is None:
            raw_history = []
            setattr(world, "_raw_brick_visibility_history", raw_history)
        raw_history.append(
            {
                "frame_id": int(getattr(world, "_frame_id", 0) or 0),
                "timestamp": time.time(),
                "found": bool(frame.get("found")),
                "conf": float(frame.get("conf", 0.0) or 0.0),
            }
        )
        if len(raw_history) > 120:
            del raw_history[:-120]
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
        # Numeric pose fields benefit from smoothing, but stack booleans already have a
        # dedicated confirmation filter in telemetry_brick. Feed the latest frame's
        # above/below observations to reduce lag vs the green marker overlays.
        latest_stack_above = bool(frame.get("brick_above"))
        latest_stack_below = bool(frame.get("brick_below"))
        _update_world_vision_safe(
            avg["found"],
            avg["dist"],
            avg["angle"],
            avg["conf"],
            avg["offset_x"],
            avg["cam_h"],
            latest_stack_above,
            latest_stack_below,
        )
        if log:
            current_obj = getattr(world, "step_state", None)
            current_name = getattr(current_obj, "value", None)
            if normalize_step_label(current_name) == "EXIT_WALL":
                pass
            elif not getattr(world, "suppress_brick_state_log", False):
                pass
    elif len(buffer) >= telemetry_brick.BRICK_SMOOTH_FRAMES:
        if _allow_raw_visibility_fallback(vision, frame):
            world._brick_inconsistent = False
            world._last_obs_note = f"{vision_backend} raw visibility fallback"
            _update_world_vision_safe(
                bool(frame.get("found")),
                frame.get("dist", 0.0),
                frame.get("angle", 0.0),
                frame.get("conf", 0.0),
                frame.get("offset_x", 0.0),
                frame.get("cam_h", 0.0),
                bool(frame.get("brick_above")),
                bool(frame.get("brick_below")),
            )
        else:
            # Mark inconsistent snapshot: force confidence to 0 and visible false until we stabilize.
            telemetry_brick.update_stack_flags_from_raw(
                world,
                bool(found),
                bool(brick_above),
                bool(brick_below),
            )
            world._brick_inconsistent = True
            world.brick["visible"] = False
            world.brick["confidence"] = 0.0
            world.brick["dist"] = 0.0
            world.brick["angle"] = 0.0
            world.brick["offset_x"] = 0.0
            world.brick["x_axis"] = 0.0
            world.brick["offset_y"] = 0.0
            world.brick["y_axis"] = 0.0
            world.brick["inCrosshairs"] = False
    else:
        if _allow_raw_visibility_fallback(vision, frame):
            world._brick_inconsistent = False
            world._last_obs_note = f"{vision_backend} raw visibility fallback"
            _update_world_vision_safe(
                bool(frame.get("found")),
                frame.get("dist", 0.0),
                frame.get("angle", 0.0),
                frame.get("conf", 0.0),
                frame.get("offset_x", 0.0),
                frame.get("cam_h", 0.0),
                bool(frame.get("brick_above")),
                bool(frame.get("brick_below")),
            )
        else:
            telemetry_brick.update_stack_flags_from_raw(
                world,
                bool(found),
                bool(brick_above),
                bool(brick_below),
            )
            world._brick_inconsistent = True
            world.brick["visible"] = False
            world.brick["confidence"] = 0.0
            world.brick["dist"] = 0.0
            world.brick["angle"] = 0.0
            world.brick["offset_x"] = 0.0
            world.brick["x_axis"] = 0.0
            world.brick["offset_y"] = 0.0
            world.brick["y_axis"] = 0.0
            world.brick["inCrosshairs"] = False
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
    if fresh_input_frame:
        # Track every fresh frame's final brick snapshot, including visible=false,
        # so lite precheck can advance on disappearance-style success gates.
        _record_smoothed_frame_snapshot(world)
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
STREAM_PINK = (180, 105, 255)
STREAM_BLUE_BRIGHT = (255, 120, 0)


def _gate_requirement_text(stats):
    if not isinstance(stats, dict):
        return ""
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        return _fmt_gate_value(min_val)
    if isinstance(max_val, bool):
        return _fmt_gate_value(max_val)
    target = stats.get("target")
    tol = stats.get("tol")
    if isinstance(target, bool):
        return _fmt_gate_value(target)
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


def _format_align_result_observation(metric_name, curr_err, prev_err, *, overshot=False):
    """
    Single canonical formatter for alignment result observations.
    Ensures all operator logs use one consistent sentence shape.
    """
    try:
        curr_err_f = float(curr_err)
        prev_err_f = float(prev_err)
    except (TypeError, ValueError):
        return None, None, None

    metric_label = str(metric_name or "").strip() or "metric"
    result_prefix_plain = "Result:"
    result_prefix_colored = f"{COLOR_BLUE_BRIGHT}Result:{COLOR_RESET}"
    if overshot:
        body = f"overshot target; previous {metric_label}={prev_err_f:+.2f}mm"
        color = COLOR_RED
        plain = f"{result_prefix_plain} {body}"
        colored = f"{result_prefix_colored} {body}"
        return plain, colored, color

    delta_local = abs(float(prev_err_f)) - abs(float(curr_err_f))
    if abs(delta_local) < 0.05:
        body = f"~unchanged; previous {metric_label}={prev_err_f:+.2f}mm"
        color = COLOR_YELLOW
        plain = f"{result_prefix_plain} {body}"
        colored = f"{result_prefix_colored} {body}"
        return plain, colored, color
    elif delta_local > 0.0:
        delta_text = f"{delta_local:+.2f}mm"
        body_tail = f"better than previous {metric_label}={prev_err_f:+.2f}mm"
        color = COLOR_GREEN
        plain = f"{result_prefix_plain} {delta_text} {body_tail}"
        colored = f"{result_prefix_colored} {color}{delta_text}{COLOR_RESET} {body_tail}"
        return plain, colored, color
    else:
        delta_text = f"{delta_local:+.2f}mm"
        body_tail = f"worse than previous {metric_label}={prev_err_f:+.2f}mm"
        color = COLOR_RED
        plain = f"{result_prefix_plain} {delta_text} {body_tail}"
        colored = f"{result_prefix_colored} {color}{delta_text}{COLOR_RESET} {body_tail}"
    return plain, colored, color


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
            if metric in ("xAxis_offset_abs", "yAxis_offset_abs"):
                return max(0.0, abs(numeric - target_num) - tol_num)
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


def _auto_diag_focus_snapshot_colored(focus, *, include_requirement=True):
    if not isinstance(focus, dict):
        return f"{COLOR_WHITE}none{COLOR_RESET}"
    metric = focus.get("metric")
    if not metric:
        return f"{COLOR_WHITE}none{COLOR_RESET}"
    value_text = _fmt_metric_value_with_unit(metric, focus.get("value"))
    head = f"{COLOR_WHITE}{metric}={value_text}{COLOR_RESET}"
    if not include_requirement:
        return head
    requirement = _gate_requirement_text(focus.get("stats") or {})
    if not requirement:
        return head
    req_color = COLOR_WHITE
    matched = focus.get("matched")
    if matched is True:
        req_color = COLOR_GREEN
    elif matched is False:
        req_color = COLOR_RED
    return f"{head} ({req_color}{requirement}{COLOR_RESET})"


def _auto_diag_entries_snapshot_colored(entries, step, *, include_requirement=True):
    if not isinstance(entries, list) or not entries:
        return f"{COLOR_WHITE}none{COLOR_RESET}"
    parts = []
    for entry in entries:
        focus = _auto_diag_focus_from_entry(entry, step)
        if isinstance(focus, dict):
            parts.append(
                _auto_diag_focus_snapshot_colored(
                    focus,
                    include_requirement=include_requirement,
                )
            )
            continue
        metric = entry.get("metric") if isinstance(entry, dict) else None
        if not metric:
            continue
        value = _auto_diag_value_from_entry(entry) if isinstance(entry, dict) else None
        value_text = _fmt_metric_value_with_unit(metric, value)
        parts.append(f"{COLOR_WHITE}{metric}={value_text}{COLOR_RESET}")
    return ", ".join(parts) if parts else f"{COLOR_WHITE}none{COLOR_RESET}"


def _auto_diag_lite_gate_detail(world, step):
    if world is None:
        return None
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    status = getattr(world, "_gatecheck_status", None)
    if not isinstance(status, dict):
        status = {}

    mode = str(status.get("mode") or getattr(world, "_gatecheck_mode", "traditional") or "traditional").strip().lower()
    checks = max(0, int(status.get("checks", 0) or 0))
    truth_ok = bool(status.get("truth_ok", False))
    lite_required = max(
        0,
        int(
            status.get(
                "lite_required",
                getattr(world, "_gatecheck_lite_required", 0),
            )
            or 0
        ),
    )
    lite_collected = max(
        0,
        int(
            status.get(
                "lite_collected",
                getattr(world, "_gatecheck_lite_collected", 0),
            )
            or 0
        ),
    )
    lite_passed = getattr(world, "_gatecheck_lite_passed", None)

    use_lite_measurement = bool(
        mode == "lite"
        or (lite_required > 0 and lite_passed is False)
    )
    measurement = None
    if use_lite_measurement:
        measurement, lite_meta = _lite_gate_measurement_for_step(world, step_key)
        if isinstance(lite_meta, dict):
            if lite_required <= 0:
                lite_required = max(0, int(lite_meta.get("required", 0) or 0))
            if lite_collected <= 0:
                lite_collected = max(0, int(lite_meta.get("collected", 0) or 0))
    rows = _success_gate_state_rows(
        world,
        step_key,
        measurement=measurement if isinstance(measurement, dict) else None,
    )
    if not rows:
        rows = _success_gate_state_rows(world, step_key)

    if mode == "lite":
        mode_text = "lite"
    elif lite_required > 0 and lite_passed is True:
        mode_text = "traditional (lite passed)"
    elif lite_required > 0 and lite_passed is False:
        mode_text = "lite pending"
    else:
        mode_text = mode or "traditional"

    if lite_required > 0:
        lite_text = f"{lite_collected}/{lite_required}"
    else:
        lite_text = "n/a"
    truth_text = "pass" if truth_ok else "wait"

    plain = f"lite gatecheck: mode={mode_text}, lite={lite_text}, sample={truth_text}, checks={checks}"
    colored = (
        f"{COLOR_GRAY}lite gatecheck: mode={mode_text}, lite={lite_text}, "
        f"sample={truth_text}, checks={checks}{COLOR_RESET}"
    )
    if bool(getattr(world, "_lite_gate_visible_false_confident_seen", False)):
        plain += "; confident_visible_recent=true"
        colored += f"{COLOR_GRAY}; confident_visible_recent=true{COLOR_RESET}"

    if rows:
        row_plain_parts = []
        row_colored_parts = []
        for row in rows:
            metric = str(row.get("metric") or "?")
            value_text = str(row.get("value_text") or "n/a")
            gate_text = str(row.get("gate_text") or "n/a")
            status_text = str(row.get("status") or "N/A").upper()
            cmp_text = "=" if status_text == "PASS" else "!=" if status_text == "FAIL" else "?"
            if status_text == "PASS":
                value_color = COLOR_GREEN
            elif status_text == "FAIL":
                value_color = COLOR_RED
            else:
                value_color = COLOR_GRAY
            row_plain_parts.append(f"{metric} ({cmp_text} Expected {gate_text} & saw {value_text})")
            row_colored_parts.append(
                f"{COLOR_GRAY}{metric} ({cmp_text} Expected {gate_text} & saw {COLOR_RESET}{value_color}({value_text}){COLOR_RESET}{COLOR_GRAY}){COLOR_RESET}"
            )
        plain += "; gates: " + "; ".join(row_plain_parts)
        colored += f"{COLOR_GRAY}; gates:{COLOR_RESET} " + f"{COLOR_GRAY}; {COLOR_RESET}".join(row_colored_parts)
    return {
        "plain": plain,
        "colored": colored,
    }


def _result_lite_gate_detail(world, step):
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    measurement, _lite_meta = _lite_gate_measurement_for_step(world, step_key)
    rows = _success_gate_state_rows(
        world,
        step_key,
        measurement=measurement if isinstance(measurement, dict) else None,
    )

    if not rows:
        detail = _auto_diag_lite_gate_detail(world, step)
        if not isinstance(detail, dict):
            return None
        plain = detail.get("plain")
        colored = detail.get("colored") or plain
        if not plain and not colored:
            return None
        token = "lite gatecheck:"
        pink_token = f"{COLOR_PINK}lite{COLOR_RESET}{COLOR_GRAY} gatecheck:"
        colored_text = str(colored).replace(token, pink_token, 1)
        return {
            "plain": plain,
            "colored": colored_text,
        }

    row_plain_parts = []
    row_colored_parts = []
    for row in rows:
        metric = str(row.get("metric") or "?")
        value_text = str(row.get("value_text") or "n/a")
        gate_text = str(row.get("gate_text") or "n/a")
        status_text = str(row.get("status") or "N/A").upper()
        cmp_text = "=" if status_text == "PASS" else "!=" if status_text == "FAIL" else "?"
        if status_text == "PASS":
            value_color = COLOR_GREEN
        elif status_text == "FAIL":
            value_color = COLOR_RED
        else:
            value_color = COLOR_GRAY

        gate_rhs = gate_text[1:] if gate_text.startswith("=") else gate_text
        row_plain_parts.append(f"{metric} ({cmp_text} Expected {gate_rhs} & saw {value_text})")
        row_colored_parts.append(
            f"{COLOR_WHITE}{metric} ({cmp_text} Expected {gate_rhs} & saw {COLOR_RESET}"
            f"{value_color}({value_text}){COLOR_RESET}{COLOR_WHITE}){COLOR_RESET}"
        )

    plain = "lite gatecheck: " + "; ".join(row_plain_parts)
    colored_text = (
        f"{COLOR_WHITE}lite gatecheck: {COLOR_RESET}"
        + f"{COLOR_WHITE}; {COLOR_RESET}".join(row_colored_parts)
    )
    return {
        "plain": plain,
        "colored": colored_text,
    }


def _switch_metric_to_success_metric(metric_name_local):
    metric_key = str(metric_name_local or "").strip().lower()
    return {
        "x_err": "xAxis_offset_abs",
        "y_err": "yAxis_offset_abs",
        "dist_err": "dist",
    }.get(metric_key, metric_key)


def _lite_gate_measurement_for_step(world, step):
    required = lite_gate_unique_frames(step)
    if required is None:
        return None, {
            "enabled": False,
            "required": 0,
            "collected": 0,
            "source": "disabled",
        }
    frames = _latest_unique_smoothed_frames(world, required)
    collected = len(frames)
    if collected < int(required):
        return None, {
            "enabled": True,
            "required": int(required),
            "collected": int(collected),
            "source": "pending",
        }
    measurement = _average_smoothed_frames(
        frames,
        step=step,
        process_rules=getattr(world, "process_rules", None),
    )
    if not isinstance(measurement, dict):
        return None, {
            "enabled": True,
            "required": int(required),
            "collected": int(collected),
            "source": "unavailable",
        }
    return measurement, {
        "enabled": True,
        "required": int(required),
        "collected": int(collected),
        "source": "lite",
    }


def _success_gate_state_rows(world, step, *, measurement=None):
    step_key = normalize_step_label(step)
    process_rules = getattr(world, "process_rules", None) or {}
    step_cfg = process_rules.get(step_key, {}) if isinstance(process_rules, dict) else {}
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    if not isinstance(success_gates, dict) or not success_gates:
        return []

    entry_map = {}
    entries = _success_gate_entries(world, step_key)
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        metric = entry.get("metric")
        if metric:
            entry_map[str(metric)] = entry

    rows = []
    for metric, stats in success_gates.items():
        metric_key = str(metric)
        stats_dict = dict(stats) if isinstance(stats, dict) else {}
        value = None
        if isinstance(measurement, dict):
            value = gate_utils.metric_value_from_measurement(measurement, metric_key)
        fallback_entry = entry_map.get(metric_key)
        if value is None:
            if isinstance(fallback_entry, dict):
                value = _auto_diag_value_from_entry(fallback_entry)

        if value is None:
            matched = None
            distance = None
        else:
            direction = metric_direction(metric_key, step_key)
            matched = bool(_gate_entry_matches(metric_key, value, stats_dict, direction))
            if (
                metric_key == "visible"
                and gate_utils.bool_gate_target(stats_dict) is False
                and isinstance(fallback_entry, dict)
                and bool(fallback_entry.get("confident_visible_recent"))
            ):
                matched = False
            distance = _gate_distance_to_success(metric_key, value, stats_dict, direction)
        requirement = _gate_requirement_text(stats_dict) or "n/a"
        value_text = _fmt_metric_value_with_unit(metric_key, value)
        if (
            metric_key == "visible"
            and gate_utils.bool_gate_target(stats_dict) is False
            and isinstance(fallback_entry, dict)
            and bool(fallback_entry.get("confident_visible_recent"))
        ):
            value_text = f"{value_text} (recent_confident_visible=true)"
        status = "PASS" if matched is True else "FAIL" if matched is False else "N/A"
        rows.append(
            {
                "metric": metric_key,
                "value_text": value_text,
                "gate_text": requirement,
                "status": status,
                "distance": distance,
            }
        )
    return rows


def _lite_measurement_rows_all_pass(world, step):
    if world is None:
        return False
    step_key = normalize_step_label(step) or str(step) or "UNKNOWN"
    measurement, _meta = _lite_gate_measurement_for_step(world, step_key)
    if not isinstance(measurement, dict):
        return False
    rows = _success_gate_state_rows(world, step_key, measurement=measurement)
    if not rows:
        return False
    return all(str(row.get("status") or "").upper() == "PASS" for row in rows)


def _gap_switch_lite_gate_reassurance(world, step, metric_name_local):
    step_key = normalize_step_label(step)
    measurement, lite_meta = _lite_gate_measurement_for_step(world, step_key)
    rows = _success_gate_state_rows(world, step_key, measurement=measurement)
    if not rows:
        plain = "Reassurance: no success gates configured for this step."
        colored = f"{COLOR_YELLOW}{plain}{COLOR_RESET}"
        return {
            "metric_plain": plain,
            "metric_colored": colored,
            "snapshot_plain_lines": ["Success gate state/gate snapshot: none"],
            "snapshot_colored_lines": [f"{COLOR_WHITE}Success gate state/gate snapshot: none{COLOR_RESET}"],
        }

    metric_key = _switch_metric_to_success_metric(metric_name_local)
    target_row = None
    for row in rows:
        if row.get("metric") == metric_key:
            target_row = row
            break
    if target_row is None and metric_key == "dist":
        for row in rows:
            if row.get("metric") == "distance":
                target_row = row
                break

    if lite_meta.get("enabled"):
        source_text = (
            f"lite {int(lite_meta.get('collected', 0))}/{int(lite_meta.get('required', 0))} unique frames"
            if isinstance(measurement, dict)
            else f"lite pending {int(lite_meta.get('collected', 0))}/{int(lite_meta.get('required', 0))} unique frames"
        )
    else:
        source_text = "lite disabled"

    if target_row is None:
        metric_plain = (
            f"Reassurance: no success gate configured for switched metric {metric_key} "
            f"({source_text})."
        )
        metric_colored = f"{COLOR_YELLOW}{metric_plain}{COLOR_RESET}"
    else:
        status = str(target_row.get("status") or "N/A")
        if status == "FAIL":
            verdict_plain = "outside success gate"
            verdict_color = COLOR_RED
        elif status == "PASS":
            verdict_plain = "inside success gate"
            verdict_color = COLOR_GREEN
        else:
            verdict_plain = "unknown vs success gate"
            verdict_color = COLOR_YELLOW
        metric_plain = (
            f"Reassurance ({source_text}): {metric_key} is {verdict_plain} "
            f"(state={target_row.get('value_text')}, gate={target_row.get('gate_text')})"
        )
        try:
            outside_mag = float(target_row.get("distance"))
        except (TypeError, ValueError):
            outside_mag = None
        if status == "FAIL" and outside_mag is not None:
            unit = _metric_display_unit(metric_key)
            if unit:
                metric_plain += f"; outside_by={outside_mag:.2f}{unit}"
            else:
                metric_plain += f"; outside_by={outside_mag:.2f}"
        metric_colored = (
            f"{COLOR_WHITE}Reassurance ({source_text}): {COLOR_RESET}"
            f"{COLOR_WHITE}{metric_key}{COLOR_RESET} is {verdict_color}{verdict_plain}{COLOR_RESET} "
            f"{COLOR_WHITE}(state={target_row.get('value_text')}, gate={target_row.get('gate_text')}){COLOR_RESET}"
        )
        if status == "FAIL" and outside_mag is not None:
            unit = _metric_display_unit(metric_key)
            if unit:
                metric_colored += f"{COLOR_WHITE}; outside_by={outside_mag:.2f}{unit}{COLOR_RESET}"
            else:
                metric_colored += f"{COLOR_WHITE}; outside_by={outside_mag:.2f}{COLOR_RESET}"

    snapshot_plain_lines = [f"Success gate state/gate snapshot ({source_text})"]
    snapshot_colored_lines = [
        f"{COLOR_WHITE}Success gate state/gate snapshot ({source_text}){COLOR_RESET}"
    ]
    for row in rows:
        status = str(row.get("status") or "N/A")
        status_color = COLOR_YELLOW
        if status == "PASS":
            status_color = COLOR_GREEN
        elif status == "FAIL":
            status_color = COLOR_RED
        snapshot_plain_lines.append(
            f"{row.get('metric')} state={row.get('value_text')} gate={row.get('gate_text')} status={status}"
        )
        snapshot_colored_lines.append(
            f"{COLOR_WHITE}{row.get('metric')}{COLOR_RESET} "
            f"state={COLOR_WHITE}{row.get('value_text')}{COLOR_RESET} "
            f"gate={COLOR_WHITE}{row.get('gate_text')}{COLOR_RESET} "
            f"status={status_color}{status}{COLOR_RESET}"
        )
    return {
        "metric_plain": metric_plain,
        "metric_colored": metric_colored,
        "snapshot_plain_lines": snapshot_plain_lines,
        "snapshot_colored_lines": snapshot_colored_lines,
    }


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


def _auto_diag_prev_observation_phrase(world, step, focus):
    if world is None or not isinstance(focus, dict):
        return None, None
    step_key = normalize_step_label(step)
    if not step_key:
        return None, None
    focus_map = getattr(world, "_auto_diag_last_pre_focus_by_step", None)
    if not isinstance(focus_map, dict):
        return None, None
    prev = focus_map.get(step_key)
    if not isinstance(prev, dict):
        return None, None

    metric = focus.get("metric")
    prev_metric = prev.get("metric")
    if not metric:
        return None, None
    if prev_metric and prev_metric != metric:
        prev_metric_text = _fmt_metric_value_with_unit(prev_metric, prev.get("value"))
        curr_metric_text = _fmt_metric_value_with_unit(metric, focus.get("value"))
        return (
            "unchanged",
            (
                f"obs switched {prev_metric}={prev_metric_text} → "
                f"{metric}={curr_metric_text}"
            ),
        )

    prev_val = prev.get("value")
    curr_val = focus.get("value")
    if isinstance(prev_val, bool) and isinstance(curr_val, bool):
        prev_bool = bool(prev_val)
        curr_bool = bool(curr_val)
        prev_txt = _fmt_gate_value(prev_bool)
        curr_txt = _fmt_gate_value(curr_bool)
        if prev_bool == curr_bool:
            return "unchanged", f"obs unchanged vs previous ({metric}={curr_txt})"
        prev_distance = prev.get("distance")
        curr_distance = focus.get("distance")
        try:
            if prev_distance is not None and curr_distance is not None:
                delta = float(prev_distance) - float(curr_distance)
                if delta > 1e-6:
                    delta_class = "closer"
                elif delta < -1e-6:
                    delta_class = "backward"
                else:
                    delta_class = "unchanged"
            else:
                delta_class = "unchanged"
        except (TypeError, ValueError):
            delta_class = "unchanged"
        return delta_class, f"obs changed vs previous ({metric}={prev_txt}→{curr_txt})"

    if metric == "visible":
        prev_val = prev.get("value")
        curr_val = focus.get("value")
        if prev_val is None or curr_val is None:
            return None, None
        prev_bool = bool(prev_val)
        curr_bool = bool(curr_val)
        prev_txt = _fmt_gate_value(prev_bool)
        curr_txt = _fmt_gate_value(curr_bool)
        if prev_bool == curr_bool:
            return "unchanged", f"obs unchanged vs previous (visible={curr_txt})"
        if curr_bool and not prev_bool:
            return "closer", f"obs changed vs previous (visible={prev_txt}→{curr_txt})"
        return "backward", f"obs changed vs previous (visible={prev_txt}→{curr_txt})"

    prev_distance = prev.get("distance")
    curr_distance = focus.get("distance")
    if prev_distance is None or curr_distance is None:
        return None, None
    try:
        delta = float(prev_distance) - float(curr_distance)
    except (TypeError, ValueError):
        return None, None
    unit = _metric_display_unit(metric)
    magnitude = f"{abs(delta):.1f}{unit}" if unit else f"{abs(delta):.1f}"
    prev_metric_text = _fmt_metric_value_with_unit(metric, prev.get("value"))
    if abs(delta) <= 1e-6:
        return "unchanged", f"obs unchanged vs previous ({metric}={prev_metric_text})"
    if delta > 0.0:
        return "closer", f"obs {magnitude} better than previous ({metric}={prev_metric_text})"
    return "backward", f"obs {magnitude} worse than previous ({metric}={prev_metric_text})"


def _ansi_to_stream_segments(colored_text):
    text = str(colored_text or "")
    if not text:
        return []
    color_map = {
        COLOR_GREEN: STREAM_GREEN,
        COLOR_RED: STREAM_RED,
        COLOR_WHITE: STREAM_WHITE,
        COLOR_BLUE_BRIGHT: STREAM_BLUE_BRIGHT,
        COLOR_ORANGE_BRIGHT: STREAM_ORANGE,
        COLOR_GRAY: STREAM_GRAY,
        COLOR_PINK: STREAM_PINK,
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


def _split_stream_segments_lines(segments):
    if not isinstance(segments, list):
        return []
    lines = []
    current = []
    for seg in segments:
        if not isinstance(seg, (tuple, list)) or not seg:
            continue
        txt = str(seg[0])
        color = seg[1] if len(seg) > 1 else STREAM_WHITE
        parts = txt.split("\n")
        for idx, part in enumerate(parts):
            if part:
                current.append((part, color))
            if idx < (len(parts) - 1):
                lines.append(current)
                current = []
    if current:
        lines.append(current)
    return [line for line in lines if isinstance(line, list) and line]


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
    action_index=None,
    pre_entries=None,
):
    step_key = normalize_step_label(step)
    post_entries = _success_gate_entries(world, step)
    pre_entries_list = [entry for entry in (pre_entries or []) if isinstance(entry, dict)]
    selected_pre = pre_focus if isinstance(pre_focus, dict) else None
    if selected_pre is None:
        if pre_entries_list:
            selected_pre = _select_auto_diag_focus(pre_entries_list, step)
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

    single_boolean_gate = _is_single_boolean_success_gate(post_entries)
    if isinstance(selected_pre, dict):
        if single_boolean_gate:
            pre_snapshot = _auto_diag_focus_observation(selected_pre)
            pre_obs_text_colored = (
                f"{COLOR_WHITE}{pre_snapshot}{COLOR_RESET}" if pre_snapshot != "none" else f"{COLOR_WHITE}none{COLOR_RESET}"
            )
        else:
            pre_snapshot = _auto_diag_focus_snapshot(selected_pre, include_requirement=True)
            pre_obs_text_colored = _auto_diag_focus_snapshot_colored(
                selected_pre,
                include_requirement=True,
            )
    else:
        pre_snapshot = str(pre_gate_text or "none")
        if (not pre_snapshot or pre_snapshot.strip().lower() == "none") and pre_entries_list:
            if single_boolean_gate:
                pre_snapshot = _auto_gate_snapshot_text(world, step_key, include_requirements=False)
                pre_obs_text_colored = _auto_diag_entries_snapshot_colored(
                    pre_entries_list,
                    step,
                    include_requirement=False,
                )
            else:
                pre_snapshot = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                pre_obs_text_colored = _auto_diag_entries_snapshot_colored(
                    pre_entries_list,
                    step,
                    include_requirement=True,
                )
        else:
            pre_obs_text_colored = f"{COLOR_WHITE}{pre_snapshot}{COLOR_RESET}"
    post_obs = _auto_diag_focus_observation(post_focus)
    post_tail = f" ({post_obs})" if post_obs and post_obs != "none" else ""
    post_tail_colored = (
        f" ({COLOR_WHITE}{post_obs}{COLOR_RESET})" if post_obs and post_obs != "none" else ""
    )

    pre_distance = selected_pre.get("distance") if isinstance(selected_pre, dict) else None
    post_distance = post_focus.get("distance") if isinstance(post_focus, dict) else None
    delta_plain, delta_colored, delta_class = _auto_diag_delta_phrase(metric, pre_distance, post_distance)

    action_plain = str(action_text or "HOLD")
    action_colored = f"{COLOR_ORANGE_BRIGHT}{action_plain}{COLOR_RESET}"
    bool_focus_metrics = {"visible", "brick_above", "brick_below", "brickAbove", "brickBelow"}
    compact_boolean_delta = bool(
        step_key in AUTO_DIAG_COMPACT_BOOLEAN_DELTA_STEPS
        and isinstance(post_focus, dict)
        and str(post_focus.get("metric") or "") in bool_focus_metrics
    )

    if isinstance(success_override, bool):
        success_hit = bool(success_override)
    else:
        success_hit = False
        matched_values = []
        for entry in post_entries or []:
            focus = _auto_diag_focus_from_entry(entry, step)
            if isinstance(focus, dict) and isinstance(focus.get("matched"), bool):
                matched_values.append(bool(focus.get("matched")))
        if matched_values:
            success_hit = all(matched_values)

    if single_boolean_gate:
        gate_status = getattr(world, "_gatecheck_status", None)
        if not isinstance(gate_status, dict):
            gate_status = {}
        lite_required = max(0, int(gate_status.get("lite_required", 0) or 0))
        lite_collected = max(0, int(gate_status.get("lite_collected", 0) or 0))
        truth_ok = bool(gate_status.get("truth_ok", False))
        gate_mode = str(gate_status.get("mode") or "").strip().lower()
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
        awaiting_confirmation = bool(post_matched) and not bool(success_hit)
        if awaiting_confirmation and gate_mode == "lite" and lite_required > 0:
            awaiting_confirmation = lite_collected < lite_required or not truth_ok
        if awaiting_confirmation:
            outcome_plain = f"resulting in gate matched; awaiting confirmation{post_tail}"
            outcome_colored = (
                f"resulting in {COLOR_YELLOW}gate matched; awaiting confirmation{COLOR_RESET}{post_tail_colored}"
            )
        elif success_hit:
            outcome_plain = f"resulting in meeting the success gates{post_tail}"
            outcome_colored = (
                f"resulting in {COLOR_GREEN}meeting the success gates{COLOR_RESET}{post_tail_colored}"
            )
        else:
            outcome_plain = f"resulting in NOT meeting the success gates{post_tail}"
            outcome_colored = (
                f"resulting in {COLOR_RED}NOT{COLOR_RESET} meeting the success gates{post_tail_colored}"
            )
    else:
        if compact_boolean_delta:
            if success_hit:
                outcome_plain = f"resulting in meeting the success gates{post_tail}"
                outcome_colored = (
                    f"resulting in {COLOR_GREEN}meeting the success gates{COLOR_RESET}{post_tail_colored}"
                )
            else:
                outcome_plain = f"resulting in NOT meeting the success gates{post_tail}"
                outcome_colored = (
                    f"resulting in {COLOR_RED}NOT{COLOR_RESET} meeting the success gates{post_tail_colored}"
                )
        else:
            outcome_plain = f"getting us {delta_plain}{post_tail}"
            outcome_colored = f"getting us {delta_colored}{post_tail_colored}"

    obs_delta_class, obs_delta_text = _auto_diag_prev_observation_phrase(world, step, selected_pre)
    if obs_delta_text:
        if delta_class in (None, "unknown") and obs_delta_class:
            delta_class = obs_delta_class
        obs_note_plain = f"; {obs_delta_text}"
        if obs_delta_class == "closer":
            obs_note_colored = f"; {COLOR_GREEN}{obs_delta_text}{COLOR_RESET}"
        elif obs_delta_class == "backward":
            obs_note_colored = f"; {COLOR_RED}{obs_delta_text}{COLOR_RESET}"
        else:
            obs_note_colored = f"; {COLOR_WHITE}{obs_delta_text}{COLOR_RESET}"
        outcome_plain += obs_note_plain
        outcome_colored += obs_note_colored

    if success_hit:
        prefix_plain = "[AUTO] SUCCESS: "
        prefix_colored = f"{COLOR_WHITE}[AUTO] {COLOR_GREEN}SUCCESS{COLOR_RESET}{COLOR_WHITE}: {COLOR_RESET}"
        gate_plain = "saw success gates"
        gate_colored = f"{COLOR_GREEN}saw success gates{COLOR_RESET}"
    else:
        prefix_plain = "[AUTO] "
        prefix_colored = f"{COLOR_WHITE}[AUTO] {COLOR_RESET}"
        gate_plain = "FAILED success gates"
        gate_colored = f"{COLOR_RED}FAILED{COLOR_RESET} success gates"

    plain = f"{prefix_plain}{gate_plain} ({pre_snapshot}), so I {action_plain}.\n{outcome_plain}."
    colored = f"{prefix_colored}{gate_colored} ({pre_obs_text_colored}), so I {action_colored}.\n{outcome_colored}."
    lite_detail = _auto_diag_lite_gate_detail(world, step_key)
    if isinstance(lite_detail, dict):
        lite_plain = str(lite_detail.get("plain") or "").strip()
        lite_colored = str(lite_detail.get("colored") or "").strip()
        if lite_plain:
            plain = f"{plain}\n{lite_plain}"
        if lite_colored:
            colored = f"{colored}\n{lite_colored}"
    return {
        "plain": plain,
        "colored": colored,
        "success_hit": bool(success_hit),
        "delta_class": delta_class or "unknown",
        "segments": _ansi_to_stream_segments(colored),
        "selected_pre": dict(selected_pre) if isinstance(selected_pre, dict) else None,
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
        world._gatecheck_lite_passed = True
        brick_ok = _evaluate_traditional_brick_success(
            world,
            step,
            telemetry_rules,
            process_rules,
        )
    elif not lite_brick_ok:
        world._gatecheck_mode = "lite"
        world._gatecheck_lite_passed = False
        brick_ok = False
    else:
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_passed = True
        brick_ok = _evaluate_traditional_brick_success(
            world,
            step,
            telemetry_rules,
            process_rules,
        )
    wall_ok = bool(telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope).ok)
    robot_ok = bool(telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules).ok)
    return brick_ok and wall_ok and robot_ok


def _store_auto_step_diagnostic(
    world,
    step,
    line,
    sent_snapshot=None,
    *,
    colored_line=None,
    segments=None,
    selected_pre=None,
):
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
    step_key = normalize_step_label(step)
    if step_key and isinstance(selected_pre, dict):
        focus_map = getattr(world, "_auto_diag_last_pre_focus_by_step", None)
        if not isinstance(focus_map, dict):
            focus_map = {}
            world._auto_diag_last_pre_focus_by_step = focus_map
        focus_map[step_key] = {
            "metric": selected_pre.get("metric"),
            "value": selected_pre.get("value"),
            "distance": selected_pre.get("distance"),
            "matched": selected_pre.get("matched"),
        }


def clear_pending_auto_step_diagnostic(world):
    if world is None:
        return
    world._pending_auto_step_diag = None


def reset_auto_step_action_stats(world, step):
    if world is None:
        return
    step_key = normalize_step_label(step)
    focus_map = getattr(world, "_auto_diag_last_pre_focus_by_step", None)
    if isinstance(focus_map, dict) and step_key in focus_map:
        focus_map.pop(step_key, None)
    world._auto_step_action_stats = {
        "step": step_key,
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
    pre_entries=None,
    action_index=None,
):
    if world is None:
        return
    try:
        action_index_val = int(action_index) if action_index is not None else None
    except (TypeError, ValueError):
        action_index_val = None
    world._pending_auto_step_diag = {
        "step": normalize_step_label(step),
        "action_text": str(action_text),
        "pre_gate_text": str(pre_gate_text or "none"),
        "pre_focus": dict(pre_focus) if isinstance(pre_focus, dict) else None,
        "pre_entries": [dict(entry) for entry in pre_entries if isinstance(entry, dict)] if isinstance(pre_entries, list) else None,
        "action_index": action_index_val,
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
    pre_entries = pending.get("pre_entries")
    action_index = pending.get("action_index")
    sent_snapshot = pending.get("sent_snapshot")
    diag = _build_auto_step_diagnostic(
        world,
        pending_step,
        action_text,
        pre_gate_text,
        pre_focus=pre_focus,
        pre_entries=pre_entries,
        action_index=action_index,
    )
    _record_auto_step_action_stats(world, pending_step, diag.get("delta_class"))
    line = diag["plain"]
    step_key = normalize_step_label(pending_step)
    step_cfg = (
        (getattr(world, "process_rules", None) or {}).get(step_key, {})
        if world is not None and step_key
        else {}
    )
    emit_each_flush = bool(isinstance(step_cfg, dict) and step_cfg.get("emit_auto_diag_each_act"))
    # Default behavior emits only at completion boundaries (force=True). Selected
    # steps can opt into per-act replay diagnostics for easier demo-following review.
    should_emit = (
        bool(emit)
        and (bool(force) or emit_each_flush)
        and not bool(diag.get("success_hit", False))
    )
    if should_emit:
        print(diag["colored"])
    _store_auto_step_diagnostic(
        world,
        pending_step,
        line,
        sent_snapshot=sent_snapshot,
        colored_line=diag.get("colored"),
        segments=diag.get("segments"),
        selected_pre=diag.get("selected_pre"),
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
    action_index=None,
    pre_entries=None,
):
    diag = _build_auto_step_diagnostic(
        world,
        step,
        action_text,
        pre_gate_text,
        pre_focus=pre_focus,
        success_override=success_override,
        action_index=action_index,
        pre_entries=pre_entries,
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
        selected_pre=diag.get("selected_pre"),
    )
    return line


def _gate_entry_matches(metric, value, stats, direction):
    stats = stats if isinstance(stats, dict) else {}
    if metric == "visible":
        if value is None:
            return False
        bool_val = bool(value)
        min_val = stats.get("min")
        max_val = stats.get("max")
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
            if metric in ("xAxis_offset_abs", "yAxis_offset_abs", "dist"):
                return abs(numeric - target_num) <= tol_num
            ok = next_module._target_tol_ok(numeric, stats, direction)
            if ok is not None:
                return bool(ok)

    min_val = stats.get("min")
    max_val = stats.get("max")
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


def _stream_success_gate_line(step, entry, *, process_rules=None):
    metric = entry.get("metric")
    stats = entry.get("stats") or {}
    value = entry.get("value")
    raw_value = entry.get("raw_value")
    requirement = _gate_requirement_text(stats)
    force_mismatch = False
    if metric == "visible":
        effective = entry.get("effective_visible")
        raw = entry.get("raw_visible")
        current = _fmt_gate_value(bool(effective))
        if raw is not None and bool(raw) != bool(effective):
            current = f"{current} raw={_fmt_gate_value(bool(raw))}"
        if bool(entry.get("confident_visible_recent")) and gate_utils.bool_gate_target(stats) is False:
            force_mismatch = True
            current = f"{current} recent_confident_visible=true"
    else:
        current = _fmt_gate_value(value) if value is not None else "n/a"
        delta_text = None
        if value is not None and not isinstance(value, bool) and isinstance(stats, dict):
            try:
                value_num = float(value)
            except (TypeError, ValueError):
                value_num = None
            if value_num is not None:
                delta_num = None
                target = stats.get("target")
                min_val = stats.get("min")
                max_val = stats.get("max")
                if target is not None and not isinstance(target, bool):
                    try:
                        delta_num = float(value_num) - float(target)
                    except (TypeError, ValueError):
                        delta_num = None
                elif min_val is not None and max_val is None and not isinstance(min_val, bool):
                    try:
                        delta_num = float(value_num) - float(min_val)
                    except (TypeError, ValueError):
                        delta_num = None
                elif max_val is not None and min_val is None and not isinstance(max_val, bool):
                    try:
                        delta_num = float(value_num) - float(max_val)
                    except (TypeError, ValueError):
                        delta_num = None
                elif min_val is not None and max_val is not None and not isinstance(min_val, bool) and not isinstance(max_val, bool):
                    try:
                        min_num = float(min_val)
                        max_num = float(max_val)
                    except (TypeError, ValueError):
                        min_num = None
                        max_num = None
                    if min_num is not None and max_num is not None:
                        if float(value_num) < float(min_num):
                            delta_num = float(value_num) - float(min_num)
                        elif float(value_num) > float(max_num):
                            delta_num = float(value_num) - float(max_num)
                        else:
                            delta_num = 0.0
                if delta_num is not None:
                    delta_text = f"{float(delta_num):+.1f}"
        if delta_text is not None:
            current = f"{_fmt_gate_value(value)} Δ{delta_text}"
        try:
            if value is not None and raw_value is not None:
                if abs(float(raw_value) - float(value)) >= 0.05:
                    current = f"{current} raw={_fmt_gate_value(raw_value)}"
        except (TypeError, ValueError):
            pass
    direction = metric_direction(metric, step, process_rules=process_rules)
    matched = _gate_entry_matches(metric, value, stats, direction)
    if force_mismatch:
        matched = False
    return {
        "segments": [
            (f"{metric}{requirement} ", STREAM_WHITE),
            (f"({current})", STREAM_GREEN if matched else STREAM_RED),
        ]
    }


def compute_stream_gate_summary(world, step, active=True, include_step_line=True):
    if not active:
        return [], None
    obj_name = normalize_step_label(step)
    if not obj_name:
        obj_name = str(step)
    nominal_only = step_is_nominal_only(step, world.process_rules)
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
        metric_lines.append(
            _stream_success_gate_line(
                obj_name,
                entry,
                process_rules=(world.process_rules or {}),
            )
        )

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

    step_cfg = (world.process_rules or {}).get(obj_name, {})
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    visible_only_success = bool(
        isinstance(success_gates, dict)
        and success_gates
        and set(str(metric) for metric in success_gates.keys()).issubset({"visible"})
    )
    # For non-visible-only steps, alignment commands are not actionable while the
    # brick is invisible because numeric offsets/distances are unavailable.
    if (not nominal_only) and (not visible_only_success) and (not bool((world.brick or {}).get("visible"))):
        suggested_cmd = None
        suggested_score = None

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
        _, pwm, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
            suggested_cmd,
            suggested_score,
        )
        suggested = action_sent_display_text(
            suggested_cmd,
            suggested_score,
            pwm=pwm,
            duration_ms=duration_ms,
        )
    summary = []
    if include_step_line:
        summary.append(f"STEP: {obj_name}")
    summary.extend(metric_lines)
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
                "y_axis": world.brick.get("y_axis", world.brick.get("offset_y")),
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


def _average_smoothed_frames(frames, *, step=None, process_rules=None):
    if not frames:
        return None

    def mean(key):
        values = [float(frame.get(key, 0.0) or 0.0) for frame in frames]
        return sum(values) / len(values) if values else 0.0

    def majority(key):
        values = [bool(frame.get(key)) for frame in frames]
        return sum(1 for value in values if value) >= (len(values) / 2.0)

    def majority_optional_bool(key):
        vals = [frame.get(key) for frame in frames if frame.get(key) is not None]
        if not vals:
            return None
        bool_vals = [bool(v) for v in vals]
        return sum(1 for value in bool_vals if value) >= (len(bool_vals) / 2.0)

    x_axis = mean("x_axis")
    y_axis = mean("y_axis")
    try:
        center_x_mm, center_y_mm = telemetry_brick.in_crosshairs_center_mm(
            None,
            step=step,
            process_rules=process_rules if isinstance(process_rules, dict) else {},
        )
    except Exception:
        center_x_mm, center_y_mm = 0.0, 0.0
    in_crosshairs = telemetry_brick._compute_in_crosshairs(
        majority("visible"),
        x_axis,
        y_axis,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
    )
    return {
        "visible": majority("visible"),
        "dist": mean("dist"),
        "angle": mean("angle"),
        "x_axis": x_axis,
        "offset_x": mean("offset_x"),
        "y_axis": y_axis,
        "offset_y": mean("offset_y"),
        "inCrosshairs": bool(in_crosshairs),
        "in_crosshairs": bool(in_crosshairs),
        "in_crosshairs_center_x": float(center_x_mm),
        "in_crosshairs_center_y": float(center_y_mm),
        "confidence": mean("confidence"),
        "brick_above": majority_optional_bool("brick_above"),
        "brick_below": majority_optional_bool("brick_below"),
        "brickAbove": majority_optional_bool("brick_above"),
        "brickBelow": majority_optional_bool("brick_below"),
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

    measurement = _average_smoothed_frames(
        frames,
        step=step,
        process_rules=process_rules,
    )
    if measurement is None:
        return False
    world._lite_gate_measurement = measurement
    world._lite_gate_frame_ids = [int(frame.get("frame_id", 0) or 0) for frame in frames]
    visible_stats = success_gates.get("visible") if isinstance(success_gates, dict) else None
    if telemetry_brick.visible_false_gate_confident_recent(world, visible_stats):
        world._lite_gate_visible_false_confident_seen = True
        return False
    world._lite_gate_visible_false_confident_seen = False
    visible_now = bool(measurement.get("visible"))
    visibility_required_metrics = {"angle_abs", "xAxis_offset_abs", "yAxis_offset_abs", "dist", "confidence"}
    saw_metric = False
    for metric, stats in success_gates.items():
        metric_key = str(metric or "").strip()
        if not metric_key:
            continue
        saw_metric = True
        if (metric_key in visibility_required_metrics) and (not visible_now):
            return False
        value = gate_utils.metric_value_from_measurement(measurement, metric_key)
        direction = metric_direction(metric_key, step, process_rules=process_rules)
        if not _gate_entry_matches(metric_key, value, stats, direction):
            return False
    return bool(saw_metric)


def _evaluate_traditional_brick_success(world, step, telemetry_rules, process_rules):
    visibility_grace_s = _visibility_grace_s(world, step)
    brick_success = telemetry_brick.evaluate_success_gates(
        world,
        step,
        telemetry_rules,
        process_rules,
        visibility_grace_s=visibility_grace_s,
    )
    return bool(brick_success.ok)


def _evaluate_instant_success_truth(world, step, *, extra_ok=True):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    brick_ok = _evaluate_traditional_brick_success(
        world,
        step,
        telemetry_rules,
        process_rules,
    )
    wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules)
    return bool(brick_ok and wall_success.ok and robot_success.ok and bool(extra_ok))


def evaluate_gate_status(world, step):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    lite_brick_ok = _evaluate_lite_gate_brick_success(world, step, process_rules)
    if lite_brick_ok is None:
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 0
        world._gatecheck_lite_collected = 0
        world._gatecheck_lite_passed = True
        brick_ok = _evaluate_traditional_brick_success(
            world,
            step,
            telemetry_rules,
            process_rules,
        )
    elif not lite_brick_ok:
        # Lite is a precheck; do not claim success until it passes.
        world._gatecheck_mode = "lite"
        world._gatecheck_lite_passed = False
        brick_ok = False
    else:
        # As soon as lite precheck passes, switch to full confirmation checks.
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_passed = True
        if _lite_only_aruco_experiment_active():
            brick_ok = True
        else:
            brick_ok = _evaluate_traditional_brick_success(
                world,
                step,
                telemetry_rules,
                process_rules,
            )
    wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules)

    success_ok = brick_ok and wall_success.ok and robot_success.ok
    confidence = step_confidence(success_ok, world, step)
    return success_ok, confidence


def update_gatecheck_with_precheck(world, step, tracker, success_ok, *, phase, log=True):
    if tracker is None:
        return False
    mode = str(getattr(world, "_gatecheck_mode", "traditional") or "traditional").strip().lower()
    # Hard rule: do not start the full gate tracker (consecutive/majority) until a
    # post-lite sample actually satisfies the current effective success gates.
    if mode != "lite":
        lite_required = int(getattr(world, "_gatecheck_lite_required", 0) or 0)
        full_checks_started = int(getattr(tracker, "total_checks", 0) or 0) > 0
        if lite_required > 0 and (not full_checks_started) and (not bool(success_ok)):
            mode = "lite"
            try:
                world._gatecheck_mode = "lite"
            except Exception:
                pass
    if mode == "lite":
        lite_checks = int(getattr(world, "_gatecheck_lite_checks", 0) or 0) + 1
        world._gatecheck_lite_checks = lite_checks
        world._gatecheck_status = {
            "step": normalize_step_label(step) or str(step),
            "phase": phase or "run",
            "mode": "lite",
            "checks": lite_checks,
            "pass": 0,
            "fail": lite_checks,
            "streak": 0,
            "need": max(1, int(getattr(tracker, "consecutive_required", 1) or 1)),
            "consecutive_ok": False,
            "window_pass": 0,
            "window_size": 0,
            "window_total": max(1, int(getattr(tracker, "majority_window", 1) or 1)),
            "window_need": max(1, int(getattr(tracker, "majority_required", 1) or 1)),
            "majority_ok": False,
            "truth_ok": False,
            "truth_by": None,
            "lite_required": int(getattr(world, "_gatecheck_lite_required", 0) or 0),
            "lite_collected": int(getattr(world, "_gatecheck_lite_collected", 0) or 0),
            "frame_id": int(getattr(world, "_frame_id", 0) or 0),
            "timestamp": time.time(),
        }
        if log:
            print_gatecheck_progress_line(world, step, log=log)
        return False

    lite_required = int(getattr(world, "_gatecheck_lite_required", 0) or 0)
    lite_passed = getattr(world, "_gatecheck_lite_passed", None)
    if lite_required > 0 and lite_passed is False:
        return False

    if _lite_only_aruco_experiment_active() and lite_required > 0:
        step_key = normalize_step_label(step) or str(step)
        phase_key = phase or "run"
        world._gatecheck_lite_checks = 0
        truth_ok = bool(success_ok)
        world._gatecheck_status = {
            "step": step_key,
            "phase": phase_key,
            "mode": "traditional",
            "checks": 1,
            "pass": 1 if truth_ok else 0,
            "fail": 0 if truth_ok else 1,
            "streak": 1 if truth_ok else 0,
            "need": 1,
            "need_pass": 1,
            "consecutive_ok": truth_ok,
            "window_pass": 1 if truth_ok else 0,
            "window_size": 1,
            "window_total": 1,
            "window_need": 1,
            "window_need_pass": 1,
            "majority_ok": truth_ok,
            "truth_ok": truth_ok,
            "truth_by": "lite_only_aruco" if truth_ok else None,
            "lite_required": int(getattr(world, "_gatecheck_lite_required", 0) or 0),
            "lite_collected": int(getattr(world, "_gatecheck_lite_collected", 0) or 0),
            "frame_id": int(getattr(world, "_frame_id", 0) or 0),
            "timestamp": time.time(),
        }
        if log:
            print_gatecheck_progress_line(world, step, log=log)
            if truth_ok:
                _log_lite_only_aruco_gatecheck_pass(world, step, phase)
        return truth_ok

    if int(getattr(tracker, "total_checks", 0) or 0) <= 0:
        print_gatecheck_entry_proof_line(
            world,
            step,
            phase=phase,
            success_ok=success_ok,
            log=log,
        )

    world._gatecheck_lite_checks = 0
    success_met = gate_utils.update_gatecheck(
        world,
        step,
        tracker,
        success_ok,
        phase=phase,
    )
    if log:
        print_gatecheck_progress_line(world, step, log=log)
    return success_met


def _new_success_metric_tracker_like(tracker):
    if tracker is None:
        return None
    try:
        clone = gate_utils.SuccessGateTracker(
            int(getattr(tracker, "consecutive_required", 1) or 1),
            int(getattr(tracker, "majority_window", 1) or 1),
            int(getattr(tracker, "majority_required", 1) or 1),
        )
    except Exception:
        return None
    for attr in ("consecutive_pass_required", "majority_pass_required"):
        if hasattr(tracker, attr):
            try:
                setattr(clone, attr, int(getattr(tracker, attr)))
            except Exception:
                pass
    return clone


def _success_metric_tracker_counts(tracker):
    if tracker is None:
        return None
    try:
        window_vals = list(getattr(tracker, "window", []) or [])
        window_pass = int(sum(1 for ok in window_vals if ok))
        return {
            "checks": int(getattr(tracker, "total_checks", 0) or 0),
            "pass": int(getattr(tracker, "total_pass", 0) or 0),
            "streak": int(getattr(tracker, "consecutive", 0) or 0),
            "need": max(1, int(getattr(tracker, "consecutive_required", 1) or 1)),
            "window_pass": int(window_pass),
            "window_total": max(1, int(getattr(tracker, "majority_window", 1) or 1)),
            "window_need": max(1, int(getattr(tracker, "majority_required", 1) or 1)),
        }
    except Exception:
        return None


def _update_success_gate_metric_tallies(world, step, tracker, *, phase):
    if world is None or tracker is None:
        return None
    step_key = normalize_step_label(step) or str(step)
    mode = str(getattr(world, "_gatecheck_mode", "traditional") or "traditional").strip().lower()
    entries = _success_gate_entries(world, step)
    tracker_sig = (
        int(getattr(tracker, "consecutive_required", 1) or 1),
        int(getattr(tracker, "majority_window", 1) or 1),
        int(getattr(tracker, "majority_required", 1) or 1),
        int(getattr(tracker, "consecutive_pass_required", getattr(tracker, "consecutive_required", 1)) or 1),
        int(getattr(tracker, "majority_pass_required", getattr(tracker, "majority_required", 1)) or 1),
    )
    state = getattr(world, "_success_gate_metric_tallies", None)
    if not isinstance(state, dict):
        state = {}
    reset_needed = (
        normalize_step_label(state.get("step")) != step_key
        or str(state.get("phase") or "") != str(phase or "")
        or tuple(state.get("tracker_sig") or ()) != tracker_sig
    )
    if reset_needed:
        state = {
            "step": step_key,
            "phase": str(phase or ""),
            "tracker_sig": tracker_sig,
            "rows": {},
        }
    rows = state.get("rows")
    if not isinstance(rows, dict):
        rows = {}
        state["rows"] = rows

    seen_metrics = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("metric") or "").strip()
        if not metric:
            continue
        seen_metrics.add(metric)
        row = rows.get(metric)
        if not isinstance(row, dict):
            row = {"metric": metric, "tracker": _new_success_metric_tracker_like(tracker)}
            rows[metric] = row
        if row.get("tracker") is None:
            row["tracker"] = _new_success_metric_tracker_like(tracker)
        value = entry.get("value")
        stats = entry.get("stats") or {}
        matched = _gate_entry_matches(metric, value, stats, metric_direction(metric, step))
        row["value"] = value
        row["matched"] = matched
        row["stats"] = stats
        row["last_seen_frame_id"] = int(getattr(world, "_frame_id", 0) or 0)
        if mode != "lite" and row.get("tracker") is not None:
            try:
                row["tracker"].update(bool(matched))
            except Exception:
                pass
        row["counts"] = _success_metric_tracker_counts(row.get("tracker"))

    for metric in list(rows.keys()):
        if metric not in seen_metrics:
            rows.pop(metric, None)

    state["mode"] = mode
    state["timestamp"] = time.time()
    world._success_gate_metric_tallies = state
    return state


def _success_gate_metric_tallies_text(world, step, *, phase=None, metrics=None, full_only=True):
    state = getattr(world, "_success_gate_metric_tallies", None)
    if not isinstance(state, dict):
        return None
    step_key = normalize_step_label(step) or str(step)
    if normalize_step_label(state.get("step")) != step_key:
        return None
    if phase is not None and str(state.get("phase") or "") != str(phase):
        return None
    mode = str(state.get("mode") or "traditional").strip().lower()
    if full_only and mode == "lite":
        return None
    rows = state.get("rows")
    if not isinstance(rows, dict) or not rows:
        return None
    metric_order = []
    if isinstance(metrics, (list, tuple)):
        metric_order.extend([str(m) for m in metrics if m is not None])
    metric_order.extend([m for m in rows.keys() if m not in metric_order])
    parts = []
    for metric in metric_order:
        row = rows.get(metric)
        if not isinstance(row, dict):
            continue
        counts = row.get("counts")
        if not isinstance(counts, dict):
            continue
        value_txt = _fmt_gate_value(row.get("value"))
        if value_txt is None:
            value_txt = "wait"
        try:
            consec_seen = int(counts.get("streak", 0) or 0)
            consec_need = max(1, int(counts.get("need", 1) or 1))
            maj_seen = int(counts.get("window_pass", 0) or 0)
            maj_total = max(1, int(counts.get("window_total", 1) or 1))
        except Exception:
            continue
        parts.append(
            f"{metric}={value_txt} ({consec_seen}/{consec_need} consec - {maj_seen}/{maj_total} maj)"
        )
    if not parts:
        return None
    return "Gate check: " + ". ".join(parts)


def print_success_gate_metric_tallies(world, step, *, phase=None, metrics=None, log=True, force=False):
    if not log or world is None:
        return
    text = _success_gate_metric_tallies_text(
        world,
        step,
        phase=phase,
        metrics=metrics,
        full_only=True,
    )
    if not text:
        return
    state = getattr(world, "_success_gate_metric_tallies", None)
    rows = (state or {}).get("rows") if isinstance(state, dict) else {}
    sig_rows = []
    for metric in (metrics or rows.keys()):
        row = rows.get(metric) if isinstance(rows, dict) else None
        counts = row.get("counts") if isinstance(row, dict) else None
        sig_rows.append(
            (
                str(metric),
                row.get("value") if isinstance(row, dict) else None,
                (counts or {}).get("streak") if isinstance(counts, dict) else None,
                (counts or {}).get("need") if isinstance(counts, dict) else None,
                (counts or {}).get("window_pass") if isinstance(counts, dict) else None,
                (counts or {}).get("window_total") if isinstance(counts, dict) else None,
            )
        )
    sig = (
        normalize_step_label(step) or str(step),
        str(phase or ""),
        tuple(sig_rows),
    )
    if not force and getattr(world, "_last_success_gate_metric_tally_sig", None) == sig:
        return
    world._last_success_gate_metric_tally_sig = sig
    print(format_headline(f"[ALIGN_TOP] {text}", COLOR_WHITE))


def observe_success_gatecheck(
    world,
    step,
    tracker,
    *,
    phase,
    log=True,
    extra_ok=True,
    hold_on_positive=True,
):
    success_ok, confidence = evaluate_gate_status(world, step)
    try:
        instant_success_ok = _evaluate_instant_success_truth(world, step, extra_ok=extra_ok)
    except Exception:
        # Minimal test doubles may omit full world state (brick/wall); in that
        # case fall back to the evaluated gate-status truth instead of failing.
        instant_success_ok = bool(success_ok and bool(extra_ok))
    _update_success_gate_metric_tallies(world, step, tracker, phase=phase)
    effective_success_ok = bool(success_ok and bool(extra_ok))
    mode_now = str(getattr(world, "_gatecheck_mode", "traditional") or "traditional").strip().lower()

    # Single-lite rule: once a lite failure has been observed for the current
    # step/phase/action state, do not keep re-running lite checks until the
    # motion state changes (new action timestamp) or lite passes.
    action_time = getattr(world, "_last_action_time", None)
    try:
        action_time_key = float(action_time) if action_time is not None else 0.0
    except (TypeError, ValueError):
        action_time_key = 0.0
    lite_latch_key = (
        normalize_step_label(step) or str(step),
        str(phase or "run"),
        action_time_key,
    )
    latched_key = getattr(world, "_gatecheck_lite_fail_latch_key", None)
    if mode_now == "lite" and not bool(effective_success_ok) and latched_key == lite_latch_key:
        return {
            "success_ok": bool(success_ok),
            "instant_success_ok": bool(instant_success_ok),
            "effective_success_ok": False,
            "success_met": False,
            "hold_for_confirm": False,
            "confidence": confidence,
        }

    gate_utils.record_success_gate_entry(world, step, effective_success_ok)
    if log:
        log_confidence(world, confidence, step)
    success_met = update_gatecheck_with_precheck(
        world,
        step,
        tracker,
        effective_success_ok,
        phase=phase,
        log=log,
    )
    if mode_now == "lite" and not bool(effective_success_ok) and not bool(success_met):
        world._gatecheck_lite_fail_latch_key = lite_latch_key
    else:
        world._gatecheck_lite_fail_latch_key = None
    hold_for_confirm = False
    if tracker is not None:
        visible_only = next_module.success_gates_visible_only(world.process_rules or {}, step)
        if bool(effective_success_ok):
            hold_for_confirm = bool(
                gate_utils.should_hold_for_success_confirmation(visible_only, tracker, success_met)
            )
    if bool(hold_on_positive) and bool(instant_success_ok) and bool(effective_success_ok) and not bool(success_met):
        hold_for_confirm = True
    # Single-lite rule: in lite mode, a non-pass should immediately return
    # control to correction planning rather than parking in confirmation hold.
    if mode_now == "lite" and not bool(success_met):
        hold_for_confirm = False
    return {
        "success_ok": bool(success_ok),
        "instant_success_ok": bool(instant_success_ok),
        "effective_success_ok": bool(effective_success_ok),
        "success_met": bool(success_met),
        "hold_for_confirm": bool(hold_for_confirm),
        "confidence": confidence,
    }


def gatecheck_after_move(world, step, tracker, phase, log=True):
    if tracker is None:
        return False
    result = observe_success_gatecheck(
        world,
        step,
        tracker,
        phase=phase,
        log=log,
    )
    return bool(result.get("success_met"))


def run_full_gatecheck_after_act(
    world,
    vision,
    step,
    tracker,
    phase,
    log=True,
    observer=None,
    required_new_frames=1,
):
    if tracker is None:
        return False
    # Single full gatecheck sample per act for all steps.
    checks_per_act = 1
    required_frames = max(0, int(required_new_frames or 0))
    truth_hit = False
    truth_by = None
    for _ in range(checks_per_act):
        if required_frames > 0:
            gate_utils.wait_for_fresh_frames(
                world,
                lambda: update_world_from_vision(world, vision, log=log),
                required_new_frames=int(required_frames),
                max_cycles=max(24, int(required_frames) * 24),
                sleep_s=0.0,
            )
        else:
            update_world_from_vision(world, vision, log=log)
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


def pre_action_success_observation(
    world,
    vision,
    step,
    tracker,
    *,
    phase,
    robot=None,
    observer=None,
    log=True,
    success_log=True,
):
    if tracker is None:
        return {"success_met": False, "hold_for_confirm": False}
    gate_utils.wait_for_fresh_frames(
        world,
        lambda: update_world_from_vision(world, vision, log=log),
        required_new_frames=1,
        max_cycles=12,
        sleep_s=0.0,
    )
    if observer:
        observer("frame", world, vision, None, None, None)
    gate_obs = observe_success_gatecheck(
        world,
        step,
        tracker,
        phase=phase,
        log=log,
    )
    success_met = bool((gate_obs or {}).get("success_met"))
    hold_for_confirm = bool((gate_obs or {}).get("hold_for_confirm"))
    if success_met:
        if robot:
            robot.stop()
        if success_log:
            print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
            print_gate_summary_line(world, tracker)
            print_success_events(world, step)
    elif hold_for_confirm and robot:
        robot.stop()
        if observer:
            observer("action", world, vision, None, 0.0, "gate confirm hold")
    return {
        "success_met": bool(success_met),
        "hold_for_confirm": bool(hold_for_confirm),
    }


def _strip_start_gate_hold_prefix(reason):
    text = str(reason or "").strip()
    if not text:
        return ""
    if text.lower().startswith("start_gate:"):
        return text.split(":", 1)[1].strip()
    return text


def _start_gate_metric_state_value(world, metric):
    metric_key = str(metric or "").strip()
    if not metric_key or world is None:
        return None

    try:
        brick = telemetry_brick.smoothed_brick_snapshot(world) or {}
    except Exception:
        brick = (getattr(world, "brick", None) or {})

    if metric_key == "visible":
        visible_now = bool(brick.get("visible"))
        if not visible_now:
            try:
                visible_now = bool(
                    telemetry_brick._recent_raw_visible(
                        world,
                        min_confidence=telemetry_brick.START_GATE_MIN_CONFIDENCE,
                        required_hits=1,
                        max_samples=telemetry_brick.BRICK_SMOOTH_FRAMES,
                    )
                )
            except Exception:
                pass
        return bool(visible_now)

    if metric_key in ("brick_above", "brickAbove"):
        if "brickAbove" in brick:
            value = brick.get("brickAbove")
            return None if value is None else bool(value)
        value = brick.get("brick_above")
        return None if value is None else bool(value)

    if metric_key in ("brick_below", "brickBelow"):
        if "brickBelow" in brick:
            value = brick.get("brickBelow")
            return None if value is None else bool(value)
        value = brick.get("brick_below")
        return None if value is None else bool(value)

    try:
        value = telemetry_brick.metric_value(brick, metric_key)
    except Exception:
        value = None
    if value is not None:
        return value

    if metric_key == "lift_height":
        return getattr(world, "lift_height", None)

    wall = getattr(world, "wall", None)
    if isinstance(wall, dict) and metric_key in wall:
        return wall.get(metric_key)

    try:
        return getattr(world, metric_key)
    except Exception:
        return None


def _start_gate_target_text(stats):
    if not isinstance(stats, dict):
        return "?"
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if isinstance(target, bool):
            return "true" if target else "false"
        try:
            tol_text = _fmt_gate_value(abs(float(tol)))
        except (TypeError, ValueError):
            tol_text = _fmt_gate_value(tol)
        return f"{_fmt_gate_value(target)} +/- {tol_text}"
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        return "true" if bool(min_val) else "false"
    if isinstance(max_val, bool):
        return "true" if bool(max_val) else "false"
    if min_val is not None and max_val is not None:
        return f"[{_fmt_gate_value(min_val)}, {_fmt_gate_value(max_val)}]"
    if min_val is not None:
        return f">={_fmt_gate_value(min_val)}"
    if max_val is not None:
        return f"<={_fmt_gate_value(max_val)}"
    mu = stats.get("mu")
    sigma = stats.get("sigma")
    if mu is not None and sigma is not None:
        return f"{_fmt_gate_value(mu)} +/- {_fmt_gate_value(sigma)}"
    return "?"


def _start_gate_metric_match(value, stats):
    if not isinstance(stats, dict):
        return None
    min_val = stats.get("min")
    max_val = stats.get("max")
    target = stats.get("target")
    tol = stats.get("tol")

    if isinstance(target, bool) or isinstance(min_val, bool) or isinstance(max_val, bool):
        if value is None:
            return None
        value_bool = bool(value)
        if isinstance(target, bool):
            return value_bool == bool(target)
        if isinstance(min_val, bool):
            return value_bool == bool(min_val)
        if isinstance(max_val, bool):
            return value_bool == bool(max_val)
        return None

    if value is None:
        return None
    try:
        value_num = float(value)
    except (TypeError, ValueError):
        return None

    if target is not None and tol is not None:
        try:
            return abs(value_num - float(target)) <= abs(float(tol))
        except (TypeError, ValueError):
            return None

    try:
        if min_val is not None and value_num < float(min_val):
            return False
        if max_val is not None and value_num > float(max_val):
            return False
    except (TypeError, ValueError):
        return None
    return True


def _start_gate_state_text(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def _format_start_gate_timeout_details(world, start_gates):
    if not isinstance(start_gates, dict) or not start_gates:
        return None
    details = []
    for metric, stats in start_gates.items():
        target_txt = _start_gate_target_text(stats)
        state_value = _start_gate_metric_state_value(world, metric)
        state_txt = _start_gate_state_text(state_value)
        match_ok = _start_gate_metric_match(state_value, stats)
        color = COLOR_GREEN if match_ok is True else COLOR_RED
        details.append(f"{metric} target={target_txt} state={color}{state_txt}{COLOR_RESET}")
    return ", ".join(details) if details else None


def _format_start_gate_status_details(world, step):
    step_key = normalize_step_label(step)
    try:
        cfg = ((getattr(world, "process_rules", None) or {}).get(step_key) or {})
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        return None
    start_gates = cfg.get("start_gates") if isinstance(cfg.get("start_gates"), dict) else {}
    wall_note = None
    try:
        wall_note = telemetry_wall.wall_origin_requirement_note(world, step_key)
    except Exception:
        wall_note = None
    detail_text = _format_start_gate_timeout_details(world, start_gates)
    if detail_text:
        if wall_note:
            return f"start gates: {detail_text}; {wall_note}"
        return f"start gates: {detail_text}"
    try:
        start_desc, _ = format_gate_lines(cfg)
    except Exception:
        start_desc = None
    if start_desc and start_desc != "none":
        if wall_note:
            return f"start gates: {start_desc}; {wall_note}"
        return f"start gates: {start_desc}"
    if wall_note:
        return wall_note
    return None


def _format_start_gate_timeout_status(world, step, last_hold_reason):
    parts = [f"timeout after {float(START_GATE_TIMEOUT_S):.2f}s"]
    blocked_detail = _strip_start_gate_hold_prefix(last_hold_reason)
    if blocked_detail:
        parts.append(f"last blocked: {blocked_detail}")
    start_detail = _format_start_gate_status_details(world, step)
    if start_detail:
        parts.append(start_detail)
    return "; ".join(parts)


def _start_gate_failure_reason_from_wait_status(start_status):
    status_text = str(start_status or "").strip()
    if not status_text:
        return "start gates not met"
    lowered = status_text.lower()
    if lowered in {"start", "success"}:
        return status_text
    return f"start gates not met: {status_text}"


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
    force_require_start_gates=False,
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
    if (not force_require_start_gates) and (not step_requires_start_gates(step_key, getattr(world, "process_rules", None))):
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
                lite_required = int(getattr(success_tracker, "lite_unique_frames", 0) or 0)
                if lite_required > 0:
                    precheck_frames = max(1, lite_required)
                else:
                    precheck_frames = max(
                        int(getattr(success_tracker, "majority_window", 1)),
                        int(getattr(success_tracker, "consecutive_required", 1)),
                        1,
                    )
            for _ in range(precheck_frames):
                update_world_from_vision(world, vision, log=log)
                if observer:
                    observer("frame", world, vision, None, None, None)
                if not allow_success or success_tracker is None:
                    continue
                gate_obs = observe_success_gatecheck(
                    world,
                    step,
                    success_tracker,
                    phase="start",
                    log=log,
                )
                success_met = bool(gate_obs.get("success_met"))
                if success_met:
                    gate_utils.store_gate_summary(world, success_tracker)
                    if robot:
                        robot.stop()
                    if log:
                        print(format_headline(f"[START] {step_key} success gates already met", COLOR_GREEN))
                    return "success"
        suppress_skip_log = bool(getattr(world, "_suppress_start_gate_skip_log", False)) if world is not None else False
        if log and not suppress_skip_log:
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
            gate_obs = observe_success_gatecheck(
                world,
                step,
                success_tracker,
                phase="start",
                log=log,
            )
            success_ok = bool(gate_obs.get("effective_success_ok"))
            success_met = bool(gate_obs.get("success_met"))
            gate_status = getattr(world, "_gatecheck_status", None)
            gate_mode = str((gate_status or {}).get("mode") or "").strip().lower()
            if success_ok and not success_seen and robot:
                robot.stop()
                success_seen = True
            if success_met:
                gate_utils.store_gate_summary(world, success_tracker)
                if robot:
                    robot.stop()
                return "success"
            if (not success_ok) and gate_mode == "lite":
                # Start gates are satisfied; if the first lite success precheck
                # fails, leave start-hold immediately and continue with normal
                # adjustment acts in the step loop instead of re-checking lite
                # in place until timeout.
                if robot:
                    robot.stop()
                return "start"
            if not success_ok:
                success_seen = False
            if bool(gate_obs.get("hold_for_confirm")):
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
                    detail = _format_start_gate_status_details(world, step)
                    if detail:
                        print(format_headline(f"[START] {step} start gates met", COLOR_GREEN, f" {detail}"))
                    else:
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
                last_cmd = cmd
                last_speed = speed
        time.sleep(CONTROL_DT)
    return _format_start_gate_timeout_status(world, step_key, last_reason)


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
        prev_log_y_err,
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
        elif ctype == "y_axis":
            y_now = local_gate_before_action.get("y_err")
            try:
                curr = abs(float(y_now))
            except (TypeError, ValueError):
                return "unknown"
            try:
                if str(prev_log_correction_type or "").strip().lower() == "y_axis" and prev_log_y_err is not None:
                    prev = abs(float(prev_log_y_err))
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

    def _unknown_metric_for_planned_action(*, cmd, cmd_reason, local_gate_before_action):
        cmd_key = str(cmd or "").strip().lower()
        reason_key = str(cmd_reason or "").strip().lower()
        local_gate = local_gate_before_action if isinstance(local_gate_before_action, dict) else {}
        brick_now = getattr(world, "brick", {}) or {}

        def _is_missing_number(value):
            try:
                float(value)
                return False
            except (TypeError, ValueError):
                return True

        if cmd_key in ("l", "r"):
            if reason_key in {"angle", "angle_abs"}:
                angle_now = brick_now.get("angle")
                if _is_missing_number(angle_now):
                    return "angle_abs"
                return None
            x_err_now = local_gate.get("x_err")
            if _is_missing_number(x_err_now):
                x_raw = brick_now.get("x_axis", brick_now.get("offset_x"))
                x_target = local_gate.get("x_target")
                if _is_missing_number(x_target):
                    x_target = 0.0
                if not _is_missing_number(x_raw):
                    try:
                        x_err_now = float(x_raw) - float(x_target)
                    except (TypeError, ValueError):
                        x_err_now = None
            if _is_missing_number(x_err_now):
                return "x_err"
            return None

        if cmd_key in ("u", "d"):
            y_err_now = local_gate.get("y_err")
            if _is_missing_number(y_err_now):
                y_raw = brick_now.get("y_axis", brick_now.get("offset_y"))
                y_target = local_gate.get("y_target")
                if _is_missing_number(y_target):
                    y_target = 0.0
                if not _is_missing_number(y_raw):
                    try:
                        y_err_now = float(y_raw) - float(y_target)
                    except (TypeError, ValueError):
                        y_err_now = None
            if _is_missing_number(y_err_now):
                return "y_err"
            return None

        if cmd_key in ("f", "b"):
            dist_now = local_gate.get("dist")
            if _is_missing_number(dist_now):
                dist_now = brick_now.get("dist")
            if _is_missing_number(dist_now):
                return "dist"
            return None

        return None

    def _emit_align_shorthand_action_line(
        *,
        local_gate_before_action,
        prev_log_correction_type,
        prev_log_x_err,
        prev_log_y_err,
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
        elif cmd_key in ("u", "d"):
            correction_type = "y_axis"
        else:
            correction_type = str(prev_log_correction_type or "").strip().lower()
            if correction_type not in {"distance", "x_axis", "y_axis"}:
                correction_type = "x_axis"

        x_err_now = local_gate_before_action.get("x_err")
        x_target = local_gate_before_action.get("x_target")
        x_tol = local_gate_before_action.get("x_tol")
        y_err_now = local_gate_before_action.get("y_err")
        y_target = local_gate_before_action.get("y_target")
        y_tol = local_gate_before_action.get("y_tol")
        dist_now = local_gate_before_action.get("dist")
        dist_target = local_gate_before_action.get("dist_target")
        dist_tol = local_gate_before_action.get("dist_tol")

        metric_color = COLOR_YELLOW
        metric_text = ""
        metric_name = "x_err"
        switch_message = None
        switch_message_colored = None
        result_observation_message = None
        result_observation_message_colored = None
        comparison_message = None
        comparison_message_colored = None
        switch_reassure_message = None
        switch_reassure_message_colored = None
        switch_snapshot_lines = None
        switch_snapshot_lines_colored = None
        prev_type = str(prev_log_correction_type or "").strip().lower()
        switched_gap_type = bool(force_gap_switch) or (
            prev_type in {"distance", "x_axis", "y_axis"} and prev_type != correction_type
        )

        def _switch_gap_message_pair(metric_name_local):
            plain = f"Switching gap type to {metric_name_local}"
            colored = f"{COLOR_ORANGE_BRIGHT}Switching{COLOR_RESET} gap type to {metric_name_local}"
            return plain, colored

        def _switch_result_observation_pair(metric_name_local, curr_err_local, prev_err_local, *, overshot_local=False):
            plain, colored, _ = _format_align_result_observation(
                metric_name_local,
                curr_err_local,
                prev_err_local,
                overshot=bool(overshot_local),
            )
            return plain, colored

        def _comparison_line_pair(metric_name_local, curr_err_local, prev_err_local, *, overshot_local=False):
            return _format_align_result_observation(
                metric_name_local,
                curr_err_local,
                prev_err_local,
                overshot=bool(overshot_local),
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
                    if prev_err_any is not None and curr_err is not None:
                        comparison_message, comparison_message_colored, cmp_color = _comparison_line_pair(
                            metric_name,
                            curr_err,
                            prev_err_any,
                        )
                        if cmp_color is not None:
                            metric_color = cmp_color
                        else:
                            metric_color = COLOR_YELLOW
                    else:
                        metric_color = COLOR_YELLOW
                else:
                    comparison_message, comparison_message_colored, cmp_color = _comparison_line_pair(
                        metric_name,
                        curr_err,
                        prev_err,
                    )
                    if cmp_color is not None:
                        metric_color = cmp_color
                err_text = f"{metric_color}{float(curr_err):+.2f}{COLOR_RESET}"
                abs_text = f"{float(curr_abs_val):.2f}"
                metric_text = (
                    f"target ({float(dist_target):.2f} ±{float(dist_tol):.2f}) "
                    f"{err_text} off={abs_text}"
                )
                if switched_gap_type:
                    if prev_err_any is not None and curr_err is not None:
                        (
                            result_observation_message,
                            result_observation_message_colored,
                        ) = _switch_result_observation_pair(metric_name, curr_err, prev_err_any)
                    switch_message, switch_message_colored = _switch_gap_message_pair(metric_name)
        else:
            if correction_type == "y_axis":
                metric_name = "y_err"
                axis_err_now = y_err_now
                axis_target = y_target
                axis_tol = y_tol
                prev_log_axis_err = prev_log_y_err
                axis_label = "y"
            else:
                metric_name = "x_err"
                axis_err_now = x_err_now
                axis_target = x_target
                axis_tol = x_tol
                prev_log_axis_err = prev_log_x_err
                axis_label = "x"
            try:
                curr_err = float(axis_err_now)
                curr_metric = abs(float(curr_err))
                curr_abs_val = float(axis_target) + float(curr_err)
            except (TypeError, ValueError):
                curr_err = None
                curr_metric = None
                curr_abs_val = None
            prev_metric = None
            prev_err = None
            prev_err_any = None
            try:
                if prev_log_axis_err is not None:
                    prev_err_any = float(prev_log_axis_err)
                if prev_log_correction_type == correction_type and prev_log_axis_err is not None:
                    prev_err = float(prev_log_axis_err)
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
                    if prev_err_any is not None and curr_err is not None:
                        comparison_message, comparison_message_colored, cmp_color = _comparison_line_pair(
                            metric_name,
                            curr_err,
                            prev_err_any,
                        )
                        if cmp_color is not None:
                            metric_color = cmp_color
                        else:
                            metric_color = COLOR_YELLOW
                    else:
                        metric_color = COLOR_YELLOW
                elif overshot:
                    comparison_message, comparison_message_colored, cmp_color = _comparison_line_pair(
                        metric_name,
                        curr_err,
                        prev_err,
                        overshot_local=True,
                    )
                    if cmp_color is not None:
                        metric_color = cmp_color
                    else:
                        metric_color = COLOR_RED
                else:
                    comparison_message, comparison_message_colored, cmp_color = _comparison_line_pair(
                        metric_name,
                        curr_err,
                        prev_err,
                    )
                    if cmp_color is not None:
                        metric_color = cmp_color
                err_text = f"{metric_color}{float(curr_err):+.2f}{COLOR_RESET}"
                abs_text = f"{float(curr_abs_val):+.2f}"
                metric_text = (
                    f"target ({float(axis_target):+.2f} ±{float(axis_tol):.2f}) "
                    f"{err_text} off={abs_text}"
                )
                if switched_gap_type:
                    overshot_any = False
                    if prev_err_any is not None and curr_err is not None:
                        try:
                            overshot_any = (
                                (float(prev_err_any) * float(curr_err)) < 0.0
                                and abs(float(prev_err_any)) > 0.30
                                and abs(float(curr_err)) > 0.30
                            )
                        except (TypeError, ValueError):
                            overshot_any = False
                        (
                            result_observation_message,
                            result_observation_message_colored,
                        ) = _switch_result_observation_pair(
                            metric_name,
                            curr_err,
                            prev_err_any,
                            overshot_local=bool(overshot_any),
                        )
                    switch_message, switch_message_colored = _switch_gap_message_pair(metric_name)

        step_key_for_switch_log = normalize_step_label(step) or str(step or "").strip().upper()
        if switched_gap_type and step_key_for_switch_log in GAP_SWITCH_REASSURE_STEPS:
            reassurance = _gap_switch_lite_gate_reassurance(world, step_key_for_switch_log, metric_name)
            if isinstance(reassurance, dict):
                switch_reassure_message = reassurance.get("metric_plain")
                switch_reassure_message_colored = reassurance.get("metric_colored")
                switch_snapshot_lines = reassurance.get("snapshot_plain_lines")
                switch_snapshot_lines_colored = reassurance.get("snapshot_colored_lines")
                if switch_message and switch_reassure_message:
                    switch_message = f"{switch_message}; {switch_reassure_message}"
                    switch_render_tail = switch_reassure_message_colored or switch_reassure_message
                    switch_message_colored = (
                        f"{switch_message_colored}; {switch_render_tail}"
                        if switch_message_colored
                        else switch_message
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
        if isinstance(action_meta, dict):
            ease_in_out_note = action_meta.get("ease_in_out_note")
            if not ease_in_out_note:
                ease_in_out_note = action_meta.get("anti_alias_note")
            if ease_in_out_note:
                parts.append(str(ease_in_out_note))
        if parts:
            motor_details = f" {COLOR_GRAY}({', '.join(parts)}){COLOR_RESET}"

        trial_num = int(getattr(world, "loop_id", 0) or 0)
        active_step_label = normalize_step_label(step) or str(step or "ALIGN").strip().upper()
        step_number = _step_number_for_label(active_step_label)
        if step_number is not None:
            step_prefix = f"{COLOR_WHITE}[step#{int(step_number)}]{COLOR_RESET} "
        else:
            step_prefix = ""
        if active_step_label == "ALIGN_BRICK":
            phase_base = "ALIGN"
        else:
            phase_base = f"{active_step_label}_ALIGN"
        phase_label = f"{phase_base}_SETTLE" if bool(settle) else phase_base
        line_prefix = f"{step_prefix}{COLOR_WHITE}[T{trial_num}.{int(action_index)} {phase_label}]"

        def _print_result_lite_gate_line():
            detail = _result_lite_gate_detail(world, step)
            if not isinstance(detail, dict):
                return
            line_text = detail.get("colored") or detail.get("plain")
            if not line_text:
                return
            print(
                f"{line_prefix} "
                f"{line_text}{COLOR_RESET}"
            )

        if switch_message:
            if result_observation_message:
                result_render = result_observation_message_colored or result_observation_message
                print(
                    f"{line_prefix} "
                    f"{result_render}{COLOR_RESET}"
                )
                _print_result_lite_gate_line()
            switch_render = switch_message_colored or switch_message
            print(
                f"{line_prefix} "
                f"{switch_render}{COLOR_RESET}"
            )
            snapshot_plain_lines = switch_snapshot_lines if isinstance(switch_snapshot_lines, list) else []
            snapshot_colored_lines = (
                switch_snapshot_lines_colored if isinstance(switch_snapshot_lines_colored, list) else []
            )
            if snapshot_plain_lines:
                for idx, plain_line in enumerate(snapshot_plain_lines):
                    if not plain_line:
                        continue
                    render_line = (
                        snapshot_colored_lines[idx]
                        if idx < len(snapshot_colored_lines) and snapshot_colored_lines[idx]
                        else plain_line
                    )
                    print(
                        f"{line_prefix} "
                        f"{render_line}{COLOR_RESET}"
                    )

                # Gap-switch result observation already emitted above in canonical form.
                comparison_message = None
                comparison_message_colored = None
        cmd_log = str(cmd).strip().lower()
        print(
            f"{line_prefix}{COLOR_RESET} "
            f"{metric_name}={metric_text} → "
            f"{COLOR_ORANGE_BRIGHT}{cmd_log} {int(score_display)}%{COLOR_RESET}{motor_details}"
        )
        if comparison_message:
            comparison_render = comparison_message_colored or comparison_message
            print(
                f"{line_prefix} "
                f"{comparison_render}{COLOR_RESET}"
            )
            _print_result_lite_gate_line()

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

    def _observed_bool(value):
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(float(value) > 0.5)
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"true", "yes", "y", "1", "pass"}:
                return True
            if key in {"false", "no", "n", "0", "wait", "unknown", "none", ""}:
                return False
        return bool(value)

    def _observe_visible(timeout_s):
        deadline = float(time.time()) + max(0.05, float(timeout_s))
        while time.time() < deadline:
            update_world_from_vision(world, vision, log=False)
            if _observed_bool((getattr(world, "brick", None) or {}).get("visible")):
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
    step_key = normalize_step_label(step)
    step_rules = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        step_rules = world.process_rules.get(step_key, {}) or {}
    visible_search_cfg = step_rules.get("search_visible_false_speed_cycle") if isinstance(step_rules, dict) else {}
    if not isinstance(visible_search_cfg, dict):
        visible_search_cfg = {}
    # Backward-compatible default keeps FIND_BRICK behavior unchanged while
    # allowing other steps (for example FIND_WALL2) to opt in via JSON.
    visible_search_cycle_enabled = bool(
        visible_search_cfg.get("enabled", normalize_step_label(step) == "FIND_BRICK")
    )
    visible_search_high_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(
            int(MAX_SPEED_SCORE),
            int(_as_int(visible_search_cfg.get("high_score"), 20)),
        ),
    )
    visible_search_low_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(
            int(MAX_SPEED_SCORE),
            int(_as_int(visible_search_cfg.get("low_score"), 5)),
        ),
    )
    visible_search_next_high = bool(visible_search_cfg.get("start_with_high", True))
    visible_search_cmd_score_overrides = {}
    visible_search_cmd_scores_cfg = visible_search_cfg.get("command_scores")
    if isinstance(visible_search_cmd_scores_cfg, dict):
        for raw_cmd, raw_score in visible_search_cmd_scores_cfg.items():
            cmd_local = str(raw_cmd or "").strip().lower()
            if cmd_local not in ("f", "b", "l", "r", "u", "d"):
                continue
            score_local = _as_int(raw_score, None)
            if score_local is None:
                continue
            score_local = max(
                int(telemetry_robot_module.SPEED_SCORE_MIN),
                min(int(MAX_SPEED_SCORE), int(score_local)),
            )
            visible_search_cmd_score_overrides[cmd_local] = int(score_local)
    visible_search_cmd_cycle_seed = []
    visible_search_cmds_cfg = visible_search_cfg.get("commands")
    if isinstance(visible_search_cmds_cfg, list):
        for raw_cmd in visible_search_cmds_cfg:
            cmd_local = str(raw_cmd or "").strip().lower()
            if cmd_local in ("f", "b", "l", "r"):
                visible_search_cmd_cycle_seed.append(cmd_local)
    visible_search_cmd_cycle = []
    visible_search_next_cmd_idx = 0
    visible_search_phase1_confirm_frames = max(
        1,
        int(_as_int(visible_search_cfg.get("phase1_visible_confirm_frames"), 1)),
    )

    def _next_visible_search_score(cmd=None):
        nonlocal visible_search_next_high
        cmd_key = str(cmd or "").strip().lower()
        score_override = visible_search_cmd_score_overrides.get(cmd_key)
        if score_override is not None:
            score_local = int(score_override)
        elif not bool(visible_search_cycle_enabled):
            score_local = int(getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1))
        else:
            score_local = int(visible_search_high_score) if bool(visible_search_next_high) else int(visible_search_low_score)
            visible_search_next_high = not bool(visible_search_next_high)
        try:
            return int(telemetry_robot_module.normalize_speed_score(score_local))
        except Exception:
            return int(score_local)

    def _next_visible_search_cmd():
        nonlocal visible_search_next_cmd_idx
        if not visible_search_cmd_cycle:
            return None
        cycle_len = len(visible_search_cmd_cycle)
        cmd_local = str(visible_search_cmd_cycle[int(visible_search_next_cmd_idx) % int(cycle_len)])
        visible_search_next_cmd_idx = (int(visible_search_next_cmd_idx) + 1) % int(cycle_len)
        return cmd_local

    operator_exception_reminder = str(step_rules.get("operator_exception_reminder") or "").strip() if isinstance(step_rules, dict) else ""
    startup_action_cfg = step_rules.get("startup_action_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(startup_action_cfg, dict):
        startup_action_cfg = {}
    startup_action_enabled = bool(startup_action_cfg.get("enabled"))
    startup_action_cmd = str(startup_action_cfg.get("command") or "").strip().lower()
    if startup_action_cmd not in ("f", "b", "l", "r", "u", "d"):
        startup_action_cmd = ""
    startup_action_acts = max(1, int(_as_int(startup_action_cfg.get("acts"), 1)))
    startup_action_score = None
    startup_score_hotkey = str(startup_action_cfg.get("score_from_hotkey") or "").strip().lower()
    if startup_score_hotkey:
        hotkey_rows = getattr(telemetry_robot_module, "HOTKEY_SPEED_SCORES", {})
        hotkey_row = hotkey_rows.get(startup_score_hotkey) if isinstance(hotkey_rows, dict) else None
        if isinstance(hotkey_row, dict):
            score_val = _as_int(hotkey_row.get("score"), None)
            if score_val is not None:
                startup_action_score = int(score_val)
    if startup_action_score is None:
        startup_action_score = int(
            _as_int(
                startup_action_cfg.get("score"),
                int(telemetry_robot_module.SPEED_SCORE_MIN),
            )
        )
    startup_action_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(startup_action_score)),
    )
    startup_action_pause_s = max(
        0.0,
        float(_as_float(startup_action_cfg.get("pause_s"), CONTROL_DT)),
    )
    startup_action_active = bool(startup_action_enabled and startup_action_cmd)
    success_gate_cfg = step_rules.get("success_gates") if isinstance(step_rules, dict) else {}
    success_gate_keys = {
        str(key) for key in (success_gate_cfg or {}).keys() if key is not None
    } if isinstance(success_gate_cfg, dict) else set()
    progress_mast_cfg = step_rules.get("progress_mast_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(progress_mast_cfg, dict):
        progress_mast_cfg = {}
    progress_mast_enabled = bool(progress_mast_cfg.get("enabled"))
    progress_mast_reminder = str(progress_mast_cfg.get("operator_log_reminder") or "").strip()
    progress_mast_trigger_fraction = max(
        0.0,
        min(1.0, float(_as_float(progress_mast_cfg.get("trigger_progress_fraction"), 0.25))),
    )
    progress_mast_cmd = str(progress_mast_cfg.get("command") or "d").strip().lower()
    if progress_mast_cmd not in ("d", "u"):
        progress_mast_cmd = "d"
    progress_mast_score = _as_int(
        progress_mast_cfg.get("score"),
        int(telemetry_robot_module.SPEED_SCORE_MIN),
    )
    progress_mast_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(progress_mast_score)),
    )
    progress_mast_acts = max(1, int(_as_int(progress_mast_cfg.get("acts"), 4)))
    progress_mast_min_gap_mm = max(
        0.0,
        float(_as_float(progress_mast_cfg.get("min_start_to_target_gap_mm"), 1.0)),
    )
    progress_mast_target_dist = _as_float(progress_mast_cfg.get("dist_target"), None)
    if progress_mast_target_dist is None and isinstance(success_gate_cfg, dict):
        dist_gate_cfg = success_gate_cfg.get("dist")
        if isinstance(dist_gate_cfg, dict):
            progress_mast_target_dist = _as_float(dist_gate_cfg.get("target"), None)
    progress_mast_active = bool(progress_mast_enabled and progress_mast_target_dist is not None)
    near_dist_vis_cfg = step_rules.get("near_dist_visibility_mast_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(near_dist_vis_cfg, dict):
        near_dist_vis_cfg = {}
    near_dist_vis_enabled = bool(near_dist_vis_cfg.get("enabled"))
    near_dist_vis_reminder = str(near_dist_vis_cfg.get("operator_log_reminder") or "").strip()
    near_dist_vis_cmd = str(near_dist_vis_cfg.get("command") or "d").strip().lower()
    if near_dist_vis_cmd not in ("d", "u"):
        near_dist_vis_cmd = "d"
    near_dist_vis_score = _as_int(
        near_dist_vis_cfg.get("score"),
        int(telemetry_robot_module.SPEED_SCORE_MIN),
    )
    near_dist_vis_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(near_dist_vis_score)),
    )
    near_dist_vis_max_attempts = max(1, int(_as_int(near_dist_vis_cfg.get("max_attempts"), 2)))
    near_dist_vis_ratio_max = max(
        0.0,
        min(1.0, float(_as_float(near_dist_vis_cfg.get("near_gate_ratio_max"), 0.2))),
    )
    near_dist_vis_observe_s = max(
        float(CONTROL_DT),
        float(_as_float(near_dist_vis_cfg.get("observe_s"), 0.35)),
    )
    near_dist_vis_target_dist = _as_float(near_dist_vis_cfg.get("dist_target"), None)
    near_dist_vis_tol = _as_float(near_dist_vis_cfg.get("dist_tol"), None)
    if isinstance(success_gate_cfg, dict):
        dist_gate_cfg_local = success_gate_cfg.get("dist")
        if isinstance(dist_gate_cfg_local, dict):
            if near_dist_vis_target_dist is None:
                near_dist_vis_target_dist = _as_float(dist_gate_cfg_local.get("target"), None)
            if near_dist_vis_tol is None:
                near_dist_vis_tol = _as_float(dist_gate_cfg_local.get("tol"), None)
    near_dist_vis_tol = abs(float(near_dist_vis_tol or 0.0))
    near_dist_vis_gap_mm_max = float(near_dist_vis_tol) * float(near_dist_vis_ratio_max)
    near_dist_vis_active = bool(
        near_dist_vis_enabled
        and near_dist_vis_target_dist is not None
        and near_dist_vis_gap_mm_max > 0.0
        and near_dist_vis_max_attempts > 0
    )
    forward_stall_cfg = (
        step_rules.get("forward_dist_stall_mast_exception")
        if isinstance(step_rules, dict)
        else {}
    )
    if not isinstance(forward_stall_cfg, dict):
        forward_stall_cfg = {}
    forward_stall_enabled = bool(forward_stall_cfg.get("enabled"))
    forward_stall_reminder = str(forward_stall_cfg.get("operator_log_reminder") or "").strip()
    forward_stall_forward_cmd = str(forward_stall_cfg.get("forward_command") or "f").strip().lower()
    if forward_stall_forward_cmd not in ("f", "b"):
        forward_stall_forward_cmd = "f"
    forward_stall_trigger_acts = max(
        1,
        int(_as_int(forward_stall_cfg.get("forward_acts_without_improvement"), 7)),
    )
    forward_stall_min_improvement_mm = max(
        0.0,
        float(_as_float(forward_stall_cfg.get("min_dist_improvement_mm"), 1.0)),
    )
    forward_stall_cmd = str(forward_stall_cfg.get("command") or "u").strip().lower()
    if forward_stall_cmd not in ("u", "d"):
        forward_stall_cmd = "u"
    forward_stall_score = _as_int(
        forward_stall_cfg.get("score"),
        int(telemetry_robot_module.SPEED_SCORE_MIN),
    )
    forward_stall_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(forward_stall_score)),
    )
    forward_stall_acts = max(1, int(_as_int(forward_stall_cfg.get("acts"), 2)))
    forward_stall_dist_target = _as_float(forward_stall_cfg.get("dist_target"), None)
    if forward_stall_dist_target is None and isinstance(success_gate_cfg, dict):
        dist_gate_cfg_local = success_gate_cfg.get("dist")
        if isinstance(dist_gate_cfg_local, dict):
            forward_stall_dist_target = _as_float(dist_gate_cfg_local.get("target"), None)
    forward_stall_active = bool(
        forward_stall_enabled
        and forward_stall_dist_target is not None
        and forward_stall_acts > 0
    )
    crosshair_gate_cfg = None
    if isinstance(success_gate_cfg, dict):
        crosshair_gate_cfg = success_gate_cfg.get("inCrosshairs")
        if crosshair_gate_cfg is None:
            crosshair_gate_cfg = success_gate_cfg.get("in_crosshairs")
    topmost_requires_crosshair_false = gate_utils.bool_gate_target(crosshair_gate_cfg) is False
    topmost_above_gate_cfg = None
    topmost_below_gate_cfg = None
    if isinstance(success_gate_cfg, dict):
        topmost_above_gate_cfg = success_gate_cfg.get("brick_above")
        if topmost_above_gate_cfg is None:
            topmost_above_gate_cfg = success_gate_cfg.get("brickAbove")
        topmost_below_gate_cfg = success_gate_cfg.get("brick_below")
        if topmost_below_gate_cfg is None:
            topmost_below_gate_cfg = success_gate_cfg.get("brickBelow")
    topmost_requires_brick_above_false = gate_utils.bool_gate_target(topmost_above_gate_cfg) is False
    topmost_requires_brick_below_false = gate_utils.bool_gate_target(topmost_below_gate_cfg) is False
    topmost_requires_stack_clear = bool(
        topmost_requires_brick_above_false or topmost_requires_brick_below_false
    )
    topmost_gate_metrics = (
        [str(key) for key in success_gate_cfg.keys() if key is not None]
        if isinstance(success_gate_cfg, dict)
        else []
    )
    if not topmost_gate_metrics and topmost_requires_crosshair_false:
        topmost_gate_metrics = ["inCrosshairs"]
    topmost_gate_target_text = format_gate_metrics(success_gate_cfg if isinstance(success_gate_cfg, dict) else {})
    if not str(topmost_gate_target_text).strip() or str(topmost_gate_target_text).strip().lower() == "none":
        topmost_gate_target_text = "inCrosshairs=false" if topmost_requires_crosshair_false else "configured success gates"
    topmost_exception_cfg = step_rules.get("topmost_crosshair_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(topmost_exception_cfg, dict):
        topmost_exception_cfg = {}
    topmost_exception_enabled = bool(topmost_exception_cfg.get("enabled"))
    topmost_exception_reminder = str(topmost_exception_cfg.get("operator_log_reminder") or "").strip()
    topmost_post_desc_cfg = topmost_exception_cfg.get("post_success_descend")
    if not isinstance(topmost_post_desc_cfg, dict):
        topmost_post_desc_cfg = {}
    topmost_post_desc_cmd = str(topmost_post_desc_cfg.get("command") or "d").strip().lower()
    if topmost_post_desc_cmd not in ("d", "u"):
        topmost_post_desc_cmd = "d"
    topmost_post_seek_true_score = _as_int(
        topmost_post_desc_cfg.get("seek_true_score"),
        int(ALIGN_BRICK_TOPMOST_LIFT_SCORE),
    )
    topmost_post_seek_true_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(topmost_post_seek_true_score)),
    )
    topmost_post_seek_true_max_acts = max(1, _as_int(topmost_post_desc_cfg.get("seek_true_max_acts"), 16))
    topmost_post_seek_false_score = _as_int(
        topmost_post_desc_cfg.get("seek_false_score"),
        int(ALIGN_BRICK_TOPMOST_LIFT_SCORE),
    )
    topmost_post_seek_false_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(topmost_post_seek_false_score)),
    )
    topmost_post_seek_false_max_acts = max(1, _as_int(topmost_post_desc_cfg.get("seek_false_max_acts"), 16))
    topmost_post_auto_continue_delay_s = max(
        0.0,
        float(_as_float(topmost_post_desc_cfg.get("auto_continue_delay_s"), 3.0)),
    )
    topmost_post_continue_blank_lines = max(
        0,
        int(_as_int(topmost_post_desc_cfg.get("post_continue_blank_lines"), 4)),
    )
    step_post_desc_cfg = step_rules.get("post_success_descend") if isinstance(step_rules, dict) else {}
    if not isinstance(step_post_desc_cfg, dict):
        step_post_desc_cfg = {}
    step_post_desc_enabled = bool(step_post_desc_cfg.get("enabled", False))
    step_post_desc_cmd = str(step_post_desc_cfg.get("command") or "d").strip().lower()
    if step_post_desc_cmd not in ("d", "u"):
        step_post_desc_cmd = "d"
    step_post_desc_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(step_post_desc_cfg.get("score"), 100))),
    )
    step_post_desc_completion_mode = str(step_post_desc_cfg.get("completion_mode") or "true_streak").strip().lower()
    if step_post_desc_completion_mode not in ("true_streak", "true_then_false_streak"):
        step_post_desc_completion_mode = "true_streak"
    step_post_desc_true_required = max(1, int(_as_int(step_post_desc_cfg.get("true_down_acts_required"), 3)))
    step_post_desc_false_after_true_required = max(
        1,
        int(_as_int(step_post_desc_cfg.get("false_after_true_down_acts_required"), 2)),
    )
    if step_post_desc_enabled:
        step_post_desc_max_acts = max(
            1,
            int(_as_int(step_post_desc_cfg.get("max_acts"), _as_int(step_post_desc_cfg.get("acts"), 16))),
        )
    else:
        step_post_desc_max_acts = 0
    step_pre_desc_cfg = step_rules.get("pre_align_descend") if isinstance(step_rules, dict) else {}
    if not isinstance(step_pre_desc_cfg, dict):
        step_pre_desc_cfg = {}
    step_pre_desc_enabled = bool(step_pre_desc_cfg.get("enabled", False))
    step_pre_desc_cmd = str(step_pre_desc_cfg.get("command") or "d").strip().lower()
    if step_pre_desc_cmd not in ("d", "u"):
        step_pre_desc_cmd = "d"
    step_pre_desc_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(step_pre_desc_cfg.get("score"), 100))),
    )
    step_pre_desc_score_after_seen_true = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(
            int(MAX_SPEED_SCORE),
            int(_as_int(step_pre_desc_cfg.get("score_after_seen_true"), int(step_pre_desc_score))),
        ),
    )
    step_pre_desc_score_when_visible_raw = step_pre_desc_cfg.get("score_when_visible")
    if step_pre_desc_score_when_visible_raw is None:
        step_pre_desc_score_when_visible = None
    else:
        step_pre_desc_score_when_visible = max(
            int(telemetry_robot_module.SPEED_SCORE_MIN),
            min(
                int(MAX_SPEED_SCORE),
                int(_as_int(step_pre_desc_score_when_visible_raw, int(step_pre_desc_score))),
            ),
        )
    step_pre_desc_confirm_frames = max(
        1,
        int(_as_int(step_pre_desc_cfg.get("confirm_frames"), 3)),
    )
    step_pre_desc_require_consistent_observation = bool(
        step_pre_desc_cfg.get("require_consistent_observation", False)
    )
    step_pre_desc_consistent_observation_max_checks = max(
        int(step_pre_desc_confirm_frames),
        int(
            _as_int(
                step_pre_desc_cfg.get("consistent_observation_max_checks"),
                max(12, int(step_pre_desc_confirm_frames) * 12),
            )
        ),
    )
    step_pre_desc_consistent_observation_require_non_wait = bool(
        step_pre_desc_cfg.get("consistent_observation_require_non_wait", True)
    )
    step_pre_desc_completion_mode = str(step_pre_desc_cfg.get("completion_mode") or "true_streak").strip().lower()
    if step_pre_desc_completion_mode not in ("true_streak", "true_then_false_streak"):
        step_pre_desc_completion_mode = "true_streak"
    step_pre_desc_true_required = max(1, int(_as_int(step_pre_desc_cfg.get("true_down_acts_required"), 3)))
    step_pre_desc_false_after_true_required = max(
        1,
        int(_as_int(step_pre_desc_cfg.get("false_after_true_down_acts_required"), 2)),
    )
    if step_pre_desc_enabled:
        step_pre_desc_max_acts = max(
            1,
            int(_as_int(step_pre_desc_cfg.get("max_acts"), _as_int(step_pre_desc_cfg.get("acts"), 16))),
        )
    else:
        step_pre_desc_max_acts = 0
    step_post_mast_cfg = step_rules.get("post_success_mast_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(step_post_mast_cfg, dict):
        step_post_mast_cfg = {}
    step_post_mast_enabled = bool(step_post_mast_cfg.get("enabled", False))
    step_post_mast_reminder = str(step_post_mast_cfg.get("operator_log_reminder") or "").strip()
    step_post_mast_cmd = str(step_post_mast_cfg.get("command") or "u").strip().lower()
    if step_post_mast_cmd not in ("u", "d"):
        step_post_mast_cmd = "u"
    step_post_mast_score = max(
        int(telemetry_robot_module.SPEED_SCORE_MIN),
        min(int(MAX_SPEED_SCORE), int(_as_int(step_post_mast_cfg.get("score"), 50))),
    )
    step_post_mast_acts = max(1, int(_as_int(step_post_mast_cfg.get("acts"), 3)))
    step_post_mast_pause_s = max(
        0.0,
        float(_as_float(step_post_mast_cfg.get("pause_s"), 0.0)),
    )
    step_post_mast_observe_between_acts = bool(step_post_mast_cfg.get("observe_between_acts", False))
    step_post_desc_run_once = False
    step_pre_desc_run_once = False
    step_post_mast_run_once = False
    stack_gate_keys = {"brick_above", "brick_below", "brickAbove", "brickBelow"}
    demo_stack_lift_actions = []
    for action in nominal_actions_from_events(segment.get("events") or []):
        cmd_val = str(action.get("cmd") or "").strip().lower()
        if cmd_val in ("u", "d"):
            demo_stack_lift_actions.append(action)
    use_demo_stack_lift_fallback = bool(
        demo_stack_lift_actions and (success_gate_keys & stack_gate_keys)
    )
    if use_demo_stack_lift_fallback:
        try:
            world._demo_stack_lift_fallback_idx = 0
        except Exception:
            pass

    def _next_demo_stack_lift_action():
        # Stack-gate steps (e.g. BRICK_LOCK) use boolean above/below signals that the
        # generic numeric align planner cannot convert into lift acts. Fall back to
        # replaying the demo's lift pulses, but only after the current gatecheck pass.
        if not use_demo_stack_lift_fallback:
            return None
        brick = getattr(world, "brick", {}) or {}
        if not bool(brick.get("visible")):
            return None
        idx = int(getattr(world, "_demo_stack_lift_fallback_idx", 0) or 0)
        action = demo_stack_lift_actions[idx % len(demo_stack_lift_actions)]
        setattr(world, "_demo_stack_lift_fallback_idx", idx + 1)
        cmd = str(action.get("cmd") or "").strip().lower()
        if cmd not in ("u", "d"):
            return None
        score = action.get("speed_score")
        try:
            score = (
                int(telemetry_robot_module.normalize_speed_score(score))
                if score is not None
                else None
            )
        except (TypeError, ValueError):
            score = None
        if score is None:
            try:
                action_power = float(action.get("power", 0.0) or 0.0)
            except (TypeError, ValueError):
                action_power = 0.0
            if action_power > 0.0:
                _, score = telemetry_robot_module.quantize_speed(
                    cmd,
                    speed=max(0.0, min(1.0, action_power / 255.0)),
                )
        if score is None:
            score = int(telemetry_robot_module.SPEED_SCORE_MIN)
        speed = telemetry_robot_module.manual_speed_for_cmd(cmd, score)
        duration_override_ms = None
        try:
            duration_override_ms = int(
                round(max(0.001, float(action.get("duration_s") or 0.0)) * 1000.0)
            )
        except (TypeError, ValueError):
            duration_override_ms = None
        return {
            "cmd": cmd,
            "speed": float(speed or 0.0),
            "speed_score": int(score),
            "reason": "demo stack lift",
            "duration_override_ms": duration_override_ms,
        }
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
    # Single-source per-act console logging for align-controller steps: keep the
    # shorthand `*_ALIGN` line as the visible attempt log, but still build/store
    # the generic auto-diagnostic internally for stats/UI.
    emit_align_auto_diag_console = False

    scan_cmd = telemetry_robot_module.resolve_scan_direction(
        world.process_rules,
        step,
        fallback=demo_single_turn_cmd or dominant_demo_turn,
    )
    if scan_cmd not in ("l", "r") and demo_single_turn_cmd in ("l", "r"):
        scan_cmd = demo_single_turn_cmd
    if visible_search_cmd_cycle_seed:
        visible_search_cmd_cycle = list(visible_search_cmd_cycle_seed)
    elif scan_cmd in ("f", "b", "l", "r"):
        visible_search_cmd_cycle = [str(scan_cmd)]
    else:
        visible_search_cmd_cycle = []
    if visible_search_cmd_cycle:
        visible_search_next_cmd_idx = int(_as_int(visible_search_cfg.get("start_cmd_index"), 0))
        visible_search_next_cmd_idx %= len(visible_search_cmd_cycle)
    else:
        visible_search_next_cmd_idx = 0

    def _apply_turn_only_demo_policy(cmd, speed, speed_score, cmd_reason, *, planner_name=None):
        # Keep gap micro-planner outputs untouched. The turn-only demo heuristic is
        # only for generic replay-style search behavior.
        if str(planner_name or "").strip().lower() == "gap":
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
    is_align_brick_step = step_key == "ALIGN_BRICK"
    is_find_topmost_step = step_key in {"FIND_TOPMOST_BRICK", "FIND_TOPMOST_BRICK_WALL"}
    # Gate-shape capability: steps with x/y/dist success gates can use the gap
    # micro-planner once visibility is established.
    step_supports_gap_micro_planner = bool(
        next_module.step_uses_gap_alignment_planner(world.process_rules or {}, step)
    )
    align_policy_cfg = step_rules.get("align_policy") if isinstance(step_rules, dict) else {}
    if not isinstance(align_policy_cfg, dict):
        align_policy_cfg = {}
    step_success_gates_cfg = step_rules.get("success_gates") if isinstance(step_rules, dict) else {}
    if not isinstance(step_success_gates_cfg, dict):
        step_success_gates_cfg = {}
    phase1_requires_visible_then_align = bool(
        visible_search_cycle_enabled
        and gate_utils.bool_gate_target(step_success_gates_cfg.get("visible")) is True
        and any(
            str(k)
            in {
                "xAxis_offset_abs",
                "yAxis_offset_abs",
                "xAxis_offset",
                "yAxis_offset",
                "x_axis",
                "y_axis",
            }
            for k in step_success_gates_cfg.keys()
        )
    )
    phase1_visible_streak = 0
    phase1_visible_passed = not bool(phase1_requires_visible_then_align)
    has_xy_target_lock_gate = bool(
        any(
            str(k)
            in {
                "xAxis_offset_abs",
                "yAxis_offset_abs",
                "xAxis_offset",
                "yAxis_offset",
                "x_axis",
                "y_axis",
            }
            for k in step_success_gates_cfg.keys()
        )
    )
    align_target_lock_mode = str(align_policy_cfg.get("target_lock_mode", "soft") or "soft").strip().lower()
    align_target_lock_hard_mode = align_target_lock_mode in {"hard", "strict", "required"}
    align_target_lock_enabled = bool(
        step_supports_gap_micro_planner
        and has_xy_target_lock_gate
        and bool(align_policy_cfg.get("target_lock_enabled", True))
        and bool(align_target_lock_hard_mode)
    )
    align_target_lock_center_prefer_enabled = bool(
        align_policy_cfg.get("target_lock_center_prefer_enabled", True)
    )
    align_target_lock_stack_signature_guard_enabled = bool(
        align_policy_cfg.get("target_lock_stack_signature_guard_enabled", False)
    )
    align_target_lock_confirm_frames = max(
        1,
        int(_as_int(align_policy_cfg.get("target_lock_confirm_frames"), 2)),
    )
    align_target_lock_hold_bad_frames = max(
        1,
        int(_as_int(align_policy_cfg.get("target_lock_hold_bad_frames"), 2)),
    )
    align_target_lock_acquire_timeout_s = max(
        float(CONTROL_DT),
        float(_as_float(align_policy_cfg.get("target_lock_acquire_timeout_s"), 0.35)),
    )
    align_target_lock_center_x_worse_margin_mm = max(
        0.0,
        float(_as_float(align_policy_cfg.get("target_lock_center_x_worse_margin_mm"), 1.5)),
    )
    align_target_lock_center_y_worse_margin_mm = max(
        0.0,
        float(_as_float(align_policy_cfg.get("target_lock_center_y_worse_margin_mm"), 2.0)),
    )
    raw_micro_adjust_crosshair_guard_cmds = align_policy_cfg.get(
        "micro_adjust_crosshair_guard_block_cmds"
    )
    if isinstance(raw_micro_adjust_crosshair_guard_cmds, (list, tuple, set)):
        micro_adjust_crosshair_guard_cmds = tuple(
            cmd
            for cmd in (
                str(item or "").strip().lower()
                for item in raw_micro_adjust_crosshair_guard_cmds
            )
            if cmd in {"u", "d"}
        )
    else:
        micro_adjust_crosshair_guard_cmds = ("d",)
    micro_adjust_crosshair_guard_enabled = bool(
        step_supports_gap_micro_planner
        and bool(align_policy_cfg.get("micro_adjust_crosshair_guard_enabled", False))
        and bool(micro_adjust_crosshair_guard_cmds)
    )
    align_gate_step_key = step
    skip_settle_pause = step_key in FAST_POST_ACT_GATECHECK_STEPS
    # ALIGN_BRICK demos are often state-only, so derived scan speed can fall back to 50%.
    # Use explicit score-1 turn speed so start-gate scan matches "1%" tuning from world_model_robot.
    if is_align_brick_step and start_cmd in ("l", "r"):
        start_speed = telemetry_robot_module.manual_speed_for_cmd(
            start_cmd,
            telemetry_robot_module.SPEED_SCORE_MIN,
        )
    # For ALIGN_BRICK, avoid any robot movement during start-gate checks. We only
    # allow motion after the per-loop gate check determines movement is needed.
    start_cmd_for_wait = start_cmd
    start_speed_for_wait = start_speed
    if step_key in {"ALIGN_BRICK", "FIND_TOPMOST_BRICK", "FIND_TOPMOST_BRICK_WALL"}:
        start_cmd_for_wait = None
        start_speed_for_wait = None

    def _read_post_desc_crosshair():
        brick_local = getattr(world, "brick", {}) or {}
        raw = brick_local.get("inCrosshairs")
        if raw is None:
            raw = brick_local.get("in_crosshairs")
        return None if raw is None else bool(raw)

    def _apply_micro_adjust_crosshair_guard(
        cmd,
        speed,
        speed_score,
        cmd_reason,
        *,
        planner_name=None,
        phase_label="align",
    ):
        if not bool(micro_adjust_crosshair_guard_enabled):
            return cmd, speed, speed_score, cmd_reason
        if str(planner_name or "").strip().lower() != "gap":
            return cmd, speed, speed_score, cmd_reason
        cmd_key = str(cmd or "").strip().lower()
        if cmd_key not in micro_adjust_crosshair_guard_cmds:
            return cmd, speed, speed_score, cmd_reason
        brick_local = getattr(world, "brick", {}) or {}
        if not bool(brick_local.get("visible")):
            return cmd, speed, speed_score, cmd_reason
        crosshair_now = _read_post_desc_crosshair()
        if crosshair_now is not True:
            return cmd, speed, speed_score, cmd_reason
        if not align_silent:
            print(
                format_headline(
                    (
                        f"[GUARD] {step_key} {str(phase_label)}: "
                        f"blocked {str(cmd_key).upper()} while inCrosshairs=YES; "
                        "holding for re-observation."
                    ),
                    COLOR_ORANGE_BRIGHT,
                )
            )
        return None, 0.0, None, f"micro_adjust_crosshair_guard_hold_{cmd_key}"

    def _format_exception_prefix_text():
        # Keep operator semantics consistent while styling only the exception token.
        return f"[{COLOR_MAGENTA_BRIGHT}exception{COLOR_WHITE}]"

    def _highlight_seen_true_text(saw_true):
        if bool(saw_true):
            return f"seen-true={COLOR_GREEN}YES{COLOR_WHITE}"
        return "seen-true=NO"

    def _highlight_false_after_true_text(count, required):
        count_int = int(count)
        required_int = int(required)
        if count_int > 0:
            count_text = f"{COLOR_GREEN}{count_int}{COLOR_WHITE}"
        else:
            count_text = str(count_int)
        return f"false-after-true={count_text}/{required_int}"

    def _print_exception_line(body_text):
        if align_silent:
            return
        print(f"{COLOR_WHITE}{_format_exception_prefix_text()} {body_text}{COLOR_RESET}")

    def _observe_descend_crosshair_confident(*, confirm_frames=1):
        # Observe several fresh frames after each pulse and use majority truth.
        required = max(1, int(confirm_frames or 1))
        threshold = max(1, int((required // 2) + 1))
        samples = []
        if robot:
            robot.stop()
        post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
        if observer:
            observer("frame", world, vision, None, None, None)
        samples.append(_read_post_desc_crosshair())
        for _ in range(1, required):
            wait_for_frame_settle(world, vision, 1, log=not align_silent)
            if observer:
                observer("frame", world, vision, None, None, None)
            samples.append(_read_post_desc_crosshair())
        yes_hits = int(sum(1 for v in samples if v is True))
        no_hits = int(sum(1 for v in samples if v is False))
        if yes_hits >= threshold and yes_hits > no_hits:
            return True, tuple(samples)
        if no_hits >= threshold and no_hits > yes_hits:
            return False, tuple(samples)
        return None, tuple(samples)

    def _crosshair_samples_consistent(samples, *, require_non_wait=True):
        if not isinstance(samples, (list, tuple)) or not samples:
            return False
        first = samples[0]
        if require_non_wait and first is None:
            return False
        return all(value == first for value in samples)

    def _crosshair_sample_text(samples):
        if not isinstance(samples, (list, tuple)):
            return ""
        parts = []
        for value in samples:
            parts.append("WAIT" if value is None else ("YES" if value else "NO"))
        return ",".join(parts)

    def _run_step_post_success_descend():
        nonlocal step_post_desc_run_once
        if not bool(step_post_desc_enabled) or int(step_post_desc_max_acts) <= 0:
            return {"enabled": False, "success": True, "reason": "disabled"}
        if bool(step_post_desc_run_once):
            return {"enabled": True, "success": True, "reason": "already completed"}
        step_post_desc_run_once = True
        post_desc_speed = telemetry_robot_module.manual_speed_for_cmd(step_post_desc_cmd, step_post_desc_score)
        if not align_silent:
            if step_post_desc_completion_mode == "true_then_false_streak":
                goal_text = (
                    f"inCrosshairs=YES then NO x{int(step_post_desc_false_after_true_required)}"
                )
            else:
                goal_text = f"inCrosshairs=YES x{int(step_post_desc_true_required)}"
            print(
                format_headline(
                    (
                        f"[EXCEPTION] [{step_key}] Post-success descend start: "
                        f"{str(step_post_desc_cmd).upper()} {int(step_post_desc_score)}%, "
                        f"goal={goal_text}, max={int(step_post_desc_max_acts)}."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        true_down_streak = 0
        saw_true = False
        false_after_true_streak = 0
        done = False
        for idx in range(1, int(step_post_desc_max_acts) + 1):
            if confirm_callback and not confirm_callback(world, vision):
                return {"enabled": True, "success": False, "reason": "confirm cancelled"}
            if observer:
                observer("analysis", world, vision, step_post_desc_cmd, post_desc_speed, "post_success_descend")
            send_robot_command(
                robot,
                world,
                step,
                step_post_desc_cmd,
                post_desc_speed,
                speed_score=step_post_desc_score,
                auto_mode=True,
            )
            post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
            update_world_from_vision(world, vision, log=not align_silent)
            if observer:
                observer("frame", world, vision, None, None, None)
                observer("action", world, vision, step_post_desc_cmd, post_desc_speed, "post_success_descend")
            crosshair_now = _read_post_desc_crosshair()
            if crosshair_now is True:
                true_down_streak = int(true_down_streak) + 1
                saw_true = True
                false_after_true_streak = 0
            elif crosshair_now is False:
                true_down_streak = 0
                if bool(saw_true):
                    false_after_true_streak = int(false_after_true_streak) + 1
            gate_done = False
            if step_post_desc_completion_mode == "true_then_false_streak":
                gate_done = bool(saw_true) and int(false_after_true_streak) >= int(step_post_desc_false_after_true_required)
                progress_text = (
                    f"seen-true={('YES' if bool(saw_true) else 'NO')}, "
                    f"false-after-true={int(false_after_true_streak)}/{int(step_post_desc_false_after_true_required)}"
                )
            else:
                gate_done = int(true_down_streak) >= int(step_post_desc_true_required)
                progress_text = f"true-streak={int(true_down_streak)}/{int(step_post_desc_true_required)}"
            if not align_silent:
                crosshair_text = "WAIT" if crosshair_now is None else ("YES" if crosshair_now else "NO")
                gate_word = f"{COLOR_GREEN}PASS{COLOR_RESET}" if bool(gate_done) else f"{COLOR_RED}FAIL{COLOR_RESET}"
                print(
                    format_headline(
                        (
                            f"[EXCEPTION] [{step_key}] Post-success descend pulse "
                            f"{int(idx)}/{int(step_post_desc_max_acts)}: "
                            f"inCrosshairs={crosshair_text}, {progress_text}, gate={gate_word}."
                        ),
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            if gate_done:
                done = True
                break
            time.sleep(CONTROL_DT)
        if not bool(done):
            if step_post_desc_completion_mode == "true_then_false_streak":
                fail_reason = (
                    f"post-success descend could not confirm inCrosshairs=YES then NO x{int(step_post_desc_false_after_true_required)}"
                )
            else:
                fail_reason = f"post-success descend could not confirm inCrosshairs=YES x{int(step_post_desc_true_required)}"
            if not align_silent:
                print(
                    format_headline(
                        f"[EXCEPTION] [{step_key}] Post-success descend complete: gate={COLOR_RED}FAIL{COLOR_RESET}.",
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            return {"enabled": True, "success": False, "reason": fail_reason}
        if not align_silent:
            print(
                format_headline(
                    f"[EXCEPTION] [{step_key}] Post-success descend complete: gate={COLOR_GREEN}PASS{COLOR_RESET}.",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        return {"enabled": True, "success": True, "reason": "post-success descend pass"}

    def _run_step_pre_align_descend():
        nonlocal step_pre_desc_run_once
        if not bool(step_pre_desc_enabled) or int(step_pre_desc_max_acts) <= 0:
            return {"enabled": False, "success": True, "reason": "disabled"}
        if bool(step_pre_desc_run_once):
            return {"enabled": True, "success": True, "reason": "already completed"}
        step_pre_desc_run_once = True
        if not align_silent:
            if step_pre_desc_completion_mode == "true_then_false_streak":
                goal_text = (
                    f"inCrosshairs=YES then NO x{int(step_pre_desc_false_after_true_required)}"
                )
            else:
                goal_text = f"inCrosshairs=YES x{int(step_pre_desc_true_required)}"
            speed_note = ""
            if int(step_pre_desc_score_after_seen_true) != int(step_pre_desc_score):
                speed_note = (
                    f", after-seen-true={int(step_pre_desc_score_after_seen_true)}%"
                )
            if int(step_pre_desc_confirm_frames) > 1:
                speed_note += f", confirm={int(step_pre_desc_confirm_frames)} frames"
            if bool(step_pre_desc_require_consistent_observation):
                speed_note += (
                    f", consistent-observe=YES(max={int(step_pre_desc_consistent_observation_max_checks)})"
                )
            _print_exception_line(
                (
                    f"[{step_key}] Pre-align descend start: "
                    f"{str(step_pre_desc_cmd).upper()} {int(step_pre_desc_score)}%, "
                    f"goal={goal_text}, max={int(step_pre_desc_max_acts)}{speed_note}."
                )
            )
        true_down_streak = 0
        saw_true = False
        false_after_true_streak = 0
        seen_true_downshift_logged = False
        done = False
        for idx in range(1, int(step_pre_desc_max_acts) + 1):
            crosshair_now, crosshair_samples = _observe_descend_crosshair_confident(
                confirm_frames=step_pre_desc_confirm_frames,
            )
            observation_checks = 1
            if bool(step_pre_desc_require_consistent_observation):
                while not _crosshair_samples_consistent(
                    crosshair_samples,
                    require_non_wait=bool(step_pre_desc_consistent_observation_require_non_wait),
                ):
                    if int(observation_checks) >= int(step_pre_desc_consistent_observation_max_checks):
                        fail_reason = (
                            f"pre-align descend could not obtain consistent inCrosshairs observation "
                            f"(obs{int(step_pre_desc_confirm_frames)}=[{_crosshair_sample_text(crosshair_samples)}], "
                            f"checks={int(observation_checks)}/{int(step_pre_desc_consistent_observation_max_checks)})"
                        )
                        if not align_silent:
                            _print_exception_line(
                                f"[{step_key}] Pre-align descend observe hold timeout: {fail_reason}."
                            )
                        return {"enabled": True, "success": False, "reason": fail_reason}
                    if not align_silent:
                        _print_exception_line(
                            (
                                f"[{step_key}] Pre-align descend observe hold: "
                                f"obs{int(step_pre_desc_confirm_frames)}=[{_crosshair_sample_text(crosshair_samples)}] "
                                f"(check {int(observation_checks)}/{int(step_pre_desc_consistent_observation_max_checks)}); "
                                "re-observing before act."
                            )
                        )
                    if observer:
                        observer("action", world, vision, None, 0.0, "pre_align_descend_wait_consistent_observation")
                    time.sleep(CONTROL_DT)
                    crosshair_now, crosshair_samples = _observe_descend_crosshair_confident(
                        confirm_frames=step_pre_desc_confirm_frames,
                    )
                    observation_checks = int(observation_checks) + 1
            saw_true_before = bool(saw_true)
            if crosshair_now is True:
                true_down_streak = int(true_down_streak) + 1
                saw_true = True
                false_after_true_streak = 0
                if (
                    (not bool(saw_true_before))
                    and int(step_pre_desc_score_after_seen_true) != int(step_pre_desc_score)
                    and (not bool(seen_true_downshift_logged))
                ):
                    _print_exception_line(
                        (
                            f"[{step_key}] Pre-align descend: seen-true latched; "
                            f"reducing {str(step_pre_desc_cmd).upper()} speed to "
                            f"{int(step_pre_desc_score_after_seen_true)}% for remaining pulses."
                        )
                    )
                    seen_true_downshift_logged = True
            elif crosshair_now is False:
                true_down_streak = 0
                if bool(saw_true):
                    false_after_true_streak = int(false_after_true_streak) + 1
            gate_done = False
            if step_pre_desc_completion_mode == "true_then_false_streak":
                gate_done = bool(saw_true) and int(false_after_true_streak) >= int(step_pre_desc_false_after_true_required)
                progress_text = (
                    f"{_highlight_seen_true_text(saw_true)}, "
                    f"{_highlight_false_after_true_text(false_after_true_streak, step_pre_desc_false_after_true_required)}"
                )
            else:
                gate_done = int(true_down_streak) >= int(step_pre_desc_true_required)
                progress_text = f"true-streak={int(true_down_streak)}/{int(step_pre_desc_true_required)}"
            pulse_score = int(step_pre_desc_score_after_seen_true) if bool(saw_true) else int(step_pre_desc_score)
            visible_now = bool((getattr(world, "brick", {}) or {}).get("visible"))
            if (step_pre_desc_score_when_visible is not None) and bool(visible_now):
                pulse_score = min(int(pulse_score), int(step_pre_desc_score_when_visible))
            pulse_speed = telemetry_robot_module.manual_speed_for_cmd(step_pre_desc_cmd, pulse_score)
            if not align_silent:
                crosshair_text = "WAIT" if crosshair_now is None else ("YES" if crosshair_now else "NO")
                gate_word = "PASS" if bool(gate_done) else "FAIL"
                samples_note = ""
                if int(step_pre_desc_confirm_frames) > 1:
                    samples_note = (
                        f", obs{int(step_pre_desc_confirm_frames)}=[{_crosshair_sample_text(crosshair_samples)}]"
                    )
                next_action = "STOP" if bool(gate_done) else f"{str(step_pre_desc_cmd).upper()} {int(pulse_score)}%"
                _print_exception_line(
                    (
                        f"[{step_key}] Pre-align descend pulse "
                        f"{int(idx)}/{int(step_pre_desc_max_acts)}: "
                        f"speed={int(pulse_score)}%, inCrosshairs={crosshair_text}{samples_note}, "
                        f"{progress_text}, gate={gate_word}, next={next_action}, obs-checks={int(observation_checks)}."
                    )
                )
            if gate_done:
                done = True
                break
            if confirm_callback and not confirm_callback(world, vision):
                return {"enabled": True, "success": False, "reason": "confirm cancelled"}
            if observer:
                observer("analysis", world, vision, step_pre_desc_cmd, pulse_speed, "pre_align_descend")
            send_robot_command(
                robot,
                world,
                step,
                step_pre_desc_cmd,
                pulse_speed,
                speed_score=pulse_score,
                auto_mode=True,
            )
            if observer:
                observer("action", world, vision, step_pre_desc_cmd, pulse_speed, "pre_align_descend")
            time.sleep(CONTROL_DT)
        if not bool(done):
            if step_pre_desc_completion_mode == "true_then_false_streak":
                fail_reason = (
                    f"pre-align descend could not confirm inCrosshairs=YES then NO x{int(step_pre_desc_false_after_true_required)}"
                )
            else:
                fail_reason = f"pre-align descend could not confirm inCrosshairs=YES x{int(step_pre_desc_true_required)}"
            if not align_silent:
                _print_exception_line(
                    f"[{step_key}] Pre-align descend complete: gate=FAIL."
                )
            return {"enabled": True, "success": False, "reason": fail_reason}
        if not align_silent:
            _print_exception_line(
                f"[{step_key}] Pre-align descend complete: gate=PASS."
            )
        return {"enabled": True, "success": True, "reason": "pre-align descend pass"}

    def _run_step_post_success_mast_exception():
        nonlocal step_post_mast_run_once
        if not bool(step_post_mast_enabled) or int(step_post_mast_acts) <= 0:
            return {"enabled": False, "success": True, "reason": "disabled"}
        if bool(step_post_mast_run_once):
            return {"enabled": True, "success": True, "reason": "already completed"}
        step_post_mast_run_once = True
        step_post_mast_speed = telemetry_robot_module.manual_speed_for_cmd(
            step_post_mast_cmd,
            step_post_mast_score,
        )
        if not align_silent:
            if step_post_mast_reminder:
                _print_exception_line(f"[{step_key}] {step_post_mast_reminder}")
            _print_exception_line(
                (
                    f"[{step_key}] Post-success mast pulse start: "
                    f"{str(step_post_mast_cmd).upper()} {int(step_post_mast_score)}% "
                    f"x{int(step_post_mast_acts)} (observe_between_acts="
                    f"{'YES' if bool(step_post_mast_observe_between_acts) else 'NO'})."
                )
            )
        for idx in range(1, int(step_post_mast_acts) + 1):
            if confirm_callback and not confirm_callback(world, vision):
                return {"enabled": True, "success": False, "reason": "confirm cancelled"}
            if observer:
                observer(
                    "analysis",
                    world,
                    vision,
                    step_post_mast_cmd,
                    step_post_mast_speed,
                    "post_success_mast_exception",
                )
            send_robot_command(
                robot,
                world,
                step,
                step_post_mast_cmd,
                step_post_mast_speed,
                speed_score=step_post_mast_score,
                auto_mode=True,
            )
            if bool(step_post_mast_observe_between_acts):
                post_act_analysis(world, vision, step=step, log=not align_silent, include_pause=False)
                update_world_from_vision(world, vision, log=not align_silent)
            if observer:
                observer(
                    "action",
                    world,
                    vision,
                    step_post_mast_cmd,
                    step_post_mast_speed,
                    "post_success_mast_exception",
                )
            if not align_silent:
                _print_exception_line(
                    (
                        f"[{step_key}] Post-success mast pulse "
                        f"{int(idx)}/{int(step_post_mast_acts)}: "
                        f"{str(step_post_mast_cmd).upper()} {int(step_post_mast_score)}%."
                    )
                )
            if step_post_mast_pause_s > 0.0:
                time.sleep(float(step_post_mast_pause_s))
        if not align_silent:
            _print_exception_line(
                f"[{step_key}] Post-success mast pulse complete."
            )
        return {"enabled": True, "success": True, "reason": "post-success mast pulse complete"}

    def _complete_alignment_success(reason_text, tracker=None):
        # Stop immediately when we declare success so no queued/lingering
        # micro-pulse can fire after a passing gatecheck.
        if robot:
            robot.stop()
        post_mast_result = _run_step_post_success_mast_exception()
        if bool((post_mast_result or {}).get("enabled")) and not bool((post_mast_result or {}).get("success")):
            if robot:
                robot.stop()
            return False, str((post_mast_result or {}).get("reason") or "post-success mast pulse failed")
        post_desc_result = _run_step_post_success_descend()
        if bool((post_desc_result or {}).get("enabled")) and not bool((post_desc_result or {}).get("success")):
            if robot:
                robot.stop()
            return False, str((post_desc_result or {}).get("reason") or "post-success descend failed")
        if robot:
            robot.stop()
        if not align_silent:
            print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
            if tracker is None:
                print_gate_summary_line(world)
            else:
                print_gate_summary_line(world, tracker)
            print_success_events(world, step)
        return True, str(reason_text or "success gate")

    start_ground_reset_cfg = step_rules.get("start_ground_reset_exception") if isinstance(step_rules, dict) else {}
    if not isinstance(start_ground_reset_cfg, dict):
        start_ground_reset_cfg = {}
    start_ground_reset_run_after_startup = bool(start_ground_reset_cfg.get("run_after_startup_action", False))
    start_ground_reset_ran = False
    if not bool(start_ground_reset_run_after_startup):
        start_ground_reset_result = _run_start_ground_reset_exception(
            world,
            vision,
            step,
            robot=robot,
            observer=observer,
            confirm_callback=confirm_callback,
            align_silent=align_silent,
        )
        start_ground_reset_ran = bool((start_ground_reset_result or {}).get("enabled"))
        if bool((start_ground_reset_result or {}).get("enabled")) and not bool((start_ground_reset_result or {}).get("success")):
            pause_after_fail(robot)
            return False, str((start_ground_reset_result or {}).get("reason") or "start ground reset failed")

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
        # Do not early-exit when step-specific startup exceptions are configured.
        # These exception phases are intentional operator-defined behavior and
        # must run before we allow a precheck success to complete the step.
        if (
            is_find_topmost_step
            or bool(step_pre_desc_enabled)
            or bool(startup_action_active)
            or bool(start_ground_reset_run_after_startup and (not start_ground_reset_ran))
        ):
            start_status = "start"
        else:
            return _complete_alignment_success("success gate")
    if start_status != "start":
        pause_after_fail(robot)
        return False, _start_gate_failure_reason_from_wait_status(start_status)

    pre_desc_result = _run_step_pre_align_descend()
    if bool((pre_desc_result or {}).get("enabled")) and not bool((pre_desc_result or {}).get("success")):
        pause_after_fail(robot)
        return False, str((pre_desc_result or {}).get("reason") or "pre-align descend failed")

    ground_up_exc = _run_ground_up_level2_exception(
        world,
        vision,
        step,
        robot=robot,
        observer=observer,
        confirm_callback=confirm_callback,
        align_silent=align_silent,
    )
    if bool((ground_up_exc or {}).get("enabled")) and bool((ground_up_exc or {}).get("handled")):
        if bool((ground_up_exc or {}).get("success")):
            return True, str((ground_up_exc or {}).get("reason") or "success gate + level2")
        pause_after_fail(robot)
        return False, str((ground_up_exc or {}).get("reason") or "ground-up level2 failed")

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
    prev_log_y_err = None
    prev_log_dist = None
    align_action_idx = 0
    stale_pre_obs_streak = 0
    stale_frame_streak = 0
    not_visible_streak = 0
    last_visible_ts = time.time()
    gap_micro_armed = False
    force_gap_switch_after_recovery = False
    last_align_gap_correction_type = None
    skip_align_gap_correction_type_once = None
    recent_gap_correction_types = collections.deque(maxlen=16)
    gap_planner_state = {}
    last_align_action = None
    last_align_signature = None
    unchanged_speed_bump_state = None
    recent_align_acts = collections.deque(maxlen=32)
    align_target_lock = None
    align_target_lock_required = bool(align_target_lock_enabled)
    align_target_lock_bootstrap_done = False
    align_target_lock_soft_bypass_active = False
    loop_id = 0
    last_no_action_gatecheck_reason = None
    last_no_action_gatecheck_log_loop = 0
    no_action_lite_failed_reason = None
    progress_mast_start_dist = None
    progress_mast_triggered = False
    near_dist_vis_attempts = 0
    near_dist_vis_reminder_logged = False
    last_visible_dist_mm = None
    forward_stall_count = 0
    forward_stall_start_dist_err = None
    forward_stall_best_dist_err = None
    forward_stall_reminder_logged = False

    def _correction_type_for_cmd_value(cmd_value):
        cmd_key = str(cmd_value or "").strip().lower()
        if cmd_key in ("f", "b"):
            return "distance"
        if cmd_key in ("l", "r"):
            return "x_axis"
        if cmd_key in ("u", "d"):
            return "y_axis"
        return None

    def _post_recovery_disqualify_types():
        available_types = []
        if isinstance(step_success_gates_cfg, dict):
            if any(key in step_success_gates_cfg for key in ("xAxis_offset_abs", "xAxis_offset", "x_axis")):
                available_types.append("x_axis")
            if any(key in step_success_gates_cfg for key in ("yAxis_offset_abs", "yAxis_offset", "y_axis")):
                available_types.append("y_axis")
            if "dist" in step_success_gates_cfg:
                available_types.append("distance")
        if not available_types:
            available_types = ["x_axis", "y_axis", "distance"]
        max_disqualify = max(1, len(set(available_types)) - 1)
        keep_cap = min(2, max_disqualify)
        picked = []
        primary = None
        if isinstance(last_align_action, dict):
            primary = _correction_type_for_cmd_value(last_align_action.get("cmd"))
        if primary is None:
            primary = str(last_align_gap_correction_type or "").strip().lower()
        if primary in available_types:
            picked.append(primary)
        for corr in reversed(list(recent_gap_correction_types)):
            corr_key = str(corr or "").strip().lower()
            if corr_key not in available_types:
                continue
            if corr_key in picked:
                continue
            picked.append(corr_key)
            if len(picked) >= keep_cap:
                break
        return list(picked[:keep_cap])

    def _set_post_recovery_disqualify_types(*, source="recovery"):
        nonlocal skip_align_gap_correction_type_once
        skip_align_gap_correction_type_once = _post_recovery_disqualify_types()
        if not align_silent and bool(skip_align_gap_correction_type_once):
            corr_text = ", ".join(str(item) for item in skip_align_gap_correction_type_once)
            print(
                format_headline(
                    f"[RECOVERY] next-loop disqualified gap types ({str(source)}): {corr_text}",
                    COLOR_ORANGE_BRIGHT,
                )
            )

    def _active_disqualified_gap_types():
        raw = skip_align_gap_correction_type_once
        if raw is None:
            return set()
        if isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = [raw]
        out = set()
        for item in items:
            corr = str(item or "").strip().lower()
            if corr in {"x_axis", "y_axis", "distance"}:
                out.add(corr)
        return out

    def _clear_align_brick_target_lock(*, clear_required=False):
        nonlocal align_target_lock, align_target_lock_required
        align_target_lock = None
        if clear_required:
            align_target_lock_required = False
        try:
            world._align_brick_target_lock = None
        except Exception:
            pass

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

    def _forward_stall_dist_err_from_local_gate(local_gate):
        if not isinstance(local_gate, dict):
            return None
        dist_now = _as_float(local_gate.get("dist"), None)
        dist_target_now = _as_float(local_gate.get("dist_target"), None)
        if dist_target_now is None:
            dist_target_now = _as_float(forward_stall_dist_target, None)
        if dist_now is None or dist_target_now is None:
            return None
        try:
            dist_err_mm = abs(float(dist_now) - float(dist_target_now))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(dist_err_mm):
            return None
        return float(dist_err_mm)

    def _reset_forward_stall_tracking(*, clear_reminder=False):
        nonlocal forward_stall_count
        nonlocal forward_stall_start_dist_err
        nonlocal forward_stall_best_dist_err
        nonlocal forward_stall_reminder_logged
        forward_stall_count = 0
        forward_stall_start_dist_err = None
        forward_stall_best_dist_err = None
        if clear_reminder:
            forward_stall_reminder_logged = False

    def _record_forward_stall_progress(*, cmd_value, dist_err_before_action, is_search_action=False):
        nonlocal forward_stall_count
        nonlocal forward_stall_start_dist_err
        nonlocal forward_stall_best_dist_err
        nonlocal forward_stall_reminder_logged
        if not bool(forward_stall_active):
            return
        cmd_key = str(cmd_value or "").strip().lower()
        if bool(is_search_action) or cmd_key != str(forward_stall_forward_cmd):
            _reset_forward_stall_tracking(clear_reminder=True)
            return
        dist_err_num = _as_float(dist_err_before_action, None)
        if dist_err_num is None:
            _reset_forward_stall_tracking(clear_reminder=True)
            return
        dist_err_num = abs(float(dist_err_num))
        if (
            int(forward_stall_count) <= 0
            or forward_stall_start_dist_err is None
            or forward_stall_best_dist_err is None
        ):
            forward_stall_count = 1
            forward_stall_start_dist_err = float(dist_err_num)
            forward_stall_best_dist_err = float(dist_err_num)
            return
        forward_stall_count = int(forward_stall_count) + 1
        forward_stall_best_dist_err = min(float(forward_stall_best_dist_err), float(dist_err_num))
        improvement_mm = max(0.0, float(forward_stall_start_dist_err) - float(forward_stall_best_dist_err))
        if float(improvement_mm) > float(forward_stall_min_improvement_mm):
            # Meaningful forward progress observed; restart a fresh streak from
            # the current best distance error so future stalls can still trigger.
            forward_stall_start_dist_err = float(forward_stall_best_dist_err)
            forward_stall_count = 1
            forward_stall_reminder_logged = False

    def _forward_stall_should_trigger(*, cmd_value, is_search_action=False):
        if not bool(forward_stall_active):
            return False, None
        cmd_key = str(cmd_value or "").strip().lower()
        if bool(is_search_action) or cmd_key != str(forward_stall_forward_cmd):
            return False, None
        if int(forward_stall_count) < int(forward_stall_trigger_acts):
            return False, None
        if forward_stall_start_dist_err is None or forward_stall_best_dist_err is None:
            return False, None
        improvement_mm = max(0.0, float(forward_stall_start_dist_err) - float(forward_stall_best_dist_err))
        return bool(float(improvement_mm) <= float(forward_stall_min_improvement_mm)), float(improvement_mm)

    def _run_forward_stall_mast_exception(*, improvement_mm):
        nonlocal last_align_action
        nonlocal forward_stall_reminder_logged
        if not bool(forward_stall_active):
            return "skipped"
        if not align_silent and forward_stall_reminder and not bool(forward_stall_reminder_logged):
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] {forward_stall_reminder}",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
            forward_stall_reminder_logged = True
        if not align_silent:
            print(
                format_headline(
                    (
                        f"[{step_key} EXCEPTION] Forward dist stall detected: "
                        f"{int(forward_stall_count)}/{int(forward_stall_trigger_acts)} "
                        f"{str(forward_stall_forward_cmd).upper()} acts, dist improvement "
                        f"{float(improvement_mm or 0.0):.2f}mm "
                        f"(need > {float(forward_stall_min_improvement_mm):.2f}mm); executing "
                        f"{str(forward_stall_cmd).upper()} {int(forward_stall_score)}% x{int(forward_stall_acts)}."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        for idx in range(1, int(forward_stall_acts) + 1):
            if confirm_callback and not confirm_callback(world, vision):
                return "confirm cancelled"
            cmd_local = str(forward_stall_cmd)
            score_local = int(forward_stall_score)
            speed_local = telemetry_robot_module.manual_speed_for_cmd(cmd_local, score_local)
            action_tag = "forward_dist_stall_mast_exception"
            if observer:
                observer("analysis", world, vision, cmd_local, speed_local, action_tag)
            action_meta_local = send_robot_command(
                robot,
                world,
                step,
                cmd_local,
                speed_local,
                speed_score=score_local,
                auto_mode=True,
            )
            recent_align_acts.append({"cmd": str(cmd_local), "score": int(score_local)})
            if isinstance(action_meta_local, dict):
                last_align_action = {
                    "cmd": str(cmd_local),
                    "cmd_sent": action_meta_local.get("cmd_sent"),
                    "score": int(score_local),
                    "pwm": action_meta_local.get("pwm"),
                    "power": action_meta_local.get("power"),
                    "duration_ms": action_meta_local.get("duration_ms"),
                }
            else:
                last_align_action = {"cmd": str(cmd_local), "score": int(score_local)}
            if confirm_callback and robot:
                robot.stop()
            post_act_analysis(
                world,
                vision,
                step=step,
                log=not align_silent,
                include_pause=False,
            )
            if observer:
                observer("action", world, vision, cmd_local, speed_local, action_tag)
            if not align_silent:
                print(
                    format_headline(
                        f"[{step_key} EXCEPTION] Pulse {int(idx)}/{int(forward_stall_acts)} complete.",
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            time.sleep(CONTROL_DT)
        if not align_silent:
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] Completed {str(forward_stall_cmd).upper()} {int(forward_stall_score)}% x{int(forward_stall_acts)}; resuming dist_err.",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        _reset_forward_stall_tracking(clear_reminder=True)
        return "ran"

    def _maybe_run_progress_mast_exception():
        nonlocal progress_mast_start_dist, progress_mast_triggered, last_align_action
        if not bool(progress_mast_active):
            return False, None
        if bool(progress_mast_triggered):
            return False, None
        brick_local = getattr(world, "brick", {}) or {}
        if not bool(brick_local.get("visible")):
            return False, None
        curr_dist_local = _as_float(brick_local.get("dist"), None)
        if curr_dist_local is None:
            return False, None
        if progress_mast_start_dist is None:
            progress_mast_start_dist = float(curr_dist_local)
            return False, None

        start_dist_local = float(progress_mast_start_dist)
        target_dist_local = float(progress_mast_target_dist)
        start_to_target = float(target_dist_local - start_dist_local)
        total_gap_abs = abs(float(start_to_target))
        if total_gap_abs < float(progress_mast_min_gap_mm):
            progress_mast_triggered = True
            if not align_silent:
                print(
                    format_headline(
                        (
                            f"[{step_key} EXCEPTION] Progress mast exception skipped: "
                            f"start dist too close to target (start={start_dist_local:.2f}, "
                            f"target={target_dist_local:.2f}, gap={total_gap_abs:.2f}mm)."
                        ),
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            return False, None

        start_to_curr = float(curr_dist_local - start_dist_local)
        if float(start_to_target) * float(start_to_curr) <= 0.0:
            return False, None

        progress_ratio = min(1.0, max(0.0, abs(float(start_to_curr)) / float(total_gap_abs)))
        if float(progress_ratio) + 1e-9 < float(progress_mast_trigger_fraction):
            return False, None

        progress_mast_triggered = True
        if not align_silent:
            if progress_mast_reminder:
                print(
                    format_headline(
                        f"[{step_key} EXCEPTION] {progress_mast_reminder}",
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            print(
                format_headline(
                    (
                        f"[{step_key} EXCEPTION] Triggered at {float(progress_ratio) * 100.0:.1f}% "
                        f"dist progress (start={start_dist_local:.2f}, now={float(curr_dist_local):.2f}, "
                        f"target={target_dist_local:.2f}); executing "
                        f"{str(progress_mast_cmd).upper()} {int(progress_mast_score)}% x{int(progress_mast_acts)}."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )

        for idx in range(1, int(progress_mast_acts) + 1):
            if confirm_callback and not confirm_callback(world, vision):
                return True, "confirm cancelled"
            cmd_local = str(progress_mast_cmd)
            score_local = int(progress_mast_score)
            speed_local = telemetry_robot_module.manual_speed_for_cmd(cmd_local, score_local)
            action_tag = "progress_mast_exception"
            if observer:
                observer("analysis", world, vision, cmd_local, speed_local, action_tag)
            action_meta_local = send_robot_command(
                robot,
                world,
                step,
                cmd_local,
                speed_local,
                speed_score=score_local,
                auto_mode=True,
            )
            recent_align_acts.append({"cmd": str(cmd_local), "score": int(score_local)})
            if isinstance(action_meta_local, dict):
                last_align_action = {
                    "cmd": str(cmd_local),
                    "cmd_sent": action_meta_local.get("cmd_sent"),
                    "score": int(score_local),
                    "pwm": action_meta_local.get("pwm"),
                    "power": action_meta_local.get("power"),
                    "duration_ms": action_meta_local.get("duration_ms"),
                }
            else:
                last_align_action = {"cmd": str(cmd_local), "score": int(score_local)}

            segments_local = []
            if isinstance(action_meta_local, dict) and isinstance(action_meta_local.get("segments"), list):
                segments_local = [seg for seg in action_meta_local.get("segments") if isinstance(seg, dict)]
            elif isinstance(action_meta_local, dict):
                segments_local = [action_meta_local]
            for seg_local in segments_local:
                duration_ms_local = (
                    seg_local.get("duration_model_ms")
                    or seg_local.get("duration_ms")
                    or int(CONTROL_DT * 1000)
                )
                power_local = seg_local.get("power")
                if power_local is None:
                    power_local = speed_local
                evt_local = MotionEvent(
                    cmd_to_motion_type(cmd_local),
                    int(float(power_local) * 255),
                    int(duration_ms_local),
                    speed_score=seg_local.get("score_effective") or seg_local.get("score_model"),
                )
                world.update_from_motion(evt_local)

            if confirm_callback and robot:
                robot.stop()
            post_act_analysis(
                world,
                vision,
                step=step,
                log=not align_silent,
                include_pause=False,
            )
            if observer:
                observer("action", world, vision, cmd_local, speed_local, action_tag)
            if not align_silent:
                print(
                    format_headline(
                        f"[{step_key} EXCEPTION] Pulse {int(idx)}/{int(progress_mast_acts)} complete.",
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            time.sleep(CONTROL_DT)

        if not align_silent:
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] Completed {str(progress_mast_cmd).upper()} {int(progress_mast_score)}% x{int(progress_mast_acts)}; resuming alignment.",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        return True, None

    def _near_dist_vis_gap_status():
        target = _as_float(near_dist_vis_target_dist, None)
        if target is None:
            return None
        candidate = _as_float(last_visible_dist_mm, None)
        if candidate is None or candidate <= 0.0:
            local_gate = _align_local_gate_status_single_source(world, step)
            candidate = _as_float((local_gate or {}).get("dist"), None)
        if candidate is None:
            return None
        try:
            candidate = float(candidate)
            target = float(target)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(candidate) or not math.isfinite(target):
            return None
        return {
            "dist": float(candidate),
            "gap_mm": abs(float(candidate) - float(target)),
        }

    def _maybe_run_near_dist_visibility_mast_exception():
        nonlocal near_dist_vis_attempts
        nonlocal near_dist_vis_reminder_logged
        nonlocal last_align_action
        nonlocal last_visible_dist_mm
        if not bool(near_dist_vis_active):
            return "skipped"
        if int(near_dist_vis_attempts) >= int(near_dist_vis_max_attempts):
            return "skipped"
        gap_status = _near_dist_vis_gap_status()
        if not isinstance(gap_status, dict):
            return "skipped"
        gap_mm = _as_float(gap_status.get("gap_mm"), None)
        dist_now = _as_float(gap_status.get("dist"), None)
        if gap_mm is None or dist_now is None:
            return "skipped"
        if float(gap_mm) > float(near_dist_vis_gap_mm_max):
            return "skipped"

        attempt_idx = int(near_dist_vis_attempts) + 1
        if not align_silent and near_dist_vis_reminder and not bool(near_dist_vis_reminder_logged):
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] {near_dist_vis_reminder}",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
            near_dist_vis_reminder_logged = True
        if not align_silent:
            print(
                format_headline(
                    (
                        f"[{step_key} EXCEPTION] Near-target visibility rescue {int(attempt_idx)}/{int(near_dist_vis_max_attempts)}: "
                        f"visible=NO, dist={float(dist_now):.2f}mm, gap={float(gap_mm):.2f}mm "
                        f"(threshold <= {float(near_dist_vis_gap_mm_max):.2f}mm = "
                        f"{float(near_dist_vis_ratio_max) * 100.0:.0f}% of tol {float(near_dist_vis_tol):.2f}mm); "
                        f"executing {str(near_dist_vis_cmd).upper()} {int(near_dist_vis_score)}%."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )

        if confirm_callback and not confirm_callback(world, vision):
            return "confirm cancelled"

        cmd_local = str(near_dist_vis_cmd)
        score_local = int(near_dist_vis_score)
        speed_local = telemetry_robot_module.manual_speed_for_cmd(cmd_local, score_local)
        action_tag = "near_dist_visibility_mast_exception"
        if observer:
            observer("analysis", world, vision, cmd_local, speed_local, action_tag)
        action_meta_local = send_robot_command(
            robot,
            world,
            step,
            cmd_local,
            speed_local,
            speed_score=score_local,
            auto_mode=True,
        )
        near_dist_vis_attempts = int(attempt_idx)
        recent_align_acts.append({"cmd": str(cmd_local), "score": int(score_local)})
        if isinstance(action_meta_local, dict):
            last_align_action = {
                "cmd": str(cmd_local),
                "cmd_sent": action_meta_local.get("cmd_sent"),
                "score": int(score_local),
                "pwm": action_meta_local.get("pwm"),
                "power": action_meta_local.get("power"),
                "duration_ms": action_meta_local.get("duration_ms"),
            }
        else:
            last_align_action = {"cmd": str(cmd_local), "score": int(score_local)}
        if observer:
            observer("action", world, vision, cmd_local, speed_local, action_tag)

        if _observe_visible(float(near_dist_vis_observe_s)):
            update_world_from_vision(world, vision, log=not align_silent)
            restored_dist = _as_float((getattr(world, "brick", {}) or {}).get("dist"), None)
            if restored_dist is not None:
                last_visible_dist_mm = float(restored_dist)
            near_dist_vis_attempts = 0
            near_dist_vis_reminder_logged = False
            if not align_silent:
                print(
                    format_headline(
                        (
                            f"[{step_key} EXCEPTION] Near-target visibility rescue recovered visibility "
                            f"after {str(cmd_local).upper()} {int(score_local)}%."
                        ),
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            return "recovered"

        if not align_silent:
            print(
                format_headline(
                    (
                        f"[{step_key} EXCEPTION] Near-target visibility rescue "
                        f"{int(attempt_idx)}/{int(near_dist_vis_max_attempts)} did not recover visibility."
                    ),
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        return "attempted"

    def _recover_visibility(source="unknown", reason=None, reverse_max_acts=None):
        source_key = str(source or "").strip().lower()
        if bool(align_target_lock_enabled):
            # Recovery moves the robot, so any existing target lock anchor is stale.
            _clear_align_brick_target_lock(clear_required=False)
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
        try:
            reverse_limit = int(reverse_max_acts) if reverse_max_acts is not None else int(ALIGN_RECOVERY_REVERSE_MAX_ACTS)
        except (TypeError, ValueError):
            reverse_limit = int(ALIGN_RECOVERY_REVERSE_MAX_ACTS)
        reverse_limit = max(1, int(reverse_limit))
        for row in reversed(list(recent_align_acts)[-int(reverse_limit):]):
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
            print(format_headline("[RECOVERY] Reverse phase did not restore visibility.", COLOR_ORANGE_BRIGHT))
            print(format_headline("[RECOVERY] Failed to restore visibility.", COLOR_RED))
        return False

    def _hold_and_recheck_visibility(rounds=ALIGN_RECOVERY_PREMOVE_RECHECK_ROUNDS):
        try:
            rounds_local = max(1, int(rounds))
        except (TypeError, ValueError):
            rounds_local = int(ALIGN_RECOVERY_PREMOVE_RECHECK_ROUNDS)
        hold_window_s = visibility_recheck_hold_seconds(vision)
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

    def _reset_stack_visibility_gate_state(*, log_reason=None):
        if not step_key in {"FIND_TOPMOST_BRICK", "FIND_TOPMOST_BRICK_WALL", "ALIGN_BRICK"}:
            return
        try:
            world._stack_visibility_gate = None
            world._stack_gatecheck_log_sig = None
            world._stack_gatecheck_log_enabled = True
            if isinstance(getattr(world, "brick", None), dict):
                world.brick["brickAbove"] = None
                world.brick["brickBelow"] = None
        except Exception:
            pass
        if not align_silent:
            suffix = f" ({str(log_reason).strip()})" if log_reason else ""
            print(format_headline(f"[ALIGN_TOP] reset stack gate state{suffix}.", COLOR_WHITE))

    def _align_brick_target_lock_frame(*, local_gate=None):
        brick = getattr(world, "brick", {}) or {}
        if local_gate is None:
            local_gate = _align_local_gate_status_single_source(
                world,
                align_gate_step_key,
            )

        def _num(value):
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        visible = bool(brick.get("visible"))
        raw_crosshair = brick.get("inCrosshairs")
        if raw_crosshair is None:
            raw_crosshair = brick.get("in_crosshairs")
        in_crosshairs_now = None if raw_crosshair is None else bool(raw_crosshair)
        return {
            "visible": visible,
            "in_crosshairs": in_crosshairs_now,
            "above": (brick.get("brickAbove") if visible else None),
            "below": (brick.get("brickBelow") if visible else None),
            "x": _num(brick.get("x_axis", brick.get("offset_x"))),
            "y": _num(brick.get("y_axis", brick.get("offset_y"))),
            "dist": _num(brick.get("dist")),
            "x_target": _num((local_gate or {}).get("x_target")),
            "y_target": _num((local_gate or {}).get("y_target")),
            "y_required": bool((local_gate or {}).get("y_required")),
            "y_tol": abs(_num((local_gate or {}).get("y_tol")) or 0.0),
            "x_tol": abs(_num((local_gate or {}).get("x_tol")) or 0.0),
            "dist_tol": abs(_num((local_gate or {}).get("dist_tol")) or 0.0),
        }

    def _align_brick_target_lock_eval(frame, *, anchor=None, guard_mode=False):
        if not isinstance(frame, dict):
            return False, "no_frame"
        if not bool(frame.get("visible")):
            return False, "not_visible"

        above_now = frame.get("above")
        below_now = frame.get("below")
        topmost_like = bool((above_now is False) and (below_now is True))
        stack_uncertain = bool((above_now is None) or (below_now is None))
        if not bool(align_target_lock_stack_signature_guard_enabled):
            topmost_like = False
            stack_uncertain = False

        x_now = frame.get("x")
        x_target = frame.get("x_target")
        y_now = frame.get("y")
        y_target = frame.get("y_target")
        y_required = bool(frame.get("y_required"))
        in_crosshairs_now = bool(frame.get("in_crosshairs") is True)
        y_tol = abs(float(frame.get("y_tol") or 0.0))
        if y_now is None and bool(y_required) and not bool(in_crosshairs_now):
            return False, "missing_y"
        if bool(y_required) and y_target is not None and not bool(in_crosshairs_now):
            y_band_scale = 1.8 if guard_mode else 1.4
            y_band = max(float(y_tol) * float(y_band_scale), 1.8)
            if abs(float(y_now) - float(y_target)) > float(y_band):
                if topmost_like:
                    return False, "topmost_signature"
                return False, "y_out_of_band"
        if topmost_like and y_target is None:
            return False, "topmost_signature"

        if isinstance(anchor, dict):
            ax = anchor.get("anchor_x")
            ay = anchor.get("anchor_y")
            ad = anchor.get("anchor_dist")
            d_now = frame.get("dist")
            x_jump_limit = max(abs(float(frame.get("x_tol") or 0.0)) * 3.0, 4.0)
            y_jump_limit = max(float(y_tol) * (2.0 if guard_mode else 1.6), 2.5)
            d_jump_limit = max(abs(float(frame.get("dist_tol") or 0.0)) * 4.0, 8.0)
            try:
                if (ax is not None) and (x_now is not None) and abs(float(x_now) - float(ax)) > float(x_jump_limit):
                    return False, "x_jump"
                if (ay is not None) and (y_now is not None) and abs(float(y_now) - float(ay)) > float(y_jump_limit):
                    return False, "y_jump"
                if (ad is not None) and (d_now is not None) and abs(float(d_now) - float(ad)) > float(d_jump_limit):
                    return False, "dist_jump"
            except (TypeError, ValueError):
                return False, "jump_parse"
            if bool(align_target_lock_center_prefer_enabled) and not bool(in_crosshairs_now):
                try:
                    anchor_x_err = anchor.get("anchor_center_x_err")
                    if anchor_x_err is None and ax is not None and x_target is not None:
                        anchor_x_err = abs(float(ax) - float(x_target))
                    x_err_now = None if (x_now is None or x_target is None) else abs(float(x_now) - float(x_target))
                    if (
                        anchor_x_err is not None
                        and x_err_now is not None
                        and x_now is not None
                        and ax is not None
                        and float(x_err_now) > (float(anchor_x_err) + float(align_target_lock_center_x_worse_margin_mm))
                        and abs(float(x_now) - float(ax)) > max(float(align_target_lock_center_x_worse_margin_mm), 1.0)
                    ):
                        return False, "x_center_worse"
                    anchor_y_err = anchor.get("anchor_center_y_err")
                    if anchor_y_err is None and ay is not None and y_target is not None:
                        anchor_y_err = abs(float(ay) - float(y_target))
                    y_err_now = None if (y_now is None or y_target is None) else abs(float(y_now) - float(y_target))
                    if (
                        anchor_y_err is not None
                        and y_err_now is not None
                        and y_now is not None
                        and ay is not None
                        and float(y_err_now) > (float(anchor_y_err) + float(align_target_lock_center_y_worse_margin_mm))
                        and abs(float(y_now) - float(ay)) > max(float(align_target_lock_center_y_worse_margin_mm), 1.0)
                    ):
                        return False, "y_center_worse"
                except (TypeError, ValueError):
                    return False, "center_parse"

        if stack_uncertain and not isinstance(anchor, dict) and y_target is None:
            return False, "stack_uncertain"

        return True, "ok"

    def _align_brick_target_lock_anchor_update(anchor, frame):
        if not isinstance(anchor, dict):
            anchor = {}
        for src_key, dst_key in (("x", "anchor_x"), ("y", "anchor_y"), ("dist", "anchor_dist")):
            current_val = frame.get(src_key) if isinstance(frame, dict) else None
            if current_val is None:
                continue
            try:
                current_val = float(current_val)
            except (TypeError, ValueError):
                continue
            prev_val = anchor.get(dst_key)
            if prev_val is None:
                anchor[dst_key] = current_val
                continue
            try:
                prev_val = float(prev_val)
            except (TypeError, ValueError):
                anchor[dst_key] = current_val
                continue
            anchor[dst_key] = (0.7 * prev_val) + (0.3 * current_val)
        x_target = frame.get("x_target") if isinstance(frame, dict) else None
        y_target = frame.get("y_target") if isinstance(frame, dict) else None
        try:
            x_target = float(x_target) if x_target is not None else None
        except (TypeError, ValueError):
            x_target = None
        try:
            y_target = float(y_target) if y_target is not None else None
        except (TypeError, ValueError):
            y_target = None
        if x_target is not None:
            anchor["x_target"] = float(x_target)
        if y_target is not None:
            anchor["y_target"] = float(y_target)
        try:
            if anchor.get("anchor_x") is not None and x_target is not None:
                anchor["anchor_center_x_err"] = abs(float(anchor.get("anchor_x")) - float(x_target))
        except (TypeError, ValueError):
            pass
        try:
            if anchor.get("anchor_y") is not None and y_target is not None:
                anchor["anchor_center_y_err"] = abs(float(anchor.get("anchor_y")) - float(y_target))
        except (TypeError, ValueError):
            pass
        anchor["updated_ts"] = time.time()
        return anchor

    def _publish_align_brick_target_lock_state():
        try:
            if isinstance(align_target_lock, dict):
                world._align_brick_target_lock = {
                    "kind": str(align_target_lock.get("kind") or "center_focus"),
                    "anchor_x": align_target_lock.get("anchor_x"),
                    "anchor_y": align_target_lock.get("anchor_y"),
                    "anchor_dist": align_target_lock.get("anchor_dist"),
                    "x_target": align_target_lock.get("x_target"),
                    "y_target": align_target_lock.get("y_target"),
                    "anchor_center_x_err": align_target_lock.get("anchor_center_x_err"),
                    "anchor_center_y_err": align_target_lock.get("anchor_center_y_err"),
                    "bad_streak": int(align_target_lock.get("bad_streak", 0) or 0),
                    "hold_bad_frames": int(align_target_lock.get("hold_bad_frames", 2) or 2),
                    "acquired_ts": align_target_lock.get("acquired_ts"),
                    "last_reason": align_target_lock.get("last_reason"),
                }
            else:
                world._align_brick_target_lock = None
        except Exception:
            pass

    def _restore_align_brick_target_lock_from_world():
        nonlocal align_target_lock, align_target_lock_required
        if not bool(align_target_lock_enabled):
            return
        raw = getattr(world, "_align_brick_target_lock", None)
        if not isinstance(raw, dict):
            return
        align_target_lock = {
            "kind": str(raw.get("kind") or "center_focus"),
            "anchor_x": raw.get("anchor_x"),
            "anchor_y": raw.get("anchor_y"),
            "anchor_dist": raw.get("anchor_dist"),
            "x_target": raw.get("x_target"),
            "y_target": raw.get("y_target"),
            "anchor_center_x_err": raw.get("anchor_center_x_err"),
            "anchor_center_y_err": raw.get("anchor_center_y_err"),
            "bad_streak": 0,
            "hold_bad_frames": int(raw.get("hold_bad_frames", align_target_lock_hold_bad_frames) or align_target_lock_hold_bad_frames),
            "last_reason": str(raw.get("last_reason") or "restored"),
            "acquired_ts": raw.get("acquired_ts") or time.time(),
            "updated_ts": time.time(),
        }
        align_target_lock_required = bool(align_target_lock_enabled)
        _publish_align_brick_target_lock_state()

    def _acquire_second_uppermost_target_lock(*, source="align_top_step_down", timeout_s=None, confirm_frames=None):
        nonlocal align_target_lock, align_target_lock_required
        if not bool(align_target_lock_enabled):
            return True, "ready"

        try:
            confirm_needed = max(
                1,
                int(confirm_frames if confirm_frames is not None else align_target_lock_confirm_frames),
            )
        except (TypeError, ValueError):
            confirm_needed = int(max(1, align_target_lock_confirm_frames))
        try:
            timeout_local = (
                float(timeout_s)
                if timeout_s is not None
                else max(float(CONTROL_DT) * float(confirm_needed + 1), float(align_target_lock_acquire_timeout_s))
            )
        except (TypeError, ValueError):
            timeout_local = max(float(CONTROL_DT), float(align_target_lock_acquire_timeout_s))
        deadline = time.time() + max(float(CONTROL_DT), float(timeout_local))
        good_streak = 0
        candidate_anchor = None
        while time.time() < deadline:
            if robot:
                robot.stop()
            update_world_from_vision(world, vision, log=not align_silent)
            if observer:
                observer("frame", world, vision, None, None, None)
            local_gate = _align_local_gate_status_single_source(
                world,
                align_gate_step_key,
            )
            frame = _align_brick_target_lock_frame(local_gate=local_gate)
            ok_frame, reason_key = _align_brick_target_lock_eval(
                frame,
                anchor=candidate_anchor,
                guard_mode=False,
            )
            if ok_frame:
                good_streak += 1
                candidate_anchor = _align_brick_target_lock_anchor_update(candidate_anchor, frame)
            else:
                good_streak = 0
                candidate_anchor = None

            if ok_frame and good_streak >= confirm_needed:
                align_target_lock = {
                    "kind": "center_focus",
                    "anchor_x": candidate_anchor.get("anchor_x") if isinstance(candidate_anchor, dict) else None,
                    "anchor_y": candidate_anchor.get("anchor_y") if isinstance(candidate_anchor, dict) else None,
                    "anchor_dist": candidate_anchor.get("anchor_dist") if isinstance(candidate_anchor, dict) else None,
                    "x_target": candidate_anchor.get("x_target") if isinstance(candidate_anchor, dict) else None,
                    "y_target": candidate_anchor.get("y_target") if isinstance(candidate_anchor, dict) else None,
                    "anchor_center_x_err": (
                        candidate_anchor.get("anchor_center_x_err") if isinstance(candidate_anchor, dict) else None
                    ),
                    "anchor_center_y_err": (
                        candidate_anchor.get("anchor_center_y_err") if isinstance(candidate_anchor, dict) else None
                    ),
                    "bad_streak": 0,
                    "hold_bad_frames": int(align_target_lock_hold_bad_frames),
                    "last_reason": "ok",
                    "acquired_ts": time.time(),
                    "updated_ts": time.time(),
                }
                align_target_lock_required = bool(align_target_lock_enabled)
                _publish_align_brick_target_lock_state()
                if not align_silent:
                    print(
                        format_headline(
                            (
                                f"[ALIGN_LOCK] acquired center-focus lock: "
                                f"x={align_target_lock.get('anchor_x')}, y={align_target_lock.get('anchor_y')}."
                            ),
                            COLOR_WHITE,
                        )
                    )
                return True, "ready"

            time.sleep(CONTROL_DT)

        _clear_align_brick_target_lock(clear_required=False)
        return False, "center-focus target lock not confirmed"

    def _guard_second_uppermost_target_lock():
        nonlocal align_target_lock
        if not bool(align_target_lock_enabled):
            return "skip"
        if not align_target_lock_required:
            return "skip"
        if not isinstance(align_target_lock, dict):
            return "missing"

        local_gate = _align_local_gate_status_single_source(
            world,
            step,
        )
        frame = _align_brick_target_lock_frame(local_gate=local_gate)
        ok_frame, reason_key = _align_brick_target_lock_eval(
            frame,
            anchor=align_target_lock,
            guard_mode=True,
        )
        if ok_frame:
            align_target_lock["bad_streak"] = 0
            align_target_lock["last_reason"] = "ok"
            _align_brick_target_lock_anchor_update(align_target_lock, frame)
            _publish_align_brick_target_lock_state()
            return "ok"

        align_target_lock["last_reason"] = str(reason_key or "unknown")
        _publish_align_brick_target_lock_state()
        if str(reason_key) == "not_visible":
            return "skip"

        bad_streak = int(align_target_lock.get("bad_streak", 0) or 0) + 1
        align_target_lock["bad_streak"] = int(bad_streak)
        hold_bad_frames = max(1, int(align_target_lock.get("hold_bad_frames", 2) or 2))
        _publish_align_brick_target_lock_state()

        if bad_streak <= hold_bad_frames:
            # Advisory-only: do not block or spam hold UI during micro-adjust flow.
            return "hold"
        _clear_align_brick_target_lock(clear_required=False)
        return "lost"

    topmost_seek_moved_last = False
    topmost_seek_last_fail_reason = None

    def _seek_align_brick_topmost_target(*, allow_fail_backoff=True):
        if not is_find_topmost_step:
            return "ready"
        nonlocal last_align_action, topmost_seek_moved_last, topmost_seek_last_fail_reason
        _clear_align_brick_target_lock(clear_required=True)
        topmost_seek_moved_last = False
        topmost_seek_last_fail_reason = None

        def _topmost_pass_counts_text():
            if not bool(topmost_requires_brick_above_false):
                return None
            status = getattr(world, "_stack_visibility_gate", None)
            if not isinstance(status, dict):
                return None
            above = status.get("above") if isinstance(status.get("above"), dict) else {}
            row = above.get("full_false") if isinstance(above.get("full_false"), dict) else {}
            if not row:
                return None
            try:
                consec_seen = int(row.get("streak", 0) or 0)
                consec_need = int(row.get("need", 1) or 1)
                maj_seen = int(row.get("window_pass", 0) or 0)
                maj_total = int(row.get("window_total", 1) or 1)
            except Exception:
                return None
            return (
                f"brickAbove=NO counts: consec {consec_seen}/{consec_need}, "
                f"maj {maj_seen}/{maj_total}"
            )

        def _run_topmost_crosshair_exception_post_success():
            nonlocal last_align_action, topmost_seek_moved_last, topmost_seek_last_fail_reason
            if not (is_find_topmost_step and topmost_exception_enabled and topmost_requires_crosshair_false):
                return True, "disabled"
            if not align_silent:
                if topmost_exception_reminder:
                    print(format_headline(f"[ALIGN_TOP] {topmost_exception_reminder}", COLOR_ORANGE_BRIGHT))
                print(
                    format_headline(
                        "[ALIGN_TOP] Special exception: post-success descent uses inCrosshairs NO->YES->NO to park just below the topmost brick.",
                        COLOR_ORANGE_BRIGHT,
                    )
                )

            def _apply_motion_segments(cmd_local, speed_local, action_meta_local):
                segments_local = []
                if isinstance(action_meta_local, dict) and isinstance(action_meta_local.get("segments"), list):
                    segments_local = [seg for seg in action_meta_local.get("segments") if isinstance(seg, dict)]
                elif isinstance(action_meta_local, dict):
                    segments_local = [action_meta_local]
                for seg_local in segments_local:
                    duration_ms_local = (
                        seg_local.get("duration_model_ms")
                        or seg_local.get("duration_ms")
                        or int(CONTROL_DT * 1000)
                    )
                    power_local = seg_local.get("power")
                    if power_local is None:
                        power_local = speed_local
                    evt_local = MotionEvent(
                        cmd_to_motion_type(cmd_local),
                        int(float(power_local) * 255),
                        int(duration_ms_local),
                        speed_score=seg_local.get("score_effective") or seg_local.get("score_model"),
                    )
                    world.update_from_motion(evt_local)

            def _observe_crosshair_state(phase_label):
                if robot:
                    robot.stop()
                record_hold_display(world, step, f"align_top: {phase_label} observe")
                update_world_from_vision(world, vision, log=not align_silent)
                if observer:
                    observer("frame", world, vision, None, None, None)
                brick_local = getattr(world, "brick", {}) or {}
                visible_local = bool(brick_local.get("visible"))
                in_crosshairs_local = brick_local.get("inCrosshairs")
                in_crosshairs_local = bool(in_crosshairs_local) if in_crosshairs_local is not None else False
                return visible_local, in_crosshairs_local

            def _run_phase(*, phase_label, action_tag, target_in_crosshairs, score_local, max_acts_local):
                nonlocal last_align_action, topmost_seek_moved_last, topmost_seek_last_fail_reason
                acts_local = 0
                target_bool = bool(target_in_crosshairs)
                while acts_local < int(max_acts_local):
                    visible_local, in_crosshairs_local = _observe_crosshair_state(phase_label)
                    if visible_local and (bool(in_crosshairs_local) == target_bool):
                        if robot:
                            robot.stop()
                        if not align_silent:
                            target_text_local = "YES" if target_bool else "NO"
                            print(
                                format_headline(
                                    f"[ALIGN_TOP] {phase_label}: reached inCrosshairs={target_text_local}.",
                                    COLOR_GREEN,
                                )
                            )
                        return True, acts_local
                    if not visible_local:
                        topmost_seek_last_fail_reason = f"{phase_label}: brick not visible"
                        if not align_silent:
                            print(
                                format_headline(
                                    f"[ALIGN_TOP] {phase_label}: visibility lost; cannot continue exception descent.",
                                    COLOR_RED,
                                )
                            )
                        return False, acts_local

                    target_text = "YES" if target_bool else "NO"
                    current_text = "YES" if bool(in_crosshairs_local) else "NO"
                    if not align_silent:
                        print(
                            format_headline(
                                f"[ALIGN_TOP] {phase_label} assess: inCrosshairs={current_text}; "
                                f"plan {topmost_post_desc_cmd.upper()} {int(score_local)}% -> {target_text}.",
                                COLOR_WHITE,
                            )
                        )
                    if confirm_callback:
                        if not confirm_callback(world, vision):
                            topmost_seek_last_fail_reason = "confirm cancelled"
                            return False, acts_local
                    speed_local = telemetry_robot_module.manual_speed_for_cmd(
                        topmost_post_desc_cmd,
                        score_local,
                    )
                    if observer:
                        observer("analysis", world, vision, topmost_post_desc_cmd, speed_local, action_tag)
                    action_meta_local = send_robot_command(
                        robot,
                        world,
                        step,
                        topmost_post_desc_cmd,
                        speed_local,
                        speed_score=score_local,
                        auto_mode=True,
                    )
                    acts_local += 1
                    topmost_seek_moved_last = True
                    recent_align_acts.append({"cmd": str(topmost_post_desc_cmd), "score": int(score_local)})
                    if isinstance(action_meta_local, dict):
                        last_align_action = {
                            "cmd": str(topmost_post_desc_cmd),
                            "cmd_sent": action_meta_local.get("cmd_sent"),
                            "score": int(score_local),
                            "pwm": action_meta_local.get("pwm"),
                            "power": action_meta_local.get("power"),
                            "duration_ms": action_meta_local.get("duration_ms"),
                        }
                    else:
                        last_align_action = {"cmd": str(topmost_post_desc_cmd), "score": int(score_local)}
                    _apply_motion_segments(topmost_post_desc_cmd, speed_local, action_meta_local)
                    if observer:
                        observer("action", world, vision, topmost_post_desc_cmd, speed_local, action_tag)
                    time.sleep(CONTROL_DT)

                topmost_seek_last_fail_reason = (
                    f"{phase_label}: max acts reached ({int(acts_local)}/{int(max_acts_local)})"
                )
                if not align_silent:
                    print(
                        format_headline(
                            f"[ALIGN_TOP] {phase_label}: {topmost_seek_last_fail_reason}.",
                            COLOR_RED,
                        )
                    )
                return False, acts_local

            phase1_ok, _phase1_acts = _run_phase(
                phase_label="exception phase 1",
                action_tag="align_top_exception_seek_true",
                target_in_crosshairs=True,
                score_local=int(topmost_post_seek_true_score),
                max_acts_local=int(topmost_post_seek_true_max_acts),
            )
            if not phase1_ok:
                return False, str(topmost_seek_last_fail_reason or "exception phase 1 failed")

            phase2_ok, _phase2_acts = _run_phase(
                phase_label="exception phase 2",
                action_tag="align_top_exception_seek_false",
                target_in_crosshairs=False,
                score_local=int(topmost_post_seek_false_score),
                max_acts_local=int(topmost_post_seek_false_max_acts),
            )
            if not phase2_ok:
                return False, str(topmost_seek_last_fail_reason or "exception phase 2 failed")

            if topmost_post_auto_continue_delay_s > 0.0:
                if robot:
                    robot.stop()
                record_hold_display(
                    world,
                    step,
                    f"align_top: auto-continue in {float(topmost_post_auto_continue_delay_s):.1f}s",
                )
                if not align_silent:
                    print(
                        format_headline(
                            f"[ALIGN_TOP] Exception complete. Auto-continue after {float(topmost_post_auto_continue_delay_s):.1f}s.",
                            COLOR_WHITE,
                        )
                    )
                time.sleep(max(float(CONTROL_DT), float(topmost_post_auto_continue_delay_s)))

            if not align_silent and topmost_post_continue_blank_lines > 0:
                for _ in range(int(topmost_post_continue_blank_lines)):
                    print("")

            return True, "positioned below topmost brick"

        def _finish_topmost_ready(success_reason):
            nonlocal topmost_seek_last_fail_reason
            topmost_seek_last_fail_reason = str(success_reason or "topmost success gates met")
            if robot:
                robot.stop()
            if not align_silent:
                counts_text = _topmost_pass_counts_text()
                if counts_text:
                    print(format_headline(f"[ALIGN_TOP] pass confidence: {counts_text}", COLOR_WHITE))
            if confirm_callback:
                record_hold_display(world, step, "align_top: manual verify uppermost brick")
                if not align_silent:
                    print(
                        format_headline(
                            "[ALIGN_TOP] Uppermost brick confirmed. Manually verify in livestream, then press Enter to proceed.",
                            COLOR_WHITE,
                        )
                    )
                if not confirm_callback(world, vision):
                    topmost_seek_last_fail_reason = "confirm cancelled"
                    return "confirm cancelled"
                if robot:
                    robot.stop()
            post_ok, post_reason = _run_topmost_crosshair_exception_post_success()
            if not post_ok:
                topmost_seek_last_fail_reason = str(post_reason or "topmost exception post-success positioning failed")
                return str(topmost_seek_last_fail_reason)
            _reset_stack_visibility_gate_state(log_reason="topmost confirmed")
            # Step 5 ends at topmost confirmation. Step 6 now handles any descent to
            # lower bricks and target locking.
            return "ready"

        lift_score = max(1, int(ALIGN_BRICK_TOPMOST_LIFT_SCORE))
        max_acts = max(1, int(ALIGN_BRICK_TOPMOST_MAX_ACTS))
        deadline = time.time() + max(float(CONTROL_DT), float(ALIGN_BRICK_TOPMOST_TIMEOUT_S))
        acts = 0
        top_not_visible_streak = 0
        top_last_visible_ts = time.time()
        visible_now = False
        above_now = None
        below_now = None

        if not align_silent:
            seek_requirements = []
            if topmost_requires_crosshair_false:
                seek_requirements.append("inCrosshairs=NO")
            if topmost_requires_brick_above_false:
                seek_requirements.append("brickAbove=NO")
            if topmost_requires_brick_below_false:
                seek_requirements.append("brickBelow=NO")
            seek_requirements_text = ", ".join(seek_requirements) if seek_requirements else "success gates"
            print(
                format_headline(
                    f"[ALIGN_TOP] Seek uppermost brick first (need {seek_requirements_text}).",
                    COLOR_WHITE,
                )
            )
            print(
                format_headline(
                    f"[ALIGN_TOP] Targeting success gates: {topmost_gate_target_text}.",
                    COLOR_WHITE,
                )
            )

        while time.time() < deadline and acts < max_acts:
            update_world_from_vision(world, vision, log=not align_silent)
            if observer:
                observer("frame", world, vision, None, None, None)

            brick = getattr(world, "brick", {}) or {}
            visible_now = bool(brick.get("visible"))
            above_now = brick.get("brickAbove") if visible_now else None
            below_now = brick.get("brickBelow") if visible_now else None
            x_now = brick.get("x_axis", brick.get("offset_x"))
            y_now = brick.get("y_axis", brick.get("offset_y"))
            dist_now = brick.get("dist")
            if visible_now:
                top_not_visible_streak = 0
                top_last_visible_ts = time.time()
            else:
                top_not_visible_streak = int(top_not_visible_streak) + 1

            top_gate_obs = observe_success_gatecheck(
                world,
                step,
                success_tracker,
                phase="align_top",
                log=not align_silent,
            )
            if bool(top_gate_obs.get("success_met")):
                return _finish_topmost_ready("step-5 success gates met")
            if bool(top_gate_obs.get("hold_for_confirm")):
                if robot:
                    robot.stop()
                topmost_seek_last_fail_reason = "confirming success gates"
                record_hold_display(world, step, "align_top: confirming success gates")
                print_success_gate_metric_tallies(
                    world,
                    step,
                    phase="align_top",
                    metrics=tuple(topmost_gate_metrics),
                    log=not align_silent,
                )
                if observer:
                    observer("action", world, vision, None, 0.0, "align_top_confirm_success")
                time.sleep(CONTROL_DT)
                continue

            # In this pre-phase, prefer lift-up to find the topmost brick and rely on the
            # existing hold/recheck visibility handling when the brick is not visible.
            if not visible_now:
                lost_for_s = max(0.0, float(time.time()) - float(top_last_visible_ts))
                confident_loss = bool(
                    int(top_not_visible_streak) >= int(ALIGN_RECOVERY_LOST_VISIBLE_FRAMES)
                    and float(lost_for_s) >= float(ALIGN_RECOVERY_LOST_VISIBLE_MIN_S)
                )
                if confident_loss:
                    reason = (
                        "ALIGN_TOP brick invisible "
                        f"for {int(top_not_visible_streak)} consecutive frames and {float(lost_for_s):.2f}s "
                        f"(threshold {int(ALIGN_RECOVERY_LOST_VISIBLE_FRAMES)} frames / {float(ALIGN_RECOVERY_LOST_VISIBLE_MIN_S):.2f}s)"
                    )
                    premove_recheck = _hold_and_recheck_visibility()
                    if bool((premove_recheck or {}).get("visible")):
                        top_not_visible_streak = 0
                        top_last_visible_ts = time.time()
                        continue
                    if _recover_visibility(
                        source="align_top",
                        reason=reason,
                        reverse_max_acts=3,
                    ):
                        topmost_seek_moved_last = True
                        top_not_visible_streak = 0
                        top_last_visible_ts = time.time()
                        continue
                    topmost_seek_last_fail_reason = "lost visibility recovery failed"
                    return "lost_vision"
                if robot:
                    robot.stop()
                topmost_seek_last_fail_reason = (
                    f"waiting for visible brick ({int(top_not_visible_streak)} invisible frames)"
                )
                record_hold_display(world, step, "align_top: waiting for visible brick")
                if observer:
                    observer("action", world, vision, None, 0.0, "align_top_wait_visible")
                time.sleep(CONTROL_DT)
                continue

            # When stack booleans are explicit success gates, do not move while their
            # signal is uncertain.
            if topmost_requires_brick_above_false and above_now is None:
                if robot:
                    robot.stop()
                topmost_seek_last_fail_reason = "waiting for brickAbove confidence"
                record_hold_display(world, step, "align_top: waiting for brickAbove confidence")
                if observer:
                    observer("action", world, vision, None, 0.0, "align_top_wait_above_confidence")
                time.sleep(CONTROL_DT)
                continue
            if topmost_requires_brick_below_false and below_now is None:
                if robot:
                    robot.stop()
                topmost_seek_last_fail_reason = "waiting for brickBelow confidence"
                record_hold_display(world, step, "align_top: waiting for brickBelow confidence")
                if observer:
                    observer("action", world, vision, None, 0.0, "align_top_wait_below_confidence")
                time.sleep(CONTROL_DT)
                continue

            # Step 5 should only seek the topmost brick and otherwise rely on the
            # existing hold/recheck visibility handling; no special back-up motions.
            cmd = "u"
            speed_score = lift_score
            action_tag = "align_top_seek"
            speed = telemetry_robot_module.manual_speed_for_cmd(cmd, speed_score)
            pre_gate_text = None
            pre_focus = None
            pre_entries = None
            if not suppress_auto_diag:
                pre_entries = _success_gate_entries(world, step)
                pre_gate_text = _auto_gate_snapshot_text(world, step, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return "confirm cancelled"
            if observer:
                observer("analysis", world, vision, cmd, speed, action_tag)
            action_meta = send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=speed_score,
                auto_mode=True,
            )
            acts += 1
            topmost_seek_moved_last = True
            topmost_seek_last_fail_reason = f"moved bot ({int(acts)}/{int(max_acts)} acts used)"

            try:
                hist_score = (
                    action_meta.get("score_effective")
                    if isinstance(action_meta, dict)
                    else None
                )
                if hist_score is None and isinstance(action_meta, dict):
                    hist_score = action_meta.get("score_model")
                if hist_score is None:
                    hist_score = speed_score
                hist_score = int(round(float(hist_score)))
            except (TypeError, ValueError):
                hist_score = int(max(1, int(speed_score or telemetry_robot_module.SPEED_SCORE_MIN)))
            hist_cmd = str(cmd).strip().lower()
            if hist_cmd not in ("f", "b", "l", "r", "u", "d"):
                hist_cmd = str(cmd).strip().lower()
            recent_align_acts.append({"cmd": str(hist_cmd), "score": int(hist_score)})
            if isinstance(action_meta, dict):
                last_align_action = {
                    "cmd": str(cmd),
                    "cmd_sent": action_meta.get("cmd_sent"),
                    "score": hist_score,
                    "pwm": action_meta.get("pwm"),
                    "power": action_meta.get("power"),
                    "duration_ms": action_meta.get("duration_ms"),
                }
            else:
                last_align_action = {"cmd": str(cmd), "score": hist_score}
            action_detail = None
            if not suppress_auto_diag:
                action_detail = auto_action_detail_text(
                    cmd,
                    hist_score,
                    action_meta=action_meta,
                )
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

            if confirm_callback and robot:
                robot.stop()
            post_act_analysis(
                world,
                vision,
                step=step,
                log=not align_silent,
                include_pause=False,
            )
            if observer:
                observer("action", world, vision, cmd, speed, action_tag)
            if not suppress_auto_diag:
                emit_auto_step_diagnostic(
                    world,
                    step,
                    action_detail,
                    pre_gate_text,
                    emit=not align_silent,
                    pre_focus=pre_focus,
                    sent_snapshot=sent_snapshot,
                    action_index=acts,
                    pre_entries=pre_entries,
                )
            time.sleep(CONTROL_DT)

        boundary_gate_obs = observe_success_gatecheck(
            world,
            step,
            success_tracker,
            phase="align_top",
            log=not align_silent,
        )
        if bool(boundary_gate_obs.get("success_met")):
            if not align_silent:
                print(
                    format_headline(
                        "[ALIGN_TOP] Success gates met at timeout/limit boundary; accepting uppermost brick.",
                        COLOR_WHITE,
                    )
            )
            return _finish_topmost_ready("step-5 success gates met at timeout/limit boundary")

        if topmost_requires_brick_above_false and visible_now and above_now is True:
            topmost_seek_last_fail_reason = "brickAbove still YES"
        elif topmost_requires_brick_above_false and visible_now and above_now is None:
            topmost_seek_last_fail_reason = "brickAbove still WAIT"
        elif topmost_requires_brick_below_false and visible_now and below_now is True:
            topmost_seek_last_fail_reason = "brickBelow still YES"
        elif topmost_requires_brick_below_false and visible_now and below_now is None:
            topmost_seek_last_fail_reason = "brickBelow still WAIT"
        elif not visible_now:
            topmost_seek_last_fail_reason = "brick not visible"
        if acts >= max_acts:
            stack_text = []
            if topmost_requires_brick_above_false:
                stack_text.append(
                    "brickAbove="
                    + ("YES" if above_now is True else "NO" if above_now is False else "WAIT")
                )
            if topmost_requires_brick_below_false:
                stack_text.append(
                    "brickBelow="
                    + ("YES" if below_now is True else "NO" if below_now is False else "WAIT")
                )
            stack_suffix = f" {' '.join(stack_text)}" if stack_text else ""
            topmost_seek_last_fail_reason = (
                f"max acts reached ({int(acts)}/{int(max_acts)})"
                + (
                    f"; last obs visible={'YES' if visible_now else 'NO'} "
                    f"{stack_suffix}".rstrip()
                )
            )
        elif time.time() >= deadline:
            stack_text = []
            if topmost_requires_brick_above_false:
                stack_text.append(
                    "brickAbove="
                    + ("YES" if above_now is True else "NO" if above_now is False else "WAIT")
                )
            if topmost_requires_brick_below_false:
                stack_text.append(
                    "brickBelow="
                    + ("YES" if below_now is True else "NO" if below_now is False else "WAIT")
                )
            stack_suffix = f" {' '.join(stack_text)}" if stack_text else ""
            topmost_seek_last_fail_reason = (
                f"timeout after {float(ALIGN_BRICK_TOPMOST_TIMEOUT_S):.2f}s"
                + (
                    f"; last obs visible={'YES' if visible_now else 'NO'} "
                    f"{stack_suffix}".rstrip()
                )
            )
        if robot:
            robot.stop()
        if not align_silent:
            print_success_gate_metric_tallies(
                world,
                step,
                phase="align_top",
                metrics=tuple(topmost_gate_metrics),
                log=True,
                force=True,
            )
            reason_tail = str(topmost_seek_last_fail_reason or "unknown").strip()
            if reason_tail:
                reason_tail = f" reason: {reason_tail}"
                print(
                    format_headline(
                        f"[ALIGN_TOP] Failed to confirm uppermost brick (need {topmost_gate_target_text}).{reason_tail}",
                        COLOR_RED,
                    )
                )
        return "topmost target not confirmed"

    _restore_align_brick_target_lock_from_world()

    topmost_seek_status = "ready"
    topmost_try_limit = max(1, int(ALIGN_BRICK_TOPMOST_GATECHECK_MAX_TRIES))
    if is_find_topmost_step:
        top_try = 0
        reset_stack_gate_before_top_try = False
        while top_try < topmost_try_limit:
            top_try += 1
            if not align_silent:
                print(
                    format_headline(
                        f"[ALIGN_TOP] gatecheck attempt {int(top_try)}/{int(topmost_try_limit)}",
                        COLOR_WHITE,
                    )
                )
            if top_try > 1 or bool(reset_stack_gate_before_top_try):
                # Restart stack-visibility confirmation from a clean state for each full attempt.
                try:
                    world._stack_visibility_gate = None
                    world._stack_gatecheck_log_sig = None
                    world.brick["brickAbove"] = None
                    world.brick["brickBelow"] = None
                except Exception:
                    pass
                reset_stack_gate_before_top_try = False
            prev_stack_gatecheck_log_enabled = bool(getattr(world, "_stack_gatecheck_log_enabled", False))
            world._stack_gatecheck_log_enabled = True
            try:
                topmost_seek_status = _seek_align_brick_topmost_target()
            finally:
                world._stack_gatecheck_log_enabled = prev_stack_gatecheck_log_enabled
            if topmost_seek_status in {"success", "ready", "confirm cancelled"}:
                break
            if bool(topmost_seek_moved_last):
                if not align_silent:
                    print(
                        format_headline(
                            f"[ALIGN_TOP] movement detected during attempt; resetting attempt limit (next attempt 1/{int(topmost_try_limit)}).",
                            COLOR_WHITE,
                        )
                    )
                reset_stack_gate_before_top_try = True
                top_try = 0
                continue
            if not align_silent and top_try < topmost_try_limit:
                fail_detail = str(topmost_seek_last_fail_reason or "").strip()
                detail_tail = f" ({fail_detail})" if fail_detail else ""
                print(
                    format_headline(
                        f"[ALIGN_TOP] attempt {int(top_try)}/{int(topmost_try_limit)} failed: {str(topmost_seek_status)}{detail_tail}",
                        COLOR_ORANGE_BRIGHT,
                    )
                )
    if is_find_topmost_step:
        if topmost_seek_status in {"success", "ready"}:
            return _complete_alignment_success("topmost target ready", tracker=success_tracker)
        if topmost_seek_status != "confirm cancelled":
            pause_after_fail(robot)
        return False, str(topmost_seek_status or "topmost target not confirmed")

    if bool(progress_mast_active) and not align_silent:
        reminder_msg = progress_mast_reminder or (
            f"Exception active: after {float(progress_mast_trigger_fraction) * 100.0:.1f}% dist progress, "
            f"run {str(progress_mast_cmd).upper()} {int(progress_mast_score)}% x{int(progress_mast_acts)}."
        )
        print(format_headline(f"[{step_key} EXCEPTION] {reminder_msg}", COLOR_MAGENTA_BRIGHT))
    if operator_exception_reminder and not align_silent:
        print(format_headline(f"[{step_key} EXCEPTION] {operator_exception_reminder}", COLOR_MAGENTA_BRIGHT))
    if startup_action_active:
        startup_action_speed = telemetry_robot_module.manual_speed_for_cmd(
            startup_action_cmd,
            startup_action_score,
        )
        if not align_silent:
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] Executing startup action {str(startup_action_cmd).upper()} {int(startup_action_score)}% x{int(startup_action_acts)}.",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        for idx in range(1, int(startup_action_acts) + 1):
            if confirm_callback and not confirm_callback(world, vision):
                return False, "confirm cancelled"
            action_tag = "startup_action_exception"
            if observer:
                observer("analysis", world, vision, startup_action_cmd, startup_action_speed, action_tag)
            action_meta_local = send_robot_command(
                robot,
                world,
                step,
                startup_action_cmd,
                startup_action_speed,
                speed_score=startup_action_score,
                auto_mode=True,
            )
            recent_align_acts.append({"cmd": str(startup_action_cmd), "score": int(startup_action_score)})
            if isinstance(action_meta_local, dict):
                last_align_action = {
                    "cmd": str(startup_action_cmd),
                    "cmd_sent": action_meta_local.get("cmd_sent"),
                    "score": int(startup_action_score),
                    "pwm": action_meta_local.get("pwm"),
                    "power": action_meta_local.get("power"),
                    "duration_ms": action_meta_local.get("duration_ms"),
                }
            else:
                last_align_action = {"cmd": str(startup_action_cmd), "score": int(startup_action_score)}
            if observer:
                observer("action", world, vision, startup_action_cmd, startup_action_speed, action_tag)
            if not align_silent:
                print(
                    format_headline(
                        f"[{step_key} EXCEPTION] Pulse {int(idx)}/{int(startup_action_acts)} complete.",
                        COLOR_MAGENTA_BRIGHT,
                    )
                )
            if int(idx) < int(startup_action_acts) and float(startup_action_pause_s) > 0.0:
                time.sleep(float(startup_action_pause_s))
    if bool(start_ground_reset_run_after_startup) and not bool(start_ground_reset_ran):
        start_ground_reset_result = _run_start_ground_reset_exception(
            world,
            vision,
            step,
            robot=robot,
            observer=observer,
            confirm_callback=confirm_callback,
            align_silent=align_silent,
        )
        start_ground_reset_ran = bool((start_ground_reset_result or {}).get("enabled"))
        if bool((start_ground_reset_result or {}).get("enabled")) and not bool((start_ground_reset_result or {}).get("success")):
            pause_after_fail(robot)
            return False, str((start_ground_reset_result or {}).get("reason") or "start ground reset failed")

    while True:
        loop_id += 1
        world.loop_id = loop_id
        step_norm = normalize_step_label(step)
        update_world_from_vision(world, vision, log=not align_silent)
        visible_now = _observed_bool((getattr(world, "brick", None) or {}).get("visible"))
        if bool(phase1_requires_visible_then_align) and not bool(phase1_visible_passed):
            if bool(visible_now):
                phase1_visible_streak = int(phase1_visible_streak) + 1
            else:
                phase1_visible_streak = 0
            if int(phase1_visible_streak) >= int(visible_search_phase1_confirm_frames):
                phase1_visible_passed = True
                if not align_silent:
                    print(
                        format_headline(
                            (
                                f"[PHASE] {step_key}: phase1 visible gate PASS "
                                f"({int(phase1_visible_streak)}/{int(visible_search_phase1_confirm_frames)}); "
                                "entering x/y micro-align."
                            ),
                            COLOR_GREEN,
                        )
                    )
        if visible_now:
            not_visible_streak = 0
            last_visible_ts = time.time()
            near_dist_vis_attempts = 0
            near_dist_vis_reminder_logged = False
            dist_visible_now = _as_float((getattr(world, "brick", {}) or {}).get("dist"), None)
            if dist_visible_now is not None:
                last_visible_dist_mm = float(dist_visible_now)
            if step_supports_gap_micro_planner:
                gap_micro_armed = True
        else:
            not_visible_streak = int(not_visible_streak) + 1
            align_target_lock_soft_bypass_active = False
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            world._last_obs_note = None
        # pause1 removed per request
        if observer:
            observer("frame", world, vision, None, None, None)
        gate_obs = None
        gate_obs = observe_success_gatecheck(
            world,
            step,
            success_tracker,
            phase="align",
            log=not align_silent,
        )
        success_met = bool(gate_obs.get("success_met"))
        if success_met:
            return _complete_alignment_success("success gate", tracker=success_tracker)
        if bool(gate_obs.get("hold_for_confirm")):
            if robot:
                robot.stop()
            if observer:
                observer("action", world, vision, None, 0.0, "gate confirm hold")
            time.sleep(CONTROL_DT)
            continue
        if bool(align_target_lock_enabled) and not bool(align_target_lock_soft_bypass_active) and bool(gap_micro_armed):
            if visible_now and not isinstance(align_target_lock, dict):
                lock_ok, lock_status = _acquire_second_uppermost_target_lock(
                    source="align_loop_bootstrap",
                    timeout_s=float(align_target_lock_acquire_timeout_s),
                    confirm_frames=int(align_target_lock_confirm_frames),
                )
                if not lock_ok:
                    # Non-blocking fallback: never stall micro-adjust flow when
                    # a center-focus lock cannot be confirmed.
                    _ = lock_status
                    _clear_align_brick_target_lock(clear_required=False)
                    align_target_lock_soft_bypass_active = True
                align_target_lock_bootstrap_done = True
            if isinstance(align_target_lock, dict):
                guard_status = _guard_second_uppermost_target_lock()
                if guard_status in {"hold", "lost"}:
                    # Never block active micro adjustments on lock drift/loss.
                    _clear_align_brick_target_lock(clear_required=False)
                    align_target_lock_soft_bypass_active = True

        exception_ran, exception_status = _maybe_run_progress_mast_exception()
        if exception_status == "confirm cancelled":
            return False, "confirm cancelled"
        if exception_ran:
            continue

        if step_supports_gap_micro_planner and gap_micro_armed and not visible_now:
            lost_for_s = max(0.0, float(time.time()) - float(last_visible_ts))
            confident_loss = bool(
                int(not_visible_streak) >= int(ALIGN_RECOVERY_LOST_VISIBLE_FRAMES)
                and float(lost_for_s) >= float(ALIGN_RECOVERY_LOST_VISIBLE_MIN_S)
            )
            if not confident_loss:
                time.sleep(CONTROL_DT)
                continue
            near_dist_vis_status = _maybe_run_near_dist_visibility_mast_exception()
            if near_dist_vis_status == "confirm cancelled":
                return False, "confirm cancelled"
            if near_dist_vis_status == "recovered":
                stale_pre_obs_streak = 0
                stale_frame_streak = 0
                not_visible_streak = 0
                last_visible_ts = time.time()
                _set_post_recovery_disqualify_types(source="near_dist_visibility_mast")
                prev_log_correction_type = None
                prev_log_x_err = None
                prev_log_y_err = None
                prev_log_dist = None
                force_gap_switch_after_recovery = True
                gap_planner_state.pop("gap_rotation", None)
                last_align_signature = None
                continue
            if (
                near_dist_vis_status == "attempted"
                and int(near_dist_vis_attempts) < int(near_dist_vis_max_attempts)
            ):
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
                _set_post_recovery_disqualify_types(source="pre_move_recheck")
                prev_log_correction_type = None
                prev_log_x_err = None
                prev_log_y_err = None
                prev_log_dist = None
                force_gap_switch_after_recovery = True
                gap_planner_state.pop("gap_rotation", None)
                last_align_signature = None
                continue
            if _recover_visibility(source="align_observe", reason=reason):
                stale_pre_obs_streak = 0
                stale_frame_streak = 0
                not_visible_streak = 0
                last_visible_ts = time.time()
                _set_post_recovery_disqualify_types(source="reverse_recovery")
                prev_log_correction_type = None
                prev_log_x_err = None
                prev_log_y_err = None
                prev_log_dist = None
                force_gap_switch_after_recovery = True
                gap_planner_state.pop("gap_rotation", None)
                last_align_signature = None
                continue
            return False, "lost_vision"

        duration_override_ms = None
        brick = getattr(world, "brick", {}) or {}
        x_gate_required = bool(
            isinstance(step_success_gates_cfg, dict)
            and any(
                key in step_success_gates_cfg
                for key in ("xAxis_offset_abs", "xAxis_offset", "x_axis")
            )
        )
        y_gate_required = bool(
            isinstance(step_success_gates_cfg, dict)
            and any(
                key in step_success_gates_cfg
                for key in ("yAxis_offset_abs", "yAxis_offset", "y_axis")
            )
        )
        x_now = brick.get("x_axis", brick.get("offset_x"))
        y_now = brick.get("y_axis", brick.get("offset_y"))
        x_missing = bool(x_gate_required and x_now is None)
        y_missing = bool(y_gate_required and y_now is None)
        phase1_pending = bool(phase1_requires_visible_then_align and not phase1_visible_passed)
        visible_search_mode = bool(
            visible_search_cycle_enabled
            and bool(visible_search_cmd_cycle)
            and (
                bool(phase1_pending)
                or (not bool(visible_now))
                or bool(x_missing or y_missing)
            )
        )
        if visible_search_mode:
            search_cmd = _next_visible_search_cmd()
            if search_cmd is None:
                visible_search_mode = False
        if visible_search_mode:
            if bool(phase1_pending):
                search_reason = f"{str(step_key or 'STEP').lower()}_search_phase1_visible"
            elif bool(x_missing or y_missing):
                search_reason = f"{str(step_key or 'STEP').lower()}_search_xy_pending"
            else:
                search_reason = f"{str(step_key or 'STEP').lower()}_search_visible_false"
            act_plan = {
                "planner": "search",
                "cmd": str(search_cmd),
                "speed": 0.0,
                "score": None,
                "reason": str(search_reason),
            }
        else:
            act_plan = next_module.select_alignment_next_act(
                process_rules=world.process_rules or {},
                learned_rules=world.learned_rules or {},
                step=step,
                x_axis_mm=brick.get("x_axis", brick.get("offset_x")),
                y_axis_mm=brick.get("y_axis", brick.get("offset_y")),
                dist_mm=brick.get("dist"),
                visible=bool(brick.get("visible")),
                angle_deg=brick.get("angle", 0.0),
                duration_s=CONTROL_DT,
                previous_correction_type=last_align_gap_correction_type,
                avoid_correction_type=skip_align_gap_correction_type_once,
                planner_state=gap_planner_state,
            )
        _log_dist_priority_cheat_act(step, act_plan, align_silent=align_silent, settle=False)
        _log_gap_rotation_tech_debt_act(world, step, act_plan, align_silent=align_silent, settle=False)
        planner_name = str(act_plan.get("planner") or "").strip().lower()
        use_micro_align_gap_planner = planner_name == "gap"
        cmd = act_plan.get("cmd")
        speed = act_plan.get("speed") or 0.0
        cmd_reason = act_plan.get("reason") or "align"
        speed_score = act_plan.get("score")
        if visible_search_mode and cmd in ("f", "b", "l", "r", "u", "d"):
            speed_score = _next_visible_search_score(cmd)
            speed_score = _apply_find_brick_turn_speed_policy(
                step_key,
                cmd,
                speed_score,
                phase="search",
            )
            if speed_score is None:
                speed_score = int(getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1))
            speed = float(telemetry_robot_module.manual_speed_for_cmd(cmd, speed_score))
        planned_corr_type = str(act_plan.get("correction_type") or "").strip().lower() or None
        active_disqualify_types = _active_disqualified_gap_types()
        if (
            use_micro_align_gap_planner
            and active_disqualify_types
            and planned_corr_type
            and planned_corr_type in active_disqualify_types
        ):
            if not align_silent:
                disq_text = ", ".join(sorted(active_disqualify_types))
                print(
                    format_headline(
                        (
                            f"[RECOVERY] planner picked disqualified gap type '{planned_corr_type}' "
                            f"(disqualified: {disq_text}); holding one loop before re-plan."
                        ),
                        COLOR_ORANGE_BRIGHT,
                    )
                )
            cmd = None
            speed = 0.0
            speed_score = None
            cmd_reason = f"recovery_disqualify_hold_{planned_corr_type}"
            # Consume one-shot recovery disqualify immediately after enforcing
            # the hold so the next loop can re-plan from fresh observations.
            skip_align_gap_correction_type_once = None
            planned_corr_type = None
            active_disqualify_types = set()
        if (
            use_micro_align_gap_planner
            and active_disqualify_types
            and planned_corr_type
            and planned_corr_type not in active_disqualify_types
        ):
            skip_align_gap_correction_type_once = None
        if cmd is None:
            demo_fallback = _next_demo_stack_lift_action()
            if isinstance(demo_fallback, dict):
                cmd = demo_fallback.get("cmd")
                speed = demo_fallback.get("speed") or 0.0
                cmd_reason = demo_fallback.get("reason") or "demo stack lift"
                speed_score = demo_fallback.get("speed_score")
                duration_override_ms = demo_fallback.get("duration_override_ms")
        if cmd is None:
            lite_fallback = _align_brick_lite_fail_fallback_action(world, step)
            if isinstance(lite_fallback, dict):
                cmd = lite_fallback.get("cmd")
                speed = float(lite_fallback.get("speed") or 0.0)
                cmd_reason = str(lite_fallback.get("reason") or "align")
                speed_score = lite_fallback.get("score")
        # Visibility-search fallback: if planner has no actionable act while
        # target is not visible, keep scanning in configured direction.
        if (
            cmd is None
            and visible_search_cycle_enabled
            and not bool((getattr(world, "brick", None) or {}).get("visible"))
            and bool(visible_search_cmd_cycle)
        ):
            search_cmd = _next_visible_search_cmd()
            if search_cmd is None:
                search_cmd = scan_cmd
            if search_cmd not in ("f", "b", "l", "r"):
                search_cmd = None
            if search_cmd is None:
                cmd_reason = f"{str(step_key or 'STEP').lower()}_scan_visible_false_no_cmd"
                speed_score = None
                speed = 0.0
            else:
                speed_for_quant = None
                if search_cmd in ("l", "r"):
                    speed_for_quant = action_speeds.get("scan")
                elif search_cmd == "f":
                    speed_for_quant = action_speeds.get("forward")
                elif search_cmd == "b":
                    speed_for_quant = action_speeds.get("backward")
                scan_score = int(getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1))
                try:
                    _q_speed, q_score = telemetry_robot_module.quantize_speed(
                        search_cmd,
                        speed=speed_for_quant,
                    )
                except Exception:
                    q_score = None
                if q_score is not None:
                    scan_score = int(q_score)
                scan_score = _apply_find_brick_turn_speed_policy(
                    step_key,
                    search_cmd,
                    scan_score,
                    phase="search",
                )
                cmd = str(search_cmd)
                speed_score = int(scan_score)
                speed = float(telemetry_robot_module.manual_speed_for_cmd(cmd, speed_score))
                cmd_reason = f"{str(step_key or 'STEP').lower()}_scan_visible_false"
        cmd, speed, speed_score, cmd_reason = _apply_micro_adjust_crosshair_guard(
            cmd,
            speed,
            speed_score,
            cmd_reason,
            planner_name=planner_name,
            phase_label="align",
        )
        # If planner reports no actionable command, immediately run gatecheck
        # confirmation (lite precheck + full tracker) instead of idling.
        if cmd is None:
            no_action_reason = str(cmd_reason or "").strip().lower() or "none"
            if str(no_action_lite_failed_reason or "") != str(no_action_reason):
                no_action_lite_failed_reason = None
            force_gatecheck_hold = True
            if (
                bool(progress_mast_active)
                and not bool(progress_mast_triggered)
                and no_action_reason != "all_gaps_within_gate"
            ):
                force_gatecheck_hold = False
            # Single-lite rule for no-action holds: after one lite precheck fail
            # for the current no-action reason, stop gatecheck-holding and let
            # the loop proceed to re-plan/adjust.
            if force_gatecheck_hold and str(no_action_lite_failed_reason or "") == str(no_action_reason):
                force_gatecheck_hold = False
            if force_gatecheck_hold:
                if not align_silent:
                    if (
                        no_action_reason != "all_gaps_within_gate"
                        and (
                            no_action_reason != last_no_action_gatecheck_reason
                            or (int(loop_id) - int(last_no_action_gatecheck_log_loop)) >= 20
                        )
                    ):
                        print(
                            format_headline(
                                (
                                    f"[WAIT] {step_key} no actionable command "
                                    f"(reason={no_action_reason}); running gatecheck hold."
                                ),
                                COLOR_WHITE,
                            )
                        )
                        last_no_action_gatecheck_reason = str(no_action_reason)
                        last_no_action_gatecheck_log_loop = int(loop_id)
                if robot:
                    robot.stop()
                if observer:
                    observer("action", world, vision, None, 0.0, "gatecheck hold")
                success_hit = run_full_gatecheck_after_act(
                    world,
                    vision,
                    step,
                    success_tracker,
                    phase="align",
                    log=not align_silent,
                    observer=observer,
                )
                if success_hit:
                    return _complete_alignment_success("success gate", tracker=success_tracker)
                gate_status = getattr(world, "_gatecheck_status", None)
                gate_mode = str((gate_status or {}).get("mode") or "").strip().lower()
                gate_truth_ok = bool((gate_status or {}).get("truth_ok", False)) if isinstance(gate_status, dict) else False
                if gate_mode == "lite" and not gate_truth_ok:
                    no_action_lite_failed_reason = str(no_action_reason)
                    if not align_silent and no_action_reason != "all_gaps_within_gate":
                        _log_gatecheck_next_action(
                            step,
                            "align",
                            cmd,
                            speed_score=speed_score,
                            reason="single lite gatecheck failed; resuming adjustment planning",
                            log=True,
                        )
                time.sleep(CONTROL_DT)
                continue
            lite_fallback = _align_brick_lite_fail_fallback_action(world, step)
            if isinstance(lite_fallback, dict):
                cmd = lite_fallback.get("cmd")
                speed = float(lite_fallback.get("speed") or 0.0)
                speed_score = lite_fallback.get("score")
                cmd_reason = str(lite_fallback.get("reason") or "align")
                no_action_lite_failed_reason = None
                if not align_silent:
                    _log_gatecheck_next_action(
                        step,
                        "align",
                        cmd,
                        speed_score=speed_score,
                        reason="single lite gatecheck failed; applying fallback adjustment act",
                        log=True,
                    )
        else:
            no_action_lite_failed_reason = None
        cmd, speed, speed_score, cmd_reason = _apply_turn_only_demo_policy(
            cmd,
            speed,
            speed_score,
            cmd_reason,
            planner_name=planner_name,
        )
        if not use_micro_align_gap_planner:
            success_ok, confidence = evaluate_gate_status(world, step)
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

        is_search_action = False
        if cmd:
            cmd_reason_key = str(cmd_reason or "").strip().lower()
            is_search_action = bool(
                "_search_" in cmd_reason_key
                or cmd_reason_key.endswith("_scan_visible_false")
            )
            local_gate_before_action = _align_local_gate_status_single_source(
                world,
                step,
            )
            dist_err_before_action = _forward_stall_dist_err_from_local_gate(local_gate_before_action)
            forward_stall_triggered, forward_stall_improvement_mm = _forward_stall_should_trigger(
                cmd_value=cmd,
                is_search_action=is_search_action,
            )
            if bool(forward_stall_triggered):
                forward_stall_status = _run_forward_stall_mast_exception(
                    improvement_mm=forward_stall_improvement_mm,
                )
                if str(forward_stall_status) == "confirm cancelled":
                    return False, "confirm cancelled"
                time.sleep(CONTROL_DT)
                continue
            if not bool(is_search_action):
                unknown_metric = _unknown_metric_for_planned_action(
                    cmd=cmd,
                    cmd_reason=cmd_reason,
                    local_gate_before_action=local_gate_before_action,
                )
                if unknown_metric:
                    if robot:
                        robot.stop()
                    if not align_silent:
                        step_label = normalize_step_label(step) or str(step or "ALIGN")
                        try:
                            score_txt = int(round(float(speed_score))) if speed_score is not None else 0
                        except (TypeError, ValueError):
                            score_txt = 0
                        print(
                            format_headline(
                                (
                                    f"[ALIGN] {step_label}: skipping {str(cmd).upper()} {int(score_txt)}% "
                                    f"(missing {str(unknown_metric)}); observing next frame."
                                ),
                                COLOR_ORANGE_BRIGHT,
                            )
                        )
                    if observer:
                        observer("action", world, vision, None, 0.0, f"missing_{str(unknown_metric)}")
                    _record_forward_stall_progress(
                        cmd_value=None,
                        dist_err_before_action=None,
                        is_search_action=True,
                    )
                    time.sleep(CONTROL_DT)
                    continue
            correction_type_for_stats = _align_correction_type_for_cmd(
                cmd,
                fallback=prev_log_correction_type,
            )
            delta_class_for_stats = _classify_align_delta_class(
                correction_type=correction_type_for_stats,
                local_gate_before_action=local_gate_before_action,
                prev_log_correction_type=prev_log_correction_type,
                prev_log_x_err=prev_log_x_err,
                prev_log_y_err=prev_log_y_err,
                prev_log_dist=prev_log_dist,
            )
            previous_cmd_for_bump = str(
                ((last_align_action or {}).get("cmd") if isinstance(last_align_action, dict) else "") or ""
            ).strip().lower()
            previous_speed_score = speed_score
            speed_score, unchanged_speed_bump_state, bumped_for_unchanged = _maybe_apply_repeated_unchanged_speed_bump(
                cmd=cmd,
                score=speed_score,
                delta_class=delta_class_for_stats,
                correction_type=correction_type_for_stats,
                error_signature=_align_error_signature_for_correction(
                    correction_type_for_stats,
                    local_gate_before_action,
                ),
                previous_cmd=previous_cmd_for_bump,
                state=unchanged_speed_bump_state,
            )
            if bool((unchanged_speed_bump_state or {}).get("exhausted")) and not bool(bumped_for_unchanged):
                if not align_silent:
                    print(
                        format_headline(
                            (
                                f"[ADAPT] repeated unchanged result on {str(cmd).upper()}: "
                                f"bump limit {int(REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT)} reached "
                                f"(cap {int(REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP)}%); "
                                "failing to avoid endless retries"
                            ),
                            COLOR_RED,
                        )
                    )
                pause_after_fail(robot)
                return False, "adaptive unchanged-result bump limit reached"
            if bumped_for_unchanged and not align_silent:
                try:
                    prev_score_txt = int(round(float(previous_speed_score)))
                except (TypeError, ValueError):
                    prev_score_txt = None
                try:
                    new_score_txt = int(round(float(speed_score)))
                except (TypeError, ValueError):
                    new_score_txt = None
                if prev_score_txt is not None and new_score_txt is not None:
                    print(
                        format_headline(
                            (
                                f"[ADAPT] repeated unchanged result on {str(cmd).upper()}: "
                                f"bumping speed {int(prev_score_txt)}% -> {int(new_score_txt)}%"
                            ),
                            COLOR_ORANGE_BRIGHT,
                        )
                    )
            pre_frame_id = getattr(world, "_frame_id", 0)
            pre_gate_text = None
            pre_focus = None
            pre_entries = None
            if not suppress_auto_diag:
                pre_entries = _success_gate_entries(world, step)
                pre_gate_text = _auto_gate_snapshot_text(world, step, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            pre_action_obs = pre_action_success_observation(
                world,
                vision,
                step,
                success_tracker,
                phase="align",
                robot=robot,
                observer=observer,
                log=not align_silent,
                success_log=not align_silent,
            )
            if bool((pre_action_obs or {}).get("success_met")):
                return _complete_alignment_success("success gate", tracker=success_tracker)
            if bool((pre_action_obs or {}).get("hold_for_confirm")):
                time.sleep(CONTROL_DT)
                continue
            # Hard no-wiggle edge guard: re-observe immediately before sending
            # motion and skip the act if gate confirmation is already passing or
            # in hold-for-confirm state.
            edge_status = getattr(world, "_gatecheck_status", None)
            if isinstance(edge_status, dict) and bool(edge_status.get("truth_ok", False)):
                return _complete_alignment_success("success gate", tracker=success_tracker)
            update_world_from_vision(world, vision, log=not align_silent)
            if observer:
                observer("frame", world, vision, None, None, None)
            edge_obs = observe_success_gatecheck(
                world,
                step,
                success_tracker,
                phase="align",
                log=False,
            )
            if bool((edge_obs or {}).get("success_met")):
                return _complete_alignment_success("success gate", tracker=success_tracker)
            if bool((edge_obs or {}).get("hold_for_confirm")):
                if robot:
                    robot.stop()
                time.sleep(CONTROL_DT)
                continue
            action_meta = send_robot_command(
                robot,
                world,
                step,
                cmd,
                speed,
                speed_score=speed_score,
                auto_mode=True,
                duration_override_ms=duration_override_ms,
                ease_in_out_enabled=False,
            )
            cmd_sent = cmd
            score_effective = speed_score
            if isinstance(action_meta, dict):
                cmd_sent = action_meta.get("cmd_sent") or cmd_sent
                score_effective = action_meta.get("score_effective")
                if score_effective is None:
                    score_effective = action_meta.get("score_model", speed_score)
            if use_micro_align_gap_planner:
                try:
                    x_sig = round(float(local_gate_before_action.get("x_err")), 2)
                except (TypeError, ValueError):
                    x_sig = None
                try:
                    y_sig = round(float(local_gate_before_action.get("y_err")), 2)
                except (TypeError, ValueError):
                    y_sig = None
                try:
                    d_sig = round(float(local_gate_before_action.get("dist")), 1)
                except (TypeError, ValueError):
                    d_sig = None
                try:
                    s_sig = int(round(float(score_effective))) if score_effective is not None else None
                except (TypeError, ValueError):
                    s_sig = None
                sig = (x_sig, y_sig, d_sig, str(cmd), s_sig)
                if sig == last_align_signature:
                    stale_pre_obs_streak = int(stale_pre_obs_streak) + 1
                else:
                    stale_pre_obs_streak = 0
                last_align_signature = sig

                if stale_pre_obs_streak >= 8:
                    if robot:
                        robot.stop()
                    warn_step = normalize_step_label(step) or str(step)
                    print(
                        format_headline(
                            f"[WARN] {warn_step} stale observation for 8 consecutive acts; treating as lost visibility.",
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
                        _set_post_recovery_disqualify_types(source="stale_observation")
                        prev_log_correction_type = None
                        prev_log_x_err = None
                        prev_log_y_err = None
                        prev_log_dist = None
                        force_gap_switch_after_recovery = True
                        gap_planner_state.pop("gap_rotation", None)
                        last_align_signature = None
                        continue
                    return False, "lost_vision_stale_observation"
            if not align_silent:
                align_action_idx = int(align_action_idx) + 1
                if (
                    "_search_" in cmd_reason_key
                    or cmd_reason_key.endswith("_scan_visible_false")
                ):
                    visible_word = (
                        f"{COLOR_GREEN}YES{COLOR_RESET}" if bool(visible_now) else f"{COLOR_RED}NO{COLOR_RESET}"
                    )
                    scan_text = action_display_text(cmd, score_effective)
                    motor_bits = []
                    if isinstance(action_meta, dict):
                        pwm_now = action_meta.get("pwm")
                        pwr_now = action_meta.get("power")
                        dur_now = action_meta.get("duration_model_ms")
                        if dur_now is None:
                            dur_now = action_meta.get("duration_ms")
                        try:
                            if pwm_now is not None:
                                motor_bits.append(f"pwm={int(round(float(pwm_now)))}")
                        except (TypeError, ValueError):
                            pass
                        try:
                            if pwr_now is not None:
                                motor_bits.append(f"pwr={float(pwr_now):.3f}")
                        except (TypeError, ValueError):
                            pass
                        try:
                            if dur_now is not None:
                                motor_bits.append(f"t={int(round(float(dur_now)))}ms")
                        except (TypeError, ValueError):
                            pass
                    motor_tail = f" ({', '.join(motor_bits)})" if motor_bits else ""
                    phase_bits = []
                    if bool(phase1_requires_visible_then_align) and not bool(phase1_visible_passed):
                        phase_bits.append(
                            f"phase1 visible gate {int(phase1_visible_streak)}/{int(visible_search_phase1_confirm_frames)}"
                        )
                    if bool(x_missing or y_missing):
                        x_state = "WAIT" if bool(x_missing) else "OK"
                        y_state = "WAIT" if bool(y_missing) else "OK"
                        phase_bits.append(f"x={x_state}, y={y_state}")
                    phase_tail = f" ({'; '.join(phase_bits)})" if phase_bits else ""
                    print(
                        format_headline(
                            f"[SEARCH] {step_key}: visible={visible_word}; scanning {scan_text}{motor_tail}{phase_tail}.",
                            COLOR_WHITE,
                        )
                    )
                else:
                    _emit_align_shorthand_action_line(
                        local_gate_before_action=local_gate_before_action,
                        prev_log_correction_type=prev_log_correction_type,
                        prev_log_x_err=prev_log_x_err,
                        prev_log_y_err=prev_log_y_err,
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
                _record_auto_step_action_stats(world, step, delta_class_for_stats)
            action_detail = None
            if not suppress_auto_diag:
                action_detail = auto_action_detail_text(
                    cmd,
                    score_effective,
                    action_meta=action_meta,
                )
            prev_log_x_err = local_gate_before_action.get("x_err")
            prev_log_y_err = local_gate_before_action.get("y_err")
            prev_log_dist = local_gate_before_action.get("dist")
            if cmd in ("f", "b"):
                prev_log_correction_type = "distance"
            elif cmd in ("l", "r"):
                prev_log_correction_type = "x_axis"
            elif cmd in ("u", "d"):
                prev_log_correction_type = "y_axis"
            else:
                prev_log_correction_type = None
            if use_micro_align_gap_planner:
                last_align_gap_correction_type = str(prev_log_correction_type or "").strip().lower() or None
                if last_align_gap_correction_type in {"x_axis", "y_axis", "distance"}:
                    recent_gap_correction_types.append(last_align_gap_correction_type)
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
            if use_micro_align_gap_planner and cmd in ("f", "b", "l", "r"):
                try:
                    hist_score = int(round(float(score_effective))) if score_effective is not None else None
                except (TypeError, ValueError):
                    hist_score = None
                if hist_score is None:
                    try:
                        hist_score = int(round(float(speed_score))) if speed_score is not None else int(DEFAULT_SPEED_SCORE)
                    except (TypeError, ValueError):
                        hist_score = int(DEFAULT_SPEED_SCORE)
                hist_cmd = str(cmd).strip().lower()
                if hist_cmd not in ("f", "b", "l", "r"):
                    hist_cmd = str(cmd).strip().lower()
                recent_align_acts.append({"cmd": str(hist_cmd), "score": int(hist_score)})
                if isinstance(action_meta, dict):
                    last_align_action = {
                        "cmd": str(cmd),
                        "cmd_sent": action_meta.get("cmd_sent"),
                        "score": hist_score,
                        "pwm": action_meta.get("pwm"),
                        "power": action_meta.get("power"),
                        "duration_ms": action_meta.get("duration_ms"),
                    }
                else:
                    last_align_action = {"cmd": str(cmd), "score": hist_score}
            if confirm_callback and robot:
                robot.stop()
            last_action_frame = None
            if not bool(is_search_action):
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
                    required_new_frames=1,
                )
            else:
                if not skip_settle_pause:
                    post_act_settle_pause(world, vision)
                success_hit = run_full_gatecheck_after_act(
                    world,
                    vision,
                    step,
                    success_tracker,
                    phase="align",
                    log=not align_silent,
                    observer=observer,
                    required_new_frames=0,
                )
            if not suppress_auto_diag:
                emit_auto_step_diagnostic(
                    world,
                    step,
                    action_detail,
                    pre_gate_text,
                    emit=(not align_silent and emit_align_auto_diag_console),
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                    action_index=align_action_idx,
                    pre_entries=pre_entries,
                )
            if success_hit:
                return _complete_alignment_success("success gate", tracker=success_tracker)
            _record_forward_stall_progress(
                cmd_value=cmd,
                dist_err_before_action=dist_err_before_action,
                is_search_action=is_search_action,
            )
            if (not skip_settle_pause) and (not bool(is_search_action)):
                post_act_settle_pause(world, vision)
            post_frame_id = getattr(world, "_frame_id", 0)
            if post_frame_id <= pre_frame_id:
                stale_frame_streak = int(stale_frame_streak) + 1
                print(format_headline("[WARN] No new frame observed after action", COLOR_RED))
                if use_micro_align_gap_planner and stale_frame_streak >= 3:
                    if robot:
                        robot.stop()
                    warn_step = normalize_step_label(step) or str(step)
                    print(
                        format_headline(
                            f"[WARN] {warn_step} frame stream appears stale; treating as lost visibility.",
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
                        _set_post_recovery_disqualify_types(source="stale_frames")
                        prev_log_correction_type = None
                        prev_log_x_err = None
                        prev_log_y_err = None
                        prev_log_dist = None
                        force_gap_switch_after_recovery = True
                        gap_planner_state.pop("gap_rotation", None)
                        last_align_signature = None
                        continue
                    return False, "lost_vision_stale_frames"
            else:
                stale_frame_streak = 0
        else:
            _record_forward_stall_progress(
                cmd_value=None,
                dist_err_before_action=None,
                is_search_action=True,
            )
            if robot:
                robot.stop()
        if observer:
            observer("action", world, vision, cmd, speed, cmd_reason)
        if not bool(is_search_action):
            time.sleep(CONTROL_DT)
        if last_action_frame:
            current_frame = {
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
                "y_axis": world.brick.get("y_axis", world.brick.get("offset_y")),
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
        gate_obs = None
        gate_obs = observe_success_gatecheck(
            world,
            step,
            settle_tracker,
            phase="settle",
            log=not align_silent,
        )
        if bool(gate_obs.get("success_met")):
            return _complete_alignment_success("success gate", tracker=settle_tracker)
        if bool(gate_obs.get("hold_for_confirm")):
            if robot:
                robot.stop()
            if observer:
                observer("action", world, vision, None, 0.0, "gate confirm hold")
            time.sleep(CONTROL_DT)
            continue

        step_norm = normalize_step_label(step)
        duration_override_ms = None
        brick = getattr(world, "brick", {}) or {}
        act_plan = next_module.select_alignment_next_act(
            process_rules=world.process_rules or {},
            learned_rules=world.learned_rules or {},
            step=step,
            x_axis_mm=brick.get("x_axis", brick.get("offset_x")),
            y_axis_mm=brick.get("y_axis", brick.get("offset_y")),
            dist_mm=brick.get("dist"),
            visible=bool(brick.get("visible")),
            angle_deg=brick.get("angle", 0.0),
            duration_s=CONTROL_DT,
            previous_correction_type=last_align_gap_correction_type,
            avoid_correction_type=skip_align_gap_correction_type_once,
            planner_state=gap_planner_state,
        )
        _log_dist_priority_cheat_act(step, act_plan, align_silent=align_silent, settle=True)
        _log_gap_rotation_tech_debt_act(world, step, act_plan, align_silent=align_silent, settle=True)
        planner_name = str(act_plan.get("planner") or "").strip().lower()
        use_micro_align_gap_planner = planner_name == "gap"
        cmd = act_plan.get("cmd")
        speed = act_plan.get("speed") or 0.0
        cmd_reason = act_plan.get("reason") or "align"
        speed_score = act_plan.get("score")
        planned_corr_type = str(act_plan.get("correction_type") or "").strip().lower() or None
        active_disqualify_types = _active_disqualified_gap_types()
        if (
            use_micro_align_gap_planner
            and active_disqualify_types
            and planned_corr_type
            and planned_corr_type in active_disqualify_types
        ):
            if not align_silent:
                disq_text = ", ".join(sorted(active_disqualify_types))
                print(
                    format_headline(
                        (
                            f"[RECOVERY] planner picked disqualified gap type '{planned_corr_type}' "
                            f"(disqualified: {disq_text}); holding one settle loop before re-plan."
                        ),
                        COLOR_ORANGE_BRIGHT,
                    )
                )
            cmd = None
            speed = 0.0
            speed_score = None
            cmd_reason = f"recovery_disqualify_hold_{planned_corr_type}"
            skip_align_gap_correction_type_once = None
            planned_corr_type = None
            active_disqualify_types = set()
        if (
            use_micro_align_gap_planner
            and active_disqualify_types
            and planned_corr_type
            and planned_corr_type not in active_disqualify_types
        ):
            skip_align_gap_correction_type_once = None
        if cmd is None:
            demo_fallback = _next_demo_stack_lift_action()
            if isinstance(demo_fallback, dict):
                cmd = demo_fallback.get("cmd")
                speed = demo_fallback.get("speed") or 0.0
                cmd_reason = demo_fallback.get("reason") or "demo stack lift"
                speed_score = demo_fallback.get("speed_score")
                duration_override_ms = demo_fallback.get("duration_override_ms")
        cmd, speed, speed_score, cmd_reason = _apply_micro_adjust_crosshair_guard(
            cmd,
            speed,
            speed_score,
            cmd_reason,
            planner_name=planner_name,
            phase_label="settle",
        )
        cmd, speed, speed_score, cmd_reason = _apply_turn_only_demo_policy(
            cmd,
            speed,
            speed_score,
            cmd_reason,
            planner_name=planner_name,
        )
        if not use_micro_align_gap_planner:
            success_ok, confidence = evaluate_gate_status(world, step)
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
            local_gate_before_action = _align_local_gate_status_single_source(
                world,
                step,
            )
            unknown_metric = _unknown_metric_for_planned_action(
                cmd=cmd,
                cmd_reason=cmd_reason,
                local_gate_before_action=local_gate_before_action,
            )
            if unknown_metric:
                if robot:
                    robot.stop()
                if not align_silent:
                    step_label = normalize_step_label(step) or str(step or "ALIGN")
                    try:
                        score_txt = int(round(float(speed_score))) if speed_score is not None else 0
                    except (TypeError, ValueError):
                        score_txt = 0
                    print(
                        format_headline(
                            (
                                f"[ALIGN] {step_label}: skipping {str(cmd).upper()} {int(score_txt)}% "
                                f"(missing {str(unknown_metric)}); observing next frame."
                            ),
                            COLOR_ORANGE_BRIGHT,
                        )
                    )
                if observer:
                    observer("action", world, vision, None, 0.0, f"missing_{str(unknown_metric)}")
                time.sleep(CONTROL_DT)
                continue
            correction_type_for_stats = _align_correction_type_for_cmd(
                cmd,
                fallback=prev_log_correction_type,
            )
            delta_class_for_stats = _classify_align_delta_class(
                correction_type=correction_type_for_stats,
                local_gate_before_action=local_gate_before_action,
                prev_log_correction_type=prev_log_correction_type,
                prev_log_x_err=prev_log_x_err,
                prev_log_y_err=prev_log_y_err,
                prev_log_dist=prev_log_dist,
            )
            previous_cmd_for_bump = str(
                ((last_align_action or {}).get("cmd") if isinstance(last_align_action, dict) else "") or ""
            ).strip().lower()
            previous_speed_score = speed_score
            speed_score, unchanged_speed_bump_state, bumped_for_unchanged = _maybe_apply_repeated_unchanged_speed_bump(
                cmd=cmd,
                score=speed_score,
                delta_class=delta_class_for_stats,
                correction_type=correction_type_for_stats,
                error_signature=_align_error_signature_for_correction(
                    correction_type_for_stats,
                    local_gate_before_action,
                ),
                previous_cmd=previous_cmd_for_bump,
                state=unchanged_speed_bump_state,
            )
            if bool((unchanged_speed_bump_state or {}).get("exhausted")) and not bool(bumped_for_unchanged):
                if not align_silent:
                    print(
                        format_headline(
                            (
                                f"[ADAPT] repeated unchanged result on {str(cmd).upper()}: "
                                f"bump limit {int(REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT)} reached "
                                f"(cap {int(REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP)}%); "
                                "failing to avoid endless retries"
                            ),
                            COLOR_RED,
                        )
                    )
                pause_after_fail(robot)
                return False, "adaptive unchanged-result bump limit reached"
            if bumped_for_unchanged and not align_silent:
                try:
                    prev_score_txt = int(round(float(previous_speed_score)))
                except (TypeError, ValueError):
                    prev_score_txt = None
                try:
                    new_score_txt = int(round(float(speed_score)))
                except (TypeError, ValueError):
                    new_score_txt = None
                if prev_score_txt is not None and new_score_txt is not None:
                    print(
                        format_headline(
                            (
                                f"[ADAPT] repeated unchanged result on {str(cmd).upper()}: "
                                f"bumping speed {int(prev_score_txt)}% -> {int(new_score_txt)}%"
                            ),
                            COLOR_ORANGE_BRIGHT,
                        )
                    )
            pre_gate_text = None
            pre_focus = None
            pre_entries = None
            if not suppress_auto_diag:
                pre_entries = _success_gate_entries(world, step)
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
                duration_override_ms=duration_override_ms,
                ease_in_out_enabled=False,
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
                    prev_log_y_err=prev_log_y_err,
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
            prev_log_y_err = local_gate_before_action.get("y_err")
            prev_log_dist = local_gate_before_action.get("dist")
            if cmd in ("f", "b"):
                prev_log_correction_type = "distance"
            elif cmd in ("l", "r"):
                prev_log_correction_type = "x_axis"
            elif cmd in ("u", "d"):
                prev_log_correction_type = "y_axis"
            else:
                prev_log_correction_type = None
            if use_micro_align_gap_planner:
                last_align_gap_correction_type = str(prev_log_correction_type or "").strip().lower() or None
                if last_align_gap_correction_type in {"x_axis", "y_axis", "distance"}:
                    recent_gap_correction_types.append(last_align_gap_correction_type)
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
                    emit=(not align_silent and emit_align_auto_diag_console),
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                    action_index=settle_action_idx,
                    pre_entries=pre_entries,
                )
            if success_hit:
                return _complete_alignment_success("success gate", tracker=settle_tracker)
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
    visible_only_tiering_enabled = bool(
        _visible_only_replay_speed_tier(world, step_key, step_cfg=cfg, cmd="f") is not None
    )
    if visible_only_tiering_enabled:
        # Tiered visible-only replay owns speed selection (normal/standard/fast).
        # Ignore demo speed scores for this step so runtime follows the tier policy.
        prefer_demo_speed = False
        loop_demo_acts = False
    demo_actions = nominal_actions_from_events(events)
    demo_actions_replay = list(demo_actions)
    if prefer_demo_speed and demo_actions:
        loop_demo_acts = True
    steps = raw_steps if loop_demo_acts else smooth_motion_steps(raw_steps)
    topmost_cfg = cfg.get("topmost_crosshair_exception") if isinstance(cfg, dict) else {}
    if not isinstance(topmost_cfg, dict):
        topmost_cfg = {}
    force_replay_for_ground_up_level2 = bool(_step_allows_topmost_ground_up_level2(step_key)) and bool(
        topmost_cfg.get("enabled")
    ) and bool(topmost_cfg.get("ground_up_level2_enabled"))
    if step_uses_alignment_control(step, world.process_rules) and not force_replay_for_ground_up_level2:
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
    start_ground_reset_result = _run_start_ground_reset_exception(
        world,
        vision,
        step_key,
        robot=robot,
        observer=observer,
        confirm_callback=confirm_callback,
        align_silent=align_silent,
    )
    if bool((start_ground_reset_result or {}).get("enabled")) and not bool((start_ground_reset_result or {}).get("success")):
        pause_after_fail(robot)
        return False, str((start_ground_reset_result or {}).get("reason") or "start ground reset failed")
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
                    post_act_analysis(world, vision, step=step, log=not align_silent)
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
    if force_replay_for_ground_up_level2:
        update_world_from_vision(world, vision, log=not align_silent)
        if observer:
            observer("frame", world, vision, None, None, None)
        if not align_silent:
            print(
                format_headline(
                    f"[{step_key} EXCEPTION] bypassing start-gate hold; entering ground-up mast workflow.",
                    COLOR_MAGENTA_BRIGHT,
                )
            )
        start_status = "start"
    else:
        start_status = wait_for_start_gates(
            world,
            vision,
            step_key,
            robot=robot,
            cmd=default_step.cmd,
            speed=start_speed,
            allow_success=success_checks_enabled,
        )
    startup_action_ran = False
    startup_action_cfg = {}
    if isinstance(getattr(world, "process_rules", None), dict):
        step_cfg_local = world.process_rules.get(step_key, {}) or {}
        if isinstance(step_cfg_local, dict):
            startup_action_cfg = step_cfg_local.get("startup_action_exception") or {}
            if not isinstance(startup_action_cfg, dict):
                startup_action_cfg = {}
            operator_exception_reminder = str(step_cfg_local.get("operator_exception_reminder") or "").strip()
            startup_enabled = bool(startup_action_cfg.get("enabled"))
            startup_cmd = str(startup_action_cfg.get("command") or "").strip().lower()
            if startup_cmd not in ("f", "b", "l", "r", "u", "d"):
                startup_cmd = ""
            startup_acts = max(1, int(_as_int(startup_action_cfg.get("acts"), 1)))
            startup_score = None
            score_hotkey = str(startup_action_cfg.get("score_from_hotkey") or "").strip().lower()
            if score_hotkey:
                hotkey_rows = getattr(telemetry_robot_module, "HOTKEY_SPEED_SCORES", {})
                hotkey_row = hotkey_rows.get(score_hotkey) if isinstance(hotkey_rows, dict) else None
                if isinstance(hotkey_row, dict):
                    score_val = _as_int(hotkey_row.get("score"), None)
                    if score_val is not None:
                        startup_score = int(score_val)
            if startup_score is None:
                startup_score = int(
                    _as_int(startup_action_cfg.get("score"), int(telemetry_robot_module.SPEED_SCORE_MIN))
                )
            startup_score = max(
                int(telemetry_robot_module.SPEED_SCORE_MIN),
                min(int(MAX_SPEED_SCORE), int(startup_score)),
            )
            if startup_enabled and startup_cmd and start_status in {"start", "success"}:
                startup_action_ran = True
                startup_speed = telemetry_robot_module.manual_speed_for_cmd(startup_cmd, startup_score)
                if operator_exception_reminder and not align_silent:
                    print(format_headline(f"[{step_key} EXCEPTION] {operator_exception_reminder}", COLOR_MAGENTA_BRIGHT))
                if not align_silent:
                    print(
                        format_headline(
                            f"[{step_key} EXCEPTION] Executing startup action {str(startup_cmd).upper()} {int(startup_score)}% x{int(startup_acts)}.",
                            COLOR_MAGENTA_BRIGHT,
                        )
                    )
                for idx in range(1, int(startup_acts) + 1):
                    if confirm_callback and not confirm_callback(world, vision):
                        if robot:
                            robot.stop()
                        return False, "confirm cancelled"
                    action_tag = "startup_action_exception"
                    if observer:
                        observer("analysis", world, vision, startup_cmd, startup_speed, action_tag)
                    send_robot_command(
                        robot,
                        world,
                        step_key,
                        startup_cmd,
                        startup_speed,
                        speed_score=startup_score,
                        auto_mode=True,
                    )
                    if observer:
                        observer("action", world, vision, startup_cmd, startup_speed, action_tag)
                    post_act_analysis(world, vision, step=step_key, log=not align_silent)
                    if not align_silent:
                        print(
                            format_headline(
                                f"[{step_key} EXCEPTION] Pulse {int(idx)}/{int(startup_acts)} complete.",
                                COLOR_MAGENTA_BRIGHT,
                            )
                        )
                    time.sleep(CONTROL_DT)
    if start_status == "success" and not startup_action_ran:
        if _step_allows_topmost_ground_up_level2(step_key):
            # Topmost steps must run the level2 mast workflow; do not short-circuit
            # on a plain success-gate pass at replay start.
            start_status = "start"
        else:
            if robot:
                robot.stop()
            print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
            print_gate_summary_line(world)
            print_success_events(world, step_key)
            return True, "success gate"
    if start_status == "success" and startup_action_ran:
        start_status = "start"
    if start_status != "start":
        pause_after_fail(robot)
        return False, _start_gate_failure_reason_from_wait_status(start_status)

    ground_up_exc = _run_ground_up_level2_exception(
        world,
        vision,
        step_key,
        robot=robot,
        observer=observer,
        confirm_callback=confirm_callback,
        align_silent=align_silent,
    )
    if bool((ground_up_exc or {}).get("enabled")) and bool((ground_up_exc or {}).get("handled")):
        if bool((ground_up_exc or {}).get("success")):
            return True, str((ground_up_exc or {}).get("reason") or "success gate + level2")
        pause_after_fail(robot)
        return False, str((ground_up_exc or {}).get("reason") or "ground-up level2 failed")

    allow_early_exit = True

    success_tracker = new_success_tracker(step, world.process_rules)
    clear_pending_auto_step_diagnostic(world)
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = float(default_step.speed or 0.0)
    demo_replay_action_idx = 0
    replay_cycle_action_idx = 0

    if loop_demo_acts and demo_actions_replay:
        while True:
            for action in demo_actions_replay:
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
                cmd_exec = str(cmd)
                score_exec = int(score)
                cmd_exec, score_exec, height_context_note, _height_used = _apply_height_intel_replay_override(
                    world,
                    step_key,
                    cmd_exec,
                    score_exec,
                    phase="demo",
                    log=not align_silent,
                )
                action_duration_s = max(0.001, float(action.get("duration_s") or CONTROL_DT))
                pre_entries = _success_gate_entries(world, step_key)
                pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step_key)
                pre_action_obs = pre_action_success_observation(
                    world,
                    vision,
                    step_key,
                    success_tracker,
                    phase="replay",
                    robot=robot,
                    observer=observer,
                    log=not align_silent,
                    success_log=not align_silent,
                )
                if bool((pre_action_obs or {}).get("success_met")):
                    return True, "success gate"
                if bool((pre_action_obs or {}).get("hold_for_confirm")):
                    time.sleep(CONTROL_DT)
                    continue
                action_meta = send_robot_command(
                    robot,
                    world,
                    step_key,
                    cmd_exec,
                    0.0,
                    speed_score=score_exec,
                    auto_mode=True,
                )
                pwm = None
                duration_ms = int(action_duration_s * 1000)
                cmd_sent = None
                if isinstance(action_meta, dict):
                    active_speed = float(action_meta.get("power", 0.0) or 0.0)
                    score_effective = action_meta.get("score_effective")
                    if score_effective is None:
                        score_effective = action_meta.get("score_model", score_exec)
                    pwm = action_meta.get("pwm")
                    cmd_sent = action_meta.get("cmd_sent")
                    duration_ms = int(
                        action_meta.get("duration_model_ms")
                        or action_meta.get("duration_ms")
                        or duration_ms
                    )
                else:
                    active_speed = 0.0
                    score_effective = score_exec
                record_action_display(
                    world,
                    step_key,
                    cmd_exec,
                    active_speed,
                    speed_score=score_effective,
                    cmd_sent=cmd_sent,
                    pwm=pwm,
                    duration_ms=duration_ms,
                    score_model=score_exec,
                )
                sent_snapshot = _capture_sent_action_snapshot(world)
                mode_suffix = f" {height_context_note}" if height_context_note else ""
                world._last_action_line = format_control_action_line(
                    cmd_exec,
                    active_speed,
                    f"demo {int(score_effective)}%{mode_suffix}",
                )
                evt = MotionEvent(
                    cmd_to_motion_type(cmd_exec),
                    int(active_speed * 255),
                    int(action_duration_s * 1000),
                )
                world.update_from_motion(evt)
                if observer:
                    observer("action", world, vision, cmd_exec, active_speed, "demo replay")
                time.sleep(action_duration_s)
                post_act_analysis(world, vision, step=step_key, log=not align_silent, include_pause=False)
                success_hit = run_full_gatecheck_after_act(
                    world,
                    vision,
                    step_key,
                    success_tracker,
                    phase="replay",
                    log=not align_silent,
                    observer=observer,
                )
                detail = auto_action_detail_text(
                    cmd_exec,
                    score_effective,
                    action_meta=action_meta,
                    context_note=height_context_note,
                )
                demo_replay_action_idx = int(demo_replay_action_idx) + 1
                emit_auto_step_diagnostic(
                    world,
                    step_key,
                    detail,
                    pre_gate_text,
                    emit=True,
                    pre_focus=pre_focus,
                    success_override=success_hit,
                    sent_snapshot=sent_snapshot,
                    action_index=demo_replay_action_idx,
                    pre_entries=pre_entries,
                )
                if success_hit:
                    if robot:
                        robot.stop()
                    if not align_silent:
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
                if observer:
                    observer("frame", world, vision, None, None, None)

                replay_action_elapsed_s = now - replay_action_start_time
                can_check_success = replay_action_elapsed_s >= (required_action_prefix_s - 1e-6)

                frame_id = int(getattr(world, "_frame_id", 0) or 0)
                fresh_frame = last_gate_frame_id is None or frame_id != last_gate_frame_id
                if fresh_frame:
                    last_gate_frame_id = frame_id

                forcing_check = now < float(force_check_until or 0.0)
                phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
                desired_score = move_score if phase == "move" else check_score
                speed_tier_label = None
                if motion_step.speed_score is not None and prefer_demo_speed:
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
                if visible_only_tiering_enabled:
                    tier_override = _visible_only_replay_speed_tier(
                        world,
                        step_key,
                        step_cfg=cfg,
                        cmd=motion_step.cmd,
                    )
                    if isinstance(tier_override, dict):
                        desired_score = int(tier_override.get("score") or desired_score)
                        speed_tier_label = str(tier_override.get("label") or "").strip().lower() or None
                if speed_tier_label is None:
                    speed_tier_label = _auto_cycle_speed_label(desired_score)

                # If we're seeing positive evidence, stay in slow/check mode while we confirm/refute.
                if allow_early_exit and can_check_success and fresh_frame and phase == "check":
                    gate_obs = observe_success_gatecheck(
                        world,
                        step_key,
                        success_tracker,
                        phase="replay",
                        log=not align_silent,
                    )
                    success_met = bool(gate_obs.get("success_met"))
                    if success_met:
                        step_success_met = True
                        break
                    if bool(gate_obs.get("hold_for_confirm")):
                        force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                        phase = "check"
                        if not visible_only_tiering_enabled:
                            desired_score = check_score
                            speed_tier_label = _auto_cycle_speed_label(desired_score)

                flush_auto_step_diagnostic(world, step_key, emit=True)
                cmd = motion_step.cmd
                if cmd:
                    cmd_exec = str(cmd)
                    score_exec = int(desired_score)
                    cmd_exec, score_exec, height_context_note, _height_used = _apply_height_intel_replay_override(
                        world,
                        step_key,
                        cmd_exec,
                        score_exec,
                        phase=phase,
                        log=not align_silent,
                    )
                    desired_score = int(score_exec)
                    pre_entries = _success_gate_entries(world, step_key)
                    pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                    pre_focus = _capture_auto_diag_focus(world, step_key)
                    pre_action_obs = pre_action_success_observation(
                        world,
                        vision,
                        step_key,
                        success_tracker,
                        phase="replay",
                        robot=robot,
                        observer=observer,
                        log=not align_silent,
                        success_log=not align_silent,
                    )
                    if bool((pre_action_obs or {}).get("success_met")):
                        flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                        return True, "success gate"
                    if bool((pre_action_obs or {}).get("hold_for_confirm")):
                        time.sleep(CONTROL_DT)
                        continue
                    action_meta = send_robot_command(
                        robot,
                        world,
                        step_key,
                        cmd_exec,
                        0.0,
                        speed_score=score_exec,
                        auto_mode=True,
                    )
                    if isinstance(action_meta, dict):
                        active_speed = float(action_meta.get("power", 0.0) or 0.0)
                        score_effective = action_meta.get("score_effective")
                        if score_effective is None:
                            score_effective = action_meta.get("score_model", score_exec)
                        cmd_sent = action_meta.get("cmd_sent") or cmd_exec
                    else:
                        active_speed = 0.0
                        score_effective = score_exec
                        cmd_sent = cmd_exec
                    action_context_note = speed_tier_label or _auto_cycle_speed_label(score_effective)
                    if height_context_note:
                        if action_context_note:
                            action_context_note = f"{action_context_note}; {height_context_note}"
                        else:
                            action_context_note = str(height_context_note)
                    mode_suffix = f" {action_context_note}" if action_context_note else ""
                    sent_snapshot = _capture_sent_action_snapshot(world)
                    world._last_action_line = format_control_action_line(
                        cmd_exec,
                        active_speed,
                        f"replay {phase} {int(score_effective)}%{mode_suffix}",
                    )
                    evt = MotionEvent(
                        cmd_to_motion_type(cmd_exec),
                        int(active_speed * 255),
                        int(CONTROL_DT * 1000),
                    )
                    world.update_from_motion(evt)
                    if visible_only_tiering_enabled:
                        world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0) or 0) + 1
                    if observer:
                        observer("action", world, vision, cmd_exec, active_speed, "replay")
                    replay_cycle_action_idx = int(replay_cycle_action_idx) + 1
                    queue_auto_step_diagnostic(
                        world,
                        step_key,
                        auto_action_detail_text(
                            cmd_exec,
                            score_effective,
                            action_meta=action_meta,
                            context_note=action_context_note,
                        ),
                        pre_gate_text,
                        pre_focus=pre_focus,
                        sent_snapshot=sent_snapshot,
                        pre_entries=pre_entries,
                        action_index=replay_cycle_action_idx,
                    )
                else:
                    if robot:
                        robot.stop()
                time.sleep(CONTROL_DT)
            if step_success_met:
                flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                if robot:
                    robot.stop()
                if not align_silent:
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

        forcing_check = now < float(force_check_until or 0.0)
        phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
        desired_score = move_score if phase == "move" else check_score
        speed_tier_label = None
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
        if visible_only_tiering_enabled:
            tier_override = _visible_only_replay_speed_tier(
                world,
                step_key,
                step_cfg=cfg,
                cmd=last_cmd,
            )
            if isinstance(tier_override, dict):
                desired_score = int(tier_override.get("score") or desired_score)
                speed_tier_label = str(tier_override.get("label") or "").strip().lower() or None
        if speed_tier_label is None:
            speed_tier_label = _auto_cycle_speed_label(desired_score)

        checked_this_frame = False
        success_met = False
        gate_obs = None
        if success_checks_enabled and fresh_frame and phase == "check":
            checked_this_frame = True
            gate_obs = observe_success_gatecheck(
                world,
                step_key,
                settle_tracker,
                phase="settle",
                log=not align_silent,
            )
            success_met = bool(gate_obs.get("success_met"))
            if success_met:
                flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                if not align_silent:
                    print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                    print_gate_summary_line(world, settle_tracker)
                    print_success_events(world, step_key)
                return True, "success gate"
            if bool(gate_obs.get("hold_for_confirm")):
                force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                phase = "check"
                if not visible_only_tiering_enabled:
                    desired_score = check_score
                    speed_tier_label = _auto_cycle_speed_label(desired_score)

        if checked_this_frame and not success_met:
            if not bool((gate_obs or {}).get("effective_success_ok")):
                _log_gatecheck_failure(world, step_key, "settle", log=not align_silent)
            if last_cmd:
                reason_text = "adaptive replay"
                if speed_tier_label:
                    reason_text = f"{reason_text}; {speed_tier_label} tier"
                _log_gatecheck_next_action(
                    step_key,
                    "settle",
                    last_cmd,
                    speed_score=desired_score,
                    reason=reason_text,
                    log=not align_silent,
                )
            else:
                _log_gatecheck_next_action(
                    step_key,
                    "settle",
                    None,
                    reason="replay has no command to continue",
                    log=not align_silent,
                )

        flush_auto_step_diagnostic(world, step_key, emit=True)
        if last_cmd:
            cmd_exec = str(last_cmd)
            score_exec = int(desired_score)
            cmd_exec, score_exec, height_context_note, _height_used = _apply_height_intel_replay_override(
                world,
                step_key,
                cmd_exec,
                score_exec,
                phase="settle",
                log=not align_silent,
            )
            desired_score = int(score_exec)
            pre_entries = _success_gate_entries(world, step_key)
            pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
            pre_focus = _capture_auto_diag_focus(world, step_key)
            pre_action_obs = pre_action_success_observation(
                world,
                vision,
                step_key,
                settle_tracker,
                phase="settle",
                robot=robot,
                observer=observer,
                log=not align_silent,
                success_log=not align_silent,
            )
            if bool((pre_action_obs or {}).get("success_met")):
                flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                return True, "success gate"
            if bool((pre_action_obs or {}).get("hold_for_confirm")):
                time.sleep(CONTROL_DT)
                continue
            action_meta = send_robot_command(
                robot,
                world,
                step_key,
                cmd_exec,
                0.0,
                speed_score=score_exec,
                auto_mode=True,
            )
            if isinstance(action_meta, dict):
                active_speed = float(action_meta.get("power", 0.0) or 0.0)
                score_effective = action_meta.get("score_effective")
                if score_effective is None:
                    score_effective = action_meta.get("score_model", score_exec)
                cmd_sent = action_meta.get("cmd_sent") or cmd_exec
            else:
                active_speed = 0.0
                score_effective = score_exec
                cmd_sent = cmd_exec
            action_context_note = speed_tier_label or _auto_cycle_speed_label(score_effective)
            if height_context_note:
                if action_context_note:
                    action_context_note = f"{action_context_note}; {height_context_note}"
                else:
                    action_context_note = str(height_context_note)
            mode_suffix = f" {action_context_note}" if action_context_note else ""
            sent_snapshot = _capture_sent_action_snapshot(world)
            world._last_action_line = format_control_action_line(
                cmd_exec,
                active_speed,
                f"settle {phase} {int(score_effective)}%{mode_suffix}",
            )
            evt = MotionEvent(
                cmd_to_motion_type(cmd_exec),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if visible_only_tiering_enabled:
                world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0) or 0) + 1
            if observer:
                observer("action", world, vision, cmd_exec, active_speed, "replay")
            replay_cycle_action_idx = int(replay_cycle_action_idx) + 1
            queue_auto_step_diagnostic(
                world,
                step_key,
                auto_action_detail_text(
                    cmd_exec,
                    score_effective,
                    action_meta=action_meta,
                    context_note=action_context_note,
                ),
                pre_gate_text,
                pre_focus=pre_focus,
                sent_snapshot=sent_snapshot,
                pre_entries=pre_entries,
                action_index=replay_cycle_action_idx,
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
            if observer:
                observer("frame", world, vision, None, None, None)

            replay_action_elapsed_s = now - replay_action_start_time
            frame_id = int(getattr(world, "_frame_id", 0) or 0)
            fresh_frame = last_gate_frame_id is None or frame_id != last_gate_frame_id
            if fresh_frame:
                last_gate_frame_id = frame_id

            forcing_check = now < float(force_check_until or 0.0)
            phase = auto_cycle_phase(replay_action_elapsed_s, force_check=forcing_check)
            desired_score = move_score if phase == "move" else check_score
            speed_tier_label = None
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
            if visible_only_tiering_enabled:
                tier_override = _visible_only_replay_speed_tier(
                    world,
                    step_key,
                    step_cfg=cfg,
                    cmd=last_cmd,
                )
                if isinstance(tier_override, dict):
                    desired_score = int(tier_override.get("score") or desired_score)
                    speed_tier_label = str(tier_override.get("label") or "").strip().lower() or None
            if speed_tier_label is None:
                speed_tier_label = _auto_cycle_speed_label(desired_score)

            checked_this_frame = False
            success_met = False
            gate_obs = None
            if success_checks_enabled and fresh_frame and phase == "check":
                checked_this_frame = True
                gate_obs = observe_success_gatecheck(
                    world,
                    step_key,
                    tail_tracker,
                    phase="tail",
                    log=not align_silent,
                )
                success_met = bool(gate_obs.get("success_met"))
                if success_met:
                    flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
                    if robot:
                        robot.stop()
                    if not align_silent:
                        print(format_headline(f"[SUCCESS] {step_key} criteria met 🎉", COLOR_GREEN))
                        print_gate_summary_line(world, tail_tracker)
                        print_success_events(world, step_key)
                    return True, "success gate"
                if bool(gate_obs.get("hold_for_confirm")):
                    force_check_until = max(float(force_check_until or 0.0), now + float(AUTO_CYCLE_CHECK_S))
                    phase = "check"
                    if not visible_only_tiering_enabled:
                        desired_score = check_score
                        speed_tier_label = _auto_cycle_speed_label(desired_score)

            if checked_this_frame and not success_met:
                if not bool((gate_obs or {}).get("effective_success_ok")):
                    _log_gatecheck_failure(world, step_key, "tail", log=not align_silent)
                if last_cmd:
                    reason_text = f"auto-cycle {phase}"
                    if speed_tier_label:
                        reason_text = f"{reason_text}; {speed_tier_label} tier"
                    _log_gatecheck_next_action(
                        step_key,
                        "tail",
                        last_cmd,
                        speed_score=desired_score,
                        reason=reason_text,
                        log=not align_silent,
                    )
                else:
                    _log_gatecheck_next_action(
                        step_key,
                        "tail",
                        None,
                        reason="replay has no command to continue",
                        log=not align_silent,
                    )

            flush_auto_step_diagnostic(world, step_key, emit=True)
            if last_cmd:
                cmd_exec = str(last_cmd)
                score_exec = int(desired_score)
                cmd_exec, score_exec, height_context_note, _height_used = _apply_height_intel_replay_override(
                    world,
                    step_key,
                    cmd_exec,
                    score_exec,
                    phase="tail",
                    log=not align_silent,
                )
                desired_score = int(score_exec)
                pre_entries = _success_gate_entries(world, step_key)
                pre_gate_text = _auto_gate_snapshot_text(world, step_key, include_requirements=True)
                pre_focus = _capture_auto_diag_focus(world, step_key)
                action_meta = send_robot_command(
                    robot,
                    world,
                    step_key,
                    cmd_exec,
                    0.0,
                    speed_score=score_exec,
                    auto_mode=True,
                )
                if isinstance(action_meta, dict):
                    active_speed = float(action_meta.get("power", 0.0) or 0.0)
                    score_effective = action_meta.get("score_effective")
                    if score_effective is None:
                        score_effective = action_meta.get("score_model", score_exec)
                    cmd_sent = action_meta.get("cmd_sent") or cmd_exec
                else:
                    active_speed = 0.0
                    score_effective = score_exec
                    cmd_sent = cmd_exec
                action_context_note = speed_tier_label or _auto_cycle_speed_label(score_effective)
                if height_context_note:
                    if action_context_note:
                        action_context_note = f"{action_context_note}; {height_context_note}"
                    else:
                        action_context_note = str(height_context_note)
                mode_suffix = f" {action_context_note}" if action_context_note else ""
                sent_snapshot = _capture_sent_action_snapshot(world)
                world._last_action_line = format_control_action_line(
                    cmd_exec,
                    active_speed,
                    f"tail {phase} {int(score_effective)}%{mode_suffix}",
                )
                evt = MotionEvent(
                    cmd_to_motion_type(cmd_exec),
                    int(active_speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
                if visible_only_tiering_enabled:
                    world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0) or 0) + 1
                if observer:
                    observer("action", world, vision, cmd_exec, active_speed, "replay")
                replay_cycle_action_idx = int(replay_cycle_action_idx) + 1
                queue_auto_step_diagnostic(
                    world,
                    step_key,
                    auto_action_detail_text(
                        cmd_exec,
                        score_effective,
                        action_meta=action_meta,
                        context_note=action_context_note,
                    ),
                    pre_gate_text,
                    pre_focus=pre_focus,
                    sent_snapshot=sent_snapshot,
                    pre_entries=pre_entries,
                    action_index=replay_cycle_action_idx,
                )
            else:
                if robot:
                    robot.stop()
            time.sleep(CONTROL_DT)

    flush_auto_step_diagnostic(world, step_key, force=True, emit=True)
    pause_after_fail(robot)
    return False, "success gate not reached"
