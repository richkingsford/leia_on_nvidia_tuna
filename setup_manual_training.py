import argparse
import cv2
import json
import math
import numpy as np
import random
import sys
import threading
import time
import tty
import termios
import statistics
import subprocess
import shutil
from pathlib import Path

from helper_robot_control import Robot
from helper_demo_log_utils import load_demo_logs, normalize_step_label, prune_log_file
from helper_gate_utils import format_gatecheck_stream_lines, load_process_steps
from helper_streaming import start_stream_server
from helper_vision_aruco import ArucoBrickVision
from helper_brick_detector_yolo import BrickDetector as YoloBrickDetector
from helper_manual_config import load_manual_training_config
from helper_vision_config import (
    active_vision_mode as _world_model_active_vision_mode,
    demos_dir_for_mode,
    demos_dirs_by_mode,
    normalize_vision_mode as _normalize_vision_mode_global,
    VISION_MODE_ARUCO as _VISION_MODE_ARUCO_GLOBAL,
    VISION_MODE_CYAN as _VISION_MODE_CYAN_GLOBAL,
)

from helper_align_profile import load_align_profile, inject_align_profile_into_learned_rules
from helper_next import (
    x_axis_curve_lookup_lines,
)
from helper_mini_hotkey_motion_calibrate import (
    RUN_LOG_FILE_DEFAULT as MINI_HOTKEY_MOTION_LOG_DEFAULT,
    run_mini_hotkey_motion_calibration,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import (
    WorldModel,
    TelemetryLogger,
    MotionEvent,
    StepState,
    draw_telemetry_overlay,
    HOTKEY_SPEED_SCORES,
    step_sequence,
    quantize_speed,
)
from telemetry_process import (
    CONTROL_DT,
    _result_lite_gate_detail,
    apply_height_inventory_adjustment_from_step,
    apply_height_snapshot_from_step,
    build_motion_sequence,
    collect_segments,
    consume_auto_step_action_stats,
    compute_stream_gate_summary,
    format_headline,
    COLOR_MAGENTA_BRIGHT,
    format_success_gate_requirements,
    height_intel_step_begin_line,
    merge_motion_steps,
    nominal_actions_from_events,
    replay_segment,
    reset_auto_step_action_stats,
    reset_repeat_act_guard,
    select_demo_segment,
    send_robot_command,
    send_robot_command_pwm,
    step_requires_start_gates,
    update_process_model_from_demos,
    update_world_from_vision,
)
from helper_gatecheck_runtime import (
    format_gate_lines,
    load_process_model,
    refresh_autobuild_config,
)
import telemetry_brick

class _LineStartTracker:
    def __init__(self):
        self.at_line_start = True

class LineStartWriter:
    def __init__(self, wrapped, tracker):
        self.wrapped = wrapped
        self.tracker = tracker

    def write(self, data):
        if not data:
            return 0
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        if not self.tracker.at_line_start and not data.startswith("\n"):
            data = "\n" + data
        lines = data.splitlines(True)
        line_start = self.tracker.at_line_start
        out = []
        for line in lines:
            if line_start:
                line = line.lstrip(" \t")
            out.append(line)
            line_start = line.endswith("\n")
        output = "".join(out)
        n = self.wrapped.write(output)
        self.tracker.at_line_start = output.endswith("\n")
        return n

    def flush(self):
        return self.wrapped.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

_line_start_tracker = _LineStartTracker()
sys.stdout = LineStartWriter(sys.stdout, _line_start_tracker)
sys.stderr = LineStartWriter(sys.stderr, _line_start_tracker)
 
# --- CONFIG ---
_MANUAL_CONFIG = load_manual_training_config()
LOG_RATE_HZ = float(_MANUAL_CONFIG.get("log_rate_hz", 10))
COMMAND_RATE_HZ = float(_MANUAL_CONFIG.get("command_rate_hz", 30))
HEARTBEAT_TIMEOUT = float(_MANUAL_CONFIG.get("heartbeat_timeout", 0.3))
STREAM_HOST = _MANUAL_CONFIG.get("stream_host", "127.0.0.1")
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))
STREAM_IMG_WIDTH = int(_MANUAL_CONFIG.get("stream_img_width", 1600))
DEMOS_DIRS_BY_MODE = demos_dirs_by_mode()
DEMOS_DIR = demos_dir_for_mode()
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
DEMO_STEPS = step_sequence()
try:
    AUTO_CONTINUE_WAIT_S = float(_MANUAL_CONFIG.get("auto_continue_wait_s", 5.0))
except (TypeError, ValueError):
    AUTO_CONTINUE_WAIT_S = 5.0
AUTO_CONTINUE_WAIT_S = max(0.0, float(AUTO_CONTINUE_WAIT_S))
try:
    RUN_LOG_RETENTION_S = float(_MANUAL_CONFIG.get("run_log_retention_s", 3600.0))
except (TypeError, ValueError):
    RUN_LOG_RETENTION_S = 3600.0
RUN_LOG_RETENTION_S = max(0.0, float(RUN_LOG_RETENTION_S))


def _build_stream_success_gate_step_options():
    options = []
    seen = set()
    source_steps = []
    process_steps = load_process_steps()
    if isinstance(process_steps, dict):
        source_steps.extend(process_steps.keys())
    source_steps.extend(getattr(obj, "value", obj) for obj in DEMO_STEPS)
    for step in source_steps:
        step_key = normalize_step_label(step)
        if not step_key or step_key in seen:
            continue
        options.append((step_key, f"{len(options) + 1}. {step_key}"))
        seen.add(step_key)
    return options


STREAM_SUCCESS_GATE_STEP_OPTIONS = _build_stream_success_gate_step_options()
_DEFAULT_STREAM_SUCCESS_GATE_STEP = (
    normalize_step_label(STREAM_SUCCESS_GATE_STEP_OPTIONS[0][0])
    if STREAM_SUCCESS_GATE_STEP_OPTIONS
    else normalize_step_label(StepState.ALIGN_BRICK.value)
)
if not _DEFAULT_STREAM_SUCCESS_GATE_STEP:
    _DEFAULT_STREAM_SUCCESS_GATE_STEP = "ALIGN_BRICK"

CTRL_HELP_LINE = (
    "[CTRL] Drive: W/S 50%, R/F 1%, T/G 100%. Turn: A/D 25%, Q/E 1%, Z/C 100%. "
    "Lift: O/K 1%, U 50%, P/L 100%. F action, ':' command, m auto, y edit hotkey vars, 1 trash log, Q quit"
)
EASE_IN_OUT_HOTKEYS = frozenset({"t", "g", "z", "c"})
ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
ANSI_GREEN_BRIGHT = "\033[92m"
ANSI_RED_BRIGHT = "\033[91m"
ANSI_RESET = "\033[0m"

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}

BRICK_STUDY_FRAMES = 4
BRICK_STUDY_SPLIT_DIFF_MM = 20.0
BRICK_STUDY_SPLIT_DIFF_DEG = 8.0
BRICK_STUDY_OUTLIER_MM = 12.0
BRICK_STUDY_OUTLIER_DEG = 6.0
BRICK_STUDY_STD_LOW_MM = 4.0
BRICK_STUDY_STD_LOW_DEG = 2.0
BRICK_STUDY_STD_HIGH_MM = 8.0
BRICK_STUDY_STD_HIGH_DEG = 4.0

VISION_MODE_ARUCO = _VISION_MODE_ARUCO_GLOBAL
VISION_MODE_CYAN = _VISION_MODE_CYAN_GLOBAL
# Backward-compatible alias.
VISION_MODE_MARKERLESS = VISION_MODE_CYAN
STREAM_VISION_MODE_OPTIONS = [
    (VISION_MODE_ARUCO, "AruCo Markers"),
    (VISION_MODE_CYAN, "Cyan Bricks"),
]
# prefer config2 (no erosion) for cyan vision by default; empirical best during manual testing
CYAN_PROFILE_DEFAULT = "config2_no_erosion"
CYAN_PROFILE_OPTIONS = [
    ("config1_defaults", "Config 1 - Defaults (baseline)"),
    ("config2_no_erosion", "Config 2 - No erosion (best angle, bricks may merge)"),
    ("config3_light_erosion", "Config 3 - Light erosion (angle vs separation balance)"),
    ("config4_heavy_erosion", "Config 4 - Heavy erosion (maximum brick separation)"),
    ("config5_tight_cyan_range", "Config 5 - Tight cyan range (cleaner contours)"),
    ("config6_wide_cyan_range", "Config 6 - Wide cyan range (catches paint variation)"),
    ("config7_smooth", "Config 7 - Smooth (very stable, slow to respond)"),
    ("config8_responsive", "Config 8 - Responsive (tracks motion, noisier)"),
    ("config9_tight_light_smooth", "Config 9 - Tight + light erosion + smooth (recommended)"),
    ("config10_hsv_disabled", "Config 10 - HSV disabled (YOLO-only baseline)"),
]
CYAN_PROFILE_PRESETS = {
    # NOTE: values are intentionally expressed in the detector runtime-tuning
    # API names to keep presets as direct set_runtime_tuning() scenario calls.
    "config1_defaults": {
        "confidence": 0.15,
        "smoothing_alpha": 0.30,
        "hsv_enabled": True,
        "hsv_erode_iterations": 2,
        "hsv_lower": [75, 60, 50],
        "hsv_upper": [115, 255, 255],
    },
    "config2_no_erosion": {
        "hsv_erode_iterations": 0,
    },
    "config3_light_erosion": {
        "hsv_erode_iterations": 1,
    },
    "config4_heavy_erosion": {
        "hsv_erode_iterations": 3,
    },
    "config5_tight_cyan_range": {
        "hsv_lower": [80, 80, 60],
        "hsv_upper": [110, 255, 255],
    },
    "config6_wide_cyan_range": {
        "hsv_lower": [70, 40, 40],
        "hsv_upper": [120, 255, 255],
    },
    "config7_smooth": {
        "smoothing_alpha": 0.15,
    },
    "config8_responsive": {
        "smoothing_alpha": 0.50,
    },
    "config9_tight_light_smooth": {
        "hsv_lower": [80, 80, 60],
        "hsv_upper": [110, 255, 255],
        "hsv_erode_iterations": 1,
        "smoothing_alpha": 0.20,
    },
    "config10_hsv_disabled": {
        "hsv_enabled": False,
    },
}
_CYAN_PROFILE_ALIASES = {
    "config1": "config1_defaults",
    "config2": "config2_no_erosion",
    "config3": "config3_light_erosion",
    "config4": "config4_heavy_erosion",
    "config5": "config5_tight_cyan_range",
    "config6": "config6_wide_cyan_range",
    "config7": "config7_smooth",
    "config8": "config8_responsive",
    "config9": "config9_tight_light_smooth",
    "config10": "config10_hsv_disabled",
    "1": "config1_defaults",
    "2": "config2_no_erosion",
    "3": "config3_light_erosion",
    "4": "config4_heavy_erosion",
    "5": "config5_tight_cyan_range",
    "6": "config6_wide_cyan_range",
    "7": "config7_smooth",
    "8": "config8_responsive",
    "9": "config9_tight_light_smooth",
    "10": "config10_hsv_disabled",
    "default": "config1_defaults",
    "baseline": "config1_defaults",
    # Backward-compatibility for old profile ids in existing local config.
    "balanced": "config1_defaults",
    "sensitive": "config1_defaults",
    "aggressive": "config1_defaults",
    "rescue": "config1_defaults",
    "stable": "config1_defaults",
    "legacy": "config1_defaults",
}
CYAN_VISIBILITY_AUTO = "auto"
CYAN_VISIBILITY_GRAY = "gray"
CYAN_VISIBILITY_CYAN = "cyan"
CYAN_VISIBILITY_OPTIONS = [
    (CYAN_VISIBILITY_AUTO, "Auto (all available)"),
    (CYAN_VISIBILITY_GRAY, "Gray Brick"),
    (CYAN_VISIBILITY_CYAN, "Cyan Brick"),
]
_CYAN_VISIBILITY_ALIASES = {
    "auto": CYAN_VISIBILITY_AUTO,
    "all": CYAN_VISIBILITY_AUTO,
    "both": CYAN_VISIBILITY_AUTO,
    "gray": CYAN_VISIBILITY_GRAY,
    "grey": CYAN_VISIBILITY_GRAY,
    "cyan": CYAN_VISIBILITY_CYAN,
}
_VISION_MODE_ALIASES = {
    "aruco": VISION_MODE_ARUCO,
    "cyan": VISION_MODE_CYAN,
    "yolo": VISION_MODE_CYAN,
    "markerless": VISION_MODE_CYAN,
}


def normalize_vision_mode(value, fallback=VISION_MODE_CYAN):
    return _normalize_vision_mode_global(value, fallback=fallback)


def hotkey_uses_ease_in_out(hotkey):
    key = str(hotkey or "").strip().lower()
    return key in EASE_IN_OUT_HOTKEYS


_DEFAULT_VISION_MODE = normalize_vision_mode(
    _world_model_active_vision_mode()
    or _MANUAL_CONFIG.get("brick_vision", VISION_MODE_CYAN),
    fallback=VISION_MODE_CYAN,
)
_DEFAULT_CYAN_PROFILE = str(
    _MANUAL_CONFIG.get(
        "cyan_profile",
        _MANUAL_CONFIG.get("markerless_profile", CYAN_PROFILE_DEFAULT),
    )
).strip().lower()
if _DEFAULT_CYAN_PROFILE not in CYAN_PROFILE_PRESETS:
    _DEFAULT_CYAN_PROFILE = CYAN_PROFILE_DEFAULT
_DEFAULT_CYAN_VISIBILITY = str(
    _MANUAL_CONFIG.get(
        "cyan_visibility",
        _MANUAL_CONFIG.get("markerless_visibility", CYAN_VISIBILITY_AUTO),
    )
).strip().lower()
_DEFAULT_CYAN_VISIBILITY = _CYAN_VISIBILITY_ALIASES.get(
    _DEFAULT_CYAN_VISIBILITY,
    CYAN_VISIBILITY_AUTO,
)
try:
    CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX = float(
        _MANUAL_CONFIG.get("cyan_experiment_smoothing_alpha_max", 0.15)
    )
except (TypeError, ValueError):
    CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX = 0.15
CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX = max(
    0.0,
    min(1.0, float(CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX)),
)

# Backward-compatible aliases while callers migrate to cyan naming.
MARKERLESS_PROFILE_BALANCED = CYAN_PROFILE_DEFAULT
MARKERLESS_PROFILE_OPTIONS = CYAN_PROFILE_OPTIONS
MARKERLESS_PROFILE_PRESETS = CYAN_PROFILE_PRESETS
_MARKERLESS_PROFILE_ALIASES = _CYAN_PROFILE_ALIASES
MARKERLESS_VISIBILITY_AUTO = CYAN_VISIBILITY_AUTO
MARKERLESS_VISIBILITY_GRAY = CYAN_VISIBILITY_GRAY
MARKERLESS_VISIBILITY_CYAN = CYAN_VISIBILITY_CYAN
MARKERLESS_VISIBILITY_OPTIONS = CYAN_VISIBILITY_OPTIONS
_MARKERLESS_VISIBILITY_ALIASES = _CYAN_VISIBILITY_ALIASES
_DEFAULT_MARKERLESS_PROFILE = _DEFAULT_CYAN_PROFILE
_DEFAULT_MARKERLESS_VISIBILITY = _DEFAULT_CYAN_VISIBILITY

AUTO_PREFLIGHT_HOTKEY_MOTION_ENABLED = bool(_MANUAL_CONFIG.get("auto_preflight_hotkey_motion_enabled", True))
AUTO_PREFLIGHT_HOTKEY_MOTION_LOG = Path(
    _MANUAL_CONFIG.get("auto_preflight_hotkey_motion_log", str(MINI_HOTKEY_MOTION_LOG_DEFAULT))
)
AUTO_PREFLIGHT_HOTKEY_KEYS = ("r", "f", "o", "k")
AUTO_PREFLIGHT_LIFT_GROUND_ENABLED = bool(_MANUAL_CONFIG.get("auto_preflight_lift_ground_enabled", True))
LIFT_GROUND_PREFLIGHT_SAMPLE_FRAMES = int(_MANUAL_CONFIG.get("lift_ground_preflight_sample_frames", 4))
LIFT_GROUND_PREFLIGHT_SAMPLE_TIMEOUT_S = float(_MANUAL_CONFIG.get("lift_ground_preflight_sample_timeout_s", 1.5))
LIFT_GROUND_PREFLIGHT_MAX_DOWN_PULSES = int(_MANUAL_CONFIG.get("lift_ground_preflight_max_down_pulses", 10))
LIFT_GROUND_PREFLIGHT_SETTLE_S = float(_MANUAL_CONFIG.get("lift_ground_preflight_settle_s", 0.05))
LIFT_GROUND_PREFLIGHT_NO_MOVE_MM = float(_MANUAL_CONFIG.get("lift_ground_preflight_no_move_mm", 0.35))
LIFT_GROUND_PREFLIGHT_STABLE_PULSES = int(_MANUAL_CONFIG.get("lift_ground_preflight_stable_pulses", 2))
LIFT_GROUND_PREFLIGHT_MIN_CONF = float(_MANUAL_CONFIG.get("lift_ground_preflight_min_conf", 50.0))
AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_ENABLED = bool(
    _MANUAL_CONFIG.get("auto_align_mini_visibility_recovery_enabled", True)
)
AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_SCORE = int(
    _MANUAL_CONFIG.get("auto_align_mini_visibility_recovery_score", 1)
)
AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_MAX_ACTS = int(
    _MANUAL_CONFIG.get("auto_align_mini_visibility_recovery_max_acts", 24)
)
AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_TIMEOUT_S = float(
    _MANUAL_CONFIG.get("auto_align_mini_visibility_recovery_timeout_s", 5.0)
)

def _vision_mode_cli_value(mode):
    mode_norm = normalize_vision_mode(mode)
    if mode_norm == VISION_MODE_CYAN:
        return VISION_MODE_CYAN
    return VISION_MODE_ARUCO


def _demos_dir_for_vision_mode(mode):
    mode_norm = normalize_vision_mode(mode, fallback=_DEFAULT_VISION_MODE)
    path = DEMOS_DIRS_BY_MODE.get(mode_norm)
    if path is None:
        path = demos_dir_for_mode(mode_norm)
    return Path(path)


def _runs_dir_for_vision_mode(mode):
    mode_norm = normalize_vision_mode(mode, fallback=_DEFAULT_VISION_MODE)
    suffix = "cyan" if mode_norm == VISION_MODE_CYAN else "aruco"
    return Path(__file__).resolve().parent / f"Runs - {suffix}"


def _run_retention_dirs():
    base_dir = Path(__file__).resolve().parent
    return (
        base_dir / "Runs - aruco",
        base_dir / "Runs - cyan",
    )


def _cleanup_stale_run_logs(*, run_folders=None, cutoff_age_s=None):
    folders = tuple(run_folders or _run_retention_dirs())
    if cutoff_age_s is None:
        cutoff_age_s = float(RUN_LOG_RETENTION_S)
    try:
        from calibration.helper_calibrate import cleanup_old_run_files

        cleanup_old_run_files(
            preserve_live_files=(),
            run_folders=folders,
            cutoff_age_s=float(cutoff_age_s),
        )
    except Exception:
        pass


def normalize_cyan_profile(value, fallback=CYAN_PROFILE_DEFAULT):
    key = str(value or "").strip().lower()
    key = _CYAN_PROFILE_ALIASES.get(key, key)
    if key in CYAN_PROFILE_PRESETS:
        return key
    fallback_key = str(fallback or "").strip().lower()
    fallback_key = _CYAN_PROFILE_ALIASES.get(fallback_key, fallback_key)
    if fallback_key in CYAN_PROFILE_PRESETS:
        return fallback_key
    return CYAN_PROFILE_DEFAULT


def cyan_profile_settings(profile):
    profile_key = normalize_cyan_profile(profile, fallback=_DEFAULT_CYAN_PROFILE)
    settings = CYAN_PROFILE_PRESETS.get(profile_key) or CYAN_PROFILE_PRESETS[CYAN_PROFILE_DEFAULT]
    return profile_key, dict(settings)


def cyan_profile_runtime_summary(runtime):
    if not isinstance(runtime, dict):
        return "runtime=unknown"
    parts = []
    conf = runtime.get("conf_threshold")
    if isinstance(conf, (int, float)):
        parts.append(f"min_conf={float(conf):.2f}")
    smooth = runtime.get("smooth_alpha")
    if isinstance(smooth, (int, float)):
        parts.append(f"smooth={float(smooth):.2f}")
    hsv_enabled = runtime.get("hsv_enabled")
    if hsv_enabled is not None:
        parts.append("hsv=on" if bool(hsv_enabled) else "hsv=off")
    erode = runtime.get("hsv_erode_iterations")
    if isinstance(erode, (int, float)):
        parts.append(f"erode={int(erode)}")
    hsv_lower = runtime.get("hsv_lower")
    hsv_upper = runtime.get("hsv_upper")
    if isinstance(hsv_lower, (list, tuple)) and isinstance(hsv_upper, (list, tuple)):
        parts.append(f"hsv_range={list(hsv_lower)}..{list(hsv_upper)}")
    return ", ".join(parts) if parts else "runtime=unchanged"


def cyan_profile_label(profile):
    profile_key = normalize_cyan_profile(profile, fallback=_DEFAULT_CYAN_PROFILE)
    for value, label in CYAN_PROFILE_OPTIONS:
        if value == profile_key:
            return label
    return profile_key


def normalize_cyan_visibility(value, fallback=CYAN_VISIBILITY_AUTO):
    key = str(value or "").strip().lower()
    mode = _CYAN_VISIBILITY_ALIASES.get(key)
    if mode:
        return mode
    fallback_key = str(fallback or "").strip().lower()
    fallback_mode = _CYAN_VISIBILITY_ALIASES.get(fallback_key)
    if fallback_mode:
        return fallback_mode
    return CYAN_VISIBILITY_AUTO


def cyan_visibility_label(mode):
    mode_key = normalize_cyan_visibility(mode, fallback=_DEFAULT_CYAN_VISIBILITY)
    for value, label in CYAN_VISIBILITY_OPTIONS:
        if value == mode_key:
            return label
    return mode_key


# Backward-compatible API names.
def normalize_markerless_profile(value, fallback=MARKERLESS_PROFILE_BALANCED):
    return normalize_cyan_profile(value, fallback=fallback)


def markerless_profile_settings(profile):
    return cyan_profile_settings(profile)


def markerless_profile_runtime_summary(runtime):
    return cyan_profile_runtime_summary(runtime)


def markerless_profile_label(profile):
    return cyan_profile_label(profile)


def normalize_markerless_visibility(value, fallback=MARKERLESS_VISIBILITY_AUTO):
    return normalize_cyan_visibility(value, fallback=fallback)


def markerless_visibility_label(mode):
    return cyan_visibility_label(mode)

STEP_CODES = {str(idx + 1): obj for idx, obj in enumerate(DEMO_STEPS)}

STEP_NAMES = {obj.value.lower(): obj for obj in DEMO_STEPS}
STEP_NAMES["wall"] = StepState.FIND_WALL
STEP_NAMES["find"] = StepState.FIND_BRICK
STEP_NAMES["supply"] = StepState.FIND_BRICK
STEP_NAMES["find_supply"] = StepState.FIND_BRICK
STEP_NAMES["avs"] = StepState.APPROACH_VECTOR_BRICK_SUPPLY
STEP_NAMES["approach_supply"] = StepState.APPROACH_VECTOR_BRICK_SUPPLY
STEP_NAMES["approach_vector_brick_supply"] = StepState.APPROACH_VECTOR_BRICK_SUPPLY
STEP_NAMES["top"] = StepState.FIND_TOPMOST_BRICK
STEP_NAMES["topmost"] = StepState.FIND_TOPMOST_BRICK
STEP_NAMES["topwall"] = StepState.FIND_TOPMOST_BRICK_WALL
STEP_NAMES["topmost_wall"] = StepState.FIND_TOPMOST_BRICK_WALL
STEP_NAMES["wall_top"] = StepState.FIND_TOPMOST_BRICK_WALL
STEP_NAMES["find_topmost_brick_wall"] = StepState.FIND_TOPMOST_BRICK_WALL
STEP_NAMES["lock"] = StepState.BRICK_LOCK
STEP_NAMES["brick_lock"] = StepState.BRICK_LOCK
STEP_NAMES["wall_lock"] = StepState.BRICK_LOCK_WALL
STEP_NAMES["brick_lock_wall"] = StepState.BRICK_LOCK_WALL
STEP_NAMES["lock_wall"] = StepState.BRICK_LOCK_WALL
STEP_NAMES["align"] = StepState.ALIGN_BRICK
STEP_NAMES["seat"] = StepState.SEAT_BRICK
STEP_NAMES["scoop"] = StepState.SEAT_BRICK
STEP_NAMES["seat2"] = StepState.SEAT_BRICK2
STEP_NAMES["seat_brick2"] = StepState.SEAT_BRICK2
STEP_NAMES["elevate"] = StepState.ELEVATE_BRICK
STEP_NAMES["lift"] = StepState.ELEVATE_BRICK
STEP_NAMES["carry"] = StepState.FIND_WALL2
STEP_NAMES["wall2"] = StepState.FIND_WALL2
STEP_NAMES["avw"] = StepState.APPROACH_VECTOR_WALL
STEP_NAMES["approach_wall"] = StepState.APPROACH_VECTOR_WALL
STEP_NAMES["approach_vector_wall"] = StepState.APPROACH_VECTOR_WALL
STEP_NAMES["position"] = StepState.POSITION_BRICK
STEP_NAMES["retreat"] = StepState.RETREAT
STEP_NAMES["place"] = StepState.RETREAT

STEP_CODE_EMOJIS = {
    "find_wall": "🟠",
    "exit_wall": "🔴",
    "find_brick": "🟠",
    "approach_vector_brick_supply": "📐",
    "find_topmost_brick": "🎩",
    "find_topmost_brick_wall": "🎩",
    "brick_lock": "🟢",
    "brick_lock_wall": "🟢",
    "align_brick": "🟢",
    "seat_brick": "🟢",
    "seat_brick2": "🟢",
    "elevate_brick": "🟠",
    "find_wall2": "🟠",
    "approach_vector_wall": "📐",
    "position_brick": "🟢",
    "retreat": "🟢",
    "place": "🟢",
}

ATTEMPT_CODES = {
    "f": "FAIL",
    "s": "SUCCESS",
    "r": "RECOVER",
    "n": "NOMINAL",
}

ATTEMPT_NAMES = {
    "fail": "FAIL",
    "failure": "FAIL",
    "success": "SUCCESS",
    "recover": "RECOVER",
    "recovery": "RECOVER",
    "nominal": "NOMINAL",
    "nom": "NOMINAL",
}

ATTEMPT_MARKERS = {
    "FAIL": ("FAIL_START", "FAIL_END"),
    "RECOVER": ("RECOVER_START", "RECOVER_END"),
    "SUCCESS": ("SUCCESS_START", "SUCCESS_END"),
    "NOMINAL": ("NOMINAL_START", "NOMINAL_END"),
}

ATTEMPT_STATUS = {
    "FAIL": "FAIL",
    "RECOVER": "RECOVERY",
    "SUCCESS": "NORMAL",
    "NOMINAL": "NOMINAL",
}


class AutoStepCancelled(Exception):
    """Raised to cooperatively abort a running auto-step (e.g. Esc pressed)."""
    pass


def _is_repeat_act_safety_fail(reason):
    text = str(reason or "").strip().lower()
    if not text:
        return False
    if "[safety-fail]" in text and "repeat" in text:
        return True
    return "repeat-act guard" in text


def step_label(obj_enum):
    return obj_enum.value

def log_line(message):
    print(str(message).strip(), flush=True)


def _command_preview_text(cmd, *, speed_score=None, pwm=None, power=None, duration_ms=None):
    cmd_key = str(cmd or "").strip().lower()
    if not cmd_key:
        return ""
    head = str(cmd_key)
    if speed_score is not None:
        try:
            head = f"{head} {int(round(float(speed_score)))}%"
        except (TypeError, ValueError):
            pass
    parts = [head]
    if pwm is not None:
        try:
            parts.append(f"pwm={int(round(float(pwm)))}")
        except (TypeError, ValueError):
            pass
    if power is not None:
        try:
            parts.append(f"pwr={float(power):.3f}")
        except (TypeError, ValueError):
            pass
    if duration_ms is not None:
        try:
            parts.append(f"t={int(round(float(duration_ms)))}ms")
        except (TypeError, ValueError):
            pass
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ({', '.join(parts[1:])})"


def _fmt_bool_text(value):
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "?"


def _colorize_match_text(text, matched):
    if matched is True:
        return f"{ANSI_GREEN_BRIGHT}{text}{ANSI_RESET}"
    if matched is False:
        return f"{ANSI_RED_BRIGHT}{text}{ANSI_RESET}"
    return str(text)


def _brick_start_gate_metric_value(world, metric):
    brick = (getattr(world, "brick", None) or {}) if world is not None else {}
    if metric == "visible":
        return bool(brick.get("visible"))
    if metric in ("brick_above", "brickAbove"):
        if "brickAbove" in brick:
            val = brick.get("brickAbove")
            return None if val is None else bool(val)
        val = brick.get("brick_above")
        return None if val is None else bool(val)
    if metric in ("brick_below", "brickBelow"):
        if "brickBelow" in brick:
            val = brick.get("brickBelow")
            return None if val is None else bool(val)
        val = brick.get("brick_below")
        return None if val is None else bool(val)
    if metric == "angle_abs":
        val = brick.get("angle")
        if val is None:
            return None
        try:
            return abs(float(val))
        except (TypeError, ValueError):
            return None
    if metric in ("xAxis_offset_abs", "xAxis_offset"):
        val = brick.get("x_axis", brick.get("offset_x"))
        if val is None:
            return None
        try:
            valf = float(val)
        except (TypeError, ValueError):
            return None
        return abs(valf) if metric.endswith("_abs") else valf
    if metric in ("yAxis_offset_abs", "yAxis_offset"):
        val = brick.get("y_axis", brick.get("offset_y"))
        if val is None:
            return None
        try:
            valf = float(val)
        except (TypeError, ValueError):
            return None
        return abs(valf) if metric.endswith("_abs") else valf
    if metric in ("dist", "distance"):
        val = brick.get("dist")
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return None


def _format_start_gate_requirement(stats):
    if not isinstance(stats, dict):
        return "?"
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if isinstance(target, bool):
            return _fmt_bool_text(bool(target))
        try:
            return f"{float(target):+.1f}+/-{abs(float(tol)):.1f}"
        except (TypeError, ValueError):
            return str(target)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        return _fmt_bool_text(bool(min_val))
    if isinstance(max_val, bool):
        return _fmt_bool_text(bool(max_val))
    if min_val is not None and max_val is not None:
        try:
            return f"[{float(min_val):.1f},{float(max_val):.1f}]"
        except (TypeError, ValueError):
            return f"[{min_val},{max_val}]"
    if min_val is not None:
        try:
            return f">={float(min_val):.1f}"
        except (TypeError, ValueError):
            return f">={min_val}"
    if max_val is not None:
        try:
            return f"<={float(max_val):.1f}"
        except (TypeError, ValueError):
            return f"<={max_val}"
    return "?"


def _start_gate_metric_match(value, stats):
    if value is None or not isinstance(stats, dict):
        return None
    min_val = stats.get("min")
    max_val = stats.get("max")
    target = stats.get("target")
    tol = stats.get("tol")
    if isinstance(value, bool):
        if isinstance(min_val, bool):
            return bool(value) == bool(min_val)
        if isinstance(max_val, bool):
            return bool(value) == bool(max_val)
        if isinstance(target, bool):
            return bool(value) == bool(target)
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if target is not None and tol is not None:
        try:
            return abs(val - float(target)) <= abs(float(tol))
        except (TypeError, ValueError):
            return None
    try:
        if min_val is not None and val < float(min_val):
            return False
        if max_val is not None and val > float(max_val):
            return False
    except (TypeError, ValueError):
        return None
    return True


def _format_start_gate_current(metric, value):
    if value is None:
        return "?"
    if isinstance(value, bool):
        return _fmt_bool_text(bool(value))
    try:
        return f"{float(value):+.1f}"
    except (TypeError, ValueError):
        return str(value)


def _format_brick_start_gate_details(world, start_gates):
    if not isinstance(start_gates, dict) or not start_gates:
        return None
    parts = []
    for metric, stats in start_gates.items():
        value = _brick_start_gate_metric_value(world, metric)
        # Only format metrics that read from brick state here.
        if value is None and metric not in ("visible", "brick_above", "brickAbove", "brick_below", "brickBelow"):
            continue
        req_txt = _format_start_gate_requirement(stats)
        cur_txt_plain = _format_start_gate_current(metric, value)
        matched = _start_gate_metric_match(value, stats)
        cur_txt = _colorize_match_text(cur_txt_plain, matched)
        parts.append(f"{metric}={req_txt} ({cur_txt})")
    return ", ".join(parts) if parts else None


def log_align_curve_lookup_table():
    for line in x_axis_curve_lookup_lines():
        log_line(str(line))


def _cmd_to_motion_action_type(cmd):
    cmd_key = str(cmd or "").strip().lower()
    return {
        "f": "forward",
        "b": "backward",
        "l": "left_turn",
        "r": "right_turn",
        "u": "mast_up",
        "d": "mast_down",
    }.get(cmd_key, "unknown")


def _demo_logging_active(app_state):
    if getattr(app_state, "active_attempt", None):
        return True
    return bool(getattr(app_state, "auto_demo_logging_active", False))


def _log_motion_event_and_state(app_state, cmd, speed, score, duration_ms=None):
    if app_state.logger is None or getattr(app_state, "logger_closed", True):
        return
    if not _demo_logging_active(app_state):
        return
    action_type = _cmd_to_motion_action_type(cmd)
    power_u8 = int(max(0.0, min(1.0, float(speed or 0.0))) * 255)
    try:
        dur_ms = int(round(float(duration_ms))) if duration_ms is not None else int(CONTROL_DT * 1000)
    except (TypeError, ValueError):
        dur_ms = int(CONTROL_DT * 1000)
    evt = MotionEvent(action_type, power_u8, max(1, int(dur_ms)), speed_score=score)
    app_state.logger.log_event(evt, app_state.world.step_state.value)
    app_state.logger.log_state(app_state.world)
    with app_state.lock:
        if getattr(app_state, "active_attempt", None):
            app_state.current_attempt_action_count = int(
                getattr(app_state, "current_attempt_action_count", 0) or 0
            ) + 1
        elif bool(getattr(app_state, "auto_demo_logging_active", False)):
            app_state.current_auto_action_count = int(
                getattr(app_state, "current_auto_action_count", 0) or 0
            ) + 1


def _apply_manual_motion_telemetry_from_send(app_state, *, cmd, speed, score, send_result=None):
    if app_state is None or app_state.world is None or not cmd:
        return
    result = send_result if isinstance(send_result, dict) else {}
    # `cmd_sent` may be a wire-level remap (e.g. lift inversion). Manual demo logs
    # and world-motion telemetry must record the semantic command the user pressed.
    cmd_for_motion = str(cmd or "").strip().lower()
    if cmd_for_motion not in ("f", "b", "l", "r", "u", "d"):
        cmd_for_motion = str(result.get("cmd_sent") or "").strip().lower()
    action_type = _cmd_to_motion_action_type(cmd_for_motion)
    if action_type == "unknown":
        return
    try:
        power_used = float(result.get("power", speed))
    except (TypeError, ValueError):
        power_used = float(speed or 0.0)
    power_used = max(0.0, min(1.0, power_used))
    try:
        duration_used_ms = int(round(float(result.get("duration_ms"))))
    except (TypeError, ValueError):
        duration_used_ms = None
    if duration_used_ms is None or duration_used_ms <= 0:
        try:
            duration_used_ms = int(round(float(app_state.active_duration_ms)))
        except (TypeError, ValueError):
            duration_used_ms = int(CONTROL_DT * 1000)
    evt = MotionEvent(
        action_type,
        int(round(power_used * 255.0)),
        max(1, int(duration_used_ms)),
        speed_score=score,
    )
    app_state.world.update_from_motion(evt)
    _log_motion_event_and_state(app_state, cmd_for_motion, power_used, score, int(duration_used_ms))


def _collect_lift_cam_h_samples(app_state, *, samples=4, timeout_s=1.5):
    vals = []
    vision = getattr(app_state, "vision", None)
    if vision is None:
        return vals
    started = time.time()
    requested = max(1, int(samples or 1))
    while len(vals) < requested and (time.time() - started) < float(timeout_s) and app_state.running:
        with app_state.vision_io_lock:
            found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            conf_val = 0.0
        try:
            cam_h_val = float(cam_h)
        except (TypeError, ValueError):
            cam_h_val = 0.0
        with app_state.lock:
            app_state.world.update_vision(
                bool(found),
                float(dist or 0.0),
                float(angle or 0.0),
                conf_val,
                float(offset_x or 0.0),
                cam_h_val,
                bool(brick_above),
                bool(brick_below),
            )
        if bool(found) and conf_val >= float(LIFT_GROUND_PREFLIGHT_MIN_CONF) and cam_h_val > 0.0:
            vals.append(float(cam_h_val))
        time.sleep(0.02)
    return vals


def _auto_align_visible_now(app_state):
    if app_state is None:
        return False
    world = getattr(app_state, "world", None)
    vision = getattr(app_state, "vision", None)
    if world is None or vision is None:
        return False
    lock = getattr(app_state, "vision_io_lock", None)
    try:
        if lock is not None:
            with lock:
                update_world_from_vision(world, vision, log=False)
        else:
            update_world_from_vision(world, vision, log=False)
    except Exception:
        return False
    brick = getattr(world, "brick", None)
    if not isinstance(brick, dict):
        return False
    return bool(brick.get("visible"))


def run_align_mini_visibility_recovery(app_state, *, step_key="ALIGN_BRICK"):
    if not AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_ENABLED:
        return {"ok": True, "skipped": True, "reason": "disabled"}
    if app_state is None or getattr(app_state, "robot", None) is None or getattr(app_state, "vision", None) is None:
        return {"ok": True, "skipped": True, "reason": "robot/vision unavailable"}

    step_name = normalize_step_label(step_key or "ALIGN_BRICK")
    if step_name != "ALIGN_BRICK":
        return {"ok": True, "skipped": True, "reason": "step not align"}

    max_acts = max(1, int(AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_MAX_ACTS))
    timeout_s = max(0.5, float(AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_TIMEOUT_S))
    score = max(1, int(AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_SCORE))

    if _auto_align_visible_now(app_state):
        log_line("[AUTO] ALIGN_BRICK mini x-axis recovery: brick already visible; no right-turn needed.")
        return {"ok": True, "visible": True, "acts": 0, "reason": "already_visible"}

    log_line(
        "[AUTO] ALIGN_BRICK mini x-axis recovery: turning right until brick is visible again."
    )

    acts = 0
    deadline = float(time.time()) + float(timeout_s)
    while bool(getattr(app_state, "running", True)) and acts < max_acts and float(time.time()) < deadline:
        send_result = send_robot_command(
            app_state.robot,
            app_state.world,
            StepState.ALIGN_BRICK,
            "r",
            0.0,
            speed_score=score,
        )
        acts += 1
        try:
            duration_ms = int(round(float((send_result or {}).get("duration_ms", 0))))
        except (TypeError, ValueError):
            duration_ms = 0
        settle_s = max(0.02, min(0.25, float(duration_ms) / 1000.0 + 0.02))
        time.sleep(settle_s)
        if _auto_align_visible_now(app_state):
            log_line(
                f"[AUTO] ALIGN_BRICK mini x-axis recovery: visible=true after {int(acts)} right-turn acts."
            )
            return {"ok": True, "visible": True, "acts": int(acts), "reason": "visible_recovered"}

    visible_now = _auto_align_visible_now(app_state)
    if bool(visible_now):
        log_line(
            f"[AUTO] ALIGN_BRICK mini x-axis recovery: visible=true after {int(acts)} right-turn acts."
        )
        return {"ok": True, "visible": True, "acts": int(acts), "reason": "visible_recovered_late"}

    log_line(
        "[AUTO] ALIGN_BRICK mini x-axis recovery: visibility not recovered; "
        f"continuing (acts={int(acts)}, timeout={float(timeout_s):.1f}s)."
    )
    return {"ok": False, "visible": False, "acts": int(acts), "reason": "visible_not_recovered"}


def _current_hotkey_row(hotkey):
    key = str(hotkey or "").strip().lower()
    row = HOTKEY_SPEED_SCORES.get(key)
    return row if isinstance(row, dict) else {}


def run_lift_ground_level_preflight(app_state, *, force=False):
    if app_state is None:
        return {"ok": False, "error": "no app state"}
    if not AUTO_PREFLIGHT_LIFT_GROUND_ENABLED and not force:
        return {"ok": True, "skipped": True, "reason": "disabled"}

    with app_state.lock:
        already = bool(getattr(app_state, "lift_ground_preflight_done", False))
        running = bool(getattr(app_state, "lift_ground_preflight_running", False))
        if already and not force:
            return {"ok": True, "skipped": True, "reason": "already calibrated"}
        if running:
            return {"ok": False, "busy": True}
        app_state.lift_ground_preflight_running = True

    try:
        if app_state.robot is None or app_state.vision is None:
            log_line("[PREFLIGHT LIFT] Skipped (robot/vision unavailable).")
            return {"ok": True, "skipped": True, "reason": "robot/vision unavailable"}

        log_line("[PREFLIGHT LIFT] Discovering lift micro movement (O/K) and finding ground level...")

        # Reuse the existing mini motion calibrator so lift minimal movement is tuned
        # together with the same logic we use for other hotkeys.
        with app_state.vision_io_lock:
            mini_result = run_mini_hotkey_motion_calibration(
                robot=app_state.robot,
                vision=app_state.vision,
                hotkeys=("o", "k"),
                log_path=AUTO_PREFLIGHT_HOTKEY_MOTION_LOG,
                log_fn=log_line,
            )
        if not bool(mini_result.get("ok")):
            log_line(f"[PREFLIGHT LIFT] Lift mini discovery failed ({mini_result.get('error')}).")
        else:
            with app_state.lock:
                app_state.preflight_checked_hotkeys.update({"o", "k"})

        baseline_vals = _collect_lift_cam_h_samples(
            app_state,
            samples=LIFT_GROUND_PREFLIGHT_SAMPLE_FRAMES,
            timeout_s=LIFT_GROUND_PREFLIGHT_SAMPLE_TIMEOUT_S,
        )
        if not baseline_vals:
            log_line("[PREFLIGHT LIFT] Could not see a brick marker to calibrate lift ground.")
            return {"ok": False, "error": "no marker samples"}
        baseline_cam_h = float(statistics.median(baseline_vals))

        row_k = _current_hotkey_row("k")
        cmd_down = str(row_k.get("cmd", "d")).strip().lower()
        if cmd_down != "d":
            cmd_down = "d"
        try:
            score_down = int(row_k.get("score", 1))
        except (TypeError, ValueError):
            score_down = 1
        duration_down_ms = row_k.get("duration_ms")

        stable_no_move = 0
        pulses_sent = 0
        last_cam_h = baseline_cam_h
        pulse_records = []
        for _ in range(max(1, int(LIFT_GROUND_PREFLIGHT_MAX_DOWN_PULSES))):
            pulses_sent += 1
            send_result = send_robot_command(
                app_state.robot,
                app_state.world,
                app_state.world.step_state,
                cmd_down,
                0.0,
                speed_score=score_down,
                duration_override_ms=duration_down_ms,
            )
            _apply_manual_motion_telemetry_from_send(
                app_state,
                cmd=cmd_down,
                speed=0.0,
                score=score_down,
                send_result=send_result,
            )
            try:
                sleep_s = max(
                    float(LIFT_GROUND_PREFLIGHT_SETTLE_S),
                    min(0.25, float((send_result or {}).get("duration_ms", 0)) / 1000.0 + 0.02),
                )
            except Exception:
                sleep_s = float(LIFT_GROUND_PREFLIGHT_SETTLE_S)
            time.sleep(sleep_s)
            vals = _collect_lift_cam_h_samples(
                app_state,
                samples=LIFT_GROUND_PREFLIGHT_SAMPLE_FRAMES,
                timeout_s=LIFT_GROUND_PREFLIGHT_SAMPLE_TIMEOUT_S,
            )
            if not vals:
                break
            cam_h_now = float(statistics.median(vals))
            delta_mm = float(cam_h_now - last_cam_h)
            pulse_records.append({"pulse": int(pulses_sent), "cam_h": cam_h_now, "delta_mm": delta_mm})
            if abs(delta_mm) <= float(LIFT_GROUND_PREFLIGHT_NO_MOVE_MM):
                stable_no_move += 1
            else:
                stable_no_move = 0
            last_cam_h = cam_h_now
            if stable_no_move >= max(1, int(LIFT_GROUND_PREFLIGHT_STABLE_PULSES)):
                break

        final_vals = _collect_lift_cam_h_samples(
            app_state,
            samples=LIFT_GROUND_PREFLIGHT_SAMPLE_FRAMES,
            timeout_s=LIFT_GROUND_PREFLIGHT_SAMPLE_TIMEOUT_S,
        )
        if not final_vals:
            final_vals = [last_cam_h]
        final_cam_h = float(statistics.median(final_vals))

        with app_state.lock:
            brick_height = float(app_state.world.height_mm or 0.0) if app_state.world.height_mm is not None else 0.0
            app_state.world.lift_height = 0.0
            app_state.world.lift_height_anchor = float(final_cam_h + brick_height)
            app_state.lift_ground_preflight_done = True
            app_state.lift_ground_preflight_cam_h = float(final_cam_h)
            app_state.lift_ground_preflight_time = float(time.time())
        refresh_brick_telemetry(app_state, read_vision=False)

        log_line(
            "[PREFLIGHT LIFT] Ground level calibrated: "
            f"cam_h={final_cam_h:.2f}mm, anchor={float(app_state.world.lift_height_anchor):.2f}, "
            f"down_pulses={int(pulses_sent)}."
        )
        return {
            "ok": True,
            "cam_h_ground": float(final_cam_h),
            "anchor": float(app_state.world.lift_height_anchor),
            "down_pulses": int(pulses_sent),
            "pulse_records": pulse_records,
            "mini_result_ok": bool(mini_result.get("ok")) if isinstance(mini_result, dict) else False,
        }
    finally:
        with app_state.lock:
            app_state.lift_ground_preflight_running = False


def run_lift_preflight_check_if_needed(app_state, *, cmd=None, force=False):
    cmd_key = str(cmd or "").strip().lower()
    if not force and cmd_key not in ("u", "d"):
        return True
    result = run_lift_ground_level_preflight(app_state, force=force)
    if result.get("busy"):
        return False
    if not bool(result.get("ok")) and not bool(result.get("skipped")):
        # Do not block manual control forever; surface error and continue.
        log_line(f"[PREFLIGHT LIFT] Continuing despite calibration issue ({result.get('error')}).")
    return True


def _begin_auto_demo_logging(app_state, obj_enum):
    ensure_log_open(app_state)
    marker = ATTEMPT_MARKERS["NOMINAL"][0]
    obj_label = step_label(obj_enum)
    app_state.logger.log_keyframe(marker, obj_label)
    app_state.world.recording_active = True
    app_state.world.attempt_status = ATTEMPT_STATUS["NOMINAL"]
    app_state.auto_demo_logging_active = True
    app_state.current_auto_step = normalize_step_label(obj_label) or obj_label
    app_state.current_auto_action_count = 0


def _end_auto_demo_logging(app_state, obj_enum):
    if app_state.logger is None or getattr(app_state, "logger_closed", True):
        app_state.auto_demo_logging_active = False
        app_state.world.recording_active = False
        app_state.world.attempt_status = "NORMAL"
        app_state.current_auto_step = None
        app_state.current_auto_action_count = 0
        return
    marker = ATTEMPT_MARKERS["NOMINAL"][1]
    obj_label = step_label(obj_enum)
    app_state.logger.log_keyframe(marker, obj_label)
    app_state.auto_demo_logging_active = False
    app_state.world.recording_active = False
    app_state.world.attempt_status = "NORMAL"
    app_state.current_auto_step = None
    app_state.current_auto_action_count = 0
    if app_state.log_path:
        valid_blocks = prune_log_file(app_state.log_path, delete_if_empty=False)
        if int(valid_blocks) <= 0:
            log_line(
                "[SESSION] No completed attempts found in the current auto log; "
                f"kept file at {app_state.log_path}."
            )


def prompt_line(prompt):
    fd_term = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd_term)
    try:
        attr = termios.tcgetattr(fd_term)
        attr[3] |= termios.ECHO | termios.ICANON
        termios.tcsetattr(fd_term, termios.TCSANOW, attr)
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
    finally:
        termios.tcsetattr(fd_term, termios.TCSANOW, old_attr)
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


def _parse_hotkey_var_triplet(raw):
    text = str(raw or "").strip()
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        pwm = int(round(float(parts[0])))
        power = float(parts[1])
        duration_ms = int(round(float(parts[2])))
    except (TypeError, ValueError):
        return None
    return pwm, power, duration_ms


def _manual_hotkey_duration_ms(app_state, cmd, duration_ms):
    try:
        base_ms = int(round(float(duration_ms)))
    except (TypeError, ValueError):
        base_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 300) or 300)
    base_ms = max(1, base_ms)
    if cmd not in ("l", "r"):
        return base_ms
    robot = getattr(app_state, "robot", None)
    last_turn_cmd = getattr(robot, "_last_turn_cmd", None) if robot is not None else None
    if last_turn_cmd != cmd:
        return max(1, int(round(base_ms * 0.5)))
    return base_ms


def _update_speed_model_vars_for_hotkey(cmd, score, pwm, power, duration_ms, hotkey=None):
    hotkey_norm = None
    if isinstance(hotkey, str) and str(hotkey).strip():
        hotkey_norm = str(hotkey).strip().lower()
    elif isinstance(HOTKEY_SPEED_SCORES, dict):
        cmd_norm = str(cmd or "").strip().lower()
        try:
            score_norm = int(telemetry_robot_module.normalize_speed_score(score))
        except Exception:
            score_norm = None
        for key, row in HOTKEY_SPEED_SCORES.items():
            if not isinstance(row, dict):
                continue
            row_cmd = str(row.get("cmd", "")).strip().lower()
            try:
                row_score = int(row.get("score"))
            except (TypeError, ValueError):
                continue
            if row_cmd == cmd_norm and score_norm is not None and row_score == score_norm:
                hotkey_norm = str(key).strip().lower()
                break

    score_key = telemetry_robot_module.normalize_speed_score(score)
    pwm_clamped = telemetry_robot_module.clamp_pwm(pwm)
    try:
        power_clamped = max(0.0, min(1.0, float(power)))
    except (TypeError, ValueError):
        power_clamped = telemetry_robot_module.pwm_to_power(pwm_clamped) or 0.0
    duration_clamped = max(1, int(round(float(duration_ms))))

    def _apply_hotkey_override_to_memory():
        if not hotkey_norm or not isinstance(HOTKEY_SPEED_SCORES, dict):
            return
        existing_hotkey = HOTKEY_SPEED_SCORES.get(hotkey_norm)
        if not isinstance(existing_hotkey, dict):
            existing_hotkey = {}
        HOTKEY_SPEED_SCORES[hotkey_norm] = {
            **existing_hotkey,
            "cmd": str(cmd),
            "score": int(score_key),
            "pwm": int(pwm_clamped),
            "power": float(power_clamped),
            "duration_ms": int(duration_clamped),
        }

    def _apply_hotkey_override_to_model(model):
        if not hotkey_norm or not isinstance(model, dict):
            return
        hotkeys_map = model.get("hotkey_speed_scores")
        if not isinstance(hotkeys_map, dict):
            hotkeys_map = {}
            model["hotkey_speed_scores"] = hotkeys_map
        existing_row = hotkeys_map.get(hotkey_norm)
        if not isinstance(existing_row, dict):
            existing_row = {}
        hotkeys_map[hotkey_norm] = {
            **existing_row,
            "cmd": str(cmd),
            "score": int(score_key),
            "pwm": int(pwm_clamped),
            "power": float(power_clamped),
            "duration_ms": int(duration_clamped),
        }

    # Lift/mast hotkeys are calibrated per-key and must not mutate shared drive
    # score maps used by auto alignment (f/b).
    if cmd in ("u", "d") and hotkey_norm:
        _apply_hotkey_override_to_memory()

        model_path = getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", None)
        if not isinstance(model_path, Path):
            return False, "Robot model path unavailable."
        try:
            if model_path.exists():
                try:
                    model = json.loads(model_path.read_text())
                except (OSError, json.JSONDecodeError):
                    model = {}
            else:
                model = {}
            if not isinstance(model, dict):
                model = {}
            _apply_hotkey_override_to_model(model)
            model_path.write_text(json.dumps(model, indent=2) + "\n")
        except OSError as exc:
            return False, f"Failed to persist world model: {exc}"

        return True, None

    if cmd == "l":
        score_map = getattr(telemetry_robot_module, "SCORE_POWER_PWM_TURN_LEFT", None)
        if not isinstance(score_map, dict):
            score_map = telemetry_robot_module.SCORE_POWER_PWM_TURN
        map_key = getattr(telemetry_robot_module, "SPEED_MAP_KEY_TURN_LEFT", "score_power_pwm_turn_left")
        duration_key = getattr(telemetry_robot_module, "SPEED_SECONDS_KEY_TURN_LEFT", "speed_score_seconds_turn_left")
    elif cmd == "r":
        score_map = getattr(telemetry_robot_module, "SCORE_POWER_PWM_TURN_RIGHT", None)
        if not isinstance(score_map, dict):
            score_map = telemetry_robot_module.SCORE_POWER_PWM_TURN
        map_key = getattr(telemetry_robot_module, "SPEED_MAP_KEY_TURN_RIGHT", "score_power_pwm_turn_right")
        duration_key = getattr(telemetry_robot_module, "SPEED_SECONDS_KEY_TURN_RIGHT", "speed_score_seconds_turn_right")
    elif cmd in ("l", "r"):
        score_map = telemetry_robot_module.SCORE_POWER_PWM_TURN
        map_key = getattr(telemetry_robot_module, "SPEED_MAP_KEY_TURN", "score_power_pwm_turn")
        duration_key = getattr(telemetry_robot_module, "SPEED_SECONDS_KEY_TURN", "speed_score_seconds_turn")
    else:
        score_map = telemetry_robot_module.SCORE_POWER_PWM_DRIVE
        map_key = getattr(telemetry_robot_module, "SPEED_MAP_KEY_DRIVE", "score_power_pwm_drive")
        duration_key = getattr(telemetry_robot_module, "SPEED_SECONDS_KEY", "speed_score_seconds")

    if not isinstance(score_map, dict):
        return False, "Speed map unavailable."

    entry = score_map.get(score_key)
    if not isinstance(entry, dict):
        entry = {}
        score_map[score_key] = entry
    entry["pwm"] = int(pwm_clamped)
    entry["power"] = float(power_clamped)
    entry["duration_ms"] = int(duration_clamped)

    duration_map = getattr(telemetry_robot_module, "SPEED_SCORE_DURATION_MS", None)
    if isinstance(duration_map, dict):
        duration_map[score_key] = int(duration_clamped)
    if score_key == int(getattr(telemetry_robot_module, "SPEED_SCORE_DEFAULT", 50)):
        telemetry_robot_module.ACT_DURATION_MS = int(duration_clamped)

    try:
        telemetry_robot_module.refresh_speed_model_baseline()
    except Exception:
        pass

    model_path = getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", None)
    if not isinstance(model_path, Path):
        return False, "Robot model path unavailable."

    try:
        if model_path.exists():
            try:
                model = json.loads(model_path.read_text())
            except (OSError, json.JSONDecodeError):
                model = {}
        else:
            model = {}
        if not isinstance(model, dict):
            model = {}

        section = model.get(map_key)
        if not isinstance(section, dict):
            section = {}
            model[map_key] = section
        score_text = str(int(score_key))
        section_entry = section.get(score_text)
        if not isinstance(section_entry, dict):
            section_entry = {}
            section[score_text] = section_entry
        section_entry["pwm"] = int(pwm_clamped)
        section_entry["power"] = float(power_clamped)

        speed_seconds = model.get(duration_key)
        if not isinstance(speed_seconds, dict):
            speed_seconds = {}
            model[duration_key] = speed_seconds
        speed_seconds[score_text] = round(float(duration_clamped) / 1000.0, 3)

        # Also persist the explicit hotkey row that was edited (if any). This is
        # required for custom/manual hotkeys that use per-key PWM/duration overrides.
        _apply_hotkey_override_to_memory()
        _apply_hotkey_override_to_model(model)

        model_path.write_text(json.dumps(model, indent=2) + "\n")
    except OSError as exc:
        return False, f"Failed to persist world model: {exc}"

    return True, None


def maybe_prompt_hotkey_var_update(app_state, cmd, score, *, enter_edit=False, hotkey=None):
    try:
        _, pwm, score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
        power = telemetry_robot_module.pwm_to_power(pwm)
    except Exception as exc:
        log_line(f"[HOTKEY] Failed to read speed vars: {exc}")
        return None

    if power is None:
        power = 0.0
    duration_used_ms = _manual_hotkey_duration_ms(app_state, cmd, duration_model_ms)

    hotkey_target = None
    hotkey_explicit = str(hotkey or "").strip().lower()
    if hotkey_explicit:
        hotkey_target = hotkey_explicit
    target = app_state.hotkey_edit_target if isinstance(app_state.hotkey_edit_target, dict) else None
    if hotkey_target is None and isinstance(target, dict):
        target_cmd = str(target.get("cmd", "")).strip().lower()
        try:
            target_score = int(target.get("score"))
        except (TypeError, ValueError):
            target_score = None
        target_hotkey = str(target.get("hotkey", "")).strip().lower()
        if target_hotkey and target_cmd == str(cmd).strip().lower() and target_score == int(score_used):
            hotkey_target = target_hotkey
    if hotkey_target is None and isinstance(HOTKEY_SPEED_SCORES, dict):
        for key, row in HOTKEY_SPEED_SCORES.items():
            if not isinstance(row, dict):
                continue
            row_cmd = str(row.get("cmd", "")).strip().lower()
            try:
                row_score = int(row.get("score"))
            except (TypeError, ValueError):
                continue
            if row_cmd == str(cmd).strip().lower() and row_score == int(score_used):
                hotkey_target = str(key).strip().lower()
                break

    # If a specific hotkey row exists, show/edit the values that manual hotkeys
    # actually use (including per-hotkey pwm/duration overrides), not only the
    # generic cmd+score model values.
    if hotkey_target and isinstance(HOTKEY_SPEED_SCORES, dict):
        hotkey_row = HOTKEY_SPEED_SCORES.get(hotkey_target)
        if isinstance(hotkey_row, dict):
            row_cmd = str(hotkey_row.get("cmd", "")).strip().lower()
            try:
                row_score = int(hotkey_row.get("score"))
            except (TypeError, ValueError):
                row_score = None
            if row_cmd == str(cmd).strip().lower() and row_score == int(score_used):
                try:
                    pwm_row = int(round(float(hotkey_row.get("pwm"))))
                except (TypeError, ValueError):
                    pwm_row = None
                if pwm_row is not None and pwm_row > 0:
                    pwm = int(telemetry_robot_module.clamp_pwm(pwm_row))
                    power_row = hotkey_row.get("power")
                    try:
                        power = max(0.0, min(1.0, float(power_row)))
                    except (TypeError, ValueError):
                        power_from_pwm = telemetry_robot_module.pwm_to_power(pwm)
                        power = float(power_from_pwm) if power_from_pwm is not None else float(power or 0.0)
                try:
                    duration_row = int(round(float(hotkey_row.get("duration_ms"))))
                except (TypeError, ValueError):
                    duration_row = None
                if duration_row is not None and duration_row > 0:
                    duration_used_ms = int(duration_row)

    if not enter_edit:
        preview_text = _command_preview_text(
            cmd,
            speed_score=score_used,
            pwm=pwm,
            power=power,
            duration_ms=duration_used_ms,
        ) or str(cmd).strip().lower()
        log_line(
            f"[HOTKEY] {preview_text} "
            f"- press y to edit vars"
        )
        return int(score_used)

    preview_text = _command_preview_text(
        cmd,
        speed_score=score_used,
        pwm=pwm,
        power=power,
        duration_ms=duration_used_ms,
    ) or str(cmd).strip().lower()
    log_line(
        f"[HOTKEY] Editing {preview_text}"
    )

    while app_state.running:
        raw = prompt_line(
            "[HOTKEY] Type 3 numbers in this exact order: pwm power duration_ms: "
        )
        parsed = _parse_hotkey_var_triplet(raw)
        if parsed is None:
            log_line("[HOTKEY] Invalid input. Enter exactly 3 numbers: pwm power duration_ms")
            continue
        pwm_new, power_new, duration_new = parsed
        ok, err = _update_speed_model_vars_for_hotkey(
            cmd,
            score_used,
            pwm_new,
            power_new,
            duration_new,
            hotkey=hotkey_target,
        )
        if not ok:
            log_line(f"[HOTKEY] Update failed: {err}")
            return
        _record_manual_hotkey_edit(
            app_state,
            hotkey=hotkey_target,
            cmd=cmd,
            score=score_used,
        )
        log_line(
            f"[HOTKEY] Updated world model for {str(cmd).upper()} {int(score_used)}% "
            f"-> (pwm={int(pwm_new)}, power={float(power_new):.3f}, {int(duration_new)}ms)"
        )
        return int(score_used)
    return int(score_used)


def format_step_codes_line():
    parts = [format_step_code_entry(idx, obj, include_emoji=True) for idx, obj in enumerate(DEMO_STEPS)]
    return "[CMD] Step codes: " + ", ".join(parts)


def format_step_code_entry(idx, obj_enum, *, include_emoji=False):
    label = str(getattr(obj_enum, "value", obj_enum)).lower()
    prefix = ""
    if include_emoji:
        emoji = STEP_CODE_EMOJIS.get(label)
        if emoji:
            prefix = str(emoji)
    return f"{prefix}{int(idx) + 1}={label}"


def step_code_for_obj(obj_enum):
    for code, obj in STEP_CODES.items():
        if obj == obj_enum:
            return code
    return normalize_step_label(getattr(obj_enum, "value", obj_enum))


def log_step_codes_list(prefix="[CMD]"):
    log_line(f"{prefix} Step codes:")
    for idx, obj in enumerate(DEMO_STEPS):
        log_line(f"{prefix}   {idx + 1}={str(obj.value).lower()}")


def next_demo_step_obj(obj_enum):
    if obj_enum is None:
        return None
    try:
        idx = DEMO_STEPS.index(obj_enum)
    except ValueError:
        return None
    total_steps = int(len(DEMO_STEPS))
    if total_steps <= 0:
        return None
    next_idx = (int(idx) + 1) % int(total_steps)
    return DEMO_STEPS[int(next_idx)]


def _auto_continue_pending_locked(app_state):
    deadline = float(getattr(app_state, "auto_continue_deadline", 0.0) or 0.0)
    target = getattr(app_state, "auto_continue_target_step", None)
    return deadline > 0.0 and target is not None


def _clear_auto_continue_locked(app_state):
    app_state.auto_continue_deadline = 0.0
    app_state.auto_continue_source_step = None
    app_state.auto_continue_target_step = None
    app_state.auto_continue_last_countdown_second = None


def _schedule_auto_continue_after_success_locked(app_state, completed_step):
    _clear_auto_continue_locked(app_state)
    wait_s = max(0.0, float(AUTO_CONTINUE_WAIT_S))
    if wait_s <= 0.0:
        return None
    next_obj = next_demo_step_obj(completed_step)
    if next_obj is None:
        return None
    app_state.auto_continue_deadline = float(time.time()) + float(wait_s)
    app_state.auto_continue_source_step = completed_step
    app_state.auto_continue_target_step = next_obj
    app_state.auto_continue_last_countdown_second = None
    return next_obj, wait_s


def _pop_due_auto_continue_locked(app_state, *, now_s=None):
    if bool(getattr(app_state, "auto_running", False)):
        return None
    if getattr(app_state, "auto_request", None) is not None:
        return None
    if not _auto_continue_pending_locked(app_state):
        return None
    now_val = float(time.time()) if now_s is None else float(now_s)
    deadline = float(getattr(app_state, "auto_continue_deadline", 0.0) or 0.0)
    if now_val < deadline:
        return None
    next_obj = getattr(app_state, "auto_continue_target_step", None)
    _clear_auto_continue_locked(app_state)
    return next_obj


def _auto_continue_countdown_message_locked(app_state, *, now_s=None):
    if not _auto_continue_pending_locked(app_state):
        return None
    if bool(getattr(app_state, "auto_running", False)):
        return None
    if getattr(app_state, "auto_request", None) is not None:
        return None

    now_val = float(time.time()) if now_s is None else float(now_s)
    deadline = float(getattr(app_state, "auto_continue_deadline", 0.0) or 0.0)
    remaining_s = float(deadline) - float(now_val)
    if remaining_s <= 0.0:
        return None

    remaining_sec = max(1, int(math.ceil(remaining_s)))
    last_sec = getattr(app_state, "auto_continue_last_countdown_second", None)
    if last_sec == remaining_sec:
        return None
    app_state.auto_continue_last_countdown_second = int(remaining_sec)

    next_obj = getattr(app_state, "auto_continue_target_step", None)
    next_code = step_code_for_obj(next_obj) if next_obj is not None else "?"
    next_name = str(getattr(next_obj, "value", next_obj or "unknown")).lower()
    esc_highlight = f"{ANSI_ORANGE_BRIGHT}escape{ANSI_RESET}"
    return (
        f"[AUTO] Auto-continue in {int(remaining_sec)}s: next step {next_code} ({next_name}). "
        f"Press Enter to run now, or {esc_highlight} to cancel."
    )


def _split_footer_sentences(line, prefix):
    text = str(line or "").strip()
    if not text:
        return []
    if prefix and text.startswith(prefix):
        text = text[len(prefix):].strip()
    parts = []
    for part in text.split(". "):
        part = part.strip()
        if not part:
            continue
        if not part.endswith("."):
            part += "."
        parts.append(part)
    return parts


def _footer_section_html(title, lines):
    rows = []
    for line in lines:
        line_text = str(line or "").strip()
        if not line_text:
            continue
        rows.append(f"<div class='footer-line'>{line_text}</div>")
    if not rows:
        return ""
    return (
        "<div class='footer-section'>"
        f"<div class='footer-title'>{title}</div>"
        + "".join(rows)
        + "</div>"
    )


def stream_footer_html():
    sections = []
    keys_line = format_hotkey_speeds()
    if keys_line:
        sections.append(
            _footer_section_html(
                "Hotkey Speeds",
                _split_footer_sentences(keys_line, "[KEYS]"),
            )
        )
    sections.append(
        _footer_section_html(
            "Step Codes",
            [format_step_code_entry(idx, obj, include_emoji=True) for idx, obj in enumerate(DEMO_STEPS)],
        )
    )
    sections = [section for section in sections if section]
    if not sections:
        return ""
    return "<div class='footer-sections'>" + "".join(sections) + "</div>"


def _open_stream_in_chrome(url):
    if not url:
        return False
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        chrome = shutil.which(candidate)
        if chrome:
            subprocess.Popen(
                [chrome, "--new-window", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    return False


def _stats_inc(counter, key, delta=1):
    if not isinstance(counter, dict):
        return
    key_text = str(key or "").strip()
    if not key_text:
        return
    counter[key_text] = int(counter.get(key_text, 0) or 0) + int(delta)


def _stats_step_key(step):
    key = normalize_step_label(step)
    if key:
        return str(key)
    return str(step or "UNKNOWN").strip().upper() or "UNKNOWN"


def _record_step_completion_actions(app_state, step, actions_required, *, source):
    try:
        actions_int = int(actions_required)
    except (TypeError, ValueError):
        return
    if actions_int < 0:
        return
    step_key = _stats_step_key(step)
    now_ts = float(time.time())
    with app_state.lock:
        by_step = getattr(app_state, "session_step_actions_by_step", None)
        if not isinstance(by_step, dict):
            by_step = {}
            app_state.session_step_actions_by_step = by_step
        bucket = by_step.get(step_key)
        if not isinstance(bucket, list):
            bucket = []
            by_step[step_key] = bucket
        bucket.append(int(actions_int))
        records = getattr(app_state, "session_step_action_records", None)
        if not isinstance(records, list):
            records = []
            app_state.session_step_action_records = records
        records.append(
            {
                "timestamp": round(now_ts, 3),
                "step": step_key,
                "actions_required": int(actions_int),
                "source": str(source or "unknown").strip().lower() or "unknown",
            }
        )


def _record_auto_step_cancel(app_state, step):
    step_key = _stats_step_key(step)
    with app_state.lock:
        app_state.session_auto_step_cancel_count = int(
            getattr(app_state, "session_auto_step_cancel_count", 0) or 0
        ) + 1
        by_step = getattr(app_state, "session_auto_step_cancel_by_step", None)
        if not isinstance(by_step, dict):
            by_step = {}
            app_state.session_auto_step_cancel_by_step = by_step
        _stats_inc(by_step, step_key, 1)


def _record_auto_step_restart(app_state, step):
    step_key = _stats_step_key(step)
    with app_state.lock:
        app_state.session_auto_step_restart_count = int(
            getattr(app_state, "session_auto_step_restart_count", 0) or 0
        ) + 1
        by_step = getattr(app_state, "session_auto_step_restart_by_step", None)
        if not isinstance(by_step, dict):
            by_step = {}
            app_state.session_auto_step_restart_by_step = by_step
        _stats_inc(by_step, step_key, 1)


def _record_manual_step_restart(app_state, step):
    step_key = _stats_step_key(step)
    with app_state.lock:
        app_state.session_manual_step_restart_count = int(
            getattr(app_state, "session_manual_step_restart_count", 0) or 0
        ) + 1
        by_step = getattr(app_state, "session_manual_step_restart_by_step", None)
        if not isinstance(by_step, dict):
            by_step = {}
            app_state.session_manual_step_restart_by_step = by_step
        _stats_inc(by_step, step_key, 1)


def _record_manual_hotkey_press(app_state, *, hotkey, cmd=None, score=None):
    hotkey_key = str(hotkey or "").strip().lower()
    if not hotkey_key:
        return
    cmd_key = str(cmd or "").strip().lower()
    with app_state.lock:
        by_hotkey = getattr(app_state, "session_manual_hotkey_press_by_hotkey", None)
        if not isinstance(by_hotkey, dict):
            by_hotkey = {}
            app_state.session_manual_hotkey_press_by_hotkey = by_hotkey
        _stats_inc(by_hotkey, hotkey_key, 1)
        app_state.session_manual_hotkey_press_count = int(
            getattr(app_state, "session_manual_hotkey_press_count", 0) or 0
        ) + 1

        by_cmd = getattr(app_state, "session_manual_hotkey_press_by_command", None)
        if not isinstance(by_cmd, dict):
            by_cmd = {}
            app_state.session_manual_hotkey_press_by_command = by_cmd
        if cmd_key:
            _stats_inc(by_cmd, cmd_key, 1)

        details = getattr(app_state, "session_manual_hotkey_press_details", None)
        if not isinstance(details, list):
            details = []
            app_state.session_manual_hotkey_press_details = details
        details.append(
            {
                "timestamp": round(time.time(), 3),
                "hotkey": hotkey_key,
                "command": cmd_key or None,
                "score": None if score is None else int(score),
            }
        )


def _record_manual_hotkey_edit(app_state, *, hotkey, cmd=None, score=None):
    hotkey_key = str(hotkey or "").strip().lower()
    if not hotkey_key:
        return
    cmd_key = str(cmd or "").strip().lower()
    with app_state.lock:
        by_hotkey = getattr(app_state, "session_manual_hotkey_edit_by_hotkey", None)
        if not isinstance(by_hotkey, dict):
            by_hotkey = {}
            app_state.session_manual_hotkey_edit_by_hotkey = by_hotkey
        _stats_inc(by_hotkey, hotkey_key, 1)
        app_state.session_manual_hotkey_edit_count = int(
            getattr(app_state, "session_manual_hotkey_edit_count", 0) or 0
        ) + 1

        details = getattr(app_state, "session_manual_hotkey_edit_details", None)
        if not isinstance(details, list):
            details = []
            app_state.session_manual_hotkey_edit_details = details
        details.append(
            {
                "timestamp": round(time.time(), 3),
                "hotkey": hotkey_key,
                "command": cmd_key or None,
                "score": None if score is None else int(score),
            }
        )


def _session_summary_payload(app_state):
    with app_state.lock:
        started_ts = float(getattr(app_state, "session_started_ts", time.time()) or time.time())
        summary_generated_ts = float(time.time())
        by_step_raw = getattr(app_state, "session_step_actions_by_step", {}) or {}
        by_step = {}
        if isinstance(by_step_raw, dict):
            by_step = {str(k): list(v) for k, v in by_step_raw.items() if isinstance(v, list)}
        auto_cancel_by_step = dict(getattr(app_state, "session_auto_step_cancel_by_step", {}) or {})
        auto_restart_by_step = dict(getattr(app_state, "session_auto_step_restart_by_step", {}) or {})
        manual_restart_by_step = dict(getattr(app_state, "session_manual_step_restart_by_step", {}) or {})
        hotkey_press_by_hotkey = dict(
            getattr(app_state, "session_manual_hotkey_press_by_hotkey", {}) or {}
        )
        hotkey_press_by_command = dict(
            getattr(app_state, "session_manual_hotkey_press_by_command", {}) or {}
        )
        hotkey_edit_by_hotkey = dict(
            getattr(app_state, "session_manual_hotkey_edit_by_hotkey", {}) or {}
        )
        run_logs_raw = list(getattr(app_state, "session_log_paths", []) or [])
        run_logs = []
        for p in run_logs_raw:
            path = Path(p)
            if path.exists():
                run_logs.append(str(path))
        action_records = list(getattr(app_state, "session_step_action_records", []) or [])
        hotkey_press_details = list(getattr(app_state, "session_manual_hotkey_press_details", []) or [])
        hotkey_edit_details = list(getattr(app_state, "session_manual_hotkey_edit_details", []) or [])
        active_vision_mode = _stream_state_vision_mode(
            app_state,
            _DEFAULT_VISION_MODE,
        )

    completion_stats = {}
    for step_key in sorted(by_step.keys()):
        samples = [int(x) for x in by_step.get(step_key, [])]
        if not samples:
            continue
        avg = float(statistics.mean(samples))
        if len(samples) >= 2:
            stddev = float(statistics.pstdev(samples))
        else:
            stddev = 0.0
        completion_stats[str(step_key)] = {
            "count": int(len(samples)),
            "average_actions": round(avg, 3),
            "stddev_actions": round(stddev, 3),
            "samples": samples,
        }

    payload = {
        "session": {
            "started_at_ts": round(started_ts, 3),
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_ts)),
            "ended_at_ts": round(summary_generated_ts, 3),
            "ended_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary_generated_ts)),
            "duration_s": round(max(0.0, summary_generated_ts - started_ts), 3),
            "active_vision_mode": str(active_vision_mode),
        },
        "run_logs": run_logs,
        "step_completion_actions": {
            "by_step": completion_stats,
            "records": action_records,
        },
        "counts": {
            "cancelled_steps_total": int(sum(int(v) for v in auto_cancel_by_step.values())),
            "cancelled_steps_by_step": auto_cancel_by_step,
            "restarted_steps_total": int(
                sum(int(v) for v in auto_restart_by_step.values())
                + sum(int(v) for v in manual_restart_by_step.values())
            ),
            "restarted_steps_auto_full_gatecheck_by_step": auto_restart_by_step,
            "restarted_steps_manual_by_step": manual_restart_by_step,
        },
        "manual_hotkeys": {
            "press_count_total": int(sum(int(v) for v in hotkey_press_by_hotkey.values())),
            "press_count_by_hotkey": hotkey_press_by_hotkey,
            "press_count_by_command": hotkey_press_by_command,
            "press_details": hotkey_press_details,
            "edit_count_total": int(sum(int(v) for v in hotkey_edit_by_hotkey.values())),
            "edit_count_by_hotkey": hotkey_edit_by_hotkey,
            "edit_details": hotkey_edit_details,
        },
    }
    return payload


def _pretty_run_timestamp(ts=None):
    local_ts = float(time.time() if ts is None else ts)
    local = time.localtime(local_ts)
    month_day = f"{time.strftime('%b', local)} {int(local.tm_mday)}"
    hhmm_ampm = time.strftime("%I%M%p", local).lstrip("0").lower()
    return f"{month_day} {hhmm_ampm}"


def _unique_json_path(folder, stem):
    target = folder / f"{stem}.json"
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = folder / f"{stem} ({index}).json"
        if not candidate.exists():
            return candidate
        index += 1





def log_session_summary(app_state):
    payload = _session_summary_payload(app_state)
    completion_by_step = (
        ((payload.get("step_completion_actions") or {}).get("by_step"))
        if isinstance(payload, dict)
        else {}
    )
    counts = (payload.get("counts") or {}) if isinstance(payload, dict) else {}
    hotkeys = (payload.get("manual_hotkeys") or {}) if isinstance(payload, dict) else {}
    log_line("[SESSION SUMMARY] Completion actions by step (avg±stddev, n):")
    if isinstance(completion_by_step, dict) and completion_by_step:
        for step_key in sorted(completion_by_step.keys()):
            row = completion_by_step.get(step_key) or {}
            avg = row.get("average_actions")
            std = row.get("stddev_actions")
            count = row.get("count")
            log_line(
                f"[SESSION SUMMARY] {step_key}: {avg} ± {std} (n={count})"
            )
    else:
        log_line("[SESSION SUMMARY] No completed step samples recorded.")

    log_line(
        "[SESSION SUMMARY] Cancelled steps: "
        f"{int((counts or {}).get('cancelled_steps_total', 0) or 0)}"
    )
    log_line(
        "[SESSION SUMMARY] Restarted steps: "
        f"{int((counts or {}).get('restarted_steps_total', 0) or 0)}"
    )
    log_line(
        "[SESSION SUMMARY] Manual hotkey presses: "
        f"{int((hotkeys or {}).get('press_count_total', 0) or 0)}"
    )
    log_line(
        "[SESSION SUMMARY] Manual hotkey edits: "
        f"{int((hotkeys or {}).get('edit_count_total', 0) or 0)}"
    )

def open_new_log(app_state):
    runs_dir = Path(
        getattr(
            app_state,
            "runs_dir",
            getattr(app_state, "demos_dir", Path(__file__).resolve().parent),
        )
    )
    _cleanup_stale_run_logs()
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = _unique_json_path(runs_dir, _pretty_run_timestamp())
    log_line(f"[SESSION] Recording Run Log to: {log_path}")
    app_state.logger = TelemetryLogger(log_path)
    app_state.logger.enabled = False  # Wait for first attempt marker
    app_state.logger_closed = False
    app_state.log_path = log_path
    session_logs = getattr(app_state, "session_log_paths", None)
    if isinstance(session_logs, list):
        session_logs.append(Path(log_path))
    app_state.world.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    app_state.world.attempt_id = 1

def ensure_log_open(app_state):
    if app_state.logger is None or app_state.logger_closed:
        open_new_log(app_state)

def close_log(app_state, marker=None):
    if app_state.logger is None or app_state.logger_closed:
        return
    if marker:
        app_state.logger.log_keyframe(marker)
    app_state.logger.enabled = False
    app_state.logger.close()
    app_state.logger_closed = True
    if app_state.log_path:
        valid_blocks = prune_log_file(app_state.log_path, delete_if_empty=False)
        if int(valid_blocks) <= 0:
            log_line(
                "[SESSION] No completed attempts found in the current log; "
                f"kept file at {app_state.log_path}."
            )


def _set_active_demos_dir_for_mode(app_state, mode):
    mode_norm = normalize_vision_mode(mode, fallback=_DEFAULT_VISION_MODE)
    next_dir = _demos_dir_for_vision_mode(mode_norm)
    next_runs_dir = _runs_dir_for_vision_mode(mode_norm)
    next_dir.mkdir(parents=True, exist_ok=True)
    next_runs_dir.mkdir(parents=True, exist_ok=True)
    current_dir = Path(getattr(app_state, "demos_dir", next_dir))
    current_runs_dir = Path(getattr(app_state, "runs_dir", next_runs_dir))
    try:
        unchanged = (
            current_dir.resolve() == next_dir.resolve()
            and current_runs_dir.resolve() == next_runs_dir.resolve()
        )
    except Exception:
        unchanged = (
            str(current_dir) == str(next_dir)
            and str(current_runs_dir) == str(next_runs_dir)
        )
    if unchanged:
        return False
    close_log(app_state, marker=None)
    app_state.demos_dir = next_dir
    app_state.runs_dir = next_runs_dir
    app_state.config_mtime = 0
    app_state.last_config_check = 0
    open_new_log(app_state)
    log_line(
        f"[DEMOS] Active folder for {_vision_mode_label(mode_norm)}: {next_dir}"
    )
    log_line(
        f"[RUNS] Active folder for {_vision_mode_label(mode_norm)}: {next_runs_dir}"
    )
    return True


def trash_current_session(app_state):
    log_path = app_state.log_path
    app_state.active_attempt = None
    app_state.step_open = False
    app_state.open_step = None
    app_state.world.recording_active = False
    app_state.world.attempt_status = "NORMAL"
    if app_state.logger is not None and not app_state.logger_closed:
        app_state.logger.enabled = False
        app_state.logger.close()
    app_state.logger_closed = True
    app_state.log_path = None
    session_logs = getattr(app_state, "session_log_paths", None)
    if isinstance(session_logs, list) and log_path is not None:
        try:
            log_resolved = Path(log_path).resolve()
            session_logs[:] = [p for p in session_logs if Path(p).resolve() != log_resolved]
        except Exception:
            session_logs[:] = [p for p in session_logs if Path(p) != Path(log_path)]
    if log_path and log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            return log_path, False
    return log_path, True

def run_auto_step(app_state, obj_enum):
    step_key = normalize_step_label(obj_enum.value)
    _set_stream_state_success_gate_step(app_state, step_key)
    print("\n" * 5, end="")
    log_line(f"[AUTO] Attempting {step_key}...")
    reset_repeat_act_guard(app_state.world, app_state.robot)

    logs = load_demo_logs(app_state.demos_dir)
    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
    refresh_autobuild_config(PROCESS_MODEL_FILE)
    if not logs:
        log_line("[AUTO] No demo logs found for auto mode.")
        return False

    segments_by_obj, _ = collect_segments(logs)
    model = load_process_model(PROCESS_MODEL_FILE)
    process_rules = model.get("steps") if isinstance(model, dict) else {}
    if not isinstance(process_rules, dict):
        process_rules = {}

    app_state.world.process_rules = process_rules
    app_state.world.rules = process_rules

    app_state.world.step_state = obj_enum
    log_line(height_intel_step_begin_line(app_state.world, step_key))
    auto_demo_logging_started = False
    app_state.auto_demo_logging_active = False

    cfg = process_rules.get(step_key, {}) if process_rules else {}
    nominal_only = bool(cfg.get("nominalDemosOnly"))
    segment, seg_type = select_demo_segment(segments_by_obj, step_key, nominal_only)
    if not segment:
        log_line(f"[AUTO] No demo segment found for {step_key}.")
        return False
    _begin_auto_demo_logging(app_state, obj_enum)
    auto_demo_logging_started = True

    events = segment.get("events") or []
    actions = nominal_actions_from_events(events)
    chassis_actions = [
        action
        for action in actions
        if str((action or {}).get("cmd") or "").strip().lower() in ("f", "b", "l", "r")
    ]
    steps = merge_motion_steps(build_motion_sequence(events))

    try:
        import telemetry_wall as telemetry_wall_module
    except Exception:
        telemetry_wall_module = None

    start_desc, success_desc = format_gate_lines(cfg)
    skip_start_gates = not step_requires_start_gates(step_key, process_rules)

    def _gate_ok(check):
        return bool(getattr(check, "ok", False))

    def _gate_reasons(check):
        raw = getattr(check, "reasons", None)
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(r) for r in raw if r]
        return [str(raw)]

    def _auto_start_gate_status_message():
        if skip_start_gates:
            status_msg_local = f"[AUTO] {step_key} start gates: SKIPPED (find/exit step)"
            if start_desc and start_desc != "none":
                status_msg_local += f" (cfg ignored: {start_desc})"
            return True, status_msg_local
        try:
            brick_check = telemetry_brick.evaluate_start_gates(
                app_state.world,
                step_key,
                {},
                process_rules,
            )
        except Exception:
            brick_check = None
        try:
            wall_env = getattr(app_state.world, "wall_envelope", None)
            if telemetry_wall_module is not None and wall_env is not None:
                wall_check = telemetry_wall_module.evaluate_start_gates(app_state.world, step_key, wall_env)
            else:
                wall_check = None
        except Exception:
            wall_check = None
        try:
            robot_check = telemetry_robot_module.evaluate_start_gates(
                app_state.world,
                step_key,
                {},
                process_rules,
            )
        except Exception:
            robot_check = None

        # Brick start-gate evaluation is required for start eligibility.
        # Optional checkers (wall/robot) should only participate when available.
        brick_start_ok = _gate_ok(brick_check)
        optional_start_ok = all(
            _gate_ok(check)
            for check in (wall_check, robot_check)
            if check is not None
        )
        start_ok = brick_start_ok and optional_start_ok
        reason_parts = []
        brick_start_gates = (cfg.get("start_gates") if isinstance(cfg, dict) else None) or {}
        for label, check in (("brick", brick_check), ("wall", wall_check), ("robot", robot_check)):
            if check is None or _gate_ok(check):
                continue
            if label == "brick":
                brick_detail = _format_brick_start_gate_details(app_state.world, brick_start_gates)
                if brick_detail:
                    reason_parts.append(f"{label}: {brick_detail}")
                    continue
            reasons = _gate_reasons(check)
            reason_parts.append(f"{label}: {', '.join(reasons)}" if reasons else f"{label}: blocked")

        status = "OK" if start_ok else "BLOCKED"
        status_msg_local = f"[AUTO] {step_key} start gates: {status}"
        if reason_parts:
            status_msg_local += " — " + "; ".join(reason_parts)
        elif start_desc and start_desc != "none":
            status_msg_local += f" (cfg: {start_desc})"
        return start_ok, status_msg_local

    _pre_start_ok, _pre_start_status_msg = _auto_start_gate_status_message()
    if not _pre_start_ok:
        log_line(_pre_start_status_msg)

    quiet_align = False
    max_full_gatecheck_attempts_default = 4
    try:
        max_full_gatecheck_attempts_default = int(
            (_MANUAL_CONFIG or {}).get("auto_full_gatecheck_max_attempts", max_full_gatecheck_attempts_default)
        )
    except (TypeError, ValueError, AttributeError):
        max_full_gatecheck_attempts_default = 4
    try:
        max_full_gatecheck_attempts = int(
            (cfg or {}).get("full_gatecheck_max_attempts", max_full_gatecheck_attempts_default)
        )
    except (TypeError, ValueError, AttributeError):
        max_full_gatecheck_attempts = int(max_full_gatecheck_attempts_default)
    max_full_gatecheck_attempts = max(1, int(max_full_gatecheck_attempts))
    max_full_gatecheck_restarts = max(0, int(max_full_gatecheck_attempts) - 1)
    success_req = format_success_gate_requirements(app_state.world, step_key)
    if success_req == "none":
        success_req = success_desc
    log_line(f"[AUTO] {step_key} demo={seg_type} success gates: {success_req}")
    log_line(f"[AUTO] {step_key} full gatecheck attempts allowed: {int(max_full_gatecheck_attempts)}.")

    if app_state.robot:
        app_state.robot.stop()
    observer = make_auto_observer(app_state)
    try:
        app_state.auto_cancel_event.clear()
    except Exception:
        pass

    def confirm_callback(_world, _vision):
        if bool(getattr(app_state, "auto_cancel_event", None)) and app_state.auto_cancel_event.is_set():
            raise AutoStepCancelled()
        return True

    def _clear_gatecheck_runtime_state():
        world_local = app_state.world
        for attr in (
            "_gatecheck_status",
            "_last_gatecheck_progress_sig",
            "_gatecheck_mode",
            "_gatecheck_lite_passed",
        ):
            try:
                setattr(world_local, attr, None)
            except Exception:
                pass
        for attr in (
            "_gatecheck_lite_checks",
            "_gatecheck_lite_required",
            "_gatecheck_lite_collected",
        ):
            try:
                setattr(world_local, attr, 0)
            except Exception:
                pass

    def _gatecheck_progress_snapshot_text():
        status = getattr(app_state.world, "_gatecheck_status", None)
        if not isinstance(status, dict):
            return None
        status_step = normalize_step_label(status.get("step")) or str(status.get("step") or "")
        if normalize_step_label(status_step) != normalize_step_label(step_key):
            return None
        phase = str(status.get("phase") or "run").strip().lower() or "run"
        mode = str(status.get("mode") or "traditional").strip().lower()
        try:
            checks = int(status.get("checks", 0) or 0)
        except Exception:
            checks = 0
        if mode == "lite":
            try:
                lite_collected = max(0, int(status.get("lite_collected", 0) or 0))
                lite_required = max(1, int(status.get("lite_required", 1) or 1))
            except Exception:
                lite_collected, lite_required = 0, 1
            return f"{phase}: lite {lite_collected}/{lite_required} (checks={checks})"
        try:
            streak = int(status.get("streak", 0) or 0)
            need = max(1, int(status.get("need", 1) or 1))
            window_pass = int(status.get("window_pass", 0) or 0)
            window_total = max(1, int(status.get("window_total", 1) or 1))
        except Exception:
            return f"{phase}: checks={checks}"
        return (
            f"{phase}: consec {streak}/{need}, majority {window_pass}/{window_total} "
            f"(checks={checks})"
        )

    def _success_gate_snapshot_detail_text():
        try:
            detail = _result_lite_gate_detail(app_state.world, step_key)
        except Exception:
            return None
        if not isinstance(detail, dict):
            return None
        text = detail.get("colored") or detail.get("plain")
        if not text:
            return None
        return str(text)

    def _log_full_gatecheck_final(outcome, *, reason=None, attempt_num=None):
        if nominal_only:
            if int(attempt_num or 1) == 1:
                log_line(f"[GATECHECK] {step_key} final: SKIPPED (nominal step)")
            return
        reason_norm_local = str(reason or "").strip().lower()
        if "cancelled by user" in reason_norm_local:
            return
        progress_text = _gatecheck_progress_snapshot_text()
        try:
            attempt_num_int = int(attempt_num) if attempt_num is not None else None
        except (TypeError, ValueError):
            attempt_num_int = None
        attempt_suffix = f" [attempt {attempt_num_int}]" if attempt_num_int and attempt_num_int > 1 else ""
        outcome_key = str(outcome or "").strip().upper()
        if outcome_key == "PASS":
            if progress_text:
                log_line(f"[GATECHECK] {step_key} final: PASS{attempt_suffix} ({progress_text})")
            else:
                log_line(f"[GATECHECK] {step_key} final: PASS{attempt_suffix} (full gatecheck met)")
            return
        if progress_text:
            log_line(f"[GATECHECK] {step_key} final: FAIL{attempt_suffix} ({progress_text})")
            return
        if "start gates not met" in reason_norm_local:
            log_line(f"[GATECHECK] {step_key} final: SKIPPED{attempt_suffix} (start gates not met)")
            success_detail = _success_gate_snapshot_detail_text()
            if success_detail:
                log_line(f"[GATECHECK-DETAIL] {step_key} final: {success_detail}")
            return
        if reason:
            log_line(f"[GATECHECK] {step_key} final: FAIL{attempt_suffix} (reason={reason})")
        else:
            log_line(f"[GATECHECK] {step_key} final: FAIL{attempt_suffix}")

    def _is_retryable_full_gatecheck_failure(reason):
        reason_norm = str(reason or "").strip().lower()
        if not reason_norm:
            return True
        if _is_repeat_act_safety_fail(reason_norm):
            return False
        if "cancelled by user" in reason_norm or "confirm cancelled" in reason_norm:
            return False
        if "start gates not met" in reason_norm:
            return False
        if "missing speedscore" in reason_norm or "invalid speedscore" in reason_norm:
            return False
        return True

    prior_suppress_skip_start_log = bool(getattr(app_state.world, "_suppress_start_gate_skip_log", False))
    app_state.world._suppress_start_gate_skip_log = True
    reset_repeat_act_guard(app_state.world, app_state.robot)
    try:
        ok = False
        reason = "cancelled by user"
        full_gatecheck_restarts_used = 0
        attempt_num = 0
        while True:
            attempt_num += 1
            reset_auto_step_action_stats(app_state.world, step_key)
            _clear_gatecheck_runtime_state()
            try:
                ok, reason = replay_segment(
                    segment,
                    step_key,
                    app_state.robot,
                    app_state.vision,
                    app_state.world,
                    observer=observer,
                    analysis_pause_s=0.0,
                    confirm_callback=confirm_callback,
                    align_silent=quiet_align,
                )
            except AutoStepCancelled:
                ok = False
                reason = "cancelled by user"
                try:
                    if app_state.robot:
                        app_state.robot.stop()
                except Exception:
                    pass
            except RuntimeError as exc:
                if _is_repeat_act_safety_fail(exc):
                    ok = False
                    reason = str(exc) or "repeat-act safety guard"
                    try:
                        if app_state.robot:
                            app_state.robot.stop()
                    except Exception:
                        pass
                    reset_repeat_act_guard(app_state.world, app_state.robot)
                else:
                    raise

            if ok:
                _log_full_gatecheck_final("PASS", reason=reason, attempt_num=attempt_num)
                break

            _log_full_gatecheck_final("FAIL", reason=reason, attempt_num=attempt_num)
            if (
                _is_retryable_full_gatecheck_failure(reason)
                and full_gatecheck_restarts_used < int(max_full_gatecheck_restarts)
                and not (bool(getattr(app_state, "auto_cancel_event", None)) and app_state.auto_cancel_event.is_set())
            ):
                full_gatecheck_restarts_used += 1
                _record_auto_step_restart(app_state, step_key)
                log_line(
                    f"[AUTO] {step_key} full gatecheck failed; restarting "
                    f"(attempt {int(attempt_num) + 1}/{int(max_full_gatecheck_attempts)})."
                )
                continue
            break
    finally:
        app_state.world._suppress_start_gate_skip_log = prior_suppress_skip_start_log
        reset_repeat_act_guard(app_state.world, app_state.robot)
        try:
            app_state.auto_cancel_event.clear()
        except Exception:
            pass
        if auto_demo_logging_started:
            _end_auto_demo_logging(app_state, obj_enum)
    if ok:
        # Emit celebration event first so optional post-success bookkeeping
        # cannot suppress gong playback on stream clients.
        _emit_stream_step_success_event(app_state, step_key)
        try:
            apply_height_snapshot_from_step(app_state.world, step_key, log=True)
        except Exception as exc:
            log_line(f"[HEIGHT INTEL] {step_key} snapshot skipped ({exc}).")
        try:
            apply_height_inventory_adjustment_from_step(app_state.world, step_key, log=True)
        except Exception as exc:
            log_line(f"[HEIGHT INTEL] {step_key} inventory adjustment skipped ({exc}).")
        auto_continue_plan = None
        with app_state.lock:
            app_state.last_auto_completed_step = obj_enum
            auto_continue_plan = _schedule_auto_continue_after_success_locked(
                app_state,
                obj_enum,
            )
        reason_text = reason.replace("_", " ") if isinstance(reason, str) else "success criteria met"
        handoff_target = None
        if isinstance(reason, str):
            handoff_prefix = "post-ground-reset handoff to "
            reason_norm = str(reason).strip()
            if reason_norm.startswith(handoff_prefix):
                handoff_target = normalize_step_label(reason_norm[len(handoff_prefix):].strip())
        if handoff_target:
            log_line(f"[AUTO HANDOFF] {step_key} -> {handoff_target}")
        log_line(f"[AUTO] {step_key} complete — {reason_text}.")
        if auto_continue_plan:
            next_obj, wait_s = auto_continue_plan
            next_code = step_code_for_obj(next_obj)
            next_name = str(getattr(next_obj, "value", next_obj)).lower()
            log_line(
                f"[AUTO] Press Enter to queue next step now, or wait {float(wait_s):.1f}s "
                f"for auto-continue to {next_code} ({next_name}). Esc cancels."
            )
        act_stats = consume_auto_step_action_stats(app_state.world, step_key)
        acts_total = int((act_stats or {}).get("total") or 0)
        acts_closer = int((act_stats or {}).get("closer") or 0)
        acts_backward = int((act_stats or {}).get("backward") or 0)
        acts_unchanged = int((act_stats or {}).get("unchanged") or 0)
        acts_unknown = int((act_stats or {}).get("unknown") or 0)
        if acts_total <= 0 and chassis_actions:
            acts_total = int(len(chassis_actions))
            acts_closer = 0
            acts_backward = 0
            acts_unchanged = 0
            acts_unknown = int(acts_total)
        parts_sum = int(acts_closer + acts_backward + acts_unchanged + acts_unknown)
        if acts_total < parts_sum:
            acts_total = int(parts_sum)
        elif acts_total > parts_sum:
            acts_unknown += int(acts_total - parts_sum)
        if int(acts_total) <= 0:
            try:
                acts_total = int(getattr(app_state, "current_auto_action_count", 0) or 0)
            except (TypeError, ValueError):
                acts_total = 0
        _record_step_completion_actions(
            app_state,
            step_key,
            int(max(0, int(acts_total))),
            source="auto",
        )
        step_code = step_code_for_obj(obj_enum)
        log_line(f"ACHIEVED step {step_code} ({step_key}) in {acts_total} acts.")
        log_line(f"  {acts_closer} closer acts")
        log_line(f"  {acts_backward} dumb acts")
        log_line(f"  {acts_unchanged} unchanged acts")
        log_line(f"  {acts_unknown} unclear acts")
    else:
        reason_norm = str(reason or "").strip().lower()
        if reason_norm == "cancelled by user":
            _record_auto_step_cancel(app_state, step_key)
            log_line(f"[AUTO] {step_key} cancelled (Esc).")
        elif _is_repeat_act_safety_fail(reason):
            log_line(str(reason))
            log_line(f"[AUTO] {step_key} cancelled (repeat-act safety guard).")
        else:
            log_line(f"[AUTO] {step_key} failed ({reason}).")
        if str(reason or "").strip().lower() == "start gates not met":
            _start_ok_now, _start_status_msg_now = _auto_start_gate_status_message()
            if not _start_ok_now:
                log_line(_start_status_msg_now)
    return ok


def _preflight_hotkey_for_motion(*, hotkey, cmd, score):
    hotkey_norm = str(hotkey or "").strip().lower()
    if hotkey_norm in AUTO_PREFLIGHT_HOTKEY_KEYS:
        return hotkey_norm

    cmd_norm = str(cmd or "").strip().lower()
    try:
        score_norm = int(score)
    except (TypeError, ValueError):
        score_norm = None

    for key in AUTO_PREFLIGHT_HOTKEY_KEYS:
        row = HOTKEY_SPEED_SCORES.get(key)
        if not isinstance(row, dict):
            continue
        row_cmd = str(row.get("cmd", "")).strip().lower()
        try:
            row_score = int(row.get("score"))
        except (TypeError, ValueError):
            continue
        if row_cmd == cmd_norm and score_norm is not None and row_score == int(score_norm):
            return str(key)
    return None


def run_hotkey_preflight_check_if_needed(app_state, *, hotkey, cmd, score):
    if not AUTO_PREFLIGHT_HOTKEY_MOTION_ENABLED:
        return True

    target_hotkey = _preflight_hotkey_for_motion(hotkey=hotkey, cmd=cmd, score=score)
    if target_hotkey is None:
        return True

    with app_state.lock:
        if target_hotkey in app_state.preflight_checked_hotkeys:
            return True
        if target_hotkey in app_state.preflight_running_hotkeys:
            return False
        app_state.preflight_running_hotkeys.add(target_hotkey)

    try:
        cmd_text = str(cmd or "").upper()
        try:
            score_text = str(int(score))
        except (TypeError, ValueError):
            score_text = "?"
        log_line(
            f"[PREFLIGHT CHECK] {target_hotkey.upper()} -> {cmd_text} "
            f"score={score_text} starting."
        )

        if app_state.robot is None or app_state.vision is None:
            log_line(
                f"[PREFLIGHT CHECK] {target_hotkey.upper()} skipped "
                "(robot/vision unavailable)."
            )
            return True

        with app_state.vision_io_lock:
            result = run_mini_hotkey_motion_calibration(
                robot=app_state.robot,
                vision=app_state.vision,
                hotkeys=(target_hotkey,),
                log_path=AUTO_PREFLIGHT_HOTKEY_MOTION_LOG,
                log_fn=log_line,
            )

        if not bool(result.get("ok")):
            log_line(
                f"[PREFLIGHT CHECK] {target_hotkey.upper()} failed; continuing "
                f"({result.get('error')})."
            )
            return True

        selected_score = None
        hotkeys_result = result.get("hotkeys")
        if isinstance(hotkeys_result, dict):
            row = hotkeys_result.get(target_hotkey)
            if isinstance(row, dict):
                try:
                    selected_score = int(row.get("selected_score"))
                except (TypeError, ValueError):
                    selected_score = None
        if selected_score is not None:
            log_line(
                f"[PREFLIGHT CHECK] {target_hotkey.upper()} complete: "
                f"selected score {selected_score}."
            )
        else:
            log_line(f"[PREFLIGHT CHECK] {target_hotkey.upper()} complete.")
        return True
    finally:
        with app_state.lock:
            app_state.preflight_running_hotkeys.discard(target_hotkey)
            app_state.preflight_checked_hotkeys.add(target_hotkey)


def update_brick_analytics(app_state):
    steps = app_state.world.process_rules or load_process_steps()
    analytics = telemetry_brick.compute_brick_analytics(
        app_state.world,
        steps,
        app_state.world.learned_rules,
        "ALIGN_BRICK",
        duration_s=CONTROL_DT,
    )
    app_state.gate_status = analytics.get("gate_status") or []
    app_state.gate_progress = analytics.get("gate_progress") or []
    app_state.brick_highlight_metric = analytics.get("highlight_metric")
    last_progress = getattr(app_state.world, "_last_gate_progress", {})
    trend = {}
    current_progress = {}
    for name, pct in app_state.gate_progress:
        current_progress[name] = pct
        prev = last_progress.get(name)
        if prev is None:
            continue
        if pct > prev + 0.1:
            trend[name] = 1
        elif pct < prev - 0.1:
            trend[name] = -1
        else:
            trend[name] = 0
    app_state.world._last_gate_progress = current_progress
    app_state.world._gate_trend = trend
    align = analytics.get("align") or {}
    last_metrics = getattr(app_state.world, "_last_align_metrics", {})
    metrics_trend = {}
    for key in ("x_axis", "angle", "dist"):
        value = align.get(key)
        prev_val = last_metrics.get(key)
        if value is None or prev_val is None:
            continue
        if value < prev_val:
            metrics_trend[key] = 1
        elif value > prev_val:
            metrics_trend[key] = -1
        else:
            metrics_trend[key] = 0
    app_state.world._last_align_metrics = {k: align.get(k) for k in ("x_axis", "angle", "dist")}
    app_state.world._align_metrics_trend = metrics_trend
    suggestion = analytics.get("suggestion")
    if suggestion:
        app_state.step_suggestions = [("ALIGN_BRICK", suggestion)]
    else:
        app_state.step_suggestions = []


def _latest_demo_mtime(demos_dir):
    latest = 0.0
    try:
        for path in Path(demos_dir).glob("*.json"):
            try:
                latest = max(latest, float(path.stat().st_mtime))
            except OSError:
                continue
    except OSError:
        return 0.0
    return latest


def refresh_world_model_from_demos(
    app_state,
    force=False,
    min_interval_s=0.5,
    *,
    force_rederive_success_gates=False,
):
    now = time.time()
    if not force and (now - app_state.last_config_check) < min_interval_s:
        return
    app_state.last_config_check = now

    demo_mtime = _latest_demo_mtime(app_state.demos_dir)
    try:
        process_mtime = float(PROCESS_MODEL_FILE.stat().st_mtime)
    except OSError:
        process_mtime = 0.0
    latest_mtime = max(demo_mtime, process_mtime)
    if not force and latest_mtime <= app_state.config_mtime:
        return

    logs = load_demo_logs(app_state.demos_dir)
    # Preserve existing process gates when no parseable demo logs are present.
    if logs:
        update_process_model_from_demos(
            logs,
            PROCESS_MODEL_FILE,
            force_rederive_success_gates=bool(force_rederive_success_gates),
        )
    refresh_autobuild_config(PROCESS_MODEL_FILE)
    model = load_process_model(PROCESS_MODEL_FILE)
    process_rules = model.get("steps") if isinstance(model, dict) else {}
    if not isinstance(process_rules, dict):
        process_rules = {}
    app_state.world.process_rules = process_rules
    app_state.world.rules = process_rules
    align_profile = load_align_profile(Path(__file__).resolve().parent)
    app_state.world.learned_rules = inject_align_profile_into_learned_rules(
        getattr(app_state.world, "learned_rules", {}),
        align_profile,
    )
    if isinstance(align_profile, dict) and align_profile:
        run_id = align_profile.get("source_run_id")
        turn_scale = align_profile.get("turn_speed_scale")
        dist_scale = align_profile.get("dist_speed_scale")
        cap = align_profile.get("max_speed_score")
        signature = (run_id, turn_scale, dist_scale, cap)
        if signature != getattr(app_state, "_last_align_profile_signature", None):
            log_line(
                "[ALIGN_PROFILE] Applied calibrate-align profile "
                f"(run_id={run_id}, turn_scale={turn_scale}, dist_scale={dist_scale}, max_speed={cap})."
            )
            app_state._last_align_profile_signature = signature
    try:
        process_mtime = float(PROCESS_MODEL_FILE.stat().st_mtime)
    except OSError:
        process_mtime = 0.0
    app_state.config_mtime = max(demo_mtime, process_mtime)
    
    # Reload telemetry correction curves from updated world_model_robot.json
    app_state.world.reload_telemetry_correction_curves()

def refresh_brick_telemetry(app_state, read_vision=True):
    if read_vision:
        update_world_from_vision(app_state.world, app_state.vision, log=False)
    update_brick_analytics(app_state)
    update_stream_frame(app_state)






def _ordinal_list(indices):
    names = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    parts = [names.get(idx, f"{idx}th") for idx in indices]
    if not parts:
        return "none", 0
    if len(parts) == 1:
        return parts[0], 1
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}", 2
    return ", ".join(parts[:-1]) + f", and {parts[-1]}", len(parts)


def _variance_label(std_dist, std_offset, std_angle):
    if std_dist <= BRICK_STUDY_STD_LOW_MM and std_offset <= BRICK_STUDY_STD_LOW_MM and std_angle <= BRICK_STUDY_STD_LOW_DEG:
        return "low variance"
    if std_dist >= BRICK_STUDY_STD_HIGH_MM or std_offset >= BRICK_STUDY_STD_HIGH_MM or std_angle >= BRICK_STUDY_STD_HIGH_DEG:
        return "high variance"
    return "med variance"


def _average_frames(frames):
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def majority(values):
        return sum(1 for v in values if v) >= (len(values) / 2.0)

    return {
        "found": majority([f["found"] for f in frames]),
        "dist": mean([f["dist"] for f in frames]),
        "angle": mean([f["angle"] for f in frames]),
        "offset_x": mean([f["offset_x"] for f in frames]),
        "offset_y": mean([f.get("offset_y", f.get("cam_h", 0.0)) for f in frames]),
        "y_axis": mean([f.get("y_axis", f.get("offset_y", f.get("cam_h", 0.0))) for f in frames]),
        "conf": mean([f["conf"] for f in frames]),
        "cam_h": mean([f["cam_h"] for f in frames]),
        "brick_above": majority([f["brick_above"] for f in frames]),
        "brick_below": majority([f["brick_below"] for f in frames]),
    }


def _cyan_average_frames(frames):
    found_frames = [frame for frame in frames if frame.get("found")]
    if not found_frames:
        return None

    def median(values):
        return statistics.median(values) if values else 0.0

    def mean(values):
        return (sum(values) / len(values)) if values else 0.0

    def majority(values):
        return sum(1 for value in values if value) >= (len(values) / 2.0)

    return {
        "found": True,
        "dist": median([frame["dist"] for frame in found_frames]),
        "angle": median([frame["angle"] for frame in found_frames]),
        "offset_x": median([frame["offset_x"] for frame in found_frames]),
        "offset_y": median([frame.get("offset_y", frame.get("cam_h", 0.0)) for frame in found_frames]),
        "y_axis": median([frame.get("y_axis", frame.get("offset_y", frame.get("cam_h", 0.0))) for frame in found_frames]),
        "conf": mean([frame["conf"] for frame in found_frames]),
        "cam_h": mean([frame["cam_h"] for frame in found_frames]),
        "brick_above": majority([frame["brick_above"] for frame in found_frames]),
        "brick_below": majority([frame["brick_below"] for frame in found_frames]),
    }


def evaluate_cyan_frames(frames):
    if len(frames) < BRICK_STUDY_FRAMES:
        return None
    found_count = sum(1 for frame in frames if frame.get("found"))
    if found_count <= 0:
        return {
            "confidence": 0,
            "message": "0% confidence (no cyan detections in the latest 4 frames)",
            "average": None,
            "reset": True,
        }
    average = _cyan_average_frames(frames)
    confidence = int(round(100 * found_count / BRICK_STUDY_FRAMES))
    return {
        "confidence": confidence,
        "message": (
            f"{confidence}% confidence ({found_count}/{BRICK_STUDY_FRAMES} "
            "cyan frames found a brick)"
        ),
        "average": average,
        "reset": True,
    }


def _markerless_average_frames(frames):
    return _cyan_average_frames(frames)


def evaluate_markerless_frames(frames):
    return evaluate_cyan_frames(frames)


def _study_snapshot_or_raw(
    study,
    *,
    found,
    angle,
    dist,
    offset_x,
    conf,
    cam_h,
    brick_above,
    brick_below,
):
    """Build a world.update_vision snapshot without stale visibility carry-over."""
    avg = (study or {}).get("average") if isinstance(study, dict) else None
    if isinstance(avg, dict):
        return {
            # Visibility must reflect the current frame, not historical average.
            "found": bool(found),
            "dist": float(avg.get("dist", dist)),
            "angle": float(avg.get("angle", angle)),
            "offset_x": float(avg.get("offset_x", offset_x)),
            "conf": float(avg.get("conf", conf)),
            "cam_h": float(avg.get("cam_h", cam_h)),
            "brick_above": bool(avg.get("brick_above", brick_above)),
            "brick_below": bool(avg.get("brick_below", brick_below)),
        }
    return {
        "found": bool(found),
        "dist": float(dist),
        "angle": float(angle),
        "offset_x": float(offset_x),
        "conf": float(conf),
        "cam_h": float(cam_h),
        "brick_above": bool(brick_above),
        "brick_below": bool(brick_below),
    }


def evaluate_brick_frames(frames):
    if len(frames) < BRICK_STUDY_FRAMES:
        return None

    first_two = frames[:2]
    last_two = frames[2:]
    avg_first = _average_frames(first_two)
    avg_last = _average_frames(last_two)
    if (
        abs(avg_first["dist"] - avg_last["dist"]) > BRICK_STUDY_SPLIT_DIFF_MM
        or abs(avg_first["offset_x"] - avg_last["offset_x"]) > BRICK_STUDY_SPLIT_DIFF_MM
        or abs(avg_first["angle"] - avg_last["angle"]) > BRICK_STUDY_SPLIT_DIFF_DEG
    ):
        return {
            "confidence": 0,
            "message": "0% confidence (frames 1 and 2 were highly different than frames 3 and 4)",
            "average": None,
            "reset": True,
        }

    med_dist = statistics.median([f["dist"] for f in frames])
    med_offset = statistics.median([f["offset_x"] for f in frames])
    med_angle = statistics.median([f["angle"] for f in frames])
    keep = []
    discarded = []
    for idx, frame in enumerate(frames, start=1):
        if not frame["found"]:
            discarded.append(idx)
            continue
        if (
            abs(frame["dist"] - med_dist) > BRICK_STUDY_OUTLIER_MM
            or abs(frame["offset_x"] - med_offset) > BRICK_STUDY_OUTLIER_MM
            or abs(frame["angle"] - med_angle) > BRICK_STUDY_OUTLIER_DEG
        ):
            discarded.append(idx)
        else:
            keep.append(frame)

    if len(keep) < 3:
        return {
            "confidence": 0,
            "message": "0% confidence (variance too high on all frames)",
            "average": None,
            "reset": True,
        }

    std_dist = statistics.pstdev([f["dist"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_MM
    std_offset = statistics.pstdev([f["offset_x"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_MM
    std_angle = statistics.pstdev([f["angle"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_DEG
    variance_label = _variance_label(std_dist, std_offset, std_angle)
    if variance_label == "high variance":
        return {
            "confidence": 0,
            "message": "0% confidence (variance too high on all frames)",
            "average": None,
            "reset": True,
        }

    confidence = int(round(100 * len(keep) / BRICK_STUDY_FRAMES))
    discarded_label, discarded_count = _ordinal_list(discarded)
    if discarded_count == 0:
        discarded_phrase = "discarded none"
    elif discarded_count == 1:
        discarded_phrase = f"discarded the {discarded_label} frame"
    else:
        discarded_phrase = f"discarded the {discarded_label} frames"
    message = (
        f"{confidence}% confidence ({len(keep)}/{BRICK_STUDY_FRAMES} frames had {variance_label}; "
        f"{discarded_phrase})"
    )
    return {
        "confidence": confidence,
        "message": message,
        "average": _average_frames(keep),
        "reset": True,
    }


def cyan_stream_debug_lines(app_state, vision):
    active_mode = _stream_state_vision_mode(app_state, _DEFAULT_VISION_MODE)
    if normalize_vision_mode(active_mode) != VISION_MODE_CYAN:
        return None

    profile = _stream_state_cyan_profile(app_state, _DEFAULT_CYAN_PROFILE)
    profile_text = cyan_profile_label(profile)
    model_name = "-"
    model_path = getattr(vision, "model_path", None)
    if model_path:
        model_name = Path(str(model_path)).name or str(model_path)

    status = str(getattr(vision, "last_status", "searching")).strip() or "searching"
    raw_count = getattr(vision, "last_raw_prediction_count", None)
    candidate_count = getattr(vision, "last_candidate_count", None)
    nms_count = getattr(vision, "last_nms_count", None)
    top_conf = getattr(vision, "last_primary_confidence", None)
    raw_max_conf = getattr(vision, "last_max_confidence", None)
    conf_threshold = getattr(vision, "conf_threshold", None)
    smooth_alpha = getattr(vision, "_smooth_alpha", None)
    input_size = getattr(vision, "input_size", None)

    raw_text = str(int(raw_count)) if isinstance(raw_count, (int, float)) else "-"
    candidate_text = str(int(candidate_count)) if isinstance(candidate_count, (int, float)) else "-"
    nms_text = str(int(nms_count)) if isinstance(nms_count, (int, float)) else "-"
    if isinstance(top_conf, (int, float)):
        top_conf_text = f"{max(0.0, min(1.0, float(top_conf))) * 100.0:.0f}%"
    else:
        top_conf_text = "-"
    if isinstance(raw_max_conf, (int, float)):
        raw_max_conf_text = f"{max(0.0, min(1.0, float(raw_max_conf))) * 100.0:.2f}%"
    else:
        raw_max_conf_text = "-"
    if isinstance(conf_threshold, (int, float)):
        threshold_text = f"{max(0.0, min(1.0, float(conf_threshold))) * 100.0:.2f}%"
    else:
        threshold_text = "-"
    if isinstance(smooth_alpha, (int, float)):
        smooth_text = f"{max(0.0, min(1.0, float(smooth_alpha))):.2f}"
    else:
        smooth_text = "-"
    input_text = f"{int(input_size)}" if isinstance(input_size, (int, float)) else "-"

    return [
        f"[ML] PROFILE: {profile_text} | MODEL: {model_name}",
        f"[ML] SEARCH: {status} | RAW:{raw_text} >THR:{candidate_text} NMS:{nms_text}",
        f"[ML] TOP CONF: {top_conf_text} | MAX RAW: {raw_max_conf_text} | MIN CONF: {threshold_text} | SMOOTH: {smooth_text} | INPUT: {input_text}px",
    ]


def markerless_stream_debug_lines(app_state, vision):
    return cyan_stream_debug_lines(app_state, vision)


def _build_cyan_mask_preview(vision, frame):
    """Build a black/white cyan mask preview from the current camera frame.

    White pixels represent values inside the active cyan HSV range.
    """
    if vision is None:
        return None
    hsv_enabled = getattr(vision, "_hsv_enabled", None)
    hsv_lower = getattr(vision, "_hsv_lower", None)
    hsv_upper = getattr(vision, "_hsv_upper", None)
    if hsv_enabled is False or hsv_lower is None or hsv_upper is None:
        return None

    src = getattr(vision, "raw_frame", None)
    if src is None:
        src = frame
    if src is None:
        return None

    try:
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        lower = np.array(hsv_lower, dtype=np.uint8)
        upper = np.array(hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        erode_iter = max(0, int(getattr(vision, "_hsv_erode_iterations", 0) or 0))
        if erode_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=erode_iter)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


def _overlay_corner_inset(frame, inset, *, title="CYAN MASK"):
    if frame is None or inset is None:
        return frame
    try:
        h, w = frame.shape[:2]
        ih, iw = inset.shape[:2]
        if h <= 0 or w <= 0 or ih <= 0 or iw <= 0:
            return frame

        target_w = max(120, int(0.28 * w))
        scale = float(target_w) / float(iw)
        target_h = max(90, int(ih * scale))
        if target_h > int(0.45 * h):
            target_h = int(0.45 * h)
            target_w = max(80, int(iw * (float(target_h) / float(ih))))

        inset_resized = cv2.resize(inset, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        margin = 10
        x1 = max(0, w - target_w - margin)
        y1 = margin
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)

        frame[y1:y2, x1:x2] = inset_resized[0 : (y2 - y1), 0 : (x2 - x1)]
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(title),
            (x1 + 6, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame
    except Exception:
        return frame


def update_stream_frame(app_state):
    vision = app_state.vision
    if vision is None or vision.current_frame is None:
        return
    frame = vision.current_frame.copy()
    with app_state.lock:
        stream_step = _stream_state_success_gate_step(
            app_state,
            getattr(app_state.world.step_state, "value", app_state.world.step_state),
        )
        step_suggestions = []
        if app_state.step_suggestions:
            step_suggestions.extend(app_state.step_suggestions)
        active = True
        gate_summary, analytics = compute_stream_gate_summary(
            app_state.world,
            stream_step,
            active=active,
            include_step_line=False,
        )
        gate_checker_summary = format_gatecheck_stream_lines(
            app_state.world,
            stream_step,
        )
        if analytics:
            app_state.brick_highlight_metric = analytics.get("highlight_metric")
        cyan_lines = cyan_stream_debug_lines(app_state, vision)
        stream_lines = []
        frame = draw_telemetry_overlay(
            frame,
            app_state.world,
            show_prompt=False,
            gate_status=None,
            gate_progress=None,
            step_suggestions=None,
            highlight_metric=app_state.brick_highlight_metric,
            loop_id=getattr(app_state.world, "loop_id", None),
            gate_summary=gate_summary if gate_summary is not None else [],
            gate_checker_summary=gate_checker_summary,
            draw_text=False,
            line_sink=stream_lines,
            show_center_line=bool(app_state.stream_state.get("show_center_line", True)),
            brick_extra_lines=cyan_lines,
            telemetry_step=stream_step,
        )
        mask_preview = _build_cyan_mask_preview(vision, frame)
        frame = _overlay_corner_inset(frame, mask_preview, title="CYAN MASK")
        app_state.current_frame = frame
    if app_state.stream_state:
        with app_state.stream_state["lock"]:
            app_state.stream_state["frame"] = frame
            app_state.stream_state["text_lines"] = stream_lines


def make_auto_observer(app_state):
    def _observer(stage, world, vision, cmd, speed, reason):
        if bool(getattr(app_state, "auto_cancel_event", None)) and app_state.auto_cancel_event.is_set():
            raise AutoStepCancelled()
        # replay_segment already refreshed world from vision before observer callbacks.
        # Avoid a second camera read here; it can stall stream updates on some cameras.
        if stage == "action" and cmd:
            score = getattr(world, "_last_action_score", None)
            duration_ms = getattr(world, "_last_action_duration_ms", None)
            sent_speed = getattr(world, "_last_action_speed", speed)
            _log_motion_event_and_state(app_state, cmd, sent_speed, score, duration_ms)
        elif stage == "frame" and _demo_logging_active(app_state):
            if app_state.logger is not None and not getattr(app_state, "logger_closed", True):
                app_state.logger.log_state(world)
        refresh_brick_telemetry(app_state, read_vision=False)
    return _observer


def make_auto_confirm(app_state):
    def _confirm(world, vision):
        with app_state.lock:
            app_state.auto_confirm_event.clear()
            app_state.auto_confirm_needed = True
            last_enter_time = app_state.last_enter_time
            start_enter_time = app_state.last_enter_time
        if time.time() - last_enter_time <= 0.75:
            with app_state.lock:
                app_state.last_enter_time = 0.0
                app_state.auto_confirm_needed = False
            return True
        log_line("[AUTO] Press Enter to execute suggested action.")
        while app_state.running:
            if bool(getattr(app_state, "auto_cancel_event", None)) and app_state.auto_cancel_event.is_set():
                with app_state.lock:
                    app_state.auto_confirm_needed = False
                raise AutoStepCancelled()
            if app_state.auto_confirm_event.is_set():
                return True
            with app_state.lock:
                recent_enter = app_state.last_enter_time
            if recent_enter > start_enter_time and time.time() - recent_enter <= 1.0:
                with app_state.lock:
                    app_state.last_enter_time = 0.0
                    app_state.auto_confirm_needed = False
                return True
            update_world_from_vision(world, vision, log=False)
            refresh_brick_telemetry(app_state, read_vision=False)
            time.sleep(0.05)
        return False
    return _confirm

def prompt_for_stage(stage, obj_label):
    if stage == "FAIL":
        return f"Press f to start failed {obj_label} demo."
    if stage == "RECOVER":
        return f"Press f to start recovery {obj_label} demo."
    if stage == "SUCCESS":
        return f"Press f to start clean {obj_label} demo."
    return None

def resolve_step_token(token):
    if not token:
        return None
    key = token.strip().lower()
    if key in STEP_CODES:
        return STEP_CODES[key]
    if key in STEP_NAMES:
        return STEP_NAMES[key]
    for name, obj in STEP_NAMES.items():
        if name.startswith(key):
            return obj
    return None

def resolve_attempt_token(token):
    if not token:
        return None
    key = token.strip().lower()
    if key in ATTEMPT_CODES:
        return ATTEMPT_CODES[key]
    if key in ATTEMPT_NAMES:
        return ATTEMPT_NAMES[key]
    for name, attempt in ATTEMPT_NAMES.items():
        if name.startswith(key):
            return attempt
    return None

def parse_text_command(text):
    if not text:
        return None, None, None, "Empty command."
    tokens = text.strip().lower().split()
    auto_mode = False
    if tokens and tokens[0] in ("auto", "robot", "bot", "run"):
        auto_mode = True
        tokens = tokens[1:]
    if not tokens:
        return auto_mode, None, None, "Missing step/attempt."
    if len(tokens) == 1 and len(tokens[0]) >= 2:
        token = tokens[0]
        if token[0].isdigit():
            digits = []
            for ch in token:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            obj_token = "".join(digits)
            attempt_token = token[len(obj_token):]
            if not attempt_token:
                return auto_mode, None, None, "Missing attempt."
        else:
            obj_token = token[0]
            attempt_token = token[1:]
    elif len(tokens) >= 2:
        obj_token = tokens[0]
        attempt_token = tokens[1]
    else:
        return auto_mode, None, None, "Missing attempt."
    obj_enum = resolve_step_token(obj_token)
    attempt = resolve_attempt_token(attempt_token)
    if not obj_enum or not attempt:
        return auto_mode, obj_enum, attempt, "Unknown step or attempt."
    return auto_mode, obj_enum, attempt, None

def print_command_help(app_state=None):
    log_line("[CMD] Enter command mode with ':'")
    log_line("[CMD] Attempt codes: f=fail, s=success, r=recover, n=nominal")
    log_line("[CMD] Example: :4s (scoop success), :4n (scoop nominal)")
    log_line("[CMD] Auto-run: type the full step number, then press Enter.")
    log_line("[CMD] Auto-run shortcut: blank Enter queues the next step after the last successful auto step.")
    log_line(f"[CMD] Auto-continue: after a successful auto-step, next step auto-queues in {float(AUTO_CONTINUE_WAIT_S):.1f}s unless cancelled.")
    log_line("[CMD] Hotkey tuning: press a movement hotkey, then press y to edit that hotkey's vars.")
    log_line("[CMD] Lift preflight: :liftcal (discover O/K micro movement + find ground level).")
    log_line("[CMD] End attempt: press ':' to finish and return to the command prompt.")
    log_step_codes_list(prefix="[CMD]")


def format_hotkey_speeds():
    categories = {
        "f": "Drive",
        "b": "Drive",
        "l": "Turn",
        "r": "Turn",
        "u": "Lift",
        "d": "Lift",
    }
    key_order = {key: idx for idx, key in enumerate([
        "W", "S", "R", "F", "T", "G",
        "A", "D", "Q", "E", "Z", "C",
        "O", "K", "U", "P", "L",
    ])}
    grouped = {}
    for key, info in HOTKEY_SPEED_SCORES.items():
        cmd = info.get("cmd")
        score = info.get("score")
        if cmd not in categories or score is None:
            continue
        grouped.setdefault(categories[cmd], {}).setdefault(score, []).append(key.upper())
    def score_display(score):
        return f"{int(score)}%"

    parts = []
    for label in ("Drive", "Turn", "Lift"):
        score_map = grouped.get(label, {})
        if not score_map:
            continue
        segments = []
        for score in sorted(score_map.keys()):
            keys = "/".join(sorted(score_map[score], key=lambda k: key_order.get(k, 999)))
            segments.append(f"{keys} {score_display(score)}")
        parts.append(f"{label}: " + ", ".join(segments))
    if not parts:
        return ""
    return "[KEYS] " + ". ".join(parts) + "."

def command_mode_exit_messages(app_state):
    if app_state.active_attempt:
        return ["[MODE] Manual mode (logging active)."]
    return ["[MODE] Manual mode and not logging."]

def handle_command_line(app_state, cmd):
    cmd_lower = (cmd or "").strip().lower()
    messages = []
    do_help = False
    exit_mode = False
    ended_info = None
    run_lift_preflight_cmd = False

    with app_state.lock:
        if cmd_lower in ("", ":"):
            if app_state.active_attempt:
                ok, msg, obj_enum, attempt_type, should_close = end_attempt(app_state)
                messages.append(msg)
                if ok:
                    ended_info = (obj_enum, attempt_type)
            exit_mode = True
            return exit_mode, do_help, messages, ended_info
        if cmd_lower in ("q", "quit", "exit"):
            app_state.running = False
            messages.append("Stopping manual recording...")
            exit_mode = True
            return exit_mode, do_help, messages, ended_info
        if cmd_lower in ("help", "h", "?"):
            do_help = True
        elif cmd_lower in ("status", "state"):
            obj = app_state.world.step_state
            attempt = app_state.active_attempt or "NONE"
            messages.append(f"[STATE] Step={step_label(obj)} Attempt={attempt}")
        elif cmd_lower in ("liftcal", "lift_cal", "lift-ground", "groundlift"):
            run_lift_preflight_cmd = True
        elif cmd_lower in ("end", "stop", "done"):
            ok, msg, obj_enum, attempt_type, should_close = end_attempt(app_state)
            messages.append(msg)
            if ok:
                ended_info = (obj_enum, attempt_type)
        else:
            auto_mode, obj_enum, attempt_type, err = parse_text_command(cmd)
            if err:
                messages.append(f"[CMD] {err}")
            elif auto_mode:
                messages.append("[AUTO] Auto-run is disabled. Use manual commands.")
                exit_mode = True
            else:
                ok, msg, ended, should_close_val = handle_attempt_command(app_state, obj_enum, attempt_type)
                messages.append(msg)
                should_close = should_close_val # Propagate
                if ok:
                    # Return to driving mode immediately on start/stop
                    exit_mode = True
                    if app_state.active_attempt:
                        messages.append("[CMD] Press ':' to finish and return to the command prompt.")
                if ended:
                    ended_info = ended

    if run_lift_preflight_cmd:
        result = run_lift_ground_level_preflight(app_state, force=True)
        if bool(result.get("ok")):
            messages.append("[PREFLIGHT LIFT] Complete.")
        elif bool(result.get("busy")):
            messages.append("[PREFLIGHT LIFT] Already running.")
        else:
            messages.append(f"[PREFLIGHT LIFT] Failed ({result.get('error')}).")

    return exit_mode, do_help, messages, ended_info


def start_attempt(app_state, obj_enum, attempt_type):
    obj_label = step_label(obj_enum)
    if app_state.step_open and app_state.open_step != obj_enum:
        return False, f"[STEP] Finish {step_label(app_state.open_step)} before switching steps."
    ensure_log_open(app_state)
    if not app_state.step_open:
        app_state.step_open = True
        app_state.open_step = obj_enum
    app_state.world.step_state = obj_enum
    marker = ATTEMPT_MARKERS[attempt_type][0]
    app_state.logger.log_keyframe(marker, obj_label)
    app_state.active_attempt = attempt_type
    app_state.current_attempt_action_count = 0
    app_state.world.attempt_status = ATTEMPT_STATUS[attempt_type]
    app_state.world.recording_active = True
    log_line(height_intel_step_begin_line(app_state.world, obj_label))
    return True, f"[OBJ] {obj_label} {attempt_type} started."

def end_attempt(app_state, complete_step=True):
    if not app_state.active_attempt:
        return False, "[OBJ] No active attempt.", None, None, False
    obj_enum = app_state.world.step_state
    obj_label = step_label(obj_enum)
    attempt_type = app_state.active_attempt
    marker = ATTEMPT_MARKERS[attempt_type][1]
    app_state.logger.log_keyframe(marker, obj_label)
    should_close = False
    try:
        attempt_actions = int(getattr(app_state, "current_attempt_action_count", 0) or 0)
    except (TypeError, ValueError):
        attempt_actions = 0

    if attempt_type == "SUCCESS":
        _emit_stream_step_success_event(app_state, obj_label)
        _record_step_completion_actions(
            app_state,
            obj_label,
            int(max(0, int(attempt_actions))),
            source="manual",
        )
        try:
            apply_height_snapshot_from_step(app_state.world, obj_label, log=True)
        except Exception as exc:
            log_line(f"[HEIGHT INTEL] {obj_label} snapshot skipped ({exc}).")
        try:
            apply_height_inventory_adjustment_from_step(app_state.world, obj_label, log=True)
        except Exception as exc:
            log_line(f"[HEIGHT INTEL] {obj_label} inventory adjustment skipped ({exc}).")
        if complete_step:
            app_state.step_open = False
            app_state.open_step = None
            current_obj = app_state.world.step_state
            app_state.world.reset_mission()
            app_state.world.step_state = current_obj
        app_state.world.recording_active = False
    else:
        # For FAIL/RECOVER, we keep it open by default to allow retry,
        # UNLESS we are explicitly closing the step.
        if complete_step:
            app_state.step_open = False
            app_state.open_step = None
        app_state.world.recording_active = False

    app_state.world.attempt_status = "NORMAL"
    app_state.active_attempt = None
    app_state.current_attempt_action_count = 0
    app_state.logger.enabled = False
    return True, f"[OBJ] {obj_label} {attempt_type} finished.", obj_enum, attempt_type, False

def handle_attempt_command(app_state, obj_enum, attempt_type):
    obj_label = step_label(obj_enum)
    ended_info = None
    ended_close = False

    # 1. If ANY recording is active, end it first.
    if app_state.active_attempt:
        ok, msg, ended_obj, ended_attempt, should_close = end_attempt(app_state, complete_step=(app_state.active_attempt == attempt_type))
        if ok:
            ended_info = (ended_obj, ended_attempt)
            ended_close = should_close
            log_line(msg)

    # 2. Check Step Constraints for the NEW attempt
    if app_state.step_open and app_state.open_step != obj_enum:
        return False, f"[STEP] Finish {step_label(app_state.open_step)} before switching steps.", ended_info, ended_close

    # 3. If they were just toggling OFF the same thing, we are done.
    if ended_info and ended_info[0] == obj_enum and ended_info[1] == attempt_type:
        return True, f"[OBJ] {obj_label} finished.", ended_info, ended_close

    # 4. Start the new attempt
    ok, msg = start_attempt(app_state, obj_enum, attempt_type)
    if ok and ended_info:
        _record_manual_step_restart(app_state, obj_label)
    return ok, msg, ended_info, ended_close

class AppState:
    def __init__(self, vision_mode=_DEFAULT_VISION_MODE):
        self.running = True
        self.active_command = None
        self.active_hotkey = None
        self.active_speed = 0.0
        self.active_speed_score = None
        self.active_duration_ms = None
        self.active_pwm_override = None
        self.last_key_time = 0

        self.hotkey_edit_target = None
        self.preflight_checked_hotkeys = set()
        self.preflight_running_hotkeys = set()
        self.lift_ground_preflight_done = False
        self.lift_ground_preflight_running = False
        self.lift_ground_preflight_cam_h = None
        self.lift_ground_preflight_time = 0.0
        # Job Status
        self.job_success = False
        self.job_success_timer = 0
        self.job_start = False
        self.job_start_timer = 0
        self.job_abort = False
        self.job_abort_timer = 0
        
        # Job Status
        
        self.lock = threading.Lock()
        self.vision_io_lock = threading.Lock()
        self.current_frame = None
        self.step_open = False
        self.open_step = None
        self.active_attempt = None
        
        # Session Setup
        vision_mode_norm = normalize_vision_mode(vision_mode, fallback=_DEFAULT_VISION_MODE)
        self.demos_dir = _demos_dir_for_vision_mode(vision_mode_norm)
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = _runs_dir_for_vision_mode(vision_mode_norm)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.world = WorldModel()
        self.logger = None
        self.logger_closed = True
        self.log_path = None
        self.session_started_ts = float(time.time())
        self.session_log_paths = []
        self.session_step_actions_by_step = {}
        self.session_step_action_records = []
        self.session_auto_step_cancel_count = 0
        self.session_auto_step_cancel_by_step = {}
        self.session_auto_step_restart_count = 0
        self.session_auto_step_restart_by_step = {}
        self.session_manual_step_restart_count = 0
        self.session_manual_step_restart_by_step = {}
        self.session_manual_hotkey_press_count = 0
        self.session_manual_hotkey_press_by_hotkey = {}
        self.session_manual_hotkey_press_by_command = {}
        self.session_manual_hotkey_press_details = []
        self.session_manual_hotkey_edit_count = 0
        self.session_manual_hotkey_edit_by_hotkey = {}
        self.session_manual_hotkey_edit_details = []
        self.current_attempt_action_count = 0
        self.current_auto_step = None
        self.current_auto_action_count = 0
        open_new_log(self)
        
        # ID Init
        self.world.step_state = StepState.ALIGN_BRICK  # Default to ALIGN_BRICK for gate testing

        self.vision = None
        self.robot = None

        self.auto_prompt = False
        self.auto_request = None
        self.auto_running = False
        self.auto_cancel_event = threading.Event()
        self.auto_demo_logging_active = False
        self.auto_confirm_needed = False
        self.auto_confirm_event = threading.Event()
        self.last_enter_time = 0.0
        self.last_auto_completed_step = None
        self.auto_continue_deadline = 0.0
        self.auto_continue_source_step = None
        self.auto_continue_target_step = None
        self.auto_continue_last_countdown_second = None

        self.gate_status = []
        self.gate_progress = []
        self.step_suggestions = []
        self.brick_highlight_metric = None
        self.brick_frame_buffer = []
        self.stream_state = {
            "frame": None,
            "lock": threading.Lock(),
            "skip_telemetry_process": True,
            "show_center_line": True,
            "vision_mode": vision_mode_norm,
            "cyan_profile": _DEFAULT_CYAN_PROFILE,
            "cyan_visibility": _DEFAULT_CYAN_VISIBILITY,
            "success_gate_step": _DEFAULT_STREAM_SUCCESS_GATE_STEP,
            "step_success_seq": 0,
            "step_success_step": None,
            "step_success_at": 0.0,
        }
        self.stream_enabled = False
        self.pending_stream_started_url = None
        # Allow telemetry_process helpers to push frames directly during auto-run pauses.
        self.world._stream_state = self.stream_state

        self.config_mtime = 0
        self.last_config_check = 0
        self._last_align_profile_signature = None

def getch():
    """Reads a single character from stdin in raw mode."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch



def keyboard_thread(app_state):
    step_code_buffer = ""

    def _clear_step_code_buffer():
        nonlocal step_code_buffer
        step_code_buffer = ""

    def _step_code_buffer_message():
        if not step_code_buffer:
            return "[AUTO] Step code buffer cleared."
        return f"[AUTO] Step code buffer: {step_code_buffer} (Enter=run, Backspace=edit)."

    def _queue_auto_step_request(obj_enum):
        if not obj_enum:
            return False, ["[AUTO] Unknown step code."]
        logs = load_demo_logs(app_state.demos_dir)
        if not logs:
            return False, ["[AUTO] No demo logs found. Record a demo first."]

        update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
        refresh_autobuild_config(PROCESS_MODEL_FILE)
        model = load_process_model(PROCESS_MODEL_FILE)
        obj_key = normalize_step_label(obj_enum.value)
        obj_cfg = (model.get("steps") or {}).get(obj_key, {})
        success_gates = obj_cfg.get("success_gates") if isinstance(obj_cfg, dict) else None
        nominal_only = isinstance(obj_cfg, dict) and obj_cfg.get("nominalDemosOnly")
        if not success_gates and not nominal_only:
            return False, [f"[AUTO] No success gates for {obj_key}. Record a success demo first."]

        with app_state.lock:
            app_state.last_key_time = time.time()
            app_state.auto_prompt = False
            try:
                app_state.auto_cancel_event.clear()
            except Exception:
                pass
            _clear_auto_continue_locked(app_state)
            app_state.auto_request = obj_enum
            app_state.active_command = None
            app_state.active_hotkey = None
            app_state.active_speed = 0.0
            app_state.active_speed_score = None
            app_state.active_duration_ms = None
            app_state.active_pwm_override = None
            _set_stream_state_success_gate_step(app_state, obj_enum.value)
        return True, []

    def _submit_step_code_buffer():
        nonlocal step_code_buffer
        token = str(step_code_buffer or "").strip()
        step_code_buffer = ""
        if not token:
            return []
        obj_enum = resolve_step_token(token)
        ok, msg_list = _queue_auto_step_request(obj_enum)
        if ok:
            step_code = step_code_for_obj(obj_enum) if obj_enum is not None else token
            step_name = str(getattr(obj_enum, "value", obj_enum or token)).lower()
            return [f"[AUTO] Queued step {step_code} ({step_name})."]
        return msg_list

    def _blank_enter_next_step():
        with app_state.lock:
            auto_running = bool(app_state.auto_running)
            last_completed = app_state.last_auto_completed_step
            auto_prompt_active = bool(app_state.auto_prompt)
        if auto_running:
            return []
        if last_completed is None:
            if auto_prompt_active:
                return ["[AUTO] Blank Enter shortcut needs a previous successful auto step."]
            return []
        next_obj = next_demo_step_obj(last_completed)
        if next_obj is None:
            last_code = step_code_for_obj(last_completed)
            last_name = str(getattr(last_completed, "value", last_completed)).lower()
            return [f"[AUTO] Blank Enter: no next step after {last_code} ({last_name})."]
        ok, msg_list = _queue_auto_step_request(next_obj)
        if not ok:
            return msg_list
        next_code = step_code_for_obj(next_obj)
        next_name = str(getattr(next_obj, "value", next_obj)).lower()
        return [f"[AUTO] Blank Enter -> queued next step {next_code} ({next_name})."]

    def _prompt_auto_step_input(initial_text=""):
        _clear_step_code_buffer()
        seed = str(initial_text or "")
        prompt = "[AUTO] Step > "

        fd_term = sys.stdin.fileno()
        try:
            attr_term = termios.tcgetattr(fd_term)
            attr_line = termios.tcgetattr(fd_term)
            attr_line[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(fd_term, termios.TCSANOW, attr_line)
        except Exception:
            attr_term = None

        line = None
        readline_mod = None
        prev_hook = None
        used_prefill = False
        if seed:
            try:
                import readline as readline_mod  # type: ignore
                if hasattr(readline_mod, "get_startup_hook"):
                    prev_hook = readline_mod.get_startup_hook()

                def _prefill_hook():
                    try:
                        readline_mod.insert_text(seed)
                        if hasattr(readline_mod, "redisplay"):
                            readline_mod.redisplay()
                    except Exception:
                        pass

                readline_mod.set_startup_hook(_prefill_hook)
                used_prefill = True
            except Exception:
                readline_mod = None
                prev_hook = None
                used_prefill = False

        try:
            if seed and not used_prefill:
                # Fallback when readline prefill is unavailable: keep the first digit(s)
                # the user already typed and let them type the remainder.
                suffix = input(f"{prompt}{seed}")
                line = f"{seed}{suffix}"
            else:
                line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            line = None
        finally:
            if readline_mod is not None:
                try:
                    readline_mod.set_startup_hook(prev_hook)
                except Exception:
                    try:
                        readline_mod.set_startup_hook(None)
                    except Exception:
                        pass
            if attr_term is not None:
                try:
                    termios.tcsetattr(fd_term, termios.TCSANOW, attr_term)
                except Exception:
                    pass
            try:
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except Exception:
                pass

        if line is None:
            return ["[AUTO] Auto step prompt cancelled."]

        token = str(line).strip()
        if not token:
            return _blank_enter_next_step()

        obj_enum = resolve_step_token(token)
        ok, msg_list = _queue_auto_step_request(obj_enum)
        if ok:
            step_code = step_code_for_obj(obj_enum) if obj_enum is not None else token
            step_name = str(getattr(obj_enum, "value", obj_enum or token)).lower()
            return [f"[AUTO] Queued step {step_code} ({step_name})."]
        return msg_list

    while app_state.running:
        ch = getch()
        if not ch:
            continue
        ch_lower = ch.lower()
        auto_continue_cleared_by_input = False
        with app_state.lock:
            if ch not in ('\n', '\r') and _auto_continue_pending_locked(app_state):
                _clear_auto_continue_locked(app_state)
                auto_continue_cleared_by_input = True
            auto_prompt = app_state.auto_prompt
            auto_confirm_needed = app_state.auto_confirm_needed

        messages = []
        pending_hotkey_action = None
        if ch == 'Q':
            _clear_step_code_buffer()
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.running = False
            messages.append("Stopping manual recording...")
        elif ch == '\x1b':
            _clear_step_code_buffer()
            robot_to_stop = None
            with app_state.lock:
                app_state.last_key_time = time.time()
                if bool(app_state.auto_running):
                    app_state.auto_cancel_event.set()
                    app_state.auto_confirm_needed = False
                    app_state.auto_confirm_event.set()
                    app_state.auto_request = None
                    robot_to_stop = app_state.robot
                    messages.append("[AUTO] Cancel requested (Esc). Stopping current auto-step...")
                elif app_state.auto_request is not None:
                    app_state.auto_request = None
                    try:
                        app_state.auto_cancel_event.clear()
                    except Exception:
                        pass
                    messages.append("[AUTO] Cleared queued auto-step request.")
                elif auto_continue_cleared_by_input:
                    messages.append("[AUTO] Auto-continue cancelled (Esc).")
                else:
                    messages.append("[AUTO] Esc: no auto-step is currently running.")
            if robot_to_stop is not None:
                try:
                    robot_to_stop.stop()
                except Exception:
                    pass
        elif ch in ('\n', '\r'):
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.last_enter_time = time.time()
                if auto_confirm_needed:
                    app_state.auto_confirm_needed = False
                    app_state.auto_confirm_event.set()
            if auto_confirm_needed:
                messages.append("[AUTO] Action confirmed.")
            elif step_code_buffer:
                messages.extend(_submit_step_code_buffer())
            else:
                messages.extend(_blank_enter_next_step())
        elif ch in ('\x7f', '\b'):
            with app_state.lock:
                app_state.last_key_time = time.time()
            if step_code_buffer:
                step_code_buffer = step_code_buffer[:-1]
                messages.append(_step_code_buffer_message())
        elif ch_lower.isdigit():
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.auto_prompt = False
            messages.extend(_prompt_auto_step_input(initial_text=ch_lower))
        elif auto_prompt:
            with app_state.lock:
                app_state.last_key_time = time.time()
                if ch_lower == 'm':
                    app_state.auto_prompt = False
                    _clear_step_code_buffer()
                    messages.append("[AUTO] Auto mode cancelled.")
                else:
                    if step_code_buffer:
                        messages.append(_step_code_buffer_message())
                    else:
                        messages.append(
                            "[AUTO] Type the full step number, then press Enter (blank Enter = next step)."
                        )
        elif ch_lower == ':':
            _clear_step_code_buffer()
            with app_state.lock:
                app_state.last_key_time = time.time()
                end_msg = None
                if app_state.active_attempt:
                    ok, msg, _, _, _ = end_attempt(app_state, complete_step=True)
                    end_msg = msg

                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None
                app_state.active_pwm_override = None
                app_state.active_pwm_override = None
            if end_msg:
                log_line(end_msg)
            log_line("[CMD] Enter command (ex: 4s, 4f, help). Use ':' or blank to exit.")
            log_step_codes_list(prefix="[CMD]")
            
            # Force enable ECHO and ICANON so input() is visible
            fd_term = sys.stdin.fileno()
            attr_term = termios.tcgetattr(fd_term)
            attr_term[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(fd_term, termios.TCSANOW, attr_term)
            
            while app_state.running:
                try:
                    cmd = input("[CMD] > ")
                except (EOFError, KeyboardInterrupt):
                    cmd = ""
                exit_mode, do_help, messages, ended_info = handle_command_line(app_state, cmd)
                if do_help:
                    print_command_help(app_state)
                for msg in messages:
                    log_line(msg)
                if ended_info:
                    pass
                if exit_mode:
                    for msg in command_mode_exit_messages(app_state):
                        log_line(msg)
                    # Flush and CLEAR messages to prevent double-printing at bottom of thread
                    messages = []
                    try:
                        termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    except:
                        pass
                    break
        elif ch_lower == 'm':
            _clear_step_code_buffer()
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.auto_prompt = False
                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None
                app_state.active_pwm_override = None
            messages.extend(_prompt_auto_step_input(initial_text=""))
        elif ch_lower == 'y':
            _clear_step_code_buffer()
            with app_state.lock:
                app_state.last_key_time = time.time()
                target = app_state.hotkey_edit_target if isinstance(app_state.hotkey_edit_target, dict) else None
            if not isinstance(target, dict):
                messages.append("[HOTKEY] Press a movement hotkey first, then press y to edit its vars.")
            else:
                cmd = target.get("cmd")
                score = target.get("score")
                if cmd is None or score is None:
                    messages.append("[HOTKEY] No valid hotkey action to edit yet.")
                else:
                    maybe_prompt_hotkey_var_update(app_state, cmd, score, enter_edit=True)
        else:
            _clear_step_code_buffer()
            with app_state.lock:
                app_state.last_key_time = time.time()
                # MOVEMENT (Heartbeat triggers)
                entry = HOTKEY_SPEED_SCORES.get(ch_lower) if isinstance(HOTKEY_SPEED_SCORES, dict) else None
                if isinstance(entry, dict):
                    cmd = str(entry.get("cmd", "")).strip().lower()
                    try:
                        score = int(entry.get("score"))
                    except (TypeError, ValueError):
                        score = None
                    try:
                        duration_ms = int(round(float(entry.get("duration_ms"))))
                    except (TypeError, ValueError):
                        duration_ms = None
                    if duration_ms is not None and duration_ms <= 0:
                        duration_ms = None
                    try:
                        pwm_override = int(round(float(entry.get("pwm"))))
                    except (TypeError, ValueError):
                        pwm_override = None
                    if pwm_override is not None and pwm_override <= 0:
                        pwm_override = None
                else:
                    cmd = None
                    score = None
                    duration_ms = None
                    pwm_override = None
                if cmd and score is not None:
                    app_state.active_command = None
                    app_state.active_hotkey = None
                    app_state.active_speed = 0.0
                    app_state.active_speed_score = None
                    app_state.active_duration_ms = None
                    app_state.active_pwm_override = None
                    pending_hotkey_action = (cmd, score, ch_lower, duration_ms, pwm_override)
                elif ch_lower in ('h', '?'):
                    print_command_help(app_state)

        if pending_hotkey_action and app_state.running:
            cmd, score, hotkey, duration_ms, pwm_override = pending_hotkey_action
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.active_command = cmd
                app_state.active_hotkey = str(hotkey).lower()
                app_state.active_speed_score = score
                app_state.active_duration_ms = int(duration_ms) if duration_ms is not None else None
                app_state.active_pwm_override = int(pwm_override) if pwm_override is not None else None
            # Execute the hotkey act first, then offer var editing for that act.
            time.sleep(1.0 / max(COMMAND_RATE_HZ, 1e-3))
            score_used = maybe_prompt_hotkey_var_update(app_state, cmd, score, enter_edit=False, hotkey=hotkey)
            with app_state.lock:
                app_state.hotkey_edit_target = {
                    "hotkey": str(hotkey).lower(),
                    "cmd": cmd,
                    "score": int(score_used) if score_used is not None else int(score),
                    "timestamp": time.time(),
                }

        for msg in messages:
            log_line(msg)

        if not app_state.running:
            break


def stream_refresh_loop(app_state):
    dt = 1.0 / max(STREAM_FPS, 1)
    while app_state.running:
        loop_start = time.time()
        if app_state.vision is not None:
            try:
                update_stream_frame(app_state)
            except Exception:
                # Keep streaming alive even if a single overlay pass fails.
                pass
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def build_vision(vision_mode, yolo_model_path=None):
    mode = normalize_vision_mode(vision_mode)
    if mode == VISION_MODE_CYAN:
        return YoloBrickDetector(debug=True, model_path=yolo_model_path)
    return ArucoBrickVision(debug=True, debug_rejected_candidates=True)


def _stream_state_vision_mode(app_state, fallback):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return normalize_vision_mode(fallback, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        value = stream_state.get("vision_mode", fallback)
    else:
        with lock:
            value = stream_state.get("vision_mode", fallback)
    return normalize_vision_mode(value, fallback=fallback)


def _set_stream_state_vision_mode(app_state, mode):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return
    lock = stream_state.get("lock")
    mode_norm = normalize_vision_mode(mode, fallback=VISION_MODE_CYAN)
    if lock is None:
        stream_state["vision_mode"] = mode_norm
        return
    with lock:
        stream_state["vision_mode"] = mode_norm


def _stream_state_cyan_profile(app_state, fallback):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return normalize_cyan_profile(fallback, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        value = stream_state.get("cyan_profile")
        if (value is None or not str(value).strip()) and "markerless_profile" in stream_state:
            value = stream_state.get("markerless_profile")
    else:
        with lock:
            value = stream_state.get("cyan_profile")
            if (value is None or not str(value).strip()) and "markerless_profile" in stream_state:
                value = stream_state.get("markerless_profile")
    if value is None or not str(value).strip():
        value = fallback
    return normalize_cyan_profile(value, fallback=fallback)


def _set_stream_state_cyan_profile(app_state, profile):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return
    lock = stream_state.get("lock")
    profile_norm = normalize_cyan_profile(profile, fallback=_DEFAULT_CYAN_PROFILE)
    if lock is None:
        stream_state["cyan_profile"] = profile_norm
        stream_state["markerless_profile"] = profile_norm
        return
    with lock:
        stream_state["cyan_profile"] = profile_norm
        stream_state["markerless_profile"] = profile_norm


def _stream_state_cyan_visibility(app_state, fallback):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return normalize_cyan_visibility(fallback, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        value = stream_state.get("cyan_visibility")
        if (value is None or not str(value).strip()) and "markerless_visibility" in stream_state:
            value = stream_state.get("markerless_visibility")
    else:
        with lock:
            value = stream_state.get("cyan_visibility")
            if (value is None or not str(value).strip()) and "markerless_visibility" in stream_state:
                value = stream_state.get("markerless_visibility")
    if value is None or not str(value).strip():
        value = fallback
    return normalize_cyan_visibility(value, fallback=fallback)


def _set_stream_state_cyan_visibility(app_state, mode):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return
    lock = stream_state.get("lock")
    mode_norm = normalize_cyan_visibility(mode, fallback=_DEFAULT_CYAN_VISIBILITY)
    if lock is None:
        stream_state["cyan_visibility"] = mode_norm
        stream_state["markerless_visibility"] = mode_norm
        return
    with lock:
        stream_state["cyan_visibility"] = mode_norm
        stream_state["markerless_visibility"] = mode_norm


def _normalize_stream_success_gate_step(step, fallback=None):
    allowed_steps = {
        normalize_step_label(value)
        for value, _label in STREAM_SUCCESS_GATE_STEP_OPTIONS
        if normalize_step_label(value)
    }
    candidate = normalize_step_label(step)
    if candidate and (not allowed_steps or candidate in allowed_steps):
        return candidate
    fallback_key = normalize_step_label(fallback)
    if fallback_key and (not allowed_steps or fallback_key in allowed_steps):
        return fallback_key
    if STREAM_SUCCESS_GATE_STEP_OPTIONS:
        first = normalize_step_label(STREAM_SUCCESS_GATE_STEP_OPTIONS[0][0])
        if first:
            return first
    if fallback_key:
        return fallback_key
    return _DEFAULT_STREAM_SUCCESS_GATE_STEP


def _stream_state_success_gate_step(app_state, fallback):
    stream_state_raw = getattr(app_state, "stream_state", None)
    stream_state = stream_state_raw if isinstance(stream_state_raw, dict) else None
    if not stream_state:
        return _normalize_stream_success_gate_step(fallback, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        value = stream_state.get("success_gate_step", fallback)
    else:
        with lock:
            value = stream_state.get("success_gate_step", fallback)
    return _normalize_stream_success_gate_step(value, fallback=fallback)


def _set_stream_state_success_gate_step(app_state, step):
    stream_state_raw = getattr(app_state, "stream_state", None)
    stream_state = stream_state_raw if isinstance(stream_state_raw, dict) else None
    if not stream_state:
        return
    world = getattr(app_state, "world", None)
    step_state = getattr(world, "step_state", None)
    fallback = getattr(step_state, "value", step_state)
    step_key = _normalize_stream_success_gate_step(step, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        stream_state["success_gate_step"] = step_key
        return
    with lock:
        stream_state["success_gate_step"] = step_key


def _emit_stream_step_success_event(app_state, step):
    stream_state_raw = getattr(app_state, "stream_state", None)
    stream_state = stream_state_raw if isinstance(stream_state_raw, dict) else None
    if not stream_state:
        return
    step_key = normalize_step_label(step) or str(step or "").strip()
    lock = stream_state.get("lock")

    def _apply():
        current_seq = stream_state.get("step_success_seq", 0)
        try:
            next_seq = int(current_seq) + 1
        except (TypeError, ValueError):
            next_seq = 1
        stream_state["step_success_seq"] = int(max(1, next_seq))
        stream_state["step_success_step"] = step_key or None
        stream_state["step_success_at"] = float(time.time())

    if lock is None:
        _apply()
        return
    with lock:
        _apply()


def _set_detector_cyan_visibility(vision, mode):
    if vision is None:
        return False
    mode_norm = normalize_cyan_visibility(mode, fallback=CYAN_VISIBILITY_AUTO)
    # Prefer explicit detector APIs if present.
    for meth_name in (
        "set_visibility_mode",
        "set_brick_visibility_mode",
        "set_color_mode",
        "set_brick_color_mode",
    ):
        meth = getattr(vision, meth_name, None)
        if callable(meth):
            try:
                meth(mode_norm)
                return True
            except TypeError:
                continue
            except Exception:
                return False
    # Try runtime-tuning style APIs used by the YOLO detector.
    set_runtime_tuning = getattr(vision, "set_runtime_tuning", None)
    if callable(set_runtime_tuning):
        for key in ("visibility_mode", "brick_visibility", "color_mode", "brick_color_mode"):
            try:
                set_runtime_tuning(**{key: mode_norm})
                return True
            except TypeError:
                continue
            except Exception:
                return False
    # Fall back to direct attributes for merged detectors that expose state.
    for attr_name in (
        "visibility_mode",
        "brick_visibility_mode",
        "color_mode",
        "brick_color_mode",
    ):
        if hasattr(vision, attr_name):
            try:
                setattr(vision, attr_name, mode_norm)
                return True
            except Exception:
                return False
    return False


def _apply_cyan_visibility(app_state, vision, mode):
    mode_norm = normalize_cyan_visibility(mode, fallback=_DEFAULT_CYAN_VISIBILITY)
    _set_stream_state_cyan_visibility(app_state, mode_norm)
    if not isinstance(vision, YoloBrickDetector):
        return mode_norm, False
    applied = _set_detector_cyan_visibility(vision, mode_norm)
    return mode_norm, bool(applied)


def _apply_cyan_profile(app_state, vision, profile):
    profile_key, settings = cyan_profile_settings(profile)
    _set_stream_state_cyan_profile(app_state, profile_key)
    if not isinstance(vision, YoloBrickDetector):
        return profile_key, settings, False

    applied_runtime = None
    set_runtime_tuning = getattr(vision, "set_runtime_tuning", None)
    if callable(set_runtime_tuning):
        applied_runtime = set_runtime_tuning(**dict(settings))
    if not isinstance(applied_runtime, dict):
        # Fallback for older detector implementations that don't expose
        # set_runtime_tuning() with full runtime snapshots.
        confidence = settings.get("confidence")
        smooth_alpha = settings.get("smoothing_alpha")
        nms_threshold = settings.get("nms_threshold")
        hsv_lower = settings.get("hsv_lower")
        hsv_upper = settings.get("hsv_upper")
        hsv_enabled = settings.get("hsv_enabled")
        hsv_erode_iterations = settings.get("hsv_erode_iterations")
        if confidence is not None:
            vision.conf_threshold = float(confidence)
        if smooth_alpha is not None:
            vision._smooth_alpha = float(smooth_alpha)
        if nms_threshold is not None:
            vision.nms_threshold = float(nms_threshold)
        if hsv_lower is not None:
            vision._hsv_lower = list(hsv_lower)
        if hsv_upper is not None:
            vision._hsv_upper = list(hsv_upper)
        if hsv_enabled is not None:
            vision._hsv_enabled = bool(hsv_enabled)
        if hsv_erode_iterations is not None:
            vision._hsv_erode_iterations = max(0, int(hsv_erode_iterations))
        applied_runtime = {
            "conf_threshold": getattr(vision, "conf_threshold", None),
            "smooth_alpha": getattr(vision, "_smooth_alpha", None),
            "nms_threshold": getattr(vision, "nms_threshold", None),
            "hsv_lower": getattr(vision, "_hsv_lower", None),
            "hsv_upper": getattr(vision, "_hsv_upper", None),
            "hsv_enabled": getattr(vision, "_hsv_enabled", None),
            "hsv_erode_iterations": getattr(vision, "_hsv_erode_iterations", None),
        }
    for attr in ("_prev_angle", "_prev_dist", "_prev_offset", "_prev_offset_y", "_center_lock_prev_center"):
        if hasattr(vision, attr):
            setattr(vision, attr, None)
    return profile_key, applied_runtime, True


def _apply_cyan_smoothing_experiment(vision, runtime_snapshot=None):
    if not isinstance(vision, YoloBrickDetector):
        return runtime_snapshot, False
    target_alpha = float(CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX)
    if target_alpha <= 0.0:
        return runtime_snapshot, False

    current_alpha = None
    if isinstance(runtime_snapshot, dict):
        current_alpha = runtime_snapshot.get("smooth_alpha")
    if current_alpha is None:
        current_alpha = getattr(vision, "_smooth_alpha", None)
    try:
        current_alpha = float(current_alpha)
    except (TypeError, ValueError):
        current_alpha = None
    if current_alpha is not None and current_alpha <= target_alpha:
        return runtime_snapshot, False

    applied_runtime = None
    set_runtime_tuning = getattr(vision, "set_runtime_tuning", None)
    if callable(set_runtime_tuning):
        applied_runtime = set_runtime_tuning(smoothing_alpha=target_alpha)
    if not isinstance(applied_runtime, dict):
        vision._smooth_alpha = float(target_alpha)
        applied_runtime = dict(runtime_snapshot) if isinstance(runtime_snapshot, dict) else {}
        applied_runtime["smooth_alpha"] = float(target_alpha)
    return applied_runtime, True


def _stream_state_markerless_profile(app_state, fallback):
    return _stream_state_cyan_profile(app_state, fallback)


def _set_stream_state_markerless_profile(app_state, profile):
    _set_stream_state_cyan_profile(app_state, profile)


def _stream_state_markerless_visibility(app_state, fallback):
    return _stream_state_cyan_visibility(app_state, fallback)


def _set_stream_state_markerless_visibility(app_state, mode):
    _set_stream_state_cyan_visibility(app_state, mode)


def _set_detector_markerless_visibility(vision, mode):
    return _set_detector_cyan_visibility(vision, mode)


def _apply_markerless_visibility(app_state, vision, mode):
    return _apply_cyan_visibility(app_state, vision, mode)


def _apply_markerless_profile(app_state, vision, profile):
    return _apply_cyan_profile(app_state, vision, profile)


def _vision_mode_label(mode):
    if normalize_vision_mode(mode) == VISION_MODE_CYAN:
        return "Cyan Bricks"
    return "AruCo Markers"


def control_loop(app_state, vision_mode="aruco", yolo_model_path=None, show_startup_help=False):
    app_state.robot = Robot()
    active_vision_mode = normalize_vision_mode(vision_mode)
    active_cyan_profile = _stream_state_cyan_profile(app_state, _DEFAULT_CYAN_PROFILE)
    active_cyan_visibility = _stream_state_cyan_visibility(
        app_state,
        _DEFAULT_CYAN_VISIBILITY,
    )
    _set_stream_state_vision_mode(app_state, active_vision_mode)
    _set_stream_state_cyan_profile(app_state, active_cyan_profile)
    _set_stream_state_cyan_visibility(app_state, active_cyan_visibility)
    if _set_active_demos_dir_for_mode(app_state, active_vision_mode):
        refresh_world_model_from_demos(app_state, force=True, min_interval_s=0.0)
    # speed_optimize=False so we get the debug markers drawn on the frame
    app_state.vision = build_vision(active_vision_mode, yolo_model_path=yolo_model_path)
    profile_key, profile_settings, profile_applied = _apply_cyan_profile(
        app_state,
        app_state.vision,
        active_cyan_profile,
    )
    active_cyan_profile = profile_key
    profile_smoothing_experiment_applied = False
    if active_vision_mode == VISION_MODE_CYAN and profile_applied:
        profile_settings, profile_smoothing_experiment_applied = _apply_cyan_smoothing_experiment(
            app_state.vision,
            profile_settings,
        )
    visibility_key, visibility_applied = _apply_cyan_visibility(
        app_state,
        app_state.vision,
        active_cyan_visibility,
    )
    active_cyan_visibility = visibility_key
    log_line(
        f"[VISION] Active mode: {_vision_mode_label(active_vision_mode)} "
        f"(--vision {_vision_mode_cli_value(active_vision_mode)})"
    )
    if active_vision_mode == VISION_MODE_CYAN and profile_applied:
        log_line(
            "[VISION] Cyan config: "
            f"{cyan_profile_label(active_cyan_profile)} "
            f"({cyan_profile_runtime_summary(profile_settings)})"
        )
    if active_vision_mode == VISION_MODE_CYAN and profile_smoothing_experiment_applied:
        smooth_alpha = profile_settings.get("smooth_alpha") if isinstance(profile_settings, dict) else None
        if isinstance(smooth_alpha, (int, float)):
            log_line(f"[VISION] Cyan smoothing experiment: smooth={float(smooth_alpha):.2f}")
    if active_vision_mode == VISION_MODE_CYAN and visibility_applied:
        log_line(
            "[VISION] Cyan visibility: "
            f"{cyan_visibility_label(active_cyan_visibility)}"
        )
    if bool(show_startup_help):
        print_command_help(app_state)
    pending_stream_url = str(getattr(app_state, "pending_stream_started_url", "") or "").strip()
    if pending_stream_url:
        log_line(f"[VISION] Stream started at {ANSI_ORANGE_BRIGHT}{pending_stream_url}{ANSI_RESET}")
        app_state.pending_stream_started_url = None

    if app_state.stream_enabled:
        stream_t = threading.Thread(target=stream_refresh_loop, args=(app_state,), daemon=True)
        stream_t.start()

    cmd_t = threading.Thread(target=command_loop, args=(app_state,), daemon=True)
    cmd_t.start()
    
    dt = 1.0 / LOG_RATE_HZ
    
    while app_state.running:
        loop_start = time.time()
        requested_cyan_profile = _stream_state_cyan_profile(
            app_state,
            active_cyan_profile,
        )
        if requested_cyan_profile != active_cyan_profile:
            active_cyan_profile = requested_cyan_profile
            if active_vision_mode == VISION_MODE_CYAN and app_state.vision is not None:
                profile_key, profile_settings, profile_applied = _apply_cyan_profile(
                    app_state,
                    app_state.vision,
                    active_cyan_profile,
                )
                active_cyan_profile = profile_key
                if profile_applied:
                    profile_settings, smoothing_experiment_applied = _apply_cyan_smoothing_experiment(
                        app_state.vision,
                        profile_settings,
                    )
                    with app_state.lock:
                        app_state.brick_frame_buffer = []
                    log_line(
                        "[VISION] Cyan config: "
                        f"{cyan_profile_label(active_cyan_profile)} "
                        f"({cyan_profile_runtime_summary(profile_settings)})"
                    )
                    smooth_alpha = profile_settings.get("smooth_alpha") if isinstance(profile_settings, dict) else None
                    if smoothing_experiment_applied and isinstance(smooth_alpha, (int, float)):
                        log_line(f"[VISION] Cyan smoothing experiment: smooth={float(smooth_alpha):.2f}")
        requested_cyan_visibility = _stream_state_cyan_visibility(
            app_state,
            active_cyan_visibility,
        )
        if requested_cyan_visibility != active_cyan_visibility:
            active_cyan_visibility = requested_cyan_visibility
            if active_vision_mode == VISION_MODE_CYAN and app_state.vision is not None:
                visibility_key, visibility_applied = _apply_cyan_visibility(
                    app_state,
                    app_state.vision,
                    active_cyan_visibility,
                )
                active_cyan_visibility = visibility_key
                if visibility_applied:
                    log_line(
                        "[VISION] Cyan visibility: "
                        f"{cyan_visibility_label(active_cyan_visibility)}"
                    )
        requested_vision_mode = _stream_state_vision_mode(app_state, active_vision_mode)
        if requested_vision_mode != active_vision_mode:
            previous_vision = app_state.vision
            try:
                if previous_vision is not None:
                    previous_vision.close()
            except Exception:
                pass
            try:
                new_vision = build_vision(requested_vision_mode, yolo_model_path=yolo_model_path)
            except Exception as exc:
                restored_vision = None
                try:
                    restored_vision = build_vision(active_vision_mode, yolo_model_path=yolo_model_path)
                    _, restored_profile_runtime, restored_profile_applied = _apply_cyan_profile(
                        app_state,
                        restored_vision,
                        active_cyan_profile,
                    )
                    if active_vision_mode == VISION_MODE_CYAN and restored_profile_applied:
                        _apply_cyan_smoothing_experiment(restored_vision, restored_profile_runtime)
                    _apply_cyan_visibility(app_state, restored_vision, active_cyan_visibility)
                except Exception as restore_exc:
                    log_line(f"[VISION] Failed to restore {_vision_mode_label(active_vision_mode)}: {restore_exc}")
                app_state.vision = restored_vision
                _set_stream_state_vision_mode(app_state, active_vision_mode)
                log_line(
                    f"[VISION] Failed to switch to {_vision_mode_label(requested_vision_mode)}: {exc}"
                )
            else:
                app_state.vision = new_vision
                active_vision_mode = requested_vision_mode
                profile_key, profile_settings, profile_applied = _apply_cyan_profile(
                    app_state,
                    app_state.vision,
                    active_cyan_profile,
                )
                active_cyan_profile = profile_key
                profile_smoothing_experiment_applied = False
                if active_vision_mode == VISION_MODE_CYAN and profile_applied:
                    profile_settings, profile_smoothing_experiment_applied = _apply_cyan_smoothing_experiment(
                        app_state.vision,
                        profile_settings,
                    )
                visibility_key, visibility_applied = _apply_cyan_visibility(
                    app_state,
                    app_state.vision,
                    active_cyan_visibility,
                )
                active_cyan_visibility = visibility_key
                with app_state.lock:
                    app_state.brick_frame_buffer = []
                _set_stream_state_vision_mode(app_state, active_vision_mode)
                demos_switched = _set_active_demos_dir_for_mode(app_state, active_vision_mode)
                if demos_switched:
                    refresh_world_model_from_demos(app_state, force=True, min_interval_s=0.0)
                log_line(
                    f"[VISION] Switched to {_vision_mode_label(active_vision_mode)} "
                    f"(--vision {_vision_mode_cli_value(active_vision_mode)})."
                )
                if active_vision_mode == VISION_MODE_CYAN and profile_applied:
                    log_line(
                        "[VISION] Cyan config: "
                        f"{cyan_profile_label(active_cyan_profile)} "
                        f"({cyan_profile_runtime_summary(profile_settings)})"
                    )
                if active_vision_mode == VISION_MODE_CYAN and profile_smoothing_experiment_applied:
                    smooth_alpha = profile_settings.get("smooth_alpha") if isinstance(profile_settings, dict) else None
                    if isinstance(smooth_alpha, (int, float)):
                        log_line(f"[VISION] Cyan smoothing experiment: smooth={float(smooth_alpha):.2f}")
                if active_vision_mode == VISION_MODE_CYAN and visibility_applied:
                    log_line(
                        "[VISION] Cyan visibility: "
                        f"{cyan_visibility_label(active_cyan_visibility)}"
                    )

        refresh_world_model_from_demos(app_state)
        auto_obj = None
        auto_continue_msg = None
        auto_continue_countdown_msg = None
        with app_state.lock:
            due_auto_obj = _pop_due_auto_continue_locked(app_state)
            if due_auto_obj is not None:
                app_state.auto_request = due_auto_obj
                step_key_due = normalize_step_label(getattr(due_auto_obj, "value", due_auto_obj))
                if step_key_due:
                    app_state.stream_state["success_gate_step"] = step_key_due
                due_code = step_code_for_obj(due_auto_obj)
                due_name = str(getattr(due_auto_obj, "value", due_auto_obj)).lower()
                auto_continue_msg = (
                    f"[AUTO] Auto-continue after {float(AUTO_CONTINUE_WAIT_S):.1f}s: "
                    f"queued step {due_code} ({due_name})."
                )
            else:
                auto_continue_countdown_msg = _auto_continue_countdown_message_locked(app_state)
            if app_state.auto_request and not app_state.auto_running:
                auto_obj = app_state.auto_request
                app_state.auto_request = None
                app_state.auto_running = True
                try:
                    app_state.auto_cancel_event.clear()
                except Exception:
                    pass
                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None

        if auto_continue_msg:
            log_line(auto_continue_msg)
        elif auto_continue_countdown_msg:
            log_line(auto_continue_countdown_msg)

        if auto_obj:
            if app_state.vision is None:
                log_line("[VISION] Auto-step skipped: no active vision backend.")
            else:
                run_auto_step(app_state, auto_obj)
            with app_state.lock:
                app_state.auto_running = False
                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None
                app_state.active_pwm_override = None
            continue
            
        # 1. Vision
        vision = app_state.vision
        if vision is None:
            time.sleep(dt)
            continue
        with app_state.vision_io_lock:
            found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
        
        # 2. Telemetry Update
        study = None
        with app_state.lock:
            app_state.brick_frame_buffer.append({
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
            })
            if len(app_state.brick_frame_buffer) > BRICK_STUDY_FRAMES:
                app_state.brick_frame_buffer.pop(0)

            if isinstance(vision, YoloBrickDetector):
                study = evaluate_cyan_frames(app_state.brick_frame_buffer)
            else:
                study = evaluate_brick_frames(app_state.brick_frame_buffer)
            if study and study.get("reset"):
                app_state.brick_frame_buffer = []
        snapshot = _study_snapshot_or_raw(
            study,
            found=found,
            angle=angle,
            dist=dist,
            offset_x=offset_x,
            conf=conf,
            cam_h=cam_h,
            brick_above=brick_above,
            brick_below=brick_below,
        )
        app_state.world.update_vision(
            snapshot["found"],
            snapshot["dist"],
            snapshot["angle"],
            snapshot["conf"],
            snapshot["offset_x"],
            snapshot["cam_h"],
            snapshot["brick_above"],
            snapshot["brick_below"],
        )
        refresh_brick_telemetry(app_state, read_vision=False)
        
        # Track Motion
        with app_state.lock:
            cmd = app_state.active_command
            speed = app_state.active_speed
            score = app_state.active_speed_score


        # Manual pulses are one-shot and are integrated/logged at send time in
        # command_loop() using the actual returned duration/PWM.
        if not (cmd and speed > 0) and _demo_logging_active(app_state):
            if app_state.logger is not None and not getattr(app_state, "logger_closed", True):
                app_state.logger.log_state(app_state.world)
        
        # 5. Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def command_loop(app_state):
    dt = 1.0 / max(COMMAND_RATE_HZ, 1e-3)
    while app_state.running:
        loop_start = time.time()
        with app_state.lock:
            if app_state.auto_running or app_state.auto_request is not None:
                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None
                app_state.active_pwm_override = None
            if time.time() - app_state.last_key_time > HEARTBEAT_TIMEOUT:
                app_state.active_command = None
                app_state.active_hotkey = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
                app_state.active_duration_ms = None
                app_state.active_pwm_override = None

            cmd = app_state.active_command
            hotkey = app_state.active_hotkey
            score = app_state.active_speed_score
            duration_override_ms = app_state.active_duration_ms
            pwm_override = app_state.active_pwm_override

        score_used = score
        if cmd and score is not None:
            quantized_speed, score_used = quantize_speed(cmd, score=score)
            speed, _ = app_state.robot.normalize_speed(cmd, quantized_speed)
            with app_state.lock:
                app_state.active_speed = speed
                if score_used is not None:
                    app_state.active_speed_score = score_used
        else:
            speed = 0.0
            with app_state.lock:
                app_state.active_speed = 0.0

        if cmd and score_used is not None and (
            speed > 0 or score_used == telemetry_robot_module.SPEED_SCORE_MIN
        ):
            ease_for_hotkey = hotkey_uses_ease_in_out(hotkey)
            send_result = None
            if pwm_override is not None:
                try:
                    pwm_override_val = telemetry_robot_module.clamp_pwm(int(pwm_override))
                except (TypeError, ValueError):
                    pwm_override_val = None
                if pwm_override_val is not None and pwm_override_val > 0:
                    try:
                        power_for_pwm = telemetry_robot_module.pwm_to_power(pwm_override_val)
                    except Exception:
                        power_for_pwm = None
                    if power_for_pwm is None:
                        power_for_pwm = float(speed)
                    if duration_override_ms is not None:
                        try:
                            duration_used_ms = max(1, int(round(float(duration_override_ms))))
                        except (TypeError, ValueError):
                            duration_used_ms = None
                    else:
                        try:
                            _, _, _, duration_used_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score_used)
                        except Exception:
                            duration_used_ms = None
                    if duration_used_ms is None or duration_used_ms <= 0:
                        duration_used_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 250) or 250)
                    send_result = send_robot_command_pwm(
                        app_state.robot,
                        app_state.world,
                        app_state.world.step_state,
                        cmd,
                        float(power_for_pwm),
                        int(pwm_override_val),
                        int(duration_used_ms),
                        speed_score=score_used,
                        auto_mode=False,
                        ease_in_out_enabled=ease_for_hotkey,
                    )
                else:
                    send_result = send_robot_command(
                        app_state.robot,
                        app_state.world,
                        app_state.world.step_state,
                        cmd,
                        speed,
                        speed_score=score_used,
                        duration_override_ms=duration_override_ms,
                        ease_in_out_enabled=ease_for_hotkey,
                    )
            else:
                send_result = send_robot_command(
                    app_state.robot,
                    app_state.world,
                    app_state.world.step_state,
                    cmd,
                    speed,
                    speed_score=score_used,
                    duration_override_ms=duration_override_ms,
                    ease_in_out_enabled=ease_for_hotkey,
                )
            _apply_manual_motion_telemetry_from_send(
                app_state,
                cmd=cmd,
                speed=speed,
                score=score_used,
                send_result=send_result,
            )
            if hotkey:
                _record_manual_hotkey_press(
                    app_state,
                    hotkey=hotkey,
                    cmd=cmd,
                    score=score_used,
                )
            if (
                hotkey
                and isinstance(send_result, dict)
                and (pwm_override is not None or duration_override_ms is not None)
            ):
                try:
                    pwm_sent = int(round(float(send_result.get("pwm"))))
                except (TypeError, ValueError):
                    pwm_sent = None
                try:
                    dur_sent = int(round(float(send_result.get("duration_ms"))))
                except (TypeError, ValueError):
                    dur_sent = None
                cmd_sent = str(send_result.get("cmd_sent") or cmd).strip().lower()
                wire = getattr(app_state.robot, "last_command", None)
                wire_text = str(wire).strip() if wire else ""
                if not wire_text:
                    pieces = [str(cmd_sent)]
                    if pwm_sent is not None:
                        pieces.append(str(int(pwm_sent)))
                    if dur_sent is not None:
                        pieces.append(str(int(dur_sent)))
                    wire_text = " ".join([p for p in pieces if p])
                log_line(
                    f"[HOTKEY SEND] {wire_text}"
                )
            refresh_brick_telemetry(app_state, read_vision=False)
            # One-shot hotkey behavior: each keypress sends one timed pulse.
            # This prevents command-loop replays when a key remains active.
            with app_state.lock:
                if app_state.active_command == cmd and app_state.active_hotkey == hotkey:
                    app_state.active_command = None
                    app_state.active_hotkey = None
                    app_state.active_speed = 0.0
                    app_state.active_speed_score = None
                    app_state.active_duration_ms = None
                    app_state.active_pwm_override = None

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brick-vision",
        "--vision",
        dest="brick_vision",
        default=_DEFAULT_VISION_MODE,
        help="Brick vision model to use (aruco/cyan)",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to YOLO ONNX model (used with --brick-vision cyan)",
    )
    parser.add_argument(
        "--cyan-profile",
        dest="cyan_profile",
        choices=[value for value, _label in CYAN_PROFILE_OPTIONS],
        default=_DEFAULT_CYAN_PROFILE,
        help="Cyan tuning profile (applies when vision mode is cyan)",
    )
    parser.add_argument(
        "--markerless-profile",
        dest="cyan_profile",
        choices=[value for value, _label in CYAN_PROFILE_OPTIONS],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stream", dest="stream", action="store_true",
                        help="Enable livestreaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable livestreaming")
    parser.set_defaults(stream=True)
    args = parser.parse_args()
    brick_vision_raw = str(args.brick_vision or "").strip().lower()
    if brick_vision_raw and brick_vision_raw not in _VISION_MODE_ALIASES:
        parser.error("--brick-vision must be one of: aruco, cyan")
    vision_mode = normalize_vision_mode(brick_vision_raw, fallback=_DEFAULT_VISION_MODE)
    cyan_profile = normalize_cyan_profile(
        args.cyan_profile,
        fallback=_DEFAULT_CYAN_PROFILE,
    )

    startup_demos_dir = _demos_dir_for_vision_mode(vision_mode)
    startup_demos_dir.mkdir(parents=True, exist_ok=True)
    logs = load_demo_logs(startup_demos_dir)
    if logs:
        update_process_model_from_demos(
            logs,
            PROCESS_MODEL_FILE,
        )

    state = AppState(vision_mode=vision_mode)
    
    # User Request: Initialize with default expectations
    setattr(state.world, "wall_height_bricks", 1)
    setattr(state.world, "brick_supply_height_bricks", 5)
    import telemetry_brick
    setattr(state.world, "wall_height_mm", float(telemetry_brick.brick_count_to_height_mm(1)))
    setattr(state.world, "brick_supply_height_mm", float(telemetry_brick.brick_count_to_height_mm(5)))
    import helper_xyz_coords
    helper_xyz_coords.sync_from_world(state.world, reason="init_default_heights")
    print(format_headline(f"[EXCEPTION] Defaulting wall_height_bricks=1 and brick_supply_height_bricks=5 start states.", COLOR_MAGENTA_BRIGHT))

    _set_stream_state_vision_mode(state, vision_mode)
    _set_stream_state_cyan_profile(state, cyan_profile)
    _set_stream_state_cyan_visibility(state, _DEFAULT_CYAN_VISIBILITY)
    _set_stream_state_success_gate_step(state, state.world.step_state.value)
    _set_active_demos_dir_for_mode(state, vision_mode)
    refresh_world_model_from_demos(
        state,
        force=True,
    )
    log_align_curve_lookup_table()
    
    # Keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    stream_server = None
    # Web Stream thread (optional)
    if args.stream:
        try:
            stream_server, url = start_stream_server(
                state.stream_state,
                title="Keyboard Training Livestream",
                header="",
                footer=stream_footer_html(),
                host=STREAM_HOST,
                port=STREAM_PORT,
                fps=STREAM_FPS,
                jpeg_quality=STREAM_JPEG_QUALITY,
                img_width=STREAM_IMG_WIDTH,
                vision_mode_options=STREAM_VISION_MODE_OPTIONS,
                cyan_profile_options=CYAN_PROFILE_OPTIONS,
                cyan_visibility_options=CYAN_VISIBILITY_OPTIONS,
                success_gate_step_options=STREAM_SUCCESS_GATE_STEP_OPTIONS,
                xyz_workspace_getter=lambda: getattr(state.world, "_xyz_workspace", None),
            )
        except Exception as exc:
            state.stream_enabled = False
            log_line(f"[VISION] Stream failed to start: {exc}")
        else:
            state.stream_enabled = True
            actual_port = getattr(stream_server, "port", STREAM_PORT)
            if actual_port != STREAM_PORT:
                log_line(f"[VISION] Stream port {STREAM_PORT} busy; using {actual_port}")
            state.pending_stream_started_url = str(url)
            log_line(f"[VISION] Livestream URL: {url}")
            _open_stream_in_chrome(url)
    else:
        state.stream_enabled = False
        log_line("[VISION] Stream disabled")
    
    try:
        control_loop(
            state,
            vision_mode=vision_mode,
            yolo_model_path=args.yolo_model,
            show_startup_help=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if stream_server is not None:
            try:
                stream_server.stop()
            except Exception:
                pass
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        close_log(state, marker=None)
        try:
            log_session_summary(state)
        except Exception as exc:
            log_line(f"[SESSION SUMMARY] Failed to log summary ({exc})")
        log_line("Shutdown complete.")
