import argparse
import json
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
from brick_detector_yolo import BrickDetector as YoloBrickDetector
from helper_manual_config import load_manual_training_config
from helper_micro_speed_adjust import micro_adjust_speed_score
from helper_align_profile import load_align_profile, inject_align_profile_into_learned_rules
from helper_mini_x_axis_calibrate import (
    MINI_TRIALS_DEFAULT as MINI_X_AXIS_TRIALS_DEFAULT,
    RUN_LOG_FILE_DEFAULT as MINI_X_AXIS_LOG_DEFAULT,
    run_mini_x_axis_calibration,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import (
    WorldModel,
    TelemetryLogger,
    MotionEvent,
    StepState,
    draw_telemetry_overlay,
    manual_key_action,
    HOTKEY_SPEED_SCORES,
    step_sequence,
    quantize_speed,
)
from telemetry_process import (
    build_motion_sequence,
    consume_auto_step_action_stats,
    compute_stream_gate_summary,
    merge_motion_steps,
    nominal_actions_from_events,
    reset_auto_step_action_stats,
    send_robot_command,
    step_requires_start_gates,
)
from autobuild import (
    collect_segments,
    CONTROL_DT,
    format_gate_lines,
    load_process_model,
    replay_segment,
    refresh_autobuild_config,
    select_demo_segment,
    update_world_from_vision,
    update_process_model_from_demos,
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
DEMOS_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
DEMO_STEPS = step_sequence()
CTRL_HELP_LINE = (
    "[CTRL] Drive: W/S 50%, R/F 1%, T/G 100%. Turn: A/D 50%, Q/E 1%, Z/C 100%. "
    "Lift: U/L 50%. F action, ':' command, m auto, y edit hotkey vars, 1 trash log, Q quit"
)
ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
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

VISION_MODE_ARUCO = "aruco"
VISION_MODE_MARKERLESS = "yolo"
STREAM_VISION_MODE_OPTIONS = [
    (VISION_MODE_ARUCO, "AruCo Markers"),
    (VISION_MODE_MARKERLESS, "Markerless"),
]
MARKERLESS_PROFILE_BALANCED = "balanced"
MARKERLESS_PROFILE_OPTIONS = [
    ("balanced", "Balanced (0.15 conf / 0.30 smooth)"),
    ("sensitive", "Sensitive (0.10 conf / 0.20 smooth)"),
    ("aggressive", "Aggressive (0.05 conf / 0.15 smooth)"),
    ("rescue", "Rescue (0.001 conf / 0.10 smooth)"),
    ("stable", "Stable (0.20 conf / 0.40 smooth)"),
    ("legacy", "Legacy (0.25 conf / 0.60 smooth)"),
]
MARKERLESS_PROFILE_PRESETS = {
    "balanced": {
        "conf_threshold": 0.15,
        "smooth_alpha": 0.30,
        "nms_threshold": 0.45,
    },
    "sensitive": {
        "conf_threshold": 0.10,
        "smooth_alpha": 0.20,
        "nms_threshold": 0.45,
    },
    "aggressive": {
        "conf_threshold": 0.05,
        "smooth_alpha": 0.15,
        "nms_threshold": 0.45,
    },
    "rescue": {
        "conf_threshold": 0.001,
        "smooth_alpha": 0.10,
        "nms_threshold": 0.45,
    },
    "stable": {
        "conf_threshold": 0.20,
        "smooth_alpha": 0.40,
        "nms_threshold": 0.45,
    },
    "legacy": {
        "conf_threshold": 0.25,
        "smooth_alpha": 0.60,
        "nms_threshold": 0.45,
    },
}
_VISION_MODE_ALIASES = {
    "aruco": VISION_MODE_ARUCO,
    "yolo": VISION_MODE_MARKERLESS,
    "markerless": VISION_MODE_MARKERLESS,
}


def normalize_vision_mode(value, fallback=VISION_MODE_ARUCO):
    key = str(value or "").strip().lower()
    mode = _VISION_MODE_ALIASES.get(key)
    if mode:
        return mode
    fallback_key = str(fallback or "").strip().lower()
    fallback_mode = _VISION_MODE_ALIASES.get(fallback_key)
    if fallback_mode:
        return fallback_mode
    return VISION_MODE_ARUCO


_DEFAULT_VISION_MODE = normalize_vision_mode(_MANUAL_CONFIG.get("brick_vision", VISION_MODE_ARUCO))
_DEFAULT_MARKERLESS_PROFILE = str(
    _MANUAL_CONFIG.get("markerless_profile", MARKERLESS_PROFILE_BALANCED)
).strip().lower()
if _DEFAULT_MARKERLESS_PROFILE not in MARKERLESS_PROFILE_PRESETS:
    _DEFAULT_MARKERLESS_PROFILE = MARKERLESS_PROFILE_BALANCED

AUTO_MINI_X_AXIS_ENABLED = bool(_MANUAL_CONFIG.get("auto_mini_x_axis_enabled", True))
AUTO_MINI_X_AXIS_STEPS = {"ALIGN_BRICK", "POSITION_BRICK"}
try:
    AUTO_MINI_X_AXIS_TRIALS = int(_MANUAL_CONFIG.get("auto_mini_x_axis_trials", MINI_X_AXIS_TRIALS_DEFAULT))
except (TypeError, ValueError):
    AUTO_MINI_X_AXIS_TRIALS = int(MINI_X_AXIS_TRIALS_DEFAULT)
AUTO_MINI_X_AXIS_TRIALS = max(1, int(AUTO_MINI_X_AXIS_TRIALS))
try:
    AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS = float(_MANUAL_CONFIG.get("auto_mini_x_axis_min_interval_hours", 12.0))
except (TypeError, ValueError):
    AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS = 12.0
AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS = max(0.0, float(AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS))
AUTO_MINI_X_AXIS_LOG = Path(_MANUAL_CONFIG.get("auto_mini_x_axis_log", str(MINI_X_AXIS_LOG_DEFAULT)))


def _vision_mode_cli_value(mode):
    mode_norm = normalize_vision_mode(mode)
    if mode_norm == VISION_MODE_MARKERLESS:
        return VISION_MODE_MARKERLESS
    return VISION_MODE_ARUCO


def normalize_markerless_profile(value, fallback=MARKERLESS_PROFILE_BALANCED):
    key = str(value or "").strip().lower()
    if key in MARKERLESS_PROFILE_PRESETS:
        return key
    fallback_key = str(fallback or "").strip().lower()
    if fallback_key in MARKERLESS_PROFILE_PRESETS:
        return fallback_key
    return MARKERLESS_PROFILE_BALANCED


def markerless_profile_settings(profile):
    profile_key = normalize_markerless_profile(profile, fallback=_DEFAULT_MARKERLESS_PROFILE)
    settings = MARKERLESS_PROFILE_PRESETS.get(profile_key) or MARKERLESS_PROFILE_PRESETS[MARKERLESS_PROFILE_BALANCED]
    return profile_key, settings


def markerless_profile_label(profile):
    profile_key = normalize_markerless_profile(profile, fallback=_DEFAULT_MARKERLESS_PROFILE)
    for value, label in MARKERLESS_PROFILE_OPTIONS:
        if value == profile_key:
            return label
    return profile_key

STEP_CODES = {str(idx + 1): obj for idx, obj in enumerate(DEMO_STEPS)}

STEP_NAMES = {obj.value.lower(): obj for obj in DEMO_STEPS}
STEP_NAMES["wall"] = StepState.FIND_WALL
STEP_NAMES["find"] = StepState.FIND_BRICK
STEP_NAMES["align"] = StepState.ALIGN_BRICK
STEP_NAMES["carry"] = StepState.FIND_WALL2
STEP_NAMES["wall2"] = StepState.FIND_WALL2
STEP_NAMES["position"] = StepState.POSITION_BRICK

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

def step_label(obj_enum):
    return obj_enum.value

def log_line(message):
    print(str(message).strip(), flush=True)


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


def _update_speed_model_vars_for_hotkey(cmd, score, pwm, power, duration_ms):
    score_key = telemetry_robot_module.normalize_speed_score(score)
    pwm_clamped = telemetry_robot_module.clamp_pwm(pwm)
    try:
        power_clamped = max(0.0, min(1.0, float(power)))
    except (TypeError, ValueError):
        power_clamped = telemetry_robot_module.pwm_to_power(pwm_clamped) or 0.0
    duration_clamped = max(1, int(round(float(duration_ms))))

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

        model_path.write_text(json.dumps(model, indent=2) + "\n")
    except OSError as exc:
        return False, f"Failed to persist world model: {exc}"

    return True, None


def maybe_prompt_hotkey_var_update(app_state, cmd, score, *, enter_edit=False):
    try:
        _, pwm, score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
        power = telemetry_robot_module.pwm_to_power(pwm)
    except Exception as exc:
        log_line(f"[HOTKEY] Failed to read speed vars: {exc}")
        return None

    if power is None:
        power = 0.0
    duration_used_ms = _manual_hotkey_duration_ms(app_state, cmd, duration_model_ms)
    if not enter_edit:
        log_line(
            f"[HOTKEY] {str(cmd).upper()} {int(score_used)}% "
            f"(pwm={int(pwm)}, power={float(power):.3f}, {int(duration_used_ms)}ms) "
            f"- press y to edit vars"
        )
        return int(score_used)

    log_line(
        f"[HOTKEY] Editing {str(cmd).upper()} {int(score_used)}% "
        f"(current pwm={int(pwm)}, power={float(power):.3f}, {int(duration_used_ms)}ms)"
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
        ok, err = _update_speed_model_vars_for_hotkey(cmd, score_used, pwm_new, power_new, duration_new)
        if not ok:
            log_line(f"[HOTKEY] Update failed: {err}")
            return
        log_line(
            f"[HOTKEY] Updated world model for {str(cmd).upper()} {int(score_used)}% "
            f"-> (pwm={int(pwm_new)}, power={float(power_new):.3f}, {int(duration_new)}ms)"
        )
        return int(score_used)
    return int(score_used)


def format_step_codes_line():
    parts = [f"{idx + 1}={obj.value.lower()}" for idx, obj in enumerate(DEMO_STEPS)]
    return "[CMD] Step codes: " + ", ".join(parts)


def step_code_for_obj(obj_enum):
    for code, obj in STEP_CODES.items():
        if obj == obj_enum:
            return code
    return normalize_step_label(getattr(obj_enum, "value", obj_enum))


def stream_footer_html():
    lines = []
    keys_line = format_hotkey_speeds()
    if keys_line:
        lines.append(keys_line)
    lines.append(format_step_codes_line())
    lines.append(CTRL_HELP_LINE)
    return "<br>".join(lines)


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

def open_new_log(app_state):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = app_state.demos_dir / f"kbd_{timestamp}.json"
    log_line(f"[SESSION] Recording Keyboard Demo to: {log_path}")
    app_state.logger = TelemetryLogger(log_path)
    app_state.logger.enabled = False  # Wait for first attempt marker
    app_state.logger_closed = False
    app_state.log_path = log_path
    app_state.world.run_id = f"run_{timestamp}"
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
        prune_log_file(app_state.log_path, delete_if_empty=True)

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
    if log_path and log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            return log_path, False
    return log_path, True

def run_auto_step(app_state, obj_enum):
    step_key = normalize_step_label(obj_enum.value)
    log_line(f"[AUTO] Attempting {step_key}...")

    if step_key in AUTO_MINI_X_AXIS_STEPS:
        if not AUTO_MINI_X_AXIS_ENABLED:
            log_line(f"[AUTO] {step_key} mini x-axis calibration disabled by config.")
        elif app_state.robot is None or app_state.vision is None:
            log_line(f"[AUTO] {step_key} mini x-axis calibration skipped (robot/vision unavailable).")
        else:
            mini_result = run_mini_x_axis_calibration(
                robot=app_state.robot,
                vision=app_state.vision,
                step_key=step_key,
                trials=AUTO_MINI_X_AXIS_TRIALS,
                log_path=AUTO_MINI_X_AXIS_LOG,
                min_interval_hours=AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS,
                log_fn=log_line,
            )
            if not bool(mini_result.get("ok")):
                log_line(
                    f"[AUTO] {step_key} mini x-axis calibration failed; continuing with demo replay "
                    f"({mini_result.get('error')})."
                )
            with app_state.lock:
                app_state.brick_frame_buffer = []

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

    cfg = process_rules.get(step_key, {}) if process_rules else {}
    nominal_only = bool(cfg.get("nominalDemosOnly"))
    segment, seg_type = select_demo_segment(segments_by_obj, step_key, nominal_only)
    if not segment:
        log_line(f"[AUTO] No demo segment found for {step_key}.")
        return False

    log_line(f"[AUTO] {step_key} demo={seg_type}")
    events = segment.get("events") or []
    actions = nominal_actions_from_events(events)
    steps = merge_motion_steps(build_motion_sequence(events))
    if actions:
        log_line(f"[AUTO] {step_key} demo acts:")
        for idx, action in enumerate(actions, start=1):
            cmd = (action.get("cmd") or "-").upper()
            score = action.get("speed_score")
            duration_s = float(action.get("duration_s") or 0.0)
            score_text = f"score={int(score)}" if score is not None else "score=?"
            log_line(f"[AUTO]   {idx}. {cmd} {score_text} dur={duration_s:.2f}s")
    else:
        log_line(f"[AUTO] {step_key} demo acts: none")
    if steps:
        log_line(f"[AUTO] {step_key} parsed steps:")
        for idx, step in enumerate(steps, start=1):
            cmd = (step.cmd or "-").upper() if hasattr(step, "cmd") else "-"
            score = getattr(step, "speed_score", None)
            duration_s = float(getattr(step, "duration_s", 0.0) or 0.0)
            score_text = f"score={int(score)}" if score is not None else "score=?"
            log_line(f"[AUTO]   {idx}. {cmd} {score_text} dur={duration_s:.2f}s")
    else:
        log_line(f"[AUTO] {step_key} parsed steps: none")

    try:
        import telemetry_wall as telemetry_wall_module
    except Exception:
        telemetry_wall_module = None

    start_desc, success_desc = format_gate_lines(cfg)
    skip_start_gates = not step_requires_start_gates(step_key, process_rules)
    if skip_start_gates:
        status_msg = f"[AUTO] {step_key} start gates: SKIPPED (find/exit step)"
        if start_desc and start_desc != "none":
            status_msg += f" (cfg ignored: {start_desc})"
        log_line(status_msg)
    else:
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

        def _gate_ok(check):
            return bool(getattr(check, "ok", False))

        def _gate_reasons(check):
            raw = getattr(check, "reasons", None)
            if not raw:
                return []
            if isinstance(raw, list):
                return [str(r) for r in raw if r]
            return [str(raw)]

        start_ok = _gate_ok(brick_check) and _gate_ok(wall_check) and _gate_ok(robot_check)
        reason_parts = []
        for label, check in (("brick", brick_check), ("wall", wall_check), ("robot", robot_check)):
            if check is None or _gate_ok(check):
                continue
            reasons = _gate_reasons(check)
            if reasons:
                reason_parts.append(f"{label}: {', '.join(reasons)}")
            else:
                reason_parts.append(f"{label}: blocked")

        status = "OK" if start_ok else "BLOCKED"
        status_msg = f"[AUTO] {step_key} start gates: {status}"
        if reason_parts:
            status_msg += " — " + "; ".join(reason_parts)
        if start_desc and start_desc != "none":
            status_msg += f" (cfg: {start_desc})"
        log_line(status_msg)

    quiet_align = False
    log_line(f"[AUTO] {step_key} demo={seg_type} success gates: {success_desc}")
    if app_state.robot:
        app_state.robot.stop()
    reset_auto_step_action_stats(app_state.world, step_key)
    observer = make_auto_observer(app_state)
    confirm_callback = None
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
    if ok:
        reason_text = reason.replace("_", " ") if isinstance(reason, str) else "success criteria met"
        log_line(f"[AUTO] {step_key} complete — {reason_text}.")
        act_stats = consume_auto_step_action_stats(app_state.world, step_key)
        acts_total = int((act_stats or {}).get("total") or 0)
        acts_closer = int((act_stats or {}).get("closer") or 0)
        acts_backward = int((act_stats or {}).get("backward") or 0)
        acts_unchanged = int((act_stats or {}).get("unchanged") or 0)
        acts_unknown = int((act_stats or {}).get("unknown") or 0)
        step_code = step_code_for_obj(obj_enum)
        log_line(
            f"ACHIEVED step {step_code} in {acts_total} acts. "
            f"{acts_closer} got us closer, {acts_backward} took us backwards, "
            f"{acts_unchanged} were unchanged, and {acts_unknown} were unclear."
        )
    else:
        log_line(f"[AUTO] {step_key} failed ({reason}).")
    return ok


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


def refresh_world_model_from_demos(app_state, force=False, min_interval_s=0.5):
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
    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
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
        log_line(
            "[ALIGN_PROFILE] Applied calibrate-align profile "
            f"(run_id={run_id}, turn_scale={turn_scale}, dist_scale={dist_scale}, max_speed={cap})."
        )
    app_state.config_mtime = latest_mtime

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
        "conf": mean([f["conf"] for f in frames]),
        "cam_h": mean([f["cam_h"] for f in frames]),
        "brick_above": majority([f["brick_above"] for f in frames]),
        "brick_below": majority([f["brick_below"] for f in frames]),
    }


def _markerless_average_frames(frames):
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
        "conf": mean([frame["conf"] for frame in found_frames]),
        "cam_h": mean([frame["cam_h"] for frame in found_frames]),
        "brick_above": majority([frame["brick_above"] for frame in found_frames]),
        "brick_below": majority([frame["brick_below"] for frame in found_frames]),
    }


def evaluate_markerless_frames(frames):
    if len(frames) < BRICK_STUDY_FRAMES:
        return None
    found_count = sum(1 for frame in frames if frame.get("found"))
    if found_count <= 0:
        return {
            "confidence": 0,
            "message": "0% confidence (no markerless detections in the latest 4 frames)",
            "average": None,
            "reset": True,
        }
    average = _markerless_average_frames(frames)
    confidence = int(round(100 * found_count / BRICK_STUDY_FRAMES))
    return {
        "confidence": confidence,
        "message": (
            f"{confidence}% confidence ({found_count}/{BRICK_STUDY_FRAMES} "
            "markerless frames found a brick)"
        ),
        "average": average,
        "reset": True,
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


def markerless_stream_debug_lines(app_state, vision):
    active_mode = _stream_state_vision_mode(app_state, _DEFAULT_VISION_MODE)
    if normalize_vision_mode(active_mode) != VISION_MODE_MARKERLESS:
        return None

    profile = _stream_state_markerless_profile(app_state, _DEFAULT_MARKERLESS_PROFILE)
    profile_text = markerless_profile_label(profile)
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


def update_stream_frame(app_state):
    vision = app_state.vision
    if vision is None or vision.current_frame is None:
        return
    frame = vision.current_frame.copy()
    with app_state.lock:
        step_suggestions = []
        if app_state.step_suggestions:
            step_suggestions.extend(app_state.step_suggestions)
        active = True
        gate_summary, analytics = compute_stream_gate_summary(
            app_state.world,
            app_state.world.step_state.value,
            active=active,
        )
        gate_checker_summary = format_gatecheck_stream_lines(
            app_state.world,
            app_state.world.step_state.value,
        )
        if analytics:
            app_state.brick_highlight_metric = analytics.get("highlight_metric")
        markerless_lines = markerless_stream_debug_lines(app_state, vision)
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
            brick_extra_lines=markerless_lines,
        )
        app_state.current_frame = frame
    if app_state.stream_state:
        with app_state.stream_state["lock"]:
            app_state.stream_state["frame"] = frame
            app_state.stream_state["text_lines"] = stream_lines


def make_auto_observer(app_state):
    def _observer(stage, world, vision, cmd, speed, reason):
        # replay_segment already refreshed world from vision before observer callbacks.
        # Avoid a second camera read here; it can stall stream updates on some cameras.
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
    log_line("[CMD] Auto mode: press step number to auto-run.")
    log_line("[CMD] Auto-run: press step number.")
    log_line("[CMD] Hotkey tuning: press a movement hotkey, then press y to edit that hotkey's vars.")
    log_line("[CMD] End attempt: press ':' to finish and return to the command prompt.")
    log_line("[CMD] Step codes:")
    for idx, obj in enumerate(DEMO_STEPS):
        log_line(f"[CMD]   {idx + 1}={obj.value.lower()}")


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
        "P", "L",
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
    app_state.world.attempt_status = ATTEMPT_STATUS[attempt_type]
    app_state.world.recording_active = True
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

    if attempt_type == "SUCCESS":
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
    return ok, msg, ended_info, ended_close

class AppState:
    def __init__(self):
        self.running = True
        self.active_command = None
        self.active_speed = 0.0
        self.active_speed_score = None
        self.last_key_time = 0
        self.micro_speed_state = {}
        self.hotkey_edit_target = None
        
        # Job Status
        self.job_success = False
        self.job_success_timer = 0
        self.job_start = False
        self.job_start_timer = 0
        self.job_abort = False
        self.job_abort_timer = 0
        
        # Job Status
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.step_open = False
        self.open_step = None
        self.active_attempt = None
        
        # Session Setup
        self.demos_dir = DEMOS_DIR
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        self.world = WorldModel()
        self.logger = None
        self.logger_closed = True
        self.log_path = None
        open_new_log(self)
        
        # ID Init
        self.world.step_state = StepState.ALIGN_BRICK  # Default to ALIGN_BRICK for gate testing

        self.vision = None
        self.robot = None

        self.auto_prompt = False
        self.auto_request = None
        self.auto_running = False
        self.auto_confirm_needed = False
        self.auto_confirm_event = threading.Event()
        self.last_enter_time = 0.0

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
            "vision_mode": _DEFAULT_VISION_MODE,
            "markerless_profile": _DEFAULT_MARKERLESS_PROFILE,
        }
        self.stream_enabled = False
        # Allow telemetry_process helpers to push frames directly during auto-run pauses.
        self.world._stream_state = self.stream_state

        self.config_mtime = 0
        self.last_config_check = 0

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
    while app_state.running:
        ch = getch()
        if not ch:
            continue
        ch_lower = ch.lower()
        with app_state.lock:
            auto_prompt = app_state.auto_prompt
            auto_confirm_needed = app_state.auto_confirm_needed

        messages = []
        pending_hotkey_action = None
        if ch == 'Q':
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.running = False
            messages.append("Stopping manual recording...")
        elif ch in ('\n', '\r'):
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.last_enter_time = time.time()
                if auto_confirm_needed:
                    app_state.auto_confirm_needed = False
                    app_state.auto_confirm_event.set()
            if auto_confirm_needed:
                messages.append("[AUTO] Action confirmed.")
        elif ch_lower.isdigit():
            obj_enum = resolve_step_token(ch_lower)
            if not obj_enum:
                messages.append("[AUTO] Unknown step code.")
            else:
                logs = load_demo_logs(app_state.demos_dir)
                if not logs:
                    messages.append("[AUTO] No demo logs found. Record a demo first.")
                else:
                    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
                    refresh_autobuild_config(PROCESS_MODEL_FILE)
                    model = load_process_model(PROCESS_MODEL_FILE)
                    obj_key = normalize_step_label(obj_enum.value)
                    obj_cfg = (model.get("steps") or {}).get(obj_key, {})
                    success_gates = obj_cfg.get("success_gates") if isinstance(obj_cfg, dict) else None
                    nominal_only = isinstance(obj_cfg, dict) and obj_cfg.get("nominalDemosOnly")
                    if not success_gates and not nominal_only:
                        messages.append(f"[AUTO] No success gates for {obj_key}. Record a success demo first.")
                    else:
                        with app_state.lock:
                            app_state.last_key_time = time.time()
                            app_state.auto_request = obj_enum
                            app_state.active_command = None
                            app_state.active_speed = 0.0
                            app_state.active_speed_score = None
        elif auto_prompt:
            with app_state.lock:
                app_state.last_key_time = time.time()
                if ch_lower == 'm':
                    app_state.auto_prompt = False
                    messages.append("[AUTO] Auto mode cancelled.")
                else:
                    obj_enum = resolve_step_token(ch_lower)
                    if obj_enum:
                        app_state.auto_prompt = False
                        app_state.auto_request = obj_enum
                        app_state.active_command = None
                        app_state.active_speed = 0.0
                        app_state.active_speed_score = None
                    else:
                        messages.append("[AUTO] Unknown step code.")
        elif ch_lower == ':':
            with app_state.lock:
                app_state.last_key_time = time.time()
                end_msg = None
                if app_state.active_attempt:
                    ok, msg, _, _, _ = end_attempt(app_state, complete_step=True)
                    end_msg = msg

                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
            if end_msg:
                log_line(end_msg)
            log_line("[CMD] Enter command (ex: 4s, 4f, help). Use ':' or blank to exit.")
            
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
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.auto_prompt = True
                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
            messages.append("[AUTO] Select a step code to run autonomously (press 'm' again to cancel).")
        elif ch_lower == 'y':
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
            with app_state.lock:
                app_state.last_key_time = time.time()
                # MOVEMENT (Heartbeat triggers)
                action = manual_key_action(ch_lower)
                if action:
                    cmd, score = action
                    app_state.active_command = None
                    app_state.active_speed = 0.0
                    app_state.active_speed_score = None
                    pending_hotkey_action = (cmd, score)
                elif ch_lower in ('h', '?'):
                    print_command_help(app_state)

        if pending_hotkey_action and app_state.running:
            cmd, score = pending_hotkey_action
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.active_command = cmd
                app_state.active_speed_score = score
            # Execute the hotkey act first, then offer var editing for that act.
            time.sleep(1.0 / max(COMMAND_RATE_HZ, 1e-3))
            score_used = maybe_prompt_hotkey_var_update(app_state, cmd, score, enter_edit=False)
            with app_state.lock:
                app_state.hotkey_edit_target = {
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
    if mode == VISION_MODE_MARKERLESS:
        return YoloBrickDetector(debug=True, model_path=yolo_model_path)
    return ArucoBrickVision(debug=True)


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
    mode_norm = normalize_vision_mode(mode, fallback=VISION_MODE_ARUCO)
    if lock is None:
        stream_state["vision_mode"] = mode_norm
        return
    with lock:
        stream_state["vision_mode"] = mode_norm


def _stream_state_markerless_profile(app_state, fallback):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return normalize_markerless_profile(fallback, fallback=fallback)
    lock = stream_state.get("lock")
    if lock is None:
        value = stream_state.get("markerless_profile", fallback)
    else:
        with lock:
            value = stream_state.get("markerless_profile", fallback)
    return normalize_markerless_profile(value, fallback=fallback)


def _set_stream_state_markerless_profile(app_state, profile):
    stream_state = app_state.stream_state if isinstance(app_state.stream_state, dict) else None
    if not stream_state:
        return
    lock = stream_state.get("lock")
    profile_norm = normalize_markerless_profile(profile, fallback=_DEFAULT_MARKERLESS_PROFILE)
    if lock is None:
        stream_state["markerless_profile"] = profile_norm
        return
    with lock:
        stream_state["markerless_profile"] = profile_norm


def _apply_markerless_profile(app_state, vision, profile):
    profile_key, settings = markerless_profile_settings(profile)
    _set_stream_state_markerless_profile(app_state, profile_key)
    if not isinstance(vision, YoloBrickDetector):
        return profile_key, settings, False

    confidence = settings.get("conf_threshold")
    smooth_alpha = settings.get("smooth_alpha")
    nms_threshold = settings.get("nms_threshold")
    if hasattr(vision, "set_runtime_tuning"):
        vision.set_runtime_tuning(
            confidence=confidence,
            smoothing_alpha=smooth_alpha,
            nms_threshold=nms_threshold,
        )
    else:
        if confidence is not None:
            vision.conf_threshold = float(confidence)
        if smooth_alpha is not None:
            vision._smooth_alpha = float(smooth_alpha)
        if nms_threshold is not None:
            vision.nms_threshold = float(nms_threshold)
    for attr in ("_prev_angle", "_prev_dist", "_prev_offset"):
        if hasattr(vision, attr):
            setattr(vision, attr, None)
    return profile_key, settings, True


def _vision_mode_label(mode):
    if normalize_vision_mode(mode) == VISION_MODE_MARKERLESS:
        return "Markerless"
    return "AruCo Markers"


def control_loop(app_state, vision_mode="aruco", yolo_model_path=None):
    app_state.robot = Robot()
    active_vision_mode = normalize_vision_mode(vision_mode)
    active_markerless_profile = _stream_state_markerless_profile(app_state, _DEFAULT_MARKERLESS_PROFILE)
    _set_stream_state_vision_mode(app_state, active_vision_mode)
    _set_stream_state_markerless_profile(app_state, active_markerless_profile)
    # speed_optimize=False so we get the debug markers drawn on the frame
    app_state.vision = build_vision(active_vision_mode, yolo_model_path=yolo_model_path)
    profile_key, profile_settings, profile_applied = _apply_markerless_profile(
        app_state,
        app_state.vision,
        active_markerless_profile,
    )
    active_markerless_profile = profile_key
    log_line(
        f"[VISION] Active mode: {_vision_mode_label(active_vision_mode)} "
        f"(--vision {_vision_mode_cli_value(active_vision_mode)})"
    )
    if active_vision_mode == VISION_MODE_MARKERLESS and profile_applied:
        log_line(
            "[VISION] Markerless config: "
            f"{markerless_profile_label(active_markerless_profile)} "
            f"(min_conf={float(profile_settings.get('conf_threshold', 0.15)):.2f}, "
            f"smooth={float(profile_settings.get('smooth_alpha', 0.30)):.2f})"
        )

    if app_state.stream_enabled:
        stream_t = threading.Thread(target=stream_refresh_loop, args=(app_state,), daemon=True)
        stream_t.start()

    cmd_t = threading.Thread(target=command_loop, args=(app_state,), daemon=True)
    cmd_t.start()
    
    dt = 1.0 / LOG_RATE_HZ
    
    while app_state.running:
        loop_start = time.time()
        requested_markerless_profile = _stream_state_markerless_profile(
            app_state,
            active_markerless_profile,
        )
        if requested_markerless_profile != active_markerless_profile:
            active_markerless_profile = requested_markerless_profile
            if active_vision_mode == VISION_MODE_MARKERLESS and app_state.vision is not None:
                profile_key, profile_settings, profile_applied = _apply_markerless_profile(
                    app_state,
                    app_state.vision,
                    active_markerless_profile,
                )
                active_markerless_profile = profile_key
                if profile_applied:
                    with app_state.lock:
                        app_state.brick_frame_buffer = []
                    log_line(
                        "[VISION] Markerless config: "
                        f"{markerless_profile_label(active_markerless_profile)} "
                        f"(min_conf={float(profile_settings.get('conf_threshold', 0.15)):.2f}, "
                        f"smooth={float(profile_settings.get('smooth_alpha', 0.30)):.2f})"
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
                    _apply_markerless_profile(app_state, restored_vision, active_markerless_profile)
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
                profile_key, profile_settings, profile_applied = _apply_markerless_profile(
                    app_state,
                    app_state.vision,
                    active_markerless_profile,
                )
                active_markerless_profile = profile_key
                with app_state.lock:
                    app_state.brick_frame_buffer = []
                _set_stream_state_vision_mode(app_state, active_vision_mode)
                log_line(
                    f"[VISION] Switched to {_vision_mode_label(active_vision_mode)} "
                    f"(--vision {_vision_mode_cli_value(active_vision_mode)})."
                )
                if active_vision_mode == VISION_MODE_MARKERLESS and profile_applied:
                    log_line(
                        "[VISION] Markerless config: "
                        f"{markerless_profile_label(active_markerless_profile)} "
                        f"(min_conf={float(profile_settings.get('conf_threshold', 0.15)):.2f}, "
                        f"smooth={float(profile_settings.get('smooth_alpha', 0.30)):.2f})"
                    )

        refresh_world_model_from_demos(app_state)
        auto_obj = None
        with app_state.lock:
            if app_state.auto_request and not app_state.auto_running:
                auto_obj = app_state.auto_request
                app_state.auto_request = None
                app_state.auto_running = True
                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None

        if auto_obj:
            if app_state.vision is None:
                log_line("[VISION] Auto-step skipped: no active vision backend.")
            else:
                run_auto_step(app_state, auto_obj)
            with app_state.lock:
                app_state.auto_running = False
                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
            continue
            
        # 1. Vision
        vision = app_state.vision
        if vision is None:
            time.sleep(dt)
            continue
        found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
        
        # 2. Telemetry Update
        study = None
        with app_state.lock:
            app_state.brick_frame_buffer.append({
                "found": bool(found),
                "dist": float(dist),
                "angle": float(angle),
                "offset_x": float(offset_x),
                "conf": float(conf),
                "cam_h": float(cam_h),
                "brick_above": bool(brick_above),
                "brick_below": bool(brick_below),
            })
            if len(app_state.brick_frame_buffer) > BRICK_STUDY_FRAMES:
                app_state.brick_frame_buffer.pop(0)

            if isinstance(vision, YoloBrickDetector):
                study = evaluate_markerless_frames(app_state.brick_frame_buffer)
            else:
                study = evaluate_brick_frames(app_state.brick_frame_buffer)
            if study and study.get("reset"):
                app_state.brick_frame_buffer = []
        if study and study.get("average"):
            avg = study["average"]
            app_state.world.update_vision(
                avg["found"],
                avg["dist"],
                avg["angle"],
                avg["conf"],
                avg["offset_x"],
                avg["cam_h"],
                avg["brick_above"],
                avg["brick_below"],
            )
        refresh_brick_telemetry(app_state, read_vision=False)
        
        # Track Motion
        with app_state.lock:
            cmd = app_state.active_command
            speed = app_state.active_speed
            score = app_state.active_speed_score

        brick = app_state.world.brick if isinstance(app_state.world.brick, dict) else {}
        metric_x_mm = None
        if brick.get("visible"):
            metric_x_mm = brick.get("x_axis", brick.get("offset_x"))

        turn_cmd = cmd if cmd in ("l", "r") else "l"
        min_turn_pwm = None
        try:
            min_turn_pwm = int(telemetry_robot_module.baseline_pwm_floor_for_cmd(turn_cmd))
        except Exception:
            try:
                min_turn_pwm = int(telemetry_robot_module.turn_pwm_floor())
            except Exception:
                min_turn_pwm = None

        score_power_pwm_turn = None
        try:
            score_power_pwm_turn = telemetry_robot_module.score_power_pwm_for_cmd(turn_cmd)
        except Exception:
            score_power_pwm_turn = None
        if not isinstance(score_power_pwm_turn, dict):
            score_power_pwm_turn = getattr(telemetry_robot_module, "SCORE_POWER_PWM_TURN", None)
        if not isinstance(score_power_pwm_turn, dict):
            score_power_pwm_turn = telemetry_robot_module.SCORE_POWER_PWM

        max_turn_pwm = None
        try:
            max_entry = score_power_pwm_turn.get(telemetry_robot_module.SPEED_SCORE_MAX)
            if isinstance(max_entry, dict):
                max_turn_pwm = int(max_entry.get("pwm"))
        except Exception:
            max_turn_pwm = None

        adjust_msg = micro_adjust_speed_score(
            app_state.micro_speed_state,
            score_power_pwm=score_power_pwm_turn,
            metric_value_mm=metric_x_mm,
            active=bool(
                cmd in ("l", "r")
                and score == telemetry_robot_module.SPEED_SCORE_MIN
                and speed > 0.0
                and app_state.world.step_state == StepState.ALIGN_BRICK
            ),
            sequence_key=cmd,
            acts=3,
            threshold_mm=0.5,
            increase_scale=1.01,
            decrease_scale=0.99,
            min_pwm=min_turn_pwm,
            max_pwm=max_turn_pwm if max_turn_pwm is not None else telemetry_robot_module.MAX_PWM,
            metric_label="x_axis",
        )
        if adjust_msg:
            log_line(adjust_msg)
        if cmd and speed > 0:
            atype = "unknown"
            if cmd == 'f': atype = "forward"
            elif cmd == 'b': atype = "backward"
            elif cmd == 'l': atype = "left_turn"
            elif cmd == 'r': atype = "right_turn"
            elif cmd == 'u': atype = "mast_up"
            elif cmd == 'd': atype = "mast_down"
            
            pwr = int(speed * 255)
            evt = MotionEvent(atype, pwr, int(dt*1000), speed_score=score)
            app_state.world.update_from_motion(evt)
            if app_state.active_attempt:
                app_state.logger.log_event(evt, app_state.world.step_state.value)
            
        # 4. Save Log (Image saving removed)
        with app_state.lock:
            if app_state.active_attempt and app_state.active_attempt != "NOMINAL":
                app_state.logger.log_state(app_state.world)
        
        # 5. Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def command_loop(app_state):
    dt = 1.0 / max(COMMAND_RATE_HZ, 1e-3)
    was_moving = False
    while app_state.running:
        loop_start = time.time()
        with app_state.lock:
            if app_state.auto_running or app_state.auto_request is not None:
                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None
            if time.time() - app_state.last_key_time > HEARTBEAT_TIMEOUT:
                app_state.active_command = None
                app_state.active_speed = 0.0
                app_state.active_speed_score = None

            cmd = app_state.active_command
            score = app_state.active_speed_score

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
            send_robot_command(
                app_state.robot,
                app_state.world,
                app_state.world.step_state,
                cmd,
                speed,
                speed_score=score_used,
            )
            was_moving = True
        elif was_moving:
            app_state.robot.stop()
            was_moving = False

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brick-vision",
        "--vision",
        dest="brick_vision",
        choices=["aruco", "yolo", "markerless"],
        default=_DEFAULT_VISION_MODE,
        help="Brick vision model to use (aruco/yolo; markerless is an alias for yolo)",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to YOLO ONNX model (used with --brick-vision yolo or --vision yolo)",
    )
    parser.add_argument(
        "--markerless-profile",
        choices=[value for value, _label in MARKERLESS_PROFILE_OPTIONS],
        default=_DEFAULT_MARKERLESS_PROFILE,
        help="Markerless tuning profile (applies when vision mode is yolo/markerless)",
    )
    parser.add_argument("--stream", dest="stream", action="store_true",
                        help="Enable livestreaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable livestreaming")
    parser.set_defaults(stream=True)
    args = parser.parse_args()
    vision_mode = normalize_vision_mode(args.brick_vision, fallback=_DEFAULT_VISION_MODE)
    markerless_profile = normalize_markerless_profile(
        args.markerless_profile,
        fallback=_DEFAULT_MARKERLESS_PROFILE,
    )

    logs = load_demo_logs(DEMOS_DIR)
    if logs:
        update_process_model_from_demos(logs, PROCESS_MODEL_FILE)

    state = AppState()
    _set_stream_state_vision_mode(state, vision_mode)
    _set_stream_state_markerless_profile(state, markerless_profile)
    refresh_world_model_from_demos(state, force=True)
    print_command_help(state)
    
    # Keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
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
                vision_mode_options=STREAM_VISION_MODE_OPTIONS,
                markerless_profile_options=MARKERLESS_PROFILE_OPTIONS,
            )
        except Exception as exc:
            state.stream_enabled = False
            log_line(f"[VISION] Stream failed to start: {exc}")
        else:
            state.stream_enabled = True
            actual_port = getattr(stream_server, "port", STREAM_PORT)
            if actual_port != STREAM_PORT:
                log_line(f"[VISION] Stream port {STREAM_PORT} busy; using {actual_port}")
            log_line(f"[VISION] Stream started at {ANSI_ORANGE_BRIGHT}{url}{ANSI_RESET}")
            _open_stream_in_chrome("http://127.0.0.1:5000/")
    else:
        state.stream_enabled = False
        log_line("[VISION] Stream disabled")
    
    try:
        control_loop(state, vision_mode=vision_mode, yolo_model_path=args.yolo_model)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        close_log(state, marker=None)
        log_line("Shutdown complete.")
