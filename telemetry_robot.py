"""
# telemetry_robot.py
-----------------
Handles the World Model and Logging for Robot Leia.
"""
import json
import copy
import math
import os
import threading
import time
import collections
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import helper_xyz_coords

# Speed/PWM tuning (single source of truth)
MIN_PWM = 36
MAX_PWM = 255
MIN_TURN_POWER = 0.064

ALIGN_MIN_SPEED = 0.2
ALIGN_MAX_SPEED = 0.28
ALIGN_MICRO_SPEED = 0.21
ALIGN_FIXED_SPEED = 0.19
ALIGN_SPEED_MIN_POWER = 0.288
ALIGN_SPEED_SLOW = ALIGN_SPEED_MIN_POWER
ALIGN_SPEED_NORMAL = 0.28
ALIGN_SPEED_FAST = 0.392
ALIGN_SPEED_SLOW_MM = 8.0
ALIGN_SPEED_FAST_MM = 18.0
ALIGN_MICRO_OFFSET_MM = 10.0
ALIGN_MICRO_ANGLE_DEG = 5.0

SPEED_SCORE_MIN = 1
SPEED_SCORE_DEFAULT = 50
SPEED_SCORE_MAX = 100
SPEED_SCORE_LEVELS = tuple(range(SPEED_SCORE_MIN, SPEED_SCORE_MAX + 1))
DEFAULT_ACT_DURATION_MS = 300
MOTION_EASE_IN_OUT_ENABLED = True
MOTION_EASE_IN_OUT_MIN_SCORE_DRIVE = 10
MOTION_EASE_IN_OUT_MIN_SCORE_TURN = 20
# Backward-compatible aggregate threshold alias.
MOTION_EASE_IN_OUT_MIN_SCORE = MOTION_EASE_IN_OUT_MIN_SCORE_DRIVE
MOTION_EASE_IN_OUT_MIN_RAMP_MS = 300
MOTION_EASE_IN_OUT_MAX_RAMP_MS = 800
MOTION_EASE_IN_OUT_RAMP_STEPS = 3
# Backward-compatible aliases for older call sites/tests.
DRIVE_ANTI_ALIAS_ENABLED = MOTION_EASE_IN_OUT_ENABLED
DRIVE_ANTI_ALIAS_MIN_SCORE = MOTION_EASE_IN_OUT_MIN_SCORE
DRIVE_ANTI_ALIAS_RAMP_MS = MOTION_EASE_IN_OUT_MIN_RAMP_MS
DRIVE_ANTI_ALIAS_RAMP_STEPS = MOTION_EASE_IN_OUT_RAMP_STEPS
SPEED_MAP_KEY_DRIVE = "score_power_pwm_drive"
SPEED_MAP_KEY_TURN = "score_power_pwm_turn"
SPEED_MAP_KEY_TURN_LEFT = "score_power_pwm_turn_left"
SPEED_MAP_KEY_TURN_RIGHT = "score_power_pwm_turn_right"
SPEED_SECONDS_KEY = "speed_score_seconds"
SPEED_SECONDS_KEY_TURN = "speed_score_seconds_turn"
SPEED_SECONDS_KEY_TURN_LEFT = "speed_score_seconds_turn_left"
SPEED_SECONDS_KEY_TURN_RIGHT = "speed_score_seconds_turn_right"
SPEED_SECONDS_KEY_MAST_UP = "speed_score_seconds_mast_up"
SPEED_SECONDS_KEY_MAST_DOWN = "speed_score_seconds_mast_down"
TURN_INTENSITY_KEY_TURN = "turn_intensity_posts_turn"
TURN_INTENSITY_KEY_TURN_LEFT = "turn_intensity_posts_turn_left"
TURN_INTENSITY_KEY_TURN_RIGHT = "turn_intensity_posts_turn_right"
MAST_INTENSITY_KEY_MAST = "mast_intensity_posts_mast"
MAST_INTENSITY_KEY_MAST_UP = "mast_intensity_posts_mast_up"
MAST_INTENSITY_KEY_MAST_DOWN = "mast_intensity_posts_mast_down"

ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"
DEFAULT_SPEED_MODEL = {
    "min_pwm": MIN_PWM,
    "max_pwm": MAX_PWM,
    "min_turn_power": MIN_TURN_POWER,
    "hotkey_speed_scores": {
        "w": {"cmd": "f", "score": 50},
        "s": {"cmd": "b", "score": 50},
        "r": {"cmd": "f", "score": 1},
        "f": {"cmd": "b", "score": 1},
        "t": {"cmd": "f", "score": 100},
        "g": {"cmd": "b", "score": 100},
        "q": {"cmd": "l", "score": 1},
        "a": {"cmd": "l", "score": 25},
        "z": {"cmd": "l", "score": 100},
        "e": {"cmd": "r", "score": 1},
        "d": {"cmd": "r", "score": 25},
        "c": {"cmd": "r", "score": 100},
        "o": {"cmd": "u", "score": 1},
        "k": {"cmd": "d", "score": 1},
        "u": {"cmd": "u", "score": 50},
        "p": {"cmd": "u", "score": 100},
        "l": {"cmd": "d", "score": 25},
    },
    SPEED_MAP_KEY_DRIVE: {
        "1": {"power": 0.064, "pwm": 50},
        "100": {"power": 1.0, "pwm": 255},
    },
    SPEED_MAP_KEY_TURN: {
        "1": {"power": 0.064, "pwm": 50},
        "100": {"power": 1.0, "pwm": 255},
    },
    # Directional turn maps (preferred). Kept separate because L/R motors can
    # respond differently at identical requested scores.
    SPEED_MAP_KEY_TURN_LEFT: {
        "1": {"power": 0.064, "pwm": 50},
        "100": {"power": 1.0, "pwm": 255},
    },
    SPEED_MAP_KEY_TURN_RIGHT: {
        "1": {"power": 0.064, "pwm": 50},
        "100": {"power": 1.0, "pwm": 255},
    },
    # Optional duration override by score (seconds).
    # If present in world_model_robot.json, these values override duration_ms.
    SPEED_SECONDS_KEY: {
        "1": 0.30,
        "100": 0.30,
    },
    # Optional directional duration overrides by score (seconds).
    SPEED_SECONDS_KEY_TURN: {
        "1": 0.30,
        "100": 0.30,
    },
    SPEED_SECONDS_KEY_TURN_LEFT: {
        "1": 0.30,
        "100": 0.30,
    },
    SPEED_SECONDS_KEY_TURN_RIGHT: {
        "1": 0.30,
        "100": 0.30,
    },
    SPEED_SECONDS_KEY_MAST_UP: {
        "1": 0.30,
        "100": 0.30,
    },
    SPEED_SECONDS_KEY_MAST_DOWN: {
        "1": 0.30,
        "100": 0.30,
    },
    # Optional fractional turn-intensity anchors (%). Values are interpolated
    # piecewise to produce PWM/power/duration for L/R turns.
    TURN_INTENSITY_KEY_TURN: {
        "0.5": {"score": 1, "duration_scale": 0.5},
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    TURN_INTENSITY_KEY_TURN_LEFT: {
        "0.5": {"score": 1, "duration_scale": 0.5},
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    TURN_INTENSITY_KEY_TURN_RIGHT: {
        "0.5": {"score": 1, "duration_scale": 0.5},
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    MAST_INTENSITY_KEY_MAST: {
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    MAST_INTENSITY_KEY_MAST_UP: {
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    MAST_INTENSITY_KEY_MAST_DOWN: {
        "1.0": {"score": 1},
        "5.0": {"score": 5},
        "25.0": {"score": 25},
        "100.0": {"score": 100},
    },
    "turn_efficiency": {
        "l": 300.0,
        "r": 300.0,
    },
    # Optional autonomous-only turn boost at score=1 (percent).
    "auto_turn_speed_boost_pct": 0.0,
}

def _brick_module():
    import telemetry_brick
    return telemetry_brick


def _wall_module():
    import telemetry_wall
    return telemetry_wall
    

def _load_speed_model(path=None):
    if path is None:
        path = ROBOT_MODEL_FILE
    
    print(f"[SYSTEM] Loading speed model from {path}...")
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            text = path.read_text()
            data = json.loads(text)
            if isinstance(data, dict):
                model = data
            else:
                print(f"[ERROR] JSON root is not a dict: {type(data)}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to load speed model: {e}")
            model = DEFAULT_SPEED_MODEL
    else:
        print(f"[WARNING] Speed model file not found: {path}")


def _closest_score(score, levels, default=SPEED_SCORE_DEFAULT):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return int(default)
    closest = None
    for candidate in levels:
        if closest is None or abs(candidate - score) < abs(closest - score):
            closest = candidate
    return int(closest if closest is not None else default)


def normalize_speed_score(score, default=SPEED_SCORE_DEFAULT):
    try:
        value = int(round(float(score)))
    except (TypeError, ValueError):
        value = int(default)
    return max(SPEED_SCORE_MIN, min(SPEED_SCORE_MAX, int(value)))


def _power_to_pwm(power, *, min_pwm=None, max_pwm=None):
    try:
        p = float(power)
    except (TypeError, ValueError):
        return None
    p = max(0.0, min(1.0, p))
    try:
        min_val = MIN_PWM if min_pwm is None else int(round(float(min_pwm)))
    except (TypeError, ValueError):
        min_val = int(MIN_PWM)
    try:
        max_val = MAX_PWM if max_pwm is None else int(round(float(max_pwm)))
    except (TypeError, ValueError):
        max_val = int(MAX_PWM)
    min_val = max(0, min(255, int(min_val)))
    max_val = max(0, min(255, int(max_val)))
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    pwm = int(round(min_val + (max_val - min_val) * p))
    return max(0, min(255, pwm))


def _pwm_to_power(pwm, *, min_pwm=None, max_pwm=None):
    try:
        raw = float(pwm)
    except (TypeError, ValueError):
        return None
    raw = max(0.0, min(255.0, raw))
    if raw <= 0.0:
        return 0.0
    try:
        min_val = MIN_PWM if min_pwm is None else int(round(float(min_pwm)))
    except (TypeError, ValueError):
        min_val = int(MIN_PWM)
    try:
        max_val = MAX_PWM if max_pwm is None else int(round(float(max_pwm)))
    except (TypeError, ValueError):
        max_val = int(MAX_PWM)
    min_val = max(0, min(255, int(min_val)))
    max_val = max(0, min(255, int(max_val)))
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    span = max(1.0, float(max_val - min_val))
    p = (raw - float(min_val)) / span
    return max(0.0, min(1.0, p))


def clamp_pwm(pwm):
    try:
        value = int(round(pwm))
    except (TypeError, ValueError):
        return 0
    return max(0, min(255, value))


def power_to_pwm(power):
    return _power_to_pwm(power)


def pwm_to_power(pwm):
    return _pwm_to_power(pwm)


def turn_pwm_floor():
    pwm = _power_to_pwm(MIN_TURN_POWER)
    if pwm is None:
        return 0
    return int(pwm)


def interp_pwm_for_score(score, slow_pwm, fast_pwm):
    score = normalize_speed_score(score)
    try:
        slow = int(round(slow_pwm))
        fast = int(round(fast_pwm))
    except (TypeError, ValueError):
        return None
    slow = clamp_pwm(slow)
    fast = clamp_pwm(fast)
    if fast < slow:
        slow, fast = fast, slow
    frac = (float(score) - float(SPEED_SCORE_MIN)) / float(SPEED_SCORE_MAX - SPEED_SCORE_MIN)
    pwm = int(round(float(slow) + (float(fast) - float(slow)) * frac))
    return clamp_pwm(pwm)


def _coerce_score_power_pwm(raw, fallback, *, min_pwm=None, max_pwm=None):
    if not isinstance(raw, dict):
        raw = fallback
    cleaned = {}
    for key, value in raw.items():
        try:
            score_key = int(float(key))
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        power_raw = value.get("power")
        pwm_raw = value.get("pwm")

        power = None
        pwm = None
        if power_raw is not None:
            try:
                power = float(power_raw)
            except (TypeError, ValueError):
                power = None
        if pwm_raw is not None:
            try:
                pwm = int(pwm_raw)
            except (TypeError, ValueError):
                pwm = None

        # Keep power/pwm consistent. PWM is authoritative because hardware sends PWM.
        if pwm is not None and power is not None:
            power = _pwm_to_power(pwm, min_pwm=min_pwm, max_pwm=max_pwm)
            pwm = max(0, min(255, int(pwm)))
        elif pwm is not None:
            pwm = max(0, min(255, int(pwm)))
            power = _pwm_to_power(pwm, min_pwm=min_pwm, max_pwm=max_pwm)
        elif power is not None:
            power = max(0.0, min(1.0, float(power)))
            pwm = _power_to_pwm(power, min_pwm=min_pwm, max_pwm=max_pwm)
        else:
            continue

        if power is None or pwm is None:
            continue

        cleaned[score_key] = {
            "power": max(0.0, min(1.0, float(power))),
            "pwm": max(0, min(255, int(pwm))),
            # Duration is sourced separately from speed_score_seconds.
            "duration_ms": DEFAULT_ACT_DURATION_MS,
        }
    return cleaned


def _coerce_hotkeys(raw, fallback, score_levels):
    if not isinstance(raw, dict):
        raw = fallback
    cleaned = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        cmd = value.get("cmd")
        if not cmd:
            continue
        score = normalize_speed_score(value.get("score"), default=SPEED_SCORE_DEFAULT)
        row = {"cmd": str(cmd), "score": score}
        pwm_raw = value.get("pwm")
        try:
            pwm_val = clamp_pwm(int(round(float(pwm_raw))))
        except (TypeError, ValueError):
            pwm_val = None
        if pwm_val is not None and pwm_val > 0:
            row["pwm"] = int(pwm_val)
            power_from_pwm = _pwm_to_power(pwm_val)
            if power_from_pwm is not None:
                row["power"] = float(power_from_pwm)
        else:
            power_raw = value.get("power")
            try:
                power_val = float(power_raw)
            except (TypeError, ValueError):
                power_val = None
            if power_val is not None:
                power_val = max(0.0, min(1.0, float(power_val)))
                row["power"] = float(power_val)
        duration_raw = value.get("duration_ms")
        try:
            duration_ms = int(round(float(duration_raw)))
        except (TypeError, ValueError):
            duration_ms = None
        if duration_ms is not None and duration_ms > 0:
            row["duration_ms"] = int(duration_ms)
        cleaned[str(key)] = row
    return cleaned


def _coerce_score_seconds(raw, fallback):
    if not isinstance(raw, dict):
        raw = fallback if isinstance(fallback, dict) else {}
    cleaned = {}
    for key, value in raw.items():
        try:
            score_key = int(float(key))
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds <= 0:
            continue
        cleaned[normalize_speed_score(score_key, default=score_key)] = float(seconds)
    return cleaned


def _coerce_command_remap(raw):
    if not isinstance(raw, dict):
        return {}
    cleaned = {}
    for key, value in raw.items():
        if key is None or value is None:
            continue
        cleaned[str(key)] = str(value)
    return cleaned


def _coerce_turn_intensity_posts(
    raw,
    fallback,
    *,
    cmd,
    low_pwm,
    high_pwm,
    duration_map,
    min_pwm=None,
    max_pwm=None,
):
    def _duration_for_score(score_key):
        exact = duration_map.get(int(score_key)) if isinstance(duration_map, dict) else None
        if exact is not None:
            try:
                return max(1, int(round(float(exact))))
            except (TypeError, ValueError):
                pass
        low = duration_map.get(SPEED_SCORE_MIN) if isinstance(duration_map, dict) else None
        high = duration_map.get(SPEED_SCORE_MAX) if isinstance(duration_map, dict) else None
        if low is None or high is None:
            return int(DEFAULT_ACT_DURATION_MS)
        try:
            low = float(low)
            high = float(high)
        except (TypeError, ValueError):
            return int(DEFAULT_ACT_DURATION_MS)
        frac = (float(score_key) - float(SPEED_SCORE_MIN)) / float(SPEED_SCORE_MAX - SPEED_SCORE_MIN)
        return max(1, int(round(low + (high - low) * frac)))

    def _resolve_entry(entry):
        if not isinstance(entry, dict):
            return None
        score_raw = entry.get("score")
        score_val = None
        if score_raw is not None:
            try:
                score_val = normalize_speed_score(score_raw)
            except Exception:
                score_val = None

        pwm = entry.get("pwm")
        power = entry.get("power")
        duration_ms = entry.get("duration_ms")
        duration_scale = entry.get("duration_scale")
        duration_s = entry.get("duration_s")

        if score_val is not None:
            pwm_scored = interp_pwm_for_score(score_val, low_pwm, high_pwm)
            if pwm_scored is not None:
                pwm = pwm if pwm is not None else pwm_scored
            if duration_ms is None:
                duration_ms = _duration_for_score(score_val)

        try:
            pwm_val = int(round(float(pwm))) if pwm is not None else None
        except (TypeError, ValueError):
            pwm_val = None
        try:
            power_val = float(power) if power is not None else None
        except (TypeError, ValueError):
            power_val = None

        if pwm_val is not None and power_val is not None:
            power_val = _pwm_to_power(pwm_val, min_pwm=min_pwm, max_pwm=max_pwm)
            pwm_val = clamp_pwm(pwm_val)
        elif pwm_val is not None:
            pwm_val = clamp_pwm(pwm_val)
            power_val = _pwm_to_power(pwm_val, min_pwm=min_pwm, max_pwm=max_pwm)
        elif power_val is not None:
            power_val = max(0.0, min(1.0, float(power_val)))
            pwm_val = _power_to_pwm(power_val, min_pwm=min_pwm, max_pwm=max_pwm)

        if pwm_val is None or power_val is None:
            return None

        if duration_ms is None and duration_s is not None:
            try:
                duration_ms = max(1, int(round(float(duration_s) * 1000.0)))
            except (TypeError, ValueError):
                duration_ms = None
        if duration_ms is None:
            duration_ms = int(DEFAULT_ACT_DURATION_MS)
        try:
            duration_ms_val = max(1, int(round(float(duration_ms))))
        except (TypeError, ValueError):
            duration_ms_val = int(DEFAULT_ACT_DURATION_MS)

        if duration_scale is not None:
            try:
                duration_ms_val = max(1, int(round(float(duration_ms_val) * float(duration_scale))))
            except (TypeError, ValueError):
                pass

        if str(cmd) in ("l", "r") and pwm_val > 0:
            pwm_val = max(turn_pwm_floor(), int(pwm_val))

        return {
            "pwm": int(pwm_val),
            "power": max(0.0, min(1.0, float(power_val))),
            "duration_ms": int(duration_ms_val),
        }

    source = raw if isinstance(raw, dict) else fallback
    cleaned = {}
    if isinstance(source, dict):
        for key, entry in source.items():
            try:
                intensity = float(key)
            except (TypeError, ValueError):
                continue
            if intensity <= 0.0:
                continue
            resolved = _resolve_entry(entry)
            if resolved is None:
                continue
            cleaned[float(intensity)] = resolved

    if cleaned:
        return cleaned

    # Last-resort default anchors from score interpolation.
    fallback_posts = {
        0.5: {"score": 1, "duration_scale": 0.5},
        1.0: {"score": 1},
        5.0: {"score": 5},
        25.0: {"score": 25},
        100.0: {"score": 100},
    }
    for intensity, entry in fallback_posts.items():
        resolved = _resolve_entry(entry)
        if resolved is not None:
            cleaned[float(intensity)] = resolved
    return cleaned


def _load_speed_model(path):
    loaded_from_file = False
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                model = data
                loaded_from_file = True
        except (OSError, json.JSONDecodeError):
            model = DEFAULT_SPEED_MODEL

    def _coerce_pwm(value, fallback):
        if value is None:
            return clamp_pwm(fallback)
        try:
            return clamp_pwm(int(round(float(value))))
        except (TypeError, ValueError):
            return clamp_pwm(fallback)

    min_pwm = _coerce_pwm(model.get("min_pwm"), DEFAULT_SPEED_MODEL.get("min_pwm", MIN_PWM))
    max_pwm = _coerce_pwm(model.get("max_pwm"), DEFAULT_SPEED_MODEL.get("max_pwm", MAX_PWM))
    if max_pwm < min_pwm:
        min_pwm, max_pwm = max_pwm, min_pwm

    try:
        min_turn_power = float(model.get("min_turn_power", DEFAULT_SPEED_MODEL.get("min_turn_power", MIN_TURN_POWER)))
    except (TypeError, ValueError):
        min_turn_power = float(DEFAULT_SPEED_MODEL.get("min_turn_power", MIN_TURN_POWER))
    min_turn_power = max(0.0, min(1.0, float(min_turn_power)))

    legacy_raw = model.get("score_power_pwm")

    drive_fallback = DEFAULT_SPEED_MODEL.get(SPEED_MAP_KEY_DRIVE)
    if not isinstance(drive_fallback, dict):
        drive_fallback = {}
    drive_raw = model.get(SPEED_MAP_KEY_DRIVE)
    if not isinstance(drive_raw, dict) and isinstance(legacy_raw, dict):
        drive_raw = legacy_raw
    drive_map = _coerce_score_power_pwm(drive_raw, drive_fallback, min_pwm=min_pwm, max_pwm=max_pwm)
    if not drive_map:
        drive_map = _coerce_score_power_pwm(drive_fallback, {}, min_pwm=min_pwm, max_pwm=max_pwm)
    if SPEED_SCORE_MIN not in drive_map or SPEED_SCORE_MAX not in drive_map:
        fallback = _coerce_score_power_pwm(drive_fallback, {}, min_pwm=min_pwm, max_pwm=max_pwm)
        for key in (SPEED_SCORE_MIN, SPEED_SCORE_MAX):
            if key not in drive_map and key in fallback:
                drive_map[key] = dict(fallback[key])

    turn_fallback = DEFAULT_SPEED_MODEL.get(SPEED_MAP_KEY_TURN)
    if not isinstance(turn_fallback, dict):
        turn_fallback = drive_fallback
    turn_raw = model.get(SPEED_MAP_KEY_TURN)
    if not isinstance(turn_raw, dict) and isinstance(legacy_raw, dict):
        turn_raw = legacy_raw
    turn_map = _coerce_score_power_pwm(turn_raw, turn_fallback, min_pwm=min_pwm, max_pwm=max_pwm)
    if not turn_map:
        turn_map = _coerce_score_power_pwm(turn_fallback, {}, min_pwm=min_pwm, max_pwm=max_pwm)
    if SPEED_SCORE_MIN not in turn_map or SPEED_SCORE_MAX not in turn_map:
        fallback = _coerce_score_power_pwm(turn_fallback, {}, min_pwm=min_pwm, max_pwm=max_pwm)
        for key in (SPEED_SCORE_MIN, SPEED_SCORE_MAX):
            if key not in turn_map and key in fallback:
                turn_map[key] = dict(fallback[key])

    # Directional turn maps (preferred). Fall back to shared turn map if absent.
    directional_turn_maps = bool(
        isinstance(model.get(SPEED_MAP_KEY_TURN_LEFT), dict)
        or isinstance(model.get(SPEED_MAP_KEY_TURN_RIGHT), dict)
    )

    turn_left_raw = model.get(SPEED_MAP_KEY_TURN_LEFT)
    if not isinstance(turn_left_raw, dict):
        turn_left_raw = turn_raw if isinstance(turn_raw, dict) else legacy_raw
    turn_left_map = _coerce_score_power_pwm(turn_left_raw, turn_map, min_pwm=min_pwm, max_pwm=max_pwm)
    if not turn_left_map:
        turn_left_map = _coerce_score_power_pwm(turn_map, {}, min_pwm=min_pwm, max_pwm=max_pwm)
    if SPEED_SCORE_MIN not in turn_left_map or SPEED_SCORE_MAX not in turn_left_map:
        fallback = _coerce_score_power_pwm(turn_map, {}, min_pwm=min_pwm, max_pwm=max_pwm)
        for key in (SPEED_SCORE_MIN, SPEED_SCORE_MAX):
            if key not in turn_left_map and key in fallback:
                turn_left_map[key] = dict(fallback[key])

    turn_right_raw = model.get(SPEED_MAP_KEY_TURN_RIGHT)
    if not isinstance(turn_right_raw, dict):
        turn_right_raw = turn_raw if isinstance(turn_raw, dict) else legacy_raw
    turn_right_map = _coerce_score_power_pwm(turn_right_raw, turn_map, min_pwm=min_pwm, max_pwm=max_pwm)
    if not turn_right_map:
        turn_right_map = _coerce_score_power_pwm(turn_map, {}, min_pwm=min_pwm, max_pwm=max_pwm)
    if SPEED_SCORE_MIN not in turn_right_map or SPEED_SCORE_MAX not in turn_right_map:
        fallback = _coerce_score_power_pwm(turn_map, {}, min_pwm=min_pwm, max_pwm=max_pwm)
        for key in (SPEED_SCORE_MIN, SPEED_SCORE_MAX):
            if key not in turn_right_map and key in fallback:
                turn_right_map[key] = dict(fallback[key])

    score_seconds = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY),
        DEFAULT_SPEED_MODEL.get(SPEED_SECONDS_KEY),
    )
    score_seconds_turn = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY_TURN),
        score_seconds,
    )
    directional_turn_duration_maps = bool(
        isinstance(model.get(SPEED_SECONDS_KEY_TURN_LEFT), dict)
        or isinstance(model.get(SPEED_SECONDS_KEY_TURN_RIGHT), dict)
    )
    score_seconds_turn_left = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY_TURN_LEFT),
        score_seconds_turn,
    )
    score_seconds_turn_right = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY_TURN_RIGHT),
        score_seconds_turn,
    )
    directional_mast_duration_maps = bool(
        isinstance(model.get(SPEED_SECONDS_KEY_MAST_UP), dict)
        or isinstance(model.get(SPEED_SECONDS_KEY_MAST_DOWN), dict)
    )
    score_seconds_mast_up = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY_MAST_UP),
        score_seconds,
    )
    score_seconds_mast_down = _coerce_score_seconds(
        model.get(SPEED_SECONDS_KEY_MAST_DOWN),
        score_seconds,
    )

    turn_intensity_raw = model.get(TURN_INTENSITY_KEY_TURN)
    if not isinstance(turn_intensity_raw, dict):
        turn_intensity_raw = DEFAULT_SPEED_MODEL.get(TURN_INTENSITY_KEY_TURN, {})
    turn_intensity_left_raw = model.get(TURN_INTENSITY_KEY_TURN_LEFT)
    if not isinstance(turn_intensity_left_raw, dict):
        turn_intensity_left_raw = turn_intensity_raw
    turn_intensity_right_raw = model.get(TURN_INTENSITY_KEY_TURN_RIGHT)
    if not isinstance(turn_intensity_right_raw, dict):
        turn_intensity_right_raw = turn_intensity_raw
    mast_intensity_raw = model.get(MAST_INTENSITY_KEY_MAST)
    if not isinstance(mast_intensity_raw, dict):
        mast_intensity_raw = DEFAULT_SPEED_MODEL.get(MAST_INTENSITY_KEY_MAST, {})
    mast_intensity_up_raw = model.get(MAST_INTENSITY_KEY_MAST_UP)
    if not isinstance(mast_intensity_up_raw, dict):
        mast_intensity_up_raw = mast_intensity_raw
    mast_intensity_down_raw = model.get(MAST_INTENSITY_KEY_MAST_DOWN)
    if not isinstance(mast_intensity_down_raw, dict):
        mast_intensity_down_raw = mast_intensity_raw

    if not score_seconds_turn:
        score_seconds_turn = dict(score_seconds)
    if not score_seconds_turn_left:
        score_seconds_turn_left = dict(score_seconds_turn)
    if not score_seconds_turn_right:
        score_seconds_turn_right = dict(score_seconds_turn)
    if not score_seconds_mast_up:
        score_seconds_mast_up = dict(score_seconds)
    if not score_seconds_mast_down:
        score_seconds_mast_down = dict(score_seconds)

    def _duration_ms_map(score_seconds_raw):
        return {
            score_key: max(1, int(round(float(seconds) * 1000.0)))
            for score_key, seconds in score_seconds_raw.items()
        }

    duration_ms_by_score = _duration_ms_map(score_seconds)
    duration_ms_by_score_turn = _duration_ms_map(score_seconds_turn)
    duration_ms_by_score_turn_left = _duration_ms_map(score_seconds_turn_left)
    duration_ms_by_score_turn_right = _duration_ms_map(score_seconds_turn_right)
    duration_ms_by_score_mast_up = _duration_ms_map(score_seconds_mast_up)
    duration_ms_by_score_mast_down = _duration_ms_map(score_seconds_mast_down)

    for score_key, duration_ms in duration_ms_by_score.items():
        entry = drive_map.get(score_key)
        if isinstance(entry, dict):
            entry["duration_ms"] = int(duration_ms)
    for score_key, duration_ms in duration_ms_by_score_turn.items():
        entry = turn_map.get(score_key)
        if isinstance(entry, dict):
            entry["duration_ms"] = int(duration_ms)
    for score_key, duration_ms in duration_ms_by_score_turn_left.items():
        entry = turn_left_map.get(score_key)
        if isinstance(entry, dict):
            entry["duration_ms"] = int(duration_ms)
    for score_key, duration_ms in duration_ms_by_score_turn_right.items():
        entry = turn_right_map.get(score_key)
        if isinstance(entry, dict):
            entry["duration_ms"] = int(duration_ms)

    low_pwm_left = ((turn_left_map.get(SPEED_SCORE_MIN) or {}).get("pwm") if isinstance(turn_left_map, dict) else None)
    high_pwm_left = ((turn_left_map.get(SPEED_SCORE_MAX) or {}).get("pwm") if isinstance(turn_left_map, dict) else None)
    low_pwm_right = ((turn_right_map.get(SPEED_SCORE_MIN) or {}).get("pwm") if isinstance(turn_right_map, dict) else None)
    high_pwm_right = ((turn_right_map.get(SPEED_SCORE_MAX) or {}).get("pwm") if isinstance(turn_right_map, dict) else None)
    try:
        low_pwm_left = int(low_pwm_left)
        high_pwm_left = int(high_pwm_left)
    except (TypeError, ValueError):
        low_pwm_left = int(MIN_PWM)
        high_pwm_left = int(MAX_PWM)
    try:
        low_pwm_right = int(low_pwm_right)
        high_pwm_right = int(high_pwm_right)
    except (TypeError, ValueError):
        low_pwm_right = int(MIN_PWM)
        high_pwm_right = int(MAX_PWM)
    low_pwm_mast = ((drive_map.get(SPEED_SCORE_MIN) or {}).get("pwm") if isinstance(drive_map, dict) else None)
    high_pwm_mast = ((drive_map.get(SPEED_SCORE_MAX) or {}).get("pwm") if isinstance(drive_map, dict) else None)
    try:
        low_pwm_mast = int(low_pwm_mast)
        high_pwm_mast = int(high_pwm_mast)
    except (TypeError, ValueError):
        low_pwm_mast = 0
        high_pwm_mast = int(MAX_PWM)

    turn_intensity_posts_left = _coerce_turn_intensity_posts(
        turn_intensity_left_raw,
        DEFAULT_SPEED_MODEL.get(TURN_INTENSITY_KEY_TURN_LEFT, {}),
        cmd="l",
        low_pwm=low_pwm_left,
        high_pwm=high_pwm_left,
        duration_map=duration_ms_by_score_turn_left,
        min_pwm=min_pwm,
        max_pwm=max_pwm,
    )
    turn_intensity_posts_right = _coerce_turn_intensity_posts(
        turn_intensity_right_raw,
        DEFAULT_SPEED_MODEL.get(TURN_INTENSITY_KEY_TURN_RIGHT, {}),
        cmd="r",
        low_pwm=low_pwm_right,
        high_pwm=high_pwm_right,
        duration_map=duration_ms_by_score_turn_right,
        min_pwm=min_pwm,
        max_pwm=max_pwm,
    )
    mast_intensity_posts_up = _coerce_turn_intensity_posts(
        mast_intensity_up_raw,
        DEFAULT_SPEED_MODEL.get(MAST_INTENSITY_KEY_MAST_UP, {}),
        cmd="u",
        low_pwm=low_pwm_mast,
        high_pwm=high_pwm_mast,
        duration_map=duration_ms_by_score_mast_up,
        min_pwm=0,
        max_pwm=max_pwm,
    )
    mast_intensity_posts_down = _coerce_turn_intensity_posts(
        mast_intensity_down_raw,
        DEFAULT_SPEED_MODEL.get(MAST_INTENSITY_KEY_MAST_DOWN, {}),
        cmd="d",
        low_pwm=low_pwm_mast,
        high_pwm=high_pwm_mast,
        duration_map=duration_ms_by_score_mast_down,
        min_pwm=0,
        max_pwm=max_pwm,
    )
    hotkey_fallback = {} if loaded_from_file else DEFAULT_SPEED_MODEL["hotkey_speed_scores"]
    hotkeys = _coerce_hotkeys(model.get("hotkey_speed_scores"), hotkey_fallback, SPEED_SCORE_LEVELS)
    if not hotkeys and not loaded_from_file:
        hotkeys = _coerce_hotkeys(DEFAULT_SPEED_MODEL["hotkey_speed_scores"], {}, SPEED_SCORE_LEVELS)
    
    turn_eff = model.get("turn_efficiency", DEFAULT_SPEED_MODEL["turn_efficiency"])
    if not isinstance(turn_eff, dict):
        turn_eff = DEFAULT_SPEED_MODEL["turn_efficiency"]
    cmd_remap = _coerce_command_remap(model.get("command_remap"))
    act_duration_ms = duration_ms_by_score.get(SPEED_SCORE_DEFAULT, DEFAULT_ACT_DURATION_MS)
    boost_raw = model.get("auto_turn_speed_boost_pct", DEFAULT_SPEED_MODEL.get("auto_turn_speed_boost_pct", 0.0))
    try:
        auto_turn_speed_boost_pct = float(boost_raw)
    except (TypeError, ValueError):
        auto_turn_speed_boost_pct = 0.0
    auto_turn_speed_boost_pct = max(0.0, min(100.0, auto_turn_speed_boost_pct))

    return (
        hotkeys,
        drive_map,
        turn_map,
        turn_left_map,
        turn_right_map,
        directional_turn_maps,
        duration_ms_by_score,
        duration_ms_by_score_turn,
        duration_ms_by_score_turn_left,
        duration_ms_by_score_turn_right,
        directional_turn_duration_maps,
        duration_ms_by_score_mast_up,
        duration_ms_by_score_mast_down,
        directional_mast_duration_maps,
        turn_eff,
        cmd_remap,
        act_duration_ms,
        auto_turn_speed_boost_pct,
        turn_intensity_posts_left,
        turn_intensity_posts_right,
        mast_intensity_posts_up,
        mast_intensity_posts_down,
        min_pwm,
        max_pwm,
        min_turn_power,
    )


(
    HOTKEY_SPEED_SCORES,
    SCORE_POWER_PWM_DRIVE,
    SCORE_POWER_PWM_TURN,
    SCORE_POWER_PWM_TURN_LEFT,
    SCORE_POWER_PWM_TURN_RIGHT,
    USE_DIRECTIONAL_TURN_MAPS,
    SPEED_SCORE_DURATION_MS,
    SPEED_SCORE_DURATION_MS_TURN,
    SPEED_SCORE_DURATION_MS_TURN_LEFT,
    SPEED_SCORE_DURATION_MS_TURN_RIGHT,
    USE_DIRECTIONAL_TURN_DURATION_MAPS,
    SPEED_SCORE_DURATION_MS_MAST_UP,
    SPEED_SCORE_DURATION_MS_MAST_DOWN,
    USE_DIRECTIONAL_MAST_DURATION_MAPS,
    TURN_EFFICIENCY,
    COMMAND_REMAP,
    ACT_DURATION_MS,
    AUTO_TURN_SPEED_BOOST_PCT,
    TURN_INTENSITY_POSTS_LEFT,
    TURN_INTENSITY_POSTS_RIGHT,
    MAST_INTENSITY_POSTS_UP,
    MAST_INTENSITY_POSTS_DOWN,
    MIN_PWM,
    MAX_PWM,
    MIN_TURN_POWER,
) = _load_speed_model(ROBOT_MODEL_FILE)

_SCORE_POWER_PWM_TURN_LOADED_ID = id(SCORE_POWER_PWM_TURN)
_SPEED_SCORE_DURATION_MS_LOADED_ID = id(SPEED_SCORE_DURATION_MS)

SCORE_POWER_PWM = SCORE_POWER_PWM_DRIVE

def _baseline_pwm_min(score_map):
    if not isinstance(score_map, dict):
        return 0
    entry = score_map.get(SPEED_SCORE_MIN)
    if not isinstance(entry, dict):
        return 0
    pwm = entry.get("pwm")
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(255, pwm_val))

SCORE_POWER_PWM_DRIVE_BASE = {}
SCORE_POWER_PWM_TURN_BASE = {}
SCORE_POWER_PWM_TURN_LEFT_BASE = {}
SCORE_POWER_PWM_TURN_RIGHT_BASE = {}
SPEED_SCORE_MIN_PWM_DRIVE_BASE = 0
SPEED_SCORE_MIN_PWM_TURN_BASE = 0
SPEED_SCORE_MIN_PWM_TURN_LEFT_BASE = 0
SPEED_SCORE_MIN_PWM_TURN_RIGHT_BASE = 0


def _safe_floor_max(*values):
    floors = []
    for value in values:
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if v > 0:
            floors.append(v)
    return max(floors) if floors else 0


def refresh_speed_model_baseline(*, drive_map=None, turn_map=None, turn_left_map=None, turn_right_map=None):
    """
    Snapshot the current speed model as the baseline (used as a hard floor so
    runtime micro-adjustments can never make the robot move slower than the
    configured 1% speed score).
    """
    global SCORE_POWER_PWM_DRIVE_BASE
    global SCORE_POWER_PWM_TURN_BASE
    global SCORE_POWER_PWM_TURN_LEFT_BASE
    global SCORE_POWER_PWM_TURN_RIGHT_BASE
    global SPEED_SCORE_MIN_PWM_DRIVE_BASE
    global SPEED_SCORE_MIN_PWM_TURN_BASE
    global SPEED_SCORE_MIN_PWM_TURN_LEFT_BASE
    global SPEED_SCORE_MIN_PWM_TURN_RIGHT_BASE

    src_drive = drive_map if isinstance(drive_map, dict) else SCORE_POWER_PWM_DRIVE
    src_turn = turn_map if isinstance(turn_map, dict) else SCORE_POWER_PWM_TURN
    src_turn_left = turn_left_map if isinstance(turn_left_map, dict) else SCORE_POWER_PWM_TURN_LEFT
    src_turn_right = turn_right_map if isinstance(turn_right_map, dict) else SCORE_POWER_PWM_TURN_RIGHT

    legacy_override = (
        isinstance(SCORE_POWER_PWM_TURN, dict)
        and id(SCORE_POWER_PWM_TURN) != _SCORE_POWER_PWM_TURN_LOADED_ID
    )
    if (not bool(USE_DIRECTIONAL_TURN_MAPS)) or legacy_override:
        src_turn_left = src_turn
        src_turn_right = src_turn

    if not isinstance(src_turn_left, dict):
        src_turn_left = src_turn
    if not isinstance(src_turn_right, dict):
        src_turn_right = src_turn

    SCORE_POWER_PWM_DRIVE_BASE = copy.deepcopy(src_drive) if isinstance(src_drive, dict) else {}
    SCORE_POWER_PWM_TURN_BASE = copy.deepcopy(src_turn) if isinstance(src_turn, dict) else {}
    SCORE_POWER_PWM_TURN_LEFT_BASE = copy.deepcopy(src_turn_left) if isinstance(src_turn_left, dict) else {}
    SCORE_POWER_PWM_TURN_RIGHT_BASE = copy.deepcopy(src_turn_right) if isinstance(src_turn_right, dict) else {}
    SPEED_SCORE_MIN_PWM_DRIVE_BASE = int(_baseline_pwm_min(SCORE_POWER_PWM_DRIVE_BASE))
    SPEED_SCORE_MIN_PWM_TURN_BASE = int(_baseline_pwm_min(SCORE_POWER_PWM_TURN_BASE))
    SPEED_SCORE_MIN_PWM_TURN_LEFT_BASE = int(_baseline_pwm_min(SCORE_POWER_PWM_TURN_LEFT_BASE))
    SPEED_SCORE_MIN_PWM_TURN_RIGHT_BASE = int(_baseline_pwm_min(SCORE_POWER_PWM_TURN_RIGHT_BASE))
    SPEED_SCORE_MIN_PWM_TURN_BASE = _safe_floor_max(
        SPEED_SCORE_MIN_PWM_TURN_BASE,
        SPEED_SCORE_MIN_PWM_TURN_LEFT_BASE,
        SPEED_SCORE_MIN_PWM_TURN_RIGHT_BASE,
    )
    return SPEED_SCORE_MIN_PWM_DRIVE_BASE, SPEED_SCORE_MIN_PWM_TURN_BASE


refresh_speed_model_baseline()


def is_valid_speed_score(score):
    try:
        value = int(round(float(score)))
    except (TypeError, ValueError):
        return False
    return SPEED_SCORE_MIN <= value <= SPEED_SCORE_MAX


def _legacy_turn_map_overridden():
    legacy = SCORE_POWER_PWM_TURN
    if not isinstance(legacy, dict):
        return False
    return id(legacy) != _SCORE_POWER_PWM_TURN_LOADED_ID


def _legacy_duration_map_overridden():
    legacy = SPEED_SCORE_DURATION_MS
    if not isinstance(legacy, dict):
        return False
    return id(legacy) != _SPEED_SCORE_DURATION_MS_LOADED_ID


def score_power_pwm_for_cmd(cmd):
    if cmd in ("l", "r"):
        # Backward compatibility: tests and tools may monkeypatch the legacy
        # shared turn map. If that happens, honor it for both directions.
        if _legacy_turn_map_overridden():
            return SCORE_POWER_PWM_TURN
        if bool(USE_DIRECTIONAL_TURN_MAPS):
            if cmd == "l" and isinstance(SCORE_POWER_PWM_TURN_LEFT, dict):
                return SCORE_POWER_PWM_TURN_LEFT
            if cmd == "r" and isinstance(SCORE_POWER_PWM_TURN_RIGHT, dict):
                return SCORE_POWER_PWM_TURN_RIGHT
        return SCORE_POWER_PWM_TURN if isinstance(SCORE_POWER_PWM_TURN, dict) else SCORE_POWER_PWM_DRIVE
    if cmd in ("f", "b"):
        return SCORE_POWER_PWM_DRIVE if isinstance(SCORE_POWER_PWM_DRIVE, dict) else SCORE_POWER_PWM_TURN
    return SCORE_POWER_PWM_DRIVE if isinstance(SCORE_POWER_PWM_DRIVE, dict) else SCORE_POWER_PWM_TURN


def _hotkey_speed_row_for_cmd_score(cmd, score):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in ("u", "d"):
        return None
    rows = HOTKEY_SPEED_SCORES if isinstance(HOTKEY_SPEED_SCORES, dict) else {}
    if not rows:
        return None
    score_key = normalize_speed_score(score)
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        row_cmd = str(row.get("cmd") or "").strip().lower()
        if row_cmd != cmd_key:
            continue
        try:
            row_score = normalize_speed_score(row.get("score"))
        except Exception:
            continue
        if int(row_score) == int(score_key):
            return row
    return None


def _hotkey_speed_override_for_cmd_score(cmd, score):
    row = _hotkey_speed_row_for_cmd_score(cmd, score)
    if not isinstance(row, dict):
        return None

    pwm = None
    pwm_raw = row.get("pwm")
    try:
        pwm = clamp_pwm(int(round(float(pwm_raw))))
    except (TypeError, ValueError):
        pwm = None
    if pwm is None or int(pwm) <= 0:
        power_raw = row.get("power")
        try:
            power_raw = float(power_raw)
        except (TypeError, ValueError):
            power_raw = None
        if power_raw is not None:
            pwm_from_power = _power_to_pwm(power_raw)
            if pwm_from_power is not None:
                pwm = clamp_pwm(int(pwm_from_power))
    if pwm is None or int(pwm) <= 0:
        return None

    power = _pwm_to_power(pwm)
    if power is None:
        power = 0.0

    duration_ms = None
    duration_raw = row.get("duration_ms")
    try:
        duration_ms = max(1, int(round(float(duration_raw))))
    except (TypeError, ValueError):
        duration_ms = None
    if duration_ms is None:
        duration_ms = _duration_ms_for_score(cmd, score)
    return float(power), int(pwm), int(duration_ms)


def baseline_pwm_floor_for_cmd(cmd):
    cmd = str(cmd) if cmd is not None else ""
    if cmd == "l":
        floor = _safe_floor_max(SPEED_SCORE_MIN_PWM_TURN_LEFT_BASE, SPEED_SCORE_MIN_PWM_TURN_BASE)
        return max(int(turn_pwm_floor()), int(floor))
    if cmd == "r":
        floor = _safe_floor_max(SPEED_SCORE_MIN_PWM_TURN_RIGHT_BASE, SPEED_SCORE_MIN_PWM_TURN_BASE)
        return max(int(turn_pwm_floor()), floor)
    if cmd in ("f", "b"):
        floor = int(SPEED_SCORE_MIN_PWM_DRIVE_BASE or 0)
        return max(0, min(255, floor))
    return 0


def _speed_pwm_endpoints(cmd):
    score_map = score_power_pwm_for_cmd(cmd)
    low_entry = score_map.get(SPEED_SCORE_MIN) if isinstance(score_map, dict) else None
    high_entry = score_map.get(SPEED_SCORE_MAX) if isinstance(score_map, dict) else None
    low_pwm = low_entry.get("pwm") if isinstance(low_entry, dict) else None
    high_pwm = high_entry.get("pwm") if isinstance(high_entry, dict) else None
    try:
        low_pwm = int(low_pwm)
    except (TypeError, ValueError):
        low_pwm = None
    try:
        high_pwm = int(high_pwm)
    except (TypeError, ValueError):
        high_pwm = None
    return low_pwm, high_pwm


def _duration_map_for_cmd(cmd):
    # Backward compatibility: if callers monkeypatch the shared map, honor it.
    if _legacy_duration_map_overridden():
        return SPEED_SCORE_DURATION_MS
    if cmd in ("l", "r"):
        if bool(USE_DIRECTIONAL_TURN_DURATION_MAPS):
            if cmd == "l" and isinstance(SPEED_SCORE_DURATION_MS_TURN_LEFT, dict):
                return SPEED_SCORE_DURATION_MS_TURN_LEFT
            if cmd == "r" and isinstance(SPEED_SCORE_DURATION_MS_TURN_RIGHT, dict):
                return SPEED_SCORE_DURATION_MS_TURN_RIGHT
        if isinstance(SPEED_SCORE_DURATION_MS_TURN, dict) and SPEED_SCORE_DURATION_MS_TURN:
            return SPEED_SCORE_DURATION_MS_TURN
        return SPEED_SCORE_DURATION_MS
    if cmd in ("u", "d"):
        if bool(USE_DIRECTIONAL_MAST_DURATION_MAPS):
            if cmd == "u" and isinstance(SPEED_SCORE_DURATION_MS_MAST_UP, dict):
                return SPEED_SCORE_DURATION_MS_MAST_UP
            if cmd == "d" and isinstance(SPEED_SCORE_DURATION_MS_MAST_DOWN, dict):
                return SPEED_SCORE_DURATION_MS_MAST_DOWN
        return SPEED_SCORE_DURATION_MS
    return SPEED_SCORE_DURATION_MS


def _duration_ms_for_score(cmd, score):
    score = normalize_speed_score(score)
    duration_map = _duration_map_for_cmd(cmd)
    exact = duration_map.get(score) if isinstance(duration_map, dict) else None
    if exact is not None:
        try:
            return max(1, int(round(exact)))
        except (TypeError, ValueError):
            pass
    low = duration_map.get(SPEED_SCORE_MIN) if isinstance(duration_map, dict) else None
    high = duration_map.get(SPEED_SCORE_MAX) if isinstance(duration_map, dict) else None
    if low is None and high is None:
        return int(ACT_DURATION_MS)
    if low is None:
        try:
            return max(1, int(round(float(high))))
        except (TypeError, ValueError):
            return int(ACT_DURATION_MS)
    if high is None:
        try:
            return max(1, int(round(float(low))))
        except (TypeError, ValueError):
            return int(ACT_DURATION_MS)
    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return int(ACT_DURATION_MS)
    frac = (float(score) - float(SPEED_SCORE_MIN)) / float(SPEED_SCORE_MAX - SPEED_SCORE_MIN)
    return max(1, int(round(low + (high - low) * frac)))


def speed_power_pwm_for_cmd(cmd, score):
    score = normalize_speed_score(score)
    hotkey_override = _hotkey_speed_override_for_cmd_score(cmd, score)
    if isinstance(hotkey_override, tuple) and len(hotkey_override) == 3:
        power, pwm, duration_ms = hotkey_override
        return float(power), int(pwm), int(score), int(duration_ms)
    score_map = score_power_pwm_for_cmd(cmd)
    exact_entry = score_map.get(score) if isinstance(score_map, dict) else None
    pwm = None
    if isinstance(exact_entry, dict):
        try:
            pwm = int(round(float(exact_entry.get("pwm"))))
        except (TypeError, ValueError):
            pwm = None
    if pwm is None:
        low_pwm, high_pwm = _speed_pwm_endpoints(cmd)
        if low_pwm is None or high_pwm is None:
            return 0.0, 0, score, int(ACT_DURATION_MS)
        pwm = interp_pwm_for_score(score, low_pwm, high_pwm)
        if pwm is None:
            return 0.0, 0, score, int(ACT_DURATION_MS)
    if cmd in ("l", "r") and pwm > 0:
        pwm = max(turn_pwm_floor(), pwm)
    if cmd in ("f", "b", "l", "r") and pwm > 0:
        pwm = max(int(baseline_pwm_floor_for_cmd(cmd)), int(pwm))
    power = _pwm_to_power(pwm) or 0.0
    duration_ms = _duration_ms_for_score(cmd, score)
    return power, pwm, score, duration_ms


def quantize_speed(cmd, speed=None, score=None):
    if score is not None:
        power, _, score_used, _ = speed_power_pwm_for_cmd(cmd, score)
        return power, score_used
    if speed is None:
        return 0.0, None
    pwm = _power_to_pwm(speed)
    if pwm is None:
        return 0.0, None
    _, low_pwm, _, _ = speed_power_pwm_for_cmd(cmd, SPEED_SCORE_MIN)
    _, high_pwm, _, _ = speed_power_pwm_for_cmd(cmd, SPEED_SCORE_MAX)
    if high_pwm <= low_pwm:
        power, _, score_used, _ = speed_power_pwm_for_cmd(cmd, SPEED_SCORE_MIN)
        return power, score_used
    frac = (float(pwm) - float(low_pwm)) / float(high_pwm - low_pwm)
    score_used = normalize_speed_score(SPEED_SCORE_MIN + frac * float(SPEED_SCORE_MAX - SPEED_SCORE_MIN))
    power, _, _, _ = speed_power_pwm_for_cmd(cmd, score_used)
    return power, int(score_used)


def manual_speed_for_cmd(cmd, score):
    power, _, _, _ = speed_power_pwm_for_cmd(cmd, score)
    return power


def ease_in_out_min_score_for_cmd(cmd):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key in ("l", "r"):
        return normalize_speed_score(MOTION_EASE_IN_OUT_MIN_SCORE_TURN)
    if cmd_key in ("f", "b"):
        return normalize_speed_score(MOTION_EASE_IN_OUT_MIN_SCORE_DRIVE)
    return normalize_speed_score(MOTION_EASE_IN_OUT_MIN_SCORE_DRIVE)


def ease_in_out_ramp_ms_for_score(score, *, cmd=None, min_score=None):
    try:
        score_val = normalize_speed_score(score)
    except Exception:
        score_val = normalize_speed_score(SPEED_SCORE_DEFAULT)
    if min_score is None:
        if cmd is not None:
            min_score = ease_in_out_min_score_for_cmd(cmd)
        else:
            min_score = MOTION_EASE_IN_OUT_MIN_SCORE_DRIVE
    min_score = normalize_speed_score(min_score)
    max_score = normalize_speed_score(SPEED_SCORE_MAX)
    try:
        min_ramp_ms = max(1, int(round(float(MOTION_EASE_IN_OUT_MIN_RAMP_MS))))
    except (TypeError, ValueError):
        min_ramp_ms = 300
    try:
        max_ramp_ms = max(min_ramp_ms, int(round(float(MOTION_EASE_IN_OUT_MAX_RAMP_MS))))
    except (TypeError, ValueError):
        max_ramp_ms = max(int(min_ramp_ms), 800)
    span = max(1, int(max_score) - int(min_score))
    frac = max(0.0, min(1.0, (float(score_val) - float(min_score)) / float(span)))
    return int(round(float(min_ramp_ms) + (float(max_ramp_ms) - float(min_ramp_ms)) * float(frac)))


def drive_ease_in_out_segments(cmd, *, speed_score=None, power=None, pwm=None, duration_ms=None):
    """
    Build a score-ramped PWM envelope for chassis commands (`f`/`b`/`l`/`r`).

    The envelope ramps up at the start and ramps down at the end while preserving
    total command duration. If the pulse is shorter than 2*ramp, ramps are
    shortened symmetrically (triangle profile).

    Returns None when easing should not apply, otherwise a dict with a
    `segments` list suitable for sequential sending.
    """
    if not bool(MOTION_EASE_IN_OUT_ENABLED):
        return None
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in ("f", "b", "l", "r"):
        return None
    try:
        total_ms = max(1, int(round(float(duration_ms or 0))))
    except (TypeError, ValueError):
        return None
    if total_ms <= 0:
        return None

    target_score = None
    try:
        if speed_score is not None:
            target_score = int(normalize_speed_score(speed_score))
    except (TypeError, ValueError):
        target_score = None
    if target_score is None:
        speed_guess = None
        if power is not None:
            try:
                speed_guess = float(power)
            except (TypeError, ValueError):
                speed_guess = None
        if speed_guess is None and pwm is not None:
            try:
                speed_guess = float(pwm_to_power(pwm) or 0.0)
            except (TypeError, ValueError):
                speed_guess = None
        if speed_guess is not None and speed_guess > 0.0:
            try:
                _, q_score = quantize_speed(cmd_key, speed=speed_guess)
                if q_score is not None:
                    target_score = int(normalize_speed_score(q_score))
            except Exception:
                target_score = None
    if target_score is None:
        return None
    threshold_score = int(ease_in_out_min_score_for_cmd(cmd_key))
    if int(target_score) < int(threshold_score):
        return None

    ramp_ms_cfg = ease_in_out_ramp_ms_for_score(target_score, min_score=threshold_score)
    ramp_ms_eff = min(int(ramp_ms_cfg), max(1, int(total_ms // 2)))
    if ramp_ms_eff <= 0:
        return None

    try:
        ramp_steps = max(1, int(round(float(MOTION_EASE_IN_OUT_RAMP_STEPS))))
    except (TypeError, ValueError):
        ramp_steps = 3

    def _split_even(total_local_ms, parts):
        parts = max(1, int(parts))
        total_local_ms = max(0, int(total_local_ms))
        base = total_local_ms // parts
        rem = total_local_ms % parts
        out = []
        for idx in range(parts):
            out.append(int(base + (1 if idx < rem else 0)))
        return out

    up_durations = _split_even(int(ramp_ms_eff), int(ramp_steps))
    down_durations = _split_even(int(ramp_ms_eff), int(ramp_steps))
    plateau_ms = max(0, int(total_ms) - int(sum(up_durations)) - int(sum(down_durations)))

    chunks = []
    for idx, chunk_ms in enumerate(up_durations, start=1):
        if chunk_ms <= 0:
            continue
        factor = float(idx) / float(max(1, int(ramp_steps)))
        chunks.append((int(chunk_ms), float(factor)))
    if plateau_ms > 0:
        chunks.append((int(plateau_ms), 1.0))
    for idx, chunk_ms in enumerate(down_durations, start=0):
        if chunk_ms <= 0:
            continue
        factor = float(max(1, int(ramp_steps) - idx)) / float(max(1, int(ramp_steps)))
        chunks.append((int(chunk_ms), float(factor)))
    if len(chunks) <= 1:
        return None

    segments = []
    for chunk_ms, factor in chunks:
        seg_score = int(round(float(target_score) * float(factor)))
        if seg_score <= 0:
            seg_score = int(SPEED_SCORE_MIN)
        seg_score = int(normalize_speed_score(seg_score))
        seg_power, seg_pwm, _, _ = speed_power_pwm_for_cmd(cmd_key, seg_score)
        segments.append(
            {
                "cmd": str(cmd_key),
                "score_model": int(seg_score),
                "power": float(seg_power or 0.0),
                "pwm": int(seg_pwm or 0),
                "duration_ms": int(chunk_ms),
            }
        )

    if not segments:
        return None

    # Merge adjacent chunks that quantize to the same PWM/score to reduce serial chatter.
    merged = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        same_profile = (
            str(prev.get("cmd")) == str(seg.get("cmd"))
            and int(prev.get("pwm") or 0) == int(seg.get("pwm") or 0)
            and int(prev.get("score_model") or 0) == int(seg.get("score_model") or 0)
        )
        if same_profile:
            prev["duration_ms"] = int(prev.get("duration_ms") or 0) + int(seg.get("duration_ms") or 0)
        else:
            merged.append(dict(seg))

    return {
        "cmd": str(cmd_key),
        "target_score": int(target_score),
        "threshold_score": int(threshold_score),
        "ramp_ms": int(ramp_ms_cfg),
        "ramp_ms_effective": int(ramp_ms_eff),
        "slice_ms": (int(round(float(ramp_ms_eff) / float(ramp_steps))) if ramp_steps > 0 else None),
        "ramp_steps": int(ramp_steps),
        "segments": merged,
    }


def drive_anti_alias_segments(cmd, *, speed_score=None, power=None, pwm=None, duration_ms=None):
    """Backward-compatible wrapper for older anti-alias naming."""
    return drive_ease_in_out_segments(
        cmd,
        speed_score=speed_score,
        power=power,
        pwm=pwm,
        duration_ms=duration_ms,
    )


def _motion_intensity_posts_for_cmd(cmd):
    if cmd == "l":
        return TURN_INTENSITY_POSTS_LEFT if isinstance(TURN_INTENSITY_POSTS_LEFT, dict) else {}
    if cmd == "r":
        return TURN_INTENSITY_POSTS_RIGHT if isinstance(TURN_INTENSITY_POSTS_RIGHT, dict) else {}
    if cmd == "u":
        return MAST_INTENSITY_POSTS_UP if isinstance(MAST_INTENSITY_POSTS_UP, dict) else {}
    if cmd == "d":
        return MAST_INTENSITY_POSTS_DOWN if isinstance(MAST_INTENSITY_POSTS_DOWN, dict) else {}
    return {}


def _intensity_posts_for_cmd(cmd):
    return _motion_intensity_posts_for_cmd(cmd)


def speed_power_pwm_for_motion_intensity(cmd, intensity_pct):
    """
    Fractional motion intensity path for L/R/U/D commands.
    Uses piecewise interpolation over configured intensity anchor posts.
    Returns (power, pwm, score_estimate, duration_ms, intensity_effective).
    """
    if cmd not in ("l", "r", "u", "d"):
        power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, intensity_pct)
        return power, pwm, score_used, duration_ms, None

    posts = _motion_intensity_posts_for_cmd(cmd)
    if not posts:
        power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, intensity_pct)
        try:
            intensity_eff = float(intensity_pct)
        except (TypeError, ValueError):
            intensity_eff = float(score_used)
        return power, pwm, score_used, duration_ms, intensity_eff

    try:
        intensity = float(intensity_pct)
    except (TypeError, ValueError):
        intensity = 1.0

    levels = sorted(float(level) for level in posts.keys())
    if not levels:
        power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, intensity_pct)
        return power, pwm, score_used, duration_ms, float(intensity)

    def _v(entry, key, default):
        try:
            return float(entry.get(key, default))
        except (TypeError, ValueError, AttributeError):
            return float(default)

    # A discovered micro profile may persist only a single tiny anchor (e.g., 0.01%).
    # Preserve that anchor for tiny requests, but use score-based scaling for normal
    # turn intensities so ALIGN logic is not flattened to one PWM/duration.
    if len(levels) == 1:
        only_level = float(levels[0])
        only = posts.get(float(only_level)) or {}
        if float(intensity) <= float(only_level):
            pwm_val = int(round(_v(only, "pwm", 0.0)))
            pwm_val = clamp_pwm(pwm_val)
            if pwm_val > 0:
                if cmd in ("l", "r"):
                    pwm_val = max(turn_pwm_floor(), int(pwm_val))
                pwm_val = max(int(baseline_pwm_floor_for_cmd(cmd)), int(pwm_val))
            duration_ms = max(1, int(round(_v(only, "duration_ms", DEFAULT_ACT_DURATION_MS))))
            power = _pwm_to_power(pwm_val)
            if power is None:
                power = 0.0
            _, score_est = quantize_speed(cmd, speed=power)
            if score_est is None:
                score_est = SPEED_SCORE_MIN
            return float(power), int(pwm_val), int(score_est), int(duration_ms), float(only_level)

        power, pwm_val, score_est, duration_ms = speed_power_pwm_for_cmd(cmd, intensity)
        anchor_duration_ms = max(1, int(round(_v(only, "duration_ms", float(duration_ms)))))
        duration_ms = max(int(duration_ms), int(anchor_duration_ms))
        return float(power), int(pwm_val), int(score_est), int(duration_ms), float(intensity)

    intensity_eff = max(float(levels[0]), min(float(levels[-1]), float(intensity)))

    low_level = levels[0]
    high_level = levels[-1]
    for level in levels:
        if level <= intensity_eff:
            low_level = level
        if level >= intensity_eff:
            high_level = level
            break

    low = posts.get(float(low_level)) or {}
    high = posts.get(float(high_level)) or low

    if float(high_level) <= float(low_level):
        frac = 0.0
    else:
        frac = (float(intensity_eff) - float(low_level)) / (float(high_level) - float(low_level))
        frac = max(0.0, min(1.0, frac))

    low_pwm = _v(low, "pwm", 0.0)
    high_pwm = _v(high, "pwm", low_pwm)
    pwm_val = int(round(low_pwm + (high_pwm - low_pwm) * frac))
    pwm_val = clamp_pwm(pwm_val)
    if pwm_val > 0:
        if cmd in ("l", "r"):
            pwm_val = max(turn_pwm_floor(), int(pwm_val))
        pwm_val = max(int(baseline_pwm_floor_for_cmd(cmd)), int(pwm_val))

    low_ms = _v(low, "duration_ms", DEFAULT_ACT_DURATION_MS)
    high_ms = _v(high, "duration_ms", low_ms)
    duration_ms = max(1, int(round(low_ms + (high_ms - low_ms) * frac)))

    power = _pwm_to_power(pwm_val)
    if power is None:
        power = 0.0
    _, score_est = quantize_speed(cmd, speed=power)
    if score_est is None:
        score_est = SPEED_SCORE_MIN
    return float(power), int(pwm_val), int(score_est), int(duration_ms), float(intensity_eff)


def speed_power_pwm_for_turn_intensity(cmd, intensity_pct):
    """
    Backward-compatible alias for older L/R turn-intensity callers.
    """
    return speed_power_pwm_for_motion_intensity(cmd, intensity_pct)


def turn_duration_scale(cmd, turn_efficiency=None, min_scale=0.5, max_scale=1.5):
    """
    Convert turn efficiency into a duration multiplier.
    Higher efficiency => shorter duration, lower efficiency => longer duration.
    """
    if cmd not in ("l", "r"):
        return 1.0
    eff_map = turn_efficiency if isinstance(turn_efficiency, dict) else TURN_EFFICIENCY
    try:
        left = float(eff_map.get("l"))
        right = float(eff_map.get("r"))
    except (TypeError, ValueError, AttributeError):
        return 1.0
    if left <= 0 or right <= 0:
        return 1.0
    avg = (left + right) / 2.0
    cmd_eff = left if cmd == "l" else right
    if cmd_eff <= 0:
        return 1.0
    raw = avg / cmd_eff
    return max(float(min_scale), min(float(max_scale), float(raw)))


def manual_key_action(key):
    entry = HOTKEY_SPEED_SCORES.get(key)
    if not entry:
        return None
    return entry["cmd"], entry["score"]


def _step_name(step):
    return _brick_module()._step_name(step)


def _build_envelope(process_rules, learned_rules, step):
    return _brick_module().build_envelope(process_rules, learned_rules, step)

METRICS_BY_STEP = {
    "ELEVATE_BRICK": ("lift_height",),
    "LIFT": ("lift_height",),
    "RETREAT": ("lift_height",),
    "PLACE": ("lift_height",),
}

METRIC_DIRECTIONS = {
    "lift_height": "band",
}


def resolve_scan_direction(process_rules, step, fallback=None):
    step_key = _step_name(step)
    cfg = {}
    if isinstance(process_rules, dict):
        raw_cfg = process_rules.get(step_key)
        if isinstance(raw_cfg, dict):
            cfg = raw_cfg

    scan_dir = cfg.get("scan_direction")
    if isinstance(scan_dir, str):
        scan_key = scan_dir.strip().lower()
        if scan_key in ("l", "left"):
            return "l"
        if scan_key in ("r", "right"):
            return "r"

    if isinstance(fallback, str):
        fallback_key = fallback.strip().lower()
        if fallback_key in ("l", "left"):
            return "l"
        if fallback_key in ("r", "right"):
            return "r"
    return None


def _target_tol_ok(value, stats, direction):
    target = stats.get("target") if isinstance(stats, dict) else None
    tol = stats.get("tol") if isinstance(stats, dict) else None
    if target is None or tol is None:
        return None
    if direction == "high":
        return value >= (target - tol)
    if direction == "low":
        return value <= (target + tol)
    return abs(value - target) <= tol


@dataclass
class MotionDelta:
    dist_mm: float = 0.0
    rot_deg: float = 0.0
    lift_mm: float = 0.0




def evaluate_start_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    return GateCheck(ok=True)


def evaluate_success_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = _build_envelope(process_rules or {}, learned_rules or {}, step)
    success_metrics = envelope.get("success") or {}
    if not success_metrics:
        return GateCheck(ok=False, reasons=["no lift success envelope"])
    stats = success_metrics.get("lift_height") or {}
    lift = world.lift_height
    ok = _target_tol_ok(lift, stats, METRIC_DIRECTIONS.get("lift_height"))
    if ok is False:
        return GateCheck(ok=False, reasons=["lift gate"])
    if ok is None:
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and lift < min_val:
            return GateCheck(ok=False, reasons=[f"lift<{min_val:.1f}mm"])
        if max_val is not None and lift > max_val:
            return GateCheck(ok=False, reasons=[f"lift>{max_val:.1f}mm"])
    return GateCheck(ok=True)


def evaluate_failure_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = _build_envelope(process_rules or {}, learned_rules or {}, step)
    failure_metrics = envelope.get("failure") or {}
    stats = failure_metrics.get("lift_height")
    if not stats:
        return GateCheck(ok=True)
    lift = world.lift_height
    min_val = stats.get("min")
    max_val = stats.get("max")
    reasons = []
    if min_val is not None and lift < min_val:
        reasons.append(f"lift<{min_val:.1f}mm")
    if max_val is not None and lift > max_val:
        reasons.append(f"lift>{max_val:.1f}mm")
    return GateCheck(ok=not reasons, reasons=reasons)


def update_from_motion(world, event):
    dt = event.duration_ms / 1000.0
    power_ratio = event.power / 255.0
    dist_pulse = 0.0
    rot_pulse = 0.0
    lift_pulse = 0.0

    if event.action_type == "forward":
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt
        rad = math.radians(world.theta)
        world.x += dist_pulse * math.cos(rad)
        world.y += dist_pulse * math.sin(rad)
    elif event.action_type == "backward":
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt
        rad = math.radians(world.theta)
        world.x -= dist_pulse * math.cos(rad)
        world.y -= dist_pulse * math.sin(rad)
    elif event.action_type == "left_turn":
        # Apply turn efficiency if available
        # Experiment found L ~88, R ~59. Scale relatively to deg_per_sec_full_speed.
        # If we use TURN_EFFICIENCY directly as a multiplier for deg_per_sec? 
        # Actually deg_per_sec_full_speed is already a 'speed'. 1.0 power = 90 deg/sec.
        # Let's use it as a 0-1 multiplier or scale relative to a baseline.
        # For now, let's just make it a direct component of the pulse.
        eff_l = world.turn_efficiency_l / 100.0 # Normalize around 100
        rot_pulse = world.deg_per_sec_full_speed * power_ratio * dt * 0.5 * eff_l
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt * 0.5
        rad = math.radians(world.theta)
        world.x += dist_pulse * math.cos(rad)
        world.y += dist_pulse * math.sin(rad)
        world.theta += rot_pulse
    elif event.action_type == "right_turn":
        eff_r = world.turn_efficiency_r / 100.0
        rot_pulse = world.deg_per_sec_full_speed * power_ratio * dt * 0.5 * eff_r
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt * 0.5
        rad = math.radians(world.theta)
        world.x += dist_pulse * math.cos(rad)
        world.y += dist_pulse * math.sin(rad)
        world.theta -= rot_pulse
    elif event.action_type == "mast_up":
        lift_pulse = world.lift_mm_per_sec * power_ratio * dt
        world.lift_height += lift_pulse
    elif event.action_type == "mast_down":
        lift_pulse = world.lift_mm_per_sec * power_ratio * dt
        world.lift_height -= lift_pulse
        if world.lift_height < 0:
            world.lift_height = 0

    return MotionDelta(dist_mm=dist_pulse, rot_deg=rot_pulse, lift_mm=lift_pulse)


def update_lift_from_vision(
    world,
    cam_h,
    brick_height,
    conf,
):
    if cam_h <= 0 or conf < 50:
        world.lift_height_source = "dead_reckon"
        world.lift_height_quality = 0.0
        return
    brick_height = brick_height or 0.0
    if world.lift_height_anchor is None:
        world.lift_height_anchor = cam_h - world.lift_height + brick_height

    vis_lift = cam_h - world.lift_height_anchor + brick_height
    world.lift_height = (0.9 * world.lift_height) + (0.1 * vis_lift)
    world.lift_height_source = "aruco_cam_h"
    world.lift_height_quality = 1.0

class StepState(Enum):
    FIND_WALL = "FIND_WALL"
    EXIT_WALL = "EXIT_WALL"
    FIND_BRICK = "FIND_BRICK"
    APPROACH_VECTOR_BRICK_SUPPLY = "APPROACH_VECTOR_BRICK_SUPPLY"
    FIND_TOPMOST_BRICK = "FIND_TOPMOST_BRICK"
    BRICK_LOCK = "BRICK_LOCK"
    ALIGN_BRICK = "ALIGN_BRICK"
    SEAT_BRICK = "SEAT_BRICK"
    SCOOP = "SEAT_BRICK"
    ELEVATE_BRICK = "ELEVATE_BRICK"
    LIFT = "ELEVATE_BRICK"
    FIND_WALL2 = "FIND_WALL2"
    APPROACH_VECTOR_WALL = "APPROACH_VECTOR_WALL"
    FIND_TOPMOST_BRICK_WALL = "FIND_TOPMOST_BRICK_WALL"
    BRICK_LOCK_WALL = "BRICK_LOCK_WALL"
    POSITION_BRICK = "POSITION_BRICK"
    SEAT_BRICK2 = "SEAT_BRICK2"
    RETREAT = "RETREAT"
    PLACE = "RETREAT"

def _cmd_for_action_type(action_type):
    return {
        "forward": "f",
        "backward": "b",
        "left_turn": "l",
        "right_turn": "r",
        "mast_up": "u",
        "mast_down": "d",
    }.get(action_type)

class MotionEvent:
    def __init__(self, action_type, power=None, duration_ms=0, speed_score=None):
        self.action_type = action_type
        self.duration_ms = int(duration_ms) if duration_ms is not None else 0
        self.timestamp = time.time()
        self.speed_score = None
        self.power = 0

        if speed_score is not None:
            try:
                self.speed_score = int(speed_score)
            except (TypeError, ValueError):
                self.speed_score = None

        if power is not None:
            try:
                self.power = int(power)
            except (TypeError, ValueError):
                self.power = 0
        elif self.speed_score is not None:
            cmd = _cmd_for_action_type(self.action_type)
            if cmd:
                power_val, _, _, _ = speed_power_pwm_for_cmd(cmd, self.speed_score)
                self.power = int(power_val * 255)

        if self.action_type in ("left_turn", "right_turn") and 0 < self.power < MIN_TURN_POWER_PWM:
            self.power = MIN_TURN_POWER_PWM

        if self.speed_score is None and self.power:
            cmd = _cmd_for_action_type(self.action_type)
            if cmd:
                _, score_used = quantize_speed(cmd, speed=self.power / 255.0)
                self.speed_score = score_used

    def to_dict(self):
        return {
            "type": self.action_type,
            "speedScore": self.speed_score,
            "timestamp": round(self.timestamp, 3)
        }

MOTION_EVENT_TYPES = {
    "forward",
    "backward",
    "left_turn",
    "right_turn",
    "mast_up",
    "mast_down"
}

WORLD_MODEL_PROCESS_FILE = Path(__file__).parent / "world_model_process.json"
WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"
WORLD_MODEL_MOTION_FILE = Path(__file__).parent / "world_model_motion.json"

DEFAULT_MM_PER_SEC_FULL_SPEED = 200.0
DEFAULT_DEG_PER_SEC_FULL_SPEED = 90.0
DEFAULT_LIFT_MM_PER_SEC = 23.5
DEFAULT_MOTION_TICK_MS = 100.0
MIN_TURN_POWER_PWM = int(math.ceil(MIN_TURN_POWER * 255))


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_motion_calibration(path=WORLD_MODEL_MOTION_FILE):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    motion = (data.get("calibration") or {}).get("motion") or {}
    return motion if isinstance(motion, dict) else {}


def motion_speeds_from_calibration(motion):
    if not isinstance(motion, dict):
        motion = {}

    mm_per_sec = _coerce_float(motion.get("mm_per_sec_full_speed"))
    deg_per_sec = _coerce_float(motion.get("deg_per_sec_full_speed"))
    lift_per_sec = _coerce_float(motion.get("mm_per_sec_mast"))

    tick_ms = _coerce_float(
        motion.get("tick_ms")
        or motion.get("command_duration_ms")
        or motion.get("cmd_duration_ms")
    )
    if tick_ms is None or tick_ms <= 0:
        tick_ms = DEFAULT_MOTION_TICK_MS
    tick_s = tick_ms / 1000.0

    if mm_per_sec is None:
        mm_per_tick = _coerce_float(motion.get("mm_per_tick"))
        if mm_per_tick is not None:
            mm_per_sec = mm_per_tick / tick_s
    if deg_per_sec is None:
        deg_per_tick = _coerce_float(motion.get("deg_per_tick"))
        if deg_per_tick is not None:
            deg_per_sec = deg_per_tick / tick_s
    if lift_per_sec is None:
        mm_per_tick_mast = _coerce_float(motion.get("mm_per_tick_mast"))
        if mm_per_tick_mast is not None:
            lift_per_sec = mm_per_tick_mast / tick_s

    return mm_per_sec, deg_per_sec, lift_per_sec

def _load_process_step_names():
    if not WORLD_MODEL_PROCESS_FILE.exists():
        return []
    try:
        with open(WORLD_MODEL_PROCESS_FILE, 'r') as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    steps = model.get("steps", {})
    if isinstance(steps, dict):
        return list(steps.keys())
    return []

def step_sequence():
    names = _load_process_step_names()
    if names:
        sequence = []
        seen = set()
        for name in names:
            normalized = _step_name(name)
            if normalized in StepState.__members__:
                obj = StepState[normalized]
                if obj not in seen:
                    sequence.append(obj)
                    seen.add(obj)
        if sequence:
            return sequence
    return list(StepState)

class WorldModel:
    def __init__(self):
        # Load Process Rules
        self.process_rules = {}
        if WORLD_MODEL_PROCESS_FILE.exists():
            try:
                with open(WORLD_MODEL_PROCESS_FILE, 'r') as f:
                    self.process_rules = json.load(f).get("steps", {})
            except: pass
        self.rules = self.process_rules
            
        self.learned_rules = {} # Rules derived from demo analysis
            
        # Robot Pose (Dead Reckoning)
        self.x = 0.0 # mm
        self.y = 0.0 # mm
        self.theta = 0.0 # degrees

        # Wall Model
        wall_module = _wall_module()
        self.wall_model = wall_module.load_wall_model()
        self.wall_envelope = wall_module.build_envelope(self.wall_model)
        self.wall = wall_module.init_wall_state(self.wall_envelope)

        # Brick Data
        self.brick = {
            "visible": False,
            "id": None,
            "dist": 0,
            "angle": 0,
            "offset_x": 0,
            "x_axis": 0,
            "offset_y": 0,
            "y_axis": 0,
            "inCrosshairs": False,
            "confidence": 0,
            "held": False,
            "brickAbove": None,
            "brickBelow": None
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)
        self.camera_height_anchor = None
        self.height_mm = None
        self.lift_height_source = "dead_reckon"
        self.lift_height_quality = 0.0
        # Height intelligence snapshots (derived from topmost-step telemetry).
        self.wall_height_mm = None
        self.wall_height_bricks = None
        self.brick_supply_height_mm = None
        self.brick_supply_height_bricks = None

        # Step
        self._step_state = None
        self._step_start_time = 0
        self._success_start_time = None
        self.step_state = StepState.FIND_BRICK
        self.attempt_status = "NORMAL" # NORMAL, FAIL, RECOVERY
        self.run_id = "unset"
        self.attempt_id = 0
        self.recording_active = False # For HUD prompt logic (Idle vs Success phase)
        
        # Alignment & Stability
        self.align_tol_angle = 5.0    # +/- Degrees
        self.align_tol_offset = 12.0  # +/- mm
        self.align_tol_dist_min = 30.0 # mm (Too close)
        self.align_tol_dist_max = 500.0 # mm (Too far)
        self.scoop_success_offset_factor = 1.2
        self.stability_count = 0
        self.stability_threshold = 10  # 10 frames @ 20Hz = 0.5 seconds
        
        self.last_visible_time = None
        self.scoop_desired_offset_x = 0.0
        self.scoop_lateral_drift = 0.0
        self.scoop_forward_preferred = False
        self.last_seen_angle = None
        self.last_seen_offset_x = None
        self.last_seen_offset_y = None
        self.last_seen_y_axis = None
        self.last_seen_dist = None
        self.last_seen_confidence = None
        
        self.last_image_file = None
        
        # Internal physics constants for dead reckoning (Calibration needed!)
        self.mm_per_sec_full_speed = DEFAULT_MM_PER_SEC_FULL_SPEED
        self.deg_per_sec_full_speed = DEFAULT_DEG_PER_SEC_FULL_SPEED
        self.lift_mm_per_sec = DEFAULT_LIFT_MM_PER_SEC
        motion = load_motion_calibration()
        mm_per_sec, deg_per_sec, lift_per_sec = motion_speeds_from_calibration(motion)
        if mm_per_sec is not None:
            self.mm_per_sec_full_speed = mm_per_sec
        if deg_per_sec is not None:
            self.deg_per_sec_full_speed = deg_per_sec
        if lift_per_sec is not None:
            self.lift_mm_per_sec = lift_per_sec
        self.lift_height_anchor = None # The Vision height at Mast=0mm
        
        # Turn Efficiencies
        self.turn_efficiency_l = TURN_EFFICIENCY.get("l", 100.0)
        self.turn_efficiency_r = TURN_EFFICIENCY.get("r", 100.0)
        
        self.action_history = collections.deque(maxlen=100)
        helper_xyz_coords.ensure_workspace(self)
        helper_xyz_coords.sync_from_world(self, reason="init")

    @property
    def step_state(self):
        return self._step_state

    @step_state.setter
    def step_state(self, value):
        if self._step_state == value:
            return
        self._step_state = value
        self._step_start_time = time.time()
        self._success_start_time = None
        self.last_visible_time = None
        # print(f"[WORLD] Step changed to {value}, timer reset.", flush=True)

    @property
    def wall_origin(self):
        return self.wall.get("origin")

    @wall_origin.setter
    def wall_origin(self, value):
        self.wall["origin"] = value
        self.wall["valid"] = value is not None

    def update_from_motion(self, event):
        """
        Updates pose based on motion events (Dead Reckoning).
        """
        delta = update_from_motion(self, event)
        brick_module = _brick_module()
        wall_module = _wall_module()
        brick_module.update_from_motion(self, event, delta)
        wall_module.update_from_motion(self, delta, self.wall_envelope)
        self.action_history.append(event)
        helper_xyz_coords.update_from_motion(self, event=event, delta=delta)

    def get_recent_net_forward_mm(self, window_s=5.0):
        """
        Calculates net forward distance (Forward - Backward) in the last window_s seconds.
        """
        now = time.time()
        cutoff = now - window_s
        net_dist = 0.0
        
        for event in reversed(self.action_history):
            if event.timestamp < cutoff:
                break
                
            dist = 0.0
            dt = event.duration_ms / 1000.0
            power_ratio = event.power / 255.0
            
            if event.action_type == "forward":
                dist = self.mm_per_sec_full_speed * power_ratio * dt
                net_dist += dist
            elif event.action_type == "backward":
                dist = self.mm_per_sec_full_speed * power_ratio * dt
                net_dist -= dist
                
        return net_dist

    def update_vision(
        self,
        found,
        dist,
        angle,
        conf,
        offset_x=0,
        cam_h=0,
        brick_above=False,
        brick_below=False,
        raw_dist=None,
    ):
        brick_module = _brick_module()
        wall_module = _wall_module()
        # Global x-axis convention: normal number line.
        # Turning camera left (brick appears further right in frame) should
        # decrease x-axis; turning right should increase x-axis.
        try:
            normalized_offset_x = -float(offset_x)
        except (TypeError, ValueError):
            normalized_offset_x = 0.0
        brick_height = brick_module.update_from_vision(
            self,
            found,
            dist,
            angle,
            conf,
            normalized_offset_x,
            cam_h,
            brick_above,
            brick_below,
            raw_dist=raw_dist,
        )
        update_lift_from_vision(
            self,
            cam_h,
            brick_height,
            conf,
        )
        wall_module.update_from_vision(self, found, dist, angle, conf, self.wall_envelope)
        helper_xyz_coords.sync_from_world(self, reason="vision")

    def get_scoop_corridor_limits(self, dist):
        brick_module = _brick_module()
        return brick_module.get_scoop_corridor_limits(self, dist)

    def compute_brick_world_xy(self, dist, angle_deg):
        brick_module = _brick_module()
        return brick_module.compute_brick_world_xy(self, dist, angle_deg)

    def is_aligned(self):
        """Returns True if metrics have been stable and centered."""
        return self.stability_count >= self.stability_threshold

    def check_step_complete(self):
        """Checks if success criteria are met using learned rules from demos."""
        wall_module = _wall_module()
        wall_check = wall_module.evaluate_success_gates(self, self.step_state, self.wall_envelope)
        if not wall_check.ok:
            return False
        obj_name = self.step_state.value

        gates = self.learned_rules.get(obj_name, {}).get("gates", {})
        success_metrics = gates.get("success", {}).get("metrics", {})
        if success_metrics:
            brick = self.brick or {}
            brick_visible = bool(brick.get("visible"))
            for metric, stats in success_metrics.items():
                if metric in ("angle_abs", "xAxis_offset_abs", "angle", "xAxis_offset", "dist", "confidence") and not brick_visible:
                    return False
                if metric == "angle_abs":
                    if abs(brick.get("angle", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "angle":
                    target = stats.get("target", 0.0)
                    tol = stats.get("tol", 0.0)
                    if abs(brick.get("angle", 0.0) - target) > tol:
                        return False
                elif metric == "xAxis_offset_abs":
                    if abs(brick.get("offset_x", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "xAxis_offset":
                    target = stats.get("target", 0.0)
                    tol = stats.get("tol", 0.0)
                    if abs(brick.get("offset_x", 0.0) - target) > tol:
                        return False
                elif metric == "dist":
                    if brick.get("dist", 0.0) > stats.get("max", 0.0):
                        return False
                elif metric == "confidence":
                    if brick.get("confidence", 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "visible":
                    if (1.0 if brick_visible else 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "lift_height":
                    lift = self.lift_height
                    if lift < stats.get("min", lift) or lift > stats.get("max", lift):
                        return False
            return True

        learned = self.learned_rules.get(obj_name, {})
        if not learned:
            return False

        target_vis = learned.get("final_visibility", True)
        if self.brick["visible"] != target_vis:
            return False

        if target_vis:
            max_x = learned.get("max_offset_x", 0)
            if abs(self.brick["offset_x"]) > max_x:
                return False
            max_ang = learned.get("max_angle", 0)
            if abs(self.brick["angle"]) > max_ang:
                return False

        return True

    def next_step(self):
        """Cycles through steps in the process order."""
        sequence = step_sequence()
        if not sequence:
            sequence = list(StepState)
        try:
            curr_idx = sequence.index(self.step_state)
        except ValueError:
            sequence = list(StepState)
            curr_idx = sequence.index(self.step_state)
        next_idx = (curr_idx + 1) % len(sequence)
        self.step_state = sequence[next_idx]
        if next_idx == 0:
            self.brick["held"] = False
        return self.step_state.value

    def get_next_step_label(self):
        """Returns the string label of the next step in sequence."""
        sequence = step_sequence()
        if not sequence:
            sequence = list(StepState)
        labels = [o.value for o in sequence]
        try:
            curr_idx = labels.index(self.step_state.value)
        except ValueError:
            labels = [o.value for o in StepState]
            curr_idx = labels.index(self.step_state.value)
        next_idx = (curr_idx + 1) % len(labels)
        return labels[next_idx]

    def reset_mission(self):
        """Resets the step state and all mission-specific flags."""
        self.step_state = StepState.FIND_BRICK
        self.brick["held"] = False
        self.stability_count = 0
        self.last_visible_time = None
        helper_xyz_coords.sync_from_world(self, reason="reset_mission")
        return self.step_state.value

    def to_dict(self):
        # Format Brick Data
        brick_fmt = self.brick.copy()
        brick_fmt.pop("id", None)
        brick_fmt.pop("confidence", None)
        # Demo logs should store one canonical horizontal/vertical offset pair.
        # Keep `x_axis`/`y_axis` and drop legacy aliases `offset_x`/`offset_y`.
        brick_fmt.pop("offset_x", None)
        brick_fmt.pop("offset_y", None)
        if self.step_state == StepState.FIND_BRICK:
            brick_fmt['dist'] = None
            brick_fmt['angle'] = None
            brick_fmt['x_axis'] = None
            brick_fmt['y_axis'] = None
            brick_fmt['brickAbove'] = None
            brick_fmt['brickBelow'] = None
        elif brick_fmt.get("visible"):
            if brick_fmt.get("dist") is not None:
                brick_fmt['dist'] = round(brick_fmt['dist'], 2)
            if brick_fmt.get("angle") is not None:
                brick_fmt['angle'] = round(brick_fmt['angle'], 3)
            if brick_fmt.get("x_axis") is not None:
                brick_fmt['x_axis'] = round(brick_fmt['x_axis'], 2)
            if brick_fmt.get("y_axis") is not None:
                brick_fmt['y_axis'] = round(brick_fmt['y_axis'], 2)
        else:
            brick_fmt['dist'] = None
            brick_fmt['angle'] = None
            brick_fmt['x_axis'] = None
            brick_fmt['y_axis'] = None

        return {
            "type": "state",
            "timestamp": round(time.time(), 3),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "robot_pose": {
                "x": round(self.x, 2), 
                "y": round(self.y, 2), 
                "theta": round(self.theta, 3),
                "height_mm": None if self.height_mm is None else round(self.height_mm, 2)
            },
            "brick": brick_fmt,
            "lift_height": round(self.lift_height, 2)
        }

class TelemetryLogger:
    def __init__(self, filename="leia_log.json"):
        self.filename = filename
        self.lock = threading.Lock()
        self.enabled = False # Don't log state until first keyframe
        # Clear old log
        with open(self.filename, 'w') as f:
            f.write("[\n") # Start JSON array
        self.first_entry = True

    def log_state(self, world_model: WorldModel):
        if not self.enabled:
            return
        data = world_model.to_dict()
        self._write_row(data)

    def log_keyframe(self, marker, step=None, timestamp=None):
        self.enabled = True # Start recording state once we have a semantic marker
        if timestamp is None:
            timestamp = time.time()
        
        data = {
            "type": "keyframe",
            "timestamp": round(timestamp, 3),
            "marker": marker
        }
        if step:
            data["step"] = step
            
        self._write_row(data)

    def _write_row(self, data):
        with self.lock:
            with open(self.filename, 'a') as f:
                if not self.first_entry:
                    f.write(",\n")
                json.dump(data, f)
                self.first_entry = False

    def log_event(self, event: MotionEvent, step=None):
        semantic_events = ['FAIL', 'RECOVERY_START', 'STEP_SUCCESS', 'JOB_SUCCESS', 'JOB_START']
        if event.action_type in semantic_events:
            self.log_keyframe(event.action_type, step, event.timestamp)
            return

        if not self.enabled:
            return

        if event.action_type not in MOTION_EVENT_TYPES:
            return

        speed_score = event.speed_score
        if speed_score is None:
            cmd = _cmd_for_action_type(event.action_type)
            if cmd:
                _, speed_score = quantize_speed(cmd, speed=event.power / 255.0)

        data = {
            "type": "action",
            "timestamp": round(event.timestamp, 3),
            "command": event.action_type,
            "speedScore": None if speed_score is None else int(speed_score)
        }

        self._write_row(data)

    def close(self):
        """
        Consolidated close method that handles JSON array termination.
        Robustly handles crashes by searching backward for the last valid '}'.
        """
        with self.lock:
            if not os.path.exists(self.filename):
                return
                
            try:
                with open(self.filename, 'rb+') as f:
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    
                    found_last_brace = False
                    # Search backwards for the last '}'
                    while pos > 0:
                        pos -= 1
                        f.seek(pos)
                        char = f.read(1)
                        if char == b'}':
                            # Found the end of a valid JSON object.
                            # Keep this row, truncate after it.
                            f.seek(pos + 1)
                            f.truncate()
                            found_last_brace = True
                            break
                        elif char == b'[': 
                            # Empty array case
                            f.seek(pos + 1)
                            f.truncate()
                            break
                    
                    # Ensure any trailing garbage (like a loose comma) is gone
                    # We already truncated at '}', so we are good.
                    
                    # Add final closing bracket
                    f.seek(0, os.SEEK_END)
                    if found_last_brace:
                        f.write(b"\n]\n")
                    else:
                        # If list was totally empty or malformed
                        f.write(b"]\n")
                        
                print(f"[LOGGER] Log closed and sanitized: {self.filename}")
            except Exception as e:
                print(f"[LOGGER] Error closing log: {e}")

    def _print_terminal(self, data):
        p = data.get('robot_pose', {'x':0, 'y':0, 'theta':0})
        b = data.get('brick', {})
        wall_state = data.get("wall") or {}
        wall = "SET" if (wall_state.get("origin") if isinstance(wall_state, dict) else None) else "UNSET"
        print(f"{'='*40}")
        print(f"TIME: {data.get('timestamp', 0):.2f}s")
        if 'step' in data:
            print(f"STEP: {data['step']}")
        print(f"WALL: {wall}")
        print(f"{'-'*40}")
        print(f"POSE:")
        print(f"  X: {p['x']:.2f} mm")
        print(f"  Y: {p['y']:.2f} mm")
        print(f"  Heading: {p['theta']:.2f}°")
        print(f"  Lift: {data.get('lift_height', 0):.2f} mm")
        print(f"{'-'*40}")
        print(f"BRICK:")
        print(f"  Visible: {b.get('visible', False)}")
        if b.get('visible'):
            print(f"  Distance: {b.get('dist', 0):.2f} mm")
            print(f"  Angle: {b.get('angle', 0):.2f}°")
            print(f"  Offset: {b.get('offset_x', 0):.2f} mm")
        print(f"{'-'*40}")
        
        print(f"{'='*40}")

# --- SHARED VISUALIZATION ---
import cv2
import numpy as np


def _stream_overlay_metric_keys_for_step(process_rules, telemetry_step):
    if not isinstance(process_rules, dict):
        return []
    step_key = None
    if telemetry_step is not None:
        if hasattr(telemetry_step, "value"):
            telemetry_step = telemetry_step.value
        raw = str(telemetry_step).strip()
        if raw:
            step_key = raw.upper()
    if not step_key:
        return []
    step_cfg = process_rules.get(step_key)
    if not isinstance(step_cfg, dict):
        return []
    gates = step_cfg.get("success_gates")
    if not isinstance(gates, dict) or not gates:
        gates = step_cfg.get("start_gates")
    if not isinstance(gates, dict) or not gates:
        return []
    keys = []
    seen = set()
    for metric in gates.keys():
        metric_key = str(metric or "").strip()
        if not metric_key or metric_key in seen:
            continue
        keys.append(metric_key)
        seen.add(metric_key)
    return keys


def draw_telemetry_overlay(
    frame,
    wm: WorldModel,
    extra_messages=None,
    reminders=None,
    gear=None,
    show_prompt=True,
    gate_status=None,
    gate_progress=None,
    step_suggestions=None,
    highlight_metric=None,
    loop_id=None,
    header_lines=None,
    gate_summary=None,
    gate_checker_summary=None,
    sidebar_mode=False,
    sidebar_width=240,
    draw_text=True,
    line_sink=None,
    show_center_line=True,
    brick_extra_lines=None,
    telemetry_step=None,
):
    """
    Simplified HUD renderer.
    - Merged step/checklist/status into single-line prompt.
    - Controls are logged in terminal, not shown on the overlay.
    - Optional gear label is handled separately.
    """
    h, w = frame.shape[:2]
    
    # --- COLORS (BGR) ---
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    ORANGE = (0, 165, 255)
    YELLOW = (0, 255, 255)

    def _bgr_to_hex(color):
        if not isinstance(color, (tuple, list)) or len(color) < 3:
            return "#ffffff"
        try:
            b, g, r = (int(color[0]), int(color[1]), int(color[2]))
        except (TypeError, ValueError):
            return "#ffffff"
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # 0. Center Alignment Line
    if show_center_line:
        cal_offset = 0
        if WORLD_MODEL_BRICK_FILE.exists():
            try:
                with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                    cal_offset = json.load(f).get('calibration', {}).get('camera_center_offset_px', 0)
            except:
                pass
        center_x = int(w // 2 + cal_offset)
        center_x = max(0, min(w - 1, center_x))
        center_y = int(h // 2)
        center_y = max(0, min(h - 1, center_y))
        guide_color = (60, 60, 60)
        cv2.line(frame, (center_x, 0), (center_x, h - 1), guide_color, 1)
        cv2.line(frame, (0, center_y), (w - 1, center_y), guide_color, 1)

    # 1. Background Panel (Left Side)
    if sidebar_mode:
        panel_w = max(200, int(sidebar_width))
        panel = np.zeros((h, panel_w, 3), dtype=frame.dtype)
        text_surface = panel
    else:
        if draw_text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        text_surface = frame
    
    # 2. Text Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1 # No bolding, as it gets fuzzy
    x_base = 12
    y_cur = 25
    line_h = 20
    
    def put_line(txt, c=WHITE, s=scale, th=thickness, thickness=None):
        nonlocal y_cur
        if thickness is not None:
            th = thickness
        if line_sink is not None:
            line_sink.append({"text": str(txt), "color": _bgr_to_hex(c)})
        if draw_text:
            cv2.putText(text_surface, txt, (x_base, y_cur), font, s, c, th)
        y_cur += line_h

    def put_line_segments(segments, s=scale, th=thickness):
        nonlocal y_cur
        x_cur = x_base
        if line_sink is not None:
            payload = []
            for seg in segments:
                if not isinstance(seg, (tuple, list)) or not seg:
                    continue
                seg_text = str(seg[0])
                seg_color = seg[1] if len(seg) > 1 else WHITE
                payload.append({"text": seg_text, "color": _bgr_to_hex(seg_color)})
            if payload:
                line_sink.append({"segments": payload})
        for seg in segments:
            if not isinstance(seg, (tuple, list)) or not seg:
                continue
            txt = str(seg[0])
            color = seg[1] if len(seg) > 1 else WHITE
            if draw_text:
                cv2.putText(text_surface, txt, (x_cur, y_cur), font, s, color, th)
                (txt_w, _), _ = cv2.getTextSize(txt, font, s, th)
                x_cur += txt_w
        y_cur += line_h

    # 3. MERGED STATE & PROMPT - REMOVED per user request
    y_cur += 5

    # 3b. Header lines (Step/Success/Suggested act)
    if header_lines:
        put_line("", WHITE, 0.35, 1)
        for line in header_lines:
            put_line(str(line), WHITE, 0.45, 1)
        put_line("", WHITE, 0.35, 1)
        y_cur += 5

    # 4. Reminders
    if reminders:
        put_line("", WHITE, 0.35, 1)
        put_line("--- REMINDERS ---", WHITE, 0.35, 1)
        if isinstance(reminders, list):
            for msg in reminders:
                put_line(str(msg), WHITE, 0.35, 1)
        else:
            put_line(str(reminders), WHITE, 0.35, 1)
        put_line("", WHITE, 0.35, 1)
        y_cur += 5

    # 4b. Success Gates (summary or current step only)
    if gate_summary is not None:
        put_line("", WHITE, 0.35, 1)
        put_line("--- SUCCESS GATES ---", WHITE, 0.35, 1)
        if gate_summary:
            for line in gate_summary:
                if isinstance(line, dict):
                    segments = line.get("segments")
                    if isinstance(segments, list) and segments:
                        put_line_segments(segments, 0.35, 1)
                        continue
                if isinstance(line, tuple):
                    text, color = line
                    put_line(str(text), color, 0.35, 1)
                else:
                    put_line(str(line), WHITE, 0.35, 1)
        else:
            put_line("(idle)", WHITE, 0.35, 1)
        put_line("", WHITE, 0.35, 1)
        y_cur += 5
    elif gate_progress is not None:
        put_line("", WHITE, 0.35, 1)
        if loop_id is not None:
            put_line(f"LOOP ID: {loop_id}", WHITE, 0.35, 1)
            y_cur += 3
        put_line("--- SUCCESS GATES ---", WHITE, 0.35, 1)
        current_obj = wm.step_state.value if wm.step_state else None
        match = None
        if gate_progress and current_obj:
            for name, pct in gate_progress:
                if str(name) == str(current_obj):
                    match = (name, pct)
                    break
        if match:
            name, pct = match
            pct_display = int(max(0.0, min(100.0, pct)))
            put_line(f"{name}: {pct_display}%", WHITE, 0.35, 1)
            if step_suggestions:
                for obj_name, suggestion in step_suggestions:
                    if str(obj_name) == str(name):
                        sug_color = WHITE
                        trend_map = getattr(wm, "_align_metrics_trend", {})
                        if suggestion.startswith(("L ", "R ")):
                            trend_val = trend_map.get("x_axis")
                        elif suggestion.startswith(("F ", "B ")):
                            trend_val = trend_map.get("dist")
                        else:
                            trend_val = 0
                        if trend_val == 1:
                            sug_color = GREEN
                        elif trend_val == -1:
                            sug_color = RED
                        put_line(f"  {suggestion}", sug_color, 0.35, 1)
        else:
            put_line("(none)", WHITE, 0.35, 1)
        put_line("", WHITE, 0.35, 1)
        y_cur += 5

    # 4c. Gate Checker (compact truth confirmation status)
    if gate_checker_summary is not None:
        put_line("", WHITE, 0.35, 1)
        put_line("--- GATE CHECKER ---", WHITE, 0.35, 1)
        lines = []
        if isinstance(gate_checker_summary, (list, tuple)):
            for line in gate_checker_summary:
                if line is None:
                    continue
                lines.append(str(line))
        elif gate_checker_summary:
            lines = [str(gate_checker_summary)]
        while len(lines) < 3:
            lines.append("-")
        for line in lines:
            put_line(line, WHITE, 0.35, 1)
        put_line("", WHITE, 0.35, 1)
        y_cur += 5

    # 5. Position Info
    put_line("", WHITE, 0.35, 1)
    put_line("--- BRICK[0] TELEMETRY ---", WHITE, 0.35, 1)
    visible_now = bool(wm.brick.get("visible"))
    in_crosshairs_now = bool(wm.brick.get("inCrosshairs"))
    x_axis = wm.brick.get("x_axis", wm.brick.get("offset_x", 0.0))
    y_axis = wm.brick.get("y_axis", wm.brick.get("offset_y", 0.0))
    obj_rules = (wm.process_rules or {}).get("ALIGN_BRICK", {}) if wm.process_rules else {}
    success_gates = (obj_rules or {}).get("success_gates") or {}
    x_prefix = "* " if highlight_metric == "xAxis_offset_abs" else ""
    y_prefix = "* " if highlight_metric in ("yAxis_offset_abs", "yAxis_offset") else ""
    angle_prefix = "* " if highlight_metric == "angle_abs" else ""
    dist_prefix = "* " if highlight_metric == "dist" else ""
    brick_conf = wm.brick.get("confidence")
    if brick_conf is None:
        brick_conf = 0.0
    stack_gate_state = getattr(wm, "_stack_visibility_gate", None)

    def _stack_bool_compact(value):
        return "true" if bool(value) else "false"

    def _stack_hud_pair(row_key, brick_key):
        row = (
            stack_gate_state.get(row_key)
            if isinstance(stack_gate_state, dict) and isinstance(stack_gate_state.get(row_key), dict)
            else {}
        )
        raw_val = row.get("raw")
        conf_val = wm.brick.get(brick_key)
        return f"raw={_stack_bool_compact(raw_val)} confident={_stack_bool_compact(conf_val)}"

    def _lift_hud_text(prefix=""):
        lift_val = getattr(wm, "lift_height", None)
        if not isinstance(lift_val, (int, float)):
            return f"{prefix}LIFT:   -"
        text = f"{prefix}LIFT:   {float(lift_val):.1f} mm"
        source = str(getattr(wm, "lift_height_source", "") or "").strip()
        try:
            quality = float(getattr(wm, "lift_height_quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0
        quality = max(0.0, min(1.0, float(quality)))
        if source and source != "dead_reckon":
            return f"{text} ({source})"
        return text

    def _height_bricks_text(value):
        try:
            bricks_val = int(round(float(value)))
        except (TypeError, ValueError):
            return "unknown"
        return f"{int(max(0, bricks_val))} bricks"

    selected_metrics = _stream_overlay_metric_keys_for_step(
        wm.process_rules or {},
        telemetry_step,
    )

    def _canon_metric(metric_name):
        key = str(metric_name or "").strip().lower()
        if key in ("xaxis_offset_abs", "x_axis", "offset_x"):
            return "xAxis_offset_abs"
        if key in ("yaxis_offset_abs", "y_axis", "offset_y"):
            return "yAxis_offset_abs"
        if key in ("angle_abs", "angle"):
            return "angle_abs"
        if key in ("dist", "distance"):
            return "dist"
        if key in ("visible",):
            return "visible"
        if key in ("confidence", "conf"):
            return "confidence"
        if key in ("brick_above", "brickabove"):
            return "brick_above"
        if key in ("brick_below", "brickbelow"):
            return "brick_below"
        if key in ("incrosshairs", "in_crosshairs"):
            return "inCrosshairs"
        if key in ("lift_height",):
            return "lift_height"
        return str(metric_name or "").strip()

    def _metric_prefix(metric_name):
        canonical = _canon_metric(metric_name)
        if canonical and canonical == _canon_metric(highlight_metric):
            return "* "
        return ""

    def _render_selected_metric(metric_name):
        metric_key = _canon_metric(metric_name)
        prefix = _metric_prefix(metric_name)
        if metric_key == "visible":
            put_line(f"{prefix}VISIBLE: {'true' if visible_now else 'false'}", WHITE, 0.38, 1)
            return True
        if metric_key == "inCrosshairs":
            put_line(f"{prefix}inCrosshairs: {'true' if in_crosshairs_now else 'false'}", WHITE, 0.38, 1)
            return True
        if metric_key == "xAxis_offset_abs":
            if visible_now:
                put_line(f"{prefix}X-AXIS: {x_axis:.1f} mm", WHITE, 0.38, 1)
            else:
                put_line(f"{prefix}X-AXIS: -", WHITE, 0.38, 1)
            return True
        if metric_key == "yAxis_offset_abs":
            if visible_now:
                put_line(f"{prefix}Y-AXIS: {y_axis:.1f} mm", WHITE, 0.38, 1)
            else:
                put_line(f"{prefix}Y-AXIS: -", WHITE, 0.38, 1)
            return True
        if metric_key == "angle_abs":
            angle_val = wm.brick.get("angle")
            if visible_now and isinstance(angle_val, (int, float)):
                put_line(f"{prefix}ANGLE:  {float(angle_val):.1f} deg", WHITE, 0.38, 1)
            else:
                put_line(f"{prefix}ANGLE:  -", WHITE, 0.38, 1)
            return True
        if metric_key == "dist":
            dist_val = wm.brick.get("dist")
            if visible_now and isinstance(dist_val, (int, float)):
                put_line(f"{prefix}DIST:   {float(dist_val):.0f} mm", WHITE, 0.38, 1)
            else:
                put_line(f"{prefix}DIST:   -", WHITE, 0.38, 1)
            return True
        if metric_key == "confidence":
            if visible_now and isinstance(brick_conf, (int, float)):
                put_line(f"{prefix}CONF:   {float(brick_conf):.0f}%", WHITE, 0.38, 1)
            else:
                put_line(f"{prefix}CONF:   -", WHITE, 0.38, 1)
            return True
        if metric_key == "brick_above":
            above_txt_local = _stack_hud_pair("above", "brickAbove")
            put_line(f"{prefix}Brick above: {above_txt_local}", WHITE, 0.38, 1)
            return True
        if metric_key == "brick_below":
            below_txt_local = _stack_hud_pair("below", "brickBelow")
            put_line(f"{prefix}Brick below: {below_txt_local}", WHITE, 0.38, 1)
            return True
        if metric_key == "lift_height":
            put_line(_lift_hud_text(prefix), WHITE, 0.38, 1)
            return True
        return False

    rendered_any_selected = False
    rendered_confidence_selected = False
    rendered_stack_above = False
    rendered_stack_below = False
    for metric_name in selected_metrics:
        metric_key = _canon_metric(metric_name)
        rendered_any_selected = _render_selected_metric(metric_name) or rendered_any_selected
        if metric_key == "confidence":
            rendered_confidence_selected = True
        if metric_key == "brick_above":
            rendered_stack_above = True
        elif metric_key == "brick_below":
            rendered_stack_below = True

    if not rendered_any_selected:
        if visible_now:
            put_line(f"{x_prefix}X-AXIS: {x_axis:.1f} mm", WHITE, 0.38, 1)
        else:
            put_line(f"{x_prefix}X-AXIS: -", WHITE, 0.38, 1)
        if visible_now:
            put_line(f"{y_prefix}Y-AXIS: {y_axis:.1f} mm", WHITE, 0.38, 1)
        else:
            put_line(f"{y_prefix}Y-AXIS: -", WHITE, 0.38, 1)
        if visible_now:
            put_line(f"{angle_prefix}ANGLE:  {wm.brick['angle']:.1f} deg", WHITE, 0.38, 1)
        else:
            put_line(f"{angle_prefix}ANGLE:  -", WHITE, 0.38, 1)
        if visible_now:
            put_line(f"{dist_prefix}DIST:   {wm.brick['dist']:.0f} mm", WHITE, 0.38, 1)
        else:
            put_line(f"{dist_prefix}DIST:   -", WHITE, 0.38, 1)
    if not rendered_confidence_selected:
        if visible_now and isinstance(brick_conf, (int, float)):
            put_line(f"CONF:   {float(brick_conf):.0f}%", WHITE, 0.38, 1)
        else:
            put_line("CONF:   -", WHITE, 0.38, 1)
    if not rendered_stack_above:
        above_txt = _stack_hud_pair("above", "brickAbove")
        put_line(f"Brick above: {above_txt}", WHITE, 0.38, 1)
    if not rendered_stack_below:
        below_txt = _stack_hud_pair("below", "brickBelow")
        put_line(f"Brick below: {below_txt}", WHITE, 0.38, 1)
    put_line(f"inCrosshairs: {'true' if in_crosshairs_now else 'false'}", WHITE, 0.38, 1)
    if brick_extra_lines:
        for line in brick_extra_lines:
            if isinstance(line, (tuple, list)) and len(line) >= 2:
                put_line(str(line[0]), line[1], 0.35, 1)
            else:
                put_line(str(line), WHITE, 0.35, 1)
    put_line("", WHITE, 0.35, 1)
    
    y_cur += 5
    put_line("", WHITE, 0.35, 1)
    put_line("--- LEIA TELEMETRY ---", WHITE, 0.35, 1)
    put_line(f"X:      {wm.x:.1f} mm", WHITE, 0.38, 1)
    put_line(f"Y:      {wm.y:.1f} mm", WHITE, 0.38, 1)
    put_line(f"THETA:  {wm.theta:.1f} deg", WHITE, 0.38, 1)
    put_line(_lift_hud_text(), WHITE, 0.38, 1)
    cam_times = getattr(wm, "_camera_frame_times", [])
    if cam_times:
        has_dupes = getattr(wm, "_camera_dupe_ms", False)
        cam_color = RED if has_dupes else WHITE
        fps = getattr(wm, "_camera_fps", None)
        fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else "-"
        cam_note = " (repeated ms stamp)" if has_dupes else ""
        put_line(f"CAMERA: {fps_str} fps{cam_note}", cam_color, 0.38, 1)
    put_line("", WHITE, 0.35, 1)

    y_cur += 8 # Spacer

    # 8. Extra Messages (Banners -> Moved to Sidebar)
    if extra_messages:
        y_cur = h - 20
        for msg in extra_messages:
             put_line(f"! {msg}", YELLOW, 0.4, 2)

    # 9. GEAR Display
    if gear:
        if line_sink is not None:
            line_sink.append({"text": f"GEAR: {gear}", "color": _bgr_to_hex(WHITE)})
        if draw_text:
            cv2.putText(text_surface, f"GEAR: {gear}", (x_base, h - 35), font, 0.4, WHITE, 2)

    if sidebar_mode:
        return cv2.hconcat([text_surface, frame])
    return frame
