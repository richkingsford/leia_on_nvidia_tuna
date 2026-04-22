#!/usr/bin/env python3
"""Run focused alignment experiments around x-axis turn behavior."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from calibration.helper_calibrate import (
    brick_pose_from_world,
    observe_pose_with_reobserve,
    read_pose,
)
from helper_close_gaps import curve_intensity_for_error, load_left_right_curve
from helper_next import compute_alignment_analytics
from helper_robot_control import Robot
from helper_turn_drive_motion import build_turn_drive_motion_plan
from helper_vision_aruco import ArucoBrickVision
from helper_vision_config import (
    VISION_MODE_ARUCO,
    VISION_MODE_CYAN,
    active_vision_mode,
    normalize_vision_mode,
)
from telemetry_process import (
    _average_smoothed_frames,
    _latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command,
    send_robot_command_pwm,
    update_world_from_vision,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import StepState, WorldModel


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_random_back_turn_experiment.json"
RUN_MEMORY_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_random_back_turn_experiment_memory.json"
TRIALS_DIR_DEFAULT = Path(__file__).resolve().parent / "trials"
TRIALS_FILE_DEFAULT = TRIALS_DIR_DEFAULT / "trials.json"
DEFAULT_SCORE = 1
DEFAULT_BACK_RANGE_MS = (500, 1500)
DEFAULT_TURN_RANGE_MS = (500, 800)
DEFAULT_SETTLE_S = 0.12
DEFAULT_OBSERVE_SAMPLES = 3
DEFAULT_OBSERVE_TIMEOUT_S = 1.5
DEFAULT_RELAXED_TIMEOUT_S = 2.8
DEFAULT_REOBSERVE_ROUNDS = 2
DEFAULT_OBSERVE_SLEEP_S = 0.05
EXPERIMENT_STEP = StepState.ALIGN_BRICK
DEFAULT_TRACK_MAX_CYCLES = 8
DEFAULT_TRACK_OBSERVE_TIMEOUT_S = 0.8
DEFAULT_TRACK_RELAXED_TIMEOUT_S = 1.2
DEFAULT_X_TURN_POST_MOVE_OBSERVE_TIMEOUT_S = 1.2
DEFAULT_X_TURN_POST_MOVE_RELAXED_TIMEOUT_S = 1.8
DEFAULT_X_TURN_POST_MOVE_REOBSERVE_ROUNDS = 2
DEFAULT_MAX_TURN_INTENSITY_PCT = 8.0
DEFAULT_MAX_DRIVE_RECOVERY_SCORE = 4
DEFAULT_MIN_TURN_INTENSITY_PCT = 0.01
DEFAULT_MIN_PROGRESS_MM = 0.25
DEFAULT_STALL_LIMIT = 2
DEFAULT_DIRECTION_FLIP_WORSEN_MM = 0.05
DEFAULT_DIRECTION_HARD_FLIP_WORSEN_MM = 0.75
DEFAULT_MAX_DIRECTION_FLIPS = 2
DEFAULT_OTHER_GATE_BLOCK_MM = 0.5
DEFAULT_SECONDARY_BPLUS_CUSHION_MM = 0.6
DEFAULT_X_TURN_EFFECTIVE_MIN_INTENSITY_PCT = 1.0
DEFAULT_X_TURN_BREAKOUT_TRIGGER_CYCLES = 2
DEFAULT_X_TURN_BREAKOUT_MIN_INTENSITY_PCT = 2.0
DEFAULT_X_TURN_BREAKOUT_MAX_INTENSITY_PCT = 4.0
DEFAULT_INITIAL_INTENSITY_SCALE = 0.75
DEFAULT_MIN_INTENSITY_SCALE = 0.40
DEFAULT_MAX_INTENSITY_SCALE = 1.00
DEFAULT_DIRECTION_CONFIRM_CYCLES = 2
DEFAULT_RECENT_EFFICACY_WINDOW = 3
DEFAULT_PLATEAU_REOBSERVE_TIMEOUT_S = 1.6
DEFAULT_PLATEAU_REOBSERVE_ROUNDS = 2
DEFAULT_PLATEAU_PROBE_SCALE = 0.70
DEFAULT_MOVING_OBSERVE_DURATION_MS = 1500
DEFAULT_MOVING_OBSERVE_SAMPLE_HZ = 20.0
DEFAULT_MOVING_OBSERVE_TRIALS = 1
DEFAULT_MOVING_OBSERVE_SEQUENCE = ("l", "r", "u", "d", "f", "b")
DEFAULT_MOVING_OBSERVE_GROUP_BY_AXIS = True
DEFAULT_MOVING_OBSERVE_RETRIGGER_FRACTION = 0.85
DEFAULT_MOVING_OBSERVE_MIN_CMD_INTERVAL_S = 0.05
DEFAULT_MOVING_OBSERVE_TURN_PULSE_MS = 220
DEFAULT_MOVING_OBSERVE_MAST_PULSE_MS = 260
DEFAULT_MOVING_OBSERVE_DRIVE_PULSE_MS = 300
DEFAULT_ALTERNATING_TURN_DRIVE_CYCLES = 10
DEFAULT_ALTERNATING_TURN_DRIVE_PHASE_DURATION_MS = 500
DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM = 0.25
DEFAULT_ALTERNATING_TURN_DRIVE_REBALANCE_MM = 3.0
DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM = 10.0
DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH = "strong"
DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS = 10
DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_DURATION_MS = 300
DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_PHASE = "back_left"
DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_SETUP_FORWARD_RANGE_MS = (300, 1000)
DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT = 75.0
TRIAL_RESULT_TBD = "___"
TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS = "adaptive_by_x_axis_sign"
TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST = "adaptive_by_run_start_dist"
ALTERNATING_TURN_DRIVE_STRENGTH_PROFILES = {
    "forward": {
        "strong": "forward_pivot",
        "medium": "forward_arc_medium",
        "light": "forward_arc_light",
    },
    "backward": {
        "strong": "backward_pivot",
        "medium": "backward_arc_medium",
        "light": "backward_arc_light",
    },
}
DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE = (
    {
        "phase": "back_left",
        "label": "BACK+LEFT",
        "cmd": "l",
        "drive_mode": "backward",
        "profile_name": "backward_pivot",
        "strength": DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
    },
    {
        "phase": "forward_right",
        "label": "FORWARD+RIGHT",
        "cmd": "r",
        "drive_mode": "forward",
        "profile_name": "forward_pivot",
        "strength": DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
    },
)
DEFAULT_X_ZERO_TRIALS = 10
DEFAULT_X_ZERO_SAMPLE_HZ = 20.0
DEFAULT_X_ZERO_RESET_TIMEOUT_S = 12.0
DEFAULT_X_ZERO_ATTEMPT_TIMEOUT_S = 12.0
DEFAULT_X_ZERO_TURN_INTENSITY_PCT = 1.0
DEFAULT_X_ZERO_RESET_TURN_INTENSITY_PCT = 1.0
DEFAULT_X_ZERO_BAND_MM = 0.5
DEFAULT_X_ZERO_STAGE_SETTLE_S = 0.05
DEFAULT_X_ZERO_MAX_ACT_DURATION_MS = 888
DEFAULT_X_ZERO_RESET_EDGE_X_MM = -35.0
DEFAULT_X_ZERO_RESET_PLATEAU_SAMPLES = 5
DEFAULT_X_ZERO_RESET_PLATEAU_SPAN_MM = 0.2

def _normalize_range(low: int, high: int) -> tuple[int, int]:
    try:
        low_val = int(round(float(low)))
    except (TypeError, ValueError):
        low_val = 0
    try:
        high_val = int(round(float(high)))
    except (TypeError, ValueError):
        high_val = 0
    low_val = max(1, int(low_val))
    high_val = max(1, int(high_val))
    if low_val > high_val:
        low_val, high_val = high_val, low_val
    return int(low_val), int(high_val)


def choose_random_back_turn_plan(
    *,
    rng: random.Random | None = None,
    back_range_ms: tuple[int, int] = DEFAULT_BACK_RANGE_MS,
    turn_range_ms: tuple[int, int] = DEFAULT_TURN_RANGE_MS,
) -> dict:
    rng_local = rng if rng is not None else random.Random()
    back_low, back_high = _normalize_range(*back_range_ms)
    turn_low, turn_high = _normalize_range(*turn_range_ms)
    back_duration_ms = int(rng_local.randint(int(back_low), int(back_high)))
    turn_cmd = str(rng_local.choice(("l", "r")))
    turn_duration_ms = int(rng_local.randint(int(turn_low), int(turn_high)))
    return {
        "back_duration_ms": int(back_duration_ms),
        "turn_cmd": str(turn_cmd),
        "turn_duration_ms": int(turn_duration_ms),
        "sequence": [
            {"cmd": "b", "duration_ms": int(back_duration_ms)},
            {"cmd": str(turn_cmd), "duration_ms": int(turn_duration_ms)},
        ],
    }


def _duration_range_snapshot(duration_range_ms) -> dict | None:
    if duration_range_ms is None:
        return None
    try:
        if isinstance(duration_range_ms, dict):
            low_src = duration_range_ms.get("min_ms")
            high_src = duration_range_ms.get("max_ms")
        else:
            low_src, high_src = duration_range_ms
        low_val, high_val = _normalize_range(low_src, high_src)
    except Exception:
        return None
    return {
        "min_ms": int(low_val),
        "max_ms": int(high_val),
    }


def _sample_duration_ms(
    *,
    duration_range_ms,
    rng: random.Random | None = None,
) -> int | None:
    snapshot = _duration_range_snapshot(duration_range_ms)
    if not isinstance(snapshot, dict):
        return None
    rng_local = rng if rng is not None else random.Random()
    return int(
        rng_local.randint(
            int(snapshot.get("min_ms") or 1),
            int(snapshot.get("max_ms") or 1),
        )
    )


def _duration_ms_to_seconds(duration_ms) -> float | None:
    value = _coerce_float(duration_ms, None)
    if value is None:
        return None
    return _round_triplet(float(value) / 1000.0)


def _evenly_distributed_durations_ms(
    *,
    duration_range_ms,
    count: int,
) -> list[int]:
    snapshot = _duration_range_snapshot(duration_range_ms)
    try:
        count_used = max(0, int(count))
    except (TypeError, ValueError):
        count_used = 0
    if count_used <= 0 or not isinstance(snapshot, dict):
        return []
    low_val = int(snapshot.get("min_ms") or 1)
    high_val = int(snapshot.get("max_ms") or 1)
    if count_used == 1:
        return [int(low_val)]
    span = float(high_val - low_val)
    step = span / float(count_used - 1)
    durations = []
    for idx in range(count_used):
        raw_value = float(low_val) + (float(idx) * step)
        duration_val = int(round(raw_value))
        duration_val = max(int(low_val), min(int(high_val), int(duration_val)))
        if durations and duration_val < durations[-1]:
            duration_val = int(durations[-1])
        durations.append(int(duration_val))
    if durations:
        durations[0] = int(low_val)
        durations[-1] = int(high_val)
    return durations


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _round_triplet(value):
    number = _coerce_float(value, None)
    if number is None:
        return None
    return round(float(number), 3)


def _opposite_turn_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "l":
        return "r"
    if cmd_key == "r":
        return "l"
    return None


def _movement_axis_for_cmd(cmd: str | None) -> str:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key in {"l", "r"}:
        return "x"
    if cmd_key in {"u", "d"}:
        return "y"
    if cmd_key in {"f", "b"}:
        return "dist"
    return "unknown"


def _group_moving_cmd_sequence(cmd_sequence: tuple[str, ...] | list[str]) -> list[dict]:
    groups = []
    by_axis = {}
    for cmd in list(cmd_sequence or []):
        cmd_key = str(cmd or "").strip().lower()
        axis = _movement_axis_for_cmd(cmd_key)
        if axis == "unknown":
            continue
        if axis not in by_axis:
            row = {"axis": str(axis), "cmds": []}
            by_axis[axis] = row
            groups.append(row)
        if cmd_key not in by_axis[axis]["cmds"]:
            by_axis[axis]["cmds"].append(str(cmd_key))
    return groups


def _build_moving_phase_specs(
    *,
    cmd_sequence: tuple[str, ...] | list[str],
    trials: int,
    group_by_axis: bool,
) -> list[dict]:
    phase_specs = []
    phase_index = 0
    trial_count = max(1, int(trials))
    seq = [str(cmd or "").strip().lower() for cmd in list(cmd_sequence or []) if str(cmd or "").strip()]
    if not bool(group_by_axis):
        for trial_index in range(1, int(trial_count) + 1):
            for cmd in seq:
                axis = _movement_axis_for_cmd(cmd)
                if axis == "unknown":
                    continue
                phase_index += 1
                phase_specs.append(
                    {
                        "phase_index": int(phase_index),
                        "trial": int(trial_index),
                        "axis": str(axis),
                        "cmd": str(cmd),
                        "block_index": int(trial_index),
                        "block_key": f"trial_{int(trial_index)}",
                    }
                )
        return phase_specs

    groups = _group_moving_cmd_sequence(tuple(seq))
    for block_index, group in enumerate(groups, start=1):
        axis = str(group.get("axis") or "")
        cmds = [str(cmd) for cmd in list(group.get("cmds") or []) if str(cmd or "").strip()]
        for trial_index in range(1, int(trial_count) + 1):
            for cmd in cmds:
                phase_index += 1
                phase_specs.append(
                    {
                        "phase_index": int(phase_index),
                        "trial": int(trial_index),
                        "axis": str(axis),
                        "cmd": str(cmd),
                        "block_index": int(block_index),
                        "block_key": f"axis_{str(axis)}",
                    }
                )
    return phase_specs


def _moving_observe_send_policy(
    *,
    cmd: str,
    phase_duration_ms: int,
) -> dict:
    axis = _movement_axis_for_cmd(cmd)
    if axis == "x":
        pulse_ms = int(DEFAULT_MOVING_OBSERVE_TURN_PULSE_MS)
    elif axis == "y":
        pulse_ms = int(DEFAULT_MOVING_OBSERVE_MAST_PULSE_MS)
    else:
        pulse_ms = int(DEFAULT_MOVING_OBSERVE_DRIVE_PULSE_MS)
    pulse_ms = max(1, min(int(phase_duration_ms), int(pulse_ms)))
    return {
        "axis": str(axis),
        "mode": "retriggered_pulses",
        "pulse_duration_ms": int(pulse_ms),
        "retrigger_fraction": float(DEFAULT_MOVING_OBSERVE_RETRIGGER_FRACTION),
        "min_interval_s": float(DEFAULT_MOVING_OBSERVE_MIN_CMD_INTERVAL_S),
    }


def _recent_mean(values) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    usable = []
    for item in values:
        try:
            usable.append(float(item))
        except (TypeError, ValueError):
            continue
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _recent_last(values) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    for item in reversed(values):
        value = _coerce_float(item, None)
        if value is not None:
            return float(value)
    return None


def _init_x_lock_controller() -> dict:
    return {
        "trusted_cmd": None,
        "trusted_confirmations": 0,
        "cmd_stats": {
            "l": {
                "recent_improvements_mm": [],
                "improve_streak": 0,
                "worsen_streak": 0,
                "last_outcome": None,
                "intensity_scale": float(DEFAULT_INITIAL_INTENSITY_SCALE),
            },
            "r": {
                "recent_improvements_mm": [],
                "improve_streak": 0,
                "worsen_streak": 0,
                "last_outcome": None,
                "intensity_scale": float(DEFAULT_INITIAL_INTENSITY_SCALE),
            },
        },
    }


def _controller_from_snapshot(snapshot: dict | None) -> dict:
    controller = _init_x_lock_controller()
    if not isinstance(snapshot, dict):
        return controller
    controller["trusted_cmd"] = str(snapshot.get("trusted_cmd") or "").strip().lower() or None
    controller["trusted_confirmations"] = max(0, int(snapshot.get("trusted_confirmations") or 0))
    stats_in = snapshot.get("cmd_stats") if isinstance(snapshot.get("cmd_stats"), dict) else {}
    stats_out = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    for cmd in ("l", "r"):
        row_in = stats_in.get(cmd) if isinstance(stats_in.get(cmd), dict) else {}
        row_out = stats_out.get(cmd) if isinstance(stats_out.get(cmd), dict) else {}
        recent = []
        for item in list(row_in.get("recent_improvements_mm") or []):
            value = _coerce_float(item, None)
            if value is not None:
                recent.append(float(value))
        row_out["recent_improvements_mm"] = recent[-int(max(1, int(DEFAULT_RECENT_EFFICACY_WINDOW))):]
        # Cross-run memory keeps aggregate evidence but resets session-local streaks.
        row_out["improve_streak"] = 0
        row_out["worsen_streak"] = 0
        row_out["last_outcome"] = None
        row_out["intensity_scale"] = float(
            _clamp(
                _coerce_float(row_in.get("intensity_scale"), DEFAULT_INITIAL_INTENSITY_SCALE),
                DEFAULT_MIN_INTENSITY_SCALE,
                DEFAULT_MAX_INTENSITY_SCALE,
            )
        )
        stats_out[cmd] = row_out
    controller["cmd_stats"] = stats_out
    trusted_cmd = str(controller.get("trusted_cmd") or "").strip().lower()
    if trusted_cmd not in {"l", "r"}:
        controller["trusted_cmd"] = None
        controller["trusted_confirmations"] = 0
        return controller
    trusted_row = stats_out.get(trusted_cmd) if isinstance(stats_out.get(trusted_cmd), dict) else {}
    trusted_mean = _recent_mean(list(trusted_row.get("recent_improvements_mm") or []))
    if (
        trusted_mean is None
        or float(trusted_mean) <= 0.0
        or int(controller.get("trusted_confirmations") or 0) < int(DEFAULT_DIRECTION_CONFIRM_CYCLES)
    ):
        controller["trusted_cmd"] = None
        controller["trusted_confirmations"] = 0
    return controller


def _controller_snapshot(controller: dict | None) -> dict:
    if not isinstance(controller, dict):
        return {}
    out = {
        "trusted_cmd": controller.get("trusted_cmd"),
        "trusted_confirmations": int(controller.get("trusted_confirmations") or 0),
        "cmd_stats": {},
    }
    stats_raw = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    for cmd in ("l", "r"):
        row = stats_raw.get(cmd) if isinstance(stats_raw.get(cmd), dict) else {}
        out["cmd_stats"][cmd] = {
            "recent_improvements_mm": [_round_triplet(item) for item in list(row.get("recent_improvements_mm") or [])],
            "recent_mean_improvement_mm": _round_triplet(_recent_mean(list(row.get("recent_improvements_mm") or []))),
            "improve_streak": int(row.get("improve_streak") or 0),
            "worsen_streak": int(row.get("worsen_streak") or 0),
            "last_outcome": row.get("last_outcome"),
            "intensity_scale": _round_triplet(row.get("intensity_scale")),
        }
    return out


def _load_x_lock_memory(memory_path: Path | None) -> dict:
    if memory_path is None:
        return _init_x_lock_controller()
    path = Path(memory_path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _init_x_lock_controller()
    snapshot = payload.get("controller") if isinstance(payload, dict) else None
    return _controller_from_snapshot(snapshot if isinstance(snapshot, dict) else payload)


def _save_x_lock_memory(
    memory_path: Path | None,
    *,
    controller: dict,
    stop_reason: str | None,
    final_pose: dict | None,
) -> str | None:
    if memory_path is None:
        return None
    payload = {
        "updated_ts": round(float(time.time()), 3),
        "stop_reason": str(stop_reason or ""),
        "controller": _controller_snapshot(controller),
        "final_pose": _pose_snapshot(final_pose),
    }
    try:
        path = Path(memory_path)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return str(path)
    except OSError:
        return None


def _update_x_lock_controller(
    controller: dict,
    *,
    cmd: str,
    x_improvement_mm: float | None,
    min_progress_mm: float,
    worsen_mm: float,
) -> dict:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return {"outcome": "ignored", "trusted_cmd": controller.get("trusted_cmd")}

    stats_by_cmd = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    row = stats_by_cmd.get(cmd_key)
    if not isinstance(row, dict):
        row = {
            "recent_improvements_mm": [],
            "improve_streak": 0,
            "worsen_streak": 0,
            "last_outcome": None,
            "intensity_scale": float(DEFAULT_INITIAL_INTENSITY_SCALE),
        }
        stats_by_cmd[cmd_key] = row
        controller["cmd_stats"] = stats_by_cmd

    improvement_val = _coerce_float(x_improvement_mm, None)
    if improvement_val is None:
        outcome = "unknown"
    elif float(improvement_val) >= float(min_progress_mm):
        outcome = "improved"
    elif float(improvement_val) <= (-1.0 * float(worsen_mm)):
        outcome = "worsened"
    else:
        outcome = "neutral"

    recent = list(row.get("recent_improvements_mm") or [])
    if improvement_val is not None:
        recent.append(float(improvement_val))
    recent = recent[-int(max(1, int(DEFAULT_RECENT_EFFICACY_WINDOW))):]
    row["recent_improvements_mm"] = recent

    current_scale = _coerce_float(row.get("intensity_scale"), DEFAULT_INITIAL_INTENSITY_SCALE)
    if outcome == "improved":
        row["improve_streak"] = int(row.get("improve_streak") or 0) + 1
        row["worsen_streak"] = 0
        current_scale = min(
            float(DEFAULT_MAX_INTENSITY_SCALE),
            float(current_scale) + 0.10,
        )
        if int(row["improve_streak"]) >= int(DEFAULT_DIRECTION_CONFIRM_CYCLES):
            controller["trusted_cmd"] = cmd_key
            controller["trusted_confirmations"] = int(row["improve_streak"])
    elif outcome == "worsened":
        row["worsen_streak"] = int(row.get("worsen_streak") or 0) + 1
        row["improve_streak"] = 0
        current_scale = max(
            float(DEFAULT_MIN_INTENSITY_SCALE),
            float(current_scale) * 0.70,
        )
        if str(controller.get("trusted_cmd") or "").strip().lower() == cmd_key:
            controller["trusted_cmd"] = None
            controller["trusted_confirmations"] = 0
    else:
        row["improve_streak"] = 0
        row["worsen_streak"] = 0
        current_scale = max(
            float(DEFAULT_MIN_INTENSITY_SCALE),
            min(float(DEFAULT_MAX_INTENSITY_SCALE), float(current_scale) * 0.95),
        )
        if str(controller.get("trusted_cmd") or "").strip().lower() == cmd_key:
            controller["trusted_cmd"] = None
            controller["trusted_confirmations"] = 0

    row["last_outcome"] = str(outcome)
    row["intensity_scale"] = float(current_scale)
    return {
        "outcome": str(outcome),
        "x_improvement_mm": _round_triplet(improvement_val),
        "trusted_cmd": controller.get("trusted_cmd"),
        "trusted_confirmations": int(controller.get("trusted_confirmations") or 0),
        "stats": _controller_snapshot(controller).get("cmd_stats", {}).get(cmd_key),
    }


def _controller_intensity_scale(controller: dict, cmd: str | None) -> float:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return float(DEFAULT_INITIAL_INTENSITY_SCALE)
    stats_by_cmd = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    row = stats_by_cmd.get(cmd_key) if isinstance(stats_by_cmd.get(cmd_key), dict) else {}
    return float(
        _clamp(
            _coerce_float(row.get("intensity_scale"), DEFAULT_INITIAL_INTENSITY_SCALE),
            DEFAULT_MIN_INTENSITY_SCALE,
            DEFAULT_MAX_INTENSITY_SCALE,
        )
    )


def _select_turn_cmd(
    controller: dict,
    *,
    analytics_cmd: str | None,
) -> tuple[str | None, str]:
    trusted_cmd = str(controller.get("trusted_cmd") or "").strip().lower()
    if trusted_cmd in {"l", "r"} and int(controller.get("trusted_confirmations") or 0) >= int(DEFAULT_DIRECTION_CONFIRM_CYCLES):
        return trusted_cmd, "trusted_confirmed"

    stats_by_cmd = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    analytics_key = str(analytics_cmd or "").strip().lower()
    if analytics_key in {"l", "r"}:
        chosen = analytics_key
        source = "analytics"
    else:
        chosen = None
        source = "none"

    if chosen in {"l", "r"}:
        chosen_row = stats_by_cmd.get(chosen) if isinstance(stats_by_cmd.get(chosen), dict) else {}
        chosen_mean = _recent_mean(list(chosen_row.get("recent_improvements_mm") or []))
        chosen_last = _recent_last(list(chosen_row.get("recent_improvements_mm") or []))
        chosen_worsen_streak = int(chosen_row.get("worsen_streak") or 0)
        opposite = _opposite_turn_cmd(chosen)
        opp_row = stats_by_cmd.get(opposite) if isinstance(stats_by_cmd.get(opposite), dict) else {}
        opp_mean = _recent_mean(list(opp_row.get("recent_improvements_mm") or []))
        opp_improve_streak = int(opp_row.get("improve_streak") or 0)
        if chosen_worsen_streak >= 2 or (
            chosen_last is not None and float(chosen_last) <= (-1.0 * float(DEFAULT_DIRECTION_HARD_FLIP_WORSEN_MM))
        ):
            return opposite, "bias_away_hard"
        if chosen_worsen_streak >= 1:
            if chosen_mean is not None and chosen_mean > 0.0:
                return chosen, "hold_positive_mean"
            return chosen, "hold_single_worsen"
        if chosen_mean is not None and chosen_mean < 0.0 and opp_mean is not None and opp_mean > chosen_mean:
            return opposite, "bias_recent_mean"
        return chosen, source

    ranked = []
    for cmd in ("l", "r"):
        row = stats_by_cmd.get(cmd) if isinstance(stats_by_cmd.get(cmd), dict) else {}
        recent_mean = _recent_mean(list(row.get("recent_improvements_mm") or []))
        ranked.append(
            (
                float(recent_mean) if recent_mean is not None else float("-inf"),
                int(row.get("improve_streak") or 0),
                str(cmd),
            )
        )
    ranked.sort(reverse=True)
    if ranked and ranked[0][0] != float("-inf"):
        return ranked[0][2], "recent_mean"
    return None, "none"


def _controller_direction_unstable(controller: dict) -> bool:
    stats_by_cmd = controller.get("cmd_stats") if isinstance(controller.get("cmd_stats"), dict) else {}
    if str(controller.get("trusted_cmd") or "").strip().lower() in {"l", "r"}:
        return False
    bad_count = 0
    for cmd in ("l", "r"):
        row = stats_by_cmd.get(cmd) if isinstance(stats_by_cmd.get(cmd), dict) else {}
        worsen_streak = int(row.get("worsen_streak") or 0)
        recent_mean = _recent_mean(list(row.get("recent_improvements_mm") or []))
        recent_last = _recent_last(list(row.get("recent_improvements_mm") or []))
        if (
            worsen_streak >= 2
            or (recent_last is not None and float(recent_last) <= (-1.0 * float(DEFAULT_DIRECTION_HARD_FLIP_WORSEN_MM)))
            or (recent_mean is not None and recent_mean <= (-1.0 * float(DEFAULT_DIRECTION_FLIP_WORSEN_MM)))
        ):
            bad_count += 1
    return bad_count >= 2


def _pose_snapshot(pose: dict | None) -> dict | None:
    if not isinstance(pose, dict):
        return None
    return {
        "offset_y": _round_triplet(pose.get("offset_y")),
        "offset_x": _round_triplet(pose.get("offset_x")),
        "dist": _round_triplet(pose.get("dist")),
        "angle": _round_triplet(pose.get("angle")),
        "confidence": _round_triplet(pose.get("confidence")),
        "obs_ts": _round_triplet(pose.get("obs_ts")),
        "pose_source": str(pose.get("pose_source") or ""),
        "lite_required_frames": (
            int(pose.get("lite_required_frames"))
            if pose.get("lite_required_frames") is not None
            else None
        ),
        "samples_used": (
            int(pose.get("samples_used"))
            if pose.get("samples_used") is not None
            else None
        ),
        "lite_frame_count": (
            int(pose.get("lite_frame_count"))
            if pose.get("lite_frame_count") is not None
            else None
        ),
        "lite_frame_first_id": (
            int(pose.get("lite_frame_first_id"))
            if pose.get("lite_frame_first_id") is not None
            else None
        ),
        "lite_frame_last_id": (
            int(pose.get("lite_frame_last_id"))
            if pose.get("lite_frame_last_id") is not None
            else None
        ),
        "lite_frame_span_s": _round_triplet(pose.get("lite_frame_span_s")),
        "sample_obs_ts_start": _round_triplet(pose.get("sample_obs_ts_start")),
        "sample_obs_ts_end": _round_triplet(pose.get("sample_obs_ts_end")),
        "sample_obs_span_s": _round_triplet(pose.get("sample_obs_span_s")),
    }


def _pose_summary_text(pose: dict | None) -> str:
    if not isinstance(pose, dict):
        return "unavailable"
    return (
        f"x={_mm_non_negative(pose.get('offset_x'))}mm "
        f"y={_mm_non_negative(pose.get('offset_y'))}mm "
        f"dist={_mm_non_negative(pose.get('dist'))}mm "
        f"angle={_round_triplet(pose.get('angle'))}deg "
        f"conf={_round_triplet(pose.get('confidence'))}% "
        f"via {str(pose.get('pose_source') or 'unknown')}"
    )


def _observation_latency_details(
    *,
    send_result: dict | None,
    pose: dict | None,
) -> dict:
    send_started_ts = _coerce_float((send_result or {}).get("send_started_ts"), None)
    send_completed_ts = _coerce_float((send_result or {}).get("send_completed_ts"), None)
    obs_ts = _coerce_float((pose or {}).get("obs_ts"), None)
    sample_obs_ts_start = _coerce_float((pose or {}).get("sample_obs_ts_start"), None)
    sample_obs_ts_end = _coerce_float((pose or {}).get("sample_obs_ts_end"), None)
    return {
        "send_started_ts": _round_triplet(send_started_ts),
        "send_completed_ts": _round_triplet(send_completed_ts),
        "observation_ts": _round_triplet(obs_ts),
        "sample_obs_ts_start": _round_triplet(sample_obs_ts_start),
        "sample_obs_ts_end": _round_triplet(sample_obs_ts_end),
        "send_to_observation_s": _round_triplet(
            (float(obs_ts) - float(send_completed_ts))
            if obs_ts is not None and send_completed_ts is not None
            else None
        ),
        "send_to_sample_start_s": _round_triplet(
            (float(sample_obs_ts_start) - float(send_completed_ts))
            if sample_obs_ts_start is not None and send_completed_ts is not None
            else None
        ),
        "send_span_s": _round_triplet(
            (float(send_completed_ts) - float(send_started_ts))
            if send_started_ts is not None and send_completed_ts is not None
            else None
        ),
        "lite_frame_count": (
            int(pose.get("lite_frame_count"))
            if isinstance(pose, dict) and pose.get("lite_frame_count") is not None
            else None
        ),
        "lite_required_frames": (
            int(pose.get("lite_required_frames"))
            if isinstance(pose, dict) and pose.get("lite_required_frames") is not None
            else None
        ),
        "samples_used": (
            int(pose.get("samples_used"))
            if isinstance(pose, dict) and pose.get("samples_used") is not None
            else None
        ),
        "lite_frame_span_s": _round_triplet((pose or {}).get("lite_frame_span_s")),
        "lite_frame_first_id": (
            int(pose.get("lite_frame_first_id"))
            if isinstance(pose, dict) and pose.get("lite_frame_first_id") is not None
            else None
        ),
        "lite_frame_last_id": (
            int(pose.get("lite_frame_last_id"))
            if isinstance(pose, dict) and pose.get("lite_frame_last_id") is not None
            else None
        ),
    }


def _observation_latency_log_text(
    *,
    stage_label: str,
    send_result: dict | None,
    pose: dict | None,
    meta: dict | None = None,
) -> str:
    details = _observation_latency_details(send_result=send_result, pose=pose)
    mode_text = str((meta or {}).get("mode") or "")
    return (
        f"{str(stage_label)} observe: mode={mode_text or 'unknown'} "
        f"send->obs={_round_triplet(details.get('send_to_observation_s'))}s "
        f"send->sample_start={_round_triplet(details.get('send_to_sample_start_s'))}s "
        f"samples={details.get('samples_used')} req_frames/sample={details.get('lite_required_frames')} "
        f"unique_frames={details.get('lite_frame_count')} "
        f"frame_ids={details.get('lite_frame_first_id')}..{details.get('lite_frame_last_id')} "
        f"frame_span={_round_triplet(details.get('lite_frame_span_s'))}s"
    )


def _latest_smoothed_frame_ts(world) -> float | None:
    history = getattr(world, "_smoothed_frame_history", None)
    if not history:
        return None
    for entry in reversed(list(history)):
        if not isinstance(entry, dict):
            continue
        ts = _coerce_float(entry.get("timestamp"), None)
        if ts is not None:
            return float(ts)
    return None


def _post_act_observe_not_before_ts(
    *,
    send_result: dict | None,
    effective_duration_ms: int | float | None,
    settle_s: float,
) -> float | None:
    send_completed_ts = _coerce_float((send_result or {}).get("send_completed_ts"), None)
    effective_duration_s = _coerce_float(effective_duration_ms, None)
    if send_completed_ts is None or effective_duration_s is None:
        return None
    return float(send_completed_ts) + max(
        float(settle_s),
        (float(effective_duration_s) / 1000.0) + 0.05,
    )


def _current_world_pose(world, *, obs_ts: float | None = None) -> dict | None:
    return brick_pose_from_world(world, obs_ts=float(obs_ts if obs_ts is not None else time.time()))


def _moving_sample_row(
    world,
    *,
    trial_index: int,
    phase_index: int,
    phase_name: str,
    cmd: str,
    phase_elapsed_s: float,
    run_elapsed_s: float,
    obs_ts: float,
) -> dict:
    pose = _current_world_pose(world, obs_ts=float(obs_ts))
    analytics = _analytics_for_pose(world, pose, step=EXPERIMENT_STEP) if pose is not None else {}
    errors = _pose_gate_errors(pose, _step_success_gates(world, EXPERIMENT_STEP)) if pose is not None else None
    brick = dict(getattr(world, "brick", {}) or {})
    return {
        "t": _round_triplet(obs_ts),
        "run_elapsed_s": _round_triplet(run_elapsed_s),
        "phase_elapsed_s": _round_triplet(phase_elapsed_s),
        "trial": int(trial_index),
        "phase_index": int(phase_index),
        "phase": str(phase_name),
        "axis": _movement_axis_for_cmd(cmd),
        "cmd": str(cmd),
        "visible": bool(brick.get("visible")),
        "confidence": _round_triplet((pose or {}).get("confidence")),
        "offset_x": _round_triplet((pose or {}).get("offset_x")),
        "offset_y": _round_triplet((pose or {}).get("offset_y")),
        "dist": _round_triplet((pose or {}).get("dist")),
        "angle": _round_triplet((pose or {}).get("angle")),
        "pose_source": str((pose or {}).get("pose_source") or brick.get("pose_source") or ""),
        "progress_pct": _progress_pct(analytics),
        "gate_error_mm": dict(errors or {}) if isinstance(errors, dict) else None,
        "camera_fps_est": _round_triplet(getattr(world, "_camera_fps", None)),
        "vision_backend": str(getattr(world, "_vision_backend", "") or ""),
    }


def _summary_stat(values: list[float]) -> dict:
    if not values:
        return {
            "start": None,
            "end": None,
            "best": None,
            "worst": None,
            "median": None,
        }
    return {
        "start": _round_triplet(values[0]),
        "end": _round_triplet(values[-1]),
        "best": _round_triplet(min(values)),
        "worst": _round_triplet(max(values)),
        "median": _round_triplet(statistics.median(values)),
    }


def _pose_metric_delta(
    before_pose: dict | None,
    after_pose: dict | None,
    metric_key: str,
) -> float | None:
    before_value = _coerce_float((before_pose or {}).get(metric_key), None)
    after_value = _coerce_float((after_pose or {}).get(metric_key), None)
    if before_value is None or after_value is None:
        return None
    return _round_triplet(float(after_value) - float(before_value))


def _normalize_turn_drive_strength(value) -> str:
    strength = str(value or "").strip().lower()
    if strength in {"strong", "medium", "light"}:
        return strength
    return str(DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH)


def _turn_drive_profile_name_for_strength(
    *,
    drive_mode: str,
    strength: str | None,
    fallback_profile_name: str | None = None,
) -> str | None:
    drive_key = str(drive_mode or "").strip().lower()
    strength_key = _normalize_turn_drive_strength(strength)
    mode_profiles = (
        ALTERNATING_TURN_DRIVE_STRENGTH_PROFILES.get(drive_key)
        if isinstance(ALTERNATING_TURN_DRIVE_STRENGTH_PROFILES.get(drive_key), dict)
        else {}
    )
    profile_name = str(mode_profiles.get(strength_key) or "").strip()
    if profile_name:
        return str(profile_name)
    fallback_name = str(fallback_profile_name or "").strip()
    return fallback_name or None


def _inverse_cmd(cmd: str | None) -> str | None:
    return {
        "l": "r",
        "r": "l",
        "f": "b",
        "b": "f",
        "u": "d",
        "d": "u",
    }.get(str(cmd or "").strip().lower())


def _inverse_drive_mode(drive_mode: str | None) -> str | None:
    drive_key = str(drive_mode or "").strip().lower()
    if drive_key == "forward":
        return "backward"
    if drive_key == "backward":
        return "forward"
    return None


def _turn_drive_strength_for_profile_name(
    profile_name: str | None,
    *,
    drive_mode: str | None = None,
) -> str | None:
    profile_key = str(profile_name or "").strip()
    drive_key = str(drive_mode or "").strip().lower() or None
    if not profile_key:
        return None
    for candidate_drive_mode, mode_profiles in ALTERNATING_TURN_DRIVE_STRENGTH_PROFILES.items():
        if drive_key is not None and str(candidate_drive_mode) != str(drive_key):
            continue
        if not isinstance(mode_profiles, dict):
            continue
        for strength_key, candidate_profile_name in mode_profiles.items():
            if str(candidate_profile_name or "").strip() == profile_key:
                return str(strength_key)
    return None


def _turn_drive_phase_label(*, cmd: str | None, drive_mode: str | None) -> str:
    cmd_key = str(cmd or "").strip().lower()
    drive_key = str(drive_mode or "").strip().lower()
    turn_text = {"l": "LEFT", "r": "RIGHT"}.get(cmd_key)
    drive_text = {"forward": "FORWARD", "backward": "BACK"}.get(drive_key)
    if turn_text and drive_text:
        return f"{drive_text}+{turn_text}"
    return str(cmd_key or drive_key or "UNKNOWN").upper()


def _adaptive_turn_label_for_drive_mode(drive_mode: str | None) -> str:
    drive_key = str(drive_mode or "").strip().lower()
    if drive_key == "forward":
        return "ADAPTIVE FORWARD TURN"
    if drive_key == "backward":
        return "ADAPTIVE BACK TURN"
    return "ADAPTIVE TURN"


def _turn_dir_for_x_axis(
    x_axis,
    *,
    fallback_cmd: str | None = None,
) -> str:
    x_val = _coerce_float(x_axis, None)
    if x_val is not None:
        return "right" if float(x_val) < 0.0 else "left"
    return "right" if str(fallback_cmd or "").strip().lower() == "r" else "left"


def _turn_cmd_for_direction(turn_dir: str | None) -> str:
    return "r" if str(turn_dir or "").strip().lower() == "right" else "l"


def _distance_hold_drive_mode(
    current_dist,
    *,
    reference_dist,
    fallback_drive_mode: str | None = None,
) -> str:
    current_val = _coerce_float(current_dist, None)
    reference_val = _coerce_float(reference_dist, None)
    if current_val is not None and reference_val is not None:
        return "backward" if float(current_val) < float(reference_val) else "forward"
    fallback = str(fallback_drive_mode or "").strip().lower()
    if fallback in {"forward", "backward"}:
        return fallback
    return "forward"


def _phase_key_for_drive_mode_and_turn_dir(
    *,
    drive_mode: str | None,
    turn_dir: str | None,
) -> str:
    drive_key = str(drive_mode or "").strip().lower()
    turn_key = str(turn_dir or "").strip().lower()
    if drive_key == "forward":
        prefix = "forward"
    elif drive_key == "backward":
        prefix = "back"
    else:
        prefix = drive_key or "turn"
    return f"{prefix}_{turn_key or 'left'}"


def _adaptive_turn_drive_phase_spec(
    *,
    drive_mode: str,
    strength: str,
    turn_dir: str,
    profile_name: str | None = None,
    pwm_override=None,
) -> dict | None:
    drive_key = str(drive_mode or "").strip().lower()
    turn_key = str(turn_dir or "").strip().lower()
    if drive_key not in {"forward", "backward"} or turn_key not in {"left", "right"}:
        return None
    cmd = _turn_cmd_for_direction(turn_key)
    profile_name_used = str(profile_name or "").strip() or _turn_drive_profile_name_for_strength(
        drive_mode=str(drive_key),
        strength=str(strength),
    )
    if not profile_name_used:
        return None
    spec = {
        "phase": _phase_key_for_drive_mode_and_turn_dir(
            drive_mode=str(drive_key),
            turn_dir=str(turn_key),
        ),
        "label": _turn_drive_phase_label(
            cmd=str(cmd),
            drive_mode=str(drive_key),
        ),
        "cmd": str(cmd),
        "drive_mode": str(drive_key),
        "profile_name": str(profile_name_used),
        "strength": str(_normalize_turn_drive_strength(strength)),
    }
    pwm_override_val = _coerce_float(pwm_override, None)
    if pwm_override_val is not None:
        spec["pwm_override"] = int(round(float(pwm_override_val)))
    return spec


def _adaptive_turn_drive_segment(
    *,
    drive_mode: str,
    strength: str,
    turn_dir: str,
    score: int,
    duration_ms: int,
    stage: str,
    profile_name: str | None = None,
    pwm_override=None,
    metadata: dict | None = None,
) -> dict | None:
    phase_spec = _adaptive_turn_drive_phase_spec(
        drive_mode=str(drive_mode),
        strength=str(strength),
        turn_dir=str(turn_dir),
        profile_name=profile_name,
        pwm_override=pwm_override,
    )
    if not isinstance(phase_spec, dict):
        return None
    plan = _build_turn_drive_plan_for_phase_spec(
        phase_spec=phase_spec,
        score=int(score),
        duration_ms=int(duration_ms),
        metadata=dict(metadata or {}),
    )
    if not isinstance(plan, dict):
        return None
    return _trial_manifest_segment(
        plan=plan,
        duration_ms=int(duration_ms),
        stage=str(stage),
    )


def _turn_drive_plan_snapshot(plan: dict | None) -> dict | None:
    if not isinstance(plan, dict):
        return None
    return {
        "cmd": str(plan.get("cmd") or ""),
        "score": int(plan.get("score") or 0),
        "duration_ms": int(plan.get("duration_ms") or 0),
        "profile_name": str(plan.get("profile_name") or ""),
        "drive_mode": str(plan.get("drive_mode") or ""),
        "inner_pwm": int(plan.get("inner_pwm") or 0),
        "outer_pwm": int(plan.get("outer_pwm") or 0),
        "action_note": str(plan.get("action_note") or ""),
        "actions": [dict(action) for action in list(plan.get("actions") or [])],
    }


def _inverse_turn_drive_plan(
    plan: dict | None,
    *,
    metadata: dict | None = None,
) -> dict | None:
    if not isinstance(plan, dict):
        return None
    inverse_cmd = _inverse_cmd(plan.get("cmd"))
    inverse_drive_mode = _inverse_drive_mode(plan.get("drive_mode"))
    strength = _turn_drive_strength_for_profile_name(
        plan.get("profile_name"),
        drive_mode=str(plan.get("drive_mode") or ""),
    )
    profile_name = _turn_drive_profile_name_for_strength(
        drive_mode=str(inverse_drive_mode or ""),
        strength=strength,
    )
    try:
        score_val = int(round(float(plan.get("score") or 0)))
        duration_ms = int(round(float(plan.get("duration_ms") or 0)))
    except (TypeError, ValueError):
        return None
    if inverse_cmd is None or inverse_drive_mode is None or profile_name is None:
        return None
    if score_val <= 0 or duration_ms <= 0:
        return None
    return build_turn_drive_motion_plan(
        cmd=str(inverse_cmd),
        score=int(score_val),
        hold_duration_ms=int(duration_ms),
        pwm_override=(
            int(round(float(plan.get("base_pwm"))))
            if _coerce_float(plan.get("base_pwm"), None) is not None
            else None
        ),
        profile_name=str(profile_name),
        drive_mode=str(inverse_drive_mode),
        metadata=dict(metadata or {}),
    )


def _build_turn_drive_plan_for_phase_spec(
    *,
    phase_spec: dict | None,
    score: int,
    duration_ms: int,
    metadata: dict | None = None,
) -> dict | None:
    if not isinstance(phase_spec, dict):
        return None
    try:
        score_val = int(round(float(score)))
        duration_val = int(round(float(duration_ms)))
    except (TypeError, ValueError):
        return None
    if score_val <= 0 or duration_val <= 0:
        return None
    return build_turn_drive_motion_plan(
        cmd=str(phase_spec.get("cmd") or ""),
        score=int(score_val),
        hold_duration_ms=int(duration_val),
        pwm_override=(
            int(round(float(phase_spec.get("pwm_override"))))
            if _coerce_float(phase_spec.get("pwm_override"), None) is not None
            else None
        ),
        profile_name=str(phase_spec.get("profile_name") or ""),
        drive_mode=str(phase_spec.get("drive_mode") or ""),
        metadata=dict(metadata or {}),
    )


def _phase_key_for_turn_drive_plan(plan: dict | None) -> str:
    if not isinstance(plan, dict):
        return ""
    return (
        _turn_drive_phase_label(
            cmd=str(plan.get("cmd") or ""),
            drive_mode=str(plan.get("drive_mode") or ""),
        )
        .lower()
        .replace("+", "_")
    )


def _trial_manifest_segment(
    *,
    plan: dict | None,
    duration_ms: int | None = None,
    stage: str,
) -> dict | None:
    if not isinstance(plan, dict):
        return None
    try:
        duration_val = int(round(float(duration_ms if duration_ms is not None else plan.get("duration_ms") or 0)))
    except (TypeError, ValueError):
        duration_val = 0
    if duration_val <= 0:
        return None
    plan_with_duration = dict(plan)
    plan_with_duration["duration_ms"] = int(duration_val)
    label = _turn_drive_phase_label(
        cmd=str(plan_with_duration.get("cmd") or ""),
        drive_mode=str(plan_with_duration.get("drive_mode") or ""),
    )
    segment = {
        "stage": str(stage or ""),
        "phase": _phase_key_for_turn_drive_plan(plan_with_duration),
        "label": str(label),
        "cmd": str(plan_with_duration.get("cmd") or ""),
        "drive_mode": str(plan_with_duration.get("drive_mode") or ""),
        "turning": str(plan_with_duration.get("cmd") or "").strip().lower() in {"l", "r"},
        "duration_ms": int(duration_val),
        "score_pct": int(plan_with_duration.get("score") or 0),
        "profile_name": str(plan_with_duration.get("profile_name") or ""),
        "action_note": str(plan_with_duration.get("action_note") or ""),
        "motor_pair": _turn_drive_motor_pair_summary(plan_with_duration),
    }
    pwm_override_val = _coerce_float(plan_with_duration.get("base_pwm"), None)
    if pwm_override_val is not None:
        segment["pwm_override"] = int(round(float(pwm_override_val)))
    return segment


def _trial_manifest_phase_template_from_segment(
    segment: dict | None,
    *,
    include_duration: bool = False,
) -> dict | None:
    if not isinstance(segment, dict):
        return None
    template = {
        "stage": str(segment.get("stage") or ""),
        "phase": str(segment.get("phase") or ""),
        "label": str(segment.get("label") or ""),
        "cmd": str(segment.get("cmd") or ""),
        "drive_mode": str(segment.get("drive_mode") or ""),
        "turning": bool(segment.get("turning")),
        "score_pct": int(segment.get("score_pct") or 0),
        "profile_name": str(segment.get("profile_name") or ""),
        "action_note": str(segment.get("action_note") or ""),
        "motor_pair": dict(segment.get("motor_pair") or {}),
    }
    pwm_override_val = _coerce_float(segment.get("pwm_override"), None)
    if pwm_override_val is not None:
        template["pwm_override"] = int(round(float(pwm_override_val)))
    if bool(include_duration):
        duration_val = int(_coerce_float(segment.get("duration_ms"), 0) or 0)
        if duration_val > 0:
            template["duration_ms"] = int(duration_val)
    return template


def _trial_manifest_segment_from_template(
    template: dict | None,
    *,
    duration_ms: int | None = None,
) -> dict | None:
    if not isinstance(template, dict):
        return None
    try:
        duration_val = int(round(float(duration_ms if duration_ms is not None else template.get("duration_ms") or 0)))
    except (TypeError, ValueError):
        duration_val = 0
    if duration_val <= 0:
        return None
    label = str(template.get("label") or _turn_drive_phase_label(
        cmd=str(template.get("cmd") or ""),
        drive_mode=str(template.get("drive_mode") or ""),
    ))
    return {
        "stage": str(template.get("stage") or ""),
        "phase": str(template.get("phase") or ""),
        "label": str(label),
        "cmd": str(template.get("cmd") or ""),
        "drive_mode": str(template.get("drive_mode") or ""),
        "turning": bool(template.get("turning")),
        "duration_ms": int(duration_val),
        "score_pct": int(template.get("score_pct") or 0),
        "profile_name": str(template.get("profile_name") or ""),
        "action_note": str(template.get("action_note") or ""),
        "motor_pair": dict(template.get("motor_pair") or {}),
        "pwm_override": (
            int(round(float(template.get("pwm_override"))))
            if _coerce_float(template.get("pwm_override"), None) is not None
            else None
        ),
    }


def _trial_results_placeholder() -> dict:
    return {
        "status": "planned",
        "start_dist_mm": TRIAL_RESULT_TBD,
        "end_dist_mm": TRIAL_RESULT_TBD,
        "dist_change_mm": TRIAL_RESULT_TBD,
        "dist_traveled_mm": TRIAL_RESULT_TBD,
        "start_x_axis_mm": TRIAL_RESULT_TBD,
        "end_x_axis_mm": TRIAL_RESULT_TBD,
        "x_axis_change_mm": TRIAL_RESULT_TBD,
        "x_gap_closed_mm": TRIAL_RESULT_TBD,
        "observation_usable": TRIAL_RESULT_TBD,
        "recovery_used": TRIAL_RESULT_TBD,
        "recovery_succeeded": TRIAL_RESULT_TBD,
        "stop_reason": TRIAL_RESULT_TBD,
    }


def _trial_manifest_compact_results_placeholder() -> dict:
    return {
        "startDist": TRIAL_RESULT_TBD,
        "distDelta": TRIAL_RESULT_TBD,
        "xDelta": TRIAL_RESULT_TBD,
        "xGapClosed": TRIAL_RESULT_TBD,
        "startConf": TRIAL_RESULT_TBD,
        "endConf": TRIAL_RESULT_TBD,
        "usable": TRIAL_RESULT_TBD,
        "status": "planned",
        "stopReason": TRIAL_RESULT_TBD,
    }


def _trial_manifest_placeholder_normalize(value):
    return TRIAL_RESULT_TBD if str(value or "").strip().upper() == "TBD" else value


def _trial_manifest_curve_pair_text(
    setup_segment: dict | None,
    measured_segment: dict | None,
    *,
    fallback: str | None = None,
) -> str:
    labels = []
    if isinstance(setup_segment, dict):
        text = str(setup_segment.get("label") or "").strip()
        if text:
            labels.append(text)
    if isinstance(measured_segment, dict):
        text = str(measured_segment.get("label") or "").strip()
        if text:
            labels.append(text)
    if labels:
        return " -> ".join(labels)
    value = str(fallback or "").strip()
    return value or TRIAL_RESULT_TBD


def _trial_manifest_motor_value(value):
    if _coerce_float(value, None) is None:
        return TRIAL_RESULT_TBD
    return int(round(float(value)))


def _trial_manifest_turn_selection_policy(manifest_or_curve) -> str:
    curve_cfg = (
        _trial_manifest_curve_config(manifest_or_curve)
        if isinstance(manifest_or_curve, dict) and "curve" in manifest_or_curve
        else dict(manifest_or_curve or {}) if isinstance(manifest_or_curve, dict) else {}
    )
    return str(curve_cfg.get("turn_selection_policy") or "").strip().lower()


def _trial_manifest_uses_adaptive_turn_selection(manifest_or_curve) -> bool:
    return _trial_manifest_turn_selection_policy(manifest_or_curve) == TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS


def _trial_manifest_distance_hold_policy(manifest_or_curve) -> str:
    curve_cfg = (
        _trial_manifest_curve_config(manifest_or_curve)
        if isinstance(manifest_or_curve, dict) and "curve" in manifest_or_curve
        else dict(manifest_or_curve or {}) if isinstance(manifest_or_curve, dict) else {}
    )
    value = str(curve_cfg.get("distance_hold_policy") or "").strip().lower()
    if value:
        return value
    setup_range = curve_cfg.get("setup_turn_duration_range_ms")
    if isinstance(setup_range, dict):
        return TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
    return ""


def _trial_manifest_uses_distance_hold(manifest_or_curve) -> bool:
    return (
        _trial_manifest_distance_hold_policy(manifest_or_curve)
        == TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
    )


def _trial_manifest_default_curve_pair_text(manifest: dict | None) -> str:
    curve_cfg = _trial_manifest_curve_config(manifest)
    value = str(curve_cfg.get("curve_pair_label") or "").strip()
    if value:
        return value
    if _trial_manifest_uses_adaptive_turn_selection(curve_cfg):
        labels = []
        setup_range = curve_cfg.get("setup_turn_duration_range_ms")
        if isinstance(setup_range, dict):
            labels.append(_adaptive_turn_label_for_drive_mode("forward"))
        labels.append(_adaptive_turn_label_for_drive_mode(
            str(((curve_cfg.get("measured_phase") or {}).get("drive_mode") or "backward"))
        ))
        return " -> ".join(labels)
    setup_segment = _trial_manifest_setup_segment({}, manifest=manifest)
    measured_segment = _trial_manifest_measured_segment({}, manifest=manifest)
    return _trial_manifest_curve_pair_text(
        setup_segment,
        measured_segment,
        fallback=None,
    )


def _trial_manifest_row_is_usable_complete(row: dict | None) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("usable") is not True:
        return False
    return all(
        _coerce_float(row.get(key), None) is not None
        for key in ("distDelta", "xDelta", "xGapClosed")
    )


def _trial_manifest_trial_rows_completed_count(trial_rows: list[dict]) -> int:
    return int(
        sum(
            1
            for row in list(trial_rows or [])
            if str((row or {}).get("status") or "").strip().lower() != "planned"
        )
    )


def _trial_manifest_merge_defaults(
    defaults: dict | None,
    existing: dict | None,
) -> dict:
    merged = dict(defaults or {})
    if not isinstance(existing, dict):
        return merged
    for key, value in existing.items():
        merged[key] = _trial_manifest_placeholder_normalize(value)
    return merged


def _trial_manifest_compact_trial_row(
    *,
    trial_index: int,
    duration_ms: int | None,
    measured_duration_ms: int | None = None,
    setup_profile: str | None = None,
    setup_score=None,
    setup_pwm=None,
    measured_profile: str | None = None,
    measured_score=None,
    measured_pwm=None,
    setup_left_motor_pwr=None,
    setup_right_motor_pwr=None,
    measured_left_motor_pwr=None,
    measured_right_motor_pwr=None,
    curve_pair: str | None = None,
    left_motor_pwr=None,
    right_motor_pwr=None,
    result_values: dict | None = None,
) -> dict:
    setup_duration_val = int(_coerce_float(duration_ms, 0) or 0)
    measured_duration_val = _coerce_float(measured_duration_ms, None)
    measured_duration_int = (
        int(round(float(measured_duration_val)))
        if measured_duration_val is not None and measured_duration_val > 0
        else None
    )
    row = {
        "trial": int(max(1, int(trial_index))),
        "setupDurationMs": int(setup_duration_val),
    }
    if measured_duration_int is not None:
        row["measuredDurationMs"] = int(measured_duration_int)
    if str(setup_profile or "").strip():
        row["setupProfile"] = str(setup_profile)
    setup_score_val = _coerce_float(setup_score, None)
    if setup_score_val is not None:
        row["setupScore"] = int(round(float(setup_score_val)))
    setup_pwm_val = _coerce_float(setup_pwm, None)
    if setup_pwm_val is not None:
        row["setupPwm"] = int(round(float(setup_pwm_val)))
    if str(measured_profile or "").strip():
        row["measuredProfile"] = str(measured_profile)
    measured_score_val = _coerce_float(measured_score, None)
    if measured_score_val is not None:
        row["measuredScore"] = int(round(float(measured_score_val)))
    measured_pwm_val = _coerce_float(measured_pwm, None)
    if measured_pwm_val is not None:
        row["measuredPwm"] = int(round(float(measured_pwm_val)))
    row["curvePair"] = str(curve_pair or TRIAL_RESULT_TBD)
    row["leftMotorPwr"] = _trial_manifest_motor_value(left_motor_pwr)
    row["rightMotorPwr"] = _trial_manifest_motor_value(right_motor_pwr)
    row["setupLeftMotorPwr"] = _trial_manifest_motor_value(setup_left_motor_pwr)
    row["setupRightMotorPwr"] = _trial_manifest_motor_value(setup_right_motor_pwr)
    row["measuredLeftMotorPwr"] = _trial_manifest_motor_value(measured_left_motor_pwr)
    row["measuredRightMotorPwr"] = _trial_manifest_motor_value(measured_right_motor_pwr)
    row.update(_trial_manifest_compact_results_placeholder())
    if isinstance(result_values, dict):
        for key in row:
            if key in {
                "trial",
                "setupProfile",
                "setupScore",
                "setupPwm",
                "measuredProfile",
                "measuredScore",
                "measuredPwm",
                "setupDurationMs",
                "curvePair",
                "leftMotorPwr",
                "rightMotorPwr",
                "setupLeftMotorPwr",
                "setupRightMotorPwr",
                "measuredLeftMotorPwr",
                "measuredRightMotorPwr",
                "measuredDurationMs",
            }:
                continue
            if key in result_values:
                row[key] = _trial_manifest_placeholder_normalize(result_values.get(key))
    return row


def _trial_manifest_curve_stats_placeholder() -> dict:
    return {
        "consistency_score_pct": TRIAL_RESULT_TBD,
        "x_gap_close_rate_pct": TRIAL_RESULT_TBD,
        "median_x_axis_change_mm": TRIAL_RESULT_TBD,
        "median_dist_change_mm": TRIAL_RESULT_TBD,
        "median_dist_traveled_mm": TRIAL_RESULT_TBD,
        "median_x_gap_closed_mm": TRIAL_RESULT_TBD,
        "median_start_confidence_pct": TRIAL_RESULT_TBD,
        "median_end_confidence_pct": TRIAL_RESULT_TBD,
        "median_confidence_delta_pct": TRIAL_RESULT_TBD,
        "usable_trials": TRIAL_RESULT_TBD,
        "production_worthy": TRIAL_RESULT_TBD,
    }


def _trial_manifest_run_status_placeholder(*, trial_count: int) -> dict:
    return {
        "status": "planned",
        "trials_requested": int(max(1, int(trial_count))),
        "trials_completed": 0,
        "stop_reason": "not_run",
        "usable_trials": TRIAL_RESULT_TBD,
        "last_started_ts": None,
        "last_completed_ts": None,
        "seconds": TRIAL_RESULT_TBD,
    }


def _trial_file_name_from_manifest(
    manifest: dict | None,
    *,
    fallback: str | None = None,
) -> str:
    if isinstance(manifest, dict):
        for key in ("name", "curve_name"):
            value = str(manifest.get(key) or "").strip()
            if value:
                return value
    value = str(fallback or "").strip()
    return value or "trials"


def _trial_manifest_curve_config(manifest: dict | None) -> dict:
    if not isinstance(manifest, dict):
        return {}
    curve = manifest.get("curve")
    if isinstance(curve, dict):
        return dict(curve)
    return dict(manifest)


def _trial_manifest_merge_trial_blocks(
    trials_forward: list[dict] | None,
    trials_backwards: list[dict] | None,
) -> list[dict]:
    by_trial = {}
    for row in list(trials_forward or []):
        if not isinstance(row, dict):
            continue
        trial_id = int(_coerce_float(row.get("trial"), 0) or 0)
        if trial_id <= 0:
            continue
        merged = dict(by_trial.get(trial_id) or {})
        merged["trial"] = int(trial_id)
        if row.get("setupDurationMs") is not None:
            merged["setupDurationMs"] = _trial_manifest_placeholder_normalize(row.get("setupDurationMs"))
        elif row.get("duration") is not None:
            merged["setupDurationMs"] = _trial_manifest_placeholder_normalize(row.get("duration"))
        for key in (
            "setupProfile",
            "setupScore",
            "setupPwm",
            "setupLeftMotorPwr",
            "setupRightMotorPwr",
            "curvePair",
            "leftMotorPwr",
            "rightMotorPwr",
        ):
            if key in row:
                merged[key] = _trial_manifest_placeholder_normalize(row.get(key))
        by_trial[int(trial_id)] = merged

    for row in list(trials_backwards or []):
        if not isinstance(row, dict):
            continue
        trial_id = int(_coerce_float(row.get("trial"), 0) or 0)
        if trial_id <= 0:
            continue
        merged = dict(by_trial.get(trial_id) or {})
        merged["trial"] = int(trial_id)
        if row.get("measuredDurationMs") is not None:
            merged["measuredDurationMs"] = _trial_manifest_placeholder_normalize(row.get("measuredDurationMs"))
        elif row.get("measured_duration") is not None:
            merged["measuredDurationMs"] = _trial_manifest_placeholder_normalize(row.get("measured_duration"))
        for key in (
            "measuredProfile",
            "measuredScore",
            "measuredPwm",
            "measuredLeftMotorPwr",
            "measuredRightMotorPwr",
            "curvePair",
            "leftMotorPwr",
            "rightMotorPwr",
            "startDist",
            "distDelta",
            "xDelta",
            "xGapClosed",
            "startConf",
            "endConf",
            "usable",
            "status",
            "stopReason",
        ):
            if key in row:
                if key in {"distDelta", "xDelta", "xGapClosed"}:
                    merged[key] = _mm_non_negative(row.get(key))
                else:
                    merged[key] = _trial_manifest_placeholder_normalize(row.get(key))
        by_trial[int(trial_id)] = merged

    return [dict(by_trial[key]) for key in sorted(by_trial.keys())]


def _trial_manifest_split_trial_blocks(trial_rows: list[dict] | None) -> tuple[list[dict], list[dict]]:
    forward_rows: list[dict] = []
    backward_rows: list[dict] = []
    for row in list(trial_rows or []):
        if not isinstance(row, dict):
            continue
        trial_id = int(_coerce_float(row.get("trial"), 0) or 0)
        if trial_id <= 0:
            continue
        forward_row = {
            "trial": int(trial_id),
            "setupDurationMs": _trial_manifest_placeholder_normalize(row.get("setupDurationMs")),
            "setupProfile": _trial_manifest_placeholder_normalize(row.get("setupProfile")),
            "setupScore": _trial_manifest_placeholder_normalize(row.get("setupScore")),
            "setupPwm": _trial_manifest_placeholder_normalize(row.get("setupPwm")),
            "setupLeftMotorPwr": _trial_manifest_placeholder_normalize(row.get("setupLeftMotorPwr")),
            "setupRightMotorPwr": _trial_manifest_placeholder_normalize(row.get("setupRightMotorPwr")),
            "curvePair": _trial_manifest_placeholder_normalize(row.get("curvePair")),
            "leftMotorPwr": _trial_manifest_placeholder_normalize(row.get("leftMotorPwr")),
            "rightMotorPwr": _trial_manifest_placeholder_normalize(row.get("rightMotorPwr")),
        }
        backward_row = {
            "trial": int(trial_id),
            "measuredDurationMs": _trial_manifest_placeholder_normalize(row.get("measuredDurationMs")),
            "measuredProfile": _trial_manifest_placeholder_normalize(row.get("measuredProfile")),
            "measuredScore": _trial_manifest_placeholder_normalize(row.get("measuredScore")),
            "measuredPwm": _trial_manifest_placeholder_normalize(row.get("measuredPwm")),
            "measuredLeftMotorPwr": _trial_manifest_placeholder_normalize(row.get("measuredLeftMotorPwr")),
            "measuredRightMotorPwr": _trial_manifest_placeholder_normalize(row.get("measuredRightMotorPwr")),
            "curvePair": _trial_manifest_placeholder_normalize(row.get("curvePair")),
            "leftMotorPwr": _trial_manifest_placeholder_normalize(row.get("leftMotorPwr")),
            "rightMotorPwr": _trial_manifest_placeholder_normalize(row.get("rightMotorPwr")),
            "startDist": _trial_manifest_placeholder_normalize(row.get("startDist")),
            "distDelta": _mm_non_negative(row.get("distDelta")),
            "xDelta": _mm_non_negative(row.get("xDelta")),
            "xGapClosed": _mm_non_negative(row.get("xGapClosed")),
            "startConf": _trial_manifest_placeholder_normalize(row.get("startConf")),
            "endConf": _trial_manifest_placeholder_normalize(row.get("endConf")),
            "usable": _trial_manifest_placeholder_normalize(row.get("usable")),
            "status": _trial_manifest_placeholder_normalize(row.get("status")),
            "stopReason": _trial_manifest_placeholder_normalize(row.get("stopReason")),
        }
        forward_rows.append(dict(forward_row))
        backward_rows.append(dict(backward_row))
    return forward_rows, backward_rows


def _trial_manifest_assign_trial_blocks(manifest: dict | None, trial_rows: list[dict] | None) -> dict | None:
    if not isinstance(manifest, dict):
        return manifest
    forward_rows, backward_rows = _trial_manifest_split_trial_blocks(trial_rows)
    manifest["trials_forward"] = list(forward_rows)
    manifest["trials_backwards"] = list(backward_rows)
    manifest.pop("trials", None)
    return manifest


def _trial_manifest_trials_list(manifest: dict | None) -> list[dict]:
    if not isinstance(manifest, dict):
        return []
    forward_rows = [dict(row) for row in list(manifest.get("trials_forward") or []) if isinstance(row, dict)]
    backward_rows = [dict(row) for row in list(manifest.get("trials_backwards") or []) if isinstance(row, dict)]
    if forward_rows or backward_rows:
        return _trial_manifest_merge_trial_blocks(forward_rows, backward_rows)
    return [dict(row) for row in list(manifest.get("trials") or []) if isinstance(row, dict)]


def _trial_manifest_sequence_from_row(row: dict | None) -> list[dict]:
    if not isinstance(row, dict):
        return []
    moves = row.get("moves")
    if isinstance(moves, list):
        return [dict(item) for item in moves if isinstance(item, dict)]
    sequence = []
    for key in ("forward_segment", "backward_segment"):
        segment = row.get(key)
        if isinstance(segment, dict):
            sequence.append(dict(segment))
    return sequence


def _trial_manifest_setup_segment(
    row: dict | None,
    *,
    manifest: dict | None = None,
) -> dict | None:
    for segment in _trial_manifest_sequence_from_row(row):
        if str(segment.get("drive_mode") or "").strip().lower() == "forward":
            return dict(segment)
    curve_cfg = _trial_manifest_curve_config(manifest)
    template = curve_cfg.get("setup_phase") if isinstance(curve_cfg.get("setup_phase"), dict) else None
    duration_val = _coerce_float((row or {}).get("setupDurationMs"), None)
    if duration_val is None:
        duration_val = _coerce_float((row or {}).get("duration"), None)
    if template is not None and duration_val is not None:
        template_used = dict(template)
        motor_pair = dict(template_used.get("motor_pair") or {})
        profile_name = str((row or {}).get("setupProfile") or "").strip()
        if profile_name:
            template_used["profile_name"] = str(profile_name)
        score_val = _coerce_float((row or {}).get("setupScore"), None)
        if score_val is not None:
            template_used["score_pct"] = int(round(float(score_val)))
        pwm_val = _coerce_float((row or {}).get("setupPwm"), None)
        if pwm_val is not None:
            template_used["pwm_override"] = int(round(float(pwm_val)))
        left_motor_val = _coerce_float((row or {}).get("setupLeftMotorPwr"), None)
        right_motor_val = _coerce_float((row or {}).get("setupRightMotorPwr"), None)
        if left_motor_val is not None:
            motor_pair["left_motor_pwm"] = int(round(float(left_motor_val)))
        if right_motor_val is not None:
            motor_pair["right_motor_pwm"] = int(round(float(right_motor_val)))
        if motor_pair:
            template_used["motor_pair"] = dict(motor_pair)
        return _trial_manifest_segment_from_template(
            template_used,
            duration_ms=int(round(float(duration_val))),
        )
    return None


def _trial_manifest_measured_segment(
    row: dict | None,
    *,
    manifest: dict | None = None,
) -> dict | None:
    for segment in _trial_manifest_sequence_from_row(row):
        if str(segment.get("drive_mode") or "").strip().lower() == "backward":
            return dict(segment)
    curve_cfg = _trial_manifest_curve_config(manifest)
    template = curve_cfg.get("measured_phase") if isinstance(curve_cfg.get("measured_phase"), dict) else None
    if template is not None:
        template_used = dict(template)
        motor_pair = dict(template_used.get("motor_pair") or {})
        profile_name = str((row or {}).get("measuredProfile") or "").strip()
        if profile_name:
            template_used["profile_name"] = str(profile_name)
        score_val = _coerce_float((row or {}).get("measuredScore"), None)
        if score_val is not None:
            template_used["score_pct"] = int(round(float(score_val)))
        pwm_val = _coerce_float((row or {}).get("measuredPwm"), None)
        if pwm_val is not None:
            template_used["pwm_override"] = int(round(float(pwm_val)))
        left_motor_val = _coerce_float((row or {}).get("measuredLeftMotorPwr"), None)
        right_motor_val = _coerce_float((row or {}).get("measuredRightMotorPwr"), None)
        if left_motor_val is not None:
            motor_pair["left_motor_pwm"] = int(round(float(left_motor_val)))
        if right_motor_val is not None:
            motor_pair["right_motor_pwm"] = int(round(float(right_motor_val)))
        if motor_pair:
            template_used["motor_pair"] = dict(motor_pair)
        measured_dur_val = _coerce_float((row or {}).get("measuredDurationMs"), None)
        if measured_dur_val is None:
            measured_dur_val = _coerce_float((row or {}).get("measured_duration"), None)
        if measured_dur_val is not None and measured_dur_val > 0:
            template_used["duration_ms"] = int(round(float(measured_dur_val)))
        return _trial_manifest_segment_from_template(template_used)
    sequence = _trial_manifest_sequence_from_row(row)
    return dict(sequence[-1]) if sequence else None


def _trial_manifest_compact_result_values_from_row(row: dict | None) -> dict:
    values = _trial_manifest_compact_results_placeholder()
    if not isinstance(row, dict):
        return values
    if any(key in row for key in values):
        for key in values:
            if key in row:
                if key in {"distDelta", "xDelta", "xGapClosed"}:
                    values[key] = _mm_non_negative(row.get(key))
                else:
                    values[key] = _trial_manifest_placeholder_normalize(row.get(key))
        values["status"] = str(values.get("status") or "planned")
        values["stopReason"] = str(values.get("stopReason") or TRIAL_RESULT_TBD)
        return values

    legacy = row.get("results") if isinstance(row.get("results"), dict) else {}
    if isinstance(legacy, dict) and legacy:
        values["status"] = str(_trial_manifest_placeholder_normalize(legacy.get("status")) or "planned")
        values["startDist"] = _trial_manifest_placeholder_normalize(legacy.get("start_dist_mm", TRIAL_RESULT_TBD))
        values["distDelta"] = _mm_non_negative(legacy.get("dist_change_mm", TRIAL_RESULT_TBD))
        values["xDelta"] = _mm_non_negative(legacy.get("x_axis_change_mm", TRIAL_RESULT_TBD))
        values["xGapClosed"] = _mm_non_negative(legacy.get("x_gap_closed_mm", TRIAL_RESULT_TBD))
        observation_usable = legacy.get("observation_usable")
        values["usable"] = (
            bool(observation_usable)
            if observation_usable in (True, False)
            else _trial_manifest_placeholder_normalize(observation_usable) if observation_usable is not None else TRIAL_RESULT_TBD
        )
        values["stopReason"] = str(_trial_manifest_placeholder_normalize(legacy.get("stop_reason")) or TRIAL_RESULT_TBD)

    pre_pose = row.get("pre_pose") if isinstance(row.get("pre_pose"), dict) else {}
    post_pose = row.get("post_pose") if isinstance(row.get("post_pose"), dict) else {}
    start_dist = _coerce_float(pre_pose.get("dist"), None)
    start_conf = _coerce_float(pre_pose.get("confidence"), None)
    end_conf = _coerce_float(post_pose.get("confidence"), None)
    if start_dist is not None:
        values["startDist"] = _round_triplet(start_dist)
    if start_conf is not None:
        values["startConf"] = _round_triplet(start_conf)
    if end_conf is not None:
        values["endConf"] = _round_triplet(end_conf)
    return values


def _trial_manifest_compact_base_values_from_row(row: dict | None) -> dict:
    if not isinstance(row, dict):
        return {}
    values = {}
    for key in ("setupProfile", "measuredProfile"):
        if key in row:
            value = str(_trial_manifest_placeholder_normalize(row.get(key)) or "").strip()
            if value:
                values[key] = value
    for key in ("setupScore", "setupPwm", "measuredScore", "measuredPwm"):
        if key in row:
            values[key] = _trial_manifest_placeholder_normalize(row.get(key))
    setup_dur = _coerce_float((row or {}).get("setupDurationMs"), None)
    if setup_dur is not None and setup_dur > 0:
        values["setupDurationMs"] = int(round(float(setup_dur)))
    measured_dur = _coerce_float((row or {}).get("measuredDurationMs"), None)
    if measured_dur is None:
        measured_dur = _coerce_float((row or {}).get("measured_duration"), None)
    if measured_dur is not None and measured_dur > 0:
        values["measuredDurationMs"] = int(round(float(measured_dur)))
        values["measured_duration"] = int(round(float(measured_dur)))
    if "curvePair" in row:
        values["curvePair"] = str(_trial_manifest_placeholder_normalize(row.get("curvePair")) or TRIAL_RESULT_TBD)
    for key in (
        "leftMotorPwr",
        "rightMotorPwr",
        "setupLeftMotorPwr",
        "setupRightMotorPwr",
        "measuredLeftMotorPwr",
        "measuredRightMotorPwr",
    ):
        if key in row:
            values[key] = _trial_manifest_placeholder_normalize(row.get(key))
    return values


def _trial_manifest_normalize_row(
    row: dict | None,
    *,
    manifest: dict | None = None,
    trial_index: int,
    keep_results: bool = True,
) -> dict:
    setup_segment = _trial_manifest_setup_segment(row, manifest=manifest)
    measured_segment = _trial_manifest_measured_segment(row, manifest=manifest)
    display_segment = setup_segment if isinstance(setup_segment, dict) else measured_segment
    duration_val = _coerce_float((row or {}).get("setupDurationMs"), None)
    if duration_val is None:
        duration_val = _coerce_float((row or {}).get("duration"), None)
    if duration_val is None and isinstance(display_segment, dict):
        duration_val = _coerce_float(display_segment.get("duration_ms"), None)
    base_values = _trial_manifest_compact_base_values_from_row(row)
    curve_pair = str(
        base_values.get("curvePair")
        or _trial_manifest_curve_pair_text(
            setup_segment,
            measured_segment,
            fallback=_trial_manifest_default_curve_pair_text(manifest),
        )
    )
    setup_profile = str(base_values.get("setupProfile") or "").strip() or None
    setup_score = base_values.get("setupScore")
    setup_pwm = base_values.get("setupPwm")
    measured_profile = str(base_values.get("measuredProfile") or "").strip() or None
    measured_score = base_values.get("measuredScore")
    measured_pwm = base_values.get("measuredPwm")
    measured_duration_from_row = base_values.get("measuredDurationMs")
    if measured_duration_from_row is None:
        measured_duration_from_row = base_values.get("measured_duration")
    left_motor_pwr = base_values.get("leftMotorPwr")
    right_motor_pwr = base_values.get("rightMotorPwr")
    setup_left_motor_pwr = base_values.get("setupLeftMotorPwr")
    setup_right_motor_pwr = base_values.get("setupRightMotorPwr")
    measured_left_motor_pwr = base_values.get("measuredLeftMotorPwr")
    measured_right_motor_pwr = base_values.get("measuredRightMotorPwr")
    if setup_left_motor_pwr is None and isinstance(setup_segment, dict):
        setup_left_motor_pwr = _coerce_float(((setup_segment.get("motor_pair") or {}).get("left_motor_pwm")), None)
    if setup_right_motor_pwr is None and isinstance(setup_segment, dict):
        setup_right_motor_pwr = _coerce_float(((setup_segment.get("motor_pair") or {}).get("right_motor_pwm")), None)
    if measured_left_motor_pwr is None and isinstance(measured_segment, dict):
        measured_left_motor_pwr = _coerce_float(((measured_segment.get("motor_pair") or {}).get("left_motor_pwm")), None)
    if measured_right_motor_pwr is None and isinstance(measured_segment, dict):
        measured_right_motor_pwr = _coerce_float(((measured_segment.get("motor_pair") or {}).get("right_motor_pwm")), None)
    if left_motor_pwr is None and isinstance(display_segment, dict):
        left_motor_pwr = _coerce_float(((display_segment.get("motor_pair") or {}).get("left_motor_pwm")), None)
    if right_motor_pwr is None and isinstance(display_segment, dict):
        right_motor_pwr = _coerce_float(((display_segment.get("motor_pair") or {}).get("right_motor_pwm")), None)
    result_values = (
        _trial_manifest_compact_result_values_from_row(row)
        if bool(keep_results)
        else _trial_manifest_compact_results_placeholder()
    )
    return _trial_manifest_compact_trial_row(
        trial_index=int(trial_index),
        duration_ms=(int(round(float(duration_val))) if duration_val is not None else 0),
        setup_profile=setup_profile,
        setup_score=setup_score,
        setup_pwm=setup_pwm,
        measured_duration_ms=measured_duration_from_row,
        measured_profile=measured_profile,
        measured_score=measured_score,
        measured_pwm=measured_pwm,
        setup_left_motor_pwr=setup_left_motor_pwr,
        setup_right_motor_pwr=setup_right_motor_pwr,
        measured_left_motor_pwr=measured_left_motor_pwr,
        measured_right_motor_pwr=measured_right_motor_pwr,
        curve_pair=str(curve_pair),
        left_motor_pwr=left_motor_pwr,
        right_motor_pwr=right_motor_pwr,
        result_values=result_values,
    )


def _build_turn_drive_plan_from_manifest_segment(
    segment: dict | None,
    *,
    metadata: dict | None = None,
) -> dict | None:
    if not isinstance(segment, dict):
        return None
    try:
        score_val = int(round(float(segment.get("score_pct") or 0)))
        duration_val = int(round(float(segment.get("duration_ms") or 0)))
    except (TypeError, ValueError):
        return None
    if score_val <= 0 or duration_val <= 0:
        return None
    motor_pair = dict(segment.get("motor_pair") or {}) if isinstance(segment.get("motor_pair"), dict) else {}
    left_pwm = _coerce_float(motor_pair.get("left_motor_pwm"), None)
    right_pwm = _coerce_float(motor_pair.get("right_motor_pwm"), None)
    left_action = str(motor_pair.get("left_motor_action") or "").strip().lower()
    right_action = str(motor_pair.get("right_motor_action") or "").strip().lower()
    if (
        left_pwm is not None
        and right_pwm is not None
        and left_action in {"f", "b", "s"}
        and right_action in {"f", "b", "s"}
    ):
        left_pwm_int = max(0, int(round(float(left_pwm))))
        right_pwm_int = max(0, int(round(float(right_pwm))))
        return {
            "cmd": str(segment.get("cmd") or ""),
            "score": int(score_val),
            "duration_ms": int(duration_val),
            "profile_name": str(segment.get("profile_name") or ""),
            "drive_mode": str(segment.get("drive_mode") or ""),
            "inner_pwm": int(min(left_pwm_int, right_pwm_int)),
            "outer_pwm": int(max(left_pwm_int, right_pwm_int)),
            "action_note": str(segment.get("action_note") or "TURN+DRIVE"),
            "actions": [
                {"target": "l", "action": str(left_action), "pwm": int(left_pwm_int)},
                {"target": "r", "action": str(right_action), "pwm": int(right_pwm_int)},
            ],
            "metadata": dict(metadata or {}),
        }
    return build_turn_drive_motion_plan(
        cmd=str(segment.get("cmd") or ""),
        score=int(score_val),
        hold_duration_ms=int(duration_val),
        pwm_override=(
            int(round(float(segment.get("pwm_override"))))
            if _coerce_float(segment.get("pwm_override"), None) is not None
            else None
        ),
        profile_name=str(segment.get("profile_name") or ""),
        drive_mode=str(segment.get("drive_mode") or ""),
        metadata=dict(metadata or {}),
    )


def _trial_manifest_segment_for_drive_mode(
    *,
    drive_mode: str | None,
    setup_segment: dict | None,
    measured_segment: dict | None,
) -> dict | None:
    drive_key = str(drive_mode or "").strip().lower()
    for segment in (setup_segment, measured_segment):
        if (
            isinstance(segment, dict)
            and str(segment.get("drive_mode") or "").strip().lower() == drive_key
        ):
            return dict(segment)
    return None


def build_close_dist_x_axis_one_act_trials_manifest(
    *,
    trials: int,
    phase: str,
    strength: str,
    phase_duration_ms: int,
    score: int = DEFAULT_SCORE,
    sequence=DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE,
    setup_forward_range_ms=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_SETUP_FORWARD_RANGE_MS,
    name: str | None = None,
    turn_selection_policy: str | None = None,
) -> dict:
    phase_spec = _resolve_turn_drive_phase_spec(
        sequence,
        phase=str(phase or ""),
        strength=str(strength or ""),
    )
    strength_used = str((phase_spec or {}).get("strength") or _normalize_turn_drive_strength(strength))
    trial_count = max(1, int(trials))
    try:
        score_used = int(round(float(score)))
    except (TypeError, ValueError):
        score_used = int(DEFAULT_SCORE)
    score_used = max(1, min(100, int(score_used)))
    duration_used_ms = max(1, int(round(float(phase_duration_ms))))
    setup_range_cfg = _duration_range_snapshot(setup_forward_range_ms)
    setup_enabled = bool(
        isinstance(setup_range_cfg, dict)
        and str((phase_spec or {}).get("drive_mode") or "").strip().lower() == "backward"
    )
    turn_selection_policy_used = str(turn_selection_policy or "").strip().lower()
    if not turn_selection_policy_used and bool(setup_enabled):
        turn_selection_policy_used = TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS
    adaptive_turn_selection = turn_selection_policy_used == TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS

    phase_name = str((phase_spec or {}).get("phase") or phase or "")
    phase_label = str((phase_spec or {}).get("label") or _turn_drive_phase_label(
        cmd=str((phase_spec or {}).get("cmd") or ""),
        drive_mode=str((phase_spec or {}).get("drive_mode") or ""),
    ))
    measured_drive_mode = str((phase_spec or {}).get("drive_mode") or "").strip().lower()
    curve_label = (
        _adaptive_turn_label_for_drive_mode(measured_drive_mode)
        if bool(adaptive_turn_selection)
        else str(phase_label)
    )
    curve_pair_label = (
        f"{_adaptive_turn_label_for_drive_mode('forward')} -> "
        f"{_adaptive_turn_label_for_drive_mode(measured_drive_mode)}"
        if bool(adaptive_turn_selection)
        else ""
    )
    manifest = {
        "file_type": "turn_drive_trials",
        "schema_version": 2,
        "name": _trial_file_name_from_manifest(None, fallback=name or phase_name),
        "curve": {
            "trial_format": "compact",
            "sequence_style": (
                "adaptive_distance_hold_then_backward_turn"
                if bool(adaptive_turn_selection) and bool(setup_enabled)
                else "distance_hold_then_backward_turn"
                if bool(setup_enabled)
                else "single_turn_drive"
            ),
            "phase": str(phase_name),
            "label": str(curve_label),
            "turn_strength": str(strength_used),
            "trial_count": int(trial_count),
            "score_pct": int(score_used),
            "measured_phase_duration_ms": int(duration_used_ms),
            "setup_turn_duration_range_ms": (dict(setup_range_cfg) if isinstance(setup_range_cfg, dict) else None),
        },
        "distribution": {},
        "checks": {
            "forward_once_then_backward_once_every_trial": False,
            "turning_whenever_moving_every_trial": False,
            "setup_durations_within_assumptions": False,
            "setup_durations_evenly_increasing": False,
        },
        "run_status": _trial_manifest_run_status_placeholder(trial_count=int(trial_count)),
        "curve_stats": _trial_manifest_curve_stats_placeholder(),
        "trials": [],
    }
    if bool(setup_enabled):
        manifest["curve"]["distance_hold_policy"] = TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
        manifest["curve"]["distance_hold_rule"] = "if dist<run start dist move back; else move forward"
    if bool(adaptive_turn_selection):
        manifest["curve"]["turn_selection_policy"] = str(turn_selection_policy_used)
        manifest["curve"]["turn_selection_rule"] = "if x_axis<0 turn right; else turn left"
        manifest["curve"]["curve_pair_label"] = str(curve_pair_label)
    if not isinstance(phase_spec, dict):
        manifest["invalid_phase"] = True
        return manifest

    measured_template = _build_turn_drive_plan_for_phase_spec(
        phase_spec=phase_spec,
        score=int(score_used),
        duration_ms=int(duration_used_ms),
        metadata={"stage": "measured_template"},
    )
    if not isinstance(measured_template, dict):
        manifest["invalid_plan"] = True
        return manifest

    setup_template = None
    setup_durations_ms = []
    if bool(setup_enabled):
        setup_template = _inverse_turn_drive_plan(
            measured_template,
            metadata={"stage": "setup_template"},
        )
        if isinstance(setup_template, dict):
            setup_durations_ms = _evenly_distributed_durations_ms(
                duration_range_ms=setup_range_cfg,
                count=int(trial_count),
            )
        else:
            manifest["setup_plan_unavailable"] = True
    setup_deltas_ms = [
        int(setup_durations_ms[idx + 1]) - int(setup_durations_ms[idx])
        for idx in range(len(setup_durations_ms) - 1)
    ]
    manifest["distribution"] = {}
    manifest["curve"]["setup_phase"] = _trial_manifest_phase_template_from_segment(
        _trial_manifest_segment(
            plan=setup_template,
            duration_ms=int(setup_durations_ms[0]) if setup_durations_ms else None,
            stage="setup_forward_turn",
        ),
    )
    manifest["curve"]["measured_phase"] = _trial_manifest_phase_template_from_segment(
        _trial_manifest_segment(
            plan=measured_template,
            duration_ms=int(duration_used_ms),
            stage="measured_backward_turn" if bool(setup_enabled) else "measured_turn_drive",
        ),
        include_duration=True,
    )
    if bool(adaptive_turn_selection):
        if isinstance(manifest["curve"].get("setup_phase"), dict):
            manifest["curve"]["setup_phase"]["label"] = _adaptive_turn_label_for_drive_mode("forward")
        if isinstance(manifest["curve"].get("measured_phase"), dict):
            manifest["curve"]["measured_phase"]["label"] = _adaptive_turn_label_for_drive_mode(measured_drive_mode)

    trial_rows = []
    forward_backward_checks = []
    turning_checks = []
    setup_duration_checks = []
    for trial_index in range(1, int(trial_count) + 1):
        forward_segment = None
        if isinstance(setup_template, dict) and setup_durations_ms:
            forward_segment = _trial_manifest_segment(
                plan=setup_template,
                duration_ms=int(setup_durations_ms[int(trial_index) - 1]),
                stage="setup_forward_turn",
            )
        backward_segment = _trial_manifest_segment(
            plan=measured_template,
            duration_ms=int(duration_used_ms),
            stage="measured_backward_turn" if bool(forward_segment) else "measured_turn_drive",
        )
        segments = [segment for segment in (forward_segment, backward_segment) if isinstance(segment, dict)]
        forward_count = sum(1 for segment in segments if str(segment.get("drive_mode") or "") == "forward")
        backward_count = sum(1 for segment in segments if str(segment.get("drive_mode") or "") == "backward")
        turning_count = sum(1 for segment in segments if bool(segment.get("turning")))
        setup_duration_ms = (
            int((forward_segment or {}).get("duration_ms") or 0)
            if isinstance(forward_segment, dict)
            else None
        )
        setup_duration_within_bounds = bool(
            not isinstance(setup_range_cfg, dict)
            or (
                setup_duration_ms is not None
                and int(setup_range_cfg.get("min_ms") or 0) <= int(setup_duration_ms) <= int(setup_range_cfg.get("max_ms") or 0)
            )
        )
        display_segment = forward_segment if isinstance(forward_segment, dict) else backward_segment
        trial_rows.append(
            _trial_manifest_compact_trial_row(
                trial_index=int(trial_index),
                duration_ms=((display_segment or {}).get("duration_ms") if isinstance(display_segment, dict) else None),
                curve_pair=(
                    str(curve_pair_label)
                    if bool(adaptive_turn_selection)
                    else _trial_manifest_curve_pair_text(forward_segment, backward_segment)
                ),
                left_motor_pwr=(
                    TRIAL_RESULT_TBD
                    if bool(adaptive_turn_selection)
                    else (
                        ((display_segment.get("motor_pair") or {}).get("left_motor_pwm"))
                        if isinstance(display_segment, dict)
                        else None
                    )
                ),
                right_motor_pwr=(
                    TRIAL_RESULT_TBD
                    if bool(adaptive_turn_selection)
                    else (
                        ((display_segment.get("motor_pair") or {}).get("right_motor_pwm"))
                        if isinstance(display_segment, dict)
                        else None
                    )
                ),
            )
        )
        forward_backward_checks.append(int(forward_count) == 1 and int(backward_count) == 1)
        turning_checks.append(bool(segments) and int(turning_count) == int(len(segments)))
        setup_duration_checks.append(bool(setup_duration_within_bounds))
    _trial_manifest_assign_trial_blocks(manifest, trial_rows)
    manifest["checks"] = {
        "forward_once_then_backward_once_every_trial": bool(trial_rows) and all(bool(value) for value in forward_backward_checks),
        "turning_whenever_moving_every_trial": bool(trial_rows) and all(bool(value) for value in turning_checks),
        "setup_durations_within_assumptions": (
            all(bool(value) for value in setup_duration_checks)
            if bool(setup_durations_ms)
            else True
        ),
        "setup_durations_evenly_increasing": (
            all(int(setup_durations_ms[idx]) <= int(setup_durations_ms[idx + 1]) for idx in range(len(setup_durations_ms) - 1))
            and (max(setup_deltas_ms) - min(setup_deltas_ms) <= 1 if setup_deltas_ms else True)
        )
        if bool(setup_durations_ms)
        else True,
    }
    return manifest


def _format_trials_manifest_text(manifest: dict | None) -> str:
    payload = dict(manifest or {}) if isinstance(manifest, dict) else manifest
    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2) + "\n"

    lines = ["{"]
    items = list(payload.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        key_text = json.dumps(str(key))
        if key in {"trials", "trials_forward", "trials_backwards"} and isinstance(value, list):
            if not value:
                lines.append(f"  {key_text}: []{comma}")
                continue
            lines.append(f"  {key_text}: [")
            for row_index, row in enumerate(value):
                row_comma = "," if row_index < len(value) - 1 else ""
                lines.append(f"    {json.dumps(row, separators=(', ', ': '))}{row_comma}")
            lines.append(f"  ]{comma}")
            continue

        value_lines = json.dumps(value, indent=2).splitlines()
        if len(value_lines) == 1:
            lines.append(f"  {key_text}: {value_lines[0]}{comma}")
            continue
        lines.append(f"  {key_text}: {value_lines[0]}")
        for inner_line in value_lines[1:-1]:
            lines.append(f"  {inner_line}")
        lines.append(f"  {value_lines[-1]}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write_trials_manifest(trials_path: Path | None, manifest: dict | None) -> str | None:
    if trials_path is None or not isinstance(manifest, dict):
        return None
    try:
        path = Path(trials_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_format_trials_manifest_text(manifest))
        return str(path)
    except OSError:
        return None


def _trial_result_number(value, *, absolute: bool = False):
    number = _coerce_float(value, None)
    if number is None:
        return TRIAL_RESULT_TBD
    if bool(absolute):
        number = abs(float(number))
    return _round_triplet(number)


def _xdelta_non_negative(value):
    number = _coerce_float(value, None)
    if number is None:
        return _trial_manifest_placeholder_normalize(value)
    return _round_triplet(abs(float(number)))


def _mm_non_negative(value):
    return _xdelta_non_negative(value)


def _mm_payload_non_negative(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            key_text = str(key or "")
            if isinstance(item, (dict, list)):
                normalized[key] = _mm_payload_non_negative(item)
                continue
            if isinstance(item, (int, float)) and (
                key_text.endswith("_mm")
                or key_text in {"offset_x", "offset_y", "dist", "x_axis", "y_axis"}
            ):
                normalized[key] = _round_triplet(abs(float(item)))
            else:
                normalized[key] = item
        return normalized
    if isinstance(value, list):
        return [_mm_payload_non_negative(item) for item in value]
    return value


def _normalize_close_dist_trial_manifest(
    manifest: dict | None,
    *,
    fallback_name: str | None = None,
) -> dict | None:
    if not isinstance(manifest, dict):
        return None
    if isinstance(manifest.get("curve"), dict) and int(manifest.get("schema_version") or 0) >= 2:
        normalized = json.loads(json.dumps(manifest))
        normalized["name"] = _trial_file_name_from_manifest(normalized, fallback=fallback_name)
        curve_cfg = _trial_manifest_curve_config(normalized)
        curve_row = dict(normalized.get("curve") or {})
        curve_row.pop("measured_phase_duration_s", None)
        curve_row.pop("setup_turn_duration_range_s", None)
        curve_row.pop("measured_turn_duration_range_s", None)
        for phase_key in ("setup_phase", "measured_phase"):
            phase_row = curve_row.get(phase_key)
            if isinstance(phase_row, dict):
                phase_copy = dict(phase_row)
                phase_copy.pop("duration_s", None)
                curve_row[phase_key] = phase_copy
        sample_trial = (_trial_manifest_trials_list(normalized) or [{}])[0]
        if not isinstance(curve_cfg.get("setup_phase"), dict):
            curve_row["setup_phase"] = _trial_manifest_phase_template_from_segment(
                _trial_manifest_setup_segment(sample_trial, manifest=normalized),
            )
        if not isinstance(curve_cfg.get("measured_phase"), dict):
            curve_row["measured_phase"] = _trial_manifest_phase_template_from_segment(
                _trial_manifest_measured_segment(sample_trial, manifest=normalized),
                include_duration=True,
            )
        if isinstance(curve_row.get("setup_turn_duration_range_ms"), dict):
            curve_row["distance_hold_policy"] = (
                str(curve_row.get("distance_hold_policy") or "").strip().lower()
                or TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
            )
            curve_row["distance_hold_rule"] = (
                str(curve_row.get("distance_hold_rule") or "").strip()
                or "if dist<run start dist move back; else move forward"
            )
            sequence_style = str(curve_row.get("sequence_style") or "").strip().lower()
            if sequence_style in {
                "",
                "adaptive_forward_turn_then_backward_turn",
                "forward_turn_then_backward_turn",
            }:
                curve_row["sequence_style"] = (
                    "adaptive_distance_hold_then_backward_turn"
                    if _trial_manifest_uses_adaptive_turn_selection(normalized)
                    else "distance_hold_then_backward_turn"
                )
        curve_row["trial_format"] = "compact"
        normalized["curve"] = curve_row
        normalized["distribution"] = {}
        normalized["run_status"] = _trial_manifest_merge_defaults(
            _trial_manifest_run_status_placeholder(
                trial_count=int((_trial_manifest_curve_config(normalized).get("trial_count") or len(_trial_manifest_trials_list(normalized)) or 1))
            ),
            normalized.get("run_status"),
        )
        normalized["curve_stats"] = _trial_manifest_merge_defaults(
            _trial_manifest_curve_stats_placeholder(),
            normalized.get("curve_stats"),
        )
        normalized_trials = _trial_manifest_trials_list(normalized)
        normalized_rows = [
            _trial_manifest_normalize_row(
                row,
                manifest=normalized,
                trial_index=int(index),
                keep_results=True,
            )
            for index, row in enumerate(normalized_trials, start=1)
        ]
        _trial_manifest_assign_trial_blocks(normalized, normalized_rows)
        return normalized
    curve_cfg = _trial_manifest_curve_config(manifest)
    normalized = build_close_dist_x_axis_one_act_trials_manifest(
        trials=int(
            len(_trial_manifest_trials_list(manifest))
            or curve_cfg.get("trial_count")
            or DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS
        ),
        phase=str(curve_cfg.get("phase") or DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_PHASE),
        strength=str(curve_cfg.get("turn_strength") or DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH),
        phase_duration_ms=int(
            curve_cfg.get("measured_phase_duration_ms")
            or DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_DURATION_MS
        ),
        score=int(curve_cfg.get("score_pct") or DEFAULT_SCORE),
        setup_forward_range_ms=curve_cfg.get("setup_turn_duration_range_ms"),
        name=_trial_file_name_from_manifest(manifest, fallback=fallback_name),
    )
    return normalized


def _reset_trial_manifest_for_run(manifest: dict | None) -> dict | None:
    if not isinstance(manifest, dict):
        return None
    curve_cfg = _trial_manifest_curve_config(manifest)
    trial_rows = _trial_manifest_trials_list(manifest)
    prepared_rows = []
    for index, row in enumerate(trial_rows, start=1):
        retry_row = _trial_manifest_normalize_row(
            row,
            manifest=manifest,
            trial_index=int(index),
            keep_results=False,
        )
        if _trial_manifest_uses_adaptive_turn_selection(manifest):
            retry_row["curvePair"] = _trial_manifest_default_curve_pair_text(manifest)
            retry_row["leftMotorPwr"] = TRIAL_RESULT_TBD
            retry_row["rightMotorPwr"] = TRIAL_RESULT_TBD
            retry_row["setupLeftMotorPwr"] = TRIAL_RESULT_TBD
            retry_row["setupRightMotorPwr"] = TRIAL_RESULT_TBD
            retry_row["measuredLeftMotorPwr"] = TRIAL_RESULT_TBD
            retry_row["measuredRightMotorPwr"] = TRIAL_RESULT_TBD
        prepared_rows.append(retry_row)
    manifest["run_status"] = _trial_manifest_run_status_placeholder(
        trial_count=int(curve_cfg.get("trial_count") or len(prepared_rows) or 1)
    )
    manifest["run_status"]["status"] = "running"
    manifest["run_status"]["stop_reason"] = "running"
    manifest["run_status"]["trials_completed"] = _trial_manifest_trial_rows_completed_count(prepared_rows)
    manifest["run_status"]["usable_trials"] = int(sum(1 for row in prepared_rows if _trial_manifest_row_is_usable_complete(row)))
    manifest["run_status"]["last_started_ts"] = _round_triplet(time.time())
    manifest["curve_stats"] = _trial_manifest_curve_stats_from_compact_rows(prepared_rows)
    _trial_manifest_assign_trial_blocks(manifest, prepared_rows)
    return manifest


def _trial_manifest_curve_stats_from_compact_rows(
    trial_rows: list[dict] | None,
    *,
    threshold_pct: float = DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT,
) -> dict:
    usable_rows = [
        dict(row)
        for row in list(trial_rows or [])
        if _trial_manifest_row_is_usable_complete(row)
    ]
    if not usable_rows:
        return _trial_manifest_curve_stats_placeholder()

    x_gap_values = [
        float(row["xGapClosed"])
        for row in usable_rows
        if _coerce_float(row.get("xGapClosed"), None) is not None
    ]
    x_delta_values = [
        float(row["xDelta"])
        for row in usable_rows
        if _coerce_float(row.get("xDelta"), None) is not None
    ]
    dist_delta_values = [
        float(row["distDelta"])
        for row in usable_rows
        if _coerce_float(row.get("distDelta"), None) is not None
    ]
    start_conf_values = [
        float(row["startConf"])
        for row in usable_rows
        if _coerce_float(row.get("startConf"), None) is not None
    ]
    end_conf_values = [
        float(row["endConf"])
        for row in usable_rows
        if _coerce_float(row.get("endConf"), None) is not None
    ]
    conf_delta_values = [
        float(row["endConf"]) - float(row["startConf"])
        for row in usable_rows
        if _coerce_float(row.get("startConf"), None) is not None
        and _coerce_float(row.get("endConf"), None) is not None
    ]
    x_gap_summary = _consistency_series_summary(
        x_gap_values,
        sign_eps_mm=float(DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM),
    )
    positive_count = int(((x_gap_summary.get("sign_counts") or {}).get("positive", 0)) or 0)
    close_rate_pct = (
        _round_triplet((float(positive_count) / float(len(usable_rows))) * 100.0)
        if usable_rows
        else None
    )
    threshold_used = max(
        0.0,
        float(_coerce_float(threshold_pct, DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT)),
    )
    return {
        "consistency_score_pct": _trial_result_number(close_rate_pct),
        "x_gap_close_rate_pct": _trial_result_number(close_rate_pct),
        "median_x_axis_change_mm": _trial_result_number(_scalar_series_summary(x_delta_values).get("median")),
        "median_dist_change_mm": _trial_result_number(_scalar_series_summary(dist_delta_values).get("median")),
        "median_dist_traveled_mm": _trial_result_number(
            _consistency_series_summary(
                dist_delta_values,
                sign_eps_mm=float(DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM),
            ).get("abs_median")
        ),
        "median_x_gap_closed_mm": _trial_result_number(_scalar_series_summary(x_gap_values).get("median")),
        "median_start_confidence_pct": _trial_result_number(_scalar_series_summary(start_conf_values).get("median")),
        "median_end_confidence_pct": _trial_result_number(_scalar_series_summary(end_conf_values).get("median")),
        "median_confidence_delta_pct": _trial_result_number(_scalar_series_summary(conf_delta_values).get("median")),
        "usable_trials": int(len(usable_rows)),
        "production_worthy": bool(
            close_rate_pct is not None and float(close_rate_pct) >= float(threshold_used)
        ),
    }


def _trial_results_from_execution_row(
    trial_row: dict | None,
    *,
    stop_reason: str | None = None,
) -> dict:
    if not isinstance(trial_row, dict):
        return _trial_manifest_compact_results_placeholder()
    delta = trial_row.get("delta_mm") if isinstance(trial_row.get("delta_mm"), dict) else {}
    curve_capture = trial_row.get("curve_capture") if isinstance(trial_row.get("curve_capture"), dict) else {}
    axis_delta_mm = None
    recovery = trial_row.get("recovery") if isinstance(trial_row.get("recovery"), dict) else {}
    setup_forward = trial_row.get("setup_forward") if isinstance(trial_row.get("setup_forward"), dict) else {}
    setup_pre_pose = setup_forward.get("pre_pose") if isinstance(setup_forward.get("pre_pose"), dict) else {}
    setup_post_pose = setup_forward.get("post_pose") if isinstance(setup_forward.get("post_pose"), dict) else {}
    setup_pre_x = _coerce_float(setup_pre_pose.get("offset_x"), None)
    setup_post_x = _coerce_float(setup_post_pose.get("offset_x"), None)
    if setup_pre_x is not None and setup_post_x is not None:
        axis_delta_mm = abs(float(setup_post_x) - float(setup_pre_x))
    else:
        fallback_delta_x = _coerce_float(delta.get("x_axis"), None)
        if fallback_delta_x is not None:
            axis_delta_mm = abs(float(fallback_delta_x))
    pre_pose = trial_row.get("pre_pose") if isinstance(trial_row.get("pre_pose"), dict) else {}
    post_pose = trial_row.get("post_pose")
    post_setup_pose = setup_forward.get("post_pose")
    if post_pose is not None:
        status = "completed"
    elif bool(recovery.get("attempted")) and bool(recovery.get("recovered")):
        status = "recovered"
    elif bool(setup_forward.get("attempted")) and post_setup_pose is None:
        status = "lost_after_setup"
    elif str(stop_reason or "").strip():
        status = "stopped"
    else:
        status = "unobserved"
    return {
        "status": str(status),
        "startDist": _trial_result_number(
            curve_capture.get("start_dist_mm", (pre_pose or {}).get("dist"))
        ),
        "distDelta": _trial_result_number(delta.get("dist"), absolute=True),
        "xDelta": _trial_result_number(axis_delta_mm, absolute=True),
        "xGapClosed": _trial_result_number(curve_capture.get("x_gap_closed_mm"), absolute=True),
        "startConf": _trial_result_number(pre_pose.get("confidence")),
        "endConf": _trial_result_number((post_pose or {}).get("confidence")),
        "usable": (
            bool(trial_row.get("observation_usable"))
            if trial_row.get("observation_usable") in (True, False)
            else TRIAL_RESULT_TBD
        ),
        "stopReason": str(stop_reason or "").strip() or TRIAL_RESULT_TBD,
    }


def _trial_manifest_row_updates_from_execution_row(trial_row: dict | None) -> dict:
    if not isinstance(trial_row, dict):
        return {}
    setup_forward = trial_row.get("setup_forward") if isinstance(trial_row.get("setup_forward"), dict) else {}
    measured_segment = {
        "label": str(trial_row.get("selected_label") or trial_row.get("label") or ""),
    }
    curve_pair = _trial_manifest_curve_pair_text(
        (setup_forward if bool(setup_forward.get("attempted")) else None),
        measured_segment,
        fallback=None,
    )
    motor_pair = (
        dict(setup_forward.get("motor_pair") or {})
        if bool(setup_forward.get("attempted"))
        else _turn_drive_motor_pair_summary(trial_row.get("plan"))
    )
    measured_motor_pair = _turn_drive_motor_pair_summary(trial_row.get("plan"))
    setup_duration_ms = _coerce_float(setup_forward.get("duration_requested_ms"), None)
    measured_duration_ms = _coerce_float((trial_row.get("plan") or {}).get("duration_ms"), None)
    return {
        "curvePair": str(curve_pair),
        "leftMotorPwr": _trial_manifest_motor_value((motor_pair or {}).get("left_motor_pwm")),
        "rightMotorPwr": _trial_manifest_motor_value((motor_pair or {}).get("right_motor_pwm")),
        "setupDurationMs": (
            int(round(float(setup_duration_ms)))
            if setup_duration_ms is not None and setup_duration_ms > 0
            else TRIAL_RESULT_TBD
        ),
        "measuredDurationMs": (
            int(round(float(measured_duration_ms)))
            if measured_duration_ms is not None and measured_duration_ms > 0
            else TRIAL_RESULT_TBD
        ),
        "setupLeftMotorPwr": _trial_manifest_motor_value((setup_forward.get("motor_pair") or {}).get("left_motor_pwm")),
        "setupRightMotorPwr": _trial_manifest_motor_value((setup_forward.get("motor_pair") or {}).get("right_motor_pwm")),
        "measuredLeftMotorPwr": _trial_manifest_motor_value((measured_motor_pair or {}).get("left_motor_pwm")),
        "measuredRightMotorPwr": _trial_manifest_motor_value((measured_motor_pair or {}).get("right_motor_pwm")),
    }


def _update_trial_manifest_trial_result(
    manifest: dict | None,
    trial_row: dict | None,
    *,
    stop_reason: str | None = None,
) -> dict | None:
    if not isinstance(manifest, dict) or not isinstance(trial_row, dict):
        return manifest
    trial_index = max(1, int(trial_row.get("trial") or 1))
    trial_rows = _trial_manifest_trials_list(manifest)
    if trial_index > len(trial_rows):
        return manifest
    updated_row = _trial_manifest_normalize_row(
        trial_rows[int(trial_index) - 1],
        manifest=manifest,
        trial_index=int(trial_index),
        keep_results=False,
    )
    updated_row.update(_trial_manifest_row_updates_from_execution_row(trial_row))
    updated_row.update(_trial_results_from_execution_row(
        trial_row,
        stop_reason=stop_reason,
    ))
    trial_rows[int(trial_index) - 1] = updated_row
    _trial_manifest_assign_trial_blocks(manifest, trial_rows)
    run_status = dict(manifest.get("run_status") or {})
    run_status["trials_completed"] = max(
        int(run_status.get("trials_completed") or 0),
        int(trial_index),
    )
    manifest["run_status"] = run_status
    return manifest


def _finalize_trial_manifest_after_run(
    manifest: dict | None,
    result: dict | None,
) -> dict | None:
    if not isinstance(manifest, dict):
        return manifest
    result_dict = dict(result or {})
    curve_assessment = (
        (result_dict.get("curve_assessment") or {})
        if isinstance(result_dict.get("curve_assessment"), dict)
        else {}
    )
    threshold_used = float(
        _coerce_float(
            curve_assessment.get("threshold_pct"),
            DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT,
        )
    )
    run_status = dict(manifest.get("run_status") or {})
    stop_reason = str(result_dict.get("stop_reason") or run_status.get("stop_reason") or "unknown")
    trial_rows = _trial_manifest_trials_list(manifest)
    computed_curve_stats = _trial_manifest_curve_stats_from_compact_rows(
        trial_rows,
        threshold_pct=float(threshold_used),
    )
    completed_total = _trial_manifest_trial_rows_completed_count(trial_rows)
    requested_total = int(
        run_status.get("trials_requested")
        or result_dict.get("trials_requested")
        or len(trial_rows)
        or 1
    )
    run_status["status"] = (
        "completed"
        if str(stop_reason) == "completed" and int(completed_total) >= int(requested_total)
        else "stopped"
    )
    run_status["trials_requested"] = int(
        requested_total
    )
    run_status["trials_completed"] = int(completed_total)
    run_status["stop_reason"] = str(stop_reason)
    run_status["usable_trials"] = int(computed_curve_stats.get("usable_trials") or 0)
    run_status["last_completed_ts"] = _round_triplet(time.time())
    run_status["seconds"] = _trial_result_number(result_dict.get("seconds"))
    manifest["run_status"] = run_status
    manifest["curve_stats"] = dict(computed_curve_stats)
    return manifest


def _run_inverse_turn_drive_recovery(
    *,
    robot,
    world,
    vision,
    prior_plan: dict | None,
    trial_index: int,
    phase_label: str,
    observe_samples: int,
    observe_timeout_s: float,
    relaxed_timeout_s: float,
    reobserve_rounds: int,
    settle_s: float,
    log_fn=None,
) -> tuple[dict, dict | None, dict]:
    logger = log_fn if callable(log_fn) else print
    recovery_row = {
        "attempted": False,
        "recovered": False,
        "reason": "not_attempted",
        "phase": "",
        "label": "",
        "plan": None,
        "motor_pair": {},
        "send_result": {},
        "pose": None,
        "meta": {},
        "summary": "not_attempted",
    }
    recovery_plan = _inverse_turn_drive_plan(
        prior_plan,
        metadata={"trial": int(trial_index), "stage": "inverse_recovery"},
    )
    if not isinstance(recovery_plan, dict):
        recovery_row["reason"] = "inverse_plan_unavailable"
        recovery_row["summary"] = "inverse plan unavailable"
        return recovery_row, None, {}

    recovery_row["attempted"] = True
    recovery_row["phase"] = _turn_drive_phase_label(
        cmd=str(recovery_plan.get("cmd") or ""),
        drive_mode=str(recovery_plan.get("drive_mode") or ""),
    ).lower().replace("+", "_")
    recovery_row["label"] = _turn_drive_phase_label(
        cmd=str(recovery_plan.get("cmd") or ""),
        drive_mode=str(recovery_plan.get("drive_mode") or ""),
    )
    recovery_row["plan"] = _turn_drive_plan_snapshot(recovery_plan)
    recovery_row["motor_pair"] = _turn_drive_motor_pair_summary(recovery_plan)
    logger(
        f"[CLOSE X+DIST] Trial {int(trial_index)} lost pose after {str(phase_label)}. "
        f"Inverse recovery via {str(recovery_row['label'])} "
        f"{str((recovery_row.get('motor_pair') or {}).get('display') or '')} "
        f"for {int(recovery_plan.get('duration_ms') or 0)}ms."
    )
    send_result = send_robot_command(
        robot,
        world,
        EXPERIMENT_STEP,
        str(recovery_plan.get("cmd") or ""),
        speed=0.0,
        speed_score=int(recovery_plan.get("score") or 0),
        duration_override_ms=int(recovery_plan.get("duration_ms") or 0),
        auto_mode=False,
        ease_in_out_enabled=False,
        half_first_turn_pulse=False,
        custom_action_specs=[dict(action) for action in list(recovery_plan.get("actions") or [])],
        action_note=str(recovery_plan.get("action_note") or "").strip() or None,
    )
    recovery_row["send_result"] = dict(send_result or {})
    if not isinstance(send_result, dict):
        recovery_row["reason"] = "inverse_send_failed"
        recovery_row["summary"] = "inverse send failed"
        return recovery_row, None, {}

    effective_duration_ms = int(
        (send_result or {}).get("duration_ms")
        or int(recovery_plan.get("duration_ms") or 0)
    )
    time.sleep(max(float(settle_s), (float(effective_duration_ms) / 1000.0) + 0.05))

    recovered_pose, recovered_meta = observe_alignment_pose(
        vision=vision,
        world=world,
        samples=int(observe_samples),
        timeout_s=float(observe_timeout_s),
        relaxed_timeout_s=float(relaxed_timeout_s),
        reobserve_rounds=int(reobserve_rounds),
        log_fn=logger,
    )
    recovery_row["pose"] = _pose_snapshot(recovered_pose)
    recovery_row["meta"] = dict(recovered_meta or {})
    recovery_row["summary"] = _pose_summary_text(recovered_pose)
    recovery_row["recovered"] = bool(recovered_pose is not None)
    recovery_row["reason"] = "recovered" if bool(recovered_pose is not None) else "still_unavailable"
    return recovery_row, recovered_pose, dict(recovered_meta or {})


def _normalize_turn_drive_sequence(sequence, *, strength: str | None = None) -> list[dict]:
    rows = []
    source = sequence if isinstance(sequence, (list, tuple)) else DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE
    override_strength = None
    if str(strength or "").strip():
        override_strength = _normalize_turn_drive_strength(strength)
    for idx, entry in enumerate(list(source), start=1):
        if not isinstance(entry, dict):
            continue
        cmd = str(entry.get("cmd") or "").strip().lower()
        drive_mode = str(entry.get("drive_mode") or "").strip().lower()
        requested_strength = override_strength or _normalize_turn_drive_strength(entry.get("strength"))
        profile_name = _turn_drive_profile_name_for_strength(
            drive_mode=str(drive_mode),
            strength=str(requested_strength),
            fallback_profile_name=str(entry.get("profile_name") or ""),
        )
        if cmd not in {"l", "r"} or drive_mode not in {"forward", "backward"} or not profile_name:
            continue
        phase = str(entry.get("phase") or f"phase_{int(idx)}").strip().lower()
        label = str(entry.get("label") or phase.replace("_", "+").upper()).strip().upper()
        rows.append(
            {
                "phase": str(phase),
                "label": str(label),
                "cmd": str(cmd),
                "drive_mode": str(drive_mode),
                "profile_name": str(profile_name),
                "strength": str(requested_strength),
            }
        )
    if rows:
        return rows
    return [dict(row) for row in DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE]


def _resolve_turn_drive_phase_spec(
    sequence,
    *,
    phase: str | None = None,
    strength: str | None = None,
) -> dict | None:
    rows = _normalize_turn_drive_sequence(sequence, strength=strength)
    if not rows:
        return None
    requested_phase = str(phase or "").strip().lower()
    if requested_phase:
        for row in rows:
            if str(row.get("phase") or "").strip().lower() == requested_phase:
                return dict(row)
    return dict(rows[0])


def _scalar_series_summary(values) -> dict:
    series = [
        float(value)
        for value in list(values or [])
        if _coerce_float(value, None) is not None
    ]
    if not series:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(series)),
        "mean": _round_triplet(statistics.mean(series)),
        "median": _round_triplet(statistics.median(series)),
        "stdev": _round_triplet(statistics.pstdev(series)) if len(series) > 1 else 0.0,
        "min": _round_triplet(min(series)),
        "max": _round_triplet(max(series)),
    }


def _format_mm_band_edge(value: float) -> str:
    number = float(value)
    if abs(float(number) - round(float(number))) <= 1e-9:
        return str(int(round(float(number))))
    return f"{float(number):.1f}"


def _distance_band_for_value(value, *, band_size_mm: float) -> dict | None:
    number = _coerce_float(value, None)
    band_size = _coerce_float(band_size_mm, None)
    if number is None or band_size is None or float(band_size) <= 0.0:
        return None
    band_index = math.floor(float(number) / float(band_size))
    band_start = float(band_index) * float(band_size)
    band_end = float(band_start) + float(band_size)
    return {
        "start_mm": _round_triplet(band_start),
        "end_mm": _round_triplet(band_end),
        "label": f"{_format_mm_band_edge(band_start)}-{_format_mm_band_edge(band_end)}mm",
    }


def _consistency_series_summary(
    values,
    *,
    sign_eps_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM,
) -> dict:
    series = [
        float(value)
        for value in list(values or [])
        if _coerce_float(value, None) is not None
    ]
    if not series:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "abs_median": None,
            "movement_rate_pct": None,
            "sign_consistency_pct": None,
            "dominant_sign": None,
            "sign_counts": {"positive": 0, "negative": 0, "flat": 0},
        }

    counts = {"positive": 0, "negative": 0, "flat": 0}
    for value in series:
        if float(value) > float(sign_eps_mm):
            counts["positive"] += 1
        elif float(value) < (-1.0 * float(sign_eps_mm)):
            counts["negative"] += 1
        else:
            counts["flat"] += 1

    non_flat_total = int(counts["positive"]) + int(counts["negative"])
    dominant_non_flat = max(int(counts["positive"]), int(counts["negative"]))
    if int(counts["positive"]) > int(counts["negative"]):
        dominant_sign = "positive"
    elif int(counts["negative"]) > int(counts["positive"]):
        dominant_sign = "negative"
    elif non_flat_total > 0:
        dominant_sign = "mixed"
    else:
        dominant_sign = "flat"

    return {
        "count": int(len(series)),
        "mean": _round_triplet(statistics.mean(series)),
        "median": _round_triplet(statistics.median(series)),
        "stdev": _round_triplet(statistics.pstdev(series)) if len(series) > 1 else 0.0,
        "min": _round_triplet(min(series)),
        "max": _round_triplet(max(series)),
        "abs_median": _round_triplet(statistics.median([abs(float(value)) for value in series])),
        "movement_rate_pct": _round_triplet((float(non_flat_total) / float(len(series))) * 100.0),
        "sign_consistency_pct": (
            _round_triplet((float(dominant_non_flat) / float(non_flat_total)) * 100.0)
            if non_flat_total > 0
            else None
        ),
        "dominant_sign": str(dominant_sign),
        "sign_counts": dict(counts),
    }


def _metric_target_gap_abs(value, stats) -> float | None:
    value_num = _coerce_float(value, None)
    target = _coerce_float((stats or {}).get("target"), None) if isinstance(stats, dict) else None
    if value_num is None or target is None:
        return None
    return _round_triplet(abs(float(value_num) - float(target)))


def _turn_drive_motor_pair_summary(plan: dict | None) -> dict:
    actions = list((plan or {}).get("actions") or []) if isinstance(plan, dict) else []
    by_target = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        target = str(action.get("target") or "").strip().lower()
        if target in {"l", "r"}:
            by_target[target] = {
                "action": str(action.get("action") or "").strip().lower(),
                "pwm": int(_coerce_float(action.get("pwm"), 0) or 0),
            }
    left = dict(by_target.get("l") or {"action": "", "pwm": 0})
    right = dict(by_target.get("r") or {"action": "", "pwm": 0})
    return {
        "left_motor_pwm": int(left.get("pwm") or 0),
        "left_motor_action": str(left.get("action") or ""),
        "right_motor_pwm": int(right.get("pwm") or 0),
        "right_motor_action": str(right.get("action") or ""),
        "display": (
            f"leftMotor={int(left.get('pwm') or 0)}({str(left.get('action') or '').upper() or '-'}) "
            f"rightMotor={int(right.get('pwm') or 0)}({str(right.get('action') or '').upper() or '-'})"
        ),
    }


def _alternating_phase_curve_capture(
    *,
    before_pose: dict | None,
    after_pose: dict | None,
    success_gates: dict | None,
) -> dict:
    x_stats = (success_gates or {}).get("xAxis_offset_abs") if isinstance(success_gates, dict) else {}
    start_dist_mm = _coerce_float((before_pose or {}).get("dist"), None)
    end_dist_mm = _coerce_float((after_pose or {}).get("dist"), None)
    start_x_offset_mm = _coerce_float((before_pose or {}).get("offset_x"), None)
    end_x_offset_mm = _coerce_float((after_pose or {}).get("offset_x"), None)
    start_x_gap_mm = _metric_target_gap_abs(start_x_offset_mm, x_stats)
    end_x_gap_mm = _metric_target_gap_abs(end_x_offset_mm, x_stats)
    x_gap_closed_mm = None
    if start_x_gap_mm is not None and end_x_gap_mm is not None:
        x_gap_closed_mm = _round_triplet(abs(float(start_x_gap_mm) - float(end_x_gap_mm)))
    return {
        "start_dist_mm": _mm_non_negative(start_dist_mm),
        "end_dist_mm": _mm_non_negative(end_dist_mm),
        "start_x_offset_mm": _mm_non_negative(start_x_offset_mm),
        "end_x_offset_mm": _mm_non_negative(end_x_offset_mm),
        "x_target_mm": _round_triplet(_coerce_float((x_stats or {}).get("target"), None)),
        "start_x_gap_mm": _mm_non_negative(start_x_gap_mm),
        "end_x_gap_mm": _mm_non_negative(end_x_gap_mm),
        "x_gap_closed_mm": _mm_non_negative(x_gap_closed_mm),
    }


def _summarize_alternating_phase_curve_reference(
    phase_rows: list[dict],
    *,
    sign_eps_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM,
    distance_band_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
) -> dict:
    usable_rows = []
    for row in list(phase_rows or []):
        if not isinstance(row, dict):
            continue
        curve_capture = row.get("curve_capture") if isinstance(row.get("curve_capture"), dict) else {}
        start_dist_mm = _coerce_float(curve_capture.get("start_dist_mm"), None)
        start_x_gap_mm = _coerce_float(curve_capture.get("start_x_gap_mm"), None)
        x_gap_closed_mm = _coerce_float(curve_capture.get("x_gap_closed_mm"), None)
        if start_dist_mm is None or start_x_gap_mm is None or x_gap_closed_mm is None:
            continue
        usable_rows.append(
            {
                "row": row,
                "curve_capture": curve_capture,
                "start_dist_mm": float(start_dist_mm),
                "start_x_gap_mm": float(start_x_gap_mm),
                "x_gap_closed_mm": float(x_gap_closed_mm),
                "x_delta_mm": _coerce_float((row.get("delta_mm") or {}).get("x_axis"), None),
                "dist_delta_mm": _coerce_float((row.get("delta_mm") or {}).get("dist"), None),
            }
        )

    sample_row = usable_rows[0]["row"] if usable_rows else {}
    motor_pair = _turn_drive_motor_pair_summary(
        sample_row.get("plan") if isinstance(sample_row.get("plan"), dict) else None
    )
    overall_close_summary = _consistency_series_summary(
        [item.get("x_gap_closed_mm") for item in usable_rows],
        sign_eps_mm=float(sign_eps_mm),
    )
    overall = {
        "runs": int(len(usable_rows)),
        "start_dist_mm": _scalar_series_summary([item.get("start_dist_mm") for item in usable_rows]),
        "start_x_gap_mm": _scalar_series_summary([item.get("start_x_gap_mm") for item in usable_rows]),
        "x_gap_closed_mm": dict(overall_close_summary),
        "x_gap_close_rate_pct": (
            _round_triplet(
                (
                    float((overall_close_summary.get("sign_counts") or {}).get("positive", 0))
                    / float(len(usable_rows))
                )
                * 100.0
            )
            if usable_rows
            else None
        ),
        "x_delta_mm": _scalar_series_summary([item.get("x_delta_mm") for item in usable_rows]),
        "dist_delta_mm": _scalar_series_summary([item.get("dist_delta_mm") for item in usable_rows]),
    }

    band_rows = []
    groups = {}
    band_size = max(0.5, float(_coerce_float(distance_band_mm, DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM)))
    for item in usable_rows:
        band = _distance_band_for_value(item.get("start_dist_mm"), band_size_mm=float(band_size))
        if not isinstance(band, dict):
            continue
        band_key = str(band.get("label") or "")
        groups.setdefault(band_key, {"band": band, "items": []})
        groups[band_key]["items"].append(item)

    def _band_sort_key(item: dict) -> float:
        return float(_coerce_float(((item.get("dist_band_mm") or {}).get("start_mm")), 0.0) or 0.0)

    for group in groups.values():
        band = dict(group.get("band") or {})
        items = list(group.get("items") or [])
        close_summary = _consistency_series_summary(
            [item.get("x_gap_closed_mm") for item in items],
            sign_eps_mm=float(sign_eps_mm),
        )
        band_rows.append(
            {
                "dist_band_mm": dict(band),
                "runs": int(len(items)),
                "median_start_dist_mm": _scalar_series_summary([item.get("start_dist_mm") for item in items]).get("median"),
                "median_start_x_gap_mm": _scalar_series_summary([item.get("start_x_gap_mm") for item in items]).get("median"),
                "median_x_gap_closed_mm": _scalar_series_summary([item.get("x_gap_closed_mm") for item in items]).get("median"),
                "median_x_delta_mm": _scalar_series_summary([item.get("x_delta_mm") for item in items]).get("median"),
                "median_dist_delta_mm": _scalar_series_summary([item.get("dist_delta_mm") for item in items]).get("median"),
                "x_gap_close_rate_pct": (
                    _round_triplet(
                        (
                            float((close_summary.get("sign_counts") or {}).get("positive", 0))
                            / float(len(items))
                        )
                        * 100.0
                    )
                    if items
                    else None
                ),
            }
        )
    band_rows = sorted(band_rows, key=_band_sort_key)

    lookup_lines = []
    label = str(sample_row.get("label") or sample_row.get("phase") or "").upper()
    strength = str(sample_row.get("strength") or "")
    prefix = f"{label} {strength}".strip()
    for band_row in band_rows:
        band = band_row.get("dist_band_mm") if isinstance(band_row.get("dist_band_mm"), dict) else {}
        median_close = _coerce_float(band_row.get("median_x_gap_closed_mm"), None)
        median_start_gap = _coerce_float(band_row.get("median_start_x_gap_mm"), None)
        consistency_pct = _coerce_float(band_row.get("x_gap_close_rate_pct"), None)
        if median_close is None or median_start_gap is None:
            continue
        lookup_lines.append(
            f"{prefix}: {str(motor_pair.get('display') or '')} at {str(band.get('label') or '')}, "
            f"median x-gap closed {float(median_close):.3f}mm from median start gap "
            f"{float(median_start_gap):.3f}mm over {int(band_row.get('runs') or 0)} runs "
            f"(consistency={_round_triplet(consistency_pct)}%)."
        )

    return {
        "label": str(sample_row.get("label") or ""),
        "phase": str(sample_row.get("phase") or ""),
        "profile_name": str(sample_row.get("profile_name") or ""),
        "strength": str(sample_row.get("strength") or ""),
        "motor_pair": motor_pair,
        "distance_band_mm": _round_triplet(band_size),
        "overall": overall,
        "distance_bands": band_rows,
        "lookup_lines": lookup_lines,
    }


def _summarize_alternating_phase_runs(
    phase_rows: list[dict],
    *,
    sign_eps_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM,
    distance_band_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
) -> dict:
    phase_rows_list = [row for row in list(phase_rows or []) if isinstance(row, dict)]
    x_values = [
        row.get("delta_mm", {}).get("x_axis")
        for row in phase_rows_list
    ]
    y_values = [
        row.get("delta_mm", {}).get("y_axis")
        for row in phase_rows_list
    ]
    dist_values = [
        row.get("delta_mm", {}).get("dist")
        for row in phase_rows_list
    ]
    start_conf_values = [
        _coerce_float(((row.get("pre_pose") or {}).get("confidence")), None)
        for row in phase_rows_list
    ]
    end_conf_values = [
        _coerce_float(((row.get("post_pose") or {}).get("confidence")), None)
        for row in phase_rows_list
    ]
    conf_delta_values = [
        _pose_metric_delta(row.get("pre_pose"), row.get("post_pose"), "confidence")
        for row in phase_rows_list
    ]
    return {
        "runs": int(len(phase_rows_list)),
        "observation_usable_runs": int(
            sum(1 for row in phase_rows_list if bool((row or {}).get("observation_usable")))
        ),
        "lost_pose_runs": int(sum(1 for row in phase_rows_list if not bool((row or {}).get("post_pose")))),
        "x_delta_mm": _consistency_series_summary(x_values, sign_eps_mm=float(sign_eps_mm)),
        "y_delta_mm": _consistency_series_summary(y_values, sign_eps_mm=float(sign_eps_mm)),
        "dist_delta_mm": _consistency_series_summary(dist_values, sign_eps_mm=float(sign_eps_mm)),
        "start_confidence_pct": _scalar_series_summary(start_conf_values),
        "end_confidence_pct": _scalar_series_summary(end_conf_values),
        "confidence_delta_pct": _scalar_series_summary(conf_delta_values),
        "curve_reference": _summarize_alternating_phase_curve_reference(
            phase_rows_list,
            sign_eps_mm=float(sign_eps_mm),
            distance_band_mm=float(distance_band_mm),
        ),
    }


def _summarize_alternating_cycle_nets(
    cycle_rows: list[dict],
    *,
    sign_eps_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_SIGN_EPS_MM,
) -> dict:
    x_values = [
        row.get("net_delta_mm", {}).get("x_axis")
        for row in list(cycle_rows or [])
        if isinstance(row, dict)
    ]
    y_values = [
        row.get("net_delta_mm", {}).get("y_axis")
        for row in list(cycle_rows or [])
        if isinstance(row, dict)
    ]
    dist_values = [
        row.get("net_delta_mm", {}).get("dist")
        for row in list(cycle_rows or [])
        if isinstance(row, dict)
    ]
    return {
        "cycles": int(len(list(cycle_rows or []))),
        "usable_cycles": int(
            sum(1 for row in list(cycle_rows or []) if bool((row or {}).get("observation_usable")))
        ),
        "x_net_delta_mm": _consistency_series_summary(x_values, sign_eps_mm=float(sign_eps_mm)),
        "y_net_delta_mm": _consistency_series_summary(y_values, sign_eps_mm=float(sign_eps_mm)),
        "dist_net_delta_mm": _consistency_series_summary(dist_values, sign_eps_mm=float(sign_eps_mm)),
    }


def _collect_alternating_curve_lookup_lines(phase_summary_by_name: dict) -> list[str]:
    lines = []
    for phase_summary in list((phase_summary_by_name or {}).values()):
        if not isinstance(phase_summary, dict):
            continue
        curve_reference = (
            phase_summary.get("curve_reference") if isinstance(phase_summary.get("curve_reference"), dict) else {}
        )
        for line in list(curve_reference.get("lookup_lines") or []):
            text = str(line or "").strip()
            if text:
                lines.append(text)
    return lines


def _curve_assessment_from_phase_summary(
    phase_summary: dict | None,
    *,
    threshold_pct: float = DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT,
) -> dict:
    curve_reference = (
        phase_summary.get("curve_reference") if isinstance((phase_summary or {}).get("curve_reference"), dict) else {}
    )
    overall = curve_reference.get("overall") if isinstance(curve_reference.get("overall"), dict) else {}
    close_rate_pct = _coerce_float(overall.get("x_gap_close_rate_pct"), None)
    usable_runs = int(overall.get("runs") or 0)
    threshold_used = max(0.0, float(_coerce_float(threshold_pct, DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT)))
    return {
        "metric": "x_gap_close_rate_pct",
        "consistency_pct": _round_triplet(close_rate_pct),
        "threshold_pct": _round_triplet(threshold_used),
        "production_worthy": bool(
            close_rate_pct is not None
            and usable_runs > 0
            and float(close_rate_pct) >= float(threshold_used)
        ),
        "usable_runs": int(usable_runs),
        "motor_pair": (
            dict(curve_reference.get("motor_pair") or {})
            if isinstance(curve_reference.get("motor_pair"), dict)
            else {}
        ),
    }


def _alternating_turn_drive_suggestion(
    *,
    phase_summary_by_name: dict,
    cycle_net_summary: dict,
    rebalance_threshold_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_REBALANCE_MM,
) -> dict:
    weakest_issue = None
    weakest_value = None
    for phase_name, phase_summary in (phase_summary_by_name or {}).items():
        if not isinstance(phase_summary, dict):
            continue
        for metric_key in ("x_delta_mm", "dist_delta_mm"):
            metric_summary = phase_summary.get(metric_key) if isinstance(phase_summary.get(metric_key), dict) else {}
            consistency_pct = _coerce_float(metric_summary.get("sign_consistency_pct"), None)
            movement_rate_pct = _coerce_float(metric_summary.get("movement_rate_pct"), None)
            if movement_rate_pct is None:
                continue
            issue_value = movement_rate_pct if consistency_pct is None else min(movement_rate_pct, consistency_pct)
            if weakest_value is None or float(issue_value) < float(weakest_value):
                weakest_value = float(issue_value)
                weakest_issue = {
                    "phase": str(phase_name),
                    "metric": str(metric_key),
                    "movement_rate_pct": _round_triplet(movement_rate_pct),
                    "sign_consistency_pct": _round_triplet(consistency_pct),
                }

    if weakest_issue is not None and float(weakest_value or 0.0) < 80.0:
        return {
            "action": "RETUNE_SINGLE_FAMILY",
            "reason": "weakest_phase_metric_is_not_repeatable_enough",
            "weakest_phase": weakest_issue.get("phase"),
            "weakest_metric": weakest_issue.get("metric"),
            "movement_rate_pct": weakest_issue.get("movement_rate_pct"),
            "sign_consistency_pct": weakest_issue.get("sign_consistency_pct"),
        }

    x_net_abs = _coerce_float(((cycle_net_summary or {}).get("x_net_delta_mm") or {}).get("abs_median"), None)
    dist_net_abs = _coerce_float(((cycle_net_summary or {}).get("dist_net_delta_mm") or {}).get("abs_median"), None)
    if (
        (x_net_abs is not None and float(x_net_abs) > float(rebalance_threshold_mm))
        or (dist_net_abs is not None and float(dist_net_abs) > float(rebalance_threshold_mm))
    ):
        return {
            "action": "REBALANCE_PHASE_DURATION",
            "reason": "phase_families_are_directional_but_cycle_net_drift_is_still_large",
            "x_net_abs_median_mm": _round_triplet(x_net_abs),
            "dist_net_abs_median_mm": _round_triplet(dist_net_abs),
        }

    return {
        "action": "PROMOTE_TO_CLAIM_WINDOW_PROBE",
        "reason": "phase_families_look_directional_and_cycle_net_drift_is_small",
        "x_net_abs_median_mm": _round_triplet(x_net_abs),
        "dist_net_abs_median_mm": _round_triplet(dist_net_abs),
    }


def _summarize_motion_phase(samples: list[dict]) -> dict:
    if not isinstance(samples, list):
        samples = []
    visible_samples = [row for row in samples if bool((row or {}).get("visible"))]
    confidences = [
        float(row["confidence"])
        for row in visible_samples
        if _coerce_float((row or {}).get("confidence"), None) is not None
    ]
    progress_values = [
        float(row["progress_pct"])
        for row in visible_samples
        if _coerce_float((row or {}).get("progress_pct"), None) is not None
    ]
    x_errors = [
        float((row.get("gate_error_mm") or {}).get("x_axis"))
        for row in visible_samples
        if _coerce_float((row.get("gate_error_mm") or {}).get("x_axis"), None) is not None
    ]
    y_errors = [
        float((row.get("gate_error_mm") or {}).get("y_axis"))
        for row in visible_samples
        if _coerce_float((row.get("gate_error_mm") or {}).get("y_axis"), None) is not None
    ]
    dist_errors = [
        float((row.get("gate_error_mm") or {}).get("dist"))
        for row in visible_samples
        if _coerce_float((row.get("gate_error_mm") or {}).get("dist"), None) is not None
    ]
    visible_rate = None
    if samples:
        visible_rate = float(len(visible_samples)) / float(len(samples))
    return {
        "sample_count": int(len(samples)),
        "visible_sample_count": int(len(visible_samples)),
        "visible_rate": _round_triplet(visible_rate),
        "confidence": _summary_stat(confidences),
        "progress_pct": _summary_stat(progress_values),
        "x_gate_error_mm": _summary_stat(x_errors),
        "y_gate_error_mm": _summary_stat(y_errors),
        "dist_gate_error_mm": _summary_stat(dist_errors),
    }


def _sample_pose_snapshot(sample: dict | None) -> dict | None:
    if not isinstance(sample, dict) or not bool(sample.get("visible")):
        return None
    return {
        "offset_y": _round_triplet(sample.get("offset_y")),
        "offset_x": _round_triplet(sample.get("offset_x")),
        "dist": _round_triplet(sample.get("dist")),
        "angle": _round_triplet(sample.get("angle")),
        "confidence": _round_triplet(sample.get("confidence")),
        "obs_ts": _round_triplet(sample.get("t")),
        "pose_source": str(sample.get("pose_source") or ""),
        "lite_required_frames": None,
        "samples_used": 1,
    }


def _x_zero_sample_row(
    world,
    *,
    trial_index: int,
    stage: str,
    sample_index: int,
    stage_elapsed_s: float,
    run_elapsed_s: float,
    obs_ts: float,
) -> dict:
    pose = _current_world_pose(world, obs_ts=float(obs_ts))
    brick = dict(getattr(world, "brick", {}) or {})
    x_axis_mm = _coerce_float((pose or {}).get("offset_x"), None)
    x_zero_error_mm = abs(float(x_axis_mm)) if x_axis_mm is not None else None
    return {
        "t": _round_triplet(obs_ts),
        "trial": int(trial_index),
        "stage": str(stage),
        "sample_index": int(sample_index),
        "run_elapsed_s": _round_triplet(run_elapsed_s),
        "stage_elapsed_s": _round_triplet(stage_elapsed_s),
        "visible": bool(brick.get("visible")),
        "confidence": _round_triplet((pose or {}).get("confidence")),
        "offset_x": _round_triplet((pose or {}).get("offset_x")),
        "offset_y": _round_triplet((pose or {}).get("offset_y")),
        "dist": _round_triplet((pose or {}).get("dist")),
        "angle": _round_triplet((pose or {}).get("angle")),
        "x_zero_error_mm": _round_triplet(x_zero_error_mm),
        "pose_source": str((pose or {}).get("pose_source") or brick.get("pose_source") or ""),
        "camera_fps_est": _round_triplet(getattr(world, "_camera_fps", None)),
        "vision_backend": str(getattr(world, "_vision_backend", "") or ""),
    }


def _summarize_x_zero_samples(samples: list[dict]) -> dict:
    if not isinstance(samples, list):
        samples = []
    visible_samples = [row for row in samples if bool((row or {}).get("visible"))]
    confidences = [
        float(row["confidence"])
        for row in visible_samples
        if _coerce_float((row or {}).get("confidence"), None) is not None
    ]
    x_values = [
        float(row["offset_x"])
        for row in visible_samples
        if _coerce_float((row or {}).get("offset_x"), None) is not None
    ]
    x_zero_errors = [
        float(row["x_zero_error_mm"])
        for row in visible_samples
        if _coerce_float((row or {}).get("x_zero_error_mm"), None) is not None
    ]
    visible_rate = None
    if samples:
        visible_rate = float(len(visible_samples)) / float(len(samples))
    return {
        "sample_count": int(len(samples)),
        "visible_sample_count": int(len(visible_samples)),
        "visible_rate": _round_triplet(visible_rate),
        "confidence": _summary_stat(confidences),
        "x_axis_mm": {
            "start": _round_triplet(x_values[0]) if x_values else None,
            "end": _round_triplet(x_values[-1]) if x_values else None,
            "min": _round_triplet(min(x_values)) if x_values else None,
            "max": _round_triplet(max(x_values)) if x_values else None,
            "median": _round_triplet(statistics.median(x_values)) if x_values else None,
        },
        "x_zero_error_mm": _summary_stat(x_zero_errors),
        "best_visible_abs_x_mm": _round_triplet(min(x_zero_errors)) if x_zero_errors else None,
    }


def _run_turn_stage(
    *,
    robot,
    world,
    vision,
    trial_index: int,
    stage: str,
    cmd: str,
    turn_intensity_pct: float,
    timeout_s: float,
    sample_hz: float,
    run_started_monotonic: float,
    stop_evaluator=None,
):
    samples = []
    state = {}
    command_duration_ms = max(1, int(round(max(0.01, float(timeout_s)) * 1000.0)))
    max_act_duration_ms = max(1, int(DEFAULT_X_ZERO_MAX_ACT_DURATION_MS))
    turn_speed_score = None
    segment_base_duration_ms = int(max_act_duration_ms)
    try:
        turn_setting = float(turn_intensity_pct)
        turn_setting_rounded = int(round(turn_setting))
        if abs(turn_setting - float(turn_setting_rounded)) <= 1e-6:
            turn_speed_score = max(
                int(telemetry_robot_module.SPEED_SCORE_MIN),
                min(int(telemetry_robot_module.SPEED_SCORE_MAX), int(turn_setting_rounded)),
            )
    except (TypeError, ValueError):
        turn_speed_score = None
    if turn_speed_score is not None:
        try:
            _power, _pwm, _score_used, modeled_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                str(cmd),
                int(turn_speed_score),
            )
            modeled_duration_ms = int(round(float(modeled_duration_ms)))
        except (TypeError, ValueError):
            modeled_duration_ms = 0
        if modeled_duration_ms > 0:
            segment_base_duration_ms = max(1, min(int(max_act_duration_ms), int(modeled_duration_ms)))
    stage_started = time.monotonic()
    stage_deadline = float(stage_started) + max(0.01, float(command_duration_ms) / 1000.0)
    sample_period_s = 1.0 / max(1.0, float(sample_hz))
    next_sample_time = float(stage_started)
    stop_reason = "duration_complete"
    send_results = []
    send_result = None
    pwm_override_active = None
    power_override_active = None

    def _send_segment(duration_ms: int):
        if pwm_override_active is not None and power_override_active is not None:
            return send_robot_command_pwm(
                robot,
                world,
                EXPERIMENT_STEP,
                str(cmd),
                float(power_override_active),
                int(pwm_override_active),
                int(duration_ms),
                speed_score=(int(turn_speed_score) if turn_speed_score is not None else None),
                auto_mode=False,
                half_first_turn_pulse=False,
                ease_in_out_enabled=False,
                respect_requested_duration=True,
            )
        send_kwargs = {
            "speed": 0.0,
            "duration_override_ms": int(duration_ms),
            "auto_mode": False,
            "ease_in_out_enabled": False,
            "half_first_turn_pulse": False,
        }
        if turn_speed_score is not None:
            send_kwargs["speed_score"] = int(turn_speed_score)
        else:
            send_kwargs["turn_intensity"] = float(turn_intensity_pct)
        return send_robot_command(
            robot,
            world,
            EXPERIMENT_STEP,
            str(cmd),
            **send_kwargs,
        )

    try:
        while True:
            now = time.monotonic()
            if not send_results or now >= float(segment_deadline):
                remaining_ms = int(round(max(0.0, float(stage_deadline) - float(now)) * 1000.0))
                if remaining_ms <= 0:
                    break
                segment_duration_ms = max(1, min(int(segment_base_duration_ms), int(remaining_ms)))
                current_send = _send_segment(segment_duration_ms)
                if not isinstance(current_send, dict):
                    send_result = dict(current_send or {})
                    return {
                        "ok": False,
                        "stop_reason": "send_failed",
                        "seconds": 0.0,
                        "send_result": send_result,
                        "send_results": list(send_results),
                        "samples": samples,
                        "sample_summary": _summarize_x_zero_samples(samples),
                        "state": state,
                        "end_pose": None,
                        "end_visible": bool((getattr(world, "brick", {}) or {}).get("visible")),
                    }
                segment_row = dict(current_send or {})
                segment_row["duration_requested_ms"] = int(segment_duration_ms)
                send_results.append(segment_row)
                if send_result is None:
                    send_result = dict(segment_row)
                effective_segment_ms = int(segment_row.get("duration_ms") or segment_duration_ms)
                segment_deadline = float(now) + max(0.01, float(effective_segment_ms) / 1000.0)
            if now >= float(next_sample_time) or not samples:
                update_world_from_vision(world, vision, log=False)
                obs_ts = time.time()
                sample = _x_zero_sample_row(
                    world,
                    trial_index=int(trial_index),
                    stage=str(stage),
                    sample_index=int(len(samples) + 1),
                    stage_elapsed_s=max(0.0, float(now) - float(stage_started)),
                    run_elapsed_s=max(0.0, float(now) - float(run_started_monotonic)),
                    obs_ts=float(obs_ts),
                )
                samples.append(sample)
                if callable(stop_evaluator):
                    verdict = stop_evaluator(sample, state, samples)
                    if isinstance(verdict, dict) and bool(verdict.get("stop")):
                        stop_reason = str(verdict.get("reason") or "condition_met")
                        break

                # On first visibility reacquire during right-turn stage, reduce
                # commanded drive immediately by 1 PWM and 0.01 power.
                if str(stage) == "right_turn_to_x_zero" and not bool(state.get("visibility_speed_reduced")):
                    if bool(state.get("visibility_reacquired")):
                        latest_send = send_results[-1] if send_results else None
                        if isinstance(latest_send, dict):
                            try:
                                base_pwm = int(round(float(latest_send.get("pwm") or 0)))
                            except (TypeError, ValueError):
                                base_pwm = 0
                            try:
                                base_power = float(latest_send.get("power") or 0.0)
                            except (TypeError, ValueError):
                                base_power = 0.0
                            if base_pwm > 0:
                                pwm_override_active = int(max(1, base_pwm - 1))
                                power_override_active = float(max(0.0, base_power - 0.01))
                                state["visibility_speed_reduced"] = True
                                state["visibility_reduced_from"] = {
                                    "pwm": int(base_pwm),
                                    "power": float(base_power),
                                }
                                state["visibility_reduced_to"] = {
                                    "pwm": int(pwm_override_active),
                                    "power": float(power_override_active),
                                }
                        state["visibility_reacquired"] = False
                next_sample_time += float(sample_period_s)
            if now >= float(stage_deadline):
                break
            sleep_until = min(float(next_sample_time), float(segment_deadline), float(stage_deadline))
            delay_s = max(0.0, float(sleep_until) - float(time.monotonic()))
            if delay_s > 0.0:
                time.sleep(delay_s)
    finally:
        stop_robot = getattr(robot, "stop", None)
        if callable(stop_robot):
            stop_robot()
    if float(DEFAULT_X_ZERO_STAGE_SETTLE_S) > 0.0:
        time.sleep(float(DEFAULT_X_ZERO_STAGE_SETTLE_S))
    update_world_from_vision(world, vision, log=False)
    end_pose = _current_world_pose(world, obs_ts=time.time())
    return {
        "ok": True,
        "stop_reason": str(stop_reason),
        "seconds": _round_triplet(max(0.0, time.monotonic() - float(stage_started))),
        "send_result": dict(send_result or {}),
        "send_results": list(send_results),
        "samples": samples,
        "sample_summary": _summarize_x_zero_samples(samples),
        "state": dict(state),
        "end_pose": _pose_snapshot(end_pose),
        "end_visible": bool((getattr(world, "brick", {}) or {}).get("visible")),
    }


def _reset_until_left_edge_evaluator(
    sample,
    state,
    _samples,
    *,
    edge_x_mm: float = DEFAULT_X_ZERO_RESET_EDGE_X_MM,
    plateau_samples: int = DEFAULT_X_ZERO_RESET_PLATEAU_SAMPLES,
    plateau_span_mm: float = DEFAULT_X_ZERO_RESET_PLATEAU_SPAN_MM,
):
    if not bool((sample or {}).get("visible")):
        return {"stop": True, "reason": "lost_visibility"}

    x_axis_mm = _coerce_float((sample or {}).get("offset_x"), None)
    if x_axis_mm is None:
        return None

    history = list(state.get("recent_visible_x_mm") or [])
    history.append(float(x_axis_mm))
    history = history[-int(max(1, int(plateau_samples))):]
    state["recent_visible_x_mm"] = history

    if float(x_axis_mm) > float(edge_x_mm):
        return None
    if len(history) < int(max(1, int(plateau_samples))):
        return None

    span_mm = float(max(history) - min(history))
    state["plateau_span_mm"] = float(span_mm)
    if float(span_mm) <= float(max(0.0, float(plateau_span_mm))):
        state["left_edge_plateau_x_mm"] = float(x_axis_mm)
        return {"stop": True, "reason": "left_edge_plateau"}
    return None


def _right_turn_to_zero_evaluator(*, zero_band_mm: float):
    zero_band_used = max(0.0, float(_coerce_float(zero_band_mm, DEFAULT_X_ZERO_BAND_MM) or DEFAULT_X_ZERO_BAND_MM))

    def _evaluate(sample, state, _samples):
        visible = bool((sample or {}).get("visible"))
        x_axis_mm = _coerce_float((sample or {}).get("offset_x"), None)
        if not visible or x_axis_mm is None:
            if bool(state.get("ever_visible")):
                invisible_streak = int(state.get("invisible_streak_after_reacquire") or 0) + 1
                state["invisible_streak_after_reacquire"] = int(invisible_streak)
                if int(invisible_streak) >= 2:
                    return {"stop": True, "reason": "visibility_lost_two_consecutive_frames"}
            return None

        state["invisible_streak_after_reacquire"] = 0

        abs_x = abs(float(x_axis_mm))
        if state.get("first_visible_pose") is None:
            state["first_visible_pose"] = _sample_pose_snapshot(sample)
            state["first_visible_sample_index"] = int((sample or {}).get("sample_index") or 0)
            state["first_visible_x_mm"] = _round_triplet(x_axis_mm)
            state["visibility_reacquired"] = True
        best_abs_x = _coerce_float(state.get("best_visible_abs_x_mm"), None)
        if best_abs_x is None or float(abs_x) < float(best_abs_x):
            state["best_visible_abs_x_mm"] = float(abs_x)
            state["best_visible_pose"] = _sample_pose_snapshot(sample)

        prev_visible_x = _coerce_float(state.get("last_visible_x_mm"), None)
        state["last_visible_x_mm"] = float(x_axis_mm)
        state["ever_visible"] = True

        if float(abs_x) <= float(zero_band_used):
            state["success_type"] = "zero_band"
            return {"stop": True, "reason": "x_axis_zero_band"}
        if prev_visible_x is not None and (
            (float(prev_visible_x) < 0.0 <= float(x_axis_mm))
            or (float(prev_visible_x) > 0.0 >= float(x_axis_mm))
        ):
            state["success_type"] = "zero_cross"
            return {"stop": True, "reason": "x_axis_zero_cross"}
        if prev_visible_x is None and float(x_axis_mm) > float(zero_band_used):
            return {"stop": True, "reason": "first_visible_positive_overshoot"}
        return None

    return _evaluate


def _summarize_single_goal_trials(trials: list[dict]) -> tuple[dict, dict]:
    if not isinstance(trials, list):
        trials = []
    attempts = [
        row.get("attempt")
        for row in trials
        if isinstance(row, dict) and isinstance(row.get("attempt"), dict)
    ]
    started = [row for row in attempts if str(row.get("status") or "") != "not_started_reset_failed"]
    successes = [
        row
        for row in started
        if str(row.get("status") or "") in {"x_axis_zero_band", "x_axis_zero_cross"}
    ]
    best_vals = [
        float(row["best_visible_abs_x_mm"])
        for row in started
        if _coerce_float(row.get("best_visible_abs_x_mm"), None) is not None
    ]
    status_counts = {}
    for row in started:
        status_key = str(row.get("status") or "unknown")
        status_counts[status_key] = int(status_counts.get(status_key, 0)) + 1
    closest_trial = None
    closest_value = None
    for row in started:
        value = _coerce_float(row.get("best_visible_abs_x_mm"), None)
        if value is None:
            continue
        if closest_value is None or float(value) < float(closest_value):
            closest_value = float(value)
            closest_trial = int(row.get("trial") or 0)

    attempts_started = len(started)
    success_count = len(successes)
    success_rate = None
    if attempts_started > 0:
        success_rate = float(success_count) / float(attempts_started)
    analysis = {
        "trials_requested": int(len(trials)),
        "attempts_started": int(attempts_started),
        "success_count": int(success_count),
        "success_rate": _round_triplet(success_rate),
        "closest_trial": int(closest_trial) if closest_trial is not None else None,
        "best_visible_abs_x_mm": _round_triplet(min(best_vals)) if best_vals else None,
        "median_best_visible_abs_x_mm": _round_triplet(statistics.median(best_vals)) if best_vals else None,
        "status_counts": status_counts,
    }
    if success_count > 0:
        suggestion = {
            "action": "keep_right_turn_stop_on_zero_cross",
            "reason": "at_least_one_trial_crossed_x_zero",
            "highlight_metric": "offset_x",
            "closest_visible_abs_x_mm": analysis.get("best_visible_abs_x_mm"),
        }
    elif int(status_counts.get("first_visible_positive_overshoot", 0)) > 0:
        suggestion = {
            "action": "increase_sampling_or_reduce_reset_depth",
            "reason": "reacquired_on_positive_side_before_zero_stop",
            "highlight_metric": "offset_x",
            "closest_visible_abs_x_mm": analysis.get("best_visible_abs_x_mm"),
        }
    elif int(status_counts.get("never_reacquired", 0)) > 0:
        suggestion = {
            "action": "increase_right_turn_time_or_start_less_hidden",
            "reason": "right_turn_only_attempts_failed_to_reacquire_visibility",
            "highlight_metric": "visible",
            "closest_visible_abs_x_mm": analysis.get("best_visible_abs_x_mm"),
        }
    else:
        suggestion = {
            "action": "keep_right_turn_and_tighten_zero_stop",
            "reason": "closest_visible_x_still_far_from_zero",
            "highlight_metric": "offset_x",
            "closest_visible_abs_x_mm": analysis.get("best_visible_abs_x_mm"),
        }
    return analysis, suggestion

def _build_experiment_vision(*, mode=None, yolo_model_path=None):
    mode_used = normalize_vision_mode(mode, fallback=active_vision_mode())
    try:
        from a_MAIN import (
            _DEFAULT_CYAN_PROFILE,
            _DEFAULT_CYAN_VISIBILITY,
            _apply_cyan_profile,
            _apply_cyan_smoothing_experiment,
            _apply_cyan_visibility,
            build_vision,
        )
    except Exception as exc:
        raise RuntimeError(
            "experiment vision must use the shared livestream vision builder"
        ) from exc

    vision = build_vision(mode_used, yolo_model_path=yolo_model_path)
    if mode_used == VISION_MODE_CYAN:
        app_state_stub = SimpleNamespace(stream_state={})
        profile_key, profile_settings, profile_applied = _apply_cyan_profile(
            app_state_stub,
            vision,
            _DEFAULT_CYAN_PROFILE,
        )
        if bool(profile_applied):
            _apply_cyan_smoothing_experiment(vision, profile_settings)
        _apply_cyan_visibility(
            app_state_stub,
            vision,
            _DEFAULT_CYAN_VISIBILITY,
        )
    return vision, mode_used


def _close_vision(vision) -> None:
    for attr in ("close", "release"):
        closer = getattr(vision, attr, None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
            return


def _read_pose_once(
    vision,
    world,
    *,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None = None,
    min_samples_required: int | None = None,
    on_vision_update=None,
):
    return read_pose(
        vision,
        world,
        samples=int(samples),
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        min_samples_required=min_samples_required,
        observe_sleep_s=float(DEFAULT_OBSERVE_SLEEP_S),
        fallback_step_label=str(EXPERIMENT_STEP.value),
        update_world_from_vision=update_world_from_vision,
        latest_unique_smoothed_frames=_latest_unique_smoothed_frames,
        average_smoothed_frames=_average_smoothed_frames,
        lite_gate_unique_frames=lite_gate_unique_frames,
        min_lite_unique_frames=3,
        on_vision_update=on_vision_update,
    )


def observe_alignment_pose(
    *,
    vision,
    world,
    samples: int = DEFAULT_OBSERVE_SAMPLES,
    timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    reobserve_rounds: int = DEFAULT_REOBSERVE_ROUNDS,
    min_sample_time: float | None = None,
    log_fn=None,
):
    logger = log_fn if callable(log_fn) else (lambda *_args, **_kwargs: None)
    pose, meta = observe_pose_with_reobserve(
        read_pose_fn=_read_pose_once,
        log_fn=logger,
        log_prefix="[RANDOM EXPERIMENT]",
        vision=vision,
        world=world,
        samples=int(samples),
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        hold_s=float(DEFAULT_SETTLE_S),
        reobserve_rounds=int(reobserve_rounds),
        relaxed_timeout_s=float(relaxed_timeout_s),
    )
    if isinstance(pose, dict):
        return pose, dict(meta or {})

    meta_out = dict(meta or {})
    diag = meta_out.get("diagnostic") if isinstance(meta_out.get("diagnostic"), dict) else {}
    raw_visible_hits = int(diag.get("raw_visible_hits") or 0)
    lite_pose_hits = int(diag.get("lite_pose_hits") or 0)
    fallback_pose = _current_world_pose(world, obs_ts=time.time())
    fallback_visible = bool((getattr(world, "brick", {}) or {}).get("visible"))
    latest_frame_ts = _latest_smoothed_frame_ts(world)
    fallback_has_pose = (
        isinstance(fallback_pose, dict)
        and _coerce_float(fallback_pose.get("offset_x"), None) is not None
        and _coerce_float(fallback_pose.get("dist"), None) is not None
    )
    if bool(fallback_visible) and bool(fallback_has_pose) and (
        int(raw_visible_hits) > 0 or int(lite_pose_hits) > 0
    ) and (
        min_sample_time is None
        or (latest_frame_ts is not None and float(latest_frame_ts) >= float(min_sample_time))
    ):
        logger(
            "[RANDOM EXPERIMENT] Observation fallback: accepting latest visible world pose "
            "after strict multi-sample observe did not converge."
        )
        meta_out["mode"] = "world_visible_fallback"
        meta_out["reobserved"] = True
        meta_out["fallback_reason"] = "strict_multisample_unavailable_but_world_visible"
        return fallback_pose, meta_out
    return pose, meta_out


def _brick_state_from_pose(pose: dict | None) -> dict:
    if not isinstance(pose, dict):
        return {
            "visible": False,
            "dist": 0.0,
            "angle": 0.0,
            "x_axis": 0.0,
            "offset_x": 0.0,
            "y_axis": 0.0,
            "offset_y": 0.0,
            "confidence": 0.0,
        }
    offset_x = _coerce_float(pose.get("offset_x"), 0.0) or 0.0
    offset_y = _coerce_float(pose.get("offset_y"), 0.0) or 0.0
    dist = _coerce_float(pose.get("dist"), 0.0) or 0.0
    angle = _coerce_float(pose.get("angle"), 0.0) or 0.0
    confidence = _coerce_float(pose.get("confidence"), 0.0) or 0.0
    return {
        "visible": True,
        "dist": float(dist),
        "angle": float(angle),
        "x_axis": float(offset_x),
        "offset_x": float(offset_x),
        "y_axis": float(offset_y),
        "offset_y": float(offset_y),
        "confidence": float(confidence),
    }


def _analytics_for_pose(world, pose: dict | None, *, step=EXPERIMENT_STEP) -> dict:
    original_brick = dict(getattr(world, "brick", {}) or {})
    original_visibility_lost_frames = getattr(world, "_visibility_lost_frames", 0)
    original_last_visible_time = getattr(world, "last_visible_time", None)
    original_last_seen_x_axis = getattr(world, "last_seen_x_axis", None)
    original_last_seen_angle = getattr(world, "last_seen_angle", None)
    original_last_seen_dist = getattr(world, "last_seen_dist", None)
    brick_state = _brick_state_from_pose(pose)
    try:
        world.brick = dict(brick_state)
        if bool(brick_state.get("visible")):
            obs_ts = _coerce_float((pose or {}).get("obs_ts"), time.time())
            world._visibility_lost_frames = 0
            world.last_visible_time = float(obs_ts or time.time())
            world.last_seen_x_axis = float(brick_state.get("x_axis", 0.0) or 0.0)
            world.last_seen_angle = float(brick_state.get("angle", 0.0) or 0.0)
            world.last_seen_dist = float(brick_state.get("dist", 0.0) or 0.0)
        else:
            world._visibility_lost_frames = max(1, int(original_visibility_lost_frames or 0))
        return dict(
            compute_alignment_analytics(
                world,
                world.process_rules,
                world.learned_rules,
                step,
                duration_s=0.05,
            )
            or {}
        )
    finally:
        world.brick = original_brick
        world._visibility_lost_frames = original_visibility_lost_frames
        world.last_visible_time = original_last_visible_time
        world.last_seen_x_axis = original_last_seen_x_axis
        world.last_seen_angle = original_last_seen_angle
        world.last_seen_dist = original_last_seen_dist


def _step_success_gates(world, step=EXPERIMENT_STEP) -> dict:
    step_key = str(getattr(step, "name", step) or "").strip().upper()
    step_cfg = (world.process_rules or {}).get(step_key)
    if not isinstance(step_cfg, dict):
        return {}
    gates = step_cfg.get("success_gates")
    return dict(gates or {}) if isinstance(gates, dict) else {}


def _metric_gate_error(value, stats) -> float | None:
    value_num = _coerce_float(value, None)
    if value_num is None or not isinstance(stats, dict):
        return None
    target = _coerce_float(stats.get("target"), None)
    tol = _coerce_float(stats.get("tol"), None)
    min_val = _coerce_float(stats.get("min"), None)
    max_val = _coerce_float(stats.get("max"), None)
    if target is not None and tol is not None:
        return max(0.0, abs(float(value_num) - float(target)) - abs(float(tol)))
    if min_val is not None and float(value_num) < float(min_val):
        return float(min_val) - float(value_num)
    if max_val is not None and float(value_num) > float(max_val):
        return float(value_num) - float(max_val)
    return 0.0


def _secondary_bplus_cushion_mm(stats) -> float:
    tol = _coerce_float((stats or {}).get("tol"), None) if isinstance(stats, dict) else None
    if tol is None:
        return float(DEFAULT_SECONDARY_BPLUS_CUSHION_MM)
    return float(max(float(DEFAULT_SECONDARY_BPLUS_CUSHION_MM), abs(float(tol)) * 0.25))


def _secondary_focus_status(success_gates: dict, errors: dict | None) -> dict:
    y_stats = success_gates.get("yAxis_offset_abs") if isinstance(success_gates, dict) else {}
    dist_stats = success_gates.get("dist") if isinstance(success_gates, dict) else {}
    y_cushion = _secondary_bplus_cushion_mm(y_stats)
    dist_cushion = _secondary_bplus_cushion_mm(dist_stats)
    y_error = _coerce_float((errors or {}).get("y_axis"), None)
    dist_error = _coerce_float((errors or {}).get("dist"), None)
    y_bplus_ok = (y_error is not None) and float(y_error) <= float(y_cushion)
    dist_bplus_ok = (dist_error is not None) and float(dist_error) <= float(dist_cushion)
    return {
        "y_bplus_cushion_mm": _round_triplet(y_cushion),
        "dist_bplus_cushion_mm": _round_triplet(dist_cushion),
        "y_bplus_ok": bool(y_bplus_ok),
        "dist_bplus_ok": bool(dist_bplus_ok),
        "secondary_focus_ready": bool(y_bplus_ok and dist_bplus_ok),
    }


def _x_turn_breakout_intensity_pct(
    x_gate_error_mm: float | None,
    *,
    max_turn_intensity_pct: float,
) -> float:
    x_error = abs(_coerce_float(x_gate_error_mm, 0.0) or 0.0)
    if x_error <= 4.0:
        target = float(DEFAULT_X_TURN_BREAKOUT_MIN_INTENSITY_PCT)
    elif x_error <= 8.0:
        target = 3.0
    else:
        target = float(DEFAULT_X_TURN_BREAKOUT_MAX_INTENSITY_PCT)
    return float(
        _clamp(
            target,
            DEFAULT_X_TURN_BREAKOUT_MIN_INTENSITY_PCT,
            min(float(DEFAULT_X_TURN_BREAKOUT_MAX_INTENSITY_PCT), float(max_turn_intensity_pct)),
        )
    )


def _dist_recovery_cmd(pose: dict | None, dist_stats: dict | None) -> str | None:
    if not isinstance(pose, dict) or not isinstance(dist_stats, dict):
        return None
    dist_value = _coerce_float(pose.get("dist"), None)
    if dist_value is None:
        return None
    target = _coerce_float(dist_stats.get("target"), None)
    min_val = _coerce_float(dist_stats.get("min"), None)
    max_val = _coerce_float(dist_stats.get("max"), None)
    if target is not None:
        if float(dist_value) > float(target):
            return "f"
        if float(dist_value) < float(target):
            return "b"
    if max_val is not None and float(dist_value) > float(max_val):
        return "f"
    if min_val is not None and float(dist_value) < float(min_val):
        return "b"
    return None


def _micro_drive_score_for_dist_error(
    dist_error_mm: float | None,
    *,
    analytics_speed_score_pct: float | None,
    max_drive_score: float,
) -> int:
    dist_err = abs(_coerce_float(dist_error_mm, 0.0) or 0.0)
    if dist_err <= 1.0:
        base_score = 1
    elif dist_err <= 2.0:
        base_score = 2
    elif dist_err <= 4.0:
        base_score = 3
    else:
        base_score = 4
    if analytics_speed_score_pct is not None:
        base_score = min(int(base_score), int(round(float(analytics_speed_score_pct))))
    return int(
        max(
            int(telemetry_robot_module.SPEED_SCORE_MIN),
            min(int(round(float(max_drive_score))), int(base_score)),
        )
    )


def _pose_gate_errors(pose: dict | None, success_gates: dict) -> dict | None:
    if not isinstance(pose, dict):
        return None
    return {
        "x_axis": _round_triplet(
            _metric_gate_error(pose.get("offset_x"), success_gates.get("xAxis_offset_abs"))
        ),
        "y_axis": _round_triplet(
            _metric_gate_error(pose.get("offset_y"), success_gates.get("yAxis_offset_abs"))
        ),
        "dist": _round_triplet(
            _metric_gate_error(pose.get("dist"), success_gates.get("dist"))
        ),
    }


def _progress_pct(analytics: dict | None) -> float | None:
    progress = _coerce_float((analytics or {}).get("progress"), None)
    if progress is None:
        return None
    return round(float(progress) * 100.0, 3)


def _suggestion_from_analytics(analytics: dict | None) -> dict:
    if not isinstance(analytics, dict) or not analytics:
        return {
            "action": "REOBSERVE",
            "reason": "no_post_observation_analytics",
            "highlight_metric": None,
            "speed_score_pct": None,
        }
    cmd = str(analytics.get("cmd") or "").strip().lower()
    speed_score = analytics.get("speed_score")
    if cmd not in {"f", "b", "l", "r"}:
        return {
            "action": "HOLD",
            "reason": "post_pose_within_current_alignment_policy",
            "highlight_metric": analytics.get("worst_metric"),
            "speed_score_pct": (
                int(round(float(speed_score)))
                if isinstance(speed_score, (int, float))
                else None
            ),
        }
    action = str(cmd).upper()
    if isinstance(speed_score, (int, float)):
        action = f"{action} {int(round(float(speed_score)))}%"
    return {
        "action": action,
        "reason": "post_pose_alignment_analytics",
        "highlight_metric": analytics.get("worst_metric"),
        "speed_score_pct": (
            int(round(float(speed_score)))
            if isinstance(speed_score, (int, float))
            else None
        ),
    }


def analyze_iteration(
    *,
    world,
    before_pose: dict | None,
    after_pose: dict | None,
    before_meta: dict | None = None,
    after_meta: dict | None = None,
) -> tuple[dict, dict]:
    success_gates = _step_success_gates(world, EXPERIMENT_STEP)
    before_analytics = _analytics_for_pose(world, before_pose, step=EXPERIMENT_STEP)
    after_analytics = _analytics_for_pose(world, after_pose, step=EXPERIMENT_STEP)
    before_errors = _pose_gate_errors(before_pose, success_gates)
    after_errors = _pose_gate_errors(after_pose, success_gates)

    error_delta = {}
    improved_metrics = []
    worsened_metrics = []
    for key in ("x_axis", "y_axis", "dist"):
        before_value = _coerce_float((before_errors or {}).get(key), None)
        after_value = _coerce_float((after_errors or {}).get(key), None)
        if before_value is None or after_value is None:
            error_delta[key] = None
            continue
        delta = float(after_value) - float(before_value)
        error_delta[key] = _round_triplet(delta)
        if delta <= -0.25:
            improved_metrics.append(key)
        elif delta >= 0.25:
            worsened_metrics.append(key)

    progress_before_pct = _progress_pct(before_analytics)
    progress_after_pct = _progress_pct(after_analytics)
    progress_delta_pct = None
    if progress_before_pct is not None and progress_after_pct is not None:
        progress_delta_pct = round(float(progress_after_pct) - float(progress_before_pct), 3)

    notes = []
    if before_pose is None:
        notes.append("No usable pre-run pose.")
    if after_pose is None:
        notes.append("No usable post-run pose.")
    if before_pose is not None and after_pose is not None:
        if progress_delta_pct is not None and progress_delta_pct > 0.5:
            primary_effect = "improved_alignment"
        elif progress_delta_pct is not None and progress_delta_pct < -0.5:
            primary_effect = "worsened_alignment"
        elif improved_metrics and not worsened_metrics:
            primary_effect = "reduced_gate_error"
        elif worsened_metrics and not improved_metrics:
            primary_effect = "increased_gate_error"
        else:
            primary_effect = "mixed_or_neutral"
    else:
        primary_effect = "insufficient_observation"

    analysis = {
        "observation_usable": bool(before_pose is not None and after_pose is not None),
        "before_mode": str((before_meta or {}).get("mode") or ""),
        "after_mode": str((after_meta or {}).get("mode") or ""),
        "progress_before_pct": progress_before_pct,
        "progress_after_pct": progress_after_pct,
        "progress_delta_pct": progress_delta_pct,
        "gate_error_before_mm": before_errors,
        "gate_error_after_mm": after_errors,
        "gate_error_delta_mm": error_delta,
        "improved_metrics": improved_metrics,
        "worsened_metrics": worsened_metrics,
        "primary_effect": primary_effect,
        "notes": notes,
    }
    suggestion = _suggestion_from_analytics(after_analytics if after_pose is not None else None)
    return analysis, suggestion


def _x_axis_turn_plan(
    *,
    world,
    pose: dict | None,
    analytics: dict | None,
    intensity_scale: float = 1.0,
    max_turn_intensity_pct: float = DEFAULT_MAX_TURN_INTENSITY_PCT,
    cmd_override: str | None = None,
    forced_min_turn_intensity_pct: float | None = None,
    other_gate_block_mm: float = DEFAULT_OTHER_GATE_BLOCK_MM,
) -> dict:
    success_gates = _step_success_gates(world, EXPERIMENT_STEP)
    errors = _pose_gate_errors(pose, success_gates) or {}
    x_gate_error_mm = _coerce_float(errors.get("x_axis"), None)
    y_gate_error_mm = _coerce_float(errors.get("y_axis"), None)
    dist_gate_error_mm = _coerce_float(errors.get("dist"), None)
    analytics = dict(analytics or {})
    cmd = str(analytics.get("cmd") or "").strip().lower()
    override_cmd = str(cmd_override or "").strip().lower()
    if override_cmd in {"l", "r"}:
        cmd = override_cmd

    base = {
        "cmd": None,
        "stage": "x_turn",
        "reason": "unavailable",
        "cmd_source": ("override" if override_cmd in {"l", "r"} else "analytics"),
        "highlight_metric": analytics.get("worst_metric"),
        "x_gate_error_mm": _round_triplet(x_gate_error_mm),
        "y_gate_error_mm": _round_triplet(y_gate_error_mm),
        "dist_gate_error_mm": _round_triplet(dist_gate_error_mm),
        "secondary_focus": _secondary_focus_status(success_gates, errors),
        "analytics_speed_score_pct": (
            int(round(float(analytics.get("speed_score"))))
            if isinstance(analytics.get("speed_score"), (int, float))
            else None
        ),
        "curve_intensity_pct": None,
        "base_turn_intensity_pct": None,
        "intensity_scale_used": _round_triplet(intensity_scale),
        "forced_floor_breakout_intensity_pct": _round_triplet(forced_min_turn_intensity_pct),
        "turn_intensity_pct": None,
        "predicted_score_pct": None,
        "predicted_duration_ms": None,
        "turn_intensity_effective_pct": None,
    }
    if not isinstance(pose, dict):
        base["reason"] = "no_pose"
        return base
    secondary_focus = base.get("secondary_focus") if isinstance(base.get("secondary_focus"), dict) else {}
    if not bool(secondary_focus.get("secondary_focus_ready")) and (
        (y_gate_error_mm or 0.0) > float(other_gate_block_mm)
        or (dist_gate_error_mm or 0.0) > float(other_gate_block_mm)
    ):
        base["reason"] = "x_axis_only_experiment_blocked_by_other_gate_errors"
        return base
    if cmd not in {"l", "r"}:
        base["reason"] = "within_x_axis_gate_or_non_turn_correction"
        return base
    if x_gate_error_mm is None or float(x_gate_error_mm) <= 0.0:
        base["reason"] = "within_x_axis_gate"
        return base

    curve = load_left_right_curve()
    curve_intensity_pct = None
    if isinstance(curve, dict):
        try:
            curve_intensity_pct = float(
                curve_intensity_for_error(curve, cmd, abs(float(x_gate_error_mm)))
            )
        except Exception:
            curve_intensity_pct = None

    analytics_speed_score_pct = _coerce_float(analytics.get("speed_score"), None)
    requested_intensity_pct = analytics_speed_score_pct
    if requested_intensity_pct is None:
        requested_intensity_pct = curve_intensity_pct
    elif curve_intensity_pct is not None:
        requested_intensity_pct = min(float(requested_intensity_pct), float(curve_intensity_pct))
    if requested_intensity_pct is None:
        requested_intensity_pct = float(max_turn_intensity_pct)

    base_turn_intensity_pct = _clamp(
        requested_intensity_pct,
        DEFAULT_MIN_TURN_INTENSITY_PCT,
        max_turn_intensity_pct,
    )
    requested_intensity_pct = float(base_turn_intensity_pct) * _clamp(
        intensity_scale,
        DEFAULT_MIN_INTENSITY_SCALE,
        DEFAULT_MAX_INTENSITY_SCALE,
    )
    if float(x_gate_error_mm) <= 0.75:
        requested_intensity_pct = float(DEFAULT_MIN_TURN_INTENSITY_PCT)
    elif float(x_gate_error_mm) <= 1.5:
        requested_intensity_pct = min(float(requested_intensity_pct), 1.0)
    requested_intensity_pct = _clamp(
        requested_intensity_pct,
        DEFAULT_MIN_TURN_INTENSITY_PCT,
        max_turn_intensity_pct,
    )
    forced_min_turn = _coerce_float(forced_min_turn_intensity_pct, None)
    if forced_min_turn is not None and float(x_gate_error_mm) > 0.0:
        requested_intensity_pct = max(float(requested_intensity_pct), float(forced_min_turn))
    if float(x_gate_error_mm) > 0.75:
        requested_intensity_pct = max(
            float(DEFAULT_X_TURN_EFFECTIVE_MIN_INTENSITY_PCT),
            float(requested_intensity_pct),
        )
        requested_intensity_pct = _clamp(
            requested_intensity_pct,
            DEFAULT_X_TURN_EFFECTIVE_MIN_INTENSITY_PCT,
            max_turn_intensity_pct,
        )

    _power, _pwm, predicted_score_pct, predicted_duration_ms, intensity_effective_pct = (
        telemetry_robot_module.speed_power_pwm_for_turn_intensity(cmd, requested_intensity_pct)
    )

    base.update(
        {
            "cmd": str(cmd),
            "reason": "targeted_x_axis_turn_tracking",
            "curve_intensity_pct": _round_triplet(curve_intensity_pct),
            "base_turn_intensity_pct": _round_triplet(base_turn_intensity_pct),
            "turn_intensity_pct": _round_triplet(requested_intensity_pct),
            "predicted_score_pct": int(predicted_score_pct),
            "predicted_duration_ms": int(predicted_duration_ms),
            "turn_intensity_effective_pct": _round_triplet(intensity_effective_pct),
        }
    )
    return base


def _dist_recovery_plan(
    *,
    world,
    pose: dict | None,
    analytics: dict | None,
    max_drive_score: float = DEFAULT_MAX_DRIVE_RECOVERY_SCORE,
    other_gate_block_mm: float = DEFAULT_OTHER_GATE_BLOCK_MM,
) -> dict:
    success_gates = _step_success_gates(world, EXPERIMENT_STEP)
    errors = _pose_gate_errors(pose, success_gates) or {}
    x_gate_error_mm = _coerce_float(errors.get("x_axis"), None)
    y_gate_error_mm = _coerce_float(errors.get("y_axis"), None)
    dist_gate_error_mm = _coerce_float(errors.get("dist"), None)
    analytics = dict(analytics or {})
    secondary_focus = _secondary_focus_status(success_gates, errors)
    analytics_speed_score_pct = (
        _coerce_float(analytics.get("speed_score"), None)
        if isinstance(analytics, dict)
        else None
    )
    base = {
        "cmd": None,
        "stage": "dist_recovery",
        "reason": "unavailable",
        "cmd_source": "range_recovery",
        "highlight_metric": "dist",
        "x_gate_error_mm": _round_triplet(x_gate_error_mm),
        "y_gate_error_mm": _round_triplet(y_gate_error_mm),
        "dist_gate_error_mm": _round_triplet(dist_gate_error_mm),
        "secondary_focus": secondary_focus,
        "analytics_speed_score_pct": (
            int(round(float(analytics_speed_score_pct)))
            if analytics_speed_score_pct is not None
            else None
        ),
        "predicted_score_pct": None,
        "predicted_duration_ms": None,
    }
    if not isinstance(pose, dict):
        base["reason"] = "no_pose"
        return base
    if not bool(secondary_focus.get("y_bplus_ok")) and (y_gate_error_mm or 0.0) > float(other_gate_block_mm):
        base["reason"] = "dist_recovery_blocked_by_y_error"
        return base
    if bool(secondary_focus.get("dist_bplus_ok")):
        base["reason"] = "dist_already_bplus"
        return base

    cmd = _dist_recovery_cmd(
        pose,
        success_gates.get("dist") if isinstance(success_gates.get("dist"), dict) else {},
    )
    if cmd not in {"f", "b"}:
        base["reason"] = "dist_recovery_cmd_unavailable"
        return base
    predicted_score_pct = _micro_drive_score_for_dist_error(
        dist_gate_error_mm,
        analytics_speed_score_pct=analytics_speed_score_pct,
        max_drive_score=max_drive_score,
    )
    _power, _pwm, predicted_score_pct, predicted_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
        cmd,
        predicted_score_pct,
    )
    base.update(
        {
            "cmd": str(cmd),
            "reason": "targeted_dist_recovery",
            "predicted_score_pct": int(predicted_score_pct),
            "predicted_duration_ms": int(predicted_duration_ms),
        }
    )
    return base


def _tracking_plan(
    *,
    world,
    pose: dict | None,
    analytics: dict | None,
    controller: dict,
    max_turn_intensity_pct: float,
    max_drive_recovery_score: float,
    pending_plateau_probe_scale: float | None = None,
    forced_x_turn_intensity_pct: float | None = None,
) -> tuple[dict, str | None]:
    recovery_plan = _dist_recovery_plan(
        world=world,
        pose=pose,
        analytics=analytics,
        max_drive_score=float(max_drive_recovery_score),
    )
    if recovery_plan.get("cmd") in {"f", "b"}:
        return recovery_plan, None

    selected_cmd, selection_source = _select_turn_cmd(
        controller,
        analytics_cmd=(analytics or {}).get("cmd"),
    )
    intensity_scale = _controller_intensity_scale(controller, selected_cmd)
    if pending_plateau_probe_scale is not None:
        intensity_scale = min(float(intensity_scale), float(pending_plateau_probe_scale))
    plan = _x_axis_turn_plan(
        world=world,
        pose=pose,
        analytics=analytics,
        intensity_scale=float(intensity_scale),
        max_turn_intensity_pct=float(max_turn_intensity_pct),
        cmd_override=selected_cmd,
        forced_min_turn_intensity_pct=forced_x_turn_intensity_pct,
    )
    plan["cmd_selection_source"] = str(selection_source)
    if pending_plateau_probe_scale is not None:
        plan["plateau_probe_scale_used"] = _round_triplet(pending_plateau_probe_scale)
    return plan, selected_cmd


def _post_move_observe_policy(
    *,
    stage: str | None,
    track_observe_timeout_s: float,
    track_relaxed_timeout_s: float,
) -> dict:
    stage_key = str(stage or "").strip().lower()
    if stage_key == "x_turn":
        return {
            "timeout_s": max(
                float(track_observe_timeout_s),
                float(DEFAULT_X_TURN_POST_MOVE_OBSERVE_TIMEOUT_S),
            ),
            "relaxed_timeout_s": max(
                float(track_relaxed_timeout_s),
                float(DEFAULT_X_TURN_POST_MOVE_RELAXED_TIMEOUT_S),
            ),
            "reobserve_rounds": int(DEFAULT_X_TURN_POST_MOVE_REOBSERVE_ROUNDS),
            "policy": "x_turn_stronger_reacquire",
        }
    return {
        "timeout_s": float(track_observe_timeout_s),
        "relaxed_timeout_s": float(track_relaxed_timeout_s),
        "reobserve_rounds": 1,
        "policy": "default_post_move_reacquire",
    }


def run_x_axis_lock_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    max_cycles: int = DEFAULT_TRACK_MAX_CYCLES,
    observe_samples: int = DEFAULT_OBSERVE_SAMPLES,
    initial_observe_timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    initial_relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    track_observe_timeout_s: float = DEFAULT_TRACK_OBSERVE_TIMEOUT_S,
    track_relaxed_timeout_s: float = DEFAULT_TRACK_RELAXED_TIMEOUT_S,
    settle_s: float = 0.02,
    max_turn_intensity_pct: float = DEFAULT_MAX_TURN_INTENSITY_PCT,
    max_drive_recovery_score: float = DEFAULT_MAX_DRIVE_RECOVERY_SCORE,
    min_progress_mm: float = DEFAULT_MIN_PROGRESS_MM,
    stall_limit: int = DEFAULT_STALL_LIMIT,
    direction_flip_worsen_mm: float = DEFAULT_DIRECTION_FLIP_WORSEN_MM,
    max_direction_flips: int = DEFAULT_MAX_DIRECTION_FLIPS,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    memory_path: Path | None = RUN_MEMORY_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    world.step_state = EXPERIMENT_STEP
    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())

    logger("[X-LOCK EXPERIMENT] Strategy: targeted x-axis lock tracking.")
    logger(f"[X-LOCK EXPERIMENT] Vision backend: {str(vision_mode_used)}.")

    initial_pose = None
    initial_meta = {}
    final_pose = None
    final_meta = {}
    cycles = []
    pulses = []
    stop_reason = "unknown"
    stagnant_cycles = 0
    direction_flip_count = 0
    controller = _load_x_lock_memory(memory_path)
    loaded_controller = _controller_snapshot(controller)
    plateau_recovery_used = False
    pending_plateau_probe_scale = None
    pending_x_floor_breakout_intensity_pct = None
    x_floor_stall_count = 0

    try:
        logger("[X-LOCK EXPERIMENT] Observing initial pose.")
        initial_pose, initial_meta = observe_alignment_pose(
            vision=vision,
            world=world,
            samples=int(observe_samples),
            timeout_s=float(initial_observe_timeout_s),
            relaxed_timeout_s=float(initial_relaxed_timeout_s),
            reobserve_rounds=int(DEFAULT_REOBSERVE_ROUNDS),
            log_fn=logger,
        )
        final_pose = initial_pose
        final_meta = dict(initial_meta or {})
        logger(f"[X-LOCK EXPERIMENT] Initial pose: {_pose_summary_text(initial_pose)}")

        if initial_pose is None:
            stop_reason = "no_initial_pose"
        else:
            current_pose = initial_pose
            current_meta = dict(initial_meta or {})
            for cycle_idx in range(1, max(1, int(max_cycles)) + 1):
                current_analytics = _analytics_for_pose(world, current_pose, step=EXPERIMENT_STEP)
                cycle_plateau_probe_scale = pending_plateau_probe_scale
                plan, selected_cmd = _tracking_plan(
                    world=world,
                    pose=current_pose,
                    analytics=current_analytics,
                    controller=controller,
                    max_turn_intensity_pct=float(max_turn_intensity_pct),
                    max_drive_recovery_score=float(max_drive_recovery_score),
                    pending_plateau_probe_scale=cycle_plateau_probe_scale,
                    forced_x_turn_intensity_pct=pending_x_floor_breakout_intensity_pct,
                )
                if cycle_plateau_probe_scale is not None:
                    pending_plateau_probe_scale = None
                if (
                    pending_x_floor_breakout_intensity_pct is not None
                    and str(plan.get("stage") or "") == "x_turn"
                ):
                    pending_x_floor_breakout_intensity_pct = None
                cycle_record = {
                    "cycle": int(cycle_idx),
                    "pre_pose": _pose_snapshot(current_pose),
                    "pre_meta": dict(current_meta or {}),
                    "pre_analytics": {
                        "progress_pct": _progress_pct(current_analytics),
                        "cmd": current_analytics.get("cmd"),
                        "speed_score_pct": current_analytics.get("speed_score"),
                        "worst_metric": current_analytics.get("worst_metric"),
                    },
                    "controller_before": _controller_snapshot(controller),
                    "plan": dict(plan),
                    "direction_flip_count_before_cycle": int(direction_flip_count),
                }

                if plan.get("cmd") not in {"l", "r", "f", "b"}:
                    stop_reason = str(plan.get("reason") or "no_turn_plan")
                    cycle_record["stopped"] = True
                    cycles.append(cycle_record)
                    break

                if str(plan.get("stage") or "") == "dist_recovery":
                    logger(
                        "[X-LOCK EXPERIMENT] Cycle "
                        f"{int(cycle_idx)}: {str(plan['cmd']).upper()} "
                        f"{int(plan['predicted_score_pct'])}% "
                        f"(duration~{int(plan['predicted_duration_ms'])}ms, "
                        f"dist_err={float(plan['dist_gate_error_mm']):.3f}mm, "
                        f"x_err={float(plan['x_gate_error_mm']):.3f}mm)."
                    )
                    send_result = send_robot_command(
                        robot,
                        world,
                        EXPERIMENT_STEP,
                        str(plan["cmd"]),
                        speed=0.0,
                        speed_score=int(plan["predicted_score_pct"] or telemetry_robot_module.SPEED_SCORE_MIN),
                        auto_mode=False,
                        ease_in_out_enabled=False,
                    )
                else:
                    logger(
                        "[X-LOCK EXPERIMENT] Cycle "
                        f"{int(cycle_idx)}: {str(plan['cmd']).upper()} "
                        f"{float(plan['turn_intensity_pct']):.2f}% "
                        f"(score~{int(plan['predicted_score_pct'])}% "
                        f"duration~{int(plan['predicted_duration_ms'])}ms, "
                        f"x_err={float(plan['x_gate_error_mm']):.3f}mm)."
                    )
                    send_result = send_robot_command(
                        robot,
                        world,
                        EXPERIMENT_STEP,
                        str(plan["cmd"]),
                        speed=0.0,
                        speed_score=int(plan["predicted_score_pct"] or telemetry_robot_module.SPEED_SCORE_MIN),
                        turn_intensity=float(plan["turn_intensity_pct"]),
                        auto_mode=False,
                        ease_in_out_enabled=False,
                        half_first_turn_pulse=False,
                    )
                cycle_record["send_result"] = dict(send_result or {})
                pulse_row = {
                    "cmd": str(plan["cmd"]),
                    "stage": str(plan.get("stage") or ""),
                    "send_result": dict(send_result or {}),
                }
                if plan.get("turn_intensity_pct") is not None:
                    pulse_row["turn_intensity_pct"] = float(plan["turn_intensity_pct"])
                if plan.get("predicted_score_pct") is not None:
                    pulse_row["score_pct"] = int(plan["predicted_score_pct"])
                pulses.append(pulse_row)
                effective_duration_ms = int(
                    (send_result or {}).get("duration_ms")
                    or int(plan["predicted_duration_ms"] or 0)
                    or 0
                )
                time.sleep(max(0.0, float(settle_s)) + max(0.0, float(effective_duration_ms) / 1000.0))

                post_move_observe_policy = _post_move_observe_policy(
                    stage=str(plan.get("stage") or ""),
                    track_observe_timeout_s=float(track_observe_timeout_s),
                    track_relaxed_timeout_s=float(track_relaxed_timeout_s),
                )
                cycle_record["post_move_observe_policy"] = dict(post_move_observe_policy)
                next_pose, next_meta = observe_alignment_pose(
                    vision=vision,
                    world=world,
                    samples=int(observe_samples),
                    timeout_s=float(post_move_observe_policy["timeout_s"]),
                    relaxed_timeout_s=float(post_move_observe_policy["relaxed_timeout_s"]),
                    reobserve_rounds=int(post_move_observe_policy["reobserve_rounds"]),
                    log_fn=logger,
                )
                cycle_record["post_pose"] = _pose_snapshot(next_pose)
                cycle_record["post_meta"] = dict(next_meta or {})
                cycle_record["post_summary"] = _pose_summary_text(next_pose)

                iteration_analysis, next_suggestion = analyze_iteration(
                    world=world,
                    before_pose=current_pose,
                    after_pose=next_pose,
                    before_meta=current_meta,
                    after_meta=next_meta,
                )
                cycle_record["iteration_analysis"] = iteration_analysis
                cycle_record["next_suggestion"] = next_suggestion
                cycles.append(cycle_record)

                if next_pose is None:
                    stop_reason = "lost_pose_after_move"
                    final_pose = None
                    final_meta = dict(next_meta or {})
                    break

                final_pose = next_pose
                final_meta = dict(next_meta or {})
                x_before = _coerce_float((iteration_analysis.get("gate_error_before_mm") or {}).get("x_axis"), None)
                x_after = _coerce_float((iteration_analysis.get("gate_error_after_mm") or {}).get("x_axis"), None)
                dist_before = _coerce_float((iteration_analysis.get("gate_error_before_mm") or {}).get("dist"), None)
                dist_after = _coerce_float((iteration_analysis.get("gate_error_after_mm") or {}).get("dist"), None)
                x_improvement_mm = None
                dist_improvement_mm = None
                if x_before is not None and x_after is not None:
                    x_improvement_mm = float(x_before) - float(x_after)
                if dist_before is not None and dist_after is not None:
                    dist_improvement_mm = float(dist_before) - float(dist_after)
                if str(plan.get("stage") or "") == "x_turn":
                    controller_update = _update_x_lock_controller(
                        controller,
                        cmd=str(plan.get("cmd") or ""),
                        x_improvement_mm=x_improvement_mm,
                        min_progress_mm=float(min_progress_mm),
                        worsen_mm=float(direction_flip_worsen_mm),
                    )
                    cycle_record["controller_update"] = dict(controller_update)
                cycle_record["controller_after"] = _controller_snapshot(controller)

                if str(plan.get("stage") or "") == "dist_recovery":
                    x_floor_stall_count = 0
                    if dist_improvement_mm is not None and dist_improvement_mm >= float(min_progress_mm):
                        stagnant_cycles = 0
                    else:
                        stagnant_cycles += 1
                else:
                    controller_stats = (
                        cycle_record.get("controller_update", {}).get("stats")
                        if isinstance(cycle_record.get("controller_update"), dict)
                        else {}
                    )
                    worsen_streak = int((controller_stats or {}).get("worsen_streak") or 0)
                    hard_worsen = bool(
                        x_improvement_mm is not None
                        and x_improvement_mm <= (-1.0 * float(DEFAULT_DIRECTION_HARD_FLIP_WORSEN_MM))
                    )
                    if hard_worsen or worsen_streak >= 2:
                        direction_flip_count += 1
                        stagnant_cycles = 0
                        cycle_record["direction_override_reason"] = (
                            f"x_gate_error_worsened_by_{abs(float(x_improvement_mm or 0.0)):.3f}mm"
                        )
                        if direction_flip_count > max(0, int(max_direction_flips)) and _controller_direction_unstable(controller):
                            stop_reason = "direction_unstable"
                            break
                    elif x_improvement_mm is not None and x_improvement_mm >= float(min_progress_mm):
                        stagnant_cycles = 0
                        direction_flip_count = 0
                        x_floor_stall_count = 0
                    else:
                        stagnant_cycles += 1
                        if (
                            bool(((plan.get("secondary_focus") or {}).get("secondary_focus_ready")))
                            and _coerce_float(plan.get("turn_intensity_pct"), 0.0) is not None
                            and float(_coerce_float(plan.get("turn_intensity_pct"), 0.0) or 0.0)
                            <= float(DEFAULT_X_TURN_EFFECTIVE_MIN_INTENSITY_PCT)
                        ):
                            x_floor_stall_count += 1
                        else:
                            x_floor_stall_count = 0
                        if x_floor_stall_count >= int(DEFAULT_X_TURN_BREAKOUT_TRIGGER_CYCLES):
                            breakout_intensity = _x_turn_breakout_intensity_pct(
                                x_after if x_after is not None else x_before,
                                max_turn_intensity_pct=float(max_turn_intensity_pct),
                            )
                            pending_x_floor_breakout_intensity_pct = float(breakout_intensity)
                            cycle_record["x_floor_breakout_scheduled"] = {
                                "turn_intensity_pct": _round_triplet(breakout_intensity),
                                "trigger_cycles": int(DEFAULT_X_TURN_BREAKOUT_TRIGGER_CYCLES),
                            }
                            x_floor_stall_count = 0

                current_pose = next_pose
                current_meta = dict(next_meta or {})
                if str((next_suggestion or {}).get("action") or "").strip().upper() == "HOLD":
                    stop_reason = "locked"
                    break
                if (
                    str(plan.get("stage") or "") == "x_turn"
                    and _controller_direction_unstable(controller)
                    and direction_flip_count > max(0, int(max_direction_flips))
                ):
                    stop_reason = "direction_unstable"
                    break
                if str(plan.get("stage") or "") == "dist_recovery" and stagnant_cycles >= max(1, int(stall_limit)):
                    stop_reason = "stalled_dist_recovery"
                    break
                if str(plan.get("stage") or "") == "x_turn" and stagnant_cycles >= max(1, int(stall_limit)):
                    if not plateau_recovery_used:
                        plateau_recovery_used = True
                        stagnant_cycles = 0
                        pending_plateau_probe_scale = float(DEFAULT_PLATEAU_PROBE_SCALE)
                        logger(
                            "[X-LOCK EXPERIMENT] Plateau detected. Reobserving longer and scheduling a lower-intensity probe."
                        )
                        rescued_pose, rescued_meta = observe_alignment_pose(
                            vision=vision,
                            world=world,
                            samples=int(observe_samples),
                            timeout_s=max(float(track_observe_timeout_s), float(DEFAULT_PLATEAU_REOBSERVE_TIMEOUT_S)),
                            relaxed_timeout_s=max(float(track_relaxed_timeout_s), float(DEFAULT_PLATEAU_REOBSERVE_TIMEOUT_S) + 0.4),
                            reobserve_rounds=int(DEFAULT_PLATEAU_REOBSERVE_ROUNDS),
                            log_fn=logger,
                        )
                        cycle_record["plateau_recovery"] = {
                            "used": True,
                            "probe_scale": _round_triplet(DEFAULT_PLATEAU_PROBE_SCALE),
                            "rescued_pose": _pose_snapshot(rescued_pose),
                            "rescued_meta": dict(rescued_meta or {}),
                        }
                        if rescued_pose is not None:
                            current_pose = rescued_pose
                            current_meta = dict(rescued_meta or {})
                            final_pose = rescued_pose
                            final_meta = dict(rescued_meta or {})
                            continue
                    stop_reason = "stalled_x_progress"
                    break
            else:
                stop_reason = "max_cycles_reached"

        final_analysis, final_suggestion = analyze_iteration(
            world=world,
            before_pose=initial_pose,
            after_pose=final_pose,
            before_meta=initial_meta,
            after_meta=final_meta,
        )
        if stop_reason == "locked":
            final_suggestion = {
                "action": "HOLD",
                "reason": "x_axis_lock_experiment_reached_gate",
                "highlight_metric": None,
                "speed_score_pct": None,
            }
        final_next_plan = None
        if final_pose is not None and stop_reason != "locked":
            final_analytics = _analytics_for_pose(world, final_pose, step=EXPERIMENT_STEP)
            final_next_plan, _ = _tracking_plan(
                world=world,
                pose=final_pose,
                analytics=final_analytics,
                controller=controller,
                max_turn_intensity_pct=float(max_turn_intensity_pct),
                max_drive_recovery_score=float(max_drive_recovery_score),
                forced_x_turn_intensity_pct=pending_x_floor_breakout_intensity_pct,
            )

        result = {
            "ok": bool(initial_pose is not None),
            "experiment_type": "targeted_x_axis_lock_tracking",
            "objective": "recover dist when needed, then maintain brick lock with repeated tiny L/R corrections under observe-act cadence",
            "seconds": float(max(0.0, time.time() - started)),
            "vision_mode": str(vision_mode_used),
            "max_cycles": int(max(1, int(max_cycles))),
            "cycles_completed": int(sum(1 for row in cycles if isinstance(row, dict) and row.get("send_result"))),
            "stop_reason": str(stop_reason),
            "pulses": pulses,
            "observation": {
                "before": _pose_snapshot(initial_pose),
                "before_meta": dict(initial_meta or {}),
                "after": _pose_snapshot(final_pose),
                "after_meta": dict(final_meta or {}),
            },
            "cycles": cycles,
            "memory_loaded": loaded_controller,
            "plateau_recovery_used": bool(plateau_recovery_used),
            "controller": _controller_snapshot(controller),
            "analysis": final_analysis,
            "suggestion": final_suggestion,
            "next_plan": final_next_plan,
        }
        memory_saved_path = _save_x_lock_memory(
            memory_path,
            controller=controller,
            stop_reason=str(stop_reason),
            final_pose=final_pose,
        )
        if memory_saved_path:
            result["memory_path"] = str(memory_saved_path)
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def run_single_goal_right_turn_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    trials: int = DEFAULT_X_ZERO_TRIALS,
    sample_hz: float = DEFAULT_X_ZERO_SAMPLE_HZ,
    reset_timeout_s: float = DEFAULT_X_ZERO_RESET_TIMEOUT_S,
    attempt_timeout_s: float = DEFAULT_X_ZERO_ATTEMPT_TIMEOUT_S,
    reset_turn_intensity_pct: float = DEFAULT_X_ZERO_RESET_TURN_INTENSITY_PCT,
    turn_intensity_pct: float = DEFAULT_X_ZERO_TURN_INTENSITY_PCT,
    zero_band_mm: float = DEFAULT_X_ZERO_BAND_MM,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    run_started_monotonic = time.monotonic()
    world.step_state = EXPERIMENT_STEP
    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())

    trial_count = max(1, int(trials))
    sample_hz_used = max(1.0, float(_coerce_float(sample_hz, DEFAULT_X_ZERO_SAMPLE_HZ) or DEFAULT_X_ZERO_SAMPLE_HZ))
    reset_timeout_used_s = max(0.25, float(_coerce_float(reset_timeout_s, DEFAULT_X_ZERO_RESET_TIMEOUT_S) or DEFAULT_X_ZERO_RESET_TIMEOUT_S))
    attempt_timeout_used_s = max(0.25, float(_coerce_float(attempt_timeout_s, DEFAULT_X_ZERO_ATTEMPT_TIMEOUT_S) or DEFAULT_X_ZERO_ATTEMPT_TIMEOUT_S))
    reset_turn_intensity_used = max(
        float(DEFAULT_MIN_TURN_INTENSITY_PCT),
        float(_coerce_float(reset_turn_intensity_pct, DEFAULT_X_ZERO_RESET_TURN_INTENSITY_PCT) or DEFAULT_X_ZERO_RESET_TURN_INTENSITY_PCT),
    )
    turn_intensity_used = max(
        float(DEFAULT_MIN_TURN_INTENSITY_PCT),
        float(_coerce_float(turn_intensity_pct, DEFAULT_X_ZERO_TURN_INTENSITY_PCT) or DEFAULT_X_ZERO_TURN_INTENSITY_PCT),
    )
    zero_band_used_mm = max(0.0, float(_coerce_float(zero_band_mm, DEFAULT_X_ZERO_BAND_MM) or DEFAULT_X_ZERO_BAND_MM))

    logger("[X-ZERO EXPERIMENT] Strategy: reset left until the brick is hidden or plateaued at the left edge, then continuous right turn until x-axis crosses zero.")
    logger(
        f"[X-ZERO EXPERIMENT] Vision backend: {str(vision_mode_used)}. "
        f"Trials={int(trial_count)} sample_hz={float(sample_hz_used):.1f} "
        f"reset_turn={float(reset_turn_intensity_used):.2f}% right_turn={float(turn_intensity_used):.2f}% "
        f"zero_band={float(zero_band_used_mm):.2f}mm."
    )

    trials_out = []
    pulses = []
    stop_reason = "completed"
    success_count = 0

    try:
        for trial_index in range(1, int(trial_count) + 1):
            trial_row = {
                "trial": int(trial_index),
            }
            update_world_from_vision(world, vision, log=False)
            trial_row["pre_reset_pose"] = _pose_snapshot(_current_world_pose(world, obs_ts=time.time()))
            trial_row["pre_reset_visible"] = bool((getattr(world, "brick", {}) or {}).get("visible"))
            logger(
                "[X-ZERO EXPERIMENT] Trial "
                f"{int(trial_index)}/{int(trial_count)}: reset left until visibility is lost or the left edge plateaus."
            )

            if not bool(trial_row["pre_reset_visible"]):
                reset_result = {
                    "status": "already_not_visible",
                    "seconds": 0.0,
                    "send_result": {},
                    "samples": [],
                    "sample_summary": _summarize_x_zero_samples([]),
                    "end_pose": None,
                    "end_visible": False,
                }
            else:
                reset_stage = _run_turn_stage(
                    robot=robot,
                    world=world,
                    vision=vision,
                    trial_index=int(trial_index),
                    stage="reset_left_to_edge",
                    cmd="l",
                    turn_intensity_pct=float(reset_turn_intensity_used),
                    timeout_s=float(reset_timeout_used_s),
                    sample_hz=float(sample_hz_used),
                    run_started_monotonic=float(run_started_monotonic),
                    stop_evaluator=lambda sample, state, samples: _reset_until_left_edge_evaluator(
                        sample,
                        state,
                        samples,
                    ),
                )
                reset_result = {
                    **reset_stage,
                    "status": (
                        "lost_visibility"
                        if str(reset_stage.get("stop_reason") or "") == "lost_visibility"
                        else (
                            "left_edge_plateau"
                            if str(reset_stage.get("stop_reason") or "") == "left_edge_plateau"
                            else "reset_timeout_visible"
                        )
                    ),
                }
                if isinstance(reset_stage.get("send_result"), dict) and reset_stage["send_result"]:
                    pulses.append(
                        {
                            "trial": int(trial_index),
                            "stage": "reset_left_to_edge",
                            "cmd": "l",
                            "turn_intensity_pct": _round_triplet(reset_turn_intensity_used),
                            "send_result": dict(reset_stage["send_result"]),
                            "send_results": list(reset_stage.get("send_results") or []),
                        }
                    )
            trial_row["reset"] = reset_result

            reset_status = str((reset_result or {}).get("status") or "")
            if reset_status not in {"lost_visibility", "already_not_visible", "left_edge_plateau"}:
                logger(
                    "[X-ZERO EXPERIMENT] Trial "
                    f"{int(trial_index)}: reset failed ({reset_status}); skipping right-turn attempt."
                )
                trial_row["attempt"] = {
                    "trial": int(trial_index),
                    "status": "not_started_reset_failed",
                    "success": False,
                    "seconds": 0.0,
                    "send_result": {},
                    "samples": [],
                    "sample_summary": _summarize_x_zero_samples([]),
                    "first_visible_pose": None,
                    "best_visible_pose": None,
                    "best_visible_abs_x_mm": None,
                    "end_pose": None,
                    "end_visible": bool((reset_result or {}).get("end_visible")),
                }
                trials_out.append(trial_row)
                continue

            logger(
                "[X-ZERO EXPERIMENT] Trial "
                f"{int(trial_index)}: continuous right turn at {float(turn_intensity_used):.2f}% until x-axis reaches zero."
            )
            attempt_stage = _run_turn_stage(
                robot=robot,
                world=world,
                vision=vision,
                trial_index=int(trial_index),
                stage="right_turn_to_x_zero",
                cmd="r",
                turn_intensity_pct=float(turn_intensity_used),
                timeout_s=float(attempt_timeout_used_s),
                sample_hz=float(sample_hz_used),
                run_started_monotonic=float(run_started_monotonic),
                stop_evaluator=_right_turn_to_zero_evaluator(zero_band_mm=float(zero_band_used_mm)),
            )
            attempt_status = str(attempt_stage.get("stop_reason") or "")
            if attempt_status == "duration_complete":
                attempt_state = attempt_stage.get("state") if isinstance(attempt_stage.get("state"), dict) else {}
                if not bool(attempt_state.get("ever_visible")):
                    attempt_status = "never_reacquired"
                else:
                    attempt_status = "attempt_timeout_before_zero"
            success = attempt_status in {"x_axis_zero_band", "x_axis_zero_cross"}
            if success:
                success_count += 1
            attempt_result = {
                **attempt_stage,
                "trial": int(trial_index),
                "status": str(attempt_status),
                "success": bool(success),
                "first_visible_pose": (attempt_stage.get("state") or {}).get("first_visible_pose"),
                "best_visible_pose": (attempt_stage.get("state") or {}).get("best_visible_pose"),
                "best_visible_abs_x_mm": _round_triplet((attempt_stage.get("state") or {}).get("best_visible_abs_x_mm")),
                "first_visible_x_mm": (attempt_stage.get("state") or {}).get("first_visible_x_mm"),
                "success_type": (attempt_stage.get("state") or {}).get("success_type"),
            }
            if isinstance(attempt_stage.get("send_result"), dict) and attempt_stage["send_result"]:
                pulses.append(
                    {
                        "trial": int(trial_index),
                        "stage": "right_turn_to_x_zero",
                        "cmd": "r",
                        "turn_intensity_pct": _round_triplet(turn_intensity_used),
                        "send_result": dict(attempt_stage["send_result"]),
                        "send_results": list(attempt_stage.get("send_results") or []),
                    }
                )
            logger(
                "[X-ZERO EXPERIMENT] Trial "
                f"{int(trial_index)} result: {str(attempt_status)} "
                f"(best |x|={attempt_result.get('best_visible_abs_x_mm')}mm)."
            )
            trial_row["attempt"] = attempt_result
            trials_out.append(trial_row)

        analysis, suggestion = _summarize_single_goal_trials(trials_out)
        result = {
            "ok": bool(len(trials_out) == int(trial_count)),
            "experiment_type": "single_goal_right_turn_x_zero",
            "objective": "from a left-edge reset, turn right continuously at 1% until x-axis reaches zero",
            "seconds": float(max(0.0, time.time() - started)),
            "vision_mode": str(vision_mode_used),
            "trials_requested": int(trial_count),
            "trials_completed": int(len(trials_out)),
            "success_count": int(success_count),
            "stop_reason": str(stop_reason),
            "sample_hz": _round_triplet(sample_hz_used),
            "reset_timeout_s": _round_triplet(reset_timeout_used_s),
            "attempt_timeout_s": _round_triplet(attempt_timeout_used_s),
            "max_act_duration_ms": int(DEFAULT_X_ZERO_MAX_ACT_DURATION_MS),
            "reset_turn_intensity_pct": _round_triplet(reset_turn_intensity_used),
            "turn_intensity_pct": _round_triplet(turn_intensity_used),
            "zero_band_mm": _round_triplet(zero_band_used_mm),
            "pulses": pulses,
            "trials": trials_out,
            "analysis": analysis,
            "suggestion": suggestion,
        }
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def run_random_back_turn_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    score: int = DEFAULT_SCORE,
    rng: random.Random | None = None,
    seed: int | None = None,
    back_range_ms: tuple[int, int] = DEFAULT_BACK_RANGE_MS,
    turn_range_ms: tuple[int, int] = DEFAULT_TURN_RANGE_MS,
    settle_s: float = DEFAULT_SETTLE_S,
    observe_samples: int = DEFAULT_OBSERVE_SAMPLES,
    observe_timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    reobserve_rounds: int = DEFAULT_REOBSERVE_ROUNDS,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    world.step_state = EXPERIMENT_STEP
    plan = choose_random_back_turn_plan(
        rng=rng,
        back_range_ms=tuple(back_range_ms),
        turn_range_ms=tuple(turn_range_ms),
    )
    try:
        score_used = int(round(float(score)))
    except (TypeError, ValueError):
        score_used = int(DEFAULT_SCORE)
    score_used = max(1, min(100, int(score_used)))
    settle_used_s = max(0.0, float(settle_s))
    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())

    logger(
        "[RANDOM EXPERIMENT] Plan: "
        f"B {int(plan['back_duration_ms'])}ms, "
        f"{str(plan['turn_cmd']).upper()} {int(plan['turn_duration_ms'])}ms "
        f"at shared {int(score_used)}%."
    )
    logger(f"[RANDOM EXPERIMENT] Vision backend: {str(vision_mode_used)}.")

    logger("[RANDOM EXPERIMENT] Observing pre-run pose.")
    before_pose, before_meta = observe_alignment_pose(
        vision=vision,
        world=world,
        samples=int(observe_samples),
        timeout_s=float(observe_timeout_s),
        relaxed_timeout_s=float(relaxed_timeout_s),
        reobserve_rounds=int(reobserve_rounds),
        log_fn=logger,
    )
    logger(f"[RANDOM EXPERIMENT] Pre-run pose: {_pose_summary_text(before_pose)}")

    pulses = []
    try:
        for pulse in list(plan.get("sequence") or []):
            cmd = str((pulse or {}).get("cmd") or "").strip().lower()
            try:
                duration_ms = int(round(float((pulse or {}).get("duration_ms"))))
            except (TypeError, ValueError):
                duration_ms = 0
            if cmd not in {"b", "l", "r"} or duration_ms <= 0:
                continue
            logger(
                f"[RANDOM EXPERIMENT] Sending {cmd.upper()} {int(score_used)}% for {int(duration_ms)}ms."
            )
            send_result = send_robot_command(
                robot,
                world,
                EXPERIMENT_STEP,
                cmd,
                speed=0.0,
                speed_score=int(score_used),
                duration_override_ms=int(duration_ms),
                auto_mode=False,
                ease_in_out_enabled=False,
            )
            pulses.append(
                {
                    "cmd": str(cmd),
                    "duration_ms": int(duration_ms),
                    "send_result": dict(send_result or {}),
                }
            )
            effective_duration_ms = int((send_result or {}).get("duration_ms") or int(duration_ms))
            time.sleep(max(settle_used_s, (float(effective_duration_ms) / 1000.0) + 0.05))

        logger("[RANDOM EXPERIMENT] Observing post-run pose.")
        after_pose, after_meta = observe_alignment_pose(
            vision=vision,
            world=world,
            samples=int(observe_samples),
            timeout_s=float(observe_timeout_s),
            relaxed_timeout_s=float(relaxed_timeout_s),
            reobserve_rounds=int(reobserve_rounds),
            log_fn=logger,
        )
        logger(f"[RANDOM EXPERIMENT] Post-run pose: {_pose_summary_text(after_pose)}")

        analysis, suggestion = analyze_iteration(
            world=world,
            before_pose=before_pose,
            after_pose=after_pose,
            before_meta=before_meta,
            after_meta=after_meta,
        )
        logger(
            "[RANDOM EXPERIMENT] Analysis: "
            f"{str(analysis.get('primary_effect') or 'unavailable')} "
            f"(progress delta={analysis.get('progress_delta_pct')})."
        )
        logger(
            "[RANDOM EXPERIMENT] Suggestion: "
            f"{str(suggestion.get('action') or 'REOBSERVE')} "
            f"[{str(suggestion.get('reason') or 'unknown')}]"
        )

        result = {
            "ok": True,
            "seconds": float(max(0.0, time.time() - started)),
            "seed": (int(seed) if seed is not None else None),
            "vision_mode": str(vision_mode_used),
            "score_pct": int(score_used),
            "plan": dict(plan),
            "pulses": pulses,
            "observation": {
                "before": _pose_snapshot(before_pose),
                "before_meta": dict(before_meta or {}),
                "after": _pose_snapshot(after_pose),
                "after_meta": dict(after_meta or {}),
            },
            "analysis": analysis,
            "suggestion": suggestion,
        }
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def run_close_dist_x_axis_one_act_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    score: int = DEFAULT_SCORE,
    trials: int = DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS,
    phase_duration_ms: int = DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_DURATION_MS,
    phase: str = DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_PHASE,
    strength: str = DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
    sequence=DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE,
    settle_s: float = DEFAULT_SETTLE_S,
    observe_samples: int = DEFAULT_OBSERVE_SAMPLES,
    observe_timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    reobserve_rounds: int = DEFAULT_REOBSERVE_ROUNDS,
    setup_forward_range_ms=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_SETUP_FORWARD_RANGE_MS,
    rng: random.Random | None = None,
    distance_band_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
    production_worthy_threshold_pct: float = DEFAULT_CURVE_PRODUCTION_WORTHY_THRESHOLD_PCT,
    trial_manifest: dict | None = None,
    trials_path: Path | None = TRIALS_FILE_DEFAULT,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
    reset_x_every_n_trials: int = 0,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    world.step_state = EXPERIMENT_STEP
    manifest_name = _trial_file_name_from_manifest(
        trial_manifest,
        fallback=(Path(trials_path).stem if trials_path is not None else phase),
    )
    provided_trial_manifest = _normalize_close_dist_trial_manifest(
        trial_manifest,
        fallback_name=manifest_name,
    )
    provided_curve_cfg = _trial_manifest_curve_config(provided_trial_manifest)
    manifest_supplied = isinstance(provided_trial_manifest, dict)
    phase_source = str((provided_curve_cfg or {}).get("phase") or phase or "")
    strength_source = str((provided_curve_cfg or {}).get("turn_strength") or strength or "")
    score_source = (provided_curve_cfg or {}).get("score_pct") if bool(manifest_supplied) else score
    duration_source = (
        (provided_curve_cfg or {}).get("measured_phase_duration_ms")
        if bool(manifest_supplied)
        else phase_duration_ms
    )
    trials_source = (
        len(_trial_manifest_trials_list(provided_trial_manifest))
        if bool(manifest_supplied) and _trial_manifest_trials_list(provided_trial_manifest)
        else ((provided_curve_cfg or {}).get("trial_count") if bool(manifest_supplied) else trials)
    )
    setup_forward_range_source = (
        (provided_curve_cfg or {}).get("setup_turn_duration_range_ms")
        if bool(manifest_supplied)
        else setup_forward_range_ms
    )
    reset_x_every_n_trials_source = (
        (provided_curve_cfg or {}).get("reset_x_every_n_trials")
        if bool(manifest_supplied)
        else reset_x_every_n_trials
    )
    phase_spec = _resolve_turn_drive_phase_spec(
        sequence,
        phase=str(phase_source),
        strength=str(strength_source),
    )
    phase_key = str((phase_spec or {}).get("phase") or "")
    phase_label = str((phase_spec or {}).get("label") or phase_key).upper()
    strength_used = str((phase_spec or {}).get("strength") or _normalize_turn_drive_strength(strength_source))
    setup_forward_cfg = _duration_range_snapshot(setup_forward_range_source)
    setup_forward_enabled = bool(
        isinstance(setup_forward_cfg, dict)
        and str((phase_spec or {}).get("drive_mode") or "").strip().lower() == "backward"
    )
    try:
        score_used = int(round(float(score_source)))
    except (TypeError, ValueError):
        score_used = int(DEFAULT_SCORE)
    score_used = max(1, min(100, int(score_used)))
    try:
        trial_count = max(1, int(trials_source))
    except (TypeError, ValueError):
        trial_count = 1
    try:
        duration_used_ms = max(1, int(round(float(duration_source))))
    except (TypeError, ValueError):
        duration_used_ms = 1
    settle_used_s = max(0.0, float(settle_s))
    try:
        reset_x_interval = max(0, int(reset_x_every_n_trials_source or 0))
    except (TypeError, ValueError):
        reset_x_interval = 0

    result = {
        "experiment_type": "close_dist_x_axis_one_act_curve",
        "objective": "repeat one combined turn+drive act and discover a distance-banded x-axis closure curve",
        "seconds": 0.0,
        "score_pct": int(score_used),
        "trials_requested": int(trial_count),
        "trials_completed": 0,
        "phase_duration_ms": int(duration_used_ms),
        "phase": str(phase_key),
        "label": str(phase_label),
        "turn_strength": str(strength_used),
        "distance_band_mm": _round_triplet(distance_band_mm),
        "reset_x_every_n_trials": int(reset_x_interval),
        "production_worthy_threshold_pct": _round_triplet(production_worthy_threshold_pct),
        "setup_forward_range_ms": (dict(setup_forward_cfg) if isinstance(setup_forward_cfg, dict) else None),
        "vision_mode": str(vision_mode or ""),
        "sequence": [dict(phase_spec or {})] if isinstance(phase_spec, dict) else [],
        "trial_manifest": {},
        "stop_reason": "not_started",
        "pulses": [],
        "trials": [],
        "x_resets": [],
        "observation": {
            "before": None,
            "before_meta": {},
            "after": None,
            "after_meta": {},
        },
        "summary": {
            "phase": {},
            "lookup_lines": [],
        },
        "curve_assessment": {
            "metric": "x_gap_close_rate_pct",
            "consistency_pct": None,
            "threshold_pct": _round_triplet(production_worthy_threshold_pct),
            "production_worthy": False,
            "usable_runs": 0,
            "motor_pair": {},
        },
        "analysis": {
            "observation_usable": False,
            "before_mode": "",
            "after_mode": "",
            "progress_before_pct": None,
            "progress_after_pct": None,
            "progress_delta_pct": None,
            "gate_error_before_mm": None,
            "gate_error_after_mm": None,
            "gate_error_delta_mm": {},
            "improved_metrics": [],
            "worsened_metrics": [],
            "primary_effect": "insufficient_observation",
            "notes": [],
        },
        "suggestion": {
            "action": "ENABLE_CUSTOM_ACTIONS",
            "reason": "robot_missing_custom_action_support",
        },
        "ok": False,
    }
    if not isinstance(phase_spec, dict):
        result["seconds"] = float(max(0.0, time.time() - started))
        result["stop_reason"] = "invalid_phase"
        result["suggestion"] = {
            "action": "SELECT_VALID_PHASE",
            "reason": "requested_phase_not_available",
        }
        return result
    if robot is None or not hasattr(robot, "send_custom_actions_pwm"):
        result["seconds"] = float(max(0.0, time.time() - started))
        result["stop_reason"] = "robot_missing_custom_action_support"
        return result

    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())
    result["vision_mode"] = str(vision_mode_used)
    if isinstance(provided_trial_manifest, dict) and _trial_manifest_trials_list(provided_trial_manifest):
        trial_manifest_used = _reset_trial_manifest_for_run(
            json.loads(json.dumps(provided_trial_manifest))
        )
    else:
        trial_manifest_used = build_close_dist_x_axis_one_act_trials_manifest(
            trials=int(trial_count),
            phase=str(phase_key),
            strength=str(strength_used),
            phase_duration_ms=int(duration_used_ms),
            score=int(score_used),
            sequence=sequence,
            setup_forward_range_ms=setup_forward_cfg,
            name=str(manifest_name),
        )
        trial_manifest_used = _reset_trial_manifest_for_run(
            json.loads(json.dumps(trial_manifest_used))
        )
    adaptive_turn_selection = _trial_manifest_uses_adaptive_turn_selection(trial_manifest_used)
    distance_hold_enabled = bool(setup_forward_enabled) and _trial_manifest_uses_distance_hold(trial_manifest_used)
    if bool(adaptive_turn_selection):
        curve_cfg_used = _trial_manifest_curve_config(trial_manifest_used)
        measured_drive_mode = str(
            ((curve_cfg_used.get("measured_phase") or {}).get("drive_mode") or (phase_spec or {}).get("drive_mode") or "")
        ).strip().lower()
        phase_key = f"adaptive_{measured_drive_mode or 'turn'}"
        phase_label = str(curve_cfg_used.get("label") or _adaptive_turn_label_for_drive_mode(measured_drive_mode)).upper()
        result["phase"] = str(phase_key)
        result["label"] = str(phase_label)
    result["trial_manifest"] = dict(trial_manifest_used or {})
    trials_manifest_path = _write_trials_manifest(trials_path, trial_manifest_used)
    if trials_manifest_path is not None:
        result["trials_path"] = str(trials_manifest_path)

    logger("[CLOSE X+DIST] Strategy: repeat one combined motion family and measure x-gap closure at the current distance.")
    logger(
        f"[CLOSE X+DIST] Vision backend: {str(vision_mode_used)}. "
        f"Trials={int(trial_count)} phase={int(duration_used_ms)}ms score={int(score_used)}% "
        f"phase={str(phase_label)} strength={str(strength_used)}."
    )
    if bool(adaptive_turn_selection):
        logger("[CLOSE X+DIST] Turn selection: if x_axis<0 turn RIGHT; otherwise turn LEFT.")
    if trials_manifest_path is not None:
        logger(f"[CLOSE X+DIST] Trial plan written to {str(trials_manifest_path)}.")
    if bool(setup_forward_enabled):
        setup_label = (
            _adaptive_turn_label_for_drive_mode("forward")
            if bool(adaptive_turn_selection)
            else str((
                _trial_manifest_setup_segment(
                    ((_trial_manifest_trials_list(trial_manifest_used) or [{}])[0]),
                    manifest=trial_manifest_used,
                )
                or {}
            ).get("label") or "FORWARD")
        )
        if bool(distance_hold_enabled):
            logger(
                "[CLOSE X+DIST] Distance hold: before each measured trial, compare current dist "
                "to the run-start dist. If current dist is smaller, send BACK; otherwise send "
                f"{str(setup_label).split('+')[0]} {int(setup_forward_cfg['min_ms'])}-{int(setup_forward_cfg['max_ms'])}ms, "
                "then reobserve and run the measured back+turn trial."
            )
        else:
            logger(
                "[CLOSE X+DIST] Trial staging: before each backward curve act, "
                f"send {str(setup_label)} {int(setup_forward_cfg['min_ms'])}-{int(setup_forward_cfg['max_ms'])}ms "
                "then reobserve and run the measured back+turn trial."
            )

    try:
        success_gates = _step_success_gates(world, EXPERIMENT_STEP)
        initial_pose, initial_meta = observe_alignment_pose(
            vision=vision,
            world=world,
            samples=int(observe_samples),
            timeout_s=float(observe_timeout_s),
            relaxed_timeout_s=float(relaxed_timeout_s),
            reobserve_rounds=int(reobserve_rounds),
            log_fn=logger,
        )
        result["observation"]["before"] = _pose_snapshot(initial_pose)
        result["observation"]["before_meta"] = dict(initial_meta or {})
        logger(f"[CLOSE X+DIST] Initial pose: {_pose_summary_text(initial_pose)}")

        final_pose = initial_pose
        final_meta = dict(initial_meta or {})
        stop_reason = "completed"
        if initial_pose is None:
            stop_reason = "no_initial_pose"
        else:
            current_pose = initial_pose
            current_meta = dict(initial_meta or {})
            run_start_dist_mm = _coerce_float((initial_pose or {}).get("dist"), None)
            trial_rows = []
            pulse_rows = []
            planned_trials = list(_trial_manifest_trials_list(trial_manifest_used) or [])
            for trial_index in range(1, int(trial_count) + 1):
                planned_trial = (
                    dict(planned_trials[int(trial_index) - 1])
                    if int(trial_index) <= len(planned_trials)
                    and isinstance(planned_trials[int(trial_index) - 1], dict)
                    else {}
                )
                planned_trial = _trial_manifest_normalize_row(
                    planned_trial,
                    manifest=trial_manifest_used,
                    trial_index=int(trial_index),
                    keep_results=True,
                )
                if _trial_manifest_row_is_usable_complete(planned_trial):
                    logger(
                        "[CLOSE X+DIST] Trial "
                        f"{int(trial_index)}/{int(trial_count)}: keeping existing usable result "
                        f"({str(planned_trial.get('curvePair') or '')})."
                    )
                    continue
                planned_forward_segment = _trial_manifest_setup_segment(
                    planned_trial,
                    manifest=trial_manifest_used,
                )
                planned_measured_segment = _trial_manifest_measured_segment(
                    planned_trial,
                    manifest=trial_manifest_used,
                )
                trial_row = {
                    "trial": int(trial_index),
                    "phase": str(phase_key),
                    "label": str(phase_label),
                    "cmd": str(phase_spec.get("cmd") or ""),
                    "drive_mode": str(phase_spec.get("drive_mode") or ""),
                    "profile_name": str(phase_spec.get("profile_name") or ""),
                    "strength": str(strength_used),
                    "selected_phase": "",
                    "selected_label": "",
                    "pre_pose": None,
                    "pre_meta": {},
                    "plan_row": dict(planned_trial),
                    "setup_forward": {
                        "attempted": False,
                        "phase": str((planned_forward_segment or {}).get("phase") or ""),
                        "label": str((planned_forward_segment or {}).get("label") or ""),
                        "cmd": str((planned_forward_segment or {}).get("cmd") or ""),
                        "drive_mode": str((planned_forward_segment or {}).get("drive_mode") or ""),
                        "pre_pose": None,
                        "pre_meta": {},
                        "pre_summary": "not_attempted",
                        "distance_hold_policy": "",
                        "distance_reference_mm": None,
                        "distance_before_mm": None,
                        "distance_delta_mm": None,
                        "turning": bool((planned_forward_segment or {}).get("turning")),
                        "duration_requested_ms": None,
                        "duration_requested_s": None,
                        "plan": None,
                        "motor_pair": {},
                        "send_result": {},
                        "post_pose": None,
                        "post_meta": {},
                        "post_summary": "not_attempted",
                        "observation_latency": {},
                    },
                    "plan": None,
                    "send_result": {},
                    "post_pose": None,
                    "post_meta": {},
                    "post_summary": "unavailable",
                    "observation_latency": {},
                    "delta_mm": {
                        "x_axis": None,
                        "y_axis": None,
                        "dist": None,
                    },
                    "curve_capture": {
                        "start_dist_mm": _round_triplet((current_pose or {}).get("dist")),
                        "end_dist_mm": None,
                        "start_x_offset_mm": _round_triplet((current_pose or {}).get("offset_x")),
                        "end_x_offset_mm": None,
                        "x_target_mm": _round_triplet(
                            _coerce_float((success_gates.get("xAxis_offset_abs") or {}).get("target"), None)
                        ),
                        "start_x_gap_mm": _metric_target_gap_abs(
                            (current_pose or {}).get("offset_x"),
                            success_gates.get("xAxis_offset_abs"),
                        ),
                        "end_x_gap_mm": None,
                        "x_gap_closed_mm": None,
                    },
                    "analysis": {
                        "observation_usable": False,
                        "primary_effect": "insufficient_observation",
                        "notes": [],
                    },
                    "next_suggestion": {
                        "action": "REOBSERVE",
                        "reason": "trial_not_started",
                    },
                    "observation_usable": False,
                    "recovery": {
                        "attempted": False,
                        "recovered": False,
                        "reason": "not_attempted",
                        "phase": "",
                        "label": "",
                        "plan": None,
                        "motor_pair": {},
                        "send_result": {},
                        "pose": None,
                        "meta": {},
                        "summary": "not_attempted",
                    },
                }
                if bool(setup_forward_enabled):
                    forward_duration_ms = int(
                        (
                            planned_trial.get("setupDurationMs")
                            or planned_trial.get("duration")
                            or (planned_forward_segment or {}).get("duration_ms")
                            or 0
                        )
                    )
                    selected_forward_segment = planned_forward_segment
                    if bool(adaptive_turn_selection):
                        setup_drive_mode = "forward"
                        if bool(distance_hold_enabled):
                            setup_drive_mode = _distance_hold_drive_mode(
                                (current_pose or {}).get("dist"),
                                reference_dist=run_start_dist_mm,
                                fallback_drive_mode=(planned_forward_segment or {}).get("drive_mode"),
                            )
                        segment_source = _trial_manifest_segment_for_drive_mode(
                            drive_mode=str(setup_drive_mode),
                            setup_segment=planned_forward_segment,
                            measured_segment=planned_measured_segment,
                        )
                        selected_forward_segment = _adaptive_turn_drive_segment(
                            drive_mode=str(setup_drive_mode),
                            strength=str(strength_used),
                            turn_dir=_turn_dir_for_x_axis(
                                (current_pose or {}).get("offset_x"),
                                fallback_cmd=(
                                    (segment_source or {}).get("cmd")
                                    or (planned_forward_segment or {}).get("cmd")
                                    or (planned_measured_segment or {}).get("cmd")
                                ),
                            ),
                            score=int(
                                _coerce_float((segment_source or {}).get("score_pct"), score_used)
                                or score_used
                            ),
                            duration_ms=int(forward_duration_ms),
                            stage="setup_distance_hold" if bool(distance_hold_enabled) else "setup_forward_turn",
                            profile_name=(str((segment_source or {}).get("profile_name") or "").strip() or None),
                            pwm_override=(segment_source or {}).get("pwm_override"),
                            metadata={
                                "trial": int(trial_index),
                                "stage": "setup_forward",
                                "policy": TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS,
                                "distance_hold_policy": (
                                    TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
                                    if bool(distance_hold_enabled)
                                    else ""
                                ),
                            },
                        )
                    if not isinstance(selected_forward_segment, dict):
                        selected_forward_segment = planned_forward_segment
                    setup_plan = _build_turn_drive_plan_from_manifest_segment(
                        selected_forward_segment,
                        metadata={"trial": int(trial_index), "stage": "setup_forward"},
                    )
                    if not isinstance(setup_plan, dict):
                        stop_reason = f"invalid_setup_plan_trial_{int(trial_index)}"
                        trial_rows.append(trial_row)
                        trial_manifest_used = _update_trial_manifest_trial_result(
                            trial_manifest_used,
                            trial_row,
                            stop_reason=stop_reason,
                        )
                        _write_trials_manifest(trials_path, trial_manifest_used)
                        logger(
                            f"[CLOSE X+DIST] Trial {int(trial_index)}: no valid setup turn+drive plan."
                        )
                        break
                    setup_motor_pair = dict((selected_forward_segment or {}).get("motor_pair") or {})
                    trial_row["setup_forward"]["phase"] = str((selected_forward_segment or {}).get("phase") or "")
                    trial_row["setup_forward"]["label"] = str((selected_forward_segment or {}).get("label") or "")
                    trial_row["setup_forward"]["cmd"] = str((selected_forward_segment or {}).get("cmd") or "")
                    trial_row["setup_forward"]["drive_mode"] = str((selected_forward_segment or {}).get("drive_mode") or "")
                    trial_row["setup_forward"]["distance_hold_policy"] = (
                        TRIAL_DISTANCE_HOLD_ADAPTIVE_BY_RUN_START_DIST
                        if bool(distance_hold_enabled)
                        else ""
                    )
                    trial_row["setup_forward"]["distance_reference_mm"] = _round_triplet(run_start_dist_mm)
                    trial_row["setup_forward"]["distance_before_mm"] = _round_triplet((current_pose or {}).get("dist"))
                    trial_row["setup_forward"]["distance_delta_mm"] = _round_triplet(
                        (
                            _coerce_float((current_pose or {}).get("dist"), None)
                            - run_start_dist_mm
                        )
                        if (
                            _coerce_float((current_pose or {}).get("dist"), None) is not None
                            and run_start_dist_mm is not None
                        )
                        else None
                    )
                    trial_row["setup_forward"]["turning"] = bool((selected_forward_segment or {}).get("turning"))
                    trial_row["setup_forward"]["attempted"] = True
                    trial_row["setup_forward"]["pre_pose"] = _pose_snapshot(current_pose)
                    trial_row["setup_forward"]["pre_meta"] = dict(current_meta or {})
                    trial_row["setup_forward"]["pre_summary"] = _pose_summary_text(current_pose)
                    trial_row["setup_forward"]["duration_requested_ms"] = int(forward_duration_ms or 0)
                    trial_row["setup_forward"]["duration_requested_s"] = _duration_ms_to_seconds(forward_duration_ms)
                    trial_row["setup_forward"]["plan"] = _turn_drive_plan_snapshot(setup_plan)
                    trial_row["setup_forward"]["motor_pair"] = dict(setup_motor_pair)
                    if bool(distance_hold_enabled):
                        logger(
                            f"[CLOSE X+DIST] Trial {int(trial_index)}/{int(trial_count)} distance hold: "
                            f"run_start_dist={_round_triplet(run_start_dist_mm)}mm "
                            f"current_dist={_round_triplet((current_pose or {}).get('dist'))}mm, "
                            f"so send {str((selected_forward_segment or {}).get('label') or 'SETUP')} "
                            f"{str(setup_motor_pair.get('display') or '')} "
                            f"for {int(forward_duration_ms or 0)}ms before {str(phase_label)}."
                        )
                    else:
                        logger(
                            f"[CLOSE X+DIST] Trial {int(trial_index)}/{int(trial_count)} staging: "
                            f"{str((selected_forward_segment or {}).get('label') or 'FORWARD')} "
                            f"{str(setup_motor_pair.get('display') or '')} "
                            f"for {int(forward_duration_ms or 0)}ms before {str(phase_label)}."
                        )
                    forward_send_result = send_robot_command(
                        robot,
                        world,
                        EXPERIMENT_STEP,
                        str(setup_plan.get("cmd") or ""),
                        speed=0.0,
                        speed_score=int(setup_plan.get("score") or score_used),
                        duration_override_ms=int(forward_duration_ms or 0),
                        auto_mode=False,
                        ease_in_out_enabled=False,
                        half_first_turn_pulse=False,
                        custom_action_specs=[dict(action) for action in list(setup_plan.get("actions") or [])],
                        action_note=str(setup_plan.get("action_note") or "").strip() or None,
                    )
                    trial_row["setup_forward"]["send_result"] = dict(forward_send_result or {})
                    pulse_rows.append(
                        {
                            "trial": int(trial_index),
                            "phase": str((selected_forward_segment or {}).get("phase") or "setup_forward"),
                            "label": str((selected_forward_segment or {}).get("label") or "SETUP+FORWARD"),
                            "strength": str(strength_used),
                            "cmd": str(setup_plan.get("cmd") or ""),
                            "drive_mode": str(setup_plan.get("drive_mode") or ""),
                            "motor_pair": dict(setup_motor_pair),
                            "send_result": dict(forward_send_result or {}),
                            "stage": "setup_forward",
                        }
                    )
                    if not isinstance(forward_send_result, dict):
                        stop_reason = f"send_failed_setup_forward_trial_{int(trial_index)}"
                        trial_rows.append(trial_row)
                        trial_manifest_used = _update_trial_manifest_trial_result(
                            trial_manifest_used,
                            trial_row,
                            stop_reason=stop_reason,
                        )
                        _write_trials_manifest(trials_path, trial_manifest_used)
                        logger(f"[CLOSE X+DIST] Trial {int(trial_index)} staging send failed.")
                        break
                    effective_forward_duration_ms = int(
                        (forward_send_result or {}).get("duration_ms")
                        or int(forward_duration_ms or 0)
                    )
                    setup_observe_not_before_ts = _post_act_observe_not_before_ts(
                        send_result=forward_send_result,
                        effective_duration_ms=effective_forward_duration_ms,
                        settle_s=float(settle_used_s),
                    )
                    setup_pose, setup_meta = observe_alignment_pose(
                        vision=vision,
                        world=world,
                        samples=int(observe_samples),
                        timeout_s=float(observe_timeout_s),
                        relaxed_timeout_s=float(relaxed_timeout_s),
                        reobserve_rounds=int(reobserve_rounds),
                        min_sample_time=setup_observe_not_before_ts,
                        log_fn=logger,
                    )
                    trial_row["setup_forward"]["post_pose"] = _pose_snapshot(setup_pose)
                    trial_row["setup_forward"]["post_meta"] = dict(setup_meta or {})
                    trial_row["setup_forward"]["post_summary"] = _pose_summary_text(setup_pose)
                    trial_row["setup_forward"]["observation_latency"] = _observation_latency_details(
                        send_result=forward_send_result,
                        pose=setup_pose,
                    )
                    logger(
                        "[CLOSE X+DIST] Trial "
                        f"{int(trial_index)} setup "
                        f"{_observation_latency_log_text(stage_label='', send_result=forward_send_result, pose=setup_pose, meta=setup_meta).strip()}"
                    )
                    if setup_pose is None:
                        stop_reason = f"lost_pose_after_setup_forward_trial_{int(trial_index)}"
                        trial_rows.append(trial_row)
                        final_pose = None
                        final_meta = dict(setup_meta or {})
                        trial_manifest_used = _update_trial_manifest_trial_result(
                            trial_manifest_used,
                            trial_row,
                            stop_reason=stop_reason,
                        )
                        _write_trials_manifest(trials_path, trial_manifest_used)
                        logger(
                            f"[CLOSE X+DIST] Trial {int(trial_index)} lost pose after setup forward."
                        )
                        break
                    current_pose = setup_pose
                    current_meta = dict(setup_meta or {})

                trial_row["pre_pose"] = _pose_snapshot(current_pose)
                trial_row["pre_meta"] = dict(current_meta or {})
                selected_measured_segment = planned_measured_segment
                measured_duration_ms = int(
                    ((planned_measured_segment or {}).get("duration_ms") or duration_used_ms)
                )
                if bool(adaptive_turn_selection):
                    measured_drive_mode = str(
                        ((planned_measured_segment or {}).get("drive_mode") or (phase_spec or {}).get("drive_mode") or "backward")
                    )
                    selected_measured_segment = _adaptive_turn_drive_segment(
                        drive_mode=str(measured_drive_mode),
                        strength=str(strength_used),
                        turn_dir=_turn_dir_for_x_axis(
                            (current_pose or {}).get("offset_x"),
                            fallback_cmd=(planned_measured_segment or {}).get("cmd") or (phase_spec or {}).get("cmd"),
                        ),
                        score=int(
                            _coerce_float((planned_measured_segment or {}).get("score_pct"), score_used)
                            or score_used
                        ),
                        duration_ms=int(measured_duration_ms),
                        stage="measured_backward_turn" if bool(setup_forward_enabled) else "measured_turn_drive",
                        profile_name=(str((planned_measured_segment or {}).get("profile_name") or "").strip() or None),
                        pwm_override=(planned_measured_segment or {}).get("pwm_override"),
                        metadata={"trial": int(trial_index), "phase": str(phase_key), "policy": TRIAL_TURN_SELECTION_ADAPTIVE_BY_X_AXIS},
                    )
                if not isinstance(selected_measured_segment, dict):
                    selected_measured_segment = planned_measured_segment
                trial_row["selected_phase"] = str((selected_measured_segment or {}).get("phase") or "")
                trial_row["selected_label"] = str((selected_measured_segment or {}).get("label") or phase_label)
                trial_row["cmd"] = str((selected_measured_segment or {}).get("cmd") or phase_spec.get("cmd") or "")
                trial_row["drive_mode"] = str((selected_measured_segment or {}).get("drive_mode") or phase_spec.get("drive_mode") or "")
                trial_row["profile_name"] = str((selected_measured_segment or {}).get("profile_name") or phase_spec.get("profile_name") or "")
                plan = _build_turn_drive_plan_from_manifest_segment(
                    selected_measured_segment,
                    metadata={"trial": int(trial_index), "phase": str(phase_key)},
                )
                if not isinstance(plan, dict):
                    plan = build_turn_drive_motion_plan(
                        cmd=str((selected_measured_segment or {}).get("cmd") or phase_spec.get("cmd") or ""),
                        score=int(score_used),
                        hold_duration_ms=int(measured_duration_ms),
                        profile_name=str((selected_measured_segment or {}).get("profile_name") or phase_spec.get("profile_name") or ""),
                        drive_mode=str((selected_measured_segment or {}).get("drive_mode") or phase_spec.get("drive_mode") or ""),
                        metadata={"trial": int(trial_index), "phase": str(phase_key)},
                    )
                if not isinstance(plan, dict):
                    stop_reason = "invalid_plan"
                    trial_rows.append(trial_row)
                    trial_manifest_used = _update_trial_manifest_trial_result(
                        trial_manifest_used,
                        trial_row,
                        stop_reason=stop_reason,
                    )
                    _write_trials_manifest(trials_path, trial_manifest_used)
                    logger(
                        f"[CLOSE X+DIST] Trial {int(trial_index)}: no valid turn+drive plan from robot model."
                    )
                    break

                trial_row["plan"] = _turn_drive_plan_snapshot(plan)
                motor_pair = _turn_drive_motor_pair_summary(trial_row.get("plan"))
                measured_label = str(trial_row.get("selected_label") or phase_label)
                measured_phase_for_pulse = str(trial_row.get("selected_phase") or phase_key)
                logger(
                    f"[CLOSE X+DIST] Trial {int(trial_index)}/{int(trial_count)}: "
                    f"{str(measured_label)} {str(strength_used)} {str(motor_pair.get('display') or '')} "
                    f"for {int(measured_duration_ms)}ms."
                )
                send_result = send_robot_command(
                    robot,
                    world,
                    EXPERIMENT_STEP,
                    str(plan.get("cmd") or ""),
                    speed=0.0,
                    speed_score=int(plan.get("score") or score_used),
                    duration_override_ms=int(plan.get("duration_ms") or duration_used_ms),
                    auto_mode=False,
                    ease_in_out_enabled=False,
                    half_first_turn_pulse=False,
                    custom_action_specs=[dict(action) for action in list(plan.get("actions") or [])],
                    action_note=str(plan.get("action_note") or "").strip() or None,
                )
                trial_row["send_result"] = dict(send_result or {})
                pulse_rows.append(
                    {
                        "trial": int(trial_index),
                        "phase": str(measured_phase_for_pulse),
                        "label": str(measured_label),
                        "strength": str(strength_used),
                        "cmd": str(plan.get("cmd") or ""),
                        "drive_mode": str(plan.get("drive_mode") or ""),
                        "motor_pair": motor_pair,
                        "send_result": dict(send_result or {}),
                    }
                )
                if not isinstance(send_result, dict):
                    stop_reason = f"send_failed_trial_{int(trial_index)}"
                    trial_rows.append(trial_row)
                    trial_manifest_used = _update_trial_manifest_trial_result(
                        trial_manifest_used,
                        trial_row,
                        stop_reason=stop_reason,
                    )
                    _write_trials_manifest(trials_path, trial_manifest_used)
                    logger(f"[CLOSE X+DIST] Trial {int(trial_index)}: send failed.")
                    break

                effective_duration_ms = int(
                    (send_result or {}).get("duration_ms")
                    or int(plan.get("duration_ms") or duration_used_ms)
                )
                measured_observe_not_before_ts = _post_act_observe_not_before_ts(
                    send_result=send_result,
                    effective_duration_ms=effective_duration_ms,
                    settle_s=float(settle_used_s),
                )

                post_pose, post_meta = observe_alignment_pose(
                    vision=vision,
                    world=world,
                    samples=int(observe_samples),
                    timeout_s=float(observe_timeout_s),
                    relaxed_timeout_s=float(relaxed_timeout_s),
                    reobserve_rounds=int(reobserve_rounds),
                    min_sample_time=measured_observe_not_before_ts,
                    log_fn=logger,
                )
                trial_row["post_pose"] = _pose_snapshot(post_pose)
                trial_row["post_meta"] = dict(post_meta or {})
                trial_row["post_summary"] = _pose_summary_text(post_pose)
                trial_row["observation_latency"] = _observation_latency_details(
                    send_result=send_result,
                    pose=post_pose,
                )
                trial_row["delta_mm"] = {
                    "x_axis": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "offset_x"), absolute=True),
                    "y_axis": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "offset_y"), absolute=True),
                    "dist": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "dist"), absolute=True),
                }
                trial_row["curve_capture"] = _alternating_phase_curve_capture(
                    before_pose=current_pose,
                    after_pose=post_pose,
                    success_gates=success_gates,
                )
                trial_analysis, next_suggestion = analyze_iteration(
                    world=world,
                    before_pose=current_pose,
                    after_pose=post_pose,
                    before_meta=current_meta,
                    after_meta=post_meta,
                )
                trial_row["analysis"] = _mm_payload_non_negative(dict(trial_analysis or {}))
                trial_row["next_suggestion"] = dict(next_suggestion or {})
                trial_row["observation_usable"] = bool((trial_analysis or {}).get("observation_usable"))
                trial_rows.append(trial_row)
                trial_manifest_used = _update_trial_manifest_trial_result(
                    trial_manifest_used,
                    trial_row,
                    stop_reason=stop_reason,
                )
                _write_trials_manifest(trials_path, trial_manifest_used)
                logger(
                    "[CLOSE X+DIST] Trial "
                    f"{int(trial_index)} measured "
                    f"{_observation_latency_log_text(stage_label='', send_result=send_result, pose=post_pose, meta=post_meta).strip()}"
                )
                logger(
                    "[CLOSE X+DIST] Trial "
                    f"{int(trial_index)} delta: "
                    f"x={trial_row['delta_mm']['x_axis']}mm "
                    f"dist={trial_row['delta_mm']['dist']}mm "
                    f"x_gap_closed={trial_row['curve_capture']['x_gap_closed_mm']}mm "
                    f"start_dist={trial_row['curve_capture']['start_dist_mm']}mm."
                )

                if post_pose is None:
                    recovery_row, recovered_pose, recovered_meta = _run_inverse_turn_drive_recovery(
                        robot=robot,
                        world=world,
                        vision=vision,
                        prior_plan=plan,
                        trial_index=int(trial_index),
                        phase_label=str(trial_row.get("selected_label") or phase_label),
                        observe_samples=int(observe_samples),
                        observe_timeout_s=float(observe_timeout_s),
                        relaxed_timeout_s=float(relaxed_timeout_s),
                        reobserve_rounds=int(reobserve_rounds),
                        settle_s=float(settle_used_s),
                        log_fn=logger,
                    )
                    trial_row["recovery"] = dict(recovery_row or {})
                    if bool((recovery_row or {}).get("attempted")):
                        pulse_rows.append(
                            {
                                "trial": int(trial_index),
                                "phase": str((recovery_row or {}).get("phase") or ""),
                                "label": str((recovery_row or {}).get("label") or ""),
                                "strength": str(strength_used),
                                "cmd": str(((recovery_row or {}).get("plan") or {}).get("cmd") or ""),
                                "drive_mode": str(((recovery_row or {}).get("plan") or {}).get("drive_mode") or ""),
                                "motor_pair": dict((recovery_row or {}).get("motor_pair") or {}),
                                "send_result": dict((recovery_row or {}).get("send_result") or {}),
                                "stage": "inverse_recovery",
                            }
                        )
                    if recovered_pose is None:
                        stop_reason = f"lost_pose_after_trial_{int(trial_index)}"
                        final_pose = None
                        final_meta = dict(recovered_meta or post_meta or {})
                        trial_manifest_used = _update_trial_manifest_trial_result(
                            trial_manifest_used,
                            trial_row,
                            stop_reason=stop_reason,
                        )
                        _write_trials_manifest(trials_path, trial_manifest_used)
                        break
                    logger(
                        f"[CLOSE X+DIST] Trial {int(trial_index)} inverse recovery pose: "
                        f"{_pose_summary_text(recovered_pose)}. Continuing."
                    )
                    trial_manifest_used = _update_trial_manifest_trial_result(
                        trial_manifest_used,
                        trial_row,
                        stop_reason=stop_reason,
                    )
                    _write_trials_manifest(trials_path, trial_manifest_used)
                    current_pose = recovered_pose
                    current_meta = dict(recovered_meta or {})
                    final_pose = recovered_pose
                    final_meta = dict(recovered_meta or {})
                    continue

                current_pose = post_pose
                current_meta = dict(post_meta or {})
                final_pose = post_pose
                final_meta = dict(post_meta or {})

                if (
                    int(reset_x_interval) > 0
                    and trial_index % int(reset_x_interval) == 0
                    and trial_index < trial_count
                ):
                    logger(
                        f"[CLOSE X+DIST] Batch reset after trial {trial_index}: "
                        "running x-axis lock to reset x to 0 before next batch."
                    )
                    reset_result = run_x_axis_lock_experiment(
                        robot=robot,
                        world=world,
                        vision=vision,
                        vision_mode=vision_mode_used,
                        observe_samples=int(observe_samples),
                        initial_observe_timeout_s=float(observe_timeout_s),
                        initial_relaxed_timeout_s=float(relaxed_timeout_s),
                        settle_s=float(settle_used_s),
                        log_path=None,
                        memory_path=None,
                        log_fn=logger,
                    )
                    result["x_resets"].append({
                        "after_trial": int(trial_index),
                        "stop_reason": str(reset_result.get("stop_reason") or ""),
                        "cycles_completed": int(reset_result.get("cycles_completed") or 0),
                        "x_before_mm": _round_triplet(
                            ((reset_result.get("observation") or {}).get("before") or {}).get("offset_x")
                        ),
                        "x_after_mm": _round_triplet(
                            ((reset_result.get("observation") or {}).get("after") or {}).get("offset_x")
                        ),
                    })
                    reset_pose, reset_meta = observe_alignment_pose(
                        vision=vision,
                        world=world,
                        samples=int(observe_samples),
                        timeout_s=float(observe_timeout_s),
                        relaxed_timeout_s=float(relaxed_timeout_s),
                        reobserve_rounds=int(reobserve_rounds),
                        log_fn=logger,
                    )
                    if reset_pose is not None:
                        current_pose = reset_pose
                        current_meta = dict(reset_meta or {})
                        final_pose = reset_pose
                        final_meta = dict(reset_meta or {})
                        logger(
                            f"[CLOSE X+DIST] Batch reset complete. "
                            f"New pose: {_pose_summary_text(reset_pose)}"
                        )
                    else:
                        logger("[CLOSE X+DIST] Batch reset: lost pose after x-lock. Stopping.")
                        stop_reason = f"lost_pose_after_x_reset_trial_{trial_index}"
                        break

            result["pulses"] = pulse_rows
            result["trials"] = trial_rows
            result["trials_completed"] = int(len(trial_rows))
            phase_summary = _summarize_alternating_phase_runs(
                trial_rows,
                distance_band_mm=float(distance_band_mm),
            )
            result["summary"] = {
                "phase": phase_summary,
                "lookup_lines": list(
                    (
                        (phase_summary.get("curve_reference") or {}).get("lookup_lines")
                        if isinstance(phase_summary.get("curve_reference"), dict)
                        else []
                    )
                    or []
                ),
            }
            result["curve_assessment"] = _curve_assessment_from_phase_summary(
                phase_summary,
                threshold_pct=float(production_worthy_threshold_pct),
            )
            for line in list(result["summary"].get("lookup_lines") or []):
                logger(f"[CLOSE X+DIST] Curve: {str(line)}")

        overall_analysis, _ = analyze_iteration(
            world=world,
            before_pose=initial_pose,
            after_pose=final_pose,
            before_meta=initial_meta,
            after_meta=final_meta,
        )
        result["analysis"] = _mm_payload_non_negative(dict(overall_analysis or {}))
        result["observation"]["after"] = _pose_snapshot(final_pose)
        result["observation"]["after_meta"] = dict(final_meta or {})
        if str(stop_reason) == "no_initial_pose":
            result["suggestion"] = {
                "action": "REOBSERVE",
                "reason": "no_initial_pose",
            }
        elif bool((result.get("curve_assessment") or {}).get("production_worthy")):
            result["suggestion"] = {
                "action": "PROMOTE_CURVE",
                "reason": "curve_meets_consistency_threshold",
                "consistency_pct": (result.get("curve_assessment") or {}).get("consistency_pct"),
                "threshold_pct": (result.get("curve_assessment") or {}).get("threshold_pct"),
            }
        else:
            result["suggestion"] = {
                "action": "RETUNE_DURATION_OR_STRENGTH",
                "reason": "curve_below_consistency_threshold",
                "consistency_pct": (result.get("curve_assessment") or {}).get("consistency_pct"),
                "threshold_pct": (result.get("curve_assessment") or {}).get("threshold_pct"),
            }
        result["stop_reason"] = str(stop_reason)
        result["seconds"] = float(max(0.0, time.time() - started))
        trial_manifest_used = _finalize_trial_manifest_after_run(
            trial_manifest_used,
            result,
        )
        manifest_curve_stats = (
            (trial_manifest_used or {}).get("curve_stats")
            if isinstance((trial_manifest_used or {}).get("curve_stats"), dict)
            else {}
        )
        result["curve_assessment"]["consistency_pct"] = manifest_curve_stats.get("consistency_score_pct")
        result["curve_assessment"]["usable_runs"] = int(manifest_curve_stats.get("usable_trials") or 0)
        result["curve_assessment"]["production_worthy"] = (
            bool(manifest_curve_stats.get("production_worthy"))
            if manifest_curve_stats.get("production_worthy") in (True, False)
            else False
        )
        completed_total = _trial_manifest_trial_rows_completed_count(
            _trial_manifest_trials_list(trial_manifest_used)
        )
        result["trials_completed"] = int(completed_total)
        result["ok"] = bool(
            str(stop_reason) == "completed"
            and int(completed_total) >= int(trial_count)
        )
        result["trial_manifest"] = dict(trial_manifest_used or {})
        trials_manifest_path = _write_trials_manifest(trials_path, trial_manifest_used)
        if trials_manifest_path is not None:
            result["trials_path"] = str(trials_manifest_path)
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def run_alternating_turn_drive_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    score: int = DEFAULT_SCORE,
    cycles: int = DEFAULT_ALTERNATING_TURN_DRIVE_CYCLES,
    phase_duration_ms: int = DEFAULT_ALTERNATING_TURN_DRIVE_PHASE_DURATION_MS,
    strength: str = DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
    distance_band_mm: float = DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
    sequence=DEFAULT_ALTERNATING_TURN_DRIVE_SEQUENCE,
    settle_s: float = DEFAULT_SETTLE_S,
    observe_samples: int = DEFAULT_OBSERVE_SAMPLES,
    observe_timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    reobserve_rounds: int = DEFAULT_REOBSERVE_ROUNDS,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    world.step_state = EXPERIMENT_STEP
    sequence_used = _normalize_turn_drive_sequence(sequence, strength=str(strength or ""))
    strength_used = (
        str(sequence_used[0].get("strength") or DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH)
        if sequence_used
        else str(DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH)
    )
    try:
        score_used = int(round(float(score)))
    except (TypeError, ValueError):
        score_used = int(DEFAULT_SCORE)
    score_used = max(1, min(100, int(score_used)))
    cycle_count = max(1, int(cycles))
    duration_used_ms = max(1, int(round(float(phase_duration_ms))))
    settle_used_s = max(0.0, float(settle_s))

    result = {
        "experiment_type": "alternating_turn_drive_consistency",
        "objective": "repeat slow backward-left then forward-right turn+drive acts and measure phase consistency",
        "seconds": 0.0,
        "score_pct": int(score_used),
        "cycles_requested": int(cycle_count),
        "cycles_completed": 0,
        "phase_duration_ms": int(duration_used_ms),
        "turn_strength": str(strength_used),
        "distance_band_mm": _round_triplet(distance_band_mm),
        "sequence": [dict(row) for row in sequence_used],
        "vision_mode": str(vision_mode or ""),
        "stop_reason": "not_started",
        "pulses": [],
        "phases": [],
        "cycles": [],
        "observation": {
            "before": None,
            "before_meta": {},
            "after": None,
            "after_meta": {},
        },
        "summary": {
            "phases": {},
            "cycle_net": {},
            "lookup_lines": [],
        },
        "analysis": {
            "observation_usable": False,
            "before_mode": "",
            "after_mode": "",
            "progress_before_pct": None,
            "progress_after_pct": None,
            "progress_delta_pct": None,
            "gate_error_before_mm": None,
            "gate_error_after_mm": None,
            "gate_error_delta_mm": {},
            "improved_metrics": [],
            "worsened_metrics": [],
            "primary_effect": "insufficient_observation",
            "notes": [],
        },
        "suggestion": {
            "action": "ENABLE_CUSTOM_ACTIONS",
            "reason": "robot_missing_custom_action_support",
        },
        "ok": False,
    }
    if robot is None or not hasattr(robot, "send_custom_actions_pwm"):
        result["seconds"] = float(max(0.0, time.time() - started))
        result["stop_reason"] = "robot_missing_custom_action_support"
        return result

    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())
    result["vision_mode"] = str(vision_mode_used)

    logger("[ALT TURN+DRIVE] Strategy: repeat fixed combined motion families and evaluate consistency.")
    logger(
        f"[ALT TURN+DRIVE] Vision backend: {str(vision_mode_used)}. "
        f"Cycles={int(cycle_count)} phase={int(duration_used_ms)}ms score={int(score_used)}% "
        f"strength={str(strength_used)}."
    )
    logger(
        "[ALT TURN+DRIVE] Sequence: "
        + " -> ".join(str(row.get("label") or row.get("phase") or "").upper() for row in sequence_used)
        + "."
    )

    try:
        success_gates = _step_success_gates(world, EXPERIMENT_STEP)
        initial_pose, initial_meta = observe_alignment_pose(
            vision=vision,
            world=world,
            samples=int(observe_samples),
            timeout_s=float(observe_timeout_s),
            relaxed_timeout_s=float(relaxed_timeout_s),
            reobserve_rounds=int(reobserve_rounds),
            log_fn=logger,
        )
        result["observation"]["before"] = _pose_snapshot(initial_pose)
        result["observation"]["before_meta"] = dict(initial_meta or {})
        logger(f"[ALT TURN+DRIVE] Initial pose: {_pose_summary_text(initial_pose)}")

        final_pose = initial_pose
        final_meta = dict(initial_meta or {})
        phase_summary_by_name = {}
        stop_reason = "completed"

        if initial_pose is None:
            stop_reason = "no_initial_pose"
        else:
            current_pose = initial_pose
            current_meta = dict(initial_meta or {})
            phase_rows = []
            cycle_rows = []
            pulse_rows = []
            phase_index = 0

            for cycle_index in range(1, int(cycle_count) + 1):
                cycle_row = {
                    "cycle": int(cycle_index),
                    "start_pose": _pose_snapshot(current_pose),
                    "start_meta": dict(current_meta or {}),
                    "phases": [],
                    "observation_usable": False,
                    "net_delta_mm": {
                        "x_axis": None,
                        "y_axis": None,
                        "dist": None,
                    },
                }
                logger(
                    "[ALT TURN+DRIVE] Cycle "
                    f"{int(cycle_index)}/{int(cycle_count)} start: {_pose_summary_text(current_pose)}"
                )
                for spec in sequence_used:
                    phase_index += 1
                    phase_key = str(spec.get("phase") or "")
                    phase_label = str(spec.get("label") or phase_key).upper()
                    plan = build_turn_drive_motion_plan(
                        cmd=str(spec.get("cmd") or ""),
                        score=int(score_used),
                        hold_duration_ms=int(duration_used_ms),
                        profile_name=str(spec.get("profile_name") or ""),
                        drive_mode=str(spec.get("drive_mode") or ""),
                        metadata={"cycle": int(cycle_index), "phase": str(phase_key)},
                    )
                    phase_row = {
                        "cycle": int(cycle_index),
                        "phase_index": int(phase_index),
                        "phase": str(phase_key),
                        "label": str(phase_label),
                        "cmd": str(spec.get("cmd") or ""),
                        "drive_mode": str(spec.get("drive_mode") or ""),
                        "profile_name": str(spec.get("profile_name") or ""),
                        "strength": str(spec.get("strength") or ""),
                        "pre_pose": _pose_snapshot(current_pose),
                        "pre_meta": dict(current_meta or {}),
                        "plan": None,
                        "send_result": {},
                        "post_pose": None,
                        "post_meta": {},
                        "post_summary": "unavailable",
                        "delta_mm": {
                            "x_axis": None,
                            "y_axis": None,
                            "dist": None,
                        },
                        "curve_capture": {
                            "start_dist_mm": _round_triplet((current_pose or {}).get("dist")),
                            "end_dist_mm": None,
                            "start_x_offset_mm": _round_triplet((current_pose or {}).get("offset_x")),
                            "end_x_offset_mm": None,
                            "x_target_mm": _round_triplet(
                                _coerce_float((success_gates.get("xAxis_offset_abs") or {}).get("target"), None)
                            ),
                            "start_x_gap_mm": _metric_target_gap_abs(
                                (current_pose or {}).get("offset_x"),
                                success_gates.get("xAxis_offset_abs"),
                            ),
                            "end_x_gap_mm": None,
                            "x_gap_closed_mm": None,
                        },
                        "analysis": {
                            "observation_usable": False,
                            "primary_effect": "insufficient_observation",
                            "notes": [],
                        },
                        "next_suggestion": {
                            "action": "REOBSERVE",
                            "reason": "phase_not_started",
                        },
                        "observation_usable": False,
                    }
                    if not isinstance(plan, dict):
                        stop_reason = f"invalid_plan_{str(phase_key)}"
                        cycle_row["phases"].append(phase_row)
                        phase_rows.append(phase_row)
                        logger(
                            f"[ALT TURN+DRIVE] Cycle {int(cycle_index)} {str(phase_label)}: "
                            "no valid turn+drive plan from robot model."
                        )
                        break

                    phase_row["plan"] = {
                        "cmd": str(plan.get("cmd") or ""),
                        "score": int(plan.get("score") or 0),
                        "duration_ms": int(plan.get("duration_ms") or 0),
                        "profile_name": str(plan.get("profile_name") or ""),
                        "drive_mode": str(plan.get("drive_mode") or ""),
                        "inner_pwm": int(plan.get("inner_pwm") or 0),
                        "outer_pwm": int(plan.get("outer_pwm") or 0),
                        "action_note": str(plan.get("action_note") or ""),
                        "actions": [dict(action) for action in list(plan.get("actions") or [])],
                    }
                    logger(
                        "[ALT TURN+DRIVE] Cycle "
                        f"{int(cycle_index)} phase {str(phase_label)}: "
                        f"{int(score_used)}% for {int(duration_used_ms)}ms via {str(plan.get('profile_name') or '')} "
                        f"({str(spec.get('strength') or '')})."
                    )
                    send_result = send_robot_command(
                        robot,
                        world,
                        EXPERIMENT_STEP,
                        str(plan.get("cmd") or ""),
                        speed=0.0,
                        speed_score=int(plan.get("score") or score_used),
                        duration_override_ms=int(plan.get("duration_ms") or duration_used_ms),
                        auto_mode=False,
                        ease_in_out_enabled=False,
                        half_first_turn_pulse=False,
                        custom_action_specs=[dict(action) for action in list(plan.get("actions") or [])],
                        action_note=str(plan.get("action_note") or "").strip() or None,
                    )
                    phase_row["send_result"] = dict(send_result or {})
                    pulse_rows.append(
                        {
                            "cycle": int(cycle_index),
                            "phase_index": int(phase_index),
                            "phase": str(phase_key),
                            "label": str(phase_label),
                            "cmd": str(plan.get("cmd") or ""),
                            "drive_mode": str(plan.get("drive_mode") or ""),
                            "strength": str(spec.get("strength") or ""),
                            "send_result": dict(send_result or {}),
                        }
                    )
                    if not isinstance(send_result, dict):
                        stop_reason = f"send_failed_{str(phase_key)}"
                        cycle_row["phases"].append(phase_row)
                        phase_rows.append(phase_row)
                        logger(
                            f"[ALT TURN+DRIVE] Cycle {int(cycle_index)} {str(phase_label)}: send failed."
                        )
                        break

                    effective_duration_ms = int(
                        (send_result or {}).get("duration_ms")
                        or int(plan.get("duration_ms") or duration_used_ms)
                    )
                    time.sleep(max(float(settle_used_s), (float(effective_duration_ms) / 1000.0) + 0.05))

                    post_pose, post_meta = observe_alignment_pose(
                        vision=vision,
                        world=world,
                        samples=int(observe_samples),
                        timeout_s=float(observe_timeout_s),
                        relaxed_timeout_s=float(relaxed_timeout_s),
                        reobserve_rounds=int(reobserve_rounds),
                        log_fn=logger,
                    )
                    phase_row["post_pose"] = _pose_snapshot(post_pose)
                    phase_row["post_meta"] = dict(post_meta or {})
                    phase_row["post_summary"] = _pose_summary_text(post_pose)
                    phase_row["delta_mm"] = {
                        "x_axis": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "offset_x"), absolute=True),
                        "y_axis": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "offset_y"), absolute=True),
                        "dist": _trial_result_number(_pose_metric_delta(current_pose, post_pose, "dist"), absolute=True),
                    }
                    phase_row["curve_capture"] = _alternating_phase_curve_capture(
                        before_pose=current_pose,
                        after_pose=post_pose,
                        success_gates=success_gates,
                    )
                    phase_analysis, next_suggestion = analyze_iteration(
                        world=world,
                        before_pose=current_pose,
                        after_pose=post_pose,
                        before_meta=current_meta,
                        after_meta=post_meta,
                    )
                    phase_row["analysis"] = _mm_payload_non_negative(dict(phase_analysis or {}))
                    phase_row["next_suggestion"] = dict(next_suggestion or {})
                    phase_row["observation_usable"] = bool((phase_analysis or {}).get("observation_usable"))
                    cycle_row["phases"].append(phase_row)
                    phase_rows.append(phase_row)
                    logger(
                        "[ALT TURN+DRIVE] Cycle "
                        f"{int(cycle_index)} {str(phase_label)} delta: "
                        f"x={phase_row['delta_mm']['x_axis']}mm "
                        f"y={phase_row['delta_mm']['y_axis']}mm "
                        f"dist={phase_row['delta_mm']['dist']}mm "
                        f"x_gap_closed={phase_row['curve_capture']['x_gap_closed_mm']}mm "
                        f"start_dist={phase_row['curve_capture']['start_dist_mm']}mm."
                    )

                    if post_pose is None:
                        stop_reason = f"lost_pose_after_{str(phase_key)}"
                        final_pose = None
                        final_meta = dict(post_meta or {})
                        break

                    current_pose = post_pose
                    current_meta = dict(post_meta or {})
                    final_pose = post_pose
                    final_meta = dict(post_meta or {})

                cycle_row["end_pose"] = _pose_snapshot(final_pose)
                cycle_row["end_meta"] = dict(final_meta or {})
                cycle_row["net_delta_mm"] = {
                    "x_axis": _trial_result_number(_pose_metric_delta(cycle_row.get("start_pose"), cycle_row.get("end_pose"), "offset_x"), absolute=True),
                    "y_axis": _trial_result_number(_pose_metric_delta(cycle_row.get("start_pose"), cycle_row.get("end_pose"), "offset_y"), absolute=True),
                    "dist": _trial_result_number(_pose_metric_delta(cycle_row.get("start_pose"), cycle_row.get("end_pose"), "dist"), absolute=True),
                }
                cycle_row["observation_usable"] = bool(
                    cycle_row.get("start_pose") is not None and cycle_row.get("end_pose") is not None
                )
                cycle_rows.append(cycle_row)
                if stop_reason != "completed":
                    break
                logger(
                    "[ALT TURN+DRIVE] Cycle "
                    f"{int(cycle_index)}/{int(cycle_count)} net delta: "
                    f"x={cycle_row['net_delta_mm']['x_axis']}mm "
                    f"y={cycle_row['net_delta_mm']['y_axis']}mm "
                    f"dist={cycle_row['net_delta_mm']['dist']}mm."
                )

            result["pulses"] = pulse_rows
            result["phases"] = phase_rows
            result["cycles"] = cycle_rows
            result["cycles_completed"] = int(
                sum(1 for row in cycle_rows if isinstance(row, dict) and row.get("phases"))
            )
            for spec in sequence_used:
                phase_key = str(spec.get("phase") or "")
                phase_summary_by_name[phase_key] = _summarize_alternating_phase_runs(
                    [
                        row
                        for row in phase_rows
                        if isinstance(row, dict) and str(row.get("phase") or "") == phase_key
                    ],
                    distance_band_mm=float(distance_band_mm),
                )
            result["summary"] = {
                "phases": phase_summary_by_name,
                "cycle_net": _summarize_alternating_cycle_nets(cycle_rows),
                "lookup_lines": _collect_alternating_curve_lookup_lines(phase_summary_by_name),
            }

        overall_analysis, _ = analyze_iteration(
            world=world,
            before_pose=initial_pose,
            after_pose=final_pose,
            before_meta=initial_meta,
            after_meta=final_meta,
        )
        result["analysis"] = _mm_payload_non_negative(dict(overall_analysis or {}))
        result["observation"]["after"] = _pose_snapshot(final_pose)
        result["observation"]["after_meta"] = dict(final_meta or {})
        result["suggestion"] = _alternating_turn_drive_suggestion(
            phase_summary_by_name=(result.get("summary") or {}).get("phases") or {},
            cycle_net_summary=(result.get("summary") or {}).get("cycle_net") or {},
        )
        result["stop_reason"] = str(stop_reason)
        result["seconds"] = float(max(0.0, time.time() - started))
        result["ok"] = bool(
            str(stop_reason) == "completed"
            and int(result.get("cycles_completed") or 0) >= int(cycle_count)
        )
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def run_observe_while_moving_experiment(
    *,
    robot,
    world,
    vision=None,
    vision_mode: str | None = None,
    yolo_model_path: str | None = None,
    score: int = DEFAULT_SCORE,
    phase_duration_ms: int = DEFAULT_MOVING_OBSERVE_DURATION_MS,
    sample_hz: float = DEFAULT_MOVING_OBSERVE_SAMPLE_HZ,
    trials: int = DEFAULT_MOVING_OBSERVE_TRIALS,
    cmd_sequence: tuple[str, ...] = DEFAULT_MOVING_OBSERVE_SEQUENCE,
    group_by_axis: bool = DEFAULT_MOVING_OBSERVE_GROUP_BY_AXIS,
    observe_samples: int = DEFAULT_OBSERVE_SAMPLES,
    observe_timeout_s: float = DEFAULT_OBSERVE_TIMEOUT_S,
    relaxed_timeout_s: float = DEFAULT_RELAXED_TIMEOUT_S,
    reobserve_rounds: int = DEFAULT_REOBSERVE_ROUNDS,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    run_started_monotonic = time.monotonic()
    world.step_state = EXPERIMENT_STEP
    created_vision = False
    if vision is None:
        vision, vision_mode_used = _build_experiment_vision(
            mode=vision_mode,
            yolo_model_path=yolo_model_path,
        )
        created_vision = True
    else:
        if vision_mode is None:
            vision_mode_used = (
                VISION_MODE_ARUCO if isinstance(vision, ArucoBrickVision) else VISION_MODE_CYAN
            )
        else:
            vision_mode_used = normalize_vision_mode(vision_mode, fallback=active_vision_mode())

    try:
        score_used = int(round(float(score)))
    except (TypeError, ValueError):
        score_used = int(DEFAULT_SCORE)
    score_used = max(1, min(100, int(score_used)))
    try:
        phase_duration_used_ms = int(round(float(phase_duration_ms)))
    except (TypeError, ValueError):
        phase_duration_used_ms = int(DEFAULT_MOVING_OBSERVE_DURATION_MS)
    phase_duration_used_ms = max(1, int(phase_duration_used_ms))
    sample_hz_used = max(1.0, float(_coerce_float(sample_hz, DEFAULT_MOVING_OBSERVE_SAMPLE_HZ) or DEFAULT_MOVING_OBSERVE_SAMPLE_HZ))
    sample_period_s = 1.0 / float(sample_hz_used)
    trial_count = max(1, int(trials))
    seq = [str(cmd or "").strip().lower() for cmd in tuple(cmd_sequence or DEFAULT_MOVING_OBSERVE_SEQUENCE)]
    seq = [cmd for cmd in seq if cmd in {"l", "r", "u", "d", "f", "b"}]
    if not seq:
        seq = list(DEFAULT_MOVING_OBSERVE_SEQUENCE)
    phase_specs = _build_moving_phase_specs(
        cmd_sequence=tuple(seq),
        trials=int(trial_count),
        group_by_axis=bool(group_by_axis),
    )

    logger("[MOVING OBSERVE] Strategy: fixed-duration observe-while-moving probe.")
    logger(
        f"[MOVING OBSERVE] Vision backend: {str(vision_mode_used)}. "
        f"Trials={int(trial_count)} cmds={','.join(seq)} score={int(score_used)}% "
        f"phase={int(phase_duration_used_ms)}ms sample_hz={float(sample_hz_used):.1f}."
    )

    initial_pose = None
    initial_meta = {}
    final_pose = None
    final_meta = {}
    phases = []
    pulses = []
    motion_samples = []
    blocks = []
    stop_reason = "completed"

    try:
        logger("[MOVING OBSERVE] Observing initial pose.")
        initial_pose, initial_meta = observe_alignment_pose(
            vision=vision,
            world=world,
            samples=int(observe_samples),
            timeout_s=float(observe_timeout_s),
            relaxed_timeout_s=float(relaxed_timeout_s),
            reobserve_rounds=int(reobserve_rounds),
            log_fn=logger,
        )
        final_pose = initial_pose
        final_meta = dict(initial_meta or {})
        logger(f"[MOVING OBSERVE] Initial pose: {_pose_summary_text(initial_pose)}")

        if initial_pose is None:
            stop_reason = "no_initial_pose"
        else:
            current_pose = initial_pose
            current_meta = dict(initial_meta or {})
            current_block = None
            current_block_key = None
            for spec in phase_specs:
                phase_index = int(spec["phase_index"])
                trial_index = int(spec["trial"])
                cmd = str(spec["cmd"])
                axis = str(spec["axis"])
                block_key = str(spec["block_key"])
                if block_key != current_block_key:
                    if isinstance(current_block, dict):
                        blocks.append(current_block)
                    if current_block_key is None:
                        block_pose = current_pose
                        block_meta = dict(current_meta or {})
                        block_source = "initial_pose"
                    else:
                        logger(f"[MOVING OBSERVE] Reobserving before axis block {str(axis).upper()}.")
                        block_pose, block_meta = observe_alignment_pose(
                            vision=vision,
                            world=world,
                            samples=int(observe_samples),
                            timeout_s=float(observe_timeout_s),
                            relaxed_timeout_s=float(relaxed_timeout_s),
                            reobserve_rounds=int(reobserve_rounds),
                            log_fn=logger,
                        )
                        block_source = "block_reobserve"
                    current_block = {
                        "block_index": int(spec["block_index"]),
                        "block_key": str(block_key),
                        "axis": str(axis),
                        "cmds": [],
                        "start_pose": _pose_snapshot(block_pose),
                        "start_meta": dict(block_meta or {}),
                        "start_source": str(block_source),
                        "phase_indices": [],
                        "completed_phases": 0,
                    }
                    current_block_key = str(block_key)
                    current_pose = block_pose
                    current_meta = dict(block_meta or {})
                    final_pose = block_pose
                    final_meta = dict(block_meta or {})
                    if block_pose is None:
                        stop_reason = "no_block_pose"
                        break

                phase_name = f"trial_{int(trial_index)}_{str(cmd)}"
                logger(
                    "[MOVING OBSERVE] Phase "
                    f"{int(phase_index)}: {str(cmd).upper()} {int(score_used)}% "
                    f"for {int(phase_duration_used_ms)}ms."
                )
                send_policy = _moving_observe_send_policy(
                    cmd=str(cmd),
                    phase_duration_ms=int(phase_duration_used_ms),
                )
                phase_record = {
                    "trial": int(trial_index),
                    "phase_index": int(phase_index),
                    "phase": str(phase_name),
                    "axis": str(axis),
                    "cmd": str(cmd),
                    "pre_pose": _pose_snapshot(current_pose),
                    "pre_meta": dict(current_meta or {}),
                    "send_policy": dict(send_policy),
                    "command_events": [],
                }
                phases.append(phase_record)
                if isinstance(current_block, dict):
                    current_block["phase_indices"].append(int(phase_index))
                    if str(cmd) not in list(current_block.get("cmds") or []):
                        current_block["cmds"].append(str(cmd))

                phase_start = time.monotonic()
                phase_end = float(phase_start) + max(0.001, float(phase_duration_used_ms) / 1000.0)
                next_cmd_time = float(phase_start)
                next_sample_time = float(phase_start)
                phase_samples = []
                first_send_result = None
                while True:
                    now = time.monotonic()
                    if now >= float(phase_end):
                        break
                    action_taken = False
                    if now >= float(next_cmd_time):
                        remaining_ms = max(1, int(round((float(phase_end) - float(now)) * 1000.0)))
                        send_duration_ms = min(int(send_policy["pulse_duration_ms"]), int(remaining_ms))
                        send_kwargs = {
                            "robot": robot,
                            "world": world,
                            "step": EXPERIMENT_STEP,
                            "cmd": str(cmd),
                            "speed": 0.0,
                            "speed_score": int(score_used),
                            "duration_override_ms": int(send_duration_ms),
                            "auto_mode": False,
                            "ease_in_out_enabled": False,
                        }
                        if str(cmd) in {"l", "r"}:
                            send_kwargs["half_first_turn_pulse"] = False
                        send_result = send_robot_command(**send_kwargs)
                        if first_send_result is None and isinstance(send_result, dict):
                            first_send_result = dict(send_result)
                        event = {
                            "phase_elapsed_s": _round_triplet(max(0.0, float(now) - float(phase_start))),
                            "duration_requested_ms": int(send_duration_ms),
                            "send_result": dict(send_result or {}),
                        }
                        phase_record["command_events"].append(event)
                        pulses.append(
                            {
                                "trial": int(trial_index),
                                "phase_index": int(phase_index),
                                "event_index": int(len(phase_record["command_events"])),
                                "cmd": str(cmd),
                                "axis": str(axis),
                                "stage": "moving_observe",
                                "score_pct": int(score_used),
                                "send_result": dict(send_result or {}),
                            }
                        )
                        if not isinstance(send_result, dict):
                            stop_reason = "send_failed"
                            break
                        effective_duration_ms = int((send_result or {}).get("duration_ms") or int(send_duration_ms))
                        next_cmd_time = float(now) + max(
                            float(send_policy["min_interval_s"]),
                            (float(effective_duration_ms) / 1000.0) * float(send_policy["retrigger_fraction"]),
                        )
                        action_taken = True
                    if now >= float(next_sample_time):
                        update_world_from_vision(world, vision, log=False)
                        obs_ts = time.time()
                        sample_row = _moving_sample_row(
                            world,
                            trial_index=int(trial_index),
                            phase_index=int(phase_index),
                            phase_name=str(phase_name),
                            cmd=str(cmd),
                            phase_elapsed_s=max(0.0, float(now) - float(phase_start)),
                            run_elapsed_s=max(0.0, float(now) - float(run_started_monotonic)),
                            obs_ts=float(obs_ts),
                        )
                        phase_samples.append(sample_row)
                        motion_samples.append(dict(sample_row))
                        next_sample_time += float(sample_period_s)
                        action_taken = True
                    if stop_reason != "completed":
                        break
                    if not action_taken:
                        sleep_until = min(float(next_cmd_time), float(next_sample_time), float(phase_end))
                        delay_s = max(0.0, float(sleep_until) - float(time.monotonic()))
                        if delay_s > 0.0:
                            time.sleep(delay_s)

                phase_record["send_result"] = dict(first_send_result or {})
                phase_record["command_send_count"] = int(len(phase_record["command_events"]))
                phase_record["motion_summary"] = _summarize_motion_phase(phase_samples)
                phase_record["motion_sample_count"] = int(len(phase_samples))
                phase_record["motion_samples"] = phase_samples
                if isinstance(current_block, dict) and int(phase_record["command_send_count"]) > 0:
                    current_block["completed_phases"] = int(current_block.get("completed_phases") or 0) + 1

                if stop_reason != "completed":
                    break

                post_pose, post_meta = observe_alignment_pose(
                    vision=vision,
                    world=world,
                    samples=int(observe_samples),
                    timeout_s=float(observe_timeout_s),
                    relaxed_timeout_s=float(relaxed_timeout_s),
                    reobserve_rounds=int(reobserve_rounds),
                    log_fn=logger,
                )
                phase_record["post_pose"] = _pose_snapshot(post_pose)
                phase_record["post_meta"] = dict(post_meta or {})
                phase_record["post_summary"] = _pose_summary_text(post_pose)
                iteration_analysis, next_suggestion = analyze_iteration(
                    world=world,
                    before_pose=current_pose,
                    after_pose=post_pose,
                    before_meta=current_meta,
                    after_meta=post_meta,
                )
                phase_record["iteration_analysis"] = iteration_analysis
                phase_record["next_suggestion"] = next_suggestion
                final_pose = post_pose
                final_meta = dict(post_meta or {})
                current_pose = post_pose
                current_meta = dict(post_meta or {})
                if post_pose is None:
                    stop_reason = "lost_pose_after_move"
                    break
            if isinstance(current_block, dict):
                blocks.append(current_block)

        final_analysis, final_suggestion = analyze_iteration(
            world=world,
            before_pose=initial_pose,
            after_pose=final_pose,
            before_meta=initial_meta,
            after_meta=final_meta,
        )
        result = {
            "ok": bool(initial_pose is not None),
            "experiment_type": "observe_while_moving_probe",
            "objective": "characterize brick visibility and pose drift while timed forward/backward motion is active",
            "seconds": float(max(0.0, time.time() - started)),
            "vision_mode": str(vision_mode_used),
            "score_pct": int(score_used),
            "phase_duration_ms": int(phase_duration_used_ms),
            "sample_hz": _round_triplet(sample_hz_used),
            "trials": int(trial_count),
            "cmd_sequence": list(seq),
            "group_by_axis": bool(group_by_axis),
            "cycles_completed": int(sum(1 for row in phases if isinstance(row, dict) and int(row.get("command_send_count") or 0) > 0)),
            "stop_reason": str(stop_reason),
            "pulses": pulses,
            "observation": {
                "before": _pose_snapshot(initial_pose),
                "before_meta": dict(initial_meta or {}),
                "after": _pose_snapshot(final_pose),
                "after_meta": dict(final_meta or {}),
            },
            "blocks": blocks,
            "phases": phases,
            "motion_samples": motion_samples,
            "analysis": final_analysis,
            "suggestion": final_suggestion,
        }
        if log_path is not None:
            try:
                Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
                result["log_path"] = str(Path(log_path))
            except OSError as exc:
                result["write_error"] = str(exc)
        return result
    finally:
        if created_vision:
            _close_vision(vision)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single-goal x-axis zero experiment by default. "
            "Use --close-dist-x-axis-one-act to repeat one combined act and discover an x+dist closure curve. "
            "Use --alternating-turn-drive for repeated backward-left / forward-right consistency probes. "
            "Use --x-lock-tracking for the older targeted x-axis lock experiment. "
            "Use --observe-while-moving for timed F/B motion sampling. "
            "Use --legacy-random-back-turn to run the old randomized back-then-turn probe."
        )
    )
    parser.add_argument("--score", type=int, default=DEFAULT_SCORE, help="Shared speed score to use for both pulses.")
    parser.add_argument("--back-min-ms", type=int, default=DEFAULT_BACK_RANGE_MS[0])
    parser.add_argument("--back-max-ms", type=int, default=DEFAULT_BACK_RANGE_MS[1])
    parser.add_argument("--turn-min-ms", type=int, default=DEFAULT_TURN_RANGE_MS[0])
    parser.add_argument("--turn-max-ms", type=int, default=DEFAULT_TURN_RANGE_MS[1])
    parser.add_argument("--vision", type=str, default=None, help="Optional vision backend override (aruco/cyan).")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Optional model path passed through to the shared livestream vision builder.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible plans.")
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_TRACK_MAX_CYCLES)
    parser.add_argument("--max-turn-intensity", type=float, default=DEFAULT_MAX_TURN_INTENSITY_PCT)
    parser.add_argument("--max-drive-recovery-score", type=int, default=DEFAULT_MAX_DRIVE_RECOVERY_SCORE)
    parser.add_argument("--observe-timeout-s", type=float, default=None)
    parser.add_argument("--relaxed-timeout-s", type=float, default=None)
    parser.add_argument("--memory", type=str, default=str(RUN_MEMORY_FILE_DEFAULT))
    parser.add_argument(
        "--x-lock-tracking",
        action="store_true",
        help="Run the older targeted x-axis lock experiment instead of the single-goal right-turn trial.",
    )
    parser.add_argument("--x-zero-trials", type=int, default=DEFAULT_X_ZERO_TRIALS)
    parser.add_argument("--x-zero-sample-hz", type=float, default=DEFAULT_X_ZERO_SAMPLE_HZ)
    parser.add_argument("--x-zero-reset-timeout-s", type=float, default=DEFAULT_X_ZERO_RESET_TIMEOUT_S)
    parser.add_argument("--x-zero-attempt-timeout-s", type=float, default=DEFAULT_X_ZERO_ATTEMPT_TIMEOUT_S)
    parser.add_argument("--x-zero-band-mm", type=float, default=DEFAULT_X_ZERO_BAND_MM)
    parser.add_argument(
        "--alternating-turn-drive",
        action="store_true",
        help="Run repeated backward-left then forward-right turn+drive cycles and summarize consistency.",
    )
    parser.add_argument(
        "--close-dist-x-axis-one-act",
        action="store_true",
        help="Run a single-family one-act x+dist closure curve study.",
    )
    parser.add_argument("--alternating-cycles", type=int, default=DEFAULT_ALTERNATING_TURN_DRIVE_CYCLES)
    parser.add_argument(
        "--alternating-duration-ms",
        type=int,
        default=DEFAULT_ALTERNATING_TURN_DRIVE_PHASE_DURATION_MS,
    )
    parser.add_argument(
        "--alternating-strength",
        type=str,
        default=DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
        help="Turn-drive family strength for alternating mode (strong/medium/light).",
    )
    parser.add_argument(
        "--alternating-dist-band-mm",
        type=float,
        default=DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
        help="Distance bucket size for alternating-mode curve reference summaries.",
    )
    parser.add_argument("--one-act-trials", type=int, default=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS)
    parser.add_argument(
        "--one-act-phase",
        type=str,
        default=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_PHASE,
        help="Combined turn+drive family to probe (default: back_left).",
    )
    parser.add_argument(
        "--one-act-duration-ms",
        type=int,
        default=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_DURATION_MS,
        help="Duration per one-act curve trial in ms.",
    )
    parser.add_argument(
        "--one-act-strength",
        type=str,
        default=DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH,
        help="Turn-drive family strength for the one-act curve study (strong/medium/light).",
    )
    parser.add_argument(
        "--one-act-dist-band-mm",
        type=float,
        default=DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM,
        help="Distance bucket size for one-act curve summaries.",
    )
    parser.add_argument(
        "--one-act-setup-forward-min-ms",
        type=int,
        default=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_SETUP_FORWARD_RANGE_MS[0],
        help="Backward one-act only: stage each trial with a forward turn+drive pulse of at least this many ms before the measured back+turn act.",
    )
    parser.add_argument(
        "--one-act-setup-forward-max-ms",
        type=int,
        default=DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_SETUP_FORWARD_RANGE_MS[1],
        help="Backward one-act only: stage each trial with a forward turn+drive pulse of at most this many ms before the measured back+turn act.",
    )
    parser.add_argument(
        "--disable-one-act-setup-forward",
        action="store_true",
        help="Disable the default forward turn+drive staging step that runs before backward one-act curve trials.",
    )
    parser.add_argument(
        "--observe-while-moving",
        action="store_true",
        help="Run a timed forward/backward moving-observation probe instead of the single-goal right-turn trial.",
    )
    parser.add_argument("--moving-duration-ms", type=int, default=DEFAULT_MOVING_OBSERVE_DURATION_MS)
    parser.add_argument("--moving-sample-hz", type=float, default=DEFAULT_MOVING_OBSERVE_SAMPLE_HZ)
    parser.add_argument("--moving-trials", type=int, default=DEFAULT_MOVING_OBSERVE_TRIALS)
    parser.add_argument(
        "--moving-cmds",
        type=str,
        default=",".join(DEFAULT_MOVING_OBSERVE_SEQUENCE),
        help="Comma-separated logical movement commands for moving observe mode (default: l,r,u,d,f,b).",
    )
    parser.add_argument(
        "--moving-interleave-trials",
        action="store_true",
        help="Interleave command types within each trial instead of running grouped axis blocks.",
    )
    parser.add_argument(
        "--legacy-random-back-turn",
        action="store_true",
        help="Run the older single random back-then-turn experiment instead of targeted x-lock tracking.",
    )
    parser.add_argument(
        "--trials-file",
        type=str,
        default=str(TRIALS_FILE_DEFAULT),
        help="Path to write the explicit per-trial motion schedule JSON.",
    )
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else None
    robot = Robot()
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK
    try:
        if bool(args.close_dist_x_axis_one_act):
            observe_timeout_s = (
                float(args.observe_timeout_s)
                if args.observe_timeout_s is not None
                else float(DEFAULT_OBSERVE_TIMEOUT_S)
            )
            relaxed_timeout_s = (
                float(args.relaxed_timeout_s)
                if args.relaxed_timeout_s is not None
                else float(DEFAULT_RELAXED_TIMEOUT_S)
            )
            result = run_close_dist_x_axis_one_act_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                score=int(args.score),
                trials=int(args.one_act_trials),
                phase_duration_ms=int(args.one_act_duration_ms),
                phase=str(args.one_act_phase),
                strength=str(args.one_act_strength),
                distance_band_mm=float(args.one_act_dist_band_mm),
                observe_timeout_s=float(observe_timeout_s),
                relaxed_timeout_s=float(relaxed_timeout_s),
                setup_forward_range_ms=(
                    None
                    if bool(args.disable_one_act_setup_forward)
                    else (
                        int(args.one_act_setup_forward_min_ms),
                        int(args.one_act_setup_forward_max_ms),
                    )
                ),
                rng=rng,
                trials_path=(Path(args.trials_file) if str(args.trials_file or "").strip() else None),
                log_path=Path(args.log),
            )
        elif bool(args.alternating_turn_drive):
            observe_timeout_s = (
                float(args.observe_timeout_s)
                if args.observe_timeout_s is not None
                else float(DEFAULT_OBSERVE_TIMEOUT_S)
            )
            relaxed_timeout_s = (
                float(args.relaxed_timeout_s)
                if args.relaxed_timeout_s is not None
                else float(DEFAULT_RELAXED_TIMEOUT_S)
            )
            result = run_alternating_turn_drive_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                score=int(args.score),
                cycles=int(args.alternating_cycles),
                phase_duration_ms=int(args.alternating_duration_ms),
                strength=str(args.alternating_strength),
                distance_band_mm=float(args.alternating_dist_band_mm),
                observe_timeout_s=float(observe_timeout_s),
                relaxed_timeout_s=float(relaxed_timeout_s),
                log_path=Path(args.log),
            )
        elif bool(args.observe_while_moving):
            observe_timeout_s = (
                float(args.observe_timeout_s)
                if args.observe_timeout_s is not None
                else float(DEFAULT_OBSERVE_TIMEOUT_S)
            )
            relaxed_timeout_s = (
                float(args.relaxed_timeout_s)
                if args.relaxed_timeout_s is not None
                else float(DEFAULT_RELAXED_TIMEOUT_S)
            )
            result = run_observe_while_moving_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                score=int(args.score),
                phase_duration_ms=int(args.moving_duration_ms),
                sample_hz=float(args.moving_sample_hz),
                trials=int(args.moving_trials),
                cmd_sequence=tuple(
                    str(part).strip().lower()
                    for part in str(args.moving_cmds or "").split(",")
                    if str(part).strip()
                ),
                group_by_axis=(not bool(args.moving_interleave_trials)),
                observe_timeout_s=float(observe_timeout_s),
                relaxed_timeout_s=float(relaxed_timeout_s),
                log_path=Path(args.log),
            )
        elif bool(args.legacy_random_back_turn):
            observe_timeout_s = (
                float(args.observe_timeout_s)
                if args.observe_timeout_s is not None
                else float(DEFAULT_OBSERVE_TIMEOUT_S)
            )
            relaxed_timeout_s = (
                float(args.relaxed_timeout_s)
                if args.relaxed_timeout_s is not None
                else float(DEFAULT_RELAXED_TIMEOUT_S)
            )
            result = run_random_back_turn_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                score=int(args.score),
                rng=rng,
                seed=args.seed,
                back_range_ms=(int(args.back_min_ms), int(args.back_max_ms)),
                turn_range_ms=(int(args.turn_min_ms), int(args.turn_max_ms)),
                observe_timeout_s=float(observe_timeout_s),
                relaxed_timeout_s=float(relaxed_timeout_s),
                log_path=Path(args.log),
            )
        elif bool(args.x_lock_tracking):
            observe_timeout_s = (
                float(args.observe_timeout_s)
                if args.observe_timeout_s is not None
                else float(DEFAULT_TRACK_OBSERVE_TIMEOUT_S)
            )
            relaxed_timeout_s = (
                float(args.relaxed_timeout_s)
                if args.relaxed_timeout_s is not None
                else float(DEFAULT_TRACK_RELAXED_TIMEOUT_S)
            )
            result = run_x_axis_lock_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                max_cycles=int(args.max_cycles),
                track_observe_timeout_s=float(observe_timeout_s),
                track_relaxed_timeout_s=float(relaxed_timeout_s),
                max_turn_intensity_pct=float(args.max_turn_intensity),
                max_drive_recovery_score=float(args.max_drive_recovery_score),
                log_path=Path(args.log),
                memory_path=Path(args.memory) if str(args.memory or "").strip() else None,
            )
        else:
            result = run_single_goal_right_turn_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                trials=int(args.x_zero_trials),
                sample_hz=float(args.x_zero_sample_hz),
                reset_timeout_s=float(args.x_zero_reset_timeout_s),
                attempt_timeout_s=float(args.x_zero_attempt_timeout_s),
                reset_turn_intensity_pct=float(DEFAULT_X_ZERO_RESET_TURN_INTENSITY_PCT),
                turn_intensity_pct=float(DEFAULT_X_ZERO_TURN_INTENSITY_PCT),
                zero_band_mm=float(args.x_zero_band_mm),
                log_path=Path(args.log),
            )
        print(json.dumps(result, indent=2))
        return 0 if bool(result.get("ok")) else 1
    except KeyboardInterrupt:
        print("\n[RANDOM EXPERIMENT] Interrupted.")
        return 130
    finally:
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
