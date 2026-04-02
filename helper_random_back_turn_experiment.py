#!/usr/bin/env python3
"""Run focused alignment experiments around x-axis turn behavior."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from calibration.helper_calibrate import (
    brick_pose_from_world,
    observe_pose_with_reobserve,
    read_pose,
)
from helper_brick_detector_yolo import (
    BrickDetector,
    CYAN_HSV_TIGHT_LOWER,
    CYAN_HSV_TIGHT_UPPER,
)
from helper_close_gaps import curve_intensity_for_error, load_left_right_curve
from helper_next import compute_alignment_analytics
from helper_robot_control import Robot
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
    update_world_from_vision,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import StepState, WorldModel


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_random_back_turn_experiment.json"
RUN_MEMORY_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_random_back_turn_experiment_memory.json"
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

_CROWN_PROFILE_RUNTIME = {
    "confidence": 0.15,
    "smoothing_alpha": 0.20,
    "hsv_enabled": True,
    "hsv_erode_iterations": 0,
    "hsv_lower": list(CYAN_HSV_TIGHT_LOWER),
    "hsv_upper": list(CYAN_HSV_TIGHT_UPPER),
}


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
    }


def _pose_summary_text(pose: dict | None) -> str:
    if not isinstance(pose, dict):
        return "unavailable"
    return (
        f"x={_round_triplet(pose.get('offset_x'))}mm "
        f"y={_round_triplet(pose.get('offset_y'))}mm "
        f"dist={_round_triplet(pose.get('dist'))}mm "
        f"angle={_round_triplet(pose.get('angle'))}deg "
        f"conf={_round_triplet(pose.get('confidence'))}% "
        f"via {str(pose.get('pose_source') or 'unknown')}"
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

    def _send_segment(duration_ms: int):
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
                return {"stop": True, "reason": "visibility_lost_after_reacquire"}
            return None

        abs_x = abs(float(x_axis_mm))
        if state.get("first_visible_pose") is None:
            state["first_visible_pose"] = _sample_pose_snapshot(sample)
            state["first_visible_sample_index"] = int((sample or {}).get("sample_index") or 0)
            state["first_visible_x_mm"] = _round_triplet(x_axis_mm)
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
    if mode_used == VISION_MODE_CYAN:
        vision = BrickDetector(debug=True, model_path=yolo_model_path)
        set_runtime_tuning = getattr(vision, "set_runtime_tuning", None)
        if callable(set_runtime_tuning):
            set_runtime_tuning(**dict(_CROWN_PROFILE_RUNTIME))
        else:
            vision.conf_threshold = float(_CROWN_PROFILE_RUNTIME["confidence"])
            vision._smooth_alpha = float(_CROWN_PROFILE_RUNTIME["smoothing_alpha"])
            vision._hsv_enabled = bool(_CROWN_PROFILE_RUNTIME["hsv_enabled"])
            vision._hsv_erode_iterations = int(_CROWN_PROFILE_RUNTIME["hsv_erode_iterations"])
            vision._hsv_lower = list(_CROWN_PROFILE_RUNTIME["hsv_lower"])
            vision._hsv_upper = list(_CROWN_PROFILE_RUNTIME["hsv_upper"])
        return vision, mode_used
    return ArucoBrickVision(debug=True, debug_rejected_candidates=True), VISION_MODE_ARUCO


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
        hold_s=float(DEFAULT_SETTLE_S),
        reobserve_rounds=int(reobserve_rounds),
        relaxed_timeout_s=float(relaxed_timeout_s),
    )
    return pose, dict(meta or {})


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
    parser.add_argument("--vision", type=str, default=None, help="Optional vision backend override (aruco/crown).")
    parser.add_argument("--yolo-model", type=str, default=None, help="Optional YOLO ONNX model path for crown mode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible plans.")
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_TRACK_MAX_CYCLES)
    parser.add_argument("--max-turn-intensity", type=float, default=DEFAULT_MAX_TURN_INTENSITY_PCT)
    parser.add_argument("--max-drive-recovery-score", type=int, default=DEFAULT_MAX_DRIVE_RECOVERY_SCORE)
    parser.add_argument("--observe-timeout-s", type=float, default=DEFAULT_TRACK_OBSERVE_TIMEOUT_S)
    parser.add_argument("--relaxed-timeout-s", type=float, default=DEFAULT_TRACK_RELAXED_TIMEOUT_S)
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
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else None
    robot = Robot()
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK
    try:
        if bool(args.observe_while_moving):
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
                observe_timeout_s=float(args.observe_timeout_s),
                relaxed_timeout_s=float(args.relaxed_timeout_s),
                log_path=Path(args.log),
            )
        elif bool(args.legacy_random_back_turn):
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
                log_path=Path(args.log),
            )
        elif bool(args.x_lock_tracking):
            result = run_x_axis_lock_experiment(
                robot=robot,
                world=world,
                vision_mode=args.vision,
                yolo_model_path=args.yolo_model,
                max_cycles=int(args.max_cycles),
                track_observe_timeout_s=float(args.observe_timeout_s),
                track_relaxed_timeout_s=float(args.relaxed_timeout_s),
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
