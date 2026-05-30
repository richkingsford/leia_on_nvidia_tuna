#!/usr/bin/env python3
"""Practice empty Step 1 with a coordinated Astolfi-style drive controller."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_astolfi_controller import AstolfiState, astolfi_wheel_command, overdamped_pd_gains
from helper_brick_detector_native_oak import BrickDetector
from helper_brick_visibility_safety import guarded_send_command_pwm, guarded_send_custom_actions_pwm
from helper_robot_control import Robot
import telemetry_robot


OUT_PATH = Path("runs/astolfi_step1_practice.jsonl")
ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"


def _load_controller_config() -> dict:
    defaults = {
        "control_hz": 20.0,
        "act_packet_ms": 100,
        "max_wheel_packet_ms": 350,
        "min_wheel_packet_ms": 100,
        "stop_offset_mm": 196.8,
        "dist_tol_mm": 50.0,
        "x_target_mm": 5.9,
        "x_tol_mm": 3.0,
        "y_target_mm": -42.7,
        "y_tol_mm": 3.0,
        "bearing_y_offset_mm": 196.8,
        "damping_ratio": 1.5,
        "dist_settle_time_s": 1.6,
        "heading_settle_time_s": 1.0,
        "linear_limit": 1.0,
        "angular_limit": 0.75,
        "wheel_limit": 1.0,
        "deadband_cmd": 0.18,
        "derivative_alpha": 0.35,
        "max_trial_s": 18.0,
        "settle_confirm_frames": 2,
    }
    try:
        raw = json.loads(ROBOT_MODEL_FILE.read_text())
        cfg = raw.get("follow_the_brick", {}).get("astolfi_step1_practice", {})
        if isinstance(cfg, dict):
            defaults.update(cfg)
    except Exception:
        pass
    return defaults


def _coerce_float(cfg: dict, key: str, fallback: float) -> float:
    try:
        return float(cfg.get(key))
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(cfg: dict, key: str, fallback: int) -> int:
    try:
        return int(round(float(cfg.get(key))))
    except (TypeError, ValueError):
        return int(fallback)


def _wheel_action(target: str, wheel_cmd: float) -> dict:
    target_key = str(target)
    if abs(float(wheel_cmd)) <= 0.0:
        return {"target": target_key, "action": "s", "pwm": 0, "duration_ms": 0}
    # Robot-forward wheel sign: left tread wire is inverted, right tread is normal.
    if target_key == "l":
        action = "b" if float(wheel_cmd) > 0.0 else "f"
    else:
        action = "f" if float(wheel_cmd) > 0.0 else "b"
    return {"target": target_key, "action": action, "pwm": 103}


def _wheel_packet_actions(left: float, right: float, cfg: dict) -> tuple[list[dict], int]:
    max_ms = max(100, _coerce_int(cfg, "max_wheel_packet_ms", 350))
    min_ms = max(100, _coerce_int(cfg, "min_wheel_packet_ms", 100))
    deadband = max(0.0, _coerce_float(cfg, "deadband_cmd", 0.18))
    wheel_limit = max(0.01, _coerce_float(cfg, "wheel_limit", 1.0))

    actions = []
    durations = []
    for target, value in (("l", left), ("r", right)):
        mag = abs(float(value)) / wheel_limit
        if mag < deadband:
            action = {"target": target, "action": "s", "pwm": 0, "duration_ms": 0}
        else:
            duration = int(round(min_ms + (max_ms - min_ms) * min(1.0, mag)))
            action = _wheel_action(target, value)
            # Hard rule: any nonzero wheel action is the 1% floor PWM.
            action["pwm"] = 103
            action["duration_ms"] = max(min_ms, duration)
            durations.append(int(action["duration_ms"]))
        actions.append(action)
    return actions, max(durations) if durations else 0


def _happy(reading: dict, cfg: dict) -> bool:
    try:
        dist_err = float(reading.get("dist_mm")) - _coerce_float(cfg, "stop_offset_mm", 196.8)
        x_err = float(reading.get("x_mm")) - _coerce_float(cfg, "x_target_mm", 5.9)
        y_err = float(reading.get("y_mm")) - _coerce_float(cfg, "y_target_mm", -42.7)
    except (TypeError, ValueError):
        return False
    return (
        abs(dist_err) <= _coerce_float(cfg, "dist_tol_mm", 11.0)
        and abs(x_err) <= _coerce_float(cfg, "x_tol_mm", 3.0)
        and abs(y_err) <= _coerce_float(cfg, "y_tol_mm", 3.0)
    )


def _snapshot(reading: dict, cfg: dict) -> dict:
    row = {"confident": bool((reading or {}).get("confident"))}
    for key in ("dist_mm", "x_mm", "y_mm", "conf", "reason"):
        row[key] = (reading or {}).get(key)
    try:
        row["dist_err_mm"] = float(reading.get("dist_mm")) - _coerce_float(cfg, "stop_offset_mm", 196.8)
        row["x_err_mm"] = float(reading.get("x_mm")) - _coerce_float(cfg, "x_target_mm", 5.9)
        row["y_err_mm"] = float(reading.get("y_mm")) - _coerce_float(cfg, "y_target_mm", -42.7)
    except (TypeError, ValueError):
        pass
    return row


def _practice_y_action_plan(reading: dict, cfg: dict) -> dict | None:
    try:
        y_mm = float((reading or {}).get("y_mm"))
    except (TypeError, ValueError):
        return None
    target = _coerce_float(cfg, "y_target_mm", -42.7)
    tol = _coerce_float(cfg, "y_tol_mm", 3.0)
    y_err = float(y_mm) - float(target)
    if abs(y_err) <= tol:
        return None
    gap_to_edge = max(0.0, abs(y_err) - tol)
    min_ms = _coerce_int(cfg, "mast_min_packet_ms", 500)
    max_ms = _coerce_int(cfg, "mast_max_packet_ms", 900)
    full_gap = max(1.0, _coerce_float(cfg, "mast_full_gap_mm", 10.0))
    duration_ms = int(round(min_ms + min(1.0, gap_to_edge / full_gap) * (max_ms - min_ms)))
    # For the current logical coordinate system, positive y error means the
    # observed brick is above the target in the image; lower the mast.
    cmd = "d" if y_err > 0.0 else "u"
    return {
        "kind": "mast",
        "cmd": cmd,
        "pwm": 255,
        "duration_ms": max(min_ms, min(max_ms, duration_ms)),
        "y_err": float(y_err),
        "y_mm": float(y_mm),
        "y_target_mm": float(target),
    }


def _pregame_recover(vision: BrickDetector, robot: Robot, cfg: dict) -> None:
    """Wait for brick visibility then nudge mast to bring y into start band."""
    vis_deadline = time.monotonic() + 8.0
    while time.monotonic() < vis_deadline:
        r = follow._read_brick_measurement(vision, jump_guard=False)
        if r.get("confident"):
            break
        follow._stop_robot(robot)
        time.sleep(0.15)

    y_target = _coerce_float(cfg, "y_target_mm", -42.7)
    start_band = 10.0
    for _ in range(6):
        r = follow._read_brick_measurement(vision, jump_guard=False)
        if not r.get("confident"):
            break
        try:
            y_err = float(r["y_mm"]) - y_target
        except (TypeError, ValueError):
            break
        if abs(y_err) <= start_band:
            break
        prev = abs(y_err)
        cmd = "d" if y_err > 0.0 else "u"
        guarded_send_command_pwm(robot, cmd, 255, duration_ms=360, reading=r)
        time.sleep(0.35)
        r2 = follow._read_brick_measurement(vision, jump_guard=False)
        if r2.get("confident"):
            try:
                if abs(float(r2["y_mm"]) - y_target) > prev + 2.0:
                    break
            except (TypeError, ValueError):
                pass
    follow._stop_robot(robot)


def _run_trial(vision: BrickDetector, robot: Robot, cfg: dict, trial: int) -> dict:
    state = AstolfiState()
    gains = overdamped_pd_gains(
        damping_ratio=_coerce_float(cfg, "damping_ratio", 1.5),
        dist_settle_time_s=_coerce_float(cfg, "dist_settle_time_s", 1.6),
        heading_settle_time_s=_coerce_float(cfg, "heading_settle_time_s", 1.0),
    )
    dt_s = 1.0 / max(1.0, _coerce_float(cfg, "control_hz", 20.0))
    max_trial_s = _coerce_float(cfg, "max_trial_s", 18.0)
    confirm_frames = max(1, _coerce_int(cfg, "settle_confirm_frames", 2))
    deadline = time.monotonic() + max_trial_s
    ticks = 0
    moves = 0
    happy_frames = 0
    last_reading = {}
    first = None
    last_control = {}

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        reading = follow._read_brick_measurement(vision, jump_guard=True)
        last_reading = dict(reading)
        if first is None and bool(reading.get("confident")):
            first = _snapshot(reading, cfg)
        if not bool(reading.get("confident")):
            follow._stop_robot(robot)
            time.sleep(dt_s)
            ticks += 1
            continue

        if _happy(reading, cfg):
            happy_frames += 1
            follow._stop_robot(robot)
            if happy_frames >= confirm_frames:
                return {
                    "trial": trial,
                    "won": True,
                    "ticks": ticks,
                    "moves": moves,
                    "first": first,
                    "final": _snapshot(reading, cfg),
                    "last_control": last_control,
                }
            time.sleep(dt_s)
            ticks += 1
            continue
        happy_frames = 0
        dist_err_now = float(reading.get("dist_mm")) - _coerce_float(cfg, "stop_offset_mm", 196.8)
        x_err_now = float(reading.get("x_mm")) - _coerce_float(cfg, "x_target_mm", 5.9)
        y_plan = _practice_y_action_plan(reading, cfg)
        mast_priority_ready = (
            abs(dist_err_now) <= _coerce_float(cfg, "mast_priority_dist_window_mm", 30.0)
            and abs(x_err_now) <= _coerce_float(cfg, "mast_priority_x_window_mm", 12.0)
        )
        if mast_priority_ready and isinstance(y_plan, dict) and abs(float(y_plan.get("y_err", 0.0))) > (
            _coerce_float(cfg, "y_tol_mm", 3.0) + _coerce_float(cfg, "mast_priority_margin_mm", 1.5)
        ):
            cmd = str(y_plan.get("cmd") or "").strip().lower()
            if cmd in {"u", "d"}:
                guarded_send_command_pwm(
                    robot,
                    cmd,
                    int(y_plan.get("pwm") or 255),
                    duration_ms=int(y_plan.get("duration_ms") or 500),
                    reading=reading,
                    context="astolfi_step1_mast_priority",
                )
                last_control = {"y_plan": dict(y_plan), "mast_priority": True}
                moves += 1
                time.sleep(max(dt_s, float(int(y_plan.get("duration_ms") or 500)) / 1000.0))
                follow._stop_robot(robot)
                ticks += 1
                continue
        x_curve_threshold = _coerce_float(cfg, "x_curve_override_abs_err_mm", 12.0)
        if abs(x_err_now) > x_curve_threshold:
            turn_cmd = follow._turn_cmd_to_close_x_gap(float(x_err_now)) or ("r" if x_err_now > 0.0 else "l")
            drive_mode = "forward" if dist_err_now >= 0.0 else "backward"
            duration_ms = _coerce_int(cfg, "x_curve_override_packet_ms", 250)
            send_result = follow._send_drive_bias(
                robot,
                turn_cmd=turn_cmd,
                drive_mode=drive_mode,
                strength="adaptive",
                duration_ms=duration_ms,
                reading=reading,
                context="astolfi_step1_x_curve_override",
            )
            last_control = {
                "x_curve_override": True,
                "turn_cmd": turn_cmd,
                "drive_mode": drive_mode,
                "duration_ms": int(duration_ms),
                "send_result": send_result,
                "y_plan": dict(y_plan) if isinstance(y_plan, dict) else None,
            }
            moves += 1
            time.sleep(max(dt_s, float(duration_ms) / 1000.0))
            follow._stop_robot(robot)
            ticks += 1
            continue

        control = astolfi_wheel_command(
            x_off_mm=(
                (float(reading.get("x_mm")) - _coerce_float(cfg, "x_target_mm", 5.9))
                * _coerce_float(cfg, "physical_heading_sign", -1.0)
            ),
            distance_mm=float(reading.get("dist_mm")),
            stop_offset_mm=_coerce_float(cfg, "stop_offset_mm", 196.8),
            bearing_y_offset_mm=_coerce_float(cfg, "bearing_y_offset_mm", 196.8),
            dt_s=dt_s,
            gains=gains,
            state=state,
            derivative_alpha=_coerce_float(cfg, "derivative_alpha", 0.35),
            linear_limit=_coerce_float(cfg, "linear_limit", 1.0),
            angular_limit=_coerce_float(cfg, "angular_limit", 0.75),
            wheel_limit=_coerce_float(cfg, "wheel_limit", 1.0),
            max_dist_derivative_m_s=_coerce_float(cfg, "max_dist_derivative_m_s", 0.08),
            max_head_derivative_rad_s=_coerce_float(cfg, "max_head_derivative_rad_s", 0.18),
        )
        same_direction_gap = _coerce_float(cfg, "same_direction_drive_until_dist_gap_mm", 20.0)
        if abs(float(control["dist_err_mm"])) > same_direction_gap:
            v_sign = 1.0 if float(control["v"]) >= 0.0 else -1.0
            left = float(control["left"])
            right = float(control["right"])
            if left * v_sign < 0.0:
                left = 0.0
            if right * v_sign < 0.0:
                right = 0.0
            control["left"] = left
            control["right"] = right
            control["same_direction_constraint"] = True
        else:
            control["same_direction_constraint"] = False
        actions, packet_ms = _wheel_packet_actions(control["left"], control["right"], cfg)
        if isinstance(y_plan, dict) and str(y_plan.get("kind") or "") == "mast":
            mast_cmd = str(y_plan.get("cmd") or "").strip().lower()
            if mast_cmd in {"u", "d"}:
                actions.append({
                    "target": "m",
                    "action": mast_cmd,
                    "pwm": int(y_plan.get("pwm") or 255),
                    "duration_ms": int(y_plan.get("duration_ms") or 500),
                })
                packet_ms = max(int(packet_ms), int(y_plan.get("duration_ms") or 500))
        last_control = dict(control)
        if isinstance(y_plan, dict):
            last_control["y_plan"] = dict(y_plan)
        if packet_ms > 0:
            label = "f" if control["v"] >= 0 else "b"
            guarded_send_custom_actions_pwm(
                robot,
                label,
                actions,
                duration_ms=packet_ms,
                reading=reading,
                context="astolfi_step1_controller",
            )
            moves += 1
            time.sleep(max(dt_s, float(packet_ms) / 1000.0))
            follow._stop_robot(robot)
        else:
            follow._stop_robot(robot)
            time.sleep(dt_s)
        elapsed = time.monotonic() - loop_start
        if elapsed < dt_s:
            time.sleep(dt_s - elapsed)
        ticks += 1

    return {
        "trial": trial,
        "won": False,
        "ticks": ticks,
        "moves": moves,
        "first": first,
        "final": _snapshot(last_reading, cfg),
        "last_control": last_control,
    }


def _practice_reset(vision: BrickDetector, robot: Robot, cfg: dict) -> dict:
    base = follow._reset_motion_config()
    reset_cfg = copy.deepcopy(base)
    reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
    reverse["post_pause_s"] = 0.2
    reverse["settle_s"] = 0.0
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    try:
        original_ms = int(round(float(straight.get("duration_ms"))))
    except (TypeError, ValueError):
        original_ms = 3000
    straight["duration_ms"] = max(1, int(round(original_ms * _coerce_float(cfg, "reset_straight_scale", 0.2))))
    reverse["straight_back_first"] = straight
    reverse["dist_target_mm"] = _coerce_float(cfg, "stop_offset_mm", 196.8)
    reverse["dist_tol_mm"] = 80.0
    reverse["x_offset_min_mm"] = 8.0
    reverse["x_offset_max_mm"] = 18.0
    reverse["target_abs_x_mm"] = 12.0
    x_curve = reverse.get("x_goal_curve") if isinstance(reverse.get("x_goal_curve"), dict) else {}
    x_curve["max_duration_ms"] = _coerce_int(cfg, "reset_x_curve_max_ms", 160)
    reverse["x_goal_curve"] = x_curve
    adjustment = reverse.get("adjustment") if isinstance(reverse.get("adjustment"), dict) else {}
    adjustment["enabled"] = True
    adjustment["max_attempts"] = 2
    adjustment["pulse_min_ms"] = 80
    adjustment["pulse_max_ms"] = 160
    reverse["adjustment"] = adjustment
    reset_cfg["reverse_turn"] = reverse
    mast_up = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    mast_up["enabled"] = False
    mast_up["practice_disabled_reason"] = "astolfi_step1_practice_keeps_mast_in_vision_zone"
    reset_cfg["mast_up"] = mast_up
    old = follow._reset_motion_config
    old_small_turn_adjust = follow._reset_small_turn_adjust

    def _practice_reset_small_turn_adjust(robot, *, turn_cmd: str, reading: dict, duration_ms: int, reset_cfg: dict, reason: str):
        corrected_turn = str(turn_cmd or "").strip().lower()
        if str(reason or "").strip().lower() == "reduce_x":
            corrected_turn = "r" if corrected_turn == "l" else "l"
        return old_small_turn_adjust(
            robot,
            turn_cmd=corrected_turn,
            reading=reading,
            duration_ms=duration_ms,
            reset_cfg=reset_cfg,
            reason=reason,
        )

    follow._reset_motion_config = lambda: copy.deepcopy(reset_cfg)
    follow._reset_small_turn_adjust = _practice_reset_small_turn_adjust
    try:
        return follow._run_reset_sequence(vision, robot, rng=random)
    finally:
        follow._reset_motion_config = old
        follow._reset_small_turn_adjust = old_small_turn_adjust


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--trials-per-loop", type=int, default=8)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    cfg = _load_controller_config()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    vision = BrickDetector(debug=True)
    robot = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        with out_path.open("w", encoding="utf-8") as out:
            for loop in range(1, max(1, args.loops) + 1):
                wins = 0
                for trial in range(1, max(1, args.trials_per_loop) + 1):
                    print(f"[ASTOLFI] loop {loop}/{args.loops} trial {trial}/{args.trials_per_loop}", flush=True)
                    _pregame_recover(vision, robot, cfg)
                    record = _run_trial(vision, robot, cfg, trial)
                    record["loop"] = loop
                    wins += 1 if record.get("won") else 0
                    if trial < args.trials_per_loop:
                        reset = _practice_reset(vision, robot, cfg)
                        record["reset"] = reset
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                print(f"[ASTOLFI] loop {loop} wins={wins}/{args.trials_per_loop}", flush=True)
        return 0
    finally:
        follow._stop_robot(robot)
        try:
            robot.close()
        except Exception:
            pass
        try:
            vision.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
