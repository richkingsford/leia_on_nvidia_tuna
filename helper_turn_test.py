#!/usr/bin/env python3
"""Build and run Leia's slow sharpening turn test sequence."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import telemetry_robot as telemetry_robot_module


DEFAULT_PHASE_DURATION_S = 5.0
DEFAULT_REQUESTED_STEP_MS = 250
DEFAULT_HOLD_AFTER_S = 2.0
DEFAULT_STOP_HOLD_STEPS = 2
DEFAULT_SPEED_SCORE = 1
DEFAULT_START_RATIO = 0.98

PHYSICAL_FORWARD = "forward"
PHYSICAL_BACKWARD = "backward"
PHYSICAL_STOP = "stop"


@dataclass(frozen=True)
class TurnTestPulse:
    phase_name: str
    logical_cmd: str
    drive_mode: str
    phase_step: int
    phase_steps: int
    duration_ms: int
    outer_target: str
    inner_target: str
    outer_direction: str
    inner_direction: str
    outer_pwm: int
    inner_pwm: int
    inner_ratio: float
    actions: tuple[dict, ...]


def _positive_int(value, fallback: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback)
    return parsed if parsed > 0 else int(fallback)


def _linear_values(start: float, end: float, count: int) -> list[float]:
    count = max(0, int(count))
    if count <= 0:
        return []
    if count == 1:
        return [float(end)]
    step = (float(end) - float(start)) / float(count - 1)
    return [float(start) + float(step) * idx for idx in range(count)]


def _turn_targets(logical_cmd: str) -> tuple[str, str]:
    cmd = str(logical_cmd or "").strip().lower()
    if cmd == "r":
        return "l", "r"
    if cmd == "l":
        return "r", "l"
    raise ValueError(f"Turn test only supports logical left/right turns, got {logical_cmd!r}.")


def _drive_actions(drive_mode: str) -> dict[str, str]:
    mode = str(drive_mode or "").strip().lower()
    if mode == PHYSICAL_FORWARD:
        return {"l": "b", "r": "f"}
    if mode == PHYSICAL_BACKWARD:
        return {"l": "f", "r": "b"}
    raise ValueError(f"Unsupported drive mode: {drive_mode!r}.")


def _opposite_drive_mode(drive_mode: str) -> str:
    mode = str(drive_mode or "").strip().lower()
    if mode == PHYSICAL_FORWARD:
        return PHYSICAL_BACKWARD
    if mode == PHYSICAL_BACKWARD:
        return PHYSICAL_FORWARD
    raise ValueError(f"Unsupported drive mode: {drive_mode!r}.")


def _physical_tread_direction(target: str, action: str) -> str:
    target_key = str(target or "").strip().lower()
    action_key = str(action or "").strip().lower()
    if action_key == "s":
        return PHYSICAL_STOP
    if (target_key, action_key) in {("l", "b"), ("r", "f")}:
        return PHYSICAL_FORWARD
    if (target_key, action_key) in {("l", "f"), ("r", "b")}:
        return PHYSICAL_BACKWARD
    return action_key or PHYSICAL_STOP


def _speed_pwm(cmd: str, score: int) -> int:
    _power, pwm, _score_used, _duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
    return int(pwm)


def _speed_duration_ms(cmd: str, score: int) -> int:
    _power, _pwm, _score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
    return int(duration_ms)


def _floor_ratio(*, inner_floor_pwm: int, outer_pwm: int) -> float:
    outer = max(1, int(outer_pwm))
    floor = max(0, int(inner_floor_pwm))
    return max(0.0, min(1.0, float(floor) / float(outer)))


def _phase_inner_plan(
    *,
    drive_mode: str,
    phase_steps: int,
    hold_after_steps: int,
    stop_hold_steps: int,
    min_active_ratio: float,
    start_ratio: float,
) -> list[tuple[str, float]]:
    phase_steps = max(1, int(phase_steps))
    hold_after_steps = max(1, min(int(hold_after_steps), int(phase_steps)))
    stop_hold_steps = max(0, min(int(stop_hold_steps), int(phase_steps) - int(hold_after_steps)))
    reverse_steps = max(0, int(phase_steps) - int(hold_after_steps) - int(stop_hold_steps))

    start = max(float(min_active_ratio), min(1.0, float(start_ratio)))
    drive_direction = str(drive_mode or "").strip().lower()
    reverse_direction = _opposite_drive_mode(drive_direction)
    moving_with_drive = [
        (drive_direction, ratio)
        for ratio in _linear_values(start, float(min_active_ratio), int(hold_after_steps))
    ]
    stopped = [(PHYSICAL_STOP, 0.0) for _idx in range(int(stop_hold_steps))]
    reversing = [
        (reverse_direction, ratio)
        for ratio in _linear_values(float(min_active_ratio), 1.0, int(reverse_steps))
    ]
    return moving_with_drive + stopped + reversing


def _build_actions(
    *,
    drive_mode: str,
    logical_cmd: str,
    outer_pwm: int,
    inner_pwm: int,
    inner_direction: str,
) -> tuple[dict, ...]:
    outer_target, inner_target = _turn_targets(logical_cmd)
    drive_actions = _drive_actions(drive_mode)
    reverse_actions = _drive_actions(_opposite_drive_mode(drive_mode))
    inner_direction_key = str(inner_direction or "").strip().lower()
    if inner_direction_key == PHYSICAL_STOP or int(inner_pwm) <= 0:
        inner_action = "s"
        inner_pwm_out = 0
    elif inner_direction_key == str(drive_mode).strip().lower():
        inner_action = drive_actions[inner_target]
        inner_pwm_out = int(inner_pwm)
    else:
        inner_action = reverse_actions[inner_target]
        inner_pwm_out = int(inner_pwm)

    by_target = {
        outer_target: {"target": outer_target, "action": drive_actions[outer_target], "pwm": int(outer_pwm)},
        inner_target: {"target": inner_target, "action": inner_action, "pwm": int(inner_pwm_out)},
    }
    return tuple(dict(by_target[target]) for target in ("l", "r") if target in by_target)


def _build_phase(
    *,
    phase_name: str,
    logical_cmd: str,
    drive_mode: str,
    phase_steps: int,
    hold_after_steps: int,
    stop_hold_steps: int,
    duration_ms: int,
    outer_pwm: int,
    inner_floor_pwm: int,
    start_ratio: float,
) -> list[TurnTestPulse]:
    min_active_ratio = _floor_ratio(inner_floor_pwm=inner_floor_pwm, outer_pwm=outer_pwm)
    inner_plan = _phase_inner_plan(
        drive_mode=drive_mode,
        phase_steps=phase_steps,
        hold_after_steps=hold_after_steps,
        stop_hold_steps=stop_hold_steps,
        min_active_ratio=min_active_ratio,
        start_ratio=start_ratio,
    )
    outer_target, inner_target = _turn_targets(logical_cmd)
    outer_direction = str(drive_mode).strip().lower()
    pulses: list[TurnTestPulse] = []
    for idx, (inner_direction, ratio) in enumerate(inner_plan):
        inner_pwm = 0 if inner_direction == PHYSICAL_STOP else int(round(float(outer_pwm) * float(ratio)))
        if inner_direction != PHYSICAL_STOP:
            inner_pwm = max(int(inner_floor_pwm), min(int(outer_pwm), int(inner_pwm)))
        actions = _build_actions(
            drive_mode=drive_mode,
            logical_cmd=logical_cmd,
            outer_pwm=int(outer_pwm),
            inner_pwm=int(inner_pwm),
            inner_direction=inner_direction,
        )
        pulses.append(
            TurnTestPulse(
                phase_name=str(phase_name),
                logical_cmd=str(logical_cmd),
                drive_mode=str(drive_mode),
                phase_step=int(idx + 1),
                phase_steps=int(phase_steps),
                duration_ms=int(duration_ms),
                outer_target=str(outer_target),
                inner_target=str(inner_target),
                outer_direction=str(outer_direction),
                inner_direction=str(inner_direction),
                outer_pwm=int(outer_pwm),
                inner_pwm=int(inner_pwm),
                inner_ratio=float(ratio),
                actions=actions,
            )
        )
    return pulses


def build_turn_test_sequence(
    *,
    phase_duration_s: float = DEFAULT_PHASE_DURATION_S,
    requested_step_ms: int = DEFAULT_REQUESTED_STEP_MS,
    hold_after_s: float = DEFAULT_HOLD_AFTER_S,
    stop_hold_steps: int = DEFAULT_STOP_HOLD_STEPS,
    speed_score: int = DEFAULT_SPEED_SCORE,
    start_ratio: float = DEFAULT_START_RATIO,
) -> list[TurnTestPulse]:
    score = telemetry_robot_module.normalize_speed_score(speed_score, default=DEFAULT_SPEED_SCORE)
    requested_ms = _positive_int(requested_step_ms, DEFAULT_REQUESTED_STEP_MS)
    phase_ms = max(requested_ms, int(round(float(phase_duration_s) * 1000.0)))
    phase_steps = max(1, int(round(float(phase_ms) / float(requested_ms))))
    hold_after_steps = max(1, int(round(float(hold_after_s) * 1000.0 / float(requested_ms))))
    hold_after_steps = min(int(hold_after_steps), int(phase_steps))

    drive_floor_pwm = max(_speed_pwm("f", score), _speed_pwm("b", score))
    forward_right_outer_pwm = max(int(drive_floor_pwm), _speed_pwm("r", score))
    backward_left_outer_pwm = max(int(drive_floor_pwm), _speed_pwm("l", score))
    effective_step_ms = max(requested_ms, _speed_duration_ms("f", score), _speed_duration_ms("b", score))

    forward_right = _build_phase(
        phase_name="forward+right",
        logical_cmd="r",
        drive_mode=PHYSICAL_FORWARD,
        phase_steps=phase_steps,
        hold_after_steps=hold_after_steps,
        stop_hold_steps=stop_hold_steps,
        duration_ms=effective_step_ms,
        outer_pwm=forward_right_outer_pwm,
        inner_floor_pwm=drive_floor_pwm,
        start_ratio=start_ratio,
    )
    backward_left = _build_phase(
        phase_name="back+left",
        logical_cmd="l",
        drive_mode=PHYSICAL_BACKWARD,
        phase_steps=phase_steps,
        hold_after_steps=hold_after_steps,
        stop_hold_steps=stop_hold_steps,
        duration_ms=effective_step_ms,
        outer_pwm=backward_left_outer_pwm,
        inner_floor_pwm=drive_floor_pwm,
        start_ratio=start_ratio,
    )
    return forward_right + backward_left


def _target_name(target: str) -> str:
    return "left" if str(target).strip().lower() == "l" else "right"


def _physical_summary_from_actions(actions: Iterable[dict]) -> str:
    parts = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        target = str(action.get("target") or "").strip().lower()
        if target not in {"l", "r"}:
            continue
        raw_action = str(action.get("action") or "").strip().lower()
        direction = _physical_tread_direction(target, raw_action)
        try:
            pwm = int(round(float(action.get("pwm") or 0)))
        except (TypeError, ValueError):
            pwm = 0
        parts.append(f"{_target_name(target)} {direction} pwm={pwm}")
    return ", ".join(parts)


def format_turn_test_pulse(
    pulse: TurnTestPulse,
    *,
    prefix: str = "[MOVE]",
    send_result: dict | None = None,
) -> str:
    actions = pulse.actions
    duration_ms = int(pulse.duration_ms)
    if isinstance(send_result, dict):
        result_actions = send_result.get("actions")
        if isinstance(result_actions, list):
            actions = tuple(dict(action) for action in result_actions if isinstance(action, dict))
        try:
            duration_ms = int(round(float(send_result.get("duration_ms") or duration_ms)))
        except (TypeError, ValueError):
            pass
    physical_summary = _physical_summary_from_actions(actions)
    return (
        f"{prefix} {pulse.phase_name} {pulse.phase_step:02d}/{pulse.phase_steps:02d}: "
        f"cmd={pulse.drive_mode.upper()}+{('RIGHT' if pulse.logical_cmd == 'r' else 'LEFT')} "
        f"t={duration_ms}ms inner={pulse.inner_direction} ratio={pulse.inner_ratio:.2f}; "
        f"{physical_summary}"
    )


def run_turn_test(
    *,
    robot=None,
    execute: bool = False,
    sequence: Iterable[TurnTestPulse] | None = None,
    pause_s: float = 0.0,
    settle_s: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = print,
    show_wire: bool = False,
) -> dict:
    pulses = list(sequence) if sequence is not None else build_turn_test_sequence()
    pause_s = max(0.0, float(pause_s or 0.0))
    settle_s = max(0.0, float(settle_s or 0.0))
    records = []

    if execute and robot is None:
        raise ValueError("execute=True requires a robot instance")

    def emit(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    if execute:
        emit("[MOVE] Sending initial stop before turn test.")
        robot.stop()
        if pause_s > 0.0:
            sleep_fn(pause_s)

    for pulse in pulses:
        prefix = "[MOVE]" if execute else "[DRY-RUN]"
        send_result = None
        if execute:
            send_result = robot.send_custom_actions_pwm(
                pulse.logical_cmd,
                list(pulse.actions),
                duration_ms=int(pulse.duration_ms),
            )
            emit(format_turn_test_pulse(pulse, prefix=prefix, send_result=send_result))
            wait_s = max(0.0, float((send_result or {}).get("duration_ms") or pulse.duration_ms) / 1000.0 + settle_s)
            if wait_s > 0.0:
                sleep_fn(wait_s)
            if show_wire and isinstance(send_result, dict) and send_result.get("wire_text"):
                emit(f"[WIRE DEBUG] {send_result['wire_text']}")
        else:
            emit(format_turn_test_pulse(pulse, prefix=prefix))

        records.append(
            {
                "phase_name": pulse.phase_name,
                "logical_cmd": pulse.logical_cmd,
                "drive_mode": pulse.drive_mode,
                "phase_step": pulse.phase_step,
                "duration_ms": pulse.duration_ms,
                "outer_target": pulse.outer_target,
                "inner_target": pulse.inner_target,
                "outer_pwm": pulse.outer_pwm,
                "inner_pwm": pulse.inner_pwm,
                "inner_direction": pulse.inner_direction,
                "inner_ratio": pulse.inner_ratio,
                "actions": [dict(action) for action in pulse.actions],
                "send_result": send_result,
            }
        )

    if execute:
        emit("[MOVE] Sending final stop after turn test.")
        robot.stop()

    return {
        "ok": True,
        "execute": bool(execute),
        "pulses": records,
    }
