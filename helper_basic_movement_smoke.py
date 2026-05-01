#!/usr/bin/env python3
"""Slow, deliberate robot movement smoke test helpers.

The sequence references the old manual hotkey rows, but does not implement a
hotkey listener. Motion values stay sourced from world_model_robot.json through
telemetry_robot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import telemetry_robot as telemetry_robot_module
from helper_manual_turn_arc_assist import build_manual_turn_arc_plan


BASIC_MOVEMENT_REFERENCE_HOTKEYS = ("r", "f", "q", "e", "o", "k")
TURN_ARC_REFERENCE_HOTKEYS = frozenset({"q", "e"})
SEND_MODE_COMMAND_PWM = "command_pwm"
SEND_MODE_TURN_ARC = "turn_arc"

_PULSE_LABELS = {
    "r": "slow forward nudge (both treads)",
    "f": "slow backward nudge (both treads)",
    "q": "slow left turn nudge (right tread)",
    "e": "slow right turn nudge (left tread)",
    "o": "slow mast up nudge",
    "k": "slow mast down nudge",
}


@dataclass
class BasicMovementPulse:
    reference_hotkey: str
    label: str
    cmd: str
    score: int
    power: float
    pwm: int
    duration_ms: int
    send_mode: str = SEND_MODE_COMMAND_PWM
    turn_arc_plan: dict | None = None


def _positive_int(value, fallback=None):
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def _power_for_pwm(pwm: int, fallback: float) -> float:
    power = telemetry_robot_module.pwm_to_power(pwm)
    if power is None:
        return float(fallback or 0.0)
    return float(power)


def _pulse_from_reference_hotkey(
    hotkey: str,
    *,
    use_turn_arc_profiles: bool = True,
) -> BasicMovementPulse:
    key = str(hotkey or "").strip().lower()
    rows = telemetry_robot_module.HOTKEY_SPEED_SCORES
    row = rows.get(key) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise KeyError(f"No world-model hotkey row found for {key!r}")

    cmd = str(row.get("cmd") or "").strip().lower()
    if cmd not in {"f", "b", "l", "r", "u", "d"}:
        raise ValueError(f"Unsupported movement command for {key!r}: {cmd!r}")

    score = telemetry_robot_module.normalize_speed_score(
        row.get("score"),
        default=telemetry_robot_module.SPEED_SCORE_MIN,
    )
    power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)

    row_pwm = _positive_int(row.get("pwm"))
    if row_pwm is not None:
        pwm = telemetry_robot_module.clamp_pwm(row_pwm)
        power = _power_for_pwm(int(pwm), float(power))

    row_duration_ms = _positive_int(row.get("duration_ms"))
    if row_duration_ms is not None:
        duration_ms = int(row_duration_ms)

    send_mode = SEND_MODE_COMMAND_PWM
    turn_arc_plan = None
    if use_turn_arc_profiles and key in TURN_ARC_REFERENCE_HOTKEYS and cmd in {"l", "r"}:
        turn_arc_plan = build_manual_turn_arc_plan(
            hotkey=key,
            cmd=cmd,
            score=int(score_used),
            hold_duration_ms=int(duration_ms),
            pwm_override=int(pwm),
        )
        if isinstance(turn_arc_plan, dict):
            send_mode = SEND_MODE_TURN_ARC
            duration_ms = int(turn_arc_plan.get("duration_ms") or duration_ms)
            pwm = int(turn_arc_plan.get("outer_pwm") or pwm)
            power = _power_for_pwm(int(pwm), float(power))

    return BasicMovementPulse(
        reference_hotkey=key,
        label=str(_PULSE_LABELS.get(key) or f"slow {cmd} nudge"),
        cmd=cmd,
        score=int(score_used),
        power=float(power),
        pwm=int(pwm),
        duration_ms=int(duration_ms),
        send_mode=send_mode,
        turn_arc_plan=turn_arc_plan,
    )


def build_basic_movement_sequence(
    reference_hotkeys: Iterable[str] = BASIC_MOVEMENT_REFERENCE_HOTKEYS,
    *,
    use_turn_arc_profiles: bool = True,
) -> list[BasicMovementPulse]:
    return [
        _pulse_from_reference_hotkey(
            hotkey,
            use_turn_arc_profiles=use_turn_arc_profiles,
        )
        for hotkey in reference_hotkeys
    ]


def format_basic_movement_pulse(pulse: BasicMovementPulse, *, prefix: str = "[MOVE]") -> str:
    mode = "arc" if pulse.send_mode == SEND_MODE_TURN_ARC else "direct"
    return (
        f"{prefix} {pulse.reference_hotkey.upper()} -> {pulse.label}: "
        f"cmd={pulse.cmd.upper()} score={int(pulse.score)} "
        f"pwm={int(pulse.pwm)} t={int(pulse.duration_ms)}ms mode={mode}"
    )


def _send_pulse(robot, pulse: BasicMovementPulse) -> dict:
    if pulse.send_mode == SEND_MODE_TURN_ARC and isinstance(pulse.turn_arc_plan, dict):
        return robot.send_custom_actions_pwm(
            pulse.cmd,
            pulse.turn_arc_plan.get("actions") or [],
            duration_ms=int(pulse.duration_ms),
        )
    return robot.send_command_pwm(
        pulse.cmd,
        int(pulse.pwm),
        duration_ms=int(pulse.duration_ms),
    )


def _stop_robot(robot) -> None:
    if robot is None or not hasattr(robot, "stop"):
        return
    robot.stop()


def run_basic_movement_smoke_test(
    *,
    robot=None,
    execute: bool = False,
    reference_hotkeys: Iterable[str] = BASIC_MOVEMENT_REFERENCE_HOTKEYS,
    repeat: int = 1,
    pause_s: float = 0.35,
    settle_s: float = 0.05,
    use_turn_arc_profiles: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = print,
    show_wire: bool = False,
) -> dict:
    sequence = build_basic_movement_sequence(
        reference_hotkeys,
        use_turn_arc_profiles=use_turn_arc_profiles,
    )
    repeat_count = max(1, int(repeat or 1))
    pause_s = max(0.0, float(pause_s or 0.0))
    settle_s = max(0.0, float(settle_s or 0.0))
    records = []

    if execute and robot is None:
        raise ValueError("execute=True requires a robot instance")

    def emit(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    if execute:
        emit("[MOVE] Sending initial stop before smoke test.")
        _stop_robot(robot)
        if pause_s > 0.0:
            sleep_fn(pause_s)

    for repeat_idx in range(repeat_count):
        if repeat_count > 1:
            emit(f"[MOVE] Smoke test pass {repeat_idx + 1}/{repeat_count}.")
        for pulse in sequence:
            prefix = "[MOVE]" if execute else "[DRY-RUN]"
            emit(format_basic_movement_pulse(pulse, prefix=prefix))
            send_result = None
            if execute:
                send_result = _send_pulse(robot, pulse)
                wait_s = max(0.0, float(pulse.duration_ms) / 1000.0 + settle_s)
                if wait_s > 0.0:
                    sleep_fn(wait_s)
                _stop_robot(robot)
                if pause_s > 0.0:
                    sleep_fn(pause_s)
                if show_wire:
                    wire_text = None
                    if isinstance(send_result, dict):
                        wire_text = send_result.get("wire_text")
                    if wire_text:
                        emit(f"[WIRE DEBUG] {wire_text}")
            records.append(
                {
                    "reference_hotkey": pulse.reference_hotkey,
                    "label": pulse.label,
                    "cmd": pulse.cmd,
                    "score": pulse.score,
                    "pwm": pulse.pwm,
                    "duration_ms": pulse.duration_ms,
                    "send_mode": pulse.send_mode,
                    "send_result": send_result,
                }
            )

    if execute:
        emit("[MOVE] Sending final stop after smoke test.")
        _stop_robot(robot)

    return {
        "ok": True,
        "execute": bool(execute),
        "repeat": int(repeat_count),
        "pulses": records,
    }
