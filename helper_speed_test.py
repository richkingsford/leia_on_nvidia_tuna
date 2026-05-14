#!/usr/bin/env python3
"""Build and run Leia's straight-line speed ramp test."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import telemetry_robot as telemetry_robot_module


DEFAULT_PHASE_DURATION_S = 2.0
DEFAULT_INTERVAL_MS = 150
DEFAULT_START_SCORE = 1
DEFAULT_END_SCORE = 2
DEFAULT_START_PWM_SCALE = 1.0


@dataclass(frozen=True)
class SpeedTestPulse:
    phase_name: str
    cmd: str
    phase_step: int
    phase_steps: int
    t_ms: int
    interval_ms: int
    score: int
    model_power: float
    model_pwm: int
    model_duration_ms: int


def _positive_int(value, fallback: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback)
    return parsed if parsed > 0 else int(fallback)


def _phase_steps(phase_duration_s: float, interval_ms: int) -> int:
    try:
        phase_ms = max(1.0, float(phase_duration_s) * 1000.0)
    except (TypeError, ValueError):
        phase_ms = float(DEFAULT_PHASE_DURATION_S * 1000.0)
    return max(1, int(math.ceil(float(phase_ms) / float(max(1, int(interval_ms))))))


def _score_for_step(step_idx: int, phase_steps: int, *, start_score: int, end_score: int) -> int:
    if int(phase_steps) <= 1:
        return int(telemetry_robot_module.normalize_speed_score(end_score))
    frac = float(step_idx) / float(int(phase_steps) - 1)
    raw = float(start_score) + (float(end_score) - float(start_score)) * frac
    return int(telemetry_robot_module.normalize_speed_score(raw))


def _uno_floor_pwm(pwm: int | float | None) -> int:
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        pwm_val = 0
    pwm_val = int(telemetry_robot_module.clamp_pwm(pwm_val))
    if pwm_val <= 0:
        return 0
    percent = int(math.ceil(float(pwm_val) * 100.0 / float(max(1, int(telemetry_robot_module.MAX_PWM)))))
    percent = max(1, min(100, int(percent)))
    return int((int(percent) * int(telemetry_robot_module.MAX_PWM)) / 100)


def _pulse_for_step(
    *,
    phase_name: str,
    cmd: str,
    step_idx: int,
    phase_steps: int,
    interval_ms: int,
    start_score: int,
    end_score: int,
    start_pwm_scale: float,
) -> SpeedTestPulse:
    score = _score_for_step(
        int(step_idx),
        int(phase_steps),
        start_score=int(start_score),
        end_score=int(end_score),
    )
    power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
    _floor_power, floor_pwm, _floor_score, _floor_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
        cmd,
        start_score,
    )
    floor_pwm = _uno_floor_pwm(floor_pwm)
    try:
        scaled_floor_pwm = int(math.ceil(float(floor_pwm) * max(0.0, float(start_pwm_scale))))
    except (TypeError, ValueError):
        scaled_floor_pwm = int(floor_pwm)
    scaled_floor_pwm = int(telemetry_robot_module.clamp_pwm(scaled_floor_pwm))
    if int(scaled_floor_pwm) > int(pwm):
        pwm = int(scaled_floor_pwm)
        power = telemetry_robot_module.pwm_to_power(int(pwm)) or float(power)
    return SpeedTestPulse(
        phase_name=str(phase_name),
        cmd=str(cmd),
        phase_step=int(step_idx + 1),
        phase_steps=int(phase_steps),
        t_ms=int(step_idx) * int(interval_ms),
        interval_ms=int(interval_ms),
        score=int(score_used),
        model_power=float(power),
        model_pwm=int(pwm),
        model_duration_ms=int(duration_ms),
    )


def build_speed_test_sequence(
    *,
    phase_duration_s: float = DEFAULT_PHASE_DURATION_S,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    end_score: int = DEFAULT_END_SCORE,
    start_pwm_scale: float = DEFAULT_START_PWM_SCALE,
    reverse_mode: str = "backward",
) -> list[SpeedTestPulse]:
    interval = _positive_int(interval_ms, DEFAULT_INTERVAL_MS)
    steps = _phase_steps(phase_duration_s, interval)
    ceiling_score = telemetry_robot_module.normalize_speed_score(end_score, default=DEFAULT_END_SCORE)
    phases = [
        ("forward ramp up", "f", DEFAULT_START_SCORE, ceiling_score),
    ]
    reverse_key = str(reverse_mode or "backward").strip().lower()
    if reverse_key == "down":
        phases.append(("forward ramp down", "f", ceiling_score, DEFAULT_START_SCORE))
    else:
        phases.append(("backward ramp up", "b", DEFAULT_START_SCORE, ceiling_score))

    sequence: list[SpeedTestPulse] = []
    for phase_name, cmd, start_score, end_score in phases:
        for step_idx in range(int(steps)):
            sequence.append(
                _pulse_for_step(
                    phase_name=phase_name,
                    cmd=cmd,
                    step_idx=step_idx,
                    phase_steps=steps,
                    interval_ms=interval,
                    start_score=start_score,
                    end_score=end_score,
                    start_pwm_scale=float(start_pwm_scale),
                )
            )
    return sequence


def _table_row(
    pulse: SpeedTestPulse,
    *,
    global_step: int,
    send_result: dict | None = None,
) -> list[str]:
    sent_power = pulse.model_power
    sent_pwm = pulse.model_pwm
    sent_ms = pulse.interval_ms
    percent = ""
    if isinstance(send_result, dict):
        try:
            sent_power = float(send_result.get("power"))
        except (TypeError, ValueError):
            sent_power = pulse.model_power
        try:
            sent_pwm = int(round(float(send_result.get("pwm"))))
        except (TypeError, ValueError):
            sent_pwm = pulse.model_pwm
        try:
            sent_ms = int(round(float(send_result.get("duration_ms"))))
        except (TypeError, ValueError):
            sent_ms = pulse.interval_ms
        if send_result.get("percent") is not None:
            percent = str(int(send_result.get("percent")))
    return [
        str(int(global_step)),
        str(int(pulse.t_ms)),
        pulse.phase_name,
        pulse.cmd.upper(),
        str(int(pulse.score)),
        f"{float(sent_power):.3f}",
        str(int(sent_pwm)),
        percent,
        str(int(pulse.interval_ms)),
        str(int(sent_ms)),
    ]


def format_speed_test_table(
    pulses: Iterable[SpeedTestPulse],
    *,
    records: Iterable[dict] | None = None,
) -> str:
    pulse_list = list(pulses)
    record_list = list(records or [])
    headers = ["#", "t_ms", "phase", "cmd", "score", "power", "pwm", "pct", "cadence_ms", "sent_ms"]
    rows = [headers]
    for idx, pulse in enumerate(pulse_list):
        send_result = None
        if idx < len(record_list) and isinstance(record_list[idx], dict):
            send_result = record_list[idx].get("send_result")
        rows.append(_table_row(pulse, global_step=idx + 1, send_result=send_result))

    widths = [0 for _ in headers]
    for row in rows:
        for col_idx, value in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(str(value)))

    def fmt(row: list[str]) -> str:
        return "  ".join(str(value).rjust(widths[col_idx]) for col_idx, value in enumerate(row))

    lines = [fmt(rows[0]), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows[1:])
    return "\n".join(lines)


def run_speed_test(
    *,
    robot=None,
    execute: bool = False,
    sequence: Iterable[SpeedTestPulse] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    log_fn: Callable[[str], None] | None = print,
) -> dict:
    pulses = list(sequence) if sequence is not None else build_speed_test_sequence()
    records = []
    if execute and robot is None:
        raise ValueError("execute=True requires a robot instance")

    def emit(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    if execute:
        emit("[SPEED] Sending initial stop, then starting continuous ramp.")
        robot.stop()
    else:
        emit("[DRY-RUN] Planned speed table:")
        emit(format_speed_test_table(pulses))

    start_s = monotonic_fn()
    cadence_s = (float(pulses[0].interval_ms) / 1000.0) if pulses else 0.0
    for idx, pulse in enumerate(pulses):
        send_result = None
        if execute:
            send_result = robot.send_command_pwm(
                pulse.cmd,
                int(pulse.model_pwm),
                duration_ms=int(pulse.interval_ms),
            )
        records.append(
            {
                "phase_name": pulse.phase_name,
                "cmd": pulse.cmd,
                "score": pulse.score,
                "model_power": pulse.model_power,
                "model_pwm": pulse.model_pwm,
                "interval_ms": pulse.interval_ms,
                "send_result": send_result,
            }
        )
        next_start_s = float(start_s) + (float(idx + 1) * float(cadence_s))
        remaining_s = float(next_start_s) - float(monotonic_fn())
        if remaining_s > 0.0:
            sleep_fn(remaining_s)

    if execute:
        robot.stop()
        emit("[SPEED] Final stop sent. Actual sent table:")
        emit(format_speed_test_table(pulses, records=records))

    return {
        "ok": True,
        "execute": bool(execute),
        "pulses": records,
    }
