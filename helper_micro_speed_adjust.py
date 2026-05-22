#!/usr/bin/env python3
"""Tiny reusable speed-map nudges from observed motion deltas."""

from __future__ import annotations

import telemetry_robot


def _coerce_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback)


def micro_adjust_speed_score(
    state: dict,
    *,
    score_power_pwm: dict,
    metric_value_mm,
    active: bool,
    acts: int,
    threshold_mm: float,
    increase_scale: float,
    decrease_scale: float,
    min_pwm: int = 1,
    max_pwm: int = 255,
):
    """Nudge score-1 PWM after a small window of observed metric movement."""
    if not bool(active) or not isinstance(score_power_pwm, dict):
        if isinstance(state, dict):
            state.pop("samples", None)
        return None
    samples = state.setdefault("samples", [])
    samples.append(_coerce_float(metric_value_mm))
    required = max(2, int(acts) + 1)
    if len(samples) < required:
        return None

    delta = abs(float(samples[-1]) - float(samples[0]))
    scale = float(increase_scale) if delta < float(threshold_mm) else float(decrease_scale)
    min_pwm_val = max(1, _coerce_int(min_pwm, 1))
    max_pwm_val = max(min_pwm_val, min(255, _coerce_int(max_pwm, 255)))

    changed = {}
    for score, row in score_power_pwm.items():
        if not isinstance(row, dict):
            continue
        old_pwm = _coerce_int(row.get("pwm"), 0)
        if old_pwm <= 0:
            continue
        new_pwm = max(min_pwm_val, min(max_pwm_val, _coerce_int(float(old_pwm) * scale, old_pwm)))
        row["pwm"] = int(new_pwm)
        row["power"] = float(telemetry_robot._pwm_to_power(int(new_pwm)) or 0.0)
        changed[score] = {"old_pwm": int(old_pwm), "new_pwm": int(new_pwm)}

    state["samples"] = [float(samples[-1])]
    if not changed:
        return None
    return {
        "reason": "micro_speed_adjust",
        "delta_mm": float(delta),
        "threshold_mm": float(threshold_mm),
        "scale": float(scale),
        "changed": changed,
    }
