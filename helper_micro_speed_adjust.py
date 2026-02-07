from __future__ import annotations

from typing import Any, MutableMapping

import telemetry_robot


def micro_adjust_speed_score(
    state: MutableMapping[str, Any],
    *,
    score_power_pwm: MutableMapping[int, dict],
    metric_value_mm: float | None,
    active: bool,
    sequence_key: str | None = None,
    score_to_adjust: int = telemetry_robot.SPEED_SCORE_MIN,
    acts: int = 3,
    threshold_mm: float = 0.5,
    increase_scale: float = 1.01,
    decrease_scale: float = 0.99,
    min_pwm: int | None = None,
    max_pwm: int = 255,
    metric_label: str = "x_axis",
) -> str | None:
    """
    Micro dynamic tuning for a single speed score.

    After `acts` active samples, compare the metric delta across those acts:
      - abs(delta) < threshold_mm  => multiply PWM by increase_scale (default +1%)
      - abs(delta) > threshold_mm  => multiply PWM by decrease_scale (default -1%)

    Returns a log-friendly string when an adjustment is applied, else None.
    """
    if not isinstance(state, dict):
        return None

    samples = state.get("samples")
    if not isinstance(samples, list):
        samples = []
        state["samples"] = samples

    if not active:
        samples.clear()
        state["sequence_key"] = None
        return None

    if state.get("sequence_key") != sequence_key:
        samples.clear()
        state["sequence_key"] = sequence_key

    try:
        metric = float(metric_value_mm)
    except (TypeError, ValueError):
        samples.clear()
        return None

    try:
        acts = int(acts)
    except (TypeError, ValueError):
        acts = 0
    if acts < 1:
        samples.clear()
        return None

    samples.append(metric)
    window = acts + 1
    if len(samples) > window:
        del samples[:-window]
    if len(samples) < window:
        return None

    try:
        threshold_val = float(threshold_mm)
    except (TypeError, ValueError):
        threshold_val = 0.0

    delta = abs(samples[-1] - samples[0])
    scale = None
    direction = None
    if threshold_val > 0.0 and delta < threshold_val:
        scale = float(increase_scale)
        direction = "up"
    elif threshold_val > 0.0 and delta > threshold_val:
        scale = float(decrease_scale)
        direction = "down"

    # Reset window for the next decision (non-overlapping windows).
    samples[:] = [samples[-1]]

    if scale is None:
        return None

    entry = score_power_pwm.get(int(score_to_adjust))
    if not isinstance(entry, dict):
        return None

    try:
        pwm_before = int(entry.get("pwm", 0))
    except (TypeError, ValueError):
        pwm_before = 0

    pwm_after = int(round(pwm_before * scale))
    if min_pwm is not None:
        try:
            pwm_min = int(round(min_pwm))
        except (TypeError, ValueError):
            pwm_min = 0
        pwm_after = max(pwm_min, pwm_after)
    pwm_after = max(0, min(int(max_pwm), pwm_after))

    power_after = telemetry_robot.pwm_to_power(pwm_after)
    if power_after is None:
        power_after = 0.0

    entry["pwm"] = pwm_after
    entry["power"] = float(power_after)

    pct = int(round(abs(scale - 1.0) * 100.0))
    sign = "+" if direction == "up" else "-"
    return (
        f"[SPEED] Micro-adjust {int(score_to_adjust)}% "
        f"({metric_label} Δ{delta:.2f}mm / {acts} acts, thr {threshold_val:.2f}mm) "
        f"-> {sign}{pct}% pwm {pwm_before}->{pwm_after}"
    )
