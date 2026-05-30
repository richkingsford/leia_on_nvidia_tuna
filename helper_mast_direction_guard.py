#!/usr/bin/env python3
"""Observed-effect guardrails for Leia's mast spool."""

from __future__ import annotations


REVERSAL_STATUSES = frozenset({"target_regressed", "cmd_direction_reversed"})


def classify_mast_y_effect(
    *,
    before_y_mm,
    after_y_mm,
    cmd: str | None = None,
    target_y_mm=None,
    min_delta_mm: float = 1.0,
) -> dict:
    """Classify what actually happened after a mast act.

    The mast fishing line can wrap around the drum and make a logical command
    stall briefly or move the opposite way.  This helper treats command labels
    as intent and judges the observed y telemetry after the act.
    """
    try:
        before_y = float(before_y_mm)
        after_y = float(after_y_mm)
    except (TypeError, ValueError):
        return {"status": "missing_y", "observed": False}

    threshold = max(0.0, float(min_delta_mm))
    delta = float(after_y - before_y)
    detail = {
        "before_y_mm": float(before_y),
        "after_y_mm": float(after_y),
        "delta_y_signed_mm": float(delta),
        "delta_y_abs_mm": abs(float(delta)),
        "min_delta_mm": float(threshold),
        "cmd": str(cmd or "").strip().lower(),
    }

    try:
        target_y = float(target_y_mm)
    except (TypeError, ValueError):
        target_y = None

    if target_y is not None:
        before_err = abs(float(before_y) - float(target_y))
        after_err = abs(float(after_y) - float(target_y))
        improvement = float(before_err - after_err)
        detail.update(
            {
                "mode": "target_error",
                "target_y_mm": float(target_y),
                "prev_y_error_mm": float(before_err),
                "now_y_error_mm": float(after_err),
                "y_error_improvement_mm": float(improvement),
            }
        )
        if improvement >= threshold:
            detail.update({"status": "target_progress", "observed": True})
            return detail
        if improvement <= -threshold:
            detail.update({"status": "target_regressed", "observed": False})
            return detail
        detail.update({"status": "target_stalled", "observed": False})
        return detail

    cmd_key = detail["cmd"]
    if abs(float(delta)) < threshold:
        detail.update({"mode": "cmd_direction", "status": "cmd_direction_stalled", "observed": False})
        return detail
    if cmd_key == "d":
        observed = delta <= -threshold
    elif cmd_key == "u":
        observed = delta >= threshold
    else:
        observed = True
    detail.update(
        {
            "mode": "cmd_direction",
            "status": "cmd_direction_progress" if observed else "cmd_direction_reversed",
            "observed": bool(observed),
        }
    )
    return detail


def mast_effect_is_reversal(effect: dict | None) -> bool:
    if not isinstance(effect, dict):
        return False
    return str(effect.get("status") or "").strip() in REVERSAL_STATUSES
