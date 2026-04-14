#!/usr/bin/env python3
"""Forward micro-adjust assist: move forward while applying slight turn bias."""

from __future__ import annotations

import telemetry_robot as telemetry_robot_module

from helper_close_gaps import x_axis_correction_cmd


DEFAULT_ASSIST_CONFIG = {
    "enabled": True,
    "require_visible": True,
    "x_deadband_mm": 0.5,
    "use_x_tol_as_deadband": True,
    "min_x_err_mm": 0.0,
    "min_reduction_ratio": 0.03,
    "max_reduction_ratio": 0.12,
    "max_x_err_mm_for_full_reduction": 20.0,
    "min_inner_ratio": 0.82,
    "action_note": "FWD+TURN-BIAS",
}


def _assist_cfg(step_rules):
    cfg = dict(DEFAULT_ASSIST_CONFIG)
    rules = step_rules if isinstance(step_rules, dict) else {}
    align_policy = rules.get("align_policy") if isinstance(rules.get("align_policy"), dict) else {}
    raw = (
        align_policy.get("forward_while_turning_assist")
        if isinstance(align_policy.get("forward_while_turning_assist"), dict)
        else {}
    )
    cfg.update(raw)
    return cfg


def _safe_float(val, fallback):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(fallback)


def build_forward_while_turning_plan(
    *,
    step_rules,
    cmd,
    speed_score,
    local_gate_status,
    hold_duration_ms,
    metadata=None,
):
    if str(cmd or "").strip().lower() != "f":
        return None

    cfg = _assist_cfg(step_rules)
    if not bool(cfg.get("enabled", True)):
        return None

    local_gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    if bool(cfg.get("require_visible", True)) and not bool(local_gate.get("visible")):
        return None

    x_err = local_gate.get("x_err")
    try:
        x_err = float(x_err)
    except (TypeError, ValueError):
        return None

    abs_x_err = abs(float(x_err))
    deadband = max(0.0, _safe_float(cfg.get("x_deadband_mm"), 0.5))
    if bool(cfg.get("use_x_tol_as_deadband", True)):
        try:
            deadband = max(deadband, abs(float(local_gate.get("x_tol"))))
        except (TypeError, ValueError):
            pass
    min_x_err_mm = max(0.0, _safe_float(cfg.get("min_x_err_mm"), 0.0))
    if abs_x_err <= max(deadband, min_x_err_mm):
        return None

    turn_cmd = str(x_axis_correction_cmd(x_err) or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return None

    try:
        score_val = telemetry_robot_module.normalize_speed_score(speed_score)
    except Exception:
        return None
    try:
        _power, base_pwm, _score_used, model_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
            "f",
            int(score_val),
        )
    except Exception:
        return None

    try:
        base_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm))))
    except (TypeError, ValueError):
        return None
    if base_pwm is None or int(base_pwm) <= 0:
        return None

    min_reduction = max(0.0, min(0.95, _safe_float(cfg.get("min_reduction_ratio"), 0.03)))
    max_reduction = max(min_reduction, min(0.95, _safe_float(cfg.get("max_reduction_ratio"), 0.12)))
    max_x_err = max(1e-6, _safe_float(cfg.get("max_x_err_mm_for_full_reduction"), 20.0))
    min_inner_ratio = max(0.0, min(1.0, _safe_float(cfg.get("min_inner_ratio"), 0.82)))

    gain = max(0.0, min(1.0, abs_x_err / max_x_err))
    reduction = float(min_reduction) + (float(max_reduction) - float(min_reduction)) * float(gain)
    inner_ratio = max(float(min_inner_ratio), 1.0 - float(reduction))
    outer_ratio = 1.0

    outer_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(outer_ratio))))
    inner_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(inner_ratio))))
    if outer_pwm is None or int(outer_pwm) <= 0:
        return None
    if inner_pwm is None:
        inner_pwm = 0

    # Forward motion uses asymmetric wire actions on Leia's drivetrain map.
    left_action = "b"
    right_action = "f"

    if turn_cmd == "l":
        actions = [
            {"target": "l", "action": left_action, "pwm": int(inner_pwm)},
            {"target": "r", "action": right_action, "pwm": int(outer_pwm)},
        ]
    else:
        actions = [
            {"target": "l", "action": left_action, "pwm": int(outer_pwm)},
            {"target": "r", "action": right_action, "pwm": int(inner_pwm)},
        ]

    duration_ms = None
    try:
        if hold_duration_ms is not None:
            duration_ms = max(1, int(round(float(hold_duration_ms))))
    except (TypeError, ValueError):
        duration_ms = None
    if duration_ms is None:
        try:
            duration_ms = max(1, int(round(float(model_duration_ms))))
        except (TypeError, ValueError):
            return None

    action_note = str(cfg.get("action_note") or "").strip() or "FWD+TURN-BIAS"
    plan = {
        "enabled": True,
        "cmd": "f",
        "score": int(score_val),
        "duration_ms": int(duration_ms),
        "base_pwm": int(base_pwm),
        "inner_pwm": int(inner_pwm),
        "outer_pwm": int(outer_pwm),
        "inner_ratio": float(inner_ratio),
        "outer_ratio": float(outer_ratio),
        "turn_cmd": str(turn_cmd),
        "x_err_mm": float(x_err),
        "action_note": str(action_note),
        "actions": actions,
    }
    if isinstance(metadata, dict) and metadata:
        plan["metadata"] = dict(metadata)
    return plan
