#!/usr/bin/env python3
"""Forward micro-adjust assist: move forward while applying slight turn bias."""

from __future__ import annotations

import helper_turn_drive_motion
import telemetry_robot as telemetry_robot_module

from helper_close_gaps import production_turn_drive_curve_plan, x_axis_correction_cmd


DEFAULT_ASSIST_CONFIG = {
    "enabled": False,
    "require_visible": True,
    "x_deadband_mm": 0.0,
    "use_x_tol_as_deadband": False,
    "min_x_err_mm": 0.0,
    "min_reduction_ratio": 0.03,
    "max_reduction_ratio": 0.12,
    "max_x_err_mm_for_full_reduction": 20.0,
    "min_inner_ratio": 0.82,
    "min_pwm_delta": 1,
    "inner_stop_x_err_mm": None,
    "max_score_boost_pct": 0,
    "max_x_err_mm_for_full_score_boost": 20.0,
    "max_dist_err_mm_for_full_score_boost": 100.0,
    "prefer_curve_for_x_axis": False,
    "curve_score_mode": "requested",
    "forward_setup_duration_mode": "max_turn_drive",
    "forward_left_curve_action_note": "FWD+LEFT-CURVE",
    "forward_right_curve_action_note": "FWD+RIGHT-CURVE",
    "backward_left_curve_action_note": "BWD+LEFT-CURVE",
    "backward_right_curve_action_note": "BWD+RIGHT-CURVE",
    "forward_action_note": "FWD+TURN-BIAS",
    "backward_action_note": "BWD+TURN-BIAS",
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


def _pwm_to_percent(pwm):
    return float(telemetry_robot_module._pwm_to_power(pwm) or 0.0) * 100.0


def _dist_forward_progress_allowed(local_gate):
    gate = local_gate if isinstance(local_gate, dict) else {}
    if not bool(gate.get("dist_required", True)):
        return True
    try:
        dist_now = float(gate.get("dist"))
        dist_target = float(gate.get("dist_target"))
    except (TypeError, ValueError):
        return False
    direction = str(gate.get("dist_direction") or "").strip().lower()
    if direction == "low":
        return bool(dist_now > dist_target)
    if direction == "high":
        return False
    return bool(dist_now >= dist_target)


def choose_forward_bias_cmd(*, step_rules, cmd, local_gate_status):
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return cmd
    cfg = _assist_cfg(step_rules)
    if not bool(cfg.get("enabled", False)) or not bool(cfg.get("prefer_forward_for_x_axis", False)):
        return cmd
    gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    if bool(cfg.get("require_visible", True)) and not bool(gate.get("visible")):
        return cmd
    try:
        x_err = abs(float(gate.get("x_err")))
    except (TypeError, ValueError):
        return cmd
    min_x_err = max(0.0, _safe_float(cfg.get("prefer_forward_min_x_err_mm"), 0.0))
    if x_err <= min_x_err:
        return cmd
    if not _dist_forward_progress_allowed(gate):
        return cmd
    return "f"


def _distance_gap_mm(local_gate):
    gate = local_gate if isinstance(local_gate, dict) else {}
    try:
        return abs(float(gate.get("dist")) - float(gate.get("dist_target")))
    except (TypeError, ValueError):
        return 0.0


def _x_axis_turn_cmd_for_assist(cfg, local_gate_status, *, turn_cmd_override=None, override_abs_x_err_mm=None):
    local_gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    try:
        x_err = float(local_gate.get("x_err"))
    except (TypeError, ValueError):
        return None, None, None

    abs_x_err = abs(float(x_err))
    if override_abs_x_err_mm is not None:
        try:
            abs_x_err = max(abs_x_err, abs(float(override_abs_x_err_mm)))
        except (TypeError, ValueError):
            pass
    if abs_x_err <= 0.0:
        return None, None, None

    deadband = max(0.0, _safe_float(cfg.get("x_deadband_mm"), 0.0))
    if bool(cfg.get("use_x_tol_as_deadband", False)):
        try:
            deadband = max(deadband, abs(float(local_gate.get("x_tol"))))
        except (TypeError, ValueError):
            pass
    min_x_err_mm = max(0.0, _safe_float(cfg.get("min_x_err_mm"), 0.0))
    if abs_x_err <= max(deadband, min_x_err_mm):
        return None, None, None

    turn_cmd = str(turn_cmd_override or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        turn_cmd = str(x_axis_correction_cmd(x_err) or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return None, None, None
    return str(turn_cmd), float(x_err), float(abs_x_err)


def _curve_action_note(cfg, *, drive_cmd, turn_cmd):
    drive_key = str(drive_cmd or "").strip().lower()
    turn_key = str(turn_cmd or "").strip().lower()
    drive_word = "forward" if drive_key == "f" else "backward"
    side_word = "right" if turn_key == "r" else "left"
    cfg_key = f"{drive_word}_{side_word}_curve_action_note"
    prefix = "FWD" if drive_key == "f" else "BWD"
    fallback = f"{prefix}+{side_word.upper()}-CURVE"
    return str(cfg.get(cfg_key) or fallback).strip() or fallback


def build_curve_drive_while_turning_plan(
    *,
    step_rules,
    cmd,
    speed_score,
    local_gate_status,
    hold_duration_ms=None,
    metadata=None,
    turn_cmd_override=None,
    override_abs_x_err_mm=None,
    trials_dir=None,
):
    drive_cmd = str(cmd or "").strip().lower()
    if drive_cmd not in {"f", "b"}:
        return None

    cfg = _assist_cfg(step_rules)
    if not bool(cfg.get("enabled", False)) or not bool(cfg.get("prefer_curve_for_x_axis", False)):
        return None

    local_gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    if bool(cfg.get("require_visible", True)) and not bool(local_gate.get("visible")):
        return None

    turn_cmd, x_err, abs_x_err = _x_axis_turn_cmd_for_assist(
        cfg,
        local_gate,
        turn_cmd_override=turn_cmd_override,
        override_abs_x_err_mm=override_abs_x_err_mm,
    )
    if turn_cmd not in {"l", "r"}:
        return None

    drive_mode = "forward" if drive_cmd == "f" else "backward"
    manifest_curve_plan = production_turn_drive_curve_plan(
        cmd=turn_cmd,
        drive_mode=drive_mode,
        current_dist_mm=local_gate.get("dist"),
        x_err_mm=x_err,
        trials_dir=trials_dir,
    )
    if not isinstance(manifest_curve_plan, dict):
        return None

    try:
        requested_score = int(telemetry_robot_module.normalize_speed_score(speed_score))
    except Exception:
        requested_score = int(telemetry_robot_module.SPEED_SCORE_MIN)
    curve_score_mode = str(cfg.get("curve_score_mode") or "requested").strip().lower()
    if curve_score_mode == "manifest":
        try:
            score_val = int(telemetry_robot_module.normalize_speed_score(manifest_curve_plan.get("score")))
        except Exception:
            score_val = int(requested_score)
    else:
        score_val = int(requested_score)

    profile_override = (
        dict(manifest_curve_plan.get("profile_override") or {})
        if isinstance(manifest_curve_plan.get("profile_override"), dict)
        else None
    )
    # At the gap safety floor (minimum score), the trial duration would massively overshoot
    # a tiny dist gap. Drop the duration override so score-based timing applies instead.
    at_gap_floor = score_val <= int(telemetry_robot_module.SPEED_SCORE_MIN)

    if isinstance(profile_override, dict) and (
        manifest_curve_plan.get("duration_override_ms") is None or at_gap_floor
    ) and str(manifest_curve_plan.get("source") or "") == "production_turn_drive_trials_forward":
        duration_mode = str(cfg.get("forward_setup_duration_mode") or "max_turn_drive").strip().lower()
        if duration_mode in helper_turn_drive_motion.VALID_DURATION_MODES:
            profile_override["duration_mode"] = duration_mode

    metadata_out = dict(metadata or {}) if isinstance(metadata, dict) else {}
    metadata_out.update(
        {
            "curve_name": manifest_curve_plan.get("curve_name"),
            "curve_value_mm": manifest_curve_plan.get("curve_value_mm"),
            "curve_trial": manifest_curve_plan.get("trial"),
            "curve_source": manifest_curve_plan.get("source"),
            "curve_drive_mode": drive_mode,
            "curve_turn_cmd": turn_cmd,
        }
    )

    # Don't apply the trial duration override when score is at the gap safety floor.
    effective_hold_ms = (
        None
        if at_gap_floor
        else (manifest_curve_plan.get("duration_override_ms") if hold_duration_ms is None else hold_duration_ms)
    )
    plan = helper_turn_drive_motion.build_turn_drive_motion_plan(
        cmd=turn_cmd,
        score=int(score_val),
        hold_duration_ms=effective_hold_ms,
        pwm_override=manifest_curve_plan.get("pwm_override"),
        profile_override=profile_override,
        drive_mode=drive_mode,
        metadata=metadata_out,
    )
    if not isinstance(plan, dict):
        return None

    plan = dict(plan)
    plan["cmd"] = str(drive_cmd)
    plan["curve_cmd"] = str(turn_cmd)
    plan["turn_cmd"] = str(turn_cmd)
    plan["score_requested"] = int(requested_score)
    plan["score"] = int(score_val)
    plan["x_err_mm"] = float(x_err)
    plan["abs_x_err_mm"] = float(abs_x_err)
    plan["dist_gap_mm"] = float(_distance_gap_mm(local_gate))
    plan["curve_assist"] = True
    plan["curve_plan"] = dict(manifest_curve_plan)
    plan["action_note"] = _curve_action_note(cfg, drive_cmd=drive_cmd, turn_cmd=turn_cmd)
    return plan


def _boosted_score(score, cfg, *, abs_x_err, dist_gap):
    try:
        requested = telemetry_robot_module.normalize_speed_score(score)
    except Exception:
        requested = int(telemetry_robot_module.SPEED_SCORE_MIN)
    max_boost = max(0, int(round(_safe_float(cfg.get("max_score_boost_pct"), 0))))
    if max_boost <= 0:
        return int(requested), 0
    max_x = max(1e-6, _safe_float(cfg.get("max_x_err_mm_for_full_score_boost"), 20.0))
    max_dist = max(1e-6, _safe_float(cfg.get("max_dist_err_mm_for_full_score_boost"), 100.0))
    x_gain = max(0.0, min(1.0, float(abs_x_err) / float(max_x)))
    dist_gain = max(0.0, min(1.0, float(dist_gap) / float(max_dist)))
    boost = int(round(float(max_boost) * min(float(x_gain), float(dist_gain))))
    score_out = telemetry_robot_module.normalize_speed_score(int(requested) + int(boost))
    return int(score_out), int(boost)


def _apply_pwm_gap(outer_pwm, inner_pwm, *, min_pwm_delta=0, min_percent_delta=0.0):
    outer = int(telemetry_robot_module.clamp_pwm(outer_pwm))
    inner = int(telemetry_robot_module.clamp_pwm(inner_pwm))
    try:
        min_pwm_delta = max(0, int(round(float(min_pwm_delta or 0))))
    except (TypeError, ValueError):
        min_pwm_delta = 0
    if min_pwm_delta > 0 and (outer - inner) < min_pwm_delta:
        inner = max(0, outer - min_pwm_delta)
    try:
        min_percent_delta = max(0.0, float(min_percent_delta or 0.0))
    except (TypeError, ValueError):
        min_percent_delta = 0.0
    if min_percent_delta > 0.0:
        while inner > 0 and (_pwm_to_percent(outer) - _pwm_to_percent(inner)) < min_percent_delta:
            inner -= 1
    return int(outer), int(telemetry_robot_module.clamp_pwm(inner))


def build_drive_while_turning_plan(
    *,
    step_rules,
    cmd,
    speed_score,
    local_gate_status,
    hold_duration_ms,
    metadata=None,
    turn_cmd_override=None,
    override_abs_x_err_mm=None,
    min_percent_delta=0.0,
):
    drive_cmd = str(cmd or "").strip().lower()
    if drive_cmd not in {"f", "b"}:
        return None

    cfg = _assist_cfg(step_rules)
    if not bool(cfg.get("enabled", False)):
        return None

    local_gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    if bool(cfg.get("require_visible", True)) and not bool(local_gate.get("visible")):
        return None

    turn_cmd, x_err, abs_x_err = _x_axis_turn_cmd_for_assist(
        cfg,
        local_gate,
        turn_cmd_override=turn_cmd_override,
        override_abs_x_err_mm=override_abs_x_err_mm,
    )
    if turn_cmd not in {"l", "r"}:
        return None


    dist_gap = _distance_gap_mm(local_gate)
    score_requested = int(telemetry_robot_module.normalize_speed_score(speed_score))
    score_val, score_boost = _boosted_score(
        score_requested,
        cfg,
        abs_x_err=abs_x_err,
        dist_gap=dist_gap,
    )
    try:
        _power, base_pwm, _score_used, model_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
            drive_cmd,
            int(score_val),
        )
    except Exception:
        return None

    base_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm or 0))))
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

    outer_pwm, inner_pwm = _apply_pwm_gap(
        outer_pwm,
        inner_pwm,
        min_pwm_delta=cfg.get("min_pwm_delta", 1),
        min_percent_delta=min_percent_delta,
    )

    inner_action_override = None
    inner_stop_x_err = _safe_float(cfg.get("inner_stop_x_err_mm"), -1.0)
    if inner_stop_x_err > 0.0 and abs_x_err >= float(inner_stop_x_err):
        inner_pwm = 0
        inner_ratio = 0.0
        inner_action_override = "s"

    if drive_cmd == "f":
        left_action = "b"
        right_action = "f"
        action_note = (
            str(cfg.get("forward_action_note") or cfg.get("action_note") or "FWD+TURN-BIAS").strip()
            or "FWD+TURN-BIAS"
        )
    else:
        left_action = "f"
        right_action = "b"
        action_note = (
            str(cfg.get("backward_action_note") or "BWD+TURN-BIAS").strip()
            or "BWD+TURN-BIAS"
        )

    if turn_cmd == "l":
        actions = [
            {"target": "l", "action": inner_action_override or left_action, "pwm": int(inner_pwm)},
            {"target": "r", "action": right_action, "pwm": int(outer_pwm)},
        ]
    else:
        actions = [
            {"target": "l", "action": left_action, "pwm": int(outer_pwm)},
            {"target": "r", "action": inner_action_override or right_action, "pwm": int(inner_pwm)},
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

    plan = {
        "enabled": True,
        "cmd": str(drive_cmd),
        "score_requested": int(score_requested),
        "score": int(score_val),
        "score_boost_pct": int(score_boost),
        "duration_ms": int(duration_ms),
        "base_pwm": int(base_pwm),
        "inner_pwm": int(inner_pwm),
        "outer_pwm": int(outer_pwm),
        "inner_ratio": float(inner_ratio),
        "outer_ratio": float(outer_ratio),
        "turn_cmd": str(turn_cmd),
        "x_err_mm": float(x_err),
        "dist_gap_mm": float(dist_gap),
        "action_note": str(action_note),
        "actions": actions,
    }
    if isinstance(metadata, dict) and metadata:
        plan["metadata"] = dict(metadata)
    return plan


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
    return build_drive_while_turning_plan(
        step_rules=step_rules,
        cmd="f",
        speed_score=speed_score,
        local_gate_status=local_gate_status,
        hold_duration_ms=hold_duration_ms,
        metadata=metadata,
    )
