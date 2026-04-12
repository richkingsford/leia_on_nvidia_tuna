#!/usr/bin/env python3
"""Generic config-driven turn+drive custom tread motion plans."""

from __future__ import annotations

import json
from pathlib import Path

import telemetry_robot as telemetry_robot_module


ASSIST_CONFIG_KEY = "turn_drive_motion_assist"
DEFAULT_ASSIST_CONFIG = {
    "enabled": True,
    "profiles": {
        "forward_pivot": {
            "drive_mode": "forward",
            "inner_ratio": 0.0,
            "outer_ratio": 1.0,
            "duration_mode": "turn",
            "action_note": "TURN+FWD",
        },
        "backward_pivot": {
            "drive_mode": "backward",
            "inner_ratio": 0.0,
            "outer_ratio": 1.0,
            "duration_mode": "turn",
            "action_note": "TURN+BWD",
        },
    },
}

VALID_DURATION_MODES = {"turn", "drive", "max_turn_drive"}


def _robot_model_path(path: Path | None = None) -> Path:
    if isinstance(path, Path):
        return path
    candidate = getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", None)
    if isinstance(candidate, Path):
        return candidate
    return Path(__file__).resolve().parent / "world_model_robot.json"


def _load_raw_model(path: Path | None = None) -> dict:
    model_path = _robot_model_path(path)
    if not model_path.exists():
        return {}
    try:
        payload = json.loads(model_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_turn_drive_motion_config(path: Path | None = None) -> dict:
    payload = _load_raw_model(path)
    raw = payload.get(ASSIST_CONFIG_KEY)
    config = {
        "enabled": bool(DEFAULT_ASSIST_CONFIG.get("enabled", True)),
        "profiles": {
            str(key): dict(value)
            for key, value in (DEFAULT_ASSIST_CONFIG.get("profiles") or {}).items()
            if isinstance(value, dict)
        },
    }
    if isinstance(raw, dict):
        config["enabled"] = bool(raw.get("enabled", config["enabled"]))
        raw_profiles = raw.get("profiles")
        if isinstance(raw_profiles, dict):
            normalized_profiles = {}
            for profile_name, value in raw_profiles.items():
                profile_key = str(profile_name or "").strip()
                if not profile_key or not isinstance(value, dict):
                    continue
                base_profile = dict((config.get("profiles") or {}).get(profile_key) or {})
                profile = dict(base_profile)
                drive_mode = str(value.get("drive_mode", profile.get("drive_mode") or "")).strip().lower()
                if drive_mode in {"forward", "backward"}:
                    profile["drive_mode"] = drive_mode
                if "inner_ratio" in value:
                    try:
                        profile["inner_ratio"] = max(0.0, min(1.0, float(value.get("inner_ratio"))))
                    except (TypeError, ValueError):
                        pass
                if "outer_ratio" in value:
                    try:
                        profile["outer_ratio"] = max(0.0, min(1.0, float(value.get("outer_ratio"))))
                    except (TypeError, ValueError):
                        pass
                duration_mode = str(value.get("duration_mode", profile.get("duration_mode") or "")).strip().lower()
                if duration_mode in VALID_DURATION_MODES:
                    profile["duration_mode"] = duration_mode
                action_note = str(value.get("action_note", profile.get("action_note") or "")).strip()
                if action_note:
                    profile["action_note"] = action_note
                normalized_profiles[profile_key] = profile
            if normalized_profiles:
                config["profiles"] = normalized_profiles
    return config


def _normalized_profile(
    *,
    config: dict | None = None,
    profile_name: str | None = None,
    profile_override: dict | None = None,
    drive_mode: str | None = None,
) -> tuple[str | None, dict | None]:
    if isinstance(profile_override, dict):
        raw_profile = dict(profile_override)
        resolved_name = (
            str(profile_name or raw_profile.get("profile_name") or "").strip() or None
        )
    else:
        assist = dict(config) if isinstance(config, dict) else load_turn_drive_motion_config()
        if not bool(assist.get("enabled")):
            return None, None
        profiles = assist.get("profiles") if isinstance(assist.get("profiles"), dict) else {}
        resolved_name = str(profile_name or "").strip() or None
        raw_profile = profiles.get(resolved_name) if resolved_name else None
        if not isinstance(raw_profile, dict) and drive_mode:
            drive_mode_key = str(drive_mode or "").strip().lower()
            for candidate_name, candidate_profile in profiles.items():
                if not isinstance(candidate_profile, dict):
                    continue
                if str(candidate_profile.get("drive_mode") or "").strip().lower() == drive_mode_key:
                    resolved_name = str(candidate_name)
                    raw_profile = candidate_profile
                    break
        if not isinstance(raw_profile, dict):
            return None, None

    profile = dict(raw_profile)
    drive_mode_key = str(drive_mode or profile.get("drive_mode") or "").strip().lower()
    if drive_mode_key not in {"forward", "backward"}:
        return None, None
    try:
        inner_ratio = max(0.0, min(1.0, float(profile.get("inner_ratio", 0.0))))
    except (TypeError, ValueError):
        inner_ratio = 0.0
    try:
        outer_ratio = max(0.0, min(1.0, float(profile.get("outer_ratio", 1.0))))
    except (TypeError, ValueError):
        outer_ratio = 1.0
    duration_mode = str(profile.get("duration_mode") or "turn").strip().lower()
    if duration_mode not in VALID_DURATION_MODES:
        duration_mode = "turn"
    action_note = str(profile.get("action_note") or "").strip()
    if not action_note:
        action_note = "TURN+FWD" if drive_mode_key == "forward" else "TURN+BWD"
    profile["drive_mode"] = drive_mode_key
    profile["inner_ratio"] = float(inner_ratio)
    profile["outer_ratio"] = float(outer_ratio)
    profile["duration_mode"] = duration_mode
    profile["action_note"] = action_note
    return resolved_name, profile


def _duration_ms_for_cmd(cmd: str | None, score: int | None) -> int | None:
    cmd_key = str(cmd or "").strip().lower()
    try:
        score_val = int(score)
    except (TypeError, ValueError):
        return None
    if cmd_key not in {"f", "b", "l", "r", "u", "d"}:
        return None
    try:
        _power, _pwm, _score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
            cmd_key,
            score_val,
        )
    except Exception:
        return None
    try:
        return max(1, int(round(float(duration_model_ms))))
    except (TypeError, ValueError):
        return None


def build_turn_drive_motion_plan(
    *,
    cmd: str | None,
    score: int | None,
    hold_duration_ms: int | None,
    pwm_override: int | None = None,
    profile_name: str | None = None,
    profile_override: dict | None = None,
    drive_mode: str | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> dict | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return None

    resolved_name, profile = _normalized_profile(
        config=config,
        profile_name=profile_name,
        profile_override=profile_override,
        drive_mode=drive_mode,
    )
    if not isinstance(profile, dict):
        return None

    try:
        score_val = int(score)
    except (TypeError, ValueError):
        return None

    try:
        base_pwm = telemetry_robot_module.clamp_pwm(int(round(float(pwm_override))))
    except (TypeError, ValueError):
        base_pwm = None
    if base_pwm is None or int(base_pwm) <= 0:
        try:
            _power, base_pwm, _score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                cmd_key,
                int(score_val),
            )
        except Exception:
            return None
    else:
        duration_model_ms = None
    try:
        base_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm))))
    except (TypeError, ValueError):
        return None
    if base_pwm is None or int(base_pwm) <= 0:
        return None

    try:
        hold_duration = max(1, int(round(float(hold_duration_ms))))
    except (TypeError, ValueError):
        hold_duration = None

    drive_mode_key = str(profile.get("drive_mode") or "").strip().lower()
    inner_ratio = float(profile.get("inner_ratio", 0.0))
    outer_ratio = float(profile.get("outer_ratio", 1.0))
    duration_mode = str(profile.get("duration_mode") or "turn").strip().lower()
    if duration_mode not in VALID_DURATION_MODES:
        duration_mode = "turn"
    outer_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(outer_ratio))))
    inner_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(inner_ratio))))
    if outer_pwm is None or int(outer_pwm) <= 0:
        return None
    if inner_pwm is None:
        inner_pwm = 0

    if drive_mode_key == "forward":
        left_action = "b"
        right_action = "f"
    elif drive_mode_key == "backward":
        left_action = "f"
        right_action = "b"
    else:
        return None

    drive_duration_model_ms = _duration_ms_for_cmd("f" if drive_mode_key == "forward" else "b", score_val)

    if cmd_key == "l":
        actions = [
            {"target": "l", "action": left_action, "pwm": int(inner_pwm)},
            {"target": "r", "action": right_action, "pwm": int(outer_pwm)},
        ]
    else:
        actions = [
            {"target": "l", "action": left_action, "pwm": int(outer_pwm)},
            {"target": "r", "action": right_action, "pwm": int(inner_pwm)},
        ]

    if hold_duration is None:
        duration_candidates = []
        if duration_mode in {"turn", "max_turn_drive"} and duration_model_ms is not None:
            try:
                duration_candidates.append(max(1, int(round(float(duration_model_ms)))))
            except (TypeError, ValueError):
                pass
        if duration_mode in {"drive", "max_turn_drive"} and drive_duration_model_ms is not None:
            try:
                duration_candidates.append(max(1, int(round(float(drive_duration_model_ms)))))
            except (TypeError, ValueError):
                pass
        if duration_candidates:
            hold_duration = max(duration_candidates) if duration_mode == "max_turn_drive" else duration_candidates[0]
        elif duration_model_ms is not None:
            try:
                hold_duration = max(1, int(round(float(duration_model_ms))))
            except (TypeError, ValueError):
                hold_duration = None
    if hold_duration is None:
        return None

    plan = {
        "enabled": True,
        "cmd": cmd_key,
        "score": int(score_val),
        "duration_ms": int(hold_duration),
        "base_pwm": int(base_pwm),
        "inner_pwm": int(inner_pwm),
        "outer_pwm": int(outer_pwm),
        "inner_ratio": float(inner_ratio),
        "outer_ratio": float(outer_ratio),
        "drive_mode": str(drive_mode_key),
        "duration_mode": str(duration_mode),
        "profile_name": resolved_name,
        "action_note": str(profile.get("action_note") or ""),
        "actions": actions,
    }
    if isinstance(metadata, dict) and metadata:
        plan["metadata"] = dict(metadata)
    return plan


def align_turn_drive_assist_request(step_rules, *, cmd: str | None, local_gate_status=None) -> dict | None:
    cfg = step_rules if isinstance(step_rules, dict) else {}
    align_policy = cfg.get("align_policy") if isinstance(cfg.get("align_policy"), dict) else {}
    assist_cfg = (
        align_policy.get("x_axis_turn_drive_assist")
        if isinstance(align_policy.get("x_axis_turn_drive_assist"), dict)
        else {}
    )
    if not bool(assist_cfg.get("enabled")):
        return None

    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return None

    local_gate = local_gate_status if isinstance(local_gate_status, dict) else {}
    if not bool(local_gate.get("visible")):
        return None

    dist_required = bool(local_gate.get("dist_required"))
    if not dist_required:
        return None

    try:
        dist_err = float(local_gate.get("dist_err"))
    except (TypeError, ValueError):
        return None
    dist_within_tol = bool(local_gate.get("dist_within_tol"))
    require_outside = bool(assist_cfg.get("require_dist_outside_gate", True))
    if require_outside and dist_within_tol:
        return None

    try:
        min_dist_outside_mm = max(0.0, float(assist_cfg.get("min_dist_outside_mm", 0.0) or 0.0))
    except (TypeError, ValueError):
        min_dist_outside_mm = 0.0
    if float(dist_err) < float(min_dist_outside_mm):
        return None

    dist_value = local_gate.get("dist")
    dist_target = local_gate.get("dist_target")
    try:
        dist_delta = float(dist_value) - float(dist_target)
    except (TypeError, ValueError):
        return None

    if dist_delta > 0.0:
        drive_mode_key = "forward"
        profile_name = str(assist_cfg.get("forward_profile") or "").strip() or None
    elif dist_delta < 0.0:
        drive_mode_key = "backward"
        profile_name = str(assist_cfg.get("backward_profile") or "").strip() or None
    else:
        return None
    if not profile_name:
        return None

    return {
        "drive_mode": str(drive_mode_key),
        "profile_name": profile_name,
    }
