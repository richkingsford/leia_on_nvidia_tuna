#!/usr/bin/env python3
"""Config-driven forward-arc assist for manual turn hotkeys."""

from __future__ import annotations

import json
from pathlib import Path

import telemetry_robot as telemetry_robot_module


ASSIST_CONFIG_KEY = "manual_turn_arc_assist"
DEFAULT_ASSIST_CONFIG = {
    "enabled": True,
    "hotkey_profiles": {
        "q": {"inner_ratio": 0.0, "outer_ratio": 1.0},
        "e": {"inner_ratio": 0.0, "outer_ratio": 1.0},
        "a": {"inner_ratio": 0.55, "outer_ratio": 1.0},
        "d": {"inner_ratio": 0.55, "outer_ratio": 1.0},
        "z": {"inner_ratio": 0.25, "outer_ratio": 1.0},
        "c": {"inner_ratio": 0.25, "outer_ratio": 1.0},
    },
}


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


def load_manual_turn_arc_assist_config(path: Path | None = None) -> dict:
    payload = _load_raw_model(path)
    raw = payload.get(ASSIST_CONFIG_KEY)
    config = {
        "enabled": bool(DEFAULT_ASSIST_CONFIG["enabled"]),
        "hotkey_profiles": {
            str(key): dict(value)
            for key, value in (DEFAULT_ASSIST_CONFIG.get("hotkey_profiles") or {}).items()
            if isinstance(value, dict)
        },
    }
    if isinstance(raw, dict):
        config["enabled"] = bool(raw.get("enabled", config["enabled"]))
        raw_profiles = raw.get("hotkey_profiles")
        if isinstance(raw_profiles, dict):
            normalized_profiles = {}
            for hotkey, value in raw_profiles.items():
                hotkey_key = str(hotkey or "").strip().lower()
                if not hotkey_key or not isinstance(value, dict):
                    continue
                base_profile = dict((config.get("hotkey_profiles") or {}).get(hotkey_key) or {})
                profile = dict(base_profile)
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
                normalized_profiles[hotkey_key] = profile
            if normalized_profiles:
                config["hotkey_profiles"] = normalized_profiles
    return config


def build_manual_turn_arc_plan(
    *,
    hotkey: str | None,
    cmd: str | None,
    score: int | None,
    hold_duration_ms: int | None,
    pwm_override: int | None = None,
    config: dict | None = None,
) -> dict | None:
    assist = dict(config) if isinstance(config, dict) else load_manual_turn_arc_assist_config()
    if not bool(assist.get("enabled")):
        return None

    hotkey_key = str(hotkey or "").strip().lower()
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in ("l", "r"):
        return None
    profiles = assist.get("hotkey_profiles") if isinstance(assist.get("hotkey_profiles"), dict) else {}
    profile = profiles.get(hotkey_key)
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
    if hold_duration is None:
        try:
            hold_duration = max(1, int(round(float(duration_model_ms))))
        except (TypeError, ValueError):
            return None

    try:
        inner_ratio = max(0.0, min(1.0, float(profile.get("inner_ratio", 0.0))))
    except (TypeError, ValueError):
        inner_ratio = 0.0
    try:
        outer_ratio = max(0.0, min(1.0, float(profile.get("outer_ratio", 1.0))))
    except (TypeError, ValueError):
        outer_ratio = 1.0

    outer_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(outer_ratio))))
    inner_pwm = telemetry_robot_module.clamp_pwm(int(round(float(base_pwm) * float(inner_ratio))))
    if outer_pwm is None or int(outer_pwm) <= 0:
        return None
    if inner_pwm is None:
        inner_pwm = 0

    if cmd_key == "l":
        actions = [
            {"target": "l", "action": "b", "pwm": int(inner_pwm)},
            {"target": "r", "action": "f", "pwm": int(outer_pwm)},
        ]
    else:
        actions = [
            {"target": "l", "action": "b", "pwm": int(outer_pwm)},
            {"target": "r", "action": "f", "pwm": int(inner_pwm)},
        ]

    return {
        "enabled": True,
        "hotkey": hotkey_key,
        "cmd": cmd_key,
        "score": int(score_val),
        "duration_ms": int(hold_duration),
        "base_pwm": int(base_pwm),
        "inner_pwm": int(inner_pwm),
        "outer_pwm": int(outer_pwm),
        "inner_ratio": float(inner_ratio),
        "outer_ratio": float(outer_ratio),
        "actions": actions,
    }


def format_manual_turn_arc_assist_line(plan: dict | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    try:
        hotkey = str(plan.get("hotkey") or "").strip().upper()
        cmd = str(plan.get("cmd") or "").strip().upper()
        outer_pwm = int(plan.get("outer_pwm") or 0)
        inner_pwm = int(plan.get("inner_pwm") or 0)
        duration_ms = int(plan.get("duration_ms") or 0)
    except (TypeError, ValueError):
        return None
    return (
        f"[HOTKEY ARC] {hotkey} -> {cmd}: outer pwm {int(outer_pwm)}, "
        f"inner pwm {int(inner_pwm)}, t={int(duration_ms)}ms."
    )


def execute_manual_turn_arc_plan(
    *,
    robot,
    hotkey: str | None,
    cmd: str | None,
    score: int | None,
    hold_duration_ms: int | None,
    pwm_override: int | None = None,
    config: dict | None = None,
) -> dict | None:
    plan = build_manual_turn_arc_plan(
        hotkey=hotkey,
        cmd=cmd,
        score=score,
        hold_duration_ms=hold_duration_ms,
        pwm_override=pwm_override,
        config=config,
    )
    if not isinstance(plan, dict):
        return None
    if robot is None or not hasattr(robot, "send_custom_actions_pwm"):
        return None
    send_result = robot.send_custom_actions_pwm(
        str(plan.get("cmd") or ""),
        plan.get("actions") or [],
        duration_ms=int(plan.get("duration_ms") or 0),
    )
    if not isinstance(send_result, dict):
        return None
    send_result["manual_turn_arc_assist"] = dict(plan)
    return send_result
