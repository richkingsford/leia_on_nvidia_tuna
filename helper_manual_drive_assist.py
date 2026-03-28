#!/usr/bin/env python3
"""Config-driven manual low-speed motion assist for hotkeys."""

from __future__ import annotations

import json
import time
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from telemetry_process import send_robot_command, send_robot_command_pwm


ASSIST_CONFIG_KEY = "manual_low_speed_drive_assist"
DEFAULT_ASSIST_CONFIG = {
    "enabled": True,
    "hotkeys": ["r", "f", "q", "e"],
    "cmds": ["f", "b", "l", "r"],
    "max_score": 1,
    "kick_score": 6,
    "kick_duration_ms": 80,
    "pause_after_kick_ms": 40,
    "hotkey_overrides": {},
}
HOTKEYS_ALL = tuple(str(item).strip().lower() for item in (DEFAULT_ASSIST_CONFIG.get("hotkeys") or []) if str(item).strip())


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


def load_manual_drive_assist_config(path: Path | None = None) -> dict:
    payload = _load_raw_model(path)
    raw = payload.get(ASSIST_CONFIG_KEY)
    config = dict(DEFAULT_ASSIST_CONFIG)
    if isinstance(raw, dict):
        config.update(raw)

    hotkeys = raw.get("hotkeys") if isinstance(raw, dict) else None
    cmds = raw.get("cmds") if isinstance(raw, dict) else None
    normalized_hotkeys = [
        str(item).strip().lower()
        for item in (hotkeys if isinstance(hotkeys, (list, tuple)) else config.get("hotkeys") or [])
        if str(item).strip()
    ]
    normalized_cmds = [
        str(item).strip().lower()
        for item in (cmds if isinstance(cmds, (list, tuple)) else config.get("cmds") or [])
        if str(item).strip()
    ]
    config["enabled"] = bool(config.get("enabled", DEFAULT_ASSIST_CONFIG["enabled"]))
    config["hotkeys"] = tuple(normalized_hotkeys)
    config["cmds"] = tuple(normalized_cmds)
    config["max_score"] = max(1, int(config.get("max_score", DEFAULT_ASSIST_CONFIG["max_score"])))
    config["kick_score"] = max(1, int(config.get("kick_score", DEFAULT_ASSIST_CONFIG["kick_score"])))
    config["kick_duration_ms"] = max(
        1,
        int(config.get("kick_duration_ms", DEFAULT_ASSIST_CONFIG["kick_duration_ms"])),
    )
    config["pause_after_kick_ms"] = max(
        0,
        int(config.get("pause_after_kick_ms", DEFAULT_ASSIST_CONFIG["pause_after_kick_ms"])),
    )
    normalized_overrides = {}
    raw_overrides = raw.get("hotkey_overrides") if isinstance(raw, dict) else None
    if isinstance(raw_overrides, dict):
        for key, value in raw_overrides.items():
            hotkey_key = str(key or "").strip().lower()
            if not hotkey_key or hotkey_key not in HOTKEYS_ALL:
                continue
            if not isinstance(value, dict):
                continue
            override = {}
            if "kick_score" in value:
                try:
                    override["kick_score"] = max(1, int(value.get("kick_score")))
                except (TypeError, ValueError):
                    pass
            if "kick_duration_ms" in value:
                try:
                    override["kick_duration_ms"] = max(1, int(value.get("kick_duration_ms")))
                except (TypeError, ValueError):
                    pass
            if "pause_after_kick_ms" in value:
                try:
                    override["pause_after_kick_ms"] = max(0, int(value.get("pause_after_kick_ms")))
                except (TypeError, ValueError):
                    pass
            if override:
                normalized_overrides[hotkey_key] = override
    config["hotkey_overrides"] = normalized_overrides
    return config


def build_manual_drive_assist_plan(
    *,
    hotkey: str | None,
    cmd: str | None,
    score: int | None,
    hold_duration_ms: int | None,
    pwm_override: int | None = None,
    config: dict | None = None,
) -> dict | None:
    assist = dict(config) if isinstance(config, dict) else load_manual_drive_assist_config()
    if not bool(assist.get("enabled")):
        return None

    hotkey_key = str(hotkey or "").strip().lower()
    cmd_key = str(cmd or "").strip().lower()
    try:
        score_val = int(score)
    except (TypeError, ValueError):
        return None

    if hotkey_key not in tuple(assist.get("hotkeys") or ()):
        return None
    if cmd_key not in tuple(assist.get("cmds") or ()):
        return None
    if int(score_val) > int(assist.get("max_score", 0) or 0):
        return None

    hotkey_override = {}
    raw_overrides = assist.get("hotkey_overrides")
    if isinstance(raw_overrides, dict):
        candidate = raw_overrides.get(hotkey_key)
        if isinstance(candidate, dict):
            hotkey_override = candidate
    kick_score = max(
        int(score_val),
        int(hotkey_override.get("kick_score", assist.get("kick_score") or score_val)),
    )
    if int(kick_score) <= int(score_val):
        return None

    try:
        hold_duration = max(1, int(hold_duration_ms))
    except (TypeError, ValueError):
        hold_duration = None
    if hold_duration is None:
        try:
            _power, _pwm, _score_used, hold_duration = telemetry_robot_module.speed_power_pwm_for_cmd(
                cmd_key,
                int(score_val),
            )
            hold_duration = max(1, int(hold_duration))
        except Exception:
            return None

    return {
        "enabled": True,
        "hotkey": hotkey_key,
        "cmd": cmd_key,
        "hold_score": int(score_val),
        "hold_duration_ms": int(hold_duration),
        "hold_pwm_override": None if pwm_override is None else max(1, int(pwm_override)),
        "kick_score": int(kick_score),
        "kick_duration_ms": max(
            1,
            int(hotkey_override.get("kick_duration_ms", assist.get("kick_duration_ms") or 1)),
        ),
        "pause_after_kick_ms": max(
            0,
            int(hotkey_override.get("pause_after_kick_ms", assist.get("pause_after_kick_ms") or 0)),
        ),
    }


def format_manual_drive_assist_line(plan: dict | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    try:
        cmd = str(plan.get("cmd") or "").strip().upper()
        hotkey = str(plan.get("hotkey") or "").strip().upper()
        kick_score = int(plan.get("kick_score"))
        kick_ms = int(plan.get("kick_duration_ms"))
        hold_score = int(plan.get("hold_score"))
        hold_ms = int(plan.get("hold_duration_ms"))
        pause_ms = int(plan.get("pause_after_kick_ms") or 0)
    except (TypeError, ValueError):
        return None
    pause_text = f" + pause {int(pause_ms)}ms" if int(pause_ms) > 0 else ""
    return (
        f"[HOTKEY ASSIST] {hotkey} -> {cmd}: kick {int(kick_score)}% for {int(kick_ms)}ms"
        f"{pause_text}, then hold {int(hold_score)}% for {int(hold_ms)}ms."
    )


def execute_manual_drive_assist_plan(
    *,
    robot,
    world,
    step_state,
    hotkey: str | None,
    cmd: str | None,
    score: int | None,
    hold_duration_ms: int | None,
    pwm_override: int | None = None,
    config: dict | None = None,
    send_robot_command_fn=send_robot_command,
    send_robot_command_pwm_fn=send_robot_command_pwm,
    sleep_fn=time.sleep,
) -> dict | None:
    plan = build_manual_drive_assist_plan(
        hotkey=hotkey,
        cmd=cmd,
        score=score,
        hold_duration_ms=hold_duration_ms,
        pwm_override=pwm_override,
        config=config,
    )
    if not isinstance(plan, dict):
        return None

    cmd_key = str(plan.get("cmd") or "").strip().lower()
    hold_score = int(plan.get("hold_score") or 0)
    kick_result = send_robot_command_fn(
        robot,
        world,
        step_state,
        cmd_key,
        0.0,
        speed_score=int(plan["kick_score"]),
        duration_override_ms=int(plan["kick_duration_ms"]),
        auto_mode=False,
        ease_in_out_enabled=False,
    )
    try:
        kick_used_ms = int(round(float((kick_result or {}).get("duration_ms") or plan["kick_duration_ms"])))
    except (TypeError, ValueError):
        kick_used_ms = int(plan["kick_duration_ms"])
    pause_after_kick_ms = max(0, int(plan.get("pause_after_kick_ms") or 0))
    wait_s = max(0.0, (float(kick_used_ms) + float(pause_after_kick_ms)) / 1000.0)
    if wait_s > 0.0:
        sleep_fn(wait_s)

    hold_duration_ms = int(plan["hold_duration_ms"])
    hold_result = None
    hold_pwm_override = plan.get("hold_pwm_override")
    if hold_pwm_override is not None:
        try:
            pwm_override_val = telemetry_robot_module.clamp_pwm(int(hold_pwm_override))
        except (TypeError, ValueError):
            pwm_override_val = None
        if pwm_override_val is not None and pwm_override_val > 0:
            try:
                power_for_pwm = telemetry_robot_module.pwm_to_power(pwm_override_val)
            except Exception:
                power_for_pwm = None
            if power_for_pwm is None:
                power_for_pwm, _, _, _ = telemetry_robot_module.speed_power_pwm_for_cmd(cmd_key, int(hold_score))
            hold_result = send_robot_command_pwm_fn(
                robot,
                world,
                step_state,
                cmd_key,
                float(power_for_pwm or 0.0),
                int(pwm_override_val),
                int(hold_duration_ms),
                speed_score=int(hold_score),
                auto_mode=False,
                ease_in_out_enabled=False,
            )
    if hold_result is None:
        hold_result = send_robot_command_fn(
            robot,
            world,
            step_state,
            cmd_key,
            0.0,
            speed_score=int(hold_score),
            duration_override_ms=int(hold_duration_ms),
            auto_mode=False,
            ease_in_out_enabled=False,
        )

    try:
        kick_power = float((kick_result or {}).get("power") or 0.0)
    except (TypeError, ValueError):
        kick_power = 0.0
    try:
        hold_power = float((hold_result or {}).get("power") or 0.0)
    except (TypeError, ValueError):
        hold_power = 0.0
    try:
        kick_pwm = int(round(float((kick_result or {}).get("pwm") or 0)))
    except (TypeError, ValueError):
        kick_pwm = 0
    try:
        hold_pwm = int(round(float((hold_result or {}).get("pwm") or 0)))
    except (TypeError, ValueError):
        hold_pwm = 0
    try:
        hold_used_ms = int(round(float((hold_result or {}).get("duration_ms") or hold_duration_ms)))
    except (TypeError, ValueError):
        hold_used_ms = int(hold_duration_ms)

    return {
        "cmd_sent": (hold_result or {}).get("cmd_sent") or cmd_key,
        "power": float(max(kick_power, hold_power)),
        "pwm": int(max(kick_pwm, hold_pwm)),
        "duration_ms": int(max(1, int(kick_used_ms) + int(hold_used_ms))),
        "score_model": int(hold_score),
        "score_effective": (hold_result or {}).get("score_effective"),
        "segments": [
            {
                "phase": "kick",
                "cmd_sent": (kick_result or {}).get("cmd_sent"),
                "pwm": int(kick_pwm),
                "duration_ms": int(kick_used_ms),
                "score_model": int(plan["kick_score"]),
            },
            {
                "phase": "hold",
                "cmd_sent": (hold_result or {}).get("cmd_sent"),
                "pwm": int(hold_pwm),
                "duration_ms": int(hold_used_ms),
                "score_model": int(hold_score),
            },
        ],
        "manual_drive_assist": dict(plan),
    }
