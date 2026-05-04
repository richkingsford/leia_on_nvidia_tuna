#!/usr/bin/env python3
"""Run one auto-step from the command line."""

from __future__ import annotations

import argparse
import json
import time

from a_MAIN import (
    VISION_MODE_CYAN,
    _DEFAULT_CYAN_PROFILE,
    _DEFAULT_CYAN_VISIBILITY,
    _apply_cyan_profile,
    _apply_cyan_visibility,
    _stream_state_cyan_profile,
    _stream_state_cyan_visibility,
    AppState,
    build_vision,
    close_log,
    refresh_brick_telemetry,
    resolve_step_token,
    run_auto_step,
)
from helper_robot_control import Robot
from helper_vision_config import normalize_vision_mode


def resolve_step_argument(token: str):
    return resolve_step_token(str(token or "").strip())


def _prime_vision(
    app_state,
    *,
    reads: int = 16,
    sleep_s: float = 0.08,
    require_visible_streak: int = 2,
) -> dict:
    reads_used = max(1, int(reads))
    required_streak = max(1, int(require_visible_streak))
    pose = {"visible": None, "dist_mm": None, "x_axis_mm": None}
    visible_streak = 0
    visible_samples = 0
    for _ in range(int(reads_used)):
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible_now = bool(brick.get("visible"))
        if visible_now:
            visible_samples += 1
            visible_streak += 1
        else:
            visible_streak = 0
        pose = {
            "visible": bool(brick.get("visible")),
            "dist_mm": brick.get("dist"),
            "x_axis_mm": brick.get("x_axis", brick.get("x")),
        }
        if visible_streak >= required_streak:
            break
        time.sleep(max(0.0, float(sleep_s)))
    pose["visible_samples"] = int(visible_samples)
    pose["reads_used"] = int(reads_used)
    return pose


def run_single_auto_step(*, step_token: str, vision_mode: str | None):
    step_obj = resolve_step_argument(step_token)
    if step_obj is None:
        return {
            "ok": False,
            "error": f"unknown_step:{step_token}",
        }

    mode_raw = None if vision_mode is None else str(vision_mode).strip().lower()
    if mode_raw in (None, "", "auto"):
        app_state = AppState()
        mode_norm = str(getattr(app_state, "active_vision_mode", "cyan"))
    else:
        mode_norm = normalize_vision_mode(mode_raw)
        app_state = AppState(vision_mode=mode_norm)
    app_state.robot = Robot()
    app_state.vision = build_vision(mode_norm)
    if normalize_vision_mode(mode_norm) == VISION_MODE_CYAN and app_state.vision is not None:
        active_cyan_profile = _stream_state_cyan_profile(app_state, _DEFAULT_CYAN_PROFILE)
        active_cyan_visibility = _stream_state_cyan_visibility(app_state, _DEFAULT_CYAN_VISIBILITY)
        _apply_cyan_profile(
            app_state,
            app_state.vision,
            active_cyan_profile,
        )
        _apply_cyan_visibility(
            app_state,
            app_state.vision,
            active_cyan_visibility,
        )
    try:
        pose_before = _prime_vision(app_state)
        print(
            "[AUTO STEP CLI] Starting "
            f"{step_obj.value} with {mode_norm} vision "
            f"(visible={pose_before.get('visible')}, "
            f"dist={pose_before.get('dist_mm')}, x_axis={pose_before.get('x_axis_mm')})."
        )
        ok = bool(run_auto_step(app_state, step_obj))
        step_key = str(getattr(step_obj, "value", step_obj))
        action_history = list(
            ((getattr(app_state, "session_step_actions_by_step", {}) or {}).get(step_key) or [])
        )
        actions_required = int(action_history[-1]) if action_history else 0
        return {
            "ok": bool(ok),
            "step": step_obj.value,
            "vision_mode": str(mode_norm),
            "pose_before": pose_before,
            "actions_required": int(actions_required),
            "run_log_path": None if getattr(app_state, "log_path", None) is None else str(app_state.log_path),
        }
    finally:
        try:
            close_log(app_state, marker=None)
        except Exception:
            pass
        try:
            import helper_xyz_coords as _xyz
            log_path = getattr(app_state, "log_path", None)
            if log_path:
                _xyz.write_run_view_from_log(log_path)
            else:
                _xyz.write_run_view_from_log()
        except Exception:
            pass
        try:
            app_state.vision.close()
        except Exception:
            pass
        try:
            app_state.robot.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one auto-step from the CLI.")
    parser.add_argument("--step", type=str, default="7", help="Step number or name. Default: 7")
    parser.add_argument(
        "--vision",
        type=str,
        choices=("auto", "cyan", "yolo", "leia", "aruco"),
        default="auto",
        help="Vision mode for the auto-step run. Default: auto (match app/world-model active mode).",
    )
    args = parser.parse_args()

    result = run_single_auto_step(
        step_token=str(args.step),
        vision_mode=(None if str(args.vision).strip().lower() == "auto" else str(args.vision)),
    )
    print(json.dumps(result, indent=2))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
