#!/usr/bin/env python3
"""Run the empty-game Step 3 mast-lift ("up") and, on success, begin the holding game.

Mirrors the modified-E2E step3_lift phase:
    result = follow._run_step3_lift_sequence(vision, robot)
    if result.holding: follow._set_game_profile("holding")
"""

from __future__ import annotations

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


def _fmt(reading, key, signed=False):
    try:
        value = float((reading or {}).get(key))
    except Exception:
        return "N/A"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def main() -> int:
    vision = None
    robot = None
    try:
        follow._set_game_profile("empty")
        vision = BrickDetector(debug=True)
        set_tuning = getattr(vision, "set_runtime_tuning", None)
        if callable(set_tuning):
            set_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = Robot()

        result = follow._run_step3_lift_sequence(vision, robot)
        ok = bool(result.get("success")) if isinstance(result, dict) else False
        holding = bool(result.get("holding")) if isinstance(result, dict) else False
        reason = result.get("reason") if isinstance(result, dict) else "no_result"
        reading = result.get("reading") if isinstance(result, dict) else None
        print(
            f"[EMPTY_S3] success={ok} holding={holding} reason={reason} "
            f"dist={_fmt(reading, 'dist_mm')}mm x={_fmt(reading, 'x_mm', True)}mm "
            f"y={_fmt(reading, 'y_mm', True)}mm",
            flush=True,
        )
        if holding:
            follow._set_game_profile("holding")
            print(f"[EMPTY_S3] profile now: {follow._active_game_profile()} (holding game begun)", flush=True)
        print("[EMPTY_S3] DONE_OK" if ok else "[EMPTY_S3] DONE_FAIL", flush=True)
        return 0 if ok else 2
    finally:
        if robot is not None:
            try:
                follow._stop_robot(robot)
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        print("[EMPTY_S3] closed", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
