#!/usr/bin/env python3
"""One-shot trigger for the POST-LIFT reset (a_follow_the_brick._run_reset_sequence).

Mirrors the game's worker setup (vision + robot), forces the holding profile, then
runs the exact reset that fires after the step-4 lift. This reset backs away, does
the x-offset/sharp-finish curve, and raises the mast.
"""
from __future__ import annotations

import a_follow_the_brick as F
from helper_brick_detector_yolo import BrickDetector


def main() -> int:
    F._set_game_profile("holding")
    print(f"[SETUP] profile={F._active_game_profile()}", flush=True)

    vision = BrickDetector(debug=True)
    vision.set_runtime_tuning(**dict(F.CROWN_PROFILE_TUNING))
    F._warmup(vision)

    robot = F.Robot()
    try:
        pregame = F._wait_for_confident_brick(vision)
        try:
            print(
                f"[PREGAME] dist={float(pregame.get('dist_mm')):.1f}mm "
                f"x={float(pregame.get('x_mm')):+.1f}mm conf={bool(pregame.get('confident'))}",
                flush=True,
            )
        except (TypeError, ValueError):
            print(f"[PREGAME] confident={bool(pregame.get('confident'))}", flush=True)

        print("[POST-LIFT RESET] Executing _run_reset_sequence ...", flush=True)
        result = F._run_reset_sequence(vision, robot)

        reading = result.get("reading") if isinstance(result, dict) else None
        rd = "N/A"
        if isinstance(reading, dict):
            try:
                rd = (
                    f"dist={float(reading.get('dist_mm')):.1f}mm "
                    f"x={float(reading.get('x_mm')):+.1f}mm "
                    f"y={float(reading.get('y_mm')):+.1f}mm "
                    f"conf={float(reading.get('confidence_pct') or 0):.0f}%"
                )
            except (TypeError, ValueError):
                rd = "numeric read unavailable"
        print(
            f"[POST-LIFT RESET] done: success={result.get('success')} "
            f"turn={result.get('turn_cmd')} reason={result.get('reason')} "
            f"mast_up_sent={result.get('mast_up_sent')} | {rd}",
            flush=True,
        )
    finally:
        F._stop_robot(robot)
        try:
            robot.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
