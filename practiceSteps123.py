#!/usr/bin/env python3
"""Drill the full Step 1 -> 2 -> 3 alignment sequence then reset, N trials.

Uses the real game loop (a_follow_the_brick._follow_loop) with complete_after_step3,
so each cycle runs Step 1 (align) -> Step 2 (align y) -> Step 3 (seat) -> reset, and
the step-4 lift is skipped. Stops after --trials completed cycles.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot

OUT_PATH = Path("runs") / f"steps123_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--duration-s", type=float, default=240.0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    follow._set_game_profile("empty")
    vision = BrickDetector(debug=True)
    robot = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        follow._set_game_profile("empty")
        follow._print_active_success_gates()
        pregame = follow._wait_for_confident_brick(vision)
        if not bool(pregame.get("confident")):
            print("[STEPS123] Pregame visibility failed; brick not confidently visible. Stopping.", flush=True)
            return 2
        print(f"[STEPS123] Running {args.trials} cycles of Step 1->2->3 + reset.", flush=True)
        stats = follow._follow_loop(
            vision,
            robot,
            duration_s=float(args.duration_s),
            reset_after_win=True,
            complete_after_step3=True,
            max_cycles=int(args.trials),
        )
        completed = int(stats.get("step3_win_count", 0))
        print(f"\n[STEPS123] completed step1-3 cycles: {completed}/{args.trials}", flush=True)
        print("[RESULTS]", flush=True)
        try:
            print(follow._format_game_results_table(stats), flush=True)
        except Exception:
            pass
        with out.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in stats.items() if _jsonable(v)}, sort_keys=True) + "\n")
        print(f"[STEPS123] data={out}", flush=True)
        return 0
    finally:
        follow._stop_robot(robot)
        try: robot.close()
        except Exception: pass
        try: vision.close()
        except Exception: pass


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
