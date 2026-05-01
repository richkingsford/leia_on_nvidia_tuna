#!/usr/bin/env python3
"""Run Leia's slow basic movement smoke test.

Dry-run is the default. Add --execute only when the robot is clear to move.
"""

from __future__ import annotations

import argparse
import os

from helper_basic_movement_smoke import run_basic_movement_smoke_test
from helper_robot_control import Robot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slowly wiggle Leia's treads and mast.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send motor commands. Without this flag, only print the planned sequence.",
    )
    parser.add_argument(
        "--port",
        help="Optional serial device path. Sets LEIA_SERIAL_PORT before connecting.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run the six-pulse sequence.",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=0.35,
        help="Pause between pulses after each timed move and stop.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.05,
        help="Extra wait after each timed move before sending stop.",
    )
    parser.add_argument(
        "--plain-turns",
        action="store_true",
        help="Use direct L/R turn commands instead of the world-model q/e arc profile.",
    )
    parser.add_argument(
        "--show-wire",
        action="store_true",
        help="Print explicit wire/debug payloads after each executed pulse.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.port:
        os.environ["LEIA_SERIAL_PORT"] = str(args.port)

    robot = None
    try:
        if args.execute:
            robot = Robot()
        result = run_basic_movement_smoke_test(
            robot=robot,
            execute=bool(args.execute),
            repeat=max(1, int(args.repeat or 1)),
            pause_s=max(0.0, float(args.pause_s or 0.0)),
            settle_s=max(0.0, float(args.settle_s or 0.0)),
            use_turn_arc_profiles=not bool(args.plain_turns),
            show_wire=bool(args.show_wire),
        )
        if not bool(result.get("execute")):
            print("[DRY-RUN] No motor commands were sent. Re-run with --execute when Leia is clear.")
        return 0
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
