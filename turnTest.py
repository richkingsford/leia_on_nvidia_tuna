#!/usr/bin/env python3
"""Run Leia's slow sharpening turn test.

Dry-run is the default. Add --execute only when the robot is clear to move.
"""

from __future__ import annotations

import argparse
import os

from helper_robot_control import Robot
from helper_turn_test import (
    DEFAULT_HOLD_AFTER_S,
    DEFAULT_PHASE_DURATION_S,
    DEFAULT_REQUESTED_STEP_MS,
    DEFAULT_SPEED_SCORE,
    DEFAULT_STOP_HOLD_STEPS,
    build_turn_test_sequence,
    run_turn_test,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slowly sharpen a forward-right turn, then mirror back-left.")
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
        "--phase-s",
        type=float,
        default=DEFAULT_PHASE_DURATION_S,
        help="Seconds requested for each turn phase.",
    )
    parser.add_argument(
        "--step-ms",
        type=int,
        default=DEFAULT_REQUESTED_STEP_MS,
        help="Requested sharpness-change interval in milliseconds.",
    )
    parser.add_argument(
        "--hold-after-s",
        type=float,
        default=DEFAULT_HOLD_AFTER_S,
        help="When to switch the inner tread to the still hold within each phase.",
    )
    parser.add_argument(
        "--stop-hold-steps",
        type=int,
        default=DEFAULT_STOP_HOLD_STEPS,
        help="Number of intervals to hold the inner tread still before reversing it.",
    )
    parser.add_argument(
        "--score",
        type=int,
        default=DEFAULT_SPEED_SCORE,
        help="World-model speed score to use. Default is the minimum score.",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=0.0,
        help="Pause before the first pulse after the initial stop.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="Extra wait after each timed pulse before sending the next pulse.",
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

    sequence = build_turn_test_sequence(
        phase_duration_s=float(args.phase_s),
        requested_step_ms=int(args.step_ms),
        hold_after_s=float(args.hold_after_s),
        stop_hold_steps=int(args.stop_hold_steps),
        speed_score=int(args.score),
    )

    robot = None
    try:
        if args.execute:
            robot = Robot()
        result = run_turn_test(
            robot=robot,
            execute=bool(args.execute),
            sequence=sequence,
            pause_s=float(args.pause_s),
            settle_s=float(args.settle_s),
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
