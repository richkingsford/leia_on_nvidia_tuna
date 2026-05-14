#!/usr/bin/env python3
"""Run Leia's straight-line speed ramp test.

Dry-run is the default. Add --execute only when the robot is clear to move.
"""

from __future__ import annotations

import argparse
import os

from helper_robot_control import Robot
from helper_speed_test import (
    DEFAULT_END_SCORE,
    DEFAULT_INTERVAL_MS,
    DEFAULT_PHASE_DURATION_S,
    DEFAULT_START_PWM_SCALE,
    build_speed_test_sequence,
    run_speed_test,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ramp forward speed, then run the reverse-direction ramp.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send motor commands. Without this flag, only print the planned table.",
    )
    parser.add_argument(
        "--port",
        help="Optional serial device path. Sets LEIA_SERIAL_PORT before connecting.",
    )
    parser.add_argument(
        "--phase-s",
        type=float,
        default=DEFAULT_PHASE_DURATION_S,
        help="Approximate seconds for each ramp phase.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Cadence for sending the next speed command.",
    )
    parser.add_argument(
        "--reverse-mode",
        choices=("backward", "down"),
        default="backward",
        help="Use backward for the second ramp, or down for a forward ramp-down.",
    )
    parser.add_argument(
        "--end-score",
        type=int,
        default=DEFAULT_END_SCORE,
        help="Upper speed-score ceiling for each ramp.",
    )
    parser.add_argument(
        "--start-pwm-scale",
        type=float,
        default=DEFAULT_START_PWM_SCALE,
        help="Multiplier for the starting PWM floor before ramping.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.port:
        os.environ["LEIA_SERIAL_PORT"] = str(args.port)

    sequence = build_speed_test_sequence(
        phase_duration_s=float(args.phase_s),
        interval_ms=int(args.interval_ms),
        reverse_mode=str(args.reverse_mode),
        end_score=int(args.end_score),
        start_pwm_scale=float(args.start_pwm_scale),
    )

    robot = None
    try:
        if args.execute:
            robot = Robot()
        result = run_speed_test(
            robot=robot,
            execute=bool(args.execute),
            sequence=sequence,
        )
        if not bool(result.get("execute")):
            print("[DRY-RUN] No motor commands were sent. Re-run with --execute when Leia is clear.")
        return 0
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
