#!/usr/bin/env python3
"""Find the mast's low-speed breakaway point with short pulses."""

from __future__ import annotations

import time

from helper_robot_control import Robot


PHYSICAL_CMD = "d"
PULSE_MS = 500
PAUSE_S = 1.0
PERCENTS = (50, 60, 70)


def _pwm_for_percent(percent: int) -> int:
    return max(1, min(255, int(round(255.0 * float(percent) / 100.0))))


def main() -> int:
    robot = Robot()
    try:
        for percent in PERCENTS:
            pwm = _pwm_for_percent(percent)
            print(f"[SLOW-SPEED] physical up {percent}% for {PULSE_MS}ms", flush=True)
            result = robot.send_command_pwm(PHYSICAL_CMD, pwm, duration_ms=PULSE_MS)
            print(result, flush=True)
            time.sleep(float(PULSE_MS) / 1000.0)
            robot.stop()
            time.sleep(PAUSE_S)
    finally:
        robot.stop()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
