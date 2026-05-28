#!/usr/bin/env python3
"""Run small deliberate robot movements while an external livestream watches."""

from __future__ import annotations

import time

from helper_robot_control import Robot
from telemetry_robot import SPEED_SCORE_MIN, speed_power_pwm_for_cmd


PULSE_MS = 270
TURN_SHARPNESS_MULTIPLIER = 1.30
PAUSE_BETWEEN_MOVES_S = 0.25
PAUSE_BETWEEN_LOOPS_S = 1.0
LOOPS = 2
SEQUENCE = (
    ("left", "l"),
    ("right", "r"),
    ("forward", "f"),
    ("back", "b"),
    ("up", "u"),
    ("down", "d"),
)


def motion_for(cmd: str) -> tuple[int, int, float]:
    duration_ms = PULSE_MS
    if cmd in {"l", "r"}:
        duration_ms = int(round(PULSE_MS * TURN_SHARPNESS_MULTIPLIER))
    power, pwm, _score, min_duration_ms = speed_power_pwm_for_cmd(cmd, SPEED_SCORE_MIN)
    duration_ms = max(duration_ms, int(round(min_duration_ms)))
    return int(round(pwm)), int(duration_ms), float(power)


def main() -> int:
    robot = Robot()
    try:
        for loop_idx in range(1, LOOPS + 1):
            print(f"[MOVE-TEST] loop {loop_idx}/{LOOPS}", flush=True)
            for label, cmd in SEQUENCE:
                pwm, duration_ms, power = motion_for(cmd)
                print(
                    f"[MOVE-TEST] {label} {duration_ms}ms at 1% "
                    f"(pwm={pwm}, power={power:.3f})",
                    flush=True,
                )
                robot.send_command_pwm(cmd, pwm, duration_ms=duration_ms)
                time.sleep((duration_ms / 1000.0) + PAUSE_BETWEEN_MOVES_S)
                robot.stop()
            if loop_idx < LOOPS:
                print(f"[MOVE-TEST] pause {PAUSE_BETWEEN_LOOPS_S:.1f}s", flush=True)
                time.sleep(PAUSE_BETWEEN_LOOPS_S)
        print("[MOVE-TEST] complete", flush=True)
        return 0
    finally:
        try:
            robot.stop()
        finally:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
