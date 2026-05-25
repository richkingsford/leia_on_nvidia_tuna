#!/usr/bin/env python3
"""Run bounded debug trials for the empty follow-the-brick game."""

from __future__ import annotations

import subprocess
import sys
import time

from helper_robot_control import Robot


TRIALS = 10
RECOVERY_UP_MS = 700
FOLLOW_CMD = [
    sys.executable,
    "a_follow_the_brick.py",
    "--game-profile",
    "empty",
    "--duration-s",
    "90",
    "--debug-mode",
    "--skip-vision-preflight",
]


def _mast_up() -> None:
    robot = Robot()
    try:
        result = robot.send_command_pwm("d", robot.MAX_PWM, duration_ms=RECOVERY_UP_MS)
        print(f"[10TRIAL] recovery up: {result}", flush=True)
    finally:
        robot.close()


def main() -> int:
    for trial in range(1, TRIALS + 1):
        if trial > 1:
            _mast_up()
            time.sleep(0.4)
        print(f"[10TRIAL] trial {trial}/{TRIALS}", flush=True)
        proc = subprocess.run(FOLLOW_CMD, text=True, capture_output=True, check=False)
        if proc.stdout:
            print(proc.stdout, end="", flush=True)
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr, flush=True)
        if "DEBUG_STEP1_TERMINATE" in proc.stdout or "Step 1 win confirmed" in proc.stdout:
            print(f"[10TRIAL] stopped after step win on trial {trial}", flush=True)
            return int(proc.returncode)
        if proc.returncode not in (0,):
            print(f"[10TRIAL] trial {trial} exited with {proc.returncode}; stopping.", flush=True)
            return int(proc.returncode)
    print("[10TRIAL] completed all trials without a Step 1 win.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
