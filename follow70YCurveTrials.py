#!/usr/bin/env python3
"""Collect Step 1 follow-game y-curve data with y-only micro resets."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from helper_robot_control import Robot


TRIALS = 70
RESET_UP_MIN_MS = 500
RESET_UP_MAX_MS = 1000
OUT_PATH = Path("runs/follow70_y_curve_trials.jsonl")
MAX_ACCEPTED_ABS_X_ERR_MM = 40.0
MAX_CONSECUTIVE_NO_WINS = 3
HAPPY_DIST_TOL_MM = 8.0
HAPPY_X_TOL_MM = 4.0
HAPPY_Y_TOL_MM = 4.0
FOLLOW_CMD = [
    sys.executable,
    "a_follow_the_brick.py",
    "--game-profile",
    "empty",
    "--duration-s",
    "35",
    "--debug-mode",
    "--skip-vision-preflight",
]

WIN_RE = re.compile(
    r"WIN #(?P<win>\d+): dist_err=(?P<dist>[+-]?\d+(?:\.\d+)?)mm "
    r"x_err=(?P<x>[+-]?\d+(?:\.\d+)?)mm\s+y_err=(?P<y>[+-]?\d+(?:\.\d+)?)mm"
)
ACT_RE = re.compile(
    r"\[FOLLOW\]\s+(?P<action>[A-Z0-9_]+)\s+dist_err=(?P<dist>[+-]?\d+(?:\.\d+)?)mm\s+"
    r"x_err=(?P<x>[+-]?\d+(?:\.\d+)?)mm\s+y_err=(?P<y>[+-]?\d+(?:\.\d+)?)mm\s+"
    r"(?:t=(?P<duration>\d+)ms\s+)?conf=(?P<conf>\d+(?:\.\d+)?)%"
)


def _mast_up(duration_ms: int) -> dict:
    robot = Robot()
    try:
        result = robot.send_command_pwm("d", robot.MAX_PWM, duration_ms=int(duration_ms))
        print(f"[70TRIAL] y micro-reset up {int(duration_ms)}ms: {result}", flush=True)
        return dict(result or {})
    finally:
        robot.close()


def _parse_trial(stdout: str, *, trial: int, reset_ms: int | None) -> dict:
    acts = []
    win = None
    for line in stdout.splitlines():
        act_match = ACT_RE.search(line)
        if act_match:
            acts.append(
                {
                    "action": act_match.group("action"),
                    "dist_err_mm": float(act_match.group("dist")),
                    "x_err_mm": float(act_match.group("x")),
                    "y_err_mm": float(act_match.group("y")),
                    "duration_ms": int(act_match.group("duration") or 0),
                    "confidence_pct": float(act_match.group("conf")),
                }
            )
        win_match = WIN_RE.search(line)
        if win_match:
            win = {
                "dist_err_mm": float(win_match.group("dist")),
                "x_err_mm": float(win_match.group("x")),
                "y_err_mm": float(win_match.group("y")),
            }
    return {
        "trial": int(trial),
        "won": win is not None,
        "win": win,
        "acts": acts,
        "micro_reset_up_ms_after": reset_ms,
    }


def _has_bad_lock(record: dict) -> bool:
    return any(abs(float(act.get("x_err_mm", 0.0))) > MAX_ACCEPTED_ABS_X_ERR_MM for act in record.get("acts") or [])


def _won_in_happy_zone(record: dict) -> bool:
    win = record.get("win")
    if not isinstance(win, dict):
        return False
    return (
        abs(float(win.get("dist_err_mm", 999.0))) <= HAPPY_DIST_TOL_MM
        and abs(float(win.get("x_err_mm", 999.0))) <= HAPPY_X_TOL_MM
        and abs(float(win.get("y_err_mm", 999.0))) <= HAPPY_Y_TOL_MM
    )


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    wins = 0
    consecutive_no_wins = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for trial in range(1, TRIALS + 1):
            print(f"[70TRIAL] trial {trial}/{TRIALS}", flush=True)
            proc = subprocess.run(FOLLOW_CMD, text=True, capture_output=True, check=False)
            if proc.stdout:
                print(proc.stdout, end="", flush=True)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr, flush=True)
            reset_ms = random.randint(RESET_UP_MIN_MS, RESET_UP_MAX_MS)
            record = _parse_trial(proc.stdout or "", trial=trial, reset_ms=reset_ms)
            record["returncode"] = int(proc.returncode)
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            completed += 1
            wins += 1 if bool(record.get("won")) else 0
            consecutive_no_wins = 0 if record.get("won") else consecutive_no_wins + 1
            bad_lock = _has_bad_lock(record)
            print(
                f"[70TRIAL] trial {trial} {'WIN' if record.get('won') else 'NO_WIN'}; "
                f"acts={len(record.get('acts') or [])}; wins={wins}/{completed}; "
                f"data={OUT_PATH}",
                flush=True,
            )
            if not _won_in_happy_zone(record):
                print(
                    "[70TRIAL] no Step 1 win; stopping without y micro-reset so the mast stays in-zone.",
                    flush=True,
                )
                return int(proc.returncode) if proc.returncode else 4
            if proc.returncode not in (0,) and not record.get("acts"):
                print(
                    f"[70TRIAL] trial {trial} exited with {proc.returncode} before motion; "
                    "stopping without y micro-reset because the brick is not confidently visible.",
                    flush=True,
                )
                return int(proc.returncode)
            if bad_lock:
                print(
                    f"[70TRIAL] trial {trial} accepted |x_err| > {MAX_ACCEPTED_ABS_X_ERR_MM:.0f}mm; "
                    "stopping before collecting polluted y-curve data.",
                    flush=True,
                )
                return 2
            if consecutive_no_wins >= MAX_CONSECUTIVE_NO_WINS:
                print(
                    f"[70TRIAL] {consecutive_no_wins} consecutive no-win trials; "
                    "stopping before further drift.",
                    flush=True,
                )
                return 3
            _mast_up(reset_ms)
            time.sleep(0.4)
            if proc.returncode not in (0,):
                print(f"[70TRIAL] trial {trial} exited with {proc.returncode}; stopping.", flush=True)
                return int(proc.returncode)
    print(f"[70TRIAL] completed {completed} trials; wins={wins}; data={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
