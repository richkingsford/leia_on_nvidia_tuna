#!/usr/bin/env python3
"""Sprint empty Step 1 wins with a simple floor-safe backoff between wins."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot
from telemetry_robot import SPEED_SCORE_MIN, speed_power_pwm_for_cmd


TARGET_WINS = 100
MAX_ATTEMPTS = 180
FOLLOW_DURATION_S = 15.0
BACKOFF_PULSE_MS = 300
BACKOFF_SETTLE_S = 0.35
POST_FAIL_RECOVER_S = 3.0
PREGAME_STABLE_FRAMES = 2
PREGAME_STABLE_WINDOW_MM = 25.0
SPRINT_GHOST_MIN_Y_MM = 999.0
OUT_PATH = Path("runs") / f"s1_empty_sprint_100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _fmt_reading(reading: dict | None) -> str:
    if not isinstance(reading, dict):
        return "no reading"
    try:
        return (
            f"dist={float(reading.get('dist_mm')):.1f}mm "
            f"x={float(reading.get('x_mm')):+.1f}mm "
            f"y={float(reading.get('y_mm')):+.1f}mm "
            f"conf={float(reading.get('conf')):.0f}%"
        )
    except (TypeError, ValueError):
        return str(reading.get("reason") or "reading unavailable")


def _read_confident(vision: BrickDetector, robot: Robot, *, reason: str, timeout_s: float = 8.0) -> dict:
    print(f"[SPRINT] visibility gate: {reason}", flush=True)
    follow._stop_robot(robot)
    follow._reset_follow_reading_history(vision, allow_large_dist_jump=True)
    try:
        setattr(vision, "_follow_last_stable_reading", None)
    except Exception:
        pass
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable: list[dict] = []
    last = follow.brick_motion_measurement_from_result(None)
    while time.monotonic() <= deadline:
        last = follow._read_brick_measurement(vision, jump_guard=True)
        if bool(last.get("confident")):
            if stable and not follow._readings_close_for_jump_confirmation(
                stable[-1],
                last,
                window_mm=PREGAME_STABLE_WINDOW_MM,
            ):
                stable.clear()
            stable.append(dict(last))
            stable = stable[-PREGAME_STABLE_FRAMES:]
            if len(stable) >= PREGAME_STABLE_FRAMES:
                chosen = follow._reading_median(stable)
                print(f"[SPRINT] stable visibility ok: {_fmt_reading(chosen)}", flush=True)
                follow._set_follow_last_stable_reading(vision, chosen)
                return chosen
        else:
            stable.clear()
        time.sleep(0.12)
    print(f"[SPRINT] stable visibility failed: {_fmt_reading(last)}", flush=True)
    return last


def _backoff_after_win(vision: BrickDetector, robot: Robot, win: dict | None) -> dict:
    _power, backoff_pwm, _score, min_duration_ms = speed_power_pwm_for_cmd("b", SPEED_SCORE_MIN)
    duration_ms = max(
        int(BACKOFF_PULSE_MS),
        int(round(float(min_duration_ms))),
    )
    before_text = _fmt_reading(win)
    print(f"[SPRINT] backoff after win: BCK {duration_ms}ms from {before_text}", flush=True)
    follow._reset_follow_reading_history(vision)
    safety_reading = follow._read_brick_measurement(vision)
    if not bool(safety_reading.get("confident")):
        safety_reading = follow._wait_for_visibility_recovery(
            vision,
            robot,
            safety_reading,
            timeout_s=4.0,
            sample_s=0.12,
            context="s1_empty_sprint_backoff_precheck",
        )
    send_result = follow.guarded_send_command_pwm(
        robot,
        "b",
        int(round(float(backoff_pwm))),
        duration_ms=duration_ms,
        reading=safety_reading if isinstance(safety_reading, dict) else None,
        context="s1_empty_sprint_backoff",
    )
    time.sleep(max(float(BACKOFF_SETTLE_S), (float(duration_ms) / 1000.0) + 0.08))
    follow._stop_robot(robot)
    follow._reset_follow_reading_history(vision)
    after = follow._read_brick_measurement(vision)
    if not bool(after.get("confident")):
        after = follow._wait_for_visibility_recovery(
            vision,
            robot,
            after,
            timeout_s=4.0,
            sample_s=0.12,
            context="s1_empty_sprint_backoff",
        )
    print(f"[SPRINT] after backoff: {_fmt_reading(after)}", flush=True)
    return {
        "send_result": send_result,
        "duration_ms": int(duration_ms),
        "after": after,
    }


def _record(attempt: int, wins: int, stats: dict, backoff: dict | None) -> dict:
    return {
        "attempt": int(attempt),
        "wins_after_attempt": int(wins),
        "won": bool(int(stats.get("win_count", 0) or 0) > 0),
        "win": stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else None,
        "latest": stats.get("latest_step1_gap") if isinstance(stats.get("latest_step1_gap"), dict) else None,
        "movement_attempts": int(stats.get("follow_attempt_count", 0) or 0),
        "miss_reasons": dict(stats.get("miss_reasons") or {}),
        "hard_ghost_stop": bool(stats.get("hard_ghost_jump_stop")),
        "stall_guard_triggered": bool(stats.get("stall_guard_triggered")),
        "backoff": backoff,
    }


def _install_sprint_ghost_filter():
    original = follow._read_brick_measurement

    def _filtered_read(vision, *args, **kwargs):
        reading = original(vision, *args, **kwargs)
        if not isinstance(reading, dict) or not bool(reading.get("confident")):
            return reading
        try:
            dist = float(reading.get("dist_mm"))
            y_mm = float(reading.get("y_mm"))
        except (TypeError, ValueError):
            return reading
        if y_mm > SPRINT_GHOST_MIN_Y_MM:
            rejected = dict(reading)
            rejected["confident"] = False
            rejected["reason"] = "s1_sprint_y_zero_ghost"
            rejected["s1_sprint_y_zero_ghost"] = True
            return rejected
        return reading

    follow._read_brick_measurement = _filtered_read
    return original


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    wins = 0
    attempts = 0
    print(
        f"[SPRINT] Empty S1 sprint: target={TARGET_WINS} wins, max_attempts={MAX_ATTEMPTS}, data={OUT_PATH}",
        flush=True,
    )
    vision = None
    robot = None
    original_read = _install_sprint_ghost_filter()
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = Robot()
        with OUT_PATH.open("w", encoding="utf-8") as out:
            while wins < TARGET_WINS and attempts < MAX_ATTEMPTS:
                attempts += 1
                print(f"[SPRINT] attempt {attempts}: wins={wins}/{TARGET_WINS}", flush=True)
                start = _read_confident(vision, robot, reason="pre-attempt")
                if not bool(start.get("confident")):
                    print("[SPRINT] no confident brick; stopping sprint.", flush=True)
                    return 3
                stats = follow._follow_loop(
                    vision,
                    robot,
                    duration_s=FOLLOW_DURATION_S,
                    reset_after_win=False,
                    stop_after_win=True,
                    debug_mode=False,
                )
                won = bool(int(stats.get("win_count", 0) or 0) > 0)
                backoff = None
                if won:
                    wins += 1
                    win = stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else {}
                    print(f"[SPRINT] WIN {wins}/{TARGET_WINS}: {_fmt_reading(win)}", flush=True)
                    if wins < TARGET_WINS:
                        backoff = _backoff_after_win(vision, robot, win)
                else:
                    latest = stats.get("latest_step1_gap") if isinstance(stats.get("latest_step1_gap"), dict) else {}
                    print(
                        f"[SPRINT] miss attempt {attempts}: {_fmt_reading(latest)} "
                        f"reasons={dict(stats.get('miss_reasons') or {})}",
                        flush=True,
                    )
                    if bool(stats.get("hard_ghost_jump_stop")):
                        print("[SPRINT] hard ghost stop; pausing to reacquire before next attempt.", flush=True)
                    time.sleep(float(POST_FAIL_RECOVER_S))
                out.write(json.dumps(_record(attempts, wins, stats, backoff), sort_keys=True) + "\n")
                out.flush()
        print(f"[SPRINT] done: wins={wins}/{TARGET_WINS} attempts={attempts} data={OUT_PATH}", flush=True)
        return 0 if wins >= TARGET_WINS else 2
    finally:
        follow._read_brick_measurement = original_read
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
