#!/usr/bin/env python3
"""Practice empty Step 1 repeatedly with a half-distance reset."""

from __future__ import annotations

import copy
import argparse
import json
import random
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


TRIALS = 70
FOLLOW_DURATION_S = 15.0
OUT_PATH = Path("runs/practice_step1_empty_70.jsonl")
RESET_RECOVERY_WAIT_S = 4.0
RESET_RECOVERY_POLL_S = 0.15
START_RECOVERY_WAIT_S = 6.0
START_RECOVERY_POLL_S = 0.15
MAX_RECOVERY_SKIPS = 20
PRACTICE_RESET_X_CURVE_MAX_MS = 75


def _half_dist_reset_config(base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    reverse = cfg.get("reverse_turn") if isinstance(cfg.get("reverse_turn"), dict) else {}
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    try:
        original_ms = int(round(float(straight.get("duration_ms"))))
    except (TypeError, ValueError):
        original_ms = int(follow.DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["duration_ms"])
    straight["duration_ms"] = max(1, int(round(float(original_ms) * 0.5)))
    straight["practice_original_duration_ms"] = int(original_ms)
    reverse["straight_back_first"] = straight
    x_goal_curve = reverse.get("x_goal_curve") if isinstance(reverse.get("x_goal_curve"), dict) else {}
    if x_goal_curve:
        try:
            original_curve_max_ms = int(round(float(x_goal_curve.get("max_duration_ms"))))
        except (TypeError, ValueError):
            original_curve_max_ms = 375
        x_goal_curve["practice_original_max_duration_ms"] = int(original_curve_max_ms)
        x_goal_curve["max_duration_ms"] = min(int(original_curve_max_ms), PRACTICE_RESET_X_CURVE_MAX_MS)
        x_goal_curve["practice_cap_reason"] = "avoid_losing_brick_after_step1_practice_reset_turn"
        reverse["x_goal_curve"] = x_goal_curve
    adjustment = reverse.get("adjustment") if isinstance(reverse.get("adjustment"), dict) else {}
    adjustment["enabled"] = False
    adjustment["practice_disabled_reason"] = "step1_empty_practice_uses_partial_reset_not_full_reset_target"
    reverse["adjustment"] = adjustment
    cfg["reverse_turn"] = reverse
    mast_up = cfg.get("mast_up") if isinstance(cfg.get("mast_up"), dict) else {}
    mast_up["enabled"] = False
    mast_up["practice_disabled_reason"] = "step1_empty_practice_keeps_mast_in_vision_zone"
    cfg["mast_up"] = mast_up
    return cfg


def _step1_record(trial: int, stats: dict, reset_result: dict | None) -> dict:
    win = stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else None
    latest = stats.get("latest_step1_gap") if isinstance(stats.get("latest_step1_gap"), dict) else None
    return {
        "trial": int(trial),
        "won": bool(int(stats.get("win_count", 0) or 0) > 0),
        "win": win,
        "latest": latest,
        "movement_attempts": int(stats.get("follow_attempt_count", 0) or 0),
        "stall_guard_triggered": bool(stats.get("stall_guard_triggered")),
        "stall_recovery_boost_count": int(stats.get("stall_recovery_boost_count", 0) or 0),
        "miss_reasons": dict(stats.get("miss_reasons") or {}),
        "reset": reset_result,
    }


def _format_win(win: dict | None) -> str:
    if not isinstance(win, dict):
        return "no Step 1 win"
    try:
        return (
            f"dist_err={float(win.get('dist_err_mm')):+.1f}mm "
            f"x_err={float(win.get('x_err_mm')):+.1f}mm "
            f"y_err={float(win.get('y_err_mm')):+.1f}mm"
        )
    except (TypeError, ValueError):
        return "Step 1 win recorded"


def _reset_result_recovered(reset_result: dict, vision: BrickDetector, robot: Robot) -> dict:
    result = dict(reset_result or {})
    if bool(result.get("success")):
        if result.get("target_met") is False:
            result["practice_target_miss_accepted"] = True
            result["reason"] = f"{result.get('reason') or 'reset'}_practice_partial_target_missed"
        return result
    reason = str(result.get("reason") or "")
    if reason not in {
        "lost_confident_brick_after_reset_x_curve",
        "lost_confident_brick_after_reset",
        "lost_confident_brick_after_reset_straight_back",
    }:
        return result
    print(
        f"[PRACTICE] reset visibility recovery: {reason}; pausing and reacquiring before giving up.",
        flush=True,
    )
    time.sleep(max(0.0, float(follow._reset_post_pause_s())))
    reading = result.get("reading") if isinstance(result.get("reading"), dict) else {}
    recovered = follow._wait_for_visibility_recovery(
        vision,
        robot,
        reading,
        timeout_s=RESET_RECOVERY_WAIT_S,
        sample_s=RESET_RECOVERY_POLL_S,
        context="practice_reset_recovery",
    )
    result["reading"] = recovered
    result["recovery_attempted"] = True
    if bool(recovered.get("confident")):
        reset_cfg = follow._reset_motion_config().get("reverse_turn")
        target_met = follow._reset_xy_target_ready(recovered, reset_cfg if isinstance(reset_cfg, dict) else {})
        result["success"] = True
        result["target_met"] = bool(target_met)
        result["reason"] = f"{reason}_visibility_recovered"
        return result
    result["recovery_failed"] = True
    return result


def _recover_start_visibility(vision: BrickDetector, robot: Robot, *, reason: str) -> bool:
    print(
        f"[PRACTICE] visibility recovery before counting next trial: {reason}; waiting for brick lock.",
        flush=True,
    )
    follow._stop_robot(robot)
    recovered = follow._wait_for_confident_brick(
        vision,
        timeout_s=START_RECOVERY_WAIT_S,
        sample_s=START_RECOVERY_POLL_S,
    )
    return bool(recovered.get("confident"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--target-wins", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--continue-on-miss", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    trials = max(1, int(args.trials))
    target_wins = max(0, int(args.target_wins))
    max_attempts = max(0, int(args.max_attempts))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    base_reset_cfg = follow._reset_motion_config()
    half_reset_cfg = _half_dist_reset_config(base_reset_cfg)
    base_straight = base_reset_cfg.get("reverse_turn", {}).get("straight_back_first", {})
    half_straight = half_reset_cfg.get("reverse_turn", {}).get("straight_back_first", {})
    print(
        f"[PRACTICE] Empty Step 1 x{trials}. Reset after each win uses half straight-back "
        f"distance duration: {int(base_straight.get('duration_ms', 0) or 0)}ms -> "
        f"{int(half_straight.get('duration_ms', 0) or 0)}ms; "
        f"practice reset x-curve capped at {PRACTICE_RESET_X_CURVE_MAX_MS}ms; "
        "reset mast lift disabled for Step 1 practice.",
        flush=True,
    )
    vision = None
    robot = None
    completed = 0
    wins = 0
    original_reset_config_fn = follow._reset_motion_config
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        print("[PRACTICE] Camera ready.", flush=True)
        follow._print_active_success_gates()
        robot = Robot()
        with out_path.open("w", encoding="utf-8") as out:
            trial = 1
            recovery_skips = 0
            while True:
                if target_wins > 0:
                    if wins >= target_wins:
                        break
                    if max_attempts > 0 and completed >= max_attempts:
                        break
                elif trial > trials:
                    break
                label_total = f"target_wins={target_wins}" if target_wins > 0 else str(trials)
                print(f"[PRACTICE] trial {trial}/{label_total}: empty Step 1 (wins={wins})", flush=True)
                if not _recover_start_visibility(
                    vision,
                    robot,
                    reason="pre-trial visibility gate",
                ):
                    recovery_skips += 1
                    if recovery_skips <= MAX_RECOVERY_SKIPS:
                        continue
                    print(
                        f"[PRACTICE] unable to recover brick vision after {recovery_skips} pre-trial waits; stopping.",
                        flush=True,
                    )
                    return 6
                stats = follow._follow_loop(
                    vision,
                    robot,
                    duration_s=FOLLOW_DURATION_S,
                    reset_after_win=False,
                    stop_after_win=True,
                    debug_mode=False,
                )
                completed += 1
                won = bool(int(stats.get("win_count", 0) or 0) > 0)
                wins += 1 if won else 0
                reset_result = None
                if not won:
                    if int(stats.get("follow_attempt_count", 0) or 0) <= 0 and _recover_start_visibility(
                        vision,
                        robot,
                        reason="no confident brick before any movement",
                    ):
                        recovery_skips += 1
                        if recovery_skips <= MAX_RECOVERY_SKIPS:
                            completed -= 1
                            continue
                    if int(stats.get("follow_attempt_count", 0) or 0) <= 0 and recovery_skips <= MAX_RECOVERY_SKIPS:
                        recovery_skips += 1
                        completed -= 1
                        continue
                    if _recover_start_visibility(
                        vision,
                        robot,
                        reason="trial missed Step 1 before reset",
                    ):
                        recovery_skips += 1
                        if recovery_skips <= MAX_RECOVERY_SKIPS:
                            record = _step1_record(trial, stats, reset_result)
                            record["recovered_without_reset"] = True
                            out.write(json.dumps(record, sort_keys=True) + "\n")
                            out.flush()
                            trial += 1
                            continue
                    record = _step1_record(trial, stats, reset_result)
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    if not bool(args.continue_on_miss):
                        print(
                            f"[PRACTICE] trial {trial} did not win; stopping before reset. "
                            f"Data: {out_path}",
                            flush=True,
                        )
                        return 4
                    print(
                        f"[PRACTICE] trial {trial} did not win; continuing without reset. "
                        f"Data: {out_path}",
                        flush=True,
                    )
                    trial += 1
                    continue
                more_to_run = (
                    (target_wins > 0 and wins < target_wins)
                    or (target_wins <= 0 and trial < trials)
                )
                if more_to_run:
                    print(f"[PRACTICE] trial {trial} win: {_format_win(stats.get('last_step1_win'))}", flush=True)
                    follow._reset_motion_config = lambda: copy.deepcopy(half_reset_cfg)
                    reset_result = follow._run_reset_sequence(vision, robot, rng=random)
                    reset_result = _reset_result_recovered(reset_result, vision, robot)
                    follow._reset_motion_config = original_reset_config_fn
                    print(
                        f"[PRACTICE] trial {trial} half-dist reset: "
                        f"success={bool(reset_result.get('success'))} reason={reset_result.get('reason')} "
                        f"turn={str(reset_result.get('turn_cmd') or '').upper()}",
                        flush=True,
                    )
                    if not bool(reset_result.get("success")):
                        if _recover_start_visibility(
                            vision,
                            robot,
                            reason=f"reset failed: {reset_result.get('reason')}",
                        ):
                            recovery_skips += 1
                            if recovery_skips <= MAX_RECOVERY_SKIPS:
                                record = _step1_record(trial, stats, reset_result)
                                record["recovered_after_failed_reset"] = True
                                out.write(json.dumps(record, sort_keys=True) + "\n")
                                out.flush()
                                trial += 1
                                continue
                        record = _step1_record(trial, stats, reset_result)
                        out.write(json.dumps(record, sort_keys=True) + "\n")
                        out.flush()
                        if not bool(args.continue_on_miss):
                            return 5
                        print(
                            f"[PRACTICE] reset failed after trial {trial}; continuing after recovery path. "
                            f"Data: {out_path}",
                            flush=True,
                        )
                    time.sleep(0.25)
                record = _step1_record(trial, stats, reset_result)
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
                trial += 1
        print(f"[PRACTICE] completed {completed} trials; wins={wins}/{completed}; data={out_path}", flush=True)
        return 0
    finally:
        follow._reset_motion_config = original_reset_config_fn
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
