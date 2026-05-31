#!/usr/bin/env python3
"""Baseline drill for empty Step 1 + Step 2, then a shortened reset.

One trial:
  1. Run the production follow loop until empty Step 1 wins and Step 2 parks.
  2. Score Step 1, Step 2, severe overshoots, and wrong-way gap acts.
  3. Run a half reset before the next trial only after Step 1 and Step 2 win.

This script intentionally uses a_follow_the_brick.py's production planner for
the baseline. Experiments should change one thing at a time and compare against
the JSONL summaries this script writes.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


OUT_PATH = Path("runs") / f"step12_empty_halfreset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


COUNT_KEYS = (
    "sample_count",
    "confident_sample_count",
    "not_confident_count",
    "follow_attempt_count",
    "win_count",
    "reset_attempt_count",
    "reset_count",
    "reset_target_met_count",
    "step2_attempt_count",
    "step2_count",
    "step2_target_met_count",
    "step2_confirmed_win_count",
    "step2_unconfirmed_win_count",
    "step2_creep_attempt_count",
)

COUNT_MAP_KEYS = (
    "act_counts",
    "sent_act_counts",
    "blocked_act_counts",
    "observed_after_act_counts",
    "no_observed_after_act_counts",
    "miss_reasons",
)

LIST_KEYS = (
    "gap_closure_samples",
    "x_curve_samples",
    "win_target_closeness_pct",
    "win_dist_target_closeness_pct",
    "win_x_target_closeness_pct",
    "win_y_target_closeness_pct",
    "step2_target_closeness_pct",
    "step2_dist_target_closeness_pct",
    "step2_x_target_closeness_pct",
    "step2_y_target_closeness_pct",
    "step2_dist_after_mm",
    "step2_x_after_mm",
    "step2_y_after_mm",
)


def _jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _compact_stats(stats: dict) -> dict:
    return {key: value for key, value in (stats or {}).items() if _jsonable(value)}


def _merge_counts(dst: dict, src: dict) -> None:
    for key in COUNT_KEYS:
        dst[key] = int(dst.get(key, 0) or 0) + int((src or {}).get(key, 0) or 0)
    for key in COUNT_MAP_KEYS:
        out = dst.setdefault(key, {})
        for name, count in ((src or {}).get(key) or {}).items():
            out[str(name)] = int(out.get(str(name), 0) or 0) + int(count or 0)
    for key in LIST_KEYS:
        values = (src or {}).get(key)
        if isinstance(values, list):
            dst.setdefault(key, []).extend(values)


def _wrong_way_samples(stats: dict) -> list[dict]:
    samples = (stats or {}).get("gap_closure_samples")
    if not isinstance(samples, list):
        return []
    return [
        row
        for row in samples
        if (
            isinstance(row, dict)
            and bool(row.get("regressed"))
            and _sample_is_primary_axis(row)
        )
    ]


def _severe_overshoot_samples(stats: dict) -> list[dict]:
    samples = (stats or {}).get("gap_closure_samples")
    if not isinstance(samples, list):
        return []
    return [
        row
        for row in samples
        if (
            isinstance(row, dict)
            and bool(row.get("overshot"))
            and not bool(row.get("overshot_within_tolerance"))
            and _sample_is_primary_axis(row)
        )
    ]


def _sample_is_primary_axis(sample: dict) -> bool:
    """Only score the axis an act was intended to close."""
    action = str((sample or {}).get("action") or "").upper()
    axis = str((sample or {}).get("axis") or "").lower()
    if "MAST" in action or action.startswith("Y_"):
        return axis == "y"
    if "TURN" in action or "BIAS" in action or "NUDGE" in action:
        return axis == "x"
    if action.startswith(("FWD", "BCK", "STEP2_PRECISION_FWD", "STEP2_PRECISION_BCK", "STEP2_CREEP")):
        return axis == "dist"
    return True


def _worst_sample(samples: list[dict], metric: str) -> dict | None:
    if not samples:
        return None
    return max(samples, key=lambda row: float(row.get(metric, 0.0) or 0.0))


def _trial_summary(trial_n: int, stats: dict) -> dict:
    wrong_way = _wrong_way_samples(stats)
    severe_overshoots = _severe_overshoot_samples(stats)
    s1_won = int((stats or {}).get("win_count", 0) or 0) >= 1
    s2_attempted = int((stats or {}).get("step2_attempt_count", 0) or 0) >= 1
    s2_won = int((stats or {}).get("step2_target_met_count", 0) or 0) >= 1
    return {
        "trial": int(trial_n),
        "s1_won": bool(s1_won),
        "s2_attempted": bool(s2_attempted),
        "s2_won": bool(s2_won),
        "clean": bool(s1_won and s2_won and not wrong_way and not severe_overshoots),
        "wrong_way_count": len(wrong_way),
        "severe_overshoot_count": len(severe_overshoots),
        "worst_wrong_way": _worst_sample(wrong_way, "regression_mm"),
        "worst_severe_overshoot": _worst_sample(severe_overshoots, "after_abs"),
        "attempts": int((stats or {}).get("follow_attempt_count", 0) or 0),
        "step2_creeps": int((stats or {}).get("step2_creep_attempt_count", 0) or 0),
        "miss_reasons": dict((stats or {}).get("miss_reasons") or {}),
        "last_step1_win": (stats or {}).get("last_step1_win"),
        "latest_step1_gap": (stats or {}).get("latest_step1_gap"),
    }


def _print_trial_line(summary: dict) -> None:
    status = "CLEAN" if summary["clean"] else "MISS"
    print(
        f"[STEP12] T{summary['trial']}: {status} "
        f"s1={int(summary['s1_won'])} s2={int(summary['s2_won'])} "
        f"wrong_way={summary['wrong_way_count']} "
        f"severe_overshoot={summary['severe_overshoot_count']} "
        f"attempts={summary['attempts']}",
        flush=True,
    )


def _required_wins_for_rate(trials: int, min_win_rate: float) -> int:
    trial_goal = max(1, int(trials))
    rate = max(0.0, min(1.0, float(min_win_rate)))
    return int(max(1, math.ceil((float(trial_goal) * float(rate)) - 1e-9)))


def _longest_clean_streak(summaries: list[dict]) -> int:
    longest = 0
    current = 0
    for row in summaries:
        if bool((row or {}).get("clean")):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _block_summary_counts(summaries: list[dict], *, requested_trials: int, min_win_rate: float) -> dict:
    trial_goal = max(1, int(requested_trials))
    completed = int(len(summaries or []))
    completed_den = max(1, completed)
    clean = sum(1 for row in summaries if bool(row.get("clean")))
    s1 = sum(1 for row in summaries if bool(row.get("s1_won")))
    s2 = sum(1 for row in summaries if bool(row.get("s2_won")))
    wrong_way = sum(int(row.get("wrong_way_count", 0) or 0) for row in summaries)
    severe = sum(int(row.get("severe_overshoot_count", 0) or 0) for row in summaries)
    required_wins = _required_wins_for_rate(trial_goal, min_win_rate)
    s1_win_rate = float(s1) / float(trial_goal)
    s2_win_rate = float(s2) / float(trial_goal)
    meets_trial_count = bool(completed >= trial_goal)
    meets_min_win_rate = bool(s1 >= required_wins and s2 >= required_wins)
    meets_safety = bool(int(severe) == 0 and int(wrong_way) == 0)
    clean_streak = _longest_clean_streak(summaries)
    return {
        "requested_trials": int(trial_goal),
        "completed_trials": int(completed),
        "clean_trials": int(clean),
        "s1_wins": int(s1),
        "s2_wins": int(s2),
        "required_wins": int(required_wins),
        "s1_win_rate": float(s1_win_rate),
        "s2_win_rate": float(s2_win_rate),
        "completed_s1_win_rate": float(s1) / float(completed_den),
        "completed_s2_win_rate": float(s2) / float(completed_den),
        "min_win_rate": max(0.0, min(1.0, float(min_win_rate))),
        "meets_trial_count": bool(meets_trial_count),
        "meets_min_win_rate": bool(meets_min_win_rate),
        "meets_safety": bool(meets_safety),
        "wrong_way_count": int(wrong_way),
        "severe_overshoot_count": int(severe),
        "longest_clean_streak": int(clean_streak),
        "meets_100x_proof": bool(
            trial_goal >= 100
            and completed >= 100
            and clean_streak >= 100
            and s1 >= 100
            and s2 >= 100
            and meets_safety
        ),
    }


def _configure_half_reset(fraction: float, *, scale_mast: bool = False) -> tuple[object, dict]:
    old_fn = follow._reset_motion_config
    base = copy.deepcopy(follow._reset_motion_config())
    cfg = copy.deepcopy(base)
    frac = max(0.1, min(1.0, float(fraction)))
    mast_up = cfg.get("mast_up") if isinstance(cfg.get("mast_up"), dict) else {}
    mast_up["enabled"] = False
    if bool(scale_mast):
        for key in ("duration_ms", "min_duration_ms", "max_duration_ms"):
            if key in mast_up:
                mast_up[key] = max(1, int(round(float(mast_up.get(key) or 0) * frac)))
    cfg["mast_up"] = mast_up
    rev = cfg.get("reverse_turn") if isinstance(cfg.get("reverse_turn"), dict) else {}

    straight = rev.get("straight_back_first") if isinstance(rev.get("straight_back_first"), dict) else {}
    for key in ("duration_ms", "duration_min_ms", "duration_max_ms"):
        if key in straight:
            straight[key] = max(1, int(round(float(straight.get(key) or 0) * frac)))
    rev["straight_back_first"] = straight

    x_curve = rev.get("x_goal_curve") if isinstance(rev.get("x_goal_curve"), dict) else {}
    if "target_fraction" in x_curve:
        x_curve["target_fraction"] = max(0.0, min(1.0, float(x_curve.get("target_fraction") or 0.0) * frac))
    for key in ("max_duration_ms", "min_duration_ms", "chunk_ms"):
        if key in x_curve:
            x_curve[key] = max(50 if key == "chunk_ms" else 1, int(round(float(x_curve.get(key) or 0) * frac)))
    rev["x_goal_curve"] = x_curve

    adjustment = rev.get("adjustment") if isinstance(rev.get("adjustment"), dict) else {}
    adjustment["enabled"] = False
    rev["adjustment"] = adjustment
    rev["post_pause_s"] = min(float(rev.get("post_pause_s", 0.5) or 0.5), 0.5)
    cfg["reverse_turn"] = rev

    follow._reset_motion_config = lambda: copy.deepcopy(cfg)
    return old_fn, cfg


def _apply_step2_strong_y_config(cfg: dict) -> None:
    step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
    targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
    step2["freeze_xz_after_xz_target"] = True
    targets["dist_tol_mm"] = 10.0
    step2["targets"] = targets
    step2["step_timeout_s"] = 35.0
    step2["precision_max_attempts"] = 20
    step2["precision_hard_max_attempts"] = 120
    step2["precision_drive_min_pulse_ms"] = 100
    step2["precision_drive_max_pulse_ms"] = 180
    step2["precision_mast_pulse_ms"] = 1000
    step2["precision_mast_small_gap_min_pulse_ms"] = 300
    step2["precision_mast_small_gap_max_pulse_ms"] = 900
    step2["precision_settle_s"] = 0.25
    cfg["step2"] = step2


def _install_close_start_escape_plan() -> object:
    old_fn = follow._follow_action_plan

    def patched(reading: dict) -> dict:
        plan = old_fn(reading)
        if not isinstance(reading, dict) or str((plan or {}).get("kind") or "") == "hold":
            return plan
        if str((plan or {}).get("reason") or "") == "protect_lower_edge":
            return plan
        try:
            dist_err = float(reading.get("dist_mm")) - float(follow._dist_target_mm())
            x_err = float(reading.get("x_mm")) - float(follow._x_target_mm())
        except (TypeError, ValueError):
            return plan
        dist_tol = float(follow._dist_tol_mm())
        dist_outside = max(0.0, abs(float(dist_err)) - float(dist_tol))
        if not (float(dist_err) < -(float(dist_tol) + 25.0) and float(dist_outside) >= 25.0):
            return plan
        try:
            pwm = int(follow._approved_straight_drive_pwm("b"))
        except Exception:
            pwm = 103
        return {
            "kind": "drive",
            "cmd": "b",
            "action": "BCK_ESCAPE",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "x_outside_mm": max(0.0, abs(float(x_err)) - float(follow._x_tol_mm())),
            "dist_outside_mm": float(dist_outside),
            "duration_ms": 450,
            "pwm": int(pwm or 103),
            "distance_creep": True,
            "reason": "too_close_straight_escape",
        }

    follow._follow_action_plan = patched
    return old_fn


def _install_step2_mast_d_only() -> object:
    old_fn = follow._step2_precision_mast_cmd
    follow._step2_precision_mast_cmd = lambda _y_err: "d"
    return old_fn


def _apply_strict_no_change_guard(cfg: dict, *, max_tries: int = 1) -> None:
    guard = cfg.get("act_stall_guard") if isinstance(cfg.get("act_stall_guard"), dict) else {}
    guard["max_no_change_tries"] = int(max(1, max_tries))
    guard["recovery_boost_enabled"] = False
    cfg["act_stall_guard"] = guard


def _pregame_unlock_needed(reading: dict) -> bool:
    if not isinstance(reading, dict):
        return False
    source = str(reading.get("vision_geometry_source") or "").lower()
    status = str(reading.get("vision_status") or "").lower()
    reason = str(reading.get("reason") or "").lower()
    if not bool(reading.get("confident")):
        return "native color rectangle mismatch" in status or reason == "not_visible"
    try:
        dist_mm = float(reading.get("dist_mm"))
    except (TypeError, ValueError):
        return False
    try:
        y_mm = float(reading.get("y_mm"))
    except (TypeError, ValueError):
        y_mm = -999.0
    return bool(dist_mm <= 130.0 and ("green_edge_close_range" in source or y_mm > -35.0))


def _pregame_mast_unlock(vision: BrickDetector, robot: Robot, reading: dict, *, trial_n: int) -> dict:
    if not _pregame_unlock_needed(reading):
        return reading
    print("[STEP12] T%d: pregame mast unlock d 500ms" % int(trial_n), flush=True)
    robot.send_command_pwm("d", 255, duration_ms=500)
    time.sleep(0.65)
    follow._stop_robot(robot)
    try:
        follow._reset_follow_reading_history(vision, allow_large_dist_jump=True)
    except TypeError:
        follow._reset_follow_reading_history(vision)
    return follow._wait_for_confident_brick(
        vision,
        timeout_s=4.0,
        sample_s=0.12,
    )


def _install_experiment(name: str) -> tuple[object, dict | None, object | None, object | None]:
    """Install a temporary follow-motion experiment for this process."""
    old_fn = follow._follow_motion_config
    old_plan_fn = None
    old_step2_mast_cmd_fn = None
    experiment = str(name or "baseline").strip().lower()
    if experiment in {"", "baseline"}:
        return old_fn, None, old_plan_fn, old_step2_mast_cmd_fn

    cfg = copy.deepcopy(follow._follow_motion_config())
    if experiment == "step2_xz_freeze":
        step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
        targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
        step2["freeze_xz_after_xz_target"] = True
        targets["dist_tol_mm"] = 10.0
        step2["targets"] = targets
        step2["step_timeout_s"] = 20.0
        step2["precision_max_attempts"] = 14
        step2["precision_hard_max_attempts"] = 80
        step2["precision_drive_min_pulse_ms"] = 100
        step2["precision_drive_max_pulse_ms"] = 180
        step2["precision_mast_pulse_ms"] = 500
        step2["precision_mast_small_gap_min_pulse_ms"] = 220
        step2["precision_mast_small_gap_max_pulse_ms"] = 500
        step2["precision_settle_s"] = 0.18
        cfg["step2"] = step2
    elif experiment == "step2_xz_freeze_strong_y":
        _apply_step2_strong_y_config(cfg)
    elif experiment == "step2_strong_y_bck_escape":
        _apply_step2_strong_y_config(cfg)
        old_plan_fn = _install_close_start_escape_plan()
    elif experiment == "step2_strong_y_pregame_mast_unlock":
        _apply_step2_strong_y_config(cfg)
    elif experiment == "step2_strong_y_half_mast_reset":
        _apply_step2_strong_y_config(cfg)
    elif experiment == "step2_strong_y_mast_d_only":
        _apply_step2_strong_y_config(cfg)
        old_step2_mast_cmd_fn = _install_step2_mast_d_only()
    elif experiment == "step2_strong_y_strict_stall":
        _apply_step2_strong_y_config(cfg)
        _apply_strict_no_change_guard(cfg, max_tries=1)
    else:
        raise ValueError(f"unknown experiment: {name}")

    follow._follow_motion_config = lambda: copy.deepcopy(cfg)
    return old_fn, cfg, old_plan_fn, old_step2_mast_cmd_fn


def _run_half_reset(
    vision: BrickDetector,
    robot: Robot,
    *,
    fraction: float,
    trial_n: int,
    scale_mast: bool = False,
) -> dict:
    old_fn, cfg = _configure_half_reset(fraction, scale_mast=scale_mast)
    try:
        print(f"[STEP12] T{trial_n}: half reset fraction={fraction:.2f}", flush=True)
        result = follow._run_reset_sequence(vision, robot, rng=random)
        return {"result": result, "config": cfg}
    finally:
        follow._reset_motion_config = old_fn


def _wait_for_pregame_visibility(vision: BrickDetector, robot: Robot, timeout_s: float) -> dict:
    reading = follow._wait_for_confident_brick(
        vision,
        timeout_s=float(timeout_s),
        sample_s=0.12,
    )
    if not bool(reading.get("confident")):
        follow._stop_robot(robot)
        reading = follow._wait_for_visibility_recovery(
            vision,
            robot,
            reading,
            timeout_s=4.0,
            sample_s=0.12,
            context="step12_pregame",
        )
    if not bool(reading.get("confident")):
        follow._stop_robot(robot)
    return reading


def _pregame_pickup_suspected(reading: dict, *, min_dist_mm: float, max_y_mm: float) -> bool:
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False
    try:
        dist_mm = float(reading.get("dist_mm"))
        y_mm = float(reading.get("y_mm"))
    except (TypeError, ValueError):
        return False
    return bool(float(dist_mm) >= float(min_dist_mm) and float(y_mm) <= float(max_y_mm))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--trial-duration-s", type=float, default=80.0)
    parser.add_argument("--reset-fraction", type=float, default=0.5)
    parser.add_argument("--pregame-timeout-s", type=float, default=8.0)
    parser.add_argument("--min-win-rate", type=float, default=0.85)
    parser.add_argument("--pickup-suspect-min-dist-mm", type=float, default=260.0)
    parser.add_argument("--pickup-suspect-max-y-mm", type=float, default=-80.0)
    parser.add_argument("--disable-pickup-suspect-guard", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--experiment",
        choices=(
            "baseline",
            "step2_xz_freeze",
            "step2_xz_freeze_strong_y",
            "step2_strong_y_bck_escape",
            "step2_strong_y_pregame_mast_unlock",
            "step2_strong_y_half_mast_reset",
            "step2_strong_y_mast_d_only",
            "step2_strong_y_strict_stall",
        ),
        default="baseline",
    )
    parser.add_argument("--reset-after-last", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    old_follow_config_fn, experiment_config, old_action_plan_fn, old_step2_mast_cmd_fn = _install_experiment(
        str(args.experiment)
    )

    aggregate: dict = {}
    summaries = []
    abort_reason = None
    vision = BrickDetector(debug=True)
    robot = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        follow._print_active_success_gates()
        with out.open("w", encoding="utf-8") as fh:
            for trial_n in range(1, int(args.trials) + 1):
                print(f"\n[STEP12] === Trial {trial_n}/{args.trials} ===", flush=True)
                follow._set_game_profile("empty")
                abort_trials = False
                pregame = _wait_for_pregame_visibility(vision, robot, float(args.pregame_timeout_s))
                if str(args.experiment) == "step2_strong_y_pregame_mast_unlock":
                    pregame = _pregame_mast_unlock(vision, robot, pregame, trial_n=trial_n)
                if not bool(pregame.get("confident")):
                    stats = follow._new_game_stats()
                    stats["sample_count"] = 1
                    stats["not_confident_count"] = 1
                    follow._bump_stat_count(stats, "miss_reasons", "pregame_no_visibility")
                    print("[STEP12] Pregame visibility failed; stopping trials with no reset.", flush=True)
                    abort_trials = True
                    abort_reason = "pregame_no_visibility"
                elif (
                    not bool(args.disable_pickup_suspect_guard)
                    and _pregame_pickup_suspected(
                        pregame,
                        min_dist_mm=float(args.pickup_suspect_min_dist_mm),
                        max_y_mm=float(args.pickup_suspect_max_y_mm),
                    )
                ):
                    stats = follow._new_game_stats()
                    stats["sample_count"] = 1
                    stats["confident_sample_count"] = 1
                    follow._bump_stat_count(stats, "miss_reasons", "pregame_pickup_suspected")
                    follow._stop_robot(robot)
                    print(
                        "[STEP12] Pregame pickup suspected "
                        f"(dist={float(pregame.get('dist_mm')):.1f}mm, y={float(pregame.get('y_mm')):+.1f}mm); "
                        "no motion, stopping trials.",
                        flush=True,
                    )
                    abort_trials = True
                    abort_reason = "pregame_pickup_suspected"
                else:
                    stats = follow._follow_loop(
                        vision,
                        robot,
                        duration_s=float(args.trial_duration_s),
                        reset_after_win=False,
                        stop_after_step2=True,
                    )
                summary = _trial_summary(trial_n, stats)
                summaries.append(summary)
                _merge_counts(aggregate, stats)
                reset_record = None
                reset_due = trial_n < int(args.trials) or bool(args.reset_after_last)
                if not abort_trials and not (bool(summary["s1_won"]) and bool(summary["s2_won"])):
                    follow._stop_robot(robot)
                    abort_trials = True
                    reset_record = {
                        "skipped": True,
                        "reason": "trial_not_s1_s2_win_stop_no_reset",
                    }
                    abort_reason = "trial_not_s1_s2_win_stop_no_reset"
                    print("[STEP12] Trial did not win both Step 1 and Step 2; no reset, stopping trials.", flush=True)
                if not abort_trials and reset_due:
                    post_trial = follow._read_brick_measurement(vision)
                    if not bool(post_trial.get("confident")):
                        follow._stop_robot(robot)
                        abort_trials = True
                        reset_record = {
                            "skipped": True,
                            "reason": "post_trial_no_confident_visibility_stop_no_reset",
                            "reading": post_trial,
                        }
                        abort_reason = "post_trial_no_confident_visibility_stop_no_reset"
                        print("[STEP12] Post-trial visibility is not confident; no reset, stopping trials.", flush=True)
                if not abort_trials and reset_due:
                    reset_record = _run_half_reset(
                        vision,
                        robot,
                        fraction=float(args.reset_fraction),
                        trial_n=trial_n,
                        scale_mast=str(args.experiment) == "step2_strong_y_half_mast_reset",
                    )
                row = {
                    "kind": "trial",
                    "experiment": str(args.experiment),
                    "summary": summary,
                    "stats": _compact_stats(stats),
                    "reset": reset_record,
                }
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                _print_trial_line(summary)
                if abort_trials:
                    break

        min_win_rate = max(0.0, min(1.0, float(args.min_win_rate)))
        block_summary = _block_summary_counts(
            summaries,
            requested_trials=int(args.trials),
            min_win_rate=min_win_rate,
        )
        final_summary = {
            "kind": "summary",
            "trials": int(args.trials),
            **block_summary,
            "aborted_early": bool(abort_reason is not None or not block_summary["meets_trial_count"]),
            "abort_reason": abort_reason,
            "out": str(out),
            "experiment": str(args.experiment),
            "experiment_config": experiment_config,
            "most_egregious_problem": follow._most_egregious_problem(aggregate),
        }
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(final_summary, sort_keys=True) + "\n")
        print(
            f"\n[STEP12] DONE completed={block_summary['completed_trials']}/{args.trials} "
            f"clean={block_summary['clean_trials']}/{args.trials} "
            f"s1={block_summary['s1_wins']}/{args.trials} ({block_summary['s1_win_rate']:.0%}) "
            f"s2={block_summary['s2_wins']}/{args.trials} ({block_summary['s2_win_rate']:.0%}) "
            f"required={block_summary['required_wins']} "
            f"wrong_way={block_summary['wrong_way_count']} "
            f"severe_overshoot={block_summary['severe_overshoot_count']} "
            f"abort={abort_reason or 'none'} data={out}",
            flush=True,
        )
        try:
            print(follow._format_game_results_table(aggregate), flush=True)
        except Exception as exc:
            print(f"[STEP12] aggregate table unavailable: {exc}", flush=True)
        return 0 if bool(block_summary["meets_trial_count"] and block_summary["meets_min_win_rate"] and block_summary["meets_safety"]) else 1
    finally:
        follow._follow_motion_config = old_follow_config_fn
        if old_action_plan_fn is not None:
            follow._follow_action_plan = old_action_plan_fn
        if old_step2_mast_cmd_fn is not None:
            follow._step2_precision_mast_cmd = old_step2_mast_cmd_fn
        follow._stop_robot(robot)
        try:
            robot.close()
        except Exception:
            pass
        try:
            vision.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
