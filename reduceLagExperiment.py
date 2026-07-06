#!/usr/bin/env python3
"""Measure Leia's sensing and motion-response timing without changing production.

The stationary-stack trial reuses the production brick reader, crawl PWM,
gentle duty-turn packet, safety gate, and serial transport. It can estimate
    vision cadence, command dispatch time, and command-to-observed-motion lag. A
    later moving-stack trial is required for true stimulus-to-response latency.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import a_follow_the_brick as follow


DEFAULT_OUTPUT_DIR = Path("/tmp/leia_reduce_lag")


class Timeline:
    def __init__(self) -> None:
        self.started_ns = time.monotonic_ns()
        self.events: list[dict] = []

    def now_ms(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1_000_000.0

    def add(self, event: str, **values) -> dict:
        row = {"event": str(event), "t_ms": round(self.now_ms(), 3), **values}
        self.events.append(row)
        return row


class FastLockedReader:
    """Accept normal single-frame movement; confirm only suspicious jumps."""

    def __init__(self, baseline: dict, *, motion_wrong_way_mm: float = 8.0) -> None:
        self.accepted = dict(baseline)
        self.jump_candidates: deque[dict] = deque(maxlen=3)
        self.motion_wrong_way_mm = max(0.0, float(motion_wrong_way_mm))

    @staticmethod
    def _delta(first: dict, second: dict) -> tuple[float, float, float] | None:
        first_dist = _number(first.get("dist_mm"))
        first_x = _number(first.get("x_mm"))
        second_dist = _number(second.get("dist_mm"))
        second_x = _number(second.get("x_mm"))
        if None in (first_dist, first_x, second_dist, second_x):
            return None
        dist_delta = float(second_dist) - float(first_dist)
        x_delta = float(second_x) - float(first_x)
        return dist_delta, x_delta, math.hypot(dist_delta, x_delta)

    def update(
        self,
        raw: dict,
        *,
        expected_motion: dict | None = None,
    ) -> tuple[dict, dict]:
        if not isinstance(raw, dict) or not bool(raw.get("confident")):
            reused = dict(self.accepted)
            reused["fast_reused_lock"] = True
            return reused, {"fresh_accepted": False, "filter_action": "reuse_lock_nonconfident"}

        jump_cfg = follow._vision_jump_guard_config()
        delta = self._delta(self.accepted, raw)
        motion_reject_reasons: list[str] = []
        if delta is not None and isinstance(expected_motion, dict):
            dist_delta, x_delta, _vector_delta = delta
            expected_dist_sign = int(expected_motion.get("dist_sign", 0) or 0)
            expected_x_sign = int(expected_motion.get("x_sign", 0) or 0)
            if expected_dist_sign < 0 and dist_delta > self.motion_wrong_way_mm:
                motion_reject_reasons.append("dist_moved_farther_than_command_allows")
            elif expected_dist_sign > 0 and dist_delta < -self.motion_wrong_way_mm:
                motion_reject_reasons.append("dist_moved_closer_than_command_allows")
            if expected_x_sign < 0 and x_delta > self.motion_wrong_way_mm:
                motion_reject_reasons.append("x_moved_opposite_right_turn")
            elif expected_x_sign > 0 and x_delta < -self.motion_wrong_way_mm:
                motion_reject_reasons.append("x_moved_opposite_left_turn")
            if abs(dist_delta) > float(jump_cfg.get("max_dist_jump_mm", 22.0)):
                motion_reject_reasons.append("dist_step_exceeds_physical_limit")
            if abs(x_delta) > float(jump_cfg.get("max_x_jump_mm", 18.0)):
                motion_reject_reasons.append("x_step_exceeds_physical_limit")
        if motion_reject_reasons:
            self.jump_candidates.clear()
            reused = dict(self.accepted)
            reused["fast_reused_lock"] = True
            return reused, {
                "fresh_accepted": False,
                "filter_action": "hold_lock_motion_inconsistent_jump",
                "motion_reject_reasons": motion_reject_reasons,
                "candidate_delta": delta,
                "expected_motion": dict(expected_motion),
            }
        suspicious = False
        if delta is not None:
            dist_delta, x_delta, vector_delta = delta
            suspicious = bool(
                abs(dist_delta) > float(jump_cfg.get("max_dist_jump_mm", 22.0))
                or abs(x_delta) > float(jump_cfg.get("max_x_jump_mm", 18.0))
                or vector_delta > float(jump_cfg.get("max_vector_jump_mm", 26.0))
            )
        if not suspicious:
            self.accepted = dict(raw)
            self.jump_candidates.clear()
            return dict(self.accepted), {
                "fresh_accepted": True,
                "filter_action": "accept_single_normal_frame",
                "accepted_delta": delta,
            }

        window_mm = float(jump_cfg.get("confirm_window_mm", 10.0))
        if self.jump_candidates:
            candidate_delta = self._delta(self.jump_candidates[-1], raw)
            agrees = bool(
                candidate_delta is not None
                and abs(candidate_delta[0]) <= window_mm
                and abs(candidate_delta[1]) <= window_mm
                and candidate_delta[2] <= window_mm * math.sqrt(2.0)
            )
            if not agrees:
                self.jump_candidates.clear()
        self.jump_candidates.append(dict(raw))
        if len(self.jump_candidates) >= 3:
            self.accepted = follow._reading_median(list(self.jump_candidates))
            self.accepted["fast_jump_confirmed"] = True
            self.jump_candidates.clear()
            return dict(self.accepted), {
                "fresh_accepted": True,
                "filter_action": "accept_confirmed_large_jump",
                "accepted_delta": delta,
            }

        reused = dict(self.accepted)
        reused["fast_reused_lock"] = True
        return reused, {
            "fresh_accepted": False,
            "filter_action": "hold_lock_suspicious_jump",
            "jump_candidate_count": len(self.jump_candidates),
            "candidate_delta": delta,
        }


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reading_fields(reading: dict | None) -> dict:
    row = reading if isinstance(reading, dict) else {}
    return {
        "confident": bool(row.get("confident")),
        "dist_mm": _number(row.get("dist_mm")),
        "x_mm": _number(row.get("x_mm")),
        "y_mm": _number(row.get("y_mm")),
        "confidence_pct": _number(row.get("confidence_pct", row.get("conf"))),
        "reason": str(row.get("reason") or ""),
        "vision_geometry_source": str(row.get("vision_geometry_source") or ""),
        "cluster_seen": row.get("empty_cluster_seen"),
        "cluster_kept": row.get("empty_cluster_kept"),
    }


def _read(vision, timeline: Timeline, *, phase: str) -> dict:
    started = timeline.now_ms()
    reading = follow._read_brick_measurement(vision, jump_guard=True)
    completed = timeline.now_ms()
    timeline.add(
        "observation",
        phase=str(phase),
        read_started_ms=round(started, 3),
        read_duration_ms=round(completed - started, 3),
        **_reading_fields(reading),
    )
    return reading


def _unsmoothed_x_mm(vision, reading: dict) -> float | None:
    dist_mm = _number(reading.get("dist_mm"))
    if dist_mm is None:
        return None
    center = getattr(vision, "_native_last_good_center", None)
    if not isinstance(center, (tuple, list)) or len(center) < 1:
        center = getattr(getattr(vision, "_detector", None), "_native_last_good_center", None)
    if not isinstance(center, (tuple, list)) or len(center) < 1:
        return None
    detector = getattr(vision, "_detector", vision)
    estimate = getattr(detector, "_estimate_offset_x_mm", None)
    if not callable(estimate):
        return None
    try:
        return float(estimate(float(center[0]), float(dist_mm)))
    except (TypeError, ValueError):
        return None


def _read_fast(
    vision,
    timeline: Timeline,
    reader: FastLockedReader,
    *,
    phase: str,
    use_unsmoothed_x: bool = True,
    expected_motion: dict | None = None,
) -> dict:
    started = timeline.now_ms()
    raw = follow._read_raw_empty_brick_measurement(vision)
    detector_smoothed_x_mm = _number(raw.get("x_mm"))
    unsmoothed_x_mm = _unsmoothed_x_mm(vision, raw)
    control_raw = dict(raw)
    if bool(use_unsmoothed_x) and unsmoothed_x_mm is not None:
        control_raw["x_mm"] = float(unsmoothed_x_mm)
        control_raw["fast_unsmoothed_x"] = True
    accepted, filter_info = reader.update(
        control_raw,
        expected_motion=expected_motion,
    )
    completed = timeline.now_ms()
    timeline.add(
        "observation",
        phase=str(phase),
        tracking_mode="fast_locked",
        read_started_ms=round(started, 3),
        read_duration_ms=round(completed - started, 3),
        raw_confident=bool(raw.get("confident")),
        raw_dist_mm=_number(raw.get("dist_mm")),
        raw_x_mm=_number(raw.get("x_mm")),
        detector_smoothed_x_mm=detector_smoothed_x_mm,
        unsmoothed_x_mm=unsmoothed_x_mm,
        steering_x_source=(
            "current_bbox_center" if bool(use_unsmoothed_x) and unsmoothed_x_mm is not None else "detector_ema"
        ),
        **filter_info,
        **_reading_fields(accepted),
    )
    return accepted


def _percentile(values: list[float], pct: float):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(pct)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
    cleaned = [float(value) for value in values]
    return {
        "count": len(cleaned),
        "median_ms": round(statistics.median(cleaned), 3),
        "p95_ms": round(float(_percentile(cleaned, 0.95)), 3),
        "max_ms": round(max(cleaned), 3),
    }


def _observed_delta_mm(baseline: dict, reading: dict) -> tuple[float | None, float | None, float | None]:
    base_dist = _number(baseline.get("dist_mm"))
    base_x = _number(baseline.get("x_mm"))
    dist = _number(reading.get("dist_mm"))
    x = _number(reading.get("x_mm"))
    if None in (base_dist, base_x, dist, x):
        return None, None, None
    dist_delta = float(dist) - float(base_dist)
    x_delta = float(x) - float(base_x)
    return dist_delta, x_delta, math.hypot(dist_delta, x_delta)


def _turn_direction(args, baseline: dict) -> tuple[str, float]:
    target = float(follow._x_target_mm())
    x_mm = _number(baseline.get("x_mm"))
    x_delta = 0.0 if x_mm is None else float(x_mm) - target
    if args.direction in {"l", "r"}:
        return str(args.direction), x_delta
    turn = follow._turn_cmd_to_close_x_gap(x_delta)
    if turn not in {"l", "r"}:
        turn = str(args.center_direction)
    return turn, x_delta


def _steering_plan(
    args,
    reading: dict,
    *,
    pwm: int,
    cfg: dict,
    force_straight: bool = False,
) -> dict:
    x_mm = _number(reading.get("x_mm"))
    x_delta = 0.0 if x_mm is None else float(x_mm) - float(follow._x_target_mm())
    deadband = float(cfg["micro_x_deadband_mm"])
    if bool(force_straight) or (args.direction == "auto" and abs(x_delta) <= deadband):
        duration_ms = int(cfg["straight_ms"])
        return {
            "mode": "straight",
            "turn_cmd": None,
            "x_delta_mm": x_delta,
            "duration_ms": duration_ms,
            "actions": follow._gap_crawl_straight_actions(pwm, duration_ms),
            "key": ("straight", duration_ms),
        }
    turn_cmd, _unused = _turn_direction(args, reading)
    both_ms = int(cfg["both_ms"])
    hold_ms = int(cfg["hold_gentle_ms"])
    return {
        "mode": "gentle_turn",
        "turn_cmd": turn_cmd,
        "x_delta_mm": x_delta,
        "duration_ms": both_ms + hold_ms,
        "actions": follow._gap_crawl_duty_turn_actions(turn_cmd, pwm, both_ms, hold_ms),
        "key": ("gentle_turn", turn_cmd, both_ms, hold_ms),
    }


def _hysteretic_steering_plan(
    args,
    reading: dict,
    *,
    pwm: int,
    cfg: dict,
    straight_latched: bool,
) -> tuple[dict, bool]:
    if args.direction != "auto":
        return _steering_plan(args, reading, pwm=pwm, cfg=cfg), False
    x_mm = _number(reading.get("x_mm"))
    x_delta = 0.0 if x_mm is None else float(x_mm) - float(follow._x_target_mm())
    abs_delta = abs(x_delta)
    if bool(straight_latched):
        next_latched = bool(abs_delta < float(args.x_hysteresis_exit_mm))
    else:
        next_latched = bool(abs_delta <= float(args.x_hysteresis_enter_mm))
    plan = _steering_plan(
        args,
        reading,
        pwm=pwm,
        cfg=cfg,
        force_straight=next_latched,
    )
    plan["x_hysteresis_straight"] = bool(next_latched)
    plan["x_hysteresis_enter_mm"] = float(args.x_hysteresis_enter_mm)
    plan["x_hysteresis_exit_mm"] = float(args.x_hysteresis_exit_mm)
    return plan, bool(next_latched)


def _immediate_turn_actions(turn_cmd: str, pwm: int, duration_ms: int) -> list[dict]:
    """Turn now: drive the outer tread and explicitly hold the inner at zero."""
    if str(turn_cmd).strip().lower() == "l":
        return [
            {"target": "r", "action": "f", "pwm": int(pwm), "duration_ms": int(duration_ms)},
            {"target": "l", "action": "s", "pwm": 0, "duration_ms": 0},
        ]
    return [
        {"target": "l", "action": "b", "pwm": int(pwm), "duration_ms": int(duration_ms)},
        {"target": "r", "action": "s", "pwm": 0, "duration_ms": 0},
    ]


def _with_driven_pwm(actions: list[dict], pwm: int) -> list[dict]:
    boosted = []
    for action in actions:
        row = dict(action)
        if str(row.get("action") or "").strip().lower() in {"f", "b"}:
            row["pwm"] = int(pwm)
        boosted.append(row)
    return boosted


def _initial_ramp_pwms(crawl_pwm: int, maximum_pwm: int, steps: int) -> list[int]:
    count = max(2, int(steps))
    lower = int(crawl_pwm)
    upper = max(lower, int(maximum_pwm))
    return [
        int(round(lower + ((upper - lower) * index / float(count - 1))))
        for index in range(count)
    ]


def _turn_first_phase_plan(
    args,
    desired: dict,
    *,
    phase: str,
    drive_pwm: int,
    turn_pwm: int,
) -> dict:
    if desired["mode"] == "straight" or phase == "straight":
        phase_interval_ms = int(args.straight_phase_ms)
        duration_ms = phase_interval_ms + int(args.packet_overlap_ms)
        return {
            **desired,
            "mode": "turn_first_straight_phase",
            "phase": "straight",
            "pwm": int(drive_pwm),
            "phase_interval_ms": phase_interval_ms,
            "duration_ms": duration_ms,
            "actions": follow._gap_crawl_straight_actions(drive_pwm, duration_ms),
            "key": ("turn_first_straight", desired.get("turn_cmd"), phase_interval_ms, duration_ms),
        }
    phase_interval_ms = int(args.turn_phase_ms)
    duration_ms = phase_interval_ms + int(args.packet_overlap_ms)
    return {
        **desired,
        "mode": "turn_first_turn_phase",
        "phase": "turn",
        "pwm": int(turn_pwm),
        "phase_interval_ms": phase_interval_ms,
        "duration_ms": duration_ms,
        "actions": _immediate_turn_actions(str(desired["turn_cmd"]), turn_pwm, duration_ms),
        "key": ("turn_first_turn", desired.get("turn_cmd"), phase_interval_ms, duration_ms),
    }


def _write_result(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_dir / f"stationary_gentle_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare_forward_clearance(
    args,
    vision,
    robot,
    timeline: Timeline,
    reading: dict,
    *,
    required_start_mm: float,
) -> tuple[dict, bool]:
    latest = dict(reading)
    reader = FastLockedReader(
        latest,
        motion_wrong_way_mm=float(args.motion_ghost_wrong_way_mm),
    )
    pwm = int(follow._crawl_forward_pwm())
    packet_ms = 300
    refresh_s = 0.2
    deadline = time.monotonic() + float(args.backup_max_s)
    next_send = time.monotonic()
    timeline.add(
        "clearance_backup_started",
        required_start_mm=round(float(required_start_mm), 3),
        max_duration_s=float(args.backup_max_s),
        pwm=pwm,
    )
    while time.monotonic() < deadline:
        dist_mm = _number(latest.get("dist_mm"))
        if dist_mm is not None and float(dist_mm) >= float(required_start_mm):
            follow._stop_robot(robot)
            timeline.add("clearance_backup_complete", dist_mm=float(dist_mm), success=True)
            return latest, True
        if follow._virtual_safety_dist_exceeded(latest):
            follow._stop_robot(robot)
            timeline.add("clearance_backup_virtual_wall_stop", dist_mm=dist_mm, success=False)
            return latest, False
        if time.monotonic() >= next_send:
            send_started_ms = timeline.now_ms()
            send = follow.guarded_send_command_pwm(
                robot,
                "b",
                pwm,
                duration_ms=packet_ms,
                reading=latest,
                context="lag_experiment_clearance_backup",
            )
            timeline.add(
                "clearance_backup_command",
                dispatch_duration_ms=round(timeline.now_ms() - send_started_ms, 3),
                blocked=bool(isinstance(send, dict) and send.get("blocked")),
                result=send,
            )
            if isinstance(send, dict) and bool(send.get("blocked")):
                follow._stop_robot(robot)
                return latest, False
            next_send = time.monotonic() + refresh_s
        latest = _read_fast(
            vision,
            timeline,
            reader,
            phase="clearance_backup",
            use_unsmoothed_x=not bool(args.disable_raw_x_steering),
        )
        if not bool(latest.get("confident")):
            follow._stop_robot(robot)
            return latest, False
    follow._stop_robot(robot)
    timeline.add(
        "clearance_backup_timeout",
        dist_mm=_number(latest.get("dist_mm")),
        success=False,
    )
    return latest, False


def run(args) -> tuple[int, dict]:
    timeline = Timeline()
    follow._set_game_profile(args.profile)
    timeline.add("experiment_started", profile=args.profile, mode="stationary_stack")

    vision = follow.BrickDetector(debug=False)
    robot = None
    motion_started_ms = None
    motion_stopped_ms = None
    first_observed_motion_ms = None
    stop_reason = "not_started"
    baseline: dict = {}

    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        follow._reset_follow_reading_history(vision, allow_large_dist_jump=True)
        for attempt in range(1, int(args.baseline_max_reads) + 1):
            baseline = _read(vision, timeline, phase="baseline")
            timeline.events[-1]["baseline_attempt"] = int(attempt)
            if bool(baseline.get("confident")):
                break
            if attempt < int(args.baseline_max_reads):
                time.sleep(float(args.baseline_poll_s))
        timeline.add("baseline_accepted", **_reading_fields(baseline))

        if not bool(baseline.get("confident")):
            stop_reason = "baseline_not_confident"
            return 2, _build_payload(
                args, timeline, baseline, stop_reason, motion_started_ms,
                motion_stopped_ms, first_observed_motion_ms,
            )

        dist_mm = _number(baseline.get("dist_mm"))
        stop_floor_mm = max(0.0, float(follow._dist_target_mm()) - float(follow._dist_lower_tol_mm()))
        clearance_scale = max(1.0, float(args.duration_s) / 1.5)
        required_start_mm = stop_floor_mm + (float(args.min_start_clearance_mm) * clearance_scale)
        timeline.add(
            "safety_gate",
            dist_mm=dist_mm,
            stop_floor_mm=round(stop_floor_mm, 3),
            required_start_mm=round(required_start_mm, 3),
            clearance_scale=round(clearance_scale, 3),
        )
        clearance_ok = bool(dist_mm is not None and float(dist_mm) >= required_start_mm)
        if not clearance_ok and not args.probe_only and bool(args.disable_auto_backup):
            stop_reason = "insufficient_forward_clearance"
            return 3, _build_payload(
                args, timeline, baseline, stop_reason, motion_started_ms,
                motion_stopped_ms, first_observed_motion_ms,
            )

        if args.tracking_mode == "fast" and args.profile != "empty":
            stop_reason = "fast_tracking_requires_empty_profile"
            return 4, _build_payload(
                args, timeline, baseline, stop_reason, motion_started_ms,
                motion_stopped_ms, first_observed_motion_ms,
            )

        cfg = follow._gap_crawl_config()
        production_crawl_pwm = int(follow._crawl_forward_pwm())
        pwm = (
            production_crawl_pwm
            if args.drive_pwm is None
            else int(args.drive_pwm)
        )
        turn_pwm = pwm if args.turn_pwm is None else int(args.turn_pwm)
        both_ms = int(cfg["both_ms"])
        hold_ms = int(cfg["hold_gentle_ms"])
        initial_x = _number(baseline.get("x_mm"))
        initial_x_delta = (
            0.0 if initial_x is None else float(initial_x) - float(follow._x_target_mm())
        )
        initial_straight_latched = bool(
            args.direction == "auto"
            and abs(initial_x_delta) <= float(args.x_hysteresis_enter_mm)
        )
        initial_plan, _unused_latch = _hysteretic_steering_plan(
            args,
            baseline,
            pwm=pwm,
            cfg=cfg,
            straight_latched=initial_straight_latched,
        )
        if args.actuation_mode == "turn_first":
            initial_plan = _turn_first_phase_plan(
                args,
                initial_plan,
                phase="turn",
                drive_pwm=pwm,
                turn_pwm=turn_pwm,
            )
        timeline.add(
            "plan",
            tracking_mode=args.tracking_mode,
            actuation_mode=args.actuation_mode,
            steering_recalculated_each_accepted_frame=True,
            turn_cmd=initial_plan["turn_cmd"],
            baseline_x_delta_mm=round(float(initial_plan["x_delta_mm"]), 3),
            pwm=pwm,
            turn_pwm=turn_pwm,
            production_crawl_pwm=production_crawl_pwm,
            both_ms=both_ms,
            hold_ms=hold_ms,
            segment_ms=int(initial_plan["duration_ms"]),
            refresh_ms=(
                int(initial_plan.get("phase_interval_ms", initial_plan["duration_ms"]))
                if args.actuation_mode == "turn_first"
                else int(args.command_refresh_ms)
            ),
            duration_s=float(args.duration_s),
            actions=initial_plan["actions"],
            initial_kick_pwm=int(args.initial_kick_pwm),
            initial_ramp_enabled=not bool(args.disable_initial_ramp),
            initial_ramp_pwms=_initial_ramp_pwms(
                turn_pwm,
                int(args.initial_ramp_max_pwm),
                int(args.initial_ramp_steps),
            ),
            x_hysteresis_enter_mm=float(args.x_hysteresis_enter_mm),
            x_hysteresis_exit_mm=float(args.x_hysteresis_exit_mm),
        )

        if args.probe_only:
            if args.tracking_mode == "fast":
                probe_reader = FastLockedReader(
                    baseline,
                    motion_wrong_way_mm=float(args.motion_ghost_wrong_way_mm),
                )
                for _index in range(int(args.probe_fast_reads)):
                    _read_fast(
                        vision,
                        timeline,
                        probe_reader,
                        phase="fast_probe",
                        use_unsmoothed_x=not bool(args.disable_raw_x_steering),
                    )
            stop_reason = "probe_only" if clearance_ok else "probe_only_insufficient_forward_clearance"
            return 0, _build_payload(
                args, timeline, baseline, stop_reason, motion_started_ms,
                motion_stopped_ms, first_observed_motion_ms,
            )

        robot = follow.Robot()
        if not clearance_ok:
            baseline, clearance_ok = _prepare_forward_clearance(
                args,
                vision,
                robot,
                timeline,
                baseline,
                required_start_mm=required_start_mm,
            )
            if not clearance_ok:
                stop_reason = "automatic_clearance_backup_failed"
                return 3, _build_payload(
                    args, timeline, baseline, stop_reason, motion_started_ms,
                    motion_stopped_ms, first_observed_motion_ms,
                )
            follow._reset_follow_reading_history(vision, allow_large_dist_jump=False)
        countdown_deadline = time.monotonic() + float(args.countdown_s)
        while time.monotonic() < countdown_deadline:
            time.sleep(min(0.05, countdown_deadline - time.monotonic()))

        fast_reader = (
            FastLockedReader(
                baseline,
                motion_wrong_way_mm=float(args.motion_ghost_wrong_way_mm),
            )
            if args.tracking_mode == "fast"
            else None
        )
        latest = baseline
        if fast_reader is not None:
            for _index in range(int(args.pre_motion_fast_reads)):
                latest = _read_fast(
                    vision,
                    timeline,
                    fast_reader,
                    phase="pre_motion",
                    use_unsmoothed_x=not bool(args.disable_raw_x_steering),
                )
            if bool(latest.get("confident")):
                baseline = dict(latest)
                fast_reader = FastLockedReader(
                    baseline,
                    motion_wrong_way_mm=float(args.motion_ghost_wrong_way_mm),
                )
                timeline.add("motion_baseline", **_reading_fields(baseline))
        live_start_dist = _number(baseline.get("dist_mm"))
        if live_start_dist is None or float(live_start_dist) < required_start_mm:
            stop_reason = "insufficient_forward_clearance_after_countdown"
            return 3, _build_payload(
                args, timeline, baseline, stop_reason, motion_started_ms,
                motion_stopped_ms, first_observed_motion_ms,
            )

        motion_started_ms = timeline.now_ms()
        deadline = time.monotonic() + float(args.duration_s)
        next_send = time.monotonic()
        last_plan_key = None
        last_desired_key = None
        live_x = _number(baseline.get("x_mm"))
        live_x_delta = (
            0.0 if live_x is None else float(live_x) - float(follow._x_target_mm())
        )
        x_straight_latched = bool(
            args.direction == "auto"
            and abs(live_x_delta) <= float(args.x_hysteresis_enter_mm)
        )
        turn_first_phase = "turn"
        ramp_pwms = _initial_ramp_pwms(
            turn_pwm,
            int(args.initial_ramp_max_pwm),
            int(args.initial_ramp_steps),
        )
        ramp_active = bool(
            not args.disable_initial_ramp and args.actuation_mode == "turn_first"
        )
        ramp_started_at = None
        ramp_stage_index = -1
        stale_frames = 0
        latest_observation_ms = next(
            (
                float(row["t_ms"])
                for row in reversed(timeline.events)
                if row.get("event") == "observation"
            ),
            None,
        )
        motion_candidates: deque[dict] = deque(
            maxlen=int(args.motion_confirm_frames)
        )
        last_expected_motion: dict | None = None
        stop_reason = "duration_complete"

        while time.monotonic() < deadline:
            desired, x_straight_latched = _hysteretic_steering_plan(
                args,
                latest,
                pwm=pwm,
                cfg=cfg,
                straight_latched=x_straight_latched,
            )
            desired_key = (desired["mode"], desired.get("turn_cmd"))
            desired_changed = last_desired_key is not None and desired_key != last_desired_key
            phase_due = time.monotonic() >= next_send
            ramp_requested_pwm = None
            ramp_stage_to_record = None
            handled_ramp = False
            if ramp_active and desired["mode"] != "straight":
                if ramp_started_at is None:
                    ramp_started_at = time.monotonic()
                ramp_elapsed_ms = (time.monotonic() - ramp_started_at) * 1000.0
                if ramp_elapsed_ms < float(args.turn_phase_ms):
                    stage_span_ms = float(args.turn_phase_ms) / float(len(ramp_pwms))
                    stage_index = min(
                        len(ramp_pwms) - 1,
                        int(ramp_elapsed_ms / max(1.0, stage_span_ms)),
                    )
                    turn_first_phase = "turn"
                    plan = _turn_first_phase_plan(
                        args,
                        desired,
                        phase="turn",
                        drive_pwm=pwm,
                        turn_pwm=turn_pwm,
                    )
                    should_send = bool(
                        last_plan_key is None
                        or desired_changed
                        or stage_index != ramp_stage_index
                    )
                    send_reason = f"initial_ramp_stage_{stage_index + 1}"
                    ramp_requested_pwm = int(ramp_pwms[stage_index])
                    ramp_stage_to_record = int(stage_index)
                    handled_ramp = True
                else:
                    ramp_active = False
                    turn_first_phase = "straight"
                    plan = _turn_first_phase_plan(
                        args,
                        desired,
                        phase="straight",
                        drive_pwm=pwm,
                        turn_pwm=turn_pwm,
                    )
                    should_send = True
                    send_reason = "initial_ramp_to_straight"
                    handled_ramp = True
            elif ramp_active:
                ramp_active = False

            if not handled_ramp and args.actuation_mode == "turn_first":
                if last_desired_key is None or desired_changed:
                    turn_first_phase = "turn" if desired["mode"] != "straight" else "straight"
                elif phase_due:
                    turn_first_phase = (
                        "straight"
                        if desired["mode"] == "straight" or turn_first_phase == "turn"
                        else "turn"
                    )
                plan = _turn_first_phase_plan(
                    args,
                    desired,
                    phase=turn_first_phase,
                    drive_pwm=pwm,
                    turn_pwm=turn_pwm,
                )
                should_send = bool(last_plan_key is None or desired_changed or phase_due)
                send_reason = (
                    "initial"
                    if last_plan_key is None
                    else "steering_change"
                    if desired_changed
                    else "phase_change"
                )
            elif not handled_ramp:
                plan = desired
                plan_changed = last_plan_key is not None and plan["key"] != last_plan_key
                should_send = bool(last_plan_key is None or plan_changed or phase_due)
                send_reason = (
                    "initial"
                    if last_plan_key is None
                    else "steering_change"
                    if plan_changed
                    else "refresh"
                )

            if should_send:
                plan_pwm = int(plan.get("pwm", pwm))
                initial_kick = bool(
                    ramp_requested_pwm is None
                    and last_plan_key is None
                    and int(args.initial_kick_pwm) > plan_pwm
                )
                requested_pwm = (
                    int(ramp_requested_pwm)
                    if ramp_requested_pwm is not None
                    else int(args.initial_kick_pwm)
                    if initial_kick
                    else plan_pwm
                )
                actions_to_send = (
                    _with_driven_pwm(plan["actions"], requested_pwm)
                    if requested_pwm != plan_pwm
                    else plan["actions"]
                )
                decision_ms = timeline.now_ms()
                observation_to_decision_ms = (
                    None
                    if latest_observation_ms is None
                    else max(0.0, decision_ms - latest_observation_ms)
                )
                send_started_ms = timeline.now_ms()
                send = follow.guarded_send_custom_actions_pwm(
                    robot,
                    "f",
                    actions_to_send,
                    duration_ms=int(plan["duration_ms"]),
                    reading=latest,
                    context="lag_experiment_stationary_gentle",
                )
                send_completed_ms = timeline.now_ms()
                timeline.add(
                    "command",
                    decision_ms=round(decision_ms, 3),
                    observation_to_decision_ms=(
                        None
                        if observation_to_decision_ms is None
                        else round(observation_to_decision_ms, 3)
                    ),
                    dispatch_duration_ms=round(send_completed_ms - send_started_ms, 3),
                    send_reason=send_reason,
                    steering_mode=plan["mode"],
                    phase=plan.get("phase"),
                    initial_kick=initial_kick,
                    initial_ramp_stage=(
                        None if ramp_stage_to_record is None else int(ramp_stage_to_record + 1)
                    ),
                    requested_pwm=requested_pwm,
                    turn_cmd=plan["turn_cmd"],
                    x_delta_mm=round(float(plan["x_delta_mm"]), 3),
                    x_hysteresis_straight=bool(x_straight_latched),
                    blocked=bool(isinstance(send, dict) and send.get("blocked")),
                    result=send,
                )
                if isinstance(send, dict) and bool(send.get("blocked")):
                    stop_reason = f"command_blocked:{send.get('reason') or 'unknown'}"
                    break
                last_plan_key = plan["key"]
                last_desired_key = desired_key
                if ramp_stage_to_record is not None:
                    ramp_stage_index = int(ramp_stage_to_record)
                next_send = time.monotonic() + (
                    float(
                        plan.get("phase_interval_ms", plan["duration_ms"])
                        if args.actuation_mode == "turn_first"
                        else args.command_refresh_ms
                    )
                    / 1000.0
                )

            expected_x_sign = 0
            if plan.get("turn_cmd") == "r" and plan.get("phase") != "straight":
                expected_x_sign = -1
            elif plan.get("turn_cmd") == "l" and plan.get("phase") != "straight":
                expected_x_sign = 1
            last_expected_motion = {
                "dist_sign": -1,
                "x_sign": expected_x_sign,
                "phase": str(plan.get("phase") or plan.get("mode") or ""),
                "turn_cmd": plan.get("turn_cmd"),
            }

            if fast_reader is not None:
                latest = _read_fast(
                    vision,
                    timeline,
                    fast_reader,
                    phase="motion",
                    use_unsmoothed_x=not bool(args.disable_raw_x_steering),
                    expected_motion=last_expected_motion,
                )
                fresh_accepted = bool(timeline.events[-1].get("fresh_accepted"))
                stale_frames = 0 if fresh_accepted else stale_frames + 1
                if stale_frames >= int(args.max_stale_frames):
                    stop_reason = "fast_tracking_stale_frame_limit"
                    break
            else:
                latest = _read(vision, timeline, phase="motion")
                fresh_accepted = True
            observation_event = timeline.events[-1]
            latest_observation_ms = float(observation_event["t_ms"])
            if not bool(latest.get("confident")):
                stop_reason = "vision_not_confident"
                break
            latest_dist = _number(latest.get("dist_mm"))
            if latest_dist is not None and latest_dist <= stop_floor_mm:
                stop_reason = "distance_safety_floor"
                break
            dist_delta, x_delta, vector_delta = _observed_delta_mm(baseline, latest)
            timeline.events[-1].update(
                {
                    "baseline_dist_delta_mm": dist_delta,
                    "baseline_x_delta_mm": x_delta,
                    "baseline_vector_delta_mm": vector_delta,
                }
            )
            if (
                bool(fresh_accepted)
                and vector_delta is not None
                and float(vector_delta) >= float(args.motion_threshold_mm)
            ):
                motion_candidates.append(
                    {
                        "t_ms": float(observation_event["t_ms"]),
                        "vector_delta_mm": float(vector_delta),
                        "dist_delta_mm": float(dist_delta),
                        "x_delta_mm": float(x_delta),
                    }
                )
            else:
                motion_candidates.clear()
            if (
                first_observed_motion_ms is None
                and len(motion_candidates) >= int(args.motion_confirm_frames)
            ):
                onset = motion_candidates[0]
                first_observed_motion_ms = float(onset["t_ms"])
                confirmed_ms = timeline.now_ms()
                timeline.add(
                    "observed_motion_onset",
                    threshold_mm=float(args.motion_threshold_mm),
                    confirmation_frames=int(args.motion_confirm_frames),
                    onset_t_ms=round(first_observed_motion_ms, 3),
                    confirmed_t_ms=round(confirmed_ms, 3),
                    confirmation_delay_ms=round(
                        confirmed_ms - first_observed_motion_ms,
                        3,
                    ),
                    vector_delta_mm=round(float(onset["vector_delta_mm"]), 3),
                    dist_delta_mm=round(float(onset["dist_delta_mm"]), 3),
                    x_delta_mm=round(float(onset["x_delta_mm"]), 3),
                )

        follow._stop_robot(robot)
        motion_stopped_ms = timeline.now_ms()
        timeline.add("motion_stopped", reason=stop_reason)

        settle_deadline = time.monotonic() + float(args.settle_observe_s)
        while time.monotonic() < settle_deadline:
            if fast_reader is not None:
                _read_fast(
                    vision,
                    timeline,
                    fast_reader,
                    phase="settle",
                    use_unsmoothed_x=not bool(args.disable_raw_x_steering),
                    expected_motion=last_expected_motion,
                )
            else:
                _read(vision, timeline, phase="settle")
    finally:
        if robot is not None:
            try:
                follow._stop_robot(robot)
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass
        try:
            vision.close()
        except Exception:
            pass

    return 0, _build_payload(
        args, timeline, baseline, stop_reason, motion_started_ms,
        motion_stopped_ms, first_observed_motion_ms,
    )


def _build_payload(
    args,
    timeline: Timeline,
    baseline: dict,
    stop_reason: str,
    motion_started_ms,
    motion_stopped_ms,
    first_observed_motion_ms,
) -> dict:
    configuration = dict(vars(args))
    configuration["output_dir"] = str(configuration.get("output_dir") or "")
    observations = [row for row in timeline.events if row.get("event") == "observation"]
    commands = [row for row in timeline.events if row.get("event") == "command"]
    motion_observations = [row for row in observations if row.get("phase") == "motion"]
    settle_observations = [row for row in observations if row.get("phase") == "settle"]
    read_durations = [float(row["read_duration_ms"]) for row in observations]
    observation_times = [float(row["t_ms"]) for row in observations]
    observation_intervals = [
        later - earlier for earlier, later in zip(observation_times, observation_times[1:])
    ]
    dispatch_durations = [float(row["dispatch_duration_ms"]) for row in commands]
    motion_times = [float(row["t_ms"]) for row in motion_observations]
    motion_intervals = [later - earlier for earlier, later in zip(motion_times, motion_times[1:])]
    command_times = [float(row["t_ms"]) for row in commands]
    command_intervals = [later - earlier for earlier, later in zip(command_times, command_times[1:])]
    observation_to_decision = [
        float(row["observation_to_decision_ms"])
        for row in commands
        if row.get("observation_to_decision_ms") is not None
    ]
    onset_event = next(
        (row for row in timeline.events if row.get("event") == "observed_motion_onset"),
        None,
    )
    filter_actions: dict[str, int] = {}
    motion_reject_reasons: dict[str, int] = {}
    for row in observations:
        action = str(row.get("filter_action") or "")
        if action:
            filter_actions[action] = int(filter_actions.get(action, 0)) + 1
        for reason in row.get("motion_reject_reasons") or []:
            key = str(reason)
            motion_reject_reasons[key] = int(motion_reject_reasons.get(key, 0)) + 1
    command_to_observed = None
    if command_times and first_observed_motion_ms is not None:
        command_to_observed = max(0.0, float(first_observed_motion_ms) - float(command_times[0]))
    decision_to_observed = None
    if commands and first_observed_motion_ms is not None:
        decision_to_observed = max(
            0.0,
            float(first_observed_motion_ms) - float(commands[0]["decision_ms"]),
        )
    actual_motion_duration = None
    if motion_started_ms is not None and motion_stopped_ms is not None:
        actual_motion_duration = max(0.0, float(motion_stopped_ms) - float(motion_started_ms))
    return {
        "schema": "leia.reduce_lag.stationary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "baseline": _reading_fields(baseline),
        "summary": {
            "stop_reason": str(stop_reason),
            "observation_count": len(observations),
            "command_count": len(commands),
            "vision_read_duration": _stats(read_durations),
            "accepted_observation_interval": _stats(observation_intervals),
            "motion_read_duration": _stats(
                [float(row["read_duration_ms"]) for row in motion_observations]
            ),
            "motion_observation_interval": _stats(motion_intervals),
            "command_dispatch_duration": _stats(dispatch_durations),
            "observation_to_decision_duration": _stats(observation_to_decision),
            "command_refresh_interval": _stats(command_intervals),
            "command_to_observed_motion_ms": (
                None if command_to_observed is None else round(command_to_observed, 3)
            ),
            "decision_to_observed_motion_ms": (
                None if decision_to_observed is None else round(decision_to_observed, 3)
            ),
            "motion_onset_threshold_mm": float(args.motion_threshold_mm),
            "motion_onset_confirmation_frames": int(args.motion_confirm_frames),
            "motion_onset_confirmation_delay_ms": (
                None
                if not isinstance(onset_event, dict)
                else _number(onset_event.get("confirmation_delay_ms"))
            ),
            "vision_filter_actions": filter_actions,
            "motion_inconsistent_reject_reasons": motion_reject_reasons,
            "actual_motion_duration_ms": (
                None if actual_motion_duration is None else round(actual_motion_duration, 3)
            ),
            "perception_lag_ms": None,
            "decision_lag_ms": None,
            "end_to_end_lag_ms": None,
            "moving_target_metrics_available": False,
            "moving_target_metrics_reason": "stationary stack has no target-motion stimulus",
            "last_motion_reading": (
                _reading_fields(motion_observations[-1]) if motion_observations else None
            ),
            "last_settle_reading": (
                _reading_fields(settle_observations[-1]) if settle_observations else None
            ),
        },
        "events": timeline.events,
    }


def _print_summary(payload: dict, output_path: Path) -> None:
    summary = payload["summary"]
    baseline = payload["baseline"]
    print(
        "[LAG] Baseline "
        f"dist={baseline.get('dist_mm')}mm x={baseline.get('x_mm')}mm "
        f"confident={baseline.get('confident')}",
        flush=True,
    )
    print(
        f"[LAG] stop={summary.get('stop_reason')} observations={summary.get('observation_count')} "
        f"commands={summary.get('command_count')}",
        flush=True,
    )
    print(f"[LAG] vision reads: {summary.get('vision_read_duration')}", flush=True)
    print(f"[LAG] observation intervals: {summary.get('accepted_observation_interval')}", flush=True)
    print(f"[LAG] command dispatch: {summary.get('command_dispatch_duration')}", flush=True)
    print(
        f"[LAG] observation -> decision: {summary.get('observation_to_decision_duration')}",
        flush=True,
    )
    print(
        "[LAG] decision -> sustained motion onset: "
        f"{summary.get('decision_to_observed_motion_ms')}ms "
        f"({summary.get('motion_onset_threshold_mm')}mm x "
        f"{summary.get('motion_onset_confirmation_frames')} frames)",
        flush=True,
    )
    print(
        f"[LAG] motion-inconsistent frames rejected: "
        f"{summary.get('vision_filter_actions', {}).get('hold_lock_motion_inconsistent_jump', 0)} "
        f"{summary.get('motion_inconsistent_reject_reasons')}",
        flush=True,
    )
    print(f"[LAG] JSON: {output_path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("empty", "holding"), default="empty")
    parser.add_argument("--tracking-mode", choices=("fast", "production"), default="fast")
    parser.add_argument("--actuation-mode", choices=("turn_first", "delayed_hold"), default="turn_first")
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--countdown-s", type=float, default=1.0)
    parser.add_argument("--settle-observe-s", type=float, default=0.8)
    parser.add_argument("--baseline-max-reads", type=int, default=8)
    parser.add_argument("--baseline-poll-s", type=float, default=0.05)
    parser.add_argument("--command-refresh-ms", type=int, default=250)
    parser.add_argument("--turn-phase-ms", type=int, default=200)
    parser.add_argument("--straight-phase-ms", type=int, default=200)
    parser.add_argument("--packet-overlap-ms", type=int, default=100)
    parser.add_argument(
        "--drive-pwm",
        type=int,
        default=None,
        help="Override the straight-crawl PWM for this isolated experiment.",
    )
    parser.add_argument(
        "--turn-pwm",
        type=int,
        default=None,
        help="Override only the driven tread during one-wheel correction phases.",
    )
    parser.add_argument("--initial-kick-pwm", type=int, default=103)
    parser.add_argument("--initial-ramp-max-pwm", type=int, default=160)
    parser.add_argument("--initial-ramp-steps", type=int, default=3)
    ramp_group = parser.add_mutually_exclusive_group()
    ramp_group.add_argument(
        "--enable-initial-ramp",
        dest="disable_initial_ramp",
        action="store_false",
    )
    ramp_group.add_argument(
        "--disable-initial-ramp",
        dest="disable_initial_ramp",
        action="store_true",
    )
    parser.set_defaults(disable_initial_ramp=True)
    parser.add_argument("--max-stale-frames", type=int, default=10)
    parser.add_argument("--probe-fast-reads", type=int, default=5)
    parser.add_argument("--pre-motion-fast-reads", type=int, default=3)
    parser.add_argument("--disable-raw-x-steering", action="store_true")
    parser.add_argument("--disable-auto-backup", action="store_true")
    parser.add_argument("--backup-max-s", type=float, default=2.0)
    parser.add_argument("--x-hysteresis-enter-mm", type=float, default=5.0)
    parser.add_argument("--x-hysteresis-exit-mm", type=float, default=7.0)
    parser.add_argument("--direction", choices=("auto", "l", "r"), default="auto")
    parser.add_argument("--center-direction", choices=("l", "r"), default="l")
    parser.add_argument("--motion-threshold-mm", type=float, default=1.0)
    parser.add_argument("--motion-confirm-frames", type=int, default=3)
    parser.add_argument(
        "--motion-ghost-wrong-way-mm",
        type=float,
        default=8.0,
        help="Reject a per-frame change this far opposite the active motor command.",
    )
    parser.add_argument("--min-start-clearance-mm", type=float, default=40.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.duration_s = max(0.1, min(5.0, float(args.duration_s)))
    args.countdown_s = max(0.0, min(10.0, float(args.countdown_s)))
    args.settle_observe_s = max(0.0, min(3.0, float(args.settle_observe_s)))
    args.baseline_max_reads = max(1, min(20, int(args.baseline_max_reads)))
    args.baseline_poll_s = max(0.0, min(0.5, float(args.baseline_poll_s)))
    args.command_refresh_ms = max(200, min(400, int(args.command_refresh_ms)))
    args.turn_phase_ms = max(200, min(1000, int(args.turn_phase_ms)))
    args.straight_phase_ms = max(200, min(1000, int(args.straight_phase_ms)))
    args.packet_overlap_ms = max(0, min(300, int(args.packet_overlap_ms)))
    if args.drive_pwm is not None:
        args.drive_pwm = max(0, min(255, int(args.drive_pwm)))
    if args.turn_pwm is not None:
        args.turn_pwm = max(0, min(255, int(args.turn_pwm)))
    args.initial_kick_pwm = max(0, min(255, int(args.initial_kick_pwm)))
    args.initial_ramp_max_pwm = max(0, min(255, int(args.initial_ramp_max_pwm)))
    args.initial_ramp_steps = max(2, min(8, int(args.initial_ramp_steps)))
    args.max_stale_frames = max(1, min(30, int(args.max_stale_frames)))
    args.probe_fast_reads = max(1, min(20, int(args.probe_fast_reads)))
    args.pre_motion_fast_reads = max(1, min(10, int(args.pre_motion_fast_reads)))
    args.backup_max_s = max(0.1, min(2.0, float(args.backup_max_s)))
    args.x_hysteresis_enter_mm = max(0.0, float(args.x_hysteresis_enter_mm))
    args.x_hysteresis_exit_mm = max(
        float(args.x_hysteresis_enter_mm),
        float(args.x_hysteresis_exit_mm),
    )
    args.motion_threshold_mm = max(1.0, float(args.motion_threshold_mm))
    args.motion_confirm_frames = max(1, min(10, int(args.motion_confirm_frames)))
    args.motion_ghost_wrong_way_mm = max(
        1.0,
        min(100.0, float(args.motion_ghost_wrong_way_mm)),
    )
    args.min_start_clearance_mm = max(0.0, float(args.min_start_clearance_mm))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    code, payload = run(args)
    output_path = _write_result(args.output_dir, payload)
    _print_summary(payload, output_path)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
