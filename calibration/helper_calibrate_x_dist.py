#!/usr/bin/env python3
"""
Combined one-shot x-axis + distance duration probe.

Each trial performs one turn act and one drive act using the same deterministic
probe duration. This captures whether a single combined correction cycle reduces
both x-axis and distance error at once.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .helper_calibrate import (
    CALIBRATION_DURATION_LIMIT_MS,
    build_linear_duration_schedule,
    build_payload as build_shared_payload,
    coerce_finite_float as shared_coerce_finite_float,
    coerce_float as shared_coerce_float,
    coerce_int as shared_coerce_int,
    ensure_run_dir,
    load_calibration_trial_speed_profile as shared_load_calibration_trial_speed_profile,
    observe_pose_with_reobserve as shared_observe_pose_with_reobserve,
    read_pose as shared_read_pose,
    resolve_calibration_trial_speed_score as shared_resolve_calibration_trial_speed_score,
    write_results as shared_write_results,
)
from helper_robot_control import Robot
from helper_vision_leia import LeiaVision
import telemetry_brick as telemetry_brick_module
from telemetry_process import (
    _average_smoothed_frames as telemetry_average_smoothed_frames,
    _latest_unique_smoothed_frames as telemetry_latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command,
    update_world_from_vision,
)
from telemetry_robot import StepState, WorldModel, normalize_speed_score

try:
    from helper_brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None

OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 1.8
POST_ACT_SETTLE_S = 0.10
POST_RESET_SETTLE_S = 0.30
OBSERVE_SAMPLES_DEFAULT = 3
SPEED_SCORE_DEFAULT = 1
MIN_DURATION_MS_DEFAULT = 200
MAX_DURATION_MS_DEFAULT = 800
DURATION_STEP_MS_DEFAULT = 20
DURATION_CEILING_MS = CALIBRATION_DURATION_LIMIT_MS
TRIAL_SPEED_MODE_DISTANCE_CURVE = "distance_curve"
TRIAL_SPEED_MODE_FIXED = "fixed"
X_AXIS_TARGET_MM_DEFAULT = 0.0
DIST_TARGET_MM_DEFAULT = 166.0
X_DEADBAND_MM_DEFAULT = 0.5
DIST_DEADBAND_MM_DEFAULT = 0.5
MAX_ACTS_PER_TRIAL_DEFAULT = 15
SUCCESS_MARGIN_MM_DEFAULT = 4.0
ACT_DURATION_GROWTH_PER_ACT = 0.12
MIN_LITE_UNIQUE_FRAMES = 3
REOBSERVE_HOLD_S = 0.12
REOBSERVE_ROUNDS = 4
RELAXED_OBSERVE_TIMEOUT_S = 4.0
RESULTS_FILE_DEFAULT: str | None = None
RUN_DIR_ARUCO = Path("Runs - aruco")
RUN_DIR_CYAN = Path("Runs - cyan")
PRESERVE_LIVE_FILES = {
    "calibrate_x_live.json",
    "calibrate_y_live.json",
    "calibrate_dist_live.json",
    "calibrate_x_dist_live.json",
}

RESET_DURATION_SCALE_START = 0.50
RESET_DURATION_SCALE_END = 0.95
RESET_DURATION_SCALE_PER_ACT = 0.10
RESET_DURATION_SCALE_MAX = 1.05
CONFIDENT_PREPOSE_WAIT_TIMEOUT_S = 10.0
CONFIDENT_PREPOSE_POLL_S = 0.05


@dataclass
class TrialResult:
    trial: int
    duration_ms: int
    score_requested: int
    turn_cmd: str | None
    drive_cmd: str | None
    turn_cmd_sent: str | None
    drive_cmd_sent: str | None
    pre_x_mm: float
    post_x_mm: float
    pre_dist_mm: float
    post_dist_mm: float
    raw_x_delta_mm: float
    raw_dist_delta_mm: float
    pre_x_error_mm: float
    post_x_error_mm: float
    pre_dist_error_mm: float
    post_dist_error_mm: float
    x_improvement_mm: float
    dist_improvement_mm: float
    combined_improvement_mm: float
    cmd_delta_mm: float
    wrong_way: bool
    pre_confidence: float
    post_confidence: float
    pre_samples_used: int | None
    post_samples_used: int | None
    pre_pose_source: str | None
    post_pose_source: str | None
    pre_observation_mode: str | None
    post_observation_mode: str | None
    post_reobserved: bool
    acts_used: int
    success_within_margin: bool
    final_combined_error_mm: float


@dataclass
class ResetEffort:
    trial: int
    reset_act: int
    cmd: str
    score_requested: int
    cmd_sent: str | None
    duration_ms: int
    pre_x_mm: float
    post_x_mm: float
    pre_dist_mm: float
    post_dist_mm: float
    raw_x_delta_mm: float
    raw_dist_delta_mm: float
    cmd_delta_mm: float
    wrong_way: bool
    pre_brick_dist_mm: float
    post_brick_dist_mm: float
    pre_confidence: float
    post_confidence: float
    pre_pose_source: str | None
    post_pose_source: str | None
    post_observation_mode: str | None


def log_line(message: str) -> None:
    print(str(message), flush=True)


def _coerce_float(value, fallback=None):
    return shared_coerce_float(value, fallback)


def _coerce_int(value, fallback=None):
    return shared_coerce_int(value, fallback)


def _coerce_finite_float(value) -> float | None:
    return shared_coerce_finite_float(value)


def _run_dir_for_vision(vision_mode: str | None) -> Path:
    mode = str(vision_mode or "").strip().lower()
    if mode == "aruco":
        return Path(RUN_DIR_ARUCO)
    return Path(RUN_DIR_CYAN)


def _trial_speed_profile_for_mode(trial_speed_mode: str | None) -> dict | None:
    mode = str(trial_speed_mode or TRIAL_SPEED_MODE_DISTANCE_CURVE).strip().lower()
    if mode == TRIAL_SPEED_MODE_FIXED:
        return None
    # Reuse distance-based trial speed profile for combined probes.
    return shared_load_calibration_trial_speed_profile("dist")


def _x_cmd_for_error(pre_x_mm: float, target_x_mm: float, deadband_mm: float) -> str | None:
    err = float(pre_x_mm) - float(target_x_mm)
    if abs(err) <= float(deadband_mm):
        return None
    return "r" if err < 0.0 else "l"


def _dist_cmd_for_error(pre_dist_mm: float, target_dist_mm: float, deadband_mm: float) -> str | None:
    err = float(pre_dist_mm) - float(target_dist_mm)
    if abs(err) <= float(deadband_mm):
        return None
    return "f" if err > 0.0 else "b"


def _read_pose(
    vision,
    world,
    *,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None = None,
    min_samples_required: int | None = None,
    on_vision_update=None,
):
    return shared_read_pose(
        vision,
        world,
        samples=samples,
        timeout_s=timeout_s,
        min_sample_time=min_sample_time,
        min_samples_required=min_samples_required,
        observe_sleep_s=OBSERVE_SLEEP_S,
        fallback_step_label="ALIGN_BRICK",
        update_world_from_vision=update_world_from_vision,
        latest_unique_smoothed_frames=telemetry_latest_unique_smoothed_frames,
        average_smoothed_frames=telemetry_average_smoothed_frames,
        lite_gate_unique_frames=lite_gate_unique_frames,
        min_lite_unique_frames=MIN_LITE_UNIQUE_FRAMES,
        on_vision_update=on_vision_update,
    )


def _observe_pose(
    *,
    vision,
    world,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None = None,
) -> tuple[dict | None, dict]:
    return shared_observe_pose_with_reobserve(
        read_pose_fn=_read_pose,
        log_fn=log_line,
        log_prefix="[CALIBRATE_X_DIST]",
        vision=vision,
        world=world,
        samples=samples,
        timeout_s=timeout_s,
        min_sample_time=min_sample_time,
        hold_s=REOBSERVE_HOLD_S,
        reobserve_rounds=REOBSERVE_ROUNDS,
        relaxed_timeout_s=RELAXED_OBSERVE_TIMEOUT_S,
    )


def _wait_for_confident_visible_read(*, vision, world, timeout_s: float, poll_s: float) -> bool:
    deadline = time.time() + max(0.1, float(timeout_s))
    while time.time() < float(deadline):
        try:
            update_world_from_vision(world, vision, log=False)
            gate = telemetry_brick_module.evaluate_start_gates(
                world,
                getattr(world, "step_state", StepState.ALIGN_BRICK),
                getattr(world, "learned_rules", {}),
                getattr(world, "process_rules", None),
            )
            if bool(getattr(gate, "ok", False)):
                return True
        except Exception:
            pass
        time.sleep(max(0.01, float(poll_s)))
    return False


def _send_fixed_score_command(*, robot, world, step: str, cmd: str, score: int, duration_ms: int) -> dict | None:
    duration_override_ms = min(max(1, int(duration_ms)), int(DURATION_CEILING_MS))
    return send_robot_command(
        robot,
        world,
        step,
        str(cmd),
        speed=0.0,
        speed_score=int(score),
        duration_override_ms=int(duration_override_ms),
    )


def _inverse_cmd(cmd: str | None) -> str | None:
    mapping = {
        "l": "r",
        "r": "l",
        "f": "b",
        "b": "f",
    }
    key = str(cmd or "").strip().lower()
    return mapping.get(key)


def _scaled_reset_duration_ms(
    *,
    base_duration_ms: int,
    trial_idx: int,
    trials_planned: int,
    reset_act_idx: int,
) -> int:
    total_trials = max(1, int(trials_planned))
    progress = 0.0
    if total_trials > 1:
        progress = float(max(0, int(trial_idx) - 1)) / float(total_trials - 1)

    run_scale = float(RESET_DURATION_SCALE_START) + (
        float(RESET_DURATION_SCALE_END) - float(RESET_DURATION_SCALE_START)
    ) * float(progress)
    per_act_boost = max(0.0, float(int(reset_act_idx) - 1)) * float(RESET_DURATION_SCALE_PER_ACT)
    final_scale = min(float(RESET_DURATION_SCALE_MAX), float(run_scale + per_act_boost))

    scaled = int(round(float(base_duration_ms) * float(final_scale)))
    return max(1, min(int(DURATION_CEILING_MS), int(scaled)))


def _needs_reset_for_next_trial(
    *,
    post_x_error_mm: float,
    post_dist_error_mm: float,
    combined_improvement_mm: float,
    x_deadband_mm: float,
    dist_deadband_mm: float,
) -> bool:
    still_outside_target = bool(
        float(post_x_error_mm) > float(x_deadband_mm)
        or float(post_dist_error_mm) > float(dist_deadband_mm)
    )
    no_or_negative_progress = bool(float(combined_improvement_mm) < -0.5)
    return bool(still_outside_target and no_or_negative_progress)


def _run_iterative_trial(
    *,
    robot,
    world,
    vision,
    pre_pose: dict,
    base_duration_ms: int,
    effective_score: int,
    center_x_mm: float,
    target_dist_mm: float,
    x_deadband_mm: float,
    dist_deadband_mm: float,
    success_margin_mm: float,
    max_acts_per_trial: int,
    observe_samples: int,
    observe_timeout_s: float,
    post_act_settle_s: float,
) -> tuple[dict | None, dict]:
    current_pose = dict(pre_pose)
    acts_used = 0
    last_turn_cmd = None
    last_drive_cmd = None
    last_turn_action = None
    last_drive_action = None
    last_post_meta = {"mode": "unknown", "reobserved": False}

    for act_idx in range(1, int(max_acts_per_trial) + 1):
        curr_x_mm = float(current_pose.get("offset_x") or 0.0)
        curr_dist_mm = float(current_pose.get("dist") or 0.0)
        curr_x_error_mm = abs(curr_x_mm - float(center_x_mm))
        curr_dist_error_mm = abs(curr_dist_mm - float(target_dist_mm))
        if curr_x_error_mm <= float(success_margin_mm) and curr_dist_error_mm <= float(success_margin_mm):
            return dict(current_pose), {
                "acts_used": int(acts_used),
                "success_within_margin": True,
                "last_turn_cmd": last_turn_cmd,
                "last_drive_cmd": last_drive_cmd,
                "last_turn_action": last_turn_action,
                "last_drive_action": last_drive_action,
                "post_meta": dict(last_post_meta),
            }

        turn_cmd = _x_cmd_for_error(curr_x_mm, center_x_mm, x_deadband_mm)
        drive_cmd = _dist_cmd_for_error(curr_dist_mm, target_dist_mm, dist_deadband_mm)
        if turn_cmd is None and drive_cmd is None:
            return dict(current_pose), {
                "acts_used": int(acts_used),
                "success_within_margin": curr_x_error_mm <= float(success_margin_mm)
                and curr_dist_error_mm <= float(success_margin_mm),
                "last_turn_cmd": last_turn_cmd,
                "last_drive_cmd": last_drive_cmd,
                "last_turn_action": last_turn_action,
                "last_drive_action": last_drive_action,
                "post_meta": dict(last_post_meta),
            }

        duration_scale = 1.0 + (max(0, int(act_idx) - 1) * float(ACT_DURATION_GROWTH_PER_ACT))
        duration_ms = max(1, min(int(DURATION_CEILING_MS), int(round(float(base_duration_ms) * duration_scale))))
        act_start_ts = time.time()
        turn_action = None
        drive_action = None
        if turn_cmd is not None:
            turn_action = _send_fixed_score_command(
                robot=robot,
                world=world,
                step="CALIBRATE_X_DIST",
                cmd=str(turn_cmd),
                score=int(effective_score),
                duration_ms=int(duration_ms),
            )
            last_turn_cmd = str(turn_cmd)
            last_turn_action = dict(turn_action or {})
        if float(post_act_settle_s) > 0.0:
            time.sleep(float(post_act_settle_s))
        if drive_cmd is not None:
            drive_action = _send_fixed_score_command(
                robot=robot,
                world=world,
                step="CALIBRATE_X_DIST",
                cmd=str(drive_cmd),
                score=int(effective_score),
                duration_ms=int(duration_ms),
            )
            last_drive_cmd = str(drive_cmd)
            last_drive_action = dict(drive_action or {})
        if float(post_act_settle_s) > 0.0:
            time.sleep(float(post_act_settle_s))

        min_sample_time = float(act_start_ts) + (float(duration_ms) / 1000.0) + float(post_act_settle_s)
        post_pose, post_meta = _observe_pose(
            vision=vision,
            world=world,
            samples=observe_samples,
            timeout_s=observe_timeout_s,
            min_sample_time=min_sample_time,
        )
        if post_pose is None:
            return None, {
                "abort_reason": f"post_pose_unavailable_act_{int(act_idx)}",
                "acts_used": int(acts_used),
            }
        current_pose = dict(post_pose)
        acts_used = int(act_idx)
        last_post_meta = dict(post_meta or {})

    final_x_mm = float(current_pose.get("offset_x") or 0.0)
    final_dist_mm = float(current_pose.get("dist") or 0.0)
    final_success = (
        abs(final_x_mm - float(center_x_mm)) <= float(success_margin_mm)
        and abs(final_dist_mm - float(target_dist_mm)) <= float(success_margin_mm)
    )
    return dict(current_pose), {
        "acts_used": int(acts_used),
        "success_within_margin": bool(final_success),
        "last_turn_cmd": last_turn_cmd,
        "last_drive_cmd": last_drive_cmd,
        "last_turn_action": last_turn_action,
        "last_drive_action": last_drive_action,
        "post_meta": dict(last_post_meta),
    }


def _build_payload(
    *,
    config: dict,
    durations_ms: list[int],
    trials: list[TrialResult],
    reset_efforts: list[ResetEffort],
    status: str,
    abort_reason: str | None,
) -> dict:
    if trials:
        x_improvements = [float(item.x_improvement_mm) for item in trials]
        dist_improvements = [float(item.dist_improvement_mm) for item in trials]
        combined_improvements = [float(item.combined_improvement_mm) for item in trials]
        successes = [bool(getattr(item, "success_within_margin", False)) for item in trials]
        acts_to_success = [
            int(getattr(item, "acts_used", 0))
            for item in trials
            if bool(getattr(item, "success_within_margin", False))
        ]
        final_combined_errors = [float(getattr(item, "final_combined_error_mm", 0.0)) for item in trials]
    else:
        x_improvements = []
        dist_improvements = []
        combined_improvements = []
        successes = []
        acts_to_success = []
        final_combined_errors = []
    return build_shared_payload(
        source="calibrate_x_dist",
        config=config,
        durations_ms=durations_ms,
        trials=trials,
        reset_efforts=reset_efforts,
        status=status,
        abort_reason=abort_reason,
        extra_fields={
            "summary_components": {
                "median_x_improvement_mm": float(statistics.median(x_improvements)) if x_improvements else None,
                "median_dist_improvement_mm": float(statistics.median(dist_improvements)) if dist_improvements else None,
                "median_combined_improvement_mm": float(statistics.median(combined_improvements)) if combined_improvements else None,
                "success_rate_within_margin": float(sum(1 for item in successes if item) / len(successes)) if successes else None,
                "median_acts_to_success": float(statistics.median(acts_to_success)) if acts_to_success else None,
                "median_final_combined_error_mm": float(statistics.median(final_combined_errors)) if final_combined_errors else None,
            }
        },
    )


def _exit_as_script(exit_code: int) -> None:
    if sys.gettrace() is not None:
        return
    raise SystemExit(int(exit_code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined one-shot x-axis + distance duration probe.")
    parser.add_argument("--trials", type=int, default=None, help="Optional trial cap; default runs full deterministic duration schedule.")
    parser.add_argument("--speed-score", type=int, default=SPEED_SCORE_DEFAULT, help=f"Fixed score used when --trial-speed-mode=fixed (default: {SPEED_SCORE_DEFAULT}).")
    parser.add_argument(
        "--trial-speed-mode",
        choices=[TRIAL_SPEED_MODE_DISTANCE_CURVE, TRIAL_SPEED_MODE_FIXED],
        default=TRIAL_SPEED_MODE_DISTANCE_CURVE,
        help="How to choose trial speed: distance_curve (default) or fixed.",
    )
    parser.add_argument("--center-x-mm", type=float, default=X_AXIS_TARGET_MM_DEFAULT)
    parser.add_argument("--target-dist-mm", type=float, default=DIST_TARGET_MM_DEFAULT)
    parser.add_argument("--x-deadband-mm", type=float, default=X_DEADBAND_MM_DEFAULT)
    parser.add_argument("--dist-deadband-mm", type=float, default=DIST_DEADBAND_MM_DEFAULT)
    parser.add_argument("--vision", choices=["leia", "yolo", "aruco"], default="yolo")
    parser.add_argument("--min-duration-ms", type=int, default=MIN_DURATION_MS_DEFAULT)
    parser.add_argument("--max-duration-ms", type=int, default=MAX_DURATION_MS_DEFAULT)
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES_DEFAULT)
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S)
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S)
    parser.add_argument("--max-acts-per-trial", type=int, default=MAX_ACTS_PER_TRIAL_DEFAULT)
    parser.add_argument("--success-margin-mm", type=float, default=SUCCESS_MARGIN_MM_DEFAULT)
    parser.add_argument(
        "--min-effective-score",
        type=int,
        default=SPEED_SCORE_DEFAULT,
        help=(
            "Lower bound applied to the resolved trial score. "
            "Use 1 to keep trials fixed at 1%% when combined with --trial-speed-mode fixed."
        ),
    )
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT)
    args = parser.parse_args()

    run_dir = _run_dir_for_vision(args.vision)
    ensure_run_dir(
        run_dir=Path(run_dir),
        preserve_live_files=PRESERVE_LIVE_FILES,
    )
    if args.results_file is None:
        args.results_file = str(Path(run_dir) / "calibrate_x_dist_live.json")

    min_duration_ms = max(1, int(args.min_duration_ms))
    max_duration_ms = max(int(min_duration_ms), int(args.max_duration_ms))
    if max_duration_ms > int(DURATION_CEILING_MS):
        max_duration_ms = int(DURATION_CEILING_MS)
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    max_acts_per_trial = max(1, int(args.max_acts_per_trial))
    success_margin_mm = max(0.1, float(args.success_margin_mm))
    min_effective_score = int(normalize_speed_score(int(args.min_effective_score)))
    trials_requested = None if args.trials is None else max(1, int(args.trials))
    speed_score = int(normalize_speed_score(int(args.speed_score)))
    trial_speed_mode = str(args.trial_speed_mode or TRIAL_SPEED_MODE_DISTANCE_CURVE).strip().lower()
    trial_speed_profile = _trial_speed_profile_for_mode(trial_speed_mode)
    center_x_mm = float(args.center_x_mm)
    target_dist_mm = float(args.target_dist_mm)
    x_deadband_mm = max(0.0, float(args.x_deadband_mm))
    dist_deadband_mm = max(0.0, float(args.dist_deadband_mm))

    durations_ms = build_linear_duration_schedule(
        trials=trials_requested,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        duration_step_ms=int(DURATION_STEP_MS_DEFAULT),
    )
    results_path = Path(args.results_file)
    trial_rows: list[TrialResult] = []
    reset_rows: list[ResetEffort] = []
    status = "completed"
    abort_reason = None

    config = {
        "trials": int(len(durations_ms)),
        "requested_trials": None if trials_requested is None else int(trials_requested),
        "speed_score": int(speed_score),
        "trial_speed_mode": str(trial_speed_mode),
        "trial_speed_score_source": "distance_curve" if isinstance(trial_speed_profile, dict) else "arg",
        "center_x_mm": float(center_x_mm),
        "target_dist_mm": float(target_dist_mm),
        "x_deadband_mm": float(x_deadband_mm),
        "dist_deadband_mm": float(dist_deadband_mm),
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "duration_step_ms": int(DURATION_STEP_MS_DEFAULT),
        "observe_samples": int(observe_samples),
        "observe_timeout_s": float(observe_timeout_s),
        "post_act_settle_s": float(post_act_settle_s),
        "post_reset_settle_s": float(POST_RESET_SETTLE_S),
        "max_acts_per_trial": int(max_acts_per_trial),
        "success_margin_mm": float(success_margin_mm),
        "min_effective_score": int(min_effective_score),
        "sequence": "turn_then_drive",
        "reset_duration_scale_start": float(RESET_DURATION_SCALE_START),
        "reset_duration_scale_end": float(RESET_DURATION_SCALE_END),
        "reset_duration_scale_per_act": float(RESET_DURATION_SCALE_PER_ACT),
        "reset_duration_scale_max": float(RESET_DURATION_SCALE_MAX),
    }
    if isinstance(trial_speed_profile, dict):
        config["trial_speed_score_profile"] = json.loads(json.dumps(trial_speed_profile))

    robot = None
    vision = None
    world = None
    try:
        world = WorldModel()
        world.step_state = StepState.ALIGN_BRICK
        world._post_action_observe_delay_s = 0.0
        robot = Robot()
        if args.vision == "yolo":
            if YoloBrickDetector is None:
                raise RuntimeError("YOLO detector module not installed; cannot use --vision yolo")
            vision = YoloBrickDetector(debug=False)
        elif args.vision == "aruco":
            from helper_vision_aruco import ArucoBrickVision

            vision = ArucoBrickVision(debug=False)
        else:
            vision = LeiaVision(debug=False)

        log_line("[CALIBRATE_X_DIST] Starting combined one-shot x+dist probe.")
        log_line(
            f"[CALIBRATE_X_DIST] trials={len(durations_ms)} durations_ms={durations_ms} speed={int(speed_score)}% "
            f"target_x={float(center_x_mm):+.2f}mm target_dist={float(target_dist_mm):.2f}mm"
        )

        for trial_idx, duration_ms in enumerate(durations_ms, start=1):
            trial_label = f"Trial {int(trial_idx)}/{int(len(durations_ms))}"
            if not _wait_for_confident_visible_read(
                vision=vision,
                world=world,
                timeout_s=float(CONFIDENT_PREPOSE_WAIT_TIMEOUT_S),
                poll_s=float(CONFIDENT_PREPOSE_POLL_S),
            ):
                status = "aborted"
                abort_reason = f"confident_read_unavailable_trial_{trial_idx}"
                log_line(
                    f"[CALIBRATE_X_DIST] {trial_label}: no telemetry-confident visible read before pre-pose. Aborting."
                )
                break
            pre_pose, pre_meta = _observe_pose(
                vision=vision,
                world=world,
                samples=observe_samples,
                timeout_s=observe_timeout_s,
            )
            if pre_pose is None:
                status = "aborted"
                abort_reason = f"pre_pose_unavailable_trial_{trial_idx}"
                log_line(f"[CALIBRATE_X_DIST] {trial_label}: no usable pre-pose. Aborting.")
                break

            pre_x_mm = float(pre_pose.get("offset_x") or 0.0)
            pre_dist_mm = float(pre_pose.get("dist") or 0.0)
            effective_score, score_meta = shared_resolve_calibration_trial_speed_score(
                observed_distance_mm=pre_dist_mm,
                requested_score=int(speed_score),
                speed_profile=trial_speed_profile,
            )
            effective_score = max(int(min_effective_score), int(effective_score))
            log_line(
                f"[CALIBRATE_X_DIST] {trial_label}: pre x={pre_x_mm:+.2f}mm dist={pre_dist_mm:.2f}mm "
                f"score={int(effective_score)}% ({str(score_meta.get('source') or 'arg')}) "
                f"target_margin={float(success_margin_mm):.2f}mm max_acts={int(max_acts_per_trial)}."
            )

            post_pose, trial_meta = _run_iterative_trial(
                robot=robot,
                world=world,
                vision=vision,
                pre_pose=pre_pose,
                base_duration_ms=int(duration_ms),
                effective_score=int(effective_score),
                center_x_mm=float(center_x_mm),
                target_dist_mm=float(target_dist_mm),
                x_deadband_mm=float(x_deadband_mm),
                dist_deadband_mm=float(dist_deadband_mm),
                success_margin_mm=float(success_margin_mm),
                max_acts_per_trial=int(max_acts_per_trial),
                observe_samples=int(observe_samples),
                observe_timeout_s=float(observe_timeout_s),
                post_act_settle_s=float(post_act_settle_s),
            )
            if post_pose is None:
                status = "aborted"
                abort_reason = str((trial_meta or {}).get("abort_reason") or f"post_pose_unavailable_trial_{trial_idx}")
                log_line(f"[CALIBRATE_X_DIST] {trial_label}: no usable post-pose. Aborting.")
                break

            turn_cmd = trial_meta.get("last_turn_cmd") if isinstance(trial_meta, dict) else None
            drive_cmd = trial_meta.get("last_drive_cmd") if isinstance(trial_meta, dict) else None
            turn_action = trial_meta.get("last_turn_action") if isinstance(trial_meta, dict) else None
            drive_action = trial_meta.get("last_drive_action") if isinstance(trial_meta, dict) else None
            post_meta = trial_meta.get("post_meta") if isinstance(trial_meta, dict) else {}
            acts_used = int((trial_meta or {}).get("acts_used") or 0)
            success_within_margin = bool((trial_meta or {}).get("success_within_margin"))

            post_x_mm = float(post_pose.get("offset_x") or 0.0)
            post_dist_mm = float(post_pose.get("dist") or 0.0)
            pre_x_error_mm = abs(pre_x_mm - float(center_x_mm))
            post_x_error_mm = abs(post_x_mm - float(center_x_mm))
            pre_dist_error_mm = abs(pre_dist_mm - float(target_dist_mm))
            post_dist_error_mm = abs(post_dist_mm - float(target_dist_mm))
            x_improvement_mm = float(pre_x_error_mm - post_x_error_mm)
            dist_improvement_mm = float(pre_dist_error_mm - post_dist_error_mm)
            combined_improvement_mm = float(x_improvement_mm + dist_improvement_mm)
            raw_x_delta_mm = float(post_x_mm - pre_x_mm)
            raw_dist_delta_mm = float(post_dist_mm - pre_dist_mm)
            cmd_delta_mm = float(math.hypot(raw_x_delta_mm, raw_dist_delta_mm))
            wrong_way = bool(combined_improvement_mm < 0.0)
            final_combined_error_mm = float(post_x_error_mm + post_dist_error_mm)

            row = TrialResult(
                trial=int(trial_idx),
                duration_ms=int(duration_ms),
                score_requested=int(effective_score),
                turn_cmd=turn_cmd,
                drive_cmd=drive_cmd,
                turn_cmd_sent=str((turn_action or {}).get("cmd_sent") or turn_cmd) if turn_cmd is not None else None,
                drive_cmd_sent=str((drive_action or {}).get("cmd_sent") or drive_cmd) if drive_cmd is not None else None,
                pre_x_mm=float(pre_x_mm),
                post_x_mm=float(post_x_mm),
                pre_dist_mm=float(pre_dist_mm),
                post_dist_mm=float(post_dist_mm),
                raw_x_delta_mm=float(raw_x_delta_mm),
                raw_dist_delta_mm=float(raw_dist_delta_mm),
                pre_x_error_mm=float(pre_x_error_mm),
                post_x_error_mm=float(post_x_error_mm),
                pre_dist_error_mm=float(pre_dist_error_mm),
                post_dist_error_mm=float(post_dist_error_mm),
                x_improvement_mm=float(x_improvement_mm),
                dist_improvement_mm=float(dist_improvement_mm),
                combined_improvement_mm=float(combined_improvement_mm),
                cmd_delta_mm=float(cmd_delta_mm),
                wrong_way=bool(wrong_way),
                pre_confidence=float(pre_pose.get("confidence") or 0.0),
                post_confidence=float(post_pose.get("confidence") or 0.0),
                pre_samples_used=_coerce_int(pre_pose.get("samples_used")),
                post_samples_used=_coerce_int(post_pose.get("samples_used")),
                pre_pose_source=str(pre_pose.get("pose_source") or "unknown"),
                post_pose_source=str(post_pose.get("pose_source") or "unknown"),
                pre_observation_mode=str((pre_meta or {}).get("mode") or "unknown"),
                post_observation_mode=str((post_meta or {}).get("mode") or "unknown"),
                post_reobserved=bool((post_meta or {}).get("reobserved")),
                acts_used=int(acts_used),
                success_within_margin=bool(success_within_margin),
                final_combined_error_mm=float(final_combined_error_mm),
            )
            trial_rows.append(row)

            log_line(
                f"[CALIBRATE_X_DIST] {trial_label}: dx={float(raw_x_delta_mm):+.2f}mm dd={float(raw_dist_delta_mm):+.2f}mm "
                f"improve_x={float(x_improvement_mm):+.2f}mm improve_d={float(dist_improvement_mm):+.2f}mm "
                f"combined={float(combined_improvement_mm):+.2f}mm acts={int(acts_used)} "
                f"success_4mm={bool(success_within_margin)}"
            )

            should_attempt_reset = bool(int(trial_idx) < int(len(durations_ms)))
            needs_reset = bool(
                should_attempt_reset
                and _needs_reset_for_next_trial(
                    post_x_error_mm=float(post_x_error_mm),
                    post_dist_error_mm=float(post_dist_error_mm),
                    combined_improvement_mm=float(combined_improvement_mm),
                    x_deadband_mm=float(x_deadband_mm),
                    dist_deadband_mm=float(dist_deadband_mm),
                )
            )
            if needs_reset:
                reset_plan: list[str] = []
                inv_drive = _inverse_cmd(drive_cmd)
                inv_turn = _inverse_cmd(turn_cmd)
                if inv_drive is not None:
                    reset_plan.append(str(inv_drive))
                if inv_turn is not None:
                    reset_plan.append(str(inv_turn))
                if reset_plan:
                    log_line(
                        f"[CALIBRATE_X_DIST] {trial_label}: reset needed before next trial; "
                        f"executing inverse sequence {','.join(item.upper() for item in reset_plan)} "
                        f"with staged reset durations."
                    )
                    reset_pre_x = float(post_x_mm)
                    reset_pre_dist = float(post_dist_mm)
                    reset_pre_conf = float(post_pose.get("confidence") or 0.0)
                    reset_pre_source = str(post_pose.get("pose_source") or "unknown")
                    reset_min_sample_time = None
                    for reset_idx, reset_cmd in enumerate(reset_plan, start=1):
                        reset_duration_ms = _scaled_reset_duration_ms(
                            base_duration_ms=int(duration_ms),
                            trial_idx=int(trial_idx),
                            trials_planned=int(len(durations_ms)),
                            reset_act_idx=int(reset_idx),
                        )
                        reset_start_ts = time.time()
                        reset_action = _send_fixed_score_command(
                            robot=robot,
                            world=world,
                            step="CALIBRATE_X_DIST_RESET",
                            cmd=str(reset_cmd),
                            score=int(effective_score),
                            duration_ms=int(reset_duration_ms),
                        )
                        if float(post_act_settle_s) > 0.0:
                            time.sleep(float(post_act_settle_s))
                        reset_min_sample_time = (
                            float(reset_start_ts)
                            + (float(reset_duration_ms) / 1000.0)
                            + float(post_act_settle_s)
                        )
                        reset_pose, reset_meta = _observe_pose(
                            vision=vision,
                            world=world,
                            samples=observe_samples,
                            timeout_s=observe_timeout_s,
                            min_sample_time=reset_min_sample_time,
                        )
                        if reset_pose is None:
                            status = "aborted"
                            abort_reason = f"reset_pose_unavailable_trial_{trial_idx}_act_{reset_idx}"
                            log_line(
                                f"[CALIBRATE_X_DIST] {trial_label}: reset act {int(reset_idx)} failed to recover pose. Aborting."
                            )
                            break
                        reset_post_x = float(reset_pose.get("offset_x") or 0.0)
                        reset_post_dist = float(reset_pose.get("dist") or 0.0)
                        reset_row = ResetEffort(
                            trial=int(trial_idx),
                            reset_act=int(reset_idx),
                            cmd=str(reset_cmd),
                            score_requested=int(effective_score),
                            cmd_sent=str((reset_action or {}).get("cmd_sent") or reset_cmd),
                            duration_ms=int(reset_duration_ms),
                            pre_x_mm=float(reset_pre_x),
                            post_x_mm=float(reset_post_x),
                            pre_dist_mm=float(reset_pre_dist),
                            post_dist_mm=float(reset_post_dist),
                            raw_x_delta_mm=float(reset_post_x - reset_pre_x),
                            raw_dist_delta_mm=float(reset_post_dist - reset_pre_dist),
                            cmd_delta_mm=float(math.hypot(reset_post_x - reset_pre_x, reset_post_dist - reset_pre_dist)),
                            wrong_way=False,
                            pre_brick_dist_mm=float(reset_pre_dist),
                            post_brick_dist_mm=float(reset_post_dist),
                            pre_confidence=float(reset_pre_conf),
                            post_confidence=float(reset_pose.get("confidence") or 0.0),
                            pre_pose_source=str(reset_pre_source),
                            post_pose_source=str(reset_pose.get("pose_source") or "unknown"),
                            post_observation_mode=str((reset_meta or {}).get("mode") or "unknown"),
                        )
                        reset_rows.append(reset_row)
                        reset_pre_x = float(reset_post_x)
                        reset_pre_dist = float(reset_post_dist)
                        reset_pre_conf = float(reset_pose.get("confidence") or 0.0)
                        reset_pre_source = str(reset_pose.get("pose_source") or "unknown")
                        log_line(
                            f"[CALIBRATE_X_DIST] {trial_label}: reset act {int(reset_idx)} "
                            f"cmd={str(reset_cmd).upper()} duration={int(reset_duration_ms)}ms "
                            f"dx={float(reset_row.raw_x_delta_mm):+.2f}mm "
                            f"dd={float(reset_row.raw_dist_delta_mm):+.2f}mm"
                        )
                    if status != "completed":
                        break
                    time.sleep(float(POST_RESET_SETTLE_S))
                    settle_pose, _ = _observe_pose(
                        vision=vision,
                        world=world,
                        samples=observe_samples,
                        timeout_s=observe_timeout_s,
                    )
                    if settle_pose is not None:
                        log_line(
                            f"[CALIBRATE_X_DIST] {trial_label}: post-reset reobserve "
                            f"x={float(settle_pose.get('offset_x') or 0.0):+.2f}mm "
                            f"dist={float(settle_pose.get('dist') or 0.0):.2f}mm (noise check)."
                        )
                    else:
                        log_line(f"[CALIBRATE_X_DIST] {trial_label}: post-reset reobserve gave no pose (proceeding).")
                else:
                    log_line(f"[CALIBRATE_X_DIST] {trial_label}: reset needed but no inverse commands were available.")
            elif should_attempt_reset:
                log_line(f"[CALIBRATE_X_DIST] {trial_label}: reset not needed before next trial.")

            shared_write_results(
                results_path,
                _build_payload(
                    config=config,
                    durations_ms=durations_ms,
                    trials=trial_rows,
                    reset_efforts=reset_rows,
                    status=status,
                    abort_reason=abort_reason,
                ),
            )
    except KeyboardInterrupt:
        status = "interrupted"
        abort_reason = "keyboard_interrupt"
        log_line("[CALIBRATE_X_DIST] Interrupted by user.")
    finally:
        shared_write_results(
            results_path,
            _build_payload(
                config=config,
                durations_ms=durations_ms,
                trials=trial_rows,
                reset_efforts=reset_rows,
                status=status,
                abort_reason=abort_reason,
            ),
        )
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass

    log_line(f"[CALIBRATE_X_DIST] Wrote results to {results_path}")
    if status != "completed":
        detail = f" reason={abort_reason}" if abort_reason else ""
        log_line(f"[CALIBRATE_X_DIST] Finished with status={status}{detail}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    _exit_as_script(main())
