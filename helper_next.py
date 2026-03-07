"""Decision helpers for choosing the robot's next action."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_step_label
from helper_vision_config import demos_dir_for_mode
from telemetry_robot import (
    manual_speed_for_cmd,
    SPEED_SCORE_DEFAULT,
    SPEED_SCORE_MIN,
    SPEED_SCORE_MAX,
    normalize_speed_score,
    ALIGN_MIN_SPEED,
    ALIGN_MAX_SPEED,
    ALIGN_MICRO_SPEED,
    ALIGN_SPEED_SLOW_MM,
    ALIGN_SPEED_FAST_MM,
    ALIGN_MICRO_OFFSET_MM,
    ALIGN_MICRO_ANGLE_DEG,
)

VISIBILITY_LOST_CONFIRM_FRAMES = 3
AUTO_SPEED_SCORE_HARD_MAX = 20
# X-axis tolerance scaling: when far from brick, be much more lenient with x-axis alignment
# This prevents hitting the brick while misaligned. Close distance = strict (1.0x), far = very lenient (6.0x)
ALIGN_BRICK_X_AXIS_TOL_FAR_SCALE = 6.0
ALIGN_STEPS_SHARED_DIST_SCORE_BANDS = (
    (60.0, int(SPEED_SCORE_MIN)),
    (80.0, 5),
)
# Distance ERROR (gap to target) speed scoring for ALIGN_BRICK micro-adjustments.
# Maps distance error MM to speed score (can be fractional for fine control).
# Lines: (error_threshold_mm, score) - if error < threshold, use this score
ALIGN_BRICK_DIST_ERROR_SCORE_BANDS = (
    (2.0, 1.0),      # dist_err < 2.0mm: score 1.0 (~3% power)
    (3.0, 1.5),      # dist_err < 3.0mm: score 1.5 (~5% power)
    (4.0, 2.0),      # dist_err < 4.0mm: score 2.0 (~6% power)
    (5.0, 2.5),      # dist_err < 5.0mm: score 2.5 (~7% power)
    (6.0, 3.0),      # dist_err < 6.0mm: score 3.0 (~10% power)
    (8.0, 4.0),      # dist_err < 8.0mm: score 4.0 (~13% power)
    (1000.0, 5.0),   # dist_err >= 8.0mm: score 5.0 (~17% power) fallback
)
ALIGN_BRICK_X_AXIS_CURVE_ALPHA = 1.67
ALIGN_BRICK_X_AXIS_CURVE_CAP = 24.78
ALIGN_BRICK_X_AXIS_CURVE_MAX_ERR_MM = 22.0
ALIGN_BRICK_X_AXIS_CURVE_BINS_MM = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 22.0)
ALIGN_BRICK_X_AXIS_ONESHOT_MIN_SCORE = 1
ALIGN_BRICK_X_AXIS_ONESHOT_MAX_SCORE = 25
ALIGN_BRICK_Y_AXIS_ERROR_SCORE_BANDS = (
    (3.0, 1),       # within 3mm -> stay near 1% lift for fine vertical centering
    (5.0, 2),
    (8.0, 3),
    (12.0, 4),
    (18.0, 5),
    (26.0, 7),
    (1000.0, 9),    # fallback for large y gaps; keep lift conservative vs x-turns
)
TURN_CURVE_X_ERR_MM_POINTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 22.0, 30.0, 40.0)
ALIGN_STEPS_SHARED_TURN_SCORE_BANDS = (
    (2.0, int(SPEED_SCORE_MIN)),
    (4.0, 2),
    (20.0, 3),
    (50.0, 5),
)
# If turn x-gap is within +/-12mm, always use 1% to reduce ping-pong.
ALIGN_STEPS_SHARED_TURN_FORCE_1_SCORE_GAP_MM = 12.0
# Extra turn-speed cap when distance is already near its success gate.
# This keeps ALIGN_BRICK (step 4) and POSITION_BRICK (step 7) from
# overshooting left/right while they are close to final placement depth.
# Near dist + modest x-axis error should bias toward 1% to prevent ping-pong.
ALIGN_STEPS_SHARED_TURN_NEAR_DIST_MM = 8.0
ALIGN_STEPS_SHARED_TURN_NEAR_DIST_X_GAP_1_SCORE_MM = 7.0
ALIGN_STEPS_SHARED_TURN_NEAR_DIST_CAP_SCORE = 2
ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE = 6

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    # This metric is treated as a signed offset with a target +/- tol band.
    # Use a non-one-sided direction so gate checks use abs(value-target) <= tol.
    "xAxis_offset_abs": "band",
    "yAxis_offset_abs": "band",
    "xAxis_offset": "band",
    "yAxis_offset": "band",
    "x_axis": "band",
    "y_axis": "band",
    "dist": "low",
    "visible": "high",
    "confidence": "high",
}

DEFAULT_DEMOS_DIR = demos_dir_for_mode()


def _step_name(step):
    return normalize_step_label(step)


ALIGN_POLICY_DEFAULTS = {
    "disabled_metrics": [],
    "metric_direction_overrides": {},
    "hold_when_not_visible": False,
    "x_axis_tol_far_scale": 1.0,
    "focus_prioritization_enabled": True,
    "focus_x_first_dist_gap_gt_mm": 1000000000.0,
    "focus_x_first_x_gap_gt_mm": 1000000000.0,
    "focus_dist_first_dist_gap_gt_mm": 150.0,
    "focus_dist_first_x_gap_lt_mm": 1000000000.0,
    "focus_dist_sticky_enabled": True,
    "focus_dist_sticky_release_mm": 100.0,
    # Optional hard override for steps that should intentionally favor distance
    # correction whenever distance is outside gate.
    "dist_priority_cheat_enabled": False,
    "dist_priority_cheat_min_ratio": 0.0,
    "dist_priority_cheat_min_outside_mm": 0.0,
    "y_axis_close_bottom_bias_enabled": True,
    "y_axis_close_bottom_dist_mm_max": 100.0,
    "y_axis_close_bottom_min_mm": 1.0,
    "dist_score_mode": "dist_value_bands",
    "turn_score_mode": "shared_turn_bands",
    "turn_fallback_score": ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE,
    "slow_fast_mm_scale": 0.25,
    "auto_speed_hard_cap_exempt": False,
    "calibration_profile_fallback_step": None,
    # Optional chunked gap rotation mode for fragile alignment steps.
    # When enabled, planner closes one gap by a bounded chunk and rotates.
    "gap_rotation_enabled": False,
    "gap_rotation_chunk_min_mm": 3.0,
    "gap_rotation_chunk_max_mm": 6.0,
    "gap_rotation_y_priority_penalty": 1.25,
    "gap_rotation_y_hold_last_mm": 3.0,
    "gap_rotation_force_recovery_switch": True,
    "gap_rotation_tech_debt_logging": False,
    # Soft anti-fixation guard for gap micro-adjustment loops. If the same
    # gap type is chosen for too many consecutive cycles and another gap is
    # still outside gate, force a switch to the alternate gap type.
    "gap_focus_max_cycles_before_switch": 10,
}


def _step_cfg(process_rules, step):
    obj_name = _step_name(step)
    cfg = {}
    if isinstance(process_rules, dict):
        raw = process_rules.get(obj_name)
        if isinstance(raw, dict):
            cfg = raw
    return obj_name, cfg


def _align_policy_for_step(process_rules, step):
    _obj_name, cfg = _step_cfg(process_rules, step)
    out = dict(ALIGN_POLICY_DEFAULTS)
    raw = cfg.get("align_policy") if isinstance(cfg, dict) else {}
    if isinstance(raw, dict):
        out.update(raw)
    if not isinstance(out.get("disabled_metrics"), (list, tuple, set)):
        out["disabled_metrics"] = []
    if not isinstance(out.get("metric_direction_overrides"), dict):
        out["metric_direction_overrides"] = {}
    return out


def metric_direction_for_step(metric, step, process_rules=None):
    direction = METRIC_DIRECTIONS.get(metric)
    if isinstance(process_rules, dict):
        policy = _align_policy_for_step(process_rules, step)
        metric_key = str(metric)
        disabled = {str(item).strip() for item in policy.get("disabled_metrics") or []}
        if metric_key in disabled:
            return None
        override = (policy.get("metric_direction_overrides") or {}).get(metric_key)
        if override is None:
            return direction
        override_key = str(override).strip().lower()
        if override_key in {"low", "high", "band"}:
            return override_key
        if override_key in {"none", "off", "disabled"}:
            return None
    return direction


def _target_tol_ok(value, stats, direction):
    target = stats.get("target") if isinstance(stats, dict) else None
    tol = stats.get("tol") if isinstance(stats, dict) else None
    if target is None or tol is None:
        return None
    if direction == "high":
        return value >= (target - tol)
    if direction == "low":
        return value <= (target + tol)
    return abs(value - target) <= tol


def _target_tol_outside_mm(value, stats, direction):
    target = stats.get("target") if isinstance(stats, dict) else None
    tol = stats.get("tol") if isinstance(stats, dict) else None
    if target is None or tol is None:
        return None
    try:
        value_num = float(value)
        target_num = float(target)
        tol_num = abs(float(tol))
    except (TypeError, ValueError):
        return None
    dir_key = str(direction or "").strip().lower()
    if dir_key == "high":
        return max(0.0, (target_num - tol_num) - value_num)
    if dir_key == "low":
        return max(0.0, value_num - (target_num + tol_num))
    return max(0.0, abs(value_num - target_num) - tol_num)


def _gate_outside_mm(value, stats, direction):
    try:
        value_num = float(value)
    except (TypeError, ValueError):
        return None
    outside_target_tol = _target_tol_outside_mm(value_num, stats, direction)
    if outside_target_tol is not None:
        return float(outside_target_tol)
    if not isinstance(stats, dict):
        return 0.0
    min_val = stats.get("min")
    max_val = stats.get("max")
    try:
        min_num = float(min_val) if min_val is not None else None
    except (TypeError, ValueError):
        min_num = None
    try:
        max_num = float(max_val) if max_val is not None else None
    except (TypeError, ValueError):
        max_num = None
    if min_num is not None and value_num < min_num:
        return float(min_num - value_num)
    if max_num is not None and value_num > max_num:
        return float(value_num - max_num)
    return 0.0


def _score_from_mm(mm_off, slow_mm, fast_mm):
    if mm_off is None:
        return SPEED_SCORE_DEFAULT
    if mm_off <= slow_mm:
        return SPEED_SCORE_MIN
    if mm_off >= fast_mm:
        return SPEED_SCORE_MAX
    return SPEED_SCORE_DEFAULT


def _banded_gap_speed_score(
    value,
    bands,
    fallback_score: int,
    *,
    use_abs=False,
    inclusive_upper=False,
):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(fallback_score)
    if use_abs:
        numeric = abs(numeric)
    for upper_bound, score in bands or ():
        try:
            bound = float(upper_bound)
        except (TypeError, ValueError):
            continue
        if numeric < bound or (inclusive_upper and numeric == bound):
            return int(score)
    return int(fallback_score)


def align_steps_dist_speed_score(dist_mm: float, fallback_score: int) -> int:
    return _banded_gap_speed_score(
        dist_mm,
        ALIGN_STEPS_SHARED_DIST_SCORE_BANDS,
        fallback_score,
    )


def align_brick_dist_error_speed_score(dist_error_mm: float) -> float:
    """
    Calculate granular speed score for ALIGN_BRICK distance micro-adjustments.
    
    Maps distance error (gap between current distance and target) to appropriate
    speed score, with fine-grained bands that support fractional scores (e.g., 1.5).
    
    This is the single source of truth for distance alignment speed logic.
    Lines defined in ALIGN_BRICK_DIST_ERROR_SCORE_BANDS in world model.
    
    Args:
        dist_error_mm: Absolute distance error in mm (should be >= 0)
    
    Returns:
        Speed score as float (e.g., 1.0, 1.5, 2.0, 2.5, 3.0, etc.)
    """
    try:
        error_mm = abs(float(dist_error_mm))
    except (TypeError, ValueError):
        return 1.0  # Default to minimal score on error
    
    # Use banded lookup: find first band where error_mm < upper_bound
    for upper_bound, score in ALIGN_BRICK_DIST_ERROR_SCORE_BANDS:
        try:
            bound = float(upper_bound)
            if error_mm < bound:
                return float(score)
        except (TypeError, ValueError):
            continue
    
    # Fallback to last score if exceeds all bands (shouldn't happen with large fallback)
    return float(ALIGN_BRICK_DIST_ERROR_SCORE_BANDS[-1][1]) if ALIGN_BRICK_DIST_ERROR_SCORE_BANDS else 1.0


def align_brick_x_axis_one_shot_score(x_err_mm: float) -> int:
    """
    Compute ALIGN_BRICK one-shot turn score from x-axis error magnitude.

    This matches calibrate-align behavior and is shared by both calibrate and
    auto step-4 decision paths.
    """
    try:
        gap_mm = abs(float(x_err_mm))
    except (TypeError, ValueError):
        gap_mm = 0.0

    max_err = max(1e-6, float(ALIGN_BRICK_X_AXIS_CURVE_MAX_ERR_MM))
    ratio = max(0.0, min(1.0, float(gap_mm) / float(max_err)))
    curved = float(ratio) ** float(ALIGN_BRICK_X_AXIS_CURVE_ALPHA)
    raw = int(round(1.0 + (float(ALIGN_BRICK_X_AXIS_CURVE_CAP) - 1.0) * curved))
    return int(
        max(
            int(ALIGN_BRICK_X_AXIS_ONESHOT_MIN_SCORE),
            min(int(raw), int(ALIGN_BRICK_X_AXIS_ONESHOT_MAX_SCORE)),
        )
    )


def align_brick_x_axis_decision_line() -> str:
    return (
        "score = clamp(round(1 + (cap - 1) * "
        "(min(|x_err_mm|, max_err_mm)/max_err_mm)^alpha), min_score, max_score) "
        f"where alpha={float(ALIGN_BRICK_X_AXIS_CURVE_ALPHA):.2f}, "
        f"cap={float(ALIGN_BRICK_X_AXIS_CURVE_CAP):.2f}, "
        f"max_err_mm={float(ALIGN_BRICK_X_AXIS_CURVE_MAX_ERR_MM):.1f}, "
        f"min_score={int(ALIGN_BRICK_X_AXIS_ONESHOT_MIN_SCORE)}, "
        f"max_score={int(ALIGN_BRICK_X_AXIS_ONESHOT_MAX_SCORE)}"
    )


def align_brick_y_axis_one_shot_score(y_err_mm: float) -> int:
    """
    Compute ALIGN_BRICK lift score from y-axis error magnitude.

    Uses conservative bands because vertical lift corrections can lose marker
    visibility more easily than turn corrections. The first band intentionally
    stays at 1% for |y_err| < 3mm.
    """
    try:
        gap_mm = abs(float(y_err_mm))
    except (TypeError, ValueError):
        gap_mm = 0.0

    for upper_bound, score in ALIGN_BRICK_Y_AXIS_ERROR_SCORE_BANDS:
        try:
            if float(gap_mm) < float(upper_bound):
                return int(
                    max(
                        int(SPEED_SCORE_MIN),
                        min(int(round(float(score))), int(SPEED_SCORE_MAX)),
                    )
                )
        except (TypeError, ValueError):
            continue
    return int(SPEED_SCORE_MIN)


def align_turn_speed_score_for_step(step, x_err_mm: float, *, dist_gate_error_mm=None, process_rules=None) -> int:
    """Return l/r speed score using world-model policy, not hard-coded step names."""
    policy = _align_policy_for_step(process_rules, step)
    mode = str(policy.get("turn_score_mode") or "shared_turn_bands").strip().lower()
    try:
        fallback_score = int(round(float(policy.get("turn_fallback_score", ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE))))
    except (TypeError, ValueError):
        fallback_score = int(ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE)
    if mode in {"x_axis_one_shot_curve", "x_axis_curve", "one_shot"}:
        return int(align_brick_x_axis_one_shot_score(x_err_mm))
    return int(
        align_steps_turn_speed_score(
            x_err_mm,
            int(fallback_score),
            dist_gate_error_mm=dist_gate_error_mm,
        )
    )


def _turn_score_cap_for_dist_gate_error(dist_gate_error_mm, x_axis_gap_mm):
    if dist_gate_error_mm is None:
        return None
    try:
        dist_err = abs(float(dist_gate_error_mm))
    except (TypeError, ValueError):
        return None
    if dist_err > float(ALIGN_STEPS_SHARED_TURN_NEAR_DIST_MM):
        return None
    try:
        x_gap = abs(float(x_axis_gap_mm))
    except (TypeError, ValueError):
        x_gap = None
    if x_gap is not None and x_gap <= float(ALIGN_STEPS_SHARED_TURN_NEAR_DIST_X_GAP_1_SCORE_MM):
        return int(SPEED_SCORE_MIN)
    return int(ALIGN_STEPS_SHARED_TURN_NEAR_DIST_CAP_SCORE)


def align_steps_turn_speed_score(
    x_axis_gap_mm: float,
    fallback_score: int,
    *,
    dist_gate_error_mm=None,
) -> int:
    try:
        x_gap_abs = abs(float(x_axis_gap_mm))
    except (TypeError, ValueError):
        x_gap_abs = None
    if x_gap_abs is not None and x_gap_abs < float(ALIGN_STEPS_SHARED_TURN_FORCE_1_SCORE_GAP_MM):
        return int(SPEED_SCORE_MIN)

    score = _banded_gap_speed_score(
        x_axis_gap_mm,
        ALIGN_STEPS_SHARED_TURN_SCORE_BANDS,
        fallback_score,
        use_abs=True,
        inclusive_upper=True,
    )
    cap = _turn_score_cap_for_dist_gate_error(dist_gate_error_mm, x_axis_gap_mm)
    if cap is not None:
        score = min(int(score), int(cap))
    return int(score)


def success_gates_visible_only(process_rules, step):
    obj_name = _step_name(step)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    if not success_metrics:
        return False
    keys = {key for key in success_metrics.keys() if key is not None}
    return keys in ({"visible"}, {"visible", "confidence"})


def _smoothstep01(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - (2.0 * x))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def align_brick_micro_forward_profile(process_rules, step: str):
    defaults = {
        "very_close_mm": 110.0,
        "close_mm": 150.0,
        "somewhat_close_mm": 170.0,
        "far_mm": 200.0,
    }
    if not isinstance(process_rules, dict):
        return defaults
    cfg = process_rules.get(step)
    if not isinstance(cfg, dict):
        return defaults
    profile = cfg.get("micro_forward_profile")
    if not isinstance(profile, dict):
        return defaults

    resolved = {}
    for key, fallback in defaults.items():
        raw = profile.get(key, fallback)
        try:
            resolved[key] = float(raw)
        except (TypeError, ValueError):
            resolved[key] = float(fallback)

    ordered = sorted(
        (
            resolved["very_close_mm"],
            resolved["close_mm"],
            resolved["somewhat_close_mm"],
            resolved["far_mm"],
        )
    )
    return {
        "very_close_mm": ordered[0],
        "close_mm": ordered[1],
        "somewhat_close_mm": ordered[2],
        "far_mm": ordered[3],
    }


def align_brick_x_axis_tol_scale(
    dist_mm: float,
    process_rules,
    step: str,
    *,
    max_scale: float = ALIGN_BRICK_X_AXIS_TOL_FAR_SCALE,
) -> float:
    """
    Calculate x-axis tolerance scaling based on distance from brick.
    
    The further from the brick, the more lenient we are with x-axis alignment.
    This prevents hitting the brick while misaligned and allows coarser x-axis
    positioning when far, with precise alignment only required when close.
    
    Returns:
        1.0 when at close distance or closer (strict alignment required)
        Up to 6.0 when at far distance or beyond (very lenient alignment)
    """
    profile = align_brick_micro_forward_profile(process_rules, step)
    close = float(profile["close_mm"])  # ~150mm - where strict alignment starts
    far = float(profile["far_mm"])      # ~200mm - where max leniency applies
    try:
        max_scale = max(1.0, float(max_scale))
    except (TypeError, ValueError):
        max_scale = float(ALIGN_BRICK_X_AXIS_TOL_FAR_SCALE)

    try:
        dist_val = float(dist_mm)
    except (TypeError, ValueError):
        return 1.0

    # At close distance or closer: strict 1.0x tolerance (precise alignment required)
    if dist_val <= close:
        return 1.0
    # If far <= close (invalid config), return max leniency
    if far <= close:
        return max_scale

    # Between close and far: smoothly scale from 1.0x to 6.0x
    # Use smoothstep for gradual transition
    t = (dist_val - close) / max(1e-6, far - close)
    return float(_lerp(1.0, max_scale, _smoothstep01(t)))


def compute_alignment_analytics(world, process_rules, learned_rules, step, duration_s=0.05):
    obj_name = _step_name(step)
    align_policy = _align_policy_for_step(process_rules, obj_name)
    raw_success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    success_metrics = dict(raw_success_metrics) if isinstance(raw_success_metrics, dict) else {}
    if "xAxis_offset_abs" not in success_metrics:
        if "x_axis" in success_metrics:
            success_metrics["xAxis_offset_abs"] = success_metrics.get("x_axis")
        elif "xAxis_offset" in success_metrics:
            success_metrics["xAxis_offset_abs"] = success_metrics.get("xAxis_offset")
    if "yAxis_offset_abs" not in success_metrics:
        if "y_axis" in success_metrics:
            success_metrics["yAxis_offset_abs"] = success_metrics.get("y_axis")
        elif "yAxis_offset" in success_metrics:
            success_metrics["yAxis_offset_abs"] = success_metrics.get("yAxis_offset")
    step_cfg = (process_rules or {}).get(obj_name, {}) or {}
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    visible_for_cmd = visible
    lost_frames = getattr(world, "_visibility_lost_frames", 0)
    last_seen_time = getattr(world, "last_visible_time", None)
    x_axis = brick.get("x_axis")
    if x_axis is None:
        x_axis = 0.0
    x_axis = float(x_axis)
    angle = float(brick.get("angle", 0.0) or 0.0)
    dist = float(brick.get("dist", 0.0) or 0.0)

    hold_when_not_visible = bool(align_policy.get("hold_when_not_visible"))
    if hold_when_not_visible and not visible:
        return {
            "progress": None,
            "worst_metric": None,
            "cmd": None,
            "speed": 0.0,
            "duration_s": duration_s,
            "x_axis": x_axis,
            "angle": angle,
            "dist": dist,
            "offsets": {},
        }
    if not visible and last_seen_time is not None and lost_frames < VISIBILITY_LOST_CONFIRM_FRAMES:
        visible_for_cmd = True
        last_x = getattr(world, "last_seen_x_axis", None)
        last_angle = getattr(world, "last_seen_angle", None)
        last_dist = getattr(world, "last_seen_dist", None)
        if last_x is not None:
            x_axis = float(last_x)
        if last_angle is not None:
            angle = float(last_angle)
        if last_dist is not None:
            dist = float(last_dist)

    metrics = {
        "xAxis_offset_abs": x_axis,
        "angle_abs": abs(angle),
        "dist": dist,
    }
    active_metrics = list(metrics.keys())
    if isinstance(success_metrics, dict) and success_metrics:
        gated_metrics = [metric for metric in active_metrics if metric in success_metrics]
        if gated_metrics:
            active_metrics = gated_metrics
    x_axis_active = "xAxis_offset_abs" in active_metrics
    signed_values = {
        "xAxis_offset_abs": x_axis,
        "angle_abs": angle,
        "dist": dist,
    }
    progress_values = []
    offsets = {}
    ratios = {}
    mm_errors = {}
    x_axis_turn_error_mm = None
    x_axis_tol_scale = 1.0
    try:
        x_axis_tol_far_scale = max(1.0, float(align_policy.get("x_axis_tol_far_scale", 1.0)))
    except (TypeError, ValueError):
        x_axis_tol_far_scale = 1.0
    if x_axis_tol_far_scale > 1.0:
        x_axis_tol_scale = align_brick_x_axis_tol_scale(
            dist,
            process_rules,
            obj_name,
            max_scale=float(x_axis_tol_far_scale),
        )
        if x_axis_tol_scale < 1.0:
            x_axis_tol_scale = 1.0

    def fallback_stats(metric):
        if metric == "xAxis_offset_abs":
            return {"max": float(getattr(world, "align_tol_offset", 12.0))}
        if metric == "angle_abs":
            return {"max": float(getattr(world, "align_tol_angle", 5.0))}
        if metric == "dist":
            return {
                "min": float(getattr(world, "align_tol_dist_min", 30.0)),
                "max": float(getattr(world, "align_tol_dist_max", 500.0)),
            }
        return {}

    def _effective_stats(metric, stats):
        if metric != "xAxis_offset_abs":
            return stats
        if not isinstance(stats, dict):
            return stats
        if x_axis_tol_scale <= 1.0:
            return stats
        scaled = dict(stats)
        tol = scaled.get("tol")
        if tol is not None:
            try:
                scaled["tol"] = float(tol) * x_axis_tol_scale
            except (TypeError, ValueError):
                pass
        max_val = scaled.get("max")
        if max_val is not None:
            try:
                scaled["max"] = float(max_val) * x_axis_tol_scale
            except (TypeError, ValueError):
                pass
        return scaled

    def metric_within_gate(stats, value):
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            return abs(value - target) <= tol
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    for metric in active_metrics:
        value = metrics.get(metric)
        stats = success_metrics.get(metric) or fallback_stats(metric)
        stats = _effective_stats(metric, stats)
        direction = metric_direction_for_step(metric, obj_name, process_rules=process_rules)
        if direction is None:
            continue
        target = stats.get("target")
        tol = stats.get("tol")
        min_val = stats.get("min")
        max_val = stats.get("max")
        error = 0.0
        signed_error = 0.0
        progress = None

        if target is not None and tol is not None:
            signed_error = value - target
            error = max(0.0, abs(signed_error) - tol)
            if tol > 0:
                distance = abs(signed_error)
                if distance <= tol:
                    progress = 1.0
                else:
                    progress = max(0.0, 1.0 - (distance - tol) / tol)
                ratios[metric] = error / max(float(tol), 1e-3)
            else:
                progress = 1.0 if signed_error == 0 else 0.0
                ratios[metric] = 1.0 if signed_error != 0 else 0.0
        else:
            if min_val is not None and value < min_val:
                signed_error = value - min_val
                error = min_val - value
            if max_val is not None and value > max_val:
                signed_error = value - max_val
                error = max(value - max_val, error)
            if min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    progress = 1.0
                else:
                    span = max(1e-3, max_val - min_val)
                    if value < min_val:
                        progress = max(0.0, 1.0 - (min_val - value) / span)
                    else:
                        progress = max(0.0, 1.0 - (value - max_val) / span)
            elif min_val is not None:
                progress = 1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3))
            elif max_val is not None:
                progress = 1.0 if value <= max_val else max(
                    0.0, 1.0 - (value - max_val) / max(max_val, 1e-3)
                )
            scale = max(abs(max_val or min_val or 1.0), 1e-3)
            ratios[metric] = error / scale

        if progress is not None:
            progress_values.append(progress)
        if metric in ("xAxis_offset_abs", "dist"):
            mm_errors[metric] = max(0.0, float(error))
        if metric == "xAxis_offset_abs":
            offsets["x_axis"] = signed_error
        elif metric == "angle_abs":
            offsets["angle"] = signed_error
        elif metric == "dist":
            offsets["dist"] = signed_error

    progress = sum(progress_values) / len(progress_values) if progress_values else None
    if x_axis_active:
        x_axis_stats = success_metrics.get("xAxis_offset_abs") or fallback_stats("xAxis_offset_abs")
        x_axis_stats = _effective_stats("xAxis_offset_abs", x_axis_stats)
        x_axis_ok = metric_within_gate(x_axis_stats, x_axis)
    else:
        x_axis_ok = True
    dist_gap_mm = None
    force_dist_focus = False
    force_x_axis_focus = False
    focus_prioritization_enabled = bool(align_policy.get("focus_prioritization_enabled"))
    if focus_prioritization_enabled:
        try:
            focus_x_first_dist_gap_gt_mm = float(align_policy.get("focus_x_first_dist_gap_gt_mm", 50.0))
        except (TypeError, ValueError):
            focus_x_first_dist_gap_gt_mm = 50.0
        try:
            focus_x_first_x_gap_gt_mm = float(align_policy.get("focus_x_first_x_gap_gt_mm", 2.0))
        except (TypeError, ValueError):
            focus_x_first_x_gap_gt_mm = 2.0
        try:
            focus_dist_first_dist_gap_gt_mm = float(align_policy.get("focus_dist_first_dist_gap_gt_mm", 150.0))
        except (TypeError, ValueError):
            focus_dist_first_dist_gap_gt_mm = 150.0
        try:
            focus_dist_first_x_gap_lt_mm = float(align_policy.get("focus_dist_first_x_gap_lt_mm", 5.0))
        except (TypeError, ValueError):
            focus_dist_first_x_gap_lt_mm = 5.0
        focus_dist_sticky_enabled = bool(align_policy.get("focus_dist_sticky_enabled", True))
        try:
            focus_dist_sticky_release_mm = float(align_policy.get("focus_dist_sticky_release_mm", 100.0))
        except (TypeError, ValueError):
            focus_dist_sticky_release_mm = 100.0
        try:
            dist_gap_mm = abs(float(offsets.get("dist", 0.0) or 0.0))
        except (TypeError, ValueError):
            dist_gap_mm = None
        if dist_gap_mm is not None:
            x_axis_gap_mm = mm_errors.get("xAxis_offset_abs", 0.0)
            sticky_focus_dist = bool(getattr(world, "_align_focus_dist", False))
            if (
                focus_dist_sticky_enabled
                and sticky_focus_dist
                and dist_gap_mm >= max(0.0, float(focus_dist_sticky_release_mm))
            ):
                force_dist_focus = True
                force_x_axis_focus = False
            elif dist_gap_mm > focus_x_first_dist_gap_gt_mm and x_axis_gap_mm > focus_x_first_x_gap_gt_mm:
                force_x_axis_focus = True
            elif dist_gap_mm > focus_dist_first_dist_gap_gt_mm and x_axis_gap_mm < focus_dist_first_x_gap_lt_mm:
                force_dist_focus = True
            else:
                force_dist_focus = False
                force_x_axis_focus = False
        focus_dist = force_dist_focus
        try:
            world._align_focus_dist = focus_dist
        except Exception:
            pass
    worst_metric = max(ratios, key=lambda m: ratios[m], default=None)
    worst_ratio = ratios.get(worst_metric, 0.0) if worst_metric else 0.0

    if success_metrics:
        all_success = True
        for metric, stats in success_metrics.items():
            if metric == "visible":
                continue
            if not isinstance(stats, dict):
                continue
            direction = metric_direction_for_step(metric, obj_name, process_rules=process_rules)
            if metric == "angle_abs":
                value = abs(angle)
            elif metric == "xAxis_offset_abs":
                value = x_axis
            elif metric == "dist":
                value = dist
            else:
                continue
            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                ok = abs(value - target) <= tol
            else:
                ok = _target_tol_ok(value, stats, direction)
            if ok is False:
                all_success = False
                break
        if all_success:
            return {
                "progress": 1.0 if progress is None else progress,
                "worst_metric": None,
                "cmd": None,
                "speed": 0.0,
                "duration_s": duration_s,
                "x_axis": x_axis,
                "angle": angle,
                "dist": dist,
                "offsets": offsets,
            }

    if worst_metric is None or worst_ratio <= 0.0:
        fallback_metric = None
        fallback_value = 0.0
        for metric, value in offsets.items():
            if abs(value) > abs(fallback_value):
                fallback_metric = metric
                fallback_value = value
        if fallback_metric and abs(fallback_value) > 0.0:
            worst_metric = {
                "x_axis": "xAxis_offset_abs",
                "angle": "angle_abs",
                "dist": "dist",
            }.get(fallback_metric)
            worst_ratio = 1.0
        else:
            if x_axis_active and not x_axis_ok:
                worst_metric = "xAxis_offset_abs"
                worst_ratio = max(ratios.get("xAxis_offset_abs", 1.0), 1.0)
            else:
                return {
                    "progress": progress,
                    "worst_metric": None,
                    "cmd": None,
                    "speed": 0.0,
                    "duration_s": duration_s,
                    "x_axis": x_axis,
                    "angle": angle,
                    "dist": dist,
                    "offsets": offsets,
                }

    # Apply focus overrides based on distance-dependent prioritization
    if force_x_axis_focus:
        worst_metric = "xAxis_offset_abs"
        worst_ratio = max(worst_ratio, ratios.get("xAxis_offset_abs", 1.0), 1.0)
    elif force_dist_focus:
        worst_metric = "dist"
        worst_ratio = max(worst_ratio, ratios.get("dist", 1.0), 1.0)

    profile = {}
    if isinstance(learned_rules, dict):
        step_profile = learned_rules.get(obj_name)
        if isinstance(step_profile, dict) and isinstance(step_profile.get("calibration_profile"), dict):
            profile = dict(step_profile.get("calibration_profile") or {})
        else:
            fallback_step = align_policy.get("calibration_profile_fallback_step")
            fallback_key = _step_name(fallback_step) if fallback_step else None
            if fallback_key:
                fallback_profile = learned_rules.get(fallback_key)
                if isinstance(fallback_profile, dict) and isinstance(fallback_profile.get("calibration_profile"), dict):
                    profile = dict(fallback_profile.get("calibration_profile") or {})

    try:
        turn_speed_scale = float(profile.get("turn_speed_scale", 1.0))
    except (TypeError, ValueError):
        turn_speed_scale = 1.0
    try:
        dist_speed_scale = float(profile.get("dist_speed_scale", 1.0))
    except (TypeError, ValueError):
        dist_speed_scale = 1.0
    turn_speed_scale = max(0.5, min(1.5, float(turn_speed_scale)))
    dist_speed_scale = max(0.5, min(1.5, float(dist_speed_scale)))

    profile_max_speed_score = None
    try:
        if profile.get("max_speed_score") is not None:
            profile_max_speed_score = normalize_speed_score(profile.get("max_speed_score"))
    except (TypeError, ValueError):
        profile_max_speed_score = None

    min_speed = ALIGN_MIN_SPEED
    max_speed = ALIGN_MAX_SPEED
    micro_speed = ALIGN_MICRO_SPEED
    micro_offset_mm = ALIGN_MICRO_OFFSET_MM
    micro_angle_deg = ALIGN_MICRO_ANGLE_DEG

    speed_factor = max(0.0, min(1.0, (worst_ratio - 1.0) / 2.0))
    speed = min_speed + (max_speed - min_speed) * speed_factor

    cmd = None
    if worst_metric == "dist":
        dist_stats = success_metrics.get("dist") or fallback_stats("dist")
        dist_min = dist_stats.get("min")
        dist_max = dist_stats.get("max")
        target = dist_stats.get("target")
        tol = dist_stats.get("tol")
        if target is not None and tol is not None:
            if dist > target + tol:
                cmd = "f"
            elif dist < target - tol:
                cmd = "b"
        if cmd is None:
            if dist_max is not None and dist > dist_max:
                cmd = "f"
            elif dist_min is not None and dist < dist_min:
                cmd = "b"

    elif worst_metric == "xAxis_offset_abs":
        signed = signed_values.get("xAxis_offset_abs", 0.0)
        stats = success_metrics.get("xAxis_offset_abs") or fallback_stats("xAxis_offset_abs")
        target = stats.get("target")
        tol = stats.get("tol")
        signed_error = signed - target if target is not None and tol is not None else signed
        x_axis_turn_error_mm = abs(float(signed_error))
        if signed_error > 0.0:
            cmd = "r"
        elif signed_error < 0.0:
            cmd = "l"
        else:
            cmd = None
        if abs(signed_error) < micro_offset_mm:
            speed = min(speed, micro_speed)
        worst_metric = "xAxis_offset_abs"

    elif worst_metric in ("angle", "angle_abs"):
        signed = signed_values.get("angle", 0.0)
        stats = success_metrics.get("angle_abs") or fallback_stats("angle_abs")
        target = stats.get("target")
        tol = stats.get("tol")
        signed_error = signed - target if target is not None and tol is not None else signed
        if signed_error > 0.0:
            cmd = "l"
        elif signed_error < 0.0:
            cmd = "r"
        else:
            cmd = None
        if abs(signed) < micro_angle_deg:
            speed = min(speed, micro_speed)
        worst_metric = "angle_abs"

    visible_only = success_gates_visible_only(process_rules, obj_name)
    if visible_only:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)
    else:
        mm_off = None
        if cmd in ("f", "b"):
            mm_off = mm_errors.get("dist")
        if (mm_off is None or mm_off <= 0.0) and worst_metric == "angle_abs":
            mm_off = None
        try:
            slow_fast_mm_scale = float(align_policy.get("slow_fast_mm_scale", 0.25))
        except (TypeError, ValueError):
            slow_fast_mm_scale = 0.25
        slow_fast_mm_scale = max(0.01, float(slow_fast_mm_scale))
        slow_mm = float(ALIGN_SPEED_SLOW_MM) * float(slow_fast_mm_scale)
        fast_mm = float(ALIGN_SPEED_FAST_MM) * float(slow_fast_mm_scale)
        dist_score_mode = str(align_policy.get("dist_score_mode") or "simple_mm_band").strip().lower()
        if cmd in ("f", "b") and dist_score_mode in {"dist_error_bands", "dist_error"}:
            dist_error_mm = None
            if isinstance(offsets, dict):
                dist_signed_err = _coerce_float(offsets.get("dist"), None)
                if dist_signed_err is not None:
                    dist_error_mm = abs(float(dist_signed_err))
            if dist_error_mm is None:
                dist_error_mm = _coerce_float(mm_errors.get("dist"), 0.0) or 0.0
            dist_score_float = align_brick_dist_error_speed_score(float(dist_error_mm))
            speed_score = int(round(float(dist_score_float)))
        elif cmd in ("f", "b") and dist_score_mode in {"dist_value_bands", "dist_profile"}:
            base_score = _score_from_mm(mm_off, slow_mm, fast_mm)
            speed_score = align_steps_dist_speed_score(dist, base_score)
        elif cmd in ("l", "r") and x_axis_turn_error_mm is not None:
            dist_gate_error_mm = mm_errors.get("dist")
            speed_score = align_turn_speed_score_for_step(
                obj_name,
                x_axis_turn_error_mm,
                dist_gate_error_mm=dist_gate_error_mm,
                process_rules=process_rules,
            )
        else:
            speed_score = _score_from_mm(mm_off, slow_mm, fast_mm)
        speed_score = normalize_speed_score(speed_score)

    if cmd in ("l", "r"):
        speed_score = normalize_speed_score(float(speed_score) * float(turn_speed_scale))
    elif cmd in ("f", "b"):
        speed_score = normalize_speed_score(float(speed_score) * float(dist_speed_scale))

    if not visible_for_cmd:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)
    max_speed_score = None
    if isinstance(step_cfg, dict) and step_cfg.get("max_speed_score") is not None:
        try:
            max_speed_score = normalize_speed_score(step_cfg.get("max_speed_score"))
        except (TypeError, ValueError):
            max_speed_score = None
    if max_speed_score is not None:
        speed_score = min(int(speed_score), int(max_speed_score))
    if profile_max_speed_score is not None:
        speed_score = min(int(speed_score), int(profile_max_speed_score))
    if not bool(align_policy.get("auto_speed_hard_cap_exempt")):
        speed_score = min(int(speed_score), int(AUTO_SPEED_SCORE_HARD_MAX))

    if cmd:
        speed = manual_speed_for_cmd(cmd, speed_score)
    else:
        speed = 0.0

    return {
        "progress": progress,
        "worst_metric": worst_metric,
        "cmd": cmd,
        "speed": speed,
        "speed_score": speed_score,
        "duration_s": duration_s,
        "x_axis": x_axis,
        "angle": angle,
        "dist": dist,
        "offsets": offsets,
    }


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Gap-closing policy: vertical (`y_err`) corrections stay in rotation, but should
# only preempt x/dist corrections when the normalized y gap is substantially larger.
# This reduces oscillatory or low-value mast adjustments during x/dist convergence.
GAP_ALIGN_Y_AXIS_PRIORITY_PENALTY = 2.0
# Treat tiny positive ratios as in-gate noise so an effectively "good" metric
# can never become the selected "worst" correction.
GAP_ALIGN_OUTSIDE_GATE_RATIO_EPS = 1e-6
# `y_err` should only be worked once the other gap(s) are nearly solved.
# Ratio is normalized "outside gate" error: 0.10 means 10% beyond the gate tolerance.
GAP_ALIGN_Y_AXIS_OTHER_GAPS_NEAR_RATIO_MAX = 0.10
# If the marker is very near the top/bottom of frame (large |y offset|), force
# y-axis correction immediately to re-center visibility before fine x/dist work.
GAP_ALIGN_Y_AXIS_EDGE_FORCE_ABS_MM_MIN = 8.0
GAP_ALIGN_Y_AXIS_EDGE_FORCE_TOL_MULT = 4.0


def _success_gates_for_step(process_rules, step):
    obj_name = _step_name(step)
    step_cfg = (process_rules or {}).get(obj_name, {}) if isinstance(process_rules, dict) else {}
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    if not isinstance(success_gates, dict):
        success_gates = {}
    return obj_name, success_gates


def step_uses_gap_alignment_planner(process_rules, step):
    """
    Return True for ALIGN-style steps that should use the brick gap micro-planner.

    The micro-planner is appropriate when the step has x/y/dist gate metrics.
    Angle-only / visible-only align steps should continue using the generic
    analytics planner.
    """
    _obj_name, success_gates = _success_gates_for_step(process_rules, step)
    if not isinstance(success_gates, dict) or not success_gates:
        return False
    gate_keys = {str(key) for key in success_gates.keys() if key is not None}
    return bool(
        gate_keys
        & {
            "xAxis_offset_abs",
            "yAxis_offset_abs",
            "xAxis_offset",
            "yAxis_offset",
            "x_axis",
            "y_axis",
            "dist",
        }
    )


def compute_alignment_decision(
    *,
    world=None,
    step="ALIGN_BRICK",
    process_rules=None,
    learned_rules=None,
    duration_s=0.05,
    visible=None,
    x_axis=None,
    angle=None,
    dist=None,
):
    """
    Single source of truth for deciding next ALIGN-style action.

    Use with either a full world model (`world=...`) or raw measurements
    (`visible`, `x_axis`, `angle`, `dist`).
    """
    if world is not None:
        rules = process_rules if process_rules is not None else getattr(world, "process_rules", {})
        learned = learned_rules if learned_rules is not None else getattr(world, "learned_rules", {})
        return compute_alignment_analytics(
            world,
            rules or {},
            learned,
            step,
            duration_s=duration_s,
        )

    class _WorldView:
        def __init__(self):
            self.brick = {
                "visible": bool(visible),
                "x_axis": _coerce_float(x_axis, 0.0) or 0.0,
                "offset_x": _coerce_float(x_axis, 0.0) or 0.0,
                "angle": _coerce_float(angle, 0.0) or 0.0,
                "dist": _coerce_float(dist, 0.0) or 0.0,
            }
            self._align_focus_dist = False
            self._visibility_lost_frames = 0
            self.last_visible_time = None
            self.last_seen_x_axis = None
            self.last_seen_angle = None
            self.last_seen_dist = None

    world_view = _WorldView()
    return compute_alignment_analytics(
        world_view,
        process_rules or {},
        learned_rules,
        step,
        duration_s=duration_s,
    )


def select_align_brick_next_act(
    *,
    process_rules,
    learned_rules=None,
    step="ALIGN_BRICK",
    x_axis_mm,
    y_axis_mm=None,
    dist_mm,
    visible=True,
    angle_deg=0.0,
    duration_s=0.05,
    previous_correction_type=None,
    avoid_correction_type=None,
    planner_state=None,
):
    """
    Single source selector for ALIGN-style x/y/dist micro-adjust next act (cmd + speed score).

    This function encapsulates the exact approach used by the best calibrate trials:
    - Direction source: `compute_alignment_decision(...)`
    - Distance score source: `align_brick_dist_error_speed_score(...)`
    - Turn score source: `align_brick_x_axis_one_shot_score(...)`
    """
    step_name = _step_name(step) or "ALIGN_BRICK"
    align_policy = _align_policy_for_step(process_rules, step_name)
    analytics = compute_alignment_decision(
        world=None,
        step=step_name,
        process_rules=process_rules,
        learned_rules=learned_rules,
        duration_s=duration_s,
        visible=bool(visible),
        x_axis=x_axis_mm,
        angle=angle_deg,
        dist=dist_mm,
    )

    prod_cmd = analytics.get("cmd")
    worst_metric = analytics.get("worst_metric")

    _obj_name, success_gates = _success_gates_for_step(process_rules, step_name)
    if not isinstance(success_gates, dict):
        success_gates = {}
    normalized_success_gates = dict(success_gates)
    if "xAxis_offset_abs" not in normalized_success_gates:
        if "x_axis" in normalized_success_gates:
            normalized_success_gates["xAxis_offset_abs"] = normalized_success_gates.get("x_axis")
        elif "xAxis_offset" in normalized_success_gates:
            normalized_success_gates["xAxis_offset_abs"] = normalized_success_gates.get("xAxis_offset")
    if "yAxis_offset_abs" not in normalized_success_gates:
        if "y_axis" in normalized_success_gates:
            normalized_success_gates["yAxis_offset_abs"] = normalized_success_gates.get("y_axis")
        elif "yAxis_offset" in normalized_success_gates:
            normalized_success_gates["yAxis_offset_abs"] = normalized_success_gates.get("yAxis_offset")

    x_stats = normalized_success_gates.get("xAxis_offset_abs")
    if not isinstance(x_stats, dict):
        x_stats = {}
    x_required = bool(x_stats)
    y_stats = normalized_success_gates.get("yAxis_offset_abs")
    if not isinstance(y_stats, dict):
        y_stats = {}
    y_required = bool(y_stats)
    d_stats = normalized_success_gates.get("dist")
    if not isinstance(d_stats, dict):
        d_stats = {}
    d_required = bool(d_stats)
    angle_stats = normalized_success_gates.get("angle_abs")
    if not isinstance(angle_stats, dict):
        angle_stats = {}
    angle_required = bool(angle_stats)

    x_target = _coerce_float(x_stats.get("target"), 0.0) or 0.0
    # `offset_y` is relative to the camera vertical center, so target remains 0 by default.
    y_target = _coerce_float(y_stats.get("target"), 0.0)
    if y_target is None:
        y_target = 0.0
    dist_target = _coerce_float(d_stats.get("target"), 0.0) or 0.0
    angle_target = _coerce_float(angle_stats.get("target"), 0.0) or 0.0

    x_tol = abs(_coerce_float(x_stats.get("tol"), 0.0) or 0.0)
    y_tol = _coerce_float(y_stats.get("tol"), None)
    if y_tol is None:
        y_tol = x_tol
    y_tol = abs(float(y_tol) or 0.0)
    dist_tol = abs(_coerce_float(d_stats.get("tol"), 0.0) or 0.0)
    angle_tol = abs(_coerce_float(angle_stats.get("tol"), 0.0) or 0.0)

    x_direction = metric_direction_for_step("xAxis_offset_abs", step_name, process_rules=process_rules)
    y_direction = metric_direction_for_step("yAxis_offset_abs", step_name, process_rules=process_rules)
    d_direction = metric_direction_for_step("dist", step_name, process_rules=process_rules)
    angle_direction = metric_direction_for_step("angle_abs", step_name, process_rules=process_rules)

    x_axis_val = _coerce_float(x_axis_mm, 0.0) or 0.0
    x_err_mm = float(x_axis_val - x_target)
    y_axis_val = _coerce_float(y_axis_mm, None)
    y_err_mm = None if y_axis_val is None else float(y_axis_val - y_target)
    dist_val = _coerce_float(dist_mm, None)
    angle_val = _coerce_float(angle_deg, 0.0) or 0.0
    angle_signed_err = float(angle_val - angle_target)
    if dist_val is None:
        dist_err_mm = float("inf")
    else:
        dist_err_mm = abs(float(dist_val - dist_target))
    x_gate_outside_mm = _gate_outside_mm(x_axis_val, x_stats, x_direction) if x_required else 0.0
    y_gate_outside_mm = (
        _gate_outside_mm(y_axis_val, y_stats, y_direction)
        if (y_required and y_axis_val is not None)
        else 0.0
    )
    dist_gate_outside_mm = (
        _gate_outside_mm(dist_val, d_stats, d_direction)
        if (d_required and dist_val is not None)
        else 0.0
    )
    angle_gate_outside_mm = (
        _gate_outside_mm(abs(float(angle_val)), angle_stats, angle_direction)
        if angle_required
        else 0.0
    )

    def _gate_ratio(outside_mm, tol_mm, *, fallback_scale=1.0):
        try:
            err = max(0.0, float(outside_mm or 0.0))
        except (TypeError, ValueError):
            return 0.0
        try:
            denom = max(float(tol_mm), float(fallback_scale), 1.0)
        except (TypeError, ValueError):
            denom = max(float(fallback_scale), 1.0)
        return float(err) / float(denom)

    x_ratio = _gate_ratio(x_gate_outside_mm, x_tol) if x_required else 0.0
    y_ratio = _gate_ratio(y_gate_outside_mm, y_tol) if (y_required and y_err_mm is not None) else 0.0
    d_ratio = (
        _gate_ratio(dist_gate_outside_mm, dist_tol, fallback_scale=max(abs(float(dist_target)), 1.0))
        if (d_required and dist_val is not None)
        else 0.0
    )
    angle_ratio = _gate_ratio(angle_gate_outside_mm, angle_tol, fallback_scale=1.0) if angle_required else 0.0
    if angle_required:
        try:
            x_gate_outside_mm = max(
                float(x_gate_outside_mm or 0.0),
                float(angle_gate_outside_mm or 0.0),
            )
        except (TypeError, ValueError):
            x_gate_outside_mm = float(angle_gate_outside_mm or 0.0)
        x_ratio = max(float(x_ratio), float(angle_ratio))
    try:
        ratio_eps = max(0.0, float(GAP_ALIGN_OUTSIDE_GATE_RATIO_EPS))
    except (TypeError, ValueError):
        ratio_eps = 0.0

    def _priority_ratio(correction_type, raw_ratio):
        try:
            ratio_val = max(0.0, float(raw_ratio or 0.0))
        except (TypeError, ValueError):
            return 0.0
        if str(correction_type or "").strip().lower() == "y_axis":
            try:
                penalty = max(
                    1.0,
                    float(
                        _coerce_float(
                            align_policy.get("gap_rotation_y_priority_penalty"),
                            GAP_ALIGN_Y_AXIS_PRIORITY_PENALTY,
                        )
                        or GAP_ALIGN_Y_AXIS_PRIORITY_PENALTY
                    ),
                )
            except (TypeError, ValueError):
                penalty = 1.0
            return float(ratio_val) / float(penalty)
        return float(ratio_val)

    def _ratio_is_near_gate(raw_ratio, *, threshold=None):
        try:
            ratio_val = max(0.0, float(raw_ratio or 0.0))
        except (TypeError, ValueError):
            return False
        if threshold is None:
            try:
                threshold = max(0.0, float(GAP_ALIGN_Y_AXIS_OTHER_GAPS_NEAR_RATIO_MAX))
            except (TypeError, ValueError):
                threshold = 0.05
        try:
            return float(ratio_val) <= float(threshold)
        except (TypeError, ValueError):
            return False

    y_other_gaps_near_ready = True
    if bool(x_stats):
        y_other_gaps_near_ready = y_other_gaps_near_ready and _ratio_is_near_gate(x_ratio)
    if bool(d_stats):
        if dist_val is None:
            y_other_gaps_near_ready = False
        else:
            y_other_gaps_near_ready = y_other_gaps_near_ready and _ratio_is_near_gate(d_ratio)

    if prod_cmd not in ("f", "b", "l", "r"):
        prod_cmd = "l" if x_err_mm <= 0.0 else "r"
        if not worst_metric:
            worst_metric = "xAxis_offset_abs"

    candidates = {}
    recovery_fallback_candidates = {}

    if x_required:
        x_candidate = {
            "cmd": "l" if x_err_mm <= 0.0 else "r",
            "correction_type": "x_axis",
            "score": int(align_brick_x_axis_one_shot_score(x_err_mm)),
            "score_float": None,
            "reason": "x_axis_alignment",
            "worst_metric": "xAxis_offset_abs",
            "ratio": float(x_ratio),
        }
        recovery_fallback_candidates["x_axis"] = dict(x_candidate)
        if x_ratio > ratio_eps:
            candidates["x_axis"] = dict(x_candidate)

    if angle_required:
        if prod_cmd in ("l", "r"):
            angle_cmd = str(prod_cmd)
        elif float(angle_signed_err) > 0.0:
            angle_cmd = "l"
        elif float(angle_signed_err) < 0.0:
            angle_cmd = "r"
        else:
            angle_cmd = None
        try:
            angle_score = int(normalize_speed_score(analytics.get("speed_score")))
        except Exception:
            angle_score = int(align_brick_x_axis_one_shot_score(float(angle_signed_err)))
        if angle_cmd in ("l", "r"):
            angle_candidate = {
                "cmd": str(angle_cmd),
                "correction_type": "x_axis",
                "score": int(angle_score),
                "score_float": None,
                "reason": "angle_alignment",
                "worst_metric": "angle_abs",
                "ratio": float(angle_ratio),
            }
            existing_recovery_x = recovery_fallback_candidates.get("x_axis")
            existing_recovery_ratio = (
                float(existing_recovery_x.get("ratio", 0.0))
                if isinstance(existing_recovery_x, dict)
                else -1.0
            )
            if float(angle_candidate["ratio"]) >= existing_recovery_ratio:
                recovery_fallback_candidates["x_axis"] = dict(angle_candidate)
            if angle_ratio > ratio_eps:
                existing_x = candidates.get("x_axis")
                existing_ratio = float(existing_x.get("ratio", 0.0)) if isinstance(existing_x, dict) else -1.0
                if float(angle_candidate["ratio"]) >= existing_ratio:
                    candidates["x_axis"] = dict(angle_candidate)

    if y_required and y_err_mm is not None:
        y_candidate = {
            "cmd": "d" if float(y_err_mm) > 0.0 else "u",
            "correction_type": "y_axis",
            "score": int(align_brick_y_axis_one_shot_score(float(y_err_mm))),
            "score_float": None,
            "reason": "y_axis_alignment",
            "worst_metric": "yAxis_offset_abs",
            "ratio": float(y_ratio),
        }
        recovery_fallback_candidates["y_axis"] = dict(y_candidate)
        if y_ratio > ratio_eps:
            candidates["y_axis"] = dict(y_candidate)

    if d_required and dist_val is not None:
        dist_cmd = None
        target_num = _coerce_float(d_stats.get("target"), None)
        tol_num = _coerce_float(d_stats.get("tol"), None)
        if target_num is not None and tol_num is not None:
            tol_abs = abs(float(tol_num))
            dir_key = str(d_direction or "").strip().lower()
            if dir_key == "high":
                if float(dist_val) < (float(target_num) - tol_abs):
                    dist_cmd = "b"
            elif dir_key == "low":
                if float(dist_val) > (float(target_num) + tol_abs):
                    dist_cmd = "f"
            else:
                if float(dist_val) > (float(target_num) + tol_abs):
                    dist_cmd = "f"
                elif float(dist_val) < (float(target_num) - tol_abs):
                    dist_cmd = "b"
        if dist_cmd is None:
            d_min = _coerce_float(d_stats.get("min"), None)
            d_max = _coerce_float(d_stats.get("max"), None)
            if d_max is not None and float(dist_val) > float(d_max):
                dist_cmd = "f"
            elif d_min is not None and float(dist_val) < float(d_min):
                dist_cmd = "b"
        if dist_cmd is None and target_num is not None:
            # Recovery fallback: when a correction type is disqualified after
            # visibility loss, allow a toward-target distance nudge even if
            # distance is currently inside its gate window.
            dir_key = str(d_direction or "").strip().lower()
            if dir_key == "low":
                if float(dist_val) > float(target_num):
                    dist_cmd = "f"
            elif dir_key == "high":
                if float(dist_val) < float(target_num):
                    dist_cmd = "b"
            else:
                if float(dist_val) > float(target_num):
                    dist_cmd = "f"
                elif float(dist_val) < float(target_num):
                    dist_cmd = "b"
        if dist_cmd is not None:
            dist_score_err_mm = dist_gate_outside_mm
            if dist_score_err_mm is None:
                dist_score_err_mm = dist_err_mm
            score_float_dist = float(align_brick_dist_error_speed_score(dist_score_err_mm))
            dist_candidate = {
                "cmd": str(dist_cmd),
                "correction_type": "distance",
                "score": int(round(score_float_dist)),
                "score_float": score_float_dist,
                "reason": "distance_alignment",
                "worst_metric": "dist",
                "ratio": float(d_ratio),
            }
            recovery_fallback_candidates["distance"] = dict(dist_candidate)
            if d_ratio > ratio_eps:
                candidates["distance"] = dict(dist_candidate)

    dist_priority_cheat_active = False
    dist_priority_cheat_context = None
    if bool(align_policy.get("dist_priority_cheat_enabled")) and "distance" in candidates:
        try:
            cheat_min_ratio = max(0.0, float(align_policy.get("dist_priority_cheat_min_ratio", 0.0)))
        except (TypeError, ValueError):
            cheat_min_ratio = 0.0
        try:
            cheat_min_outside_mm = max(0.0, float(align_policy.get("dist_priority_cheat_min_outside_mm", 0.0)))
        except (TypeError, ValueError):
            cheat_min_outside_mm = 0.0
        try:
            dist_ratio_now = max(0.0, float(d_ratio or 0.0))
        except (TypeError, ValueError):
            dist_ratio_now = 0.0
        try:
            dist_outside_now = max(0.0, float(dist_gate_outside_mm or 0.0))
        except (TypeError, ValueError):
            dist_outside_now = 0.0
        if dist_ratio_now > max(float(ratio_eps), float(cheat_min_ratio)) and dist_outside_now >= float(
            cheat_min_outside_mm
        ):
            dist_priority_cheat_active = True
            dist_priority_cheat_context = {
                "dist_ratio": float(dist_ratio_now),
                "dist_outside_mm": float(dist_outside_now),
                "x_ratio": float(x_ratio),
                "x_outside_mm": float(x_gate_outside_mm or 0.0),
                "y_ratio": float(y_ratio),
                "y_outside_mm": float(y_gate_outside_mm or 0.0),
                "min_ratio": float(cheat_min_ratio),
                "min_outside_mm": float(cheat_min_outside_mm),
            }

    y_edge_force_triggered = False
    y_edge_force_threshold_mm = None
    if "y_axis" in candidates and y_err_mm is not None:
        y_edge_force_enabled = bool(align_policy.get("y_axis_edge_force_enabled", True))
        raw_edge_force_abs_mm = _coerce_float(align_policy.get("y_axis_edge_force_abs_mm"), None)
        if raw_edge_force_abs_mm is not None and raw_edge_force_abs_mm > 0.0:
            y_edge_force_threshold_mm = float(raw_edge_force_abs_mm)
        else:
            y_edge_force_threshold_mm = max(
                float(GAP_ALIGN_Y_AXIS_EDGE_FORCE_ABS_MM_MIN),
                float(y_tol) * float(GAP_ALIGN_Y_AXIS_EDGE_FORCE_TOL_MULT),
            )
        if y_edge_force_enabled:
            y_edge_force_triggered = abs(float(y_err_mm)) > float(y_edge_force_threshold_mm)

    y_close_bottom_bias_triggered = False
    y_close_bottom_bias_dist_threshold_mm = None
    y_close_bottom_bias_min_mm = None
    if "y_axis" in candidates and y_err_mm is not None and dist_val is not None:
        y_close_bottom_bias_enabled = bool(align_policy.get("y_axis_close_bottom_bias_enabled", True))
        raw_dist_bias_threshold = _coerce_float(align_policy.get("y_axis_close_bottom_dist_mm_max"), None)
        raw_bottom_min_mm = _coerce_float(align_policy.get("y_axis_close_bottom_min_mm"), None)
        if raw_dist_bias_threshold is not None and raw_dist_bias_threshold > 0.0:
            y_close_bottom_bias_dist_threshold_mm = float(raw_dist_bias_threshold)
        else:
            y_close_bottom_bias_dist_threshold_mm = 100.0
        if raw_bottom_min_mm is not None and raw_bottom_min_mm >= 0.0:
            y_close_bottom_bias_min_mm = float(raw_bottom_min_mm)
        else:
            y_close_bottom_bias_min_mm = max(1.0, 0.5 * float(y_tol))
        if y_close_bottom_bias_enabled:
            y_close_bottom_bias_triggered = bool(
                float(dist_val) < float(y_close_bottom_bias_dist_threshold_mm)
                and float(y_err_mm) >= float(y_close_bottom_bias_min_mm)
            )

    # Vertical camera-center alignment:
    # Positive y offset means marker is below center, so move mast down (`d`) to
    # bring the marker upward in the image. Negative y offset uses mast up (`u`).
    use_y_axis_correction = bool(
        "y_axis" in candidates
        and (
            bool(y_edge_force_triggered)
            or bool(y_close_bottom_bias_triggered)
            or (
                y_other_gaps_near_ready
                and _priority_ratio("y_axis", y_ratio) >= max(float(x_ratio), float(d_ratio))
            )
        )
    )

    # Single-source x-vs-dist safety focus (same intent as the best ALIGN_BRICK
    # behavior): when far from target and x is off, fix x first; only force depth
    # focus when extremely far and x is already reasonably aligned.
    force_x_axis_focus = False
    force_dist_focus = False
    try:
        x_axis_gap_mm = abs(float(x_err_mm))
        dist_gap_mm_signed = (float(dist_val) - float(dist_target)) if dist_val is not None else None
        dist_gap_mm = abs(float(dist_gap_mm_signed)) if dist_gap_mm_signed is not None else None
    except (TypeError, ValueError):
        x_axis_gap_mm = None
        dist_gap_mm = None
    if x_required and d_required and dist_gap_mm is not None and x_axis_gap_mm is not None:
        if dist_gap_mm > 50.0 and x_axis_gap_mm > 2.0:
            force_x_axis_focus = True
        elif dist_gap_mm > 150.0 and x_axis_gap_mm < 5.0:
            force_dist_focus = True

    def _best_alternative(
        excluded_types,
        *,
        include_recovery_fallback=False,
        require_outside_gate=True,
        enforce_y_near_ready=True,
    ):
        excluded = {
            str(item).strip().lower()
            for item in (excluded_types or set())
            if item is not None and str(item).strip()
        }
        viable = [
            dict(v)
            for k, v in candidates.items()
            if (
                str(k) not in excluded
                and (
                    not bool(require_outside_gate)
                    or float(v.get("ratio", 0.0) or 0.0) > ratio_eps
                )
                and (
                    not bool(enforce_y_near_ready)
                    or str(v.get("correction_type") or "").strip().lower() != "y_axis"
                    or bool(y_other_gaps_near_ready)
                )
            )
        ]
        if not viable and include_recovery_fallback:
            viable = [
                dict(v)
                for k, v in recovery_fallback_candidates.items()
                if (
                    str(k) not in excluded
                    and str(v.get("correction_type") or "").strip().lower() not in avoid_corr_set
                    and float(v.get("ratio", 0.0) or 0.0) > ratio_eps
                    and (
                        not bool(enforce_y_near_ready)
                        or str(v.get("correction_type") or "").strip().lower() != "y_axis"
                        or bool(y_other_gaps_near_ready)
                    )
                )
            ]
        if not viable:
            return None
        viable.sort(
            key=lambda row: (
                _priority_ratio(row.get("correction_type"), row.get("ratio", 0.0)),
                1.0 if str(row.get("correction_type")) != "y_axis" else 0.0,
            ),
            reverse=True,
        )
        return viable[0]

    prev_corr = str(previous_correction_type or "").strip().lower()

    def _normalize_avoid_corr_types(raw):
        if raw is None:
            return set()
        if isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = [raw]
        out = []
        for item in items:
            corr = str(item or "").strip().lower()
            if corr in {"x_axis", "y_axis", "distance"} and corr not in out:
                out.append(corr)
        return set(out)

    avoid_corr_set = _normalize_avoid_corr_types(avoid_correction_type)

    if bool(y_edge_force_triggered) and "y_axis" in candidates and "y_axis" not in avoid_corr_set:
        chosen = dict(candidates["y_axis"])
        chosen["reason"] = "y_axis_edge_force"
    elif bool(y_close_bottom_bias_triggered) and "y_axis" in candidates and "y_axis" not in avoid_corr_set:
        chosen = dict(candidates["y_axis"])
        chosen["reason"] = "y_axis_close_bottom_bias"
    elif dist_priority_cheat_active and "distance" in candidates and "distance" not in avoid_corr_set:
        chosen = dict(candidates["distance"])
        chosen["reason"] = "distance_priority_cheat"
        chosen["_cheat_dist_priority"] = True
    elif force_x_axis_focus and "x_axis" in candidates:
        chosen = dict(candidates["x_axis"])
    elif force_dist_focus and "distance" in candidates:
        chosen = dict(candidates["distance"])
    elif use_y_axis_correction:
        chosen = dict(candidates["y_axis"])
    else:
        best_primary = _best_alternative(set())
        if best_primary is not None:
            chosen = dict(best_primary)
        else:
            # Hard rule: if every gated metric is currently within its success gate,
            # do not invent a correction. Hold still and wait for gate confirmation.
            chosen = {
                "cmd": None,
                "correction_type": None,
                "score": 0,
                "score_float": None,
                "reason": "all_gaps_within_gate",
                "worst_metric": None,
                "ratio": 0.0,
            }

    cheat_locked_choice = bool(chosen.get("_cheat_dist_priority"))

    rotation_override = False
    chosen_type = str(chosen.get("correction_type") or "").strip().lower()
    if avoid_corr_set and chosen_type in avoid_corr_set:
        alt = _best_alternative(set(avoid_corr_set))
        if alt is None:
            alt = _best_alternative(
                set(avoid_corr_set),
                include_recovery_fallback=True,
                require_outside_gate=False,
                enforce_y_near_ready=False,
            )
        if alt is not None:
            chosen = alt
            chosen_type = str(chosen.get("correction_type") or "").strip().lower()
            rotation_override = True
    elif not cheat_locked_choice:
        if prev_corr and chosen_type == prev_corr:
            alt = _best_alternative({prev_corr})
            if alt is not None:
                chosen = alt
                chosen_type = str(chosen.get("correction_type") or "").strip().lower()
                rotation_override = True

    gap_rotation_active = False
    gap_rotation_chunk_switch = False
    gap_rotation_chunk_progress_mm = None
    gap_rotation_chunk_target_mm = None
    gap_rotation_y_hold_active = False
    gap_rotation_non_repeat_override = False
    gap_rotation_tech_debt_notes = []
    gap_rotation_enabled = bool(align_policy.get("gap_rotation_enabled", False))
    if gap_rotation_enabled:
        try:
            chunk_min_mm = max(0.5, float(_coerce_float(align_policy.get("gap_rotation_chunk_min_mm"), 3.0) or 3.0))
        except (TypeError, ValueError):
            chunk_min_mm = 3.0
        try:
            chunk_max_mm = max(chunk_min_mm, float(_coerce_float(align_policy.get("gap_rotation_chunk_max_mm"), 6.0) or 6.0))
        except (TypeError, ValueError):
            chunk_max_mm = max(chunk_min_mm, 6.0)
        try:
            y_hold_last_mm = max(0.0, float(_coerce_float(align_policy.get("gap_rotation_y_hold_last_mm"), 3.0) or 3.0))
        except (TypeError, ValueError):
            y_hold_last_mm = 3.0
        force_recovery_switch = bool(align_policy.get("gap_rotation_force_recovery_switch", True))
        log_tech_debt = bool(align_policy.get("gap_rotation_tech_debt_logging", False))

        outside_mm_by_type = {
            "x_axis": max(0.0, float(x_gate_outside_mm or 0.0)),
            "y_axis": max(0.0, float(y_gate_outside_mm or 0.0)),
            "distance": max(0.0, float(dist_gate_outside_mm or 0.0)),
        }

        non_y_outside_present = any(outside_mm_by_type.get(t, 0.0) > float(ratio_eps) for t in ("x_axis", "distance"))
        y_gap_now = outside_mm_by_type.get("y_axis", 0.0)
        gap_rotation_y_hold_active = bool(
            y_gap_now > float(ratio_eps) and y_gap_now <= float(y_hold_last_mm) and non_y_outside_present
        )

        if not isinstance(planner_state, dict):
            planner_state = {}
        rotation_state = planner_state.get("gap_rotation")
        if not isinstance(rotation_state, dict):
            rotation_state = {}
            planner_state["gap_rotation"] = rotation_state

        def _rotation_candidate(corr_type, *, include_recovery=False):
            corr_key = str(corr_type or "").strip().lower()
            row = candidates.get(corr_key)
            if row is None and include_recovery:
                row = recovery_fallback_candidates.get(corr_key)
            return dict(row) if isinstance(row, dict) else None

        def _chunk_target_for_gap(gap_mm):
            try:
                gap_val = max(0.0, float(gap_mm or 0.0))
            except (TypeError, ValueError):
                gap_val = 0.0
            if gap_val <= float(chunk_min_mm):
                return float(gap_val)
            scaled = max(float(chunk_min_mm), min(float(chunk_max_mm), float(gap_val) * 0.5))
            return float(scaled)

        def _pick_rotation_candidate(
            *,
            excluded_types=None,
            include_recovery=False,
            require_outside=True,
            allow_reserved_y=False,
        ):
            excluded = {str(item).strip().lower() for item in (excluded_types or []) if item is not None}
            pool = []
            for corr_key in ("x_axis", "distance", "y_axis"):
                if corr_key in excluded:
                    continue
                row = _rotation_candidate(corr_key, include_recovery=include_recovery)
                if row is None:
                    continue
                gap_now_local = max(0.0, float(outside_mm_by_type.get(corr_key, 0.0) or 0.0))
                if require_outside and gap_now_local <= float(ratio_eps):
                    continue
                if (
                    corr_key == "y_axis"
                    and bool(gap_rotation_y_hold_active)
                    and not bool(allow_reserved_y)
                ):
                    continue
                score = _priority_ratio(corr_key, row.get("ratio", 0.0))
                if score <= 0.0:
                    score = float(gap_now_local)
                pool.append((float(score), float(gap_now_local), 1.0 if corr_key != "y_axis" else 0.0, corr_key, row))
            if not pool:
                return None
            pool.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            _score, _gap, _non_y_bias, corr_key, row = pool[0]
            return corr_key, dict(row)

        active_type = str(rotation_state.get("active_type") or "").strip().lower()
        if active_type not in {"x_axis", "y_axis", "distance"}:
            active_type = None
            rotation_state["active_type"] = None
        active_start_gap = _coerce_float(rotation_state.get("chunk_start_gap_mm"), None)
        active_target_gap = _coerce_float(rotation_state.get("chunk_target_mm"), None)
        active_gap_now = outside_mm_by_type.get(active_type, 0.0) if active_type else None
        if active_type and active_start_gap is not None and active_target_gap is not None and active_gap_now is not None:
            gap_rotation_chunk_progress_mm = max(0.0, float(active_start_gap) - float(active_gap_now))
            gap_rotation_chunk_target_mm = max(0.0, float(active_target_gap))
            if (
                float(active_gap_now) <= float(ratio_eps)
                or float(gap_rotation_chunk_progress_mm) >= max(0.0, float(active_target_gap))
            ):
                rotation_state["last_completed_type"] = str(active_type)
                rotation_state["active_type"] = None
                rotation_state["chunk_start_gap_mm"] = None
                rotation_state["chunk_target_mm"] = None
                active_type = None
                gap_rotation_chunk_switch = True

        if not active_type:
            excluded = set()
            if force_recovery_switch and avoid_corr_set:
                excluded.update(set(avoid_corr_set))
            last_completed = str(rotation_state.get("last_completed_type") or "").strip().lower()
            if last_completed in {"x_axis", "y_axis", "distance"}:
                excluded.add(last_completed)
            picked = _pick_rotation_candidate(excluded_types=excluded, include_recovery=False, require_outside=True)
            if picked is None and last_completed in excluded:
                excluded_wo_last = {item for item in excluded if item != last_completed}
                picked = _pick_rotation_candidate(
                    excluded_types=excluded_wo_last,
                    include_recovery=False,
                    require_outside=True,
                )
            if picked is None and force_recovery_switch and avoid_corr_set:
                picked = _pick_rotation_candidate(
                    excluded_types=set(avoid_corr_set),
                    include_recovery=True,
                    require_outside=True,
                    allow_reserved_y=True,
                )
                if picked is not None:
                    gap_rotation_non_repeat_override = True
            if picked is not None:
                active_type, _picked_row = picked
                active_gap_now = outside_mm_by_type.get(active_type, 0.0)
                active_start_gap = max(0.0, float(active_gap_now or 0.0))
                active_target_gap = _chunk_target_for_gap(active_start_gap)
                rotation_state["active_type"] = str(active_type)
                rotation_state["chunk_start_gap_mm"] = float(active_start_gap)
                rotation_state["chunk_target_mm"] = float(active_target_gap)
                gap_rotation_chunk_progress_mm = 0.0
                gap_rotation_chunk_target_mm = float(active_target_gap)

        if active_type:
            forced = _rotation_candidate(active_type, include_recovery=True)
            if forced is not None:
                prev_type_for_override = str(chosen.get("correction_type") or "").strip().lower()
                chosen = dict(forced)
                if gap_rotation_non_repeat_override:
                    chosen["reason"] = "gap_rotation_recovery_nonrepeat"
                elif bool(gap_rotation_chunk_switch):
                    chosen["reason"] = "gap_rotation_chunk_switch"
                else:
                    chosen["reason"] = "gap_rotation_chunk_follow"
                chosen["_gap_rotation_active"] = True
                chosen["_gap_rotation_chunk_switch"] = bool(gap_rotation_chunk_switch)
                if prev_type_for_override != str(chosen.get("correction_type") or "").strip().lower():
                    rotation_override = True
                chosen_type = str(chosen.get("correction_type") or "").strip().lower()
                gap_rotation_active = True

        if log_tech_debt and gap_rotation_active:
            chunk_prog_txt = (
                "N/A" if gap_rotation_chunk_progress_mm is None else f"{float(gap_rotation_chunk_progress_mm):.1f}"
            )
            chunk_goal_txt = (
                "N/A" if gap_rotation_chunk_target_mm is None else f"{float(gap_rotation_chunk_target_mm):.1f}"
            )
            gap_rotation_tech_debt_notes = [
                (
                    "TECH-DEBT hack active: chunked gap rotation "
                    f"({float(chunk_min_mm):.1f}-{float(chunk_max_mm):.1f}mm). "
                    f"chunk={chunk_prog_txt}/{chunk_goal_txt}mm; y_last_hold<={float(y_hold_last_mm):.1f}mm="
                    f"{'ON' if gap_rotation_y_hold_active else 'OFF'}."
                )
            ]

    gap_focus_cycle_switch = False
    gap_focus_cycle_only_one_remaining = False
    gap_focus_cycle_count = 0
    try:
        gap_focus_cycle_cap = int(
            round(
                float(
                    _coerce_float(
                        align_policy.get("gap_focus_max_cycles_before_switch"),
                        10,
                    )
                    or 0.0
                )
            )
        )
    except (TypeError, ValueError):
        gap_focus_cycle_cap = 10
    if gap_focus_cycle_cap < 1:
        gap_focus_cycle_cap = 0

    if not isinstance(planner_state, dict):
        planner_state = {}
    gap_focus_state = planner_state.get("gap_focus_cycle_guard")
    if not isinstance(gap_focus_state, dict):
        gap_focus_state = {}
        planner_state["gap_focus_cycle_guard"] = gap_focus_state

    chosen_type = str(chosen.get("correction_type") or "").strip().lower()
    chosen_cmd = str(chosen.get("cmd") or "").strip().lower()
    active_type_set = {"x_axis", "y_axis", "distance"}
    valid_chosen_type = chosen_type in active_type_set and chosen_cmd in {"f", "b", "l", "r", "u", "d"}
    if valid_chosen_type:
        prev_focus_type = str(gap_focus_state.get("active_type") or "").strip().lower()
        try:
            prev_focus_count = int(round(float(gap_focus_state.get("count") or 0)))
        except (TypeError, ValueError):
            prev_focus_count = 0
        if prev_focus_type == chosen_type:
            gap_focus_cycle_count = max(0, int(prev_focus_count)) + 1
        else:
            gap_focus_cycle_count = 1

        if gap_focus_cycle_cap > 0 and gap_focus_cycle_count > int(gap_focus_cycle_cap):
            excluded_for_switch = set(avoid_corr_set)
            excluded_for_switch.add(chosen_type)
            alt = _best_alternative(
                excluded_for_switch,
                include_recovery_fallback=False,
                require_outside_gate=True,
                enforce_y_near_ready=True,
            )
            if alt is None:
                alt = _best_alternative(
                    excluded_for_switch,
                    include_recovery_fallback=False,
                    require_outside_gate=True,
                    enforce_y_near_ready=False,
                )
            if alt is not None:
                prev_type_for_override = chosen_type
                chosen = dict(alt)
                chosen["reason"] = "gap_focus_cycle_switch"
                chosen_type = str(chosen.get("correction_type") or "").strip().lower()
                chosen_cmd = str(chosen.get("cmd") or "").strip().lower()
                valid_chosen_type = chosen_type in active_type_set and chosen_cmd in {"f", "b", "l", "r", "u", "d"}
                gap_focus_cycle_switch = True
                gap_focus_cycle_count = 1 if valid_chosen_type else 0
                if valid_chosen_type and prev_type_for_override != chosen_type:
                    rotation_override = True
            else:
                gap_focus_cycle_only_one_remaining = True

        if valid_chosen_type:
            gap_focus_state["active_type"] = str(chosen_type)
            gap_focus_state["count"] = int(max(1, gap_focus_cycle_count))
        else:
            gap_focus_state["active_type"] = None
            gap_focus_state["count"] = 0
    else:
        gap_focus_state["active_type"] = None
        gap_focus_state["count"] = 0

    correction_type = chosen.get("correction_type")
    score_float = chosen.get("score_float")
    try:
        score_int = int(chosen.get("score"))
    except (TypeError, ValueError):
        score_int = 0
    prod_cmd = chosen.get("cmd")
    if prod_cmd is not None:
        prod_cmd = str(prod_cmd).strip().lower()
    if prod_cmd not in {"f", "b", "l", "r", "u", "d"}:
        prod_cmd = None
    reason = str(chosen.get("reason"))
    worst_metric = chosen.get("worst_metric")

    # Single-solution rule for gap micro-adjustments: keep the score selected by
    # this gap planner (x/dist/y one-shot logic). Do not override from the generic
    # analytics planner, which can create a second conflicting path across steps.

    return {
        "cmd": prod_cmd,
        "correction_type": (None if correction_type is None else str(correction_type)),
        "score": int(score_int),
        "score_float": score_float,
        "reason": str(reason),
        "worst_metric": worst_metric,
        "rotation_override": bool(rotation_override),
        "cheat_dist_priority": bool(chosen.get("_cheat_dist_priority")),
        "cheat_dist_priority_context": (
            dict(dist_priority_cheat_context) if isinstance(dist_priority_cheat_context, dict) else None
        ),
        "gap_rotation_active": bool(gap_rotation_active),
        "gap_rotation_chunk_switch": bool(gap_rotation_chunk_switch),
        "gap_rotation_chunk_progress_mm": (
            None if gap_rotation_chunk_progress_mm is None else float(gap_rotation_chunk_progress_mm)
        ),
        "gap_rotation_chunk_target_mm": (
            None if gap_rotation_chunk_target_mm is None else float(gap_rotation_chunk_target_mm)
        ),
        "gap_rotation_y_hold_active": bool(gap_rotation_y_hold_active),
        "gap_rotation_non_repeat_override": bool(gap_rotation_non_repeat_override),
        "gap_focus_cycle_cap": (None if gap_focus_cycle_cap <= 0 else int(gap_focus_cycle_cap)),
        "gap_focus_cycle_count": int(max(0, gap_focus_cycle_count)),
        "gap_focus_cycle_switch": bool(gap_focus_cycle_switch),
        "gap_focus_cycle_only_one_remaining": bool(gap_focus_cycle_only_one_remaining),
        "exception_tech_debt_notes": (
            list(gap_rotation_tech_debt_notes) if isinstance(gap_rotation_tech_debt_notes, list) else []
        ),
        "x_err_mm": float(x_err_mm),
        "y_err_mm": (None if y_err_mm is None else float(y_err_mm)),
        "dist_err_mm": float(dist_err_mm),
        "dist_target_mm": float(dist_target),
        "x_tol_mm": float(x_tol),
        "y_tol_mm": float(y_tol),
        "y_axis_edge_force_triggered": bool(y_edge_force_triggered),
        "y_axis_edge_force_threshold_mm": (
            None if y_edge_force_threshold_mm is None else float(y_edge_force_threshold_mm)
        ),
        "y_axis_close_bottom_bias_triggered": bool(y_close_bottom_bias_triggered),
        "y_axis_close_bottom_bias_dist_threshold_mm": (
            None
            if y_close_bottom_bias_dist_threshold_mm is None
            else float(y_close_bottom_bias_dist_threshold_mm)
        ),
        "y_axis_close_bottom_bias_min_mm": (
            None if y_close_bottom_bias_min_mm is None else float(y_close_bottom_bias_min_mm)
        ),
    }


def select_alignment_next_act(
    *,
    process_rules,
    learned_rules=None,
    step="ALIGN_BRICK",
    x_axis_mm,
    y_axis_mm=None,
    dist_mm,
    visible=True,
    angle_deg=0.0,
    duration_s=0.05,
    previous_correction_type=None,
    avoid_correction_type=None,
    planner_state=None,
):
    """
    Unified ALIGN-step next-act selector.

    Uses the gap micro-planner for brick-placement style steps (x/y/dist gates),
    and falls back to the generic analytics planner for angle-only / visible-only
    align steps.
    """
    # Gap-planner steps still need a search phase while target visibility is false.
    # Switch to micro-gap corrections only once visibility is confirmed.
    use_gap_planner = bool(step_uses_gap_alignment_planner(process_rules, step) and bool(visible))
    if use_gap_planner:
        plan = select_align_brick_next_act(
            process_rules=process_rules,
            learned_rules=learned_rules,
            step=step,
            x_axis_mm=x_axis_mm,
            y_axis_mm=y_axis_mm,
            dist_mm=dist_mm,
            visible=visible,
            angle_deg=angle_deg,
            duration_s=duration_s,
            previous_correction_type=previous_correction_type,
            avoid_correction_type=avoid_correction_type,
            planner_state=planner_state,
        )
        cmd = plan.get("cmd")
        score = plan.get("score")
        speed = manual_speed_for_cmd(cmd, score) if cmd and score is not None else 0.0
        out = dict(plan)
        out["speed"] = float(speed or 0.0)
        out["planner"] = "gap"
        return out

    analytics = compute_alignment_decision(
        world=None,
        step=step,
        process_rules=process_rules,
        learned_rules=learned_rules,
        duration_s=duration_s,
        visible=bool(visible),
        x_axis=x_axis_mm,
        angle=angle_deg,
        dist=dist_mm,
    )
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") or 0.0
    score = analytics.get("speed_score")
    if cmd in ("f", "b"):
        correction_type = "distance"
    elif cmd in ("l", "r"):
        correction_type = "x_axis"
    elif cmd in ("u", "d"):
        correction_type = "y_axis"
    else:
        correction_type = None
    return {
        "cmd": cmd,
        "correction_type": correction_type,
        "score": score,
        "speed": float(speed or 0.0),
        "reason": analytics.get("worst_metric") or "align",
        "worst_metric": analytics.get("worst_metric"),
        "planner": "generic",
    }


def align_local_gate_status(world, step="ALIGN_BRICK", process_rules=None):
    """
    Evaluate strict local ALIGN_BRICK-style gate truth from the current brick state.

    Returns a dict with bools and per-metric errors so callers can combine this with
    broader gate checker signals (consecutive/majority trackers).
    """
    rules = process_rules if process_rules is not None else getattr(world, "process_rules", {})
    _obj_name, success_gates = _success_gates_for_step(rules, step)
    brick = getattr(world, "brick", {}) or {}

    visible = bool(brick.get("visible"))
    x_axis_raw = brick.get("x_axis", brick.get("offset_x"))
    y_axis_raw = brick.get("y_axis", brick.get("offset_y"))
    dist_raw = brick.get("dist")

    x_axis = _coerce_float(x_axis_raw, None)
    y_axis = _coerce_float(y_axis_raw, None)
    dist_val = _coerce_float(dist_raw, None)

    x_stats = success_gates.get("xAxis_offset_abs") if isinstance(success_gates, dict) else {}
    if not isinstance(x_stats, dict):
        x_stats = {}
    x_required = bool(isinstance(x_stats, dict) and x_stats)
    x_target = _coerce_float(x_stats.get("target"), 0.0)
    x_tol = abs(_coerce_float(x_stats.get("tol"), 0.0) or 0.0)
    x_err = None if x_axis is None else float(x_axis - x_target)
    x_abs_err = None if x_err is None else abs(float(x_err))
    x_within_tol = (not x_required) or bool(x_abs_err is not None and x_abs_err <= x_tol)

    y_stats = success_gates.get("yAxis_offset_abs") if isinstance(success_gates, dict) else {}
    if not isinstance(y_stats, dict):
        y_stats = {}
    y_required = bool(y_stats)
    y_target = _coerce_float(y_stats.get("target"), 0.0)
    if y_target is None:
        y_target = 0.0
    y_tol = _coerce_float(y_stats.get("tol"), None)
    if y_tol is None:
        y_tol = x_tol
    y_tol = abs(float(y_tol) or 0.0)
    y_err = None if y_axis is None else float(y_axis - y_target)
    y_abs_err = None if y_err is None else abs(float(y_err))
    y_within_tol = (not y_required) or bool(y_abs_err is not None and y_abs_err <= y_tol)

    d_stats = success_gates.get("dist") if isinstance(success_gates, dict) else {}
    if not isinstance(d_stats, dict):
        d_stats = {}
    d_required = bool(d_stats)
    d_target = _coerce_float(d_stats.get("target"), 0.0)
    d_tol = abs(_coerce_float(d_stats.get("tol"), 0.0) or 0.0)
    dist_err = None if dist_val is None else abs(float(dist_val - d_target))
    d_direction = metric_direction_for_step("dist", _step_name(step), process_rules=rules)
    dist_match = None
    if d_required and dist_val is not None:
        dist_match = _target_tol_ok(float(dist_val), d_stats, d_direction)
        if dist_match is None:
            d_min = _coerce_float(d_stats.get("min"), None)
            d_max = _coerce_float(d_stats.get("max"), None)
            dist_match = True
            if d_min is not None and float(dist_val) < float(d_min):
                dist_match = False
            if d_max is not None and float(dist_val) > float(d_max):
                dist_match = False
    dist_within_tol = (not d_required) or bool(dist_match)

    ok = bool(
        visible
        and x_within_tol
        and dist_within_tol
        and (y_within_tol if y_required else True)
    )
    return {
        "ok": ok,
        "visible": bool(visible),
        "x_axis": x_axis,
        "x_target": x_target,
        "x_tol": x_tol,
        "x_err": x_err,
        "x_abs_err": x_abs_err,
        "x_within_tol": x_within_tol,
        "x_required": x_required,
        "y_axis": y_axis,
        "y_target": y_target,
        "y_tol": y_tol,
        "y_err": y_err,
        "y_abs_err": y_abs_err,
        "y_within_tol": y_within_tol,
        "y_required": y_required,
        "dist": dist_val,
        "dist_target": d_target,
        "dist_tol": d_tol,
        "dist_err": dist_err,
        "dist_within_tol": dist_within_tol,
        "dist_required": d_required,
    }


def format_align_pre_observation_text(
    *,
    x_err_mm,
    x_tol_mm,
    x_target_mm=0.0,
    offset_x=None,
    dist_mm=None,
):
    """Build a concise calibrate-style pre-action observation string."""
    parts = []
    x_err = _coerce_float(x_err_mm, None)
    x_tol = abs(_coerce_float(x_tol_mm, 0.0) or 0.0)
    x_target = _coerce_float(x_target_mm, 0.0) or 0.0
    if x_err is not None:
        parts.append(f"x_err={x_err:+.2f}mm")
        parts.append(f"gate=[{-x_tol:+.2f},{x_tol:+.2f}]mm")
        parts.append(f"target={x_target:+.2f}mm")
    else:
        parts.append("x_err=unknown")

    offset_val = _coerce_float(offset_x, None)
    if offset_val is not None:
        parts.append(f"offset={offset_val:+.2f}mm")

    dist_val = _coerce_float(dist_mm, None)
    if dist_val is not None:
        parts.append(f"dist={dist_val:.1f}mm")

    return " ".join(parts)


def build_align_result_delta_obj(
    *,
    correction_type,
    prev_x_err_mm=None,
    curr_x_err_mm=None,
    prev_dist_mm=None,
    curr_dist_mm=None,
    dist_target_mm=0.0,
    stable_deadzone_mm=0.02,
):
    """Build `result_delta_obj` payload for format_shorthand_calibration_line."""
    corr = str(correction_type or "").strip().lower()
    deadzone = abs(_coerce_float(stable_deadzone_mm, 0.02) or 0.02)

    if corr == "distance":
        prev_dist = _coerce_float(prev_dist_mm, None)
        curr_dist = _coerce_float(curr_dist_mm, None)
        target = _coerce_float(dist_target_mm, 0.0) or 0.0
        if prev_dist is None or curr_dist is None:
            return None
        prev_gap = abs(prev_dist - target)
        curr_gap = abs(curr_dist - target)
        delta = float(prev_gap - curr_gap)
        if delta > deadzone:
            delta_class = "closer"
            change_text = f"improved {abs(delta):.2f}mm"
        elif delta < -deadzone:
            delta_class = "backward"
            change_text = f"worsened {abs(delta):.2f}mm"
        else:
            delta_class = "unchanged"
            change_text = "no change"
        return {
            "delta_class": delta_class,
            "delta_text": f"dist {prev_dist:.1f}→{curr_dist:.1f}mm ({change_text})",
        }

    prev_x = _coerce_float(prev_x_err_mm, None)
    curr_x = _coerce_float(curr_x_err_mm, None)
    if prev_x is None or curr_x is None:
        return None
    prev_abs = abs(prev_x)
    curr_abs = abs(curr_x)
    delta = float(prev_abs - curr_abs)
    if delta > deadzone:
        delta_class = "closer"
        change_text = f"improved {abs(delta):.2f}mm"
    elif delta < -deadzone:
        delta_class = "backward"
        change_text = f"worsened {abs(delta):.2f}mm"
    else:
        delta_class = "unchanged"
        change_text = "no change"
    return {
        "delta_class": delta_class,
        "delta_text": f"x_err {prev_abs:.2f}→{curr_abs:.2f}mm ({change_text})",
    }


"""
Brick alignment telemetry helpers and correction suggestions.
"""


def offset_side_label(offset_x):
    if offset_x is None:
        return ""
    if offset_x > 0:
        return "right"
    if offset_x < 0:
        return "left"
    return "center"


def offset_marker_direction(offset_x):
    side = offset_side_label(offset_x)
    if side == "left":
        return "left of the marker"
    if side == "right":
        return "right of the marker"
    return ""


def offset_gap_phrase(offset_x):
    side = offset_side_label(offset_x)
    if side == "right":
        return "between the right side of the robot and the aruco marker"
    if side == "left":
        return "between the left side of the robot and the aruco marker"
    return "between the robot and the aruco marker"


def distance_marker_direction(dist, gates):
    if dist is None:
        return ""
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        if dist > target + tol:
            return "in front of the marker"
        if dist < target - tol:
            return "behind the marker"
        return ""
    if max_val is not None and dist > max_val:
        return "in front of the marker"
    if min_val is not None and dist < min_val:
        return "behind the marker"
    return ""


def worst_offset_direction(metric, measurement, gates):
    if not measurement:
        return ""
    if metric == "xAxis_offset_abs":
        x_axis = measurement.get("x_axis")
        if x_axis is None:
            x_axis = measurement.get("offset_x")
        return offset_marker_direction(x_axis)
    if metric == "dist":
        return distance_marker_direction(measurement.get("dist"), gates)
    return ""


def gap_direction_from_cmd(axis, cmd):
    if axis == "angle":
        return "to the right" if cmd == "l" else "to the left"
    if axis == "offset":
        return ""
    if axis == "distance":
        return "in front" if cmd == "f" else "behind"
    return ""


def distance_correction_cmd(measurement, gates):
    if not measurement:
        return None
    dist = measurement.get("dist")
    if dist is None:
        return None
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        if dist > target + tol:
            return "f"
        if dist < target - tol:
            return "b"
        return None
    if max_val is not None and dist > max_val:
        return "f"
    if min_val is not None and dist < min_val:
        return "b"
    return None


def distance_gap_value(dist, gates):
    if dist is None:
        return None
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        return abs(dist - target)
    if max_val is not None and dist > max_val:
        return dist - max_val
    if min_val is not None and dist < min_val:
        return min_val - dist
    return None


def offset_gap_value(offset, gates):
    if offset is None:
        return None
    stats = (gates or {}).get("xAxis_offset_abs") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    abs_offset = abs(offset)
    if target is not None and tol is not None:
        return abs(abs_offset - target)
    if max_val is not None and abs_offset > max_val:
        return abs_offset - max_val
    if min_val is not None and abs_offset < min_val:
        return min_val - abs_offset
    return None


def suggested_minor_correction(brick, success_gates):
    if not brick or not brick.get("visible"):
        return None
    cmd = distance_correction_cmd(brick, success_gates)
    if cmd:
        return "forward" if cmd == "f" else "backward"
    return None


@dataclass
class BrickAlignmentState:
    dist: float
    offset: float
    angle: float
    visible: bool

    @classmethod
    def from_brick(cls, brick: Optional[dict]) -> "BrickAlignmentState":
        if not brick:
            return cls(0.0, 0.0, 0.0, False)
        dist = brick.get("dist")
        offset = brick.get("offset_x")
        angle = brick.get("angle")
        return cls(
            dist=float(dist) if dist is not None else 0.0,
            offset=float(offset) if offset is not None else 0.0,
            angle=float(angle) if angle is not None else 0.0,
            visible=bool(brick.get("visible")),
        )


@dataclass
class BrickAdjustment:
    mode: str
    distance_delta: float
    offset_delta: float
    angle_delta: float
    confidence: float = 0.0


class AlignmentEnvelope:
    def __init__(self, max_samples: int = 2048, neighbors: int = 6):
        self.max_samples = max_samples
        self.neighbors = max(1, neighbors)
        self.samples: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    def _normalize(self, state: BrickAlignmentState) -> Tuple[float, float, float]:
        dist = max(0.0, min(500.0, state.dist)) / 500.0
        offset = max(-200.0, min(200.0, state.offset)) / 200.0
        angle = max(-180.0, min(180.0, state.angle)) / 180.0
        return dist, offset, angle

    def record_transition(self, previous: BrickAlignmentState, current: BrickAlignmentState) -> None:
        if not (previous.visible and current.visible):
            return
        delta_dist = current.dist - previous.dist
        delta_offset = current.offset - previous.offset
        delta_angle = current.angle - previous.angle
        if (
            abs(delta_dist) < 0.3
            and abs(delta_offset) < 0.3
            and abs(delta_angle) < 0.25
        ):
            return
        features = self._normalize(previous)
        delta = (delta_dist, delta_offset, delta_angle)
        self.samples.append((features, delta))
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(
        self, state: BrickAlignmentState
    ) -> Optional[Tuple[float, float, float, float]]:
        if not self.samples or not state.visible:
            return None
        query = self._normalize(state)
        scored = []
        for features, delta in self.samples:
            dist = self._distance(query, features)
            scored.append((dist, delta))
        scored.sort(key=lambda pair: pair[0])
        top = scored[: min(self.neighbors, len(scored))]
        total_weight = 0.0
        weighted_dist = 0.0
        weighted_offset = 0.0
        weighted_angle = 0.0
        for dist, delta in top:
            weight = 1.0 / (dist + 1e-3)
            total_weight += weight
            weighted_dist += delta[0] * weight
            weighted_offset += delta[1] * weight
            weighted_angle += delta[2] * weight
        if total_weight == 0.0:
            return None
        confidence = min(1.0, len(top) / self.neighbors)
        return (
            weighted_dist / total_weight,
            weighted_offset / total_weight,
            weighted_angle / total_weight,
            confidence,
        )

    def learn_from_demos(self, demos_dir: Optional[Path] = None, session: Optional[str] = None) -> None:
        demos_dir = Path(demos_dir) if demos_dir else DEFAULT_DEMOS_DIR
        if not demos_dir.exists():
            return
        logs = load_demo_logs(demos_dir, session)
        for _, rows in logs:
            segments = extract_attempt_segments(rows)
            for seg in segments:
                states = seg.get("states") or []
                sorted_states = sorted(states, key=lambda row: row.get("timestamp", 0.0))
                for prev, curr in zip(sorted_states, sorted_states[1:]):
                    prev_state = BrickAlignmentState.from_brick(prev.get("brick"))
                    curr_state = BrickAlignmentState.from_brick(curr.get("brick"))
                    self.record_transition(prev_state, curr_state)


class BrickAlignmentController:
    APPROACH_DISTANCE_THRESHOLD = 70.0
    APPROACH_OFFSET_THRESHOLD = 40.0
    APPROACH_FALLBACK_DISTANCE_GAIN = 0.3
    APPROACH_FALLBACK_OFFSET_GAIN = 0.35
    APPROACH_FALLBACK_ANGLE_GAIN = 0.6
    MICRO_DISTANCE_GAIN = 0.4
    MICRO_OFFSET_GAIN = 0.45

    def __init__(self, demos_dir: Optional[Path] = None):
        self.demos_dir = Path(demos_dir) if demos_dir else DEFAULT_DEMOS_DIR
        self.envelope = AlignmentEnvelope()
        self._last_state: Optional[BrickAlignmentState] = None
        self.envelope.learn_from_demos(self.demos_dir)

    def _choose_mode(self, state: BrickAlignmentState) -> str:
        if not state.visible:
            return "unknown"
        if state.dist > self.APPROACH_DISTANCE_THRESHOLD or abs(state.offset) > self.APPROACH_OFFSET_THRESHOLD:
            return "approach"
        return "micro"

    def _register_telemetry(self, state: BrickAlignmentState) -> None:
        if self._last_state:
            self.envelope.record_transition(self._last_state, state)
        self._last_state = state

    def next_adjustment(self, brick: Optional[dict]) -> Optional[BrickAdjustment]:
        state = BrickAlignmentState.from_brick(brick)
        if not state.visible:
            self._register_telemetry(state)
            return None
        self._register_telemetry(state)
        mode = self._choose_mode(state)
        if mode == "approach":
            return self._approach_adjustment(state)
        return self._micro_adjustment(state)

    def _approach_adjustment(self, state: BrickAlignmentState) -> BrickAdjustment:
        prediction = self.envelope.predict(state)
        if prediction:
            dist_delta, offset_delta, angle_delta, confidence = prediction
        else:
            dist_delta = -state.dist * self.APPROACH_FALLBACK_DISTANCE_GAIN
            offset_delta = -state.offset * self.APPROACH_FALLBACK_OFFSET_GAIN
            angle_delta = -state.angle * self.APPROACH_FALLBACK_ANGLE_GAIN
            confidence = 0.0
        if abs(angle_delta) < 1e-3:
            angle_delta = -state.angle * self.APPROACH_FALLBACK_ANGLE_GAIN
        return BrickAdjustment(
            mode="approach",
            distance_delta=dist_delta,
            offset_delta=offset_delta,
            angle_delta=angle_delta,
            confidence=confidence,
        )

    def _micro_adjustment(self, state: BrickAlignmentState) -> BrickAdjustment:
        return BrickAdjustment(
            mode="micro",
            distance_delta=-state.dist * self.MICRO_DISTANCE_GAIN,
            offset_delta=-state.offset * self.MICRO_OFFSET_GAIN,
            angle_delta=0.0,
            confidence=1.0,
        )
