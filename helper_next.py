"""Decision helpers for choosing the robot's next action."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_step_label
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
    "dist": "low",
    "visible": "high",
    "confidence": "high",
}

DEFAULT_DEMOS_DIR = Path(__file__).resolve().parent / "demos"


def _step_name(step):
    return normalize_step_label(step)


def metric_direction_for_step(metric, step):
    direction = METRIC_DIRECTIONS.get(metric)
    obj_name = _step_name(step)
    if obj_name == "FIND_BRICK" and metric == "dist":
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


def align_turn_speed_score_for_step(step, x_err_mm: float, *, dist_gate_error_mm=None) -> int:
    """Return shared l/r speed score for a step using the active paradigm."""
    step_key = _step_name(step)
    if step_key == "ALIGN_BRICK":
        return int(align_brick_x_axis_one_shot_score(x_err_mm))
    if step_key == "POSITION_BRICK":
        return int(
            align_steps_turn_speed_score(
                x_err_mm,
                ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE,
                dist_gate_error_mm=dist_gate_error_mm,
            )
        )
    return int(
        align_steps_turn_speed_score(
            x_err_mm,
            ALIGN_STEPS_SHARED_TURN_FALLBACK_SCORE,
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
    return keys == {"visible"}


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


def align_brick_x_axis_tol_scale(dist_mm: float, process_rules, step: str) -> float:
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
    max_scale = float(ALIGN_BRICK_X_AXIS_TOL_FAR_SCALE)  # 6.0x

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
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
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

    # Auto-step ALIGN_BRICK micro-adjustments should never act blindly. If the
    # brick isn't visible, HOLD and let vision settle/reacquire.
    if obj_name == "ALIGN_BRICK" and not visible:
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
    if obj_name == "ALIGN_BRICK":
        x_axis_tol_scale = align_brick_x_axis_tol_scale(dist, process_rules, obj_name)
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
        direction = metric_direction_for_step(metric, obj_name)
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
    force_x_axis_focus = False  # New: prioritize x-axis when far from brick
    if obj_name == "ALIGN_BRICK":
        try:
            dist_gap_mm = abs(float(offsets.get("dist", 0.0) or 0.0))
        except (TypeError, ValueError):
            dist_gap_mm = None
        
        # CRITICAL SAFETY: When far from brick, prioritize x-axis alignment first
        # This prevents hitting the brick while misaligned on x-axis
        # Only focus on distance when x-axis is good OR distance is extreme (>150mm)
        if dist_gap_mm is not None:
            # If we're far (>50mm from target) and x-axis needs work, prioritize x-axis
            x_axis_gap_mm = mm_errors.get("xAxis_offset_abs", 0.0)
            if dist_gap_mm > 50.0 and x_axis_gap_mm > 2.0:
                force_x_axis_focus = True
            # Only force distance focus if extremely far AND x-axis is acceptable
            elif dist_gap_mm > 150.0 and x_axis_gap_mm < 5.0:
                force_dist_focus = True
            # Medium distance (50-150mm): use natural worst_metric selection
            else:
                force_dist_focus = False
                force_x_axis_focus = False
        
        # Update sticky focus state (legacy compatibility)
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
            direction = metric_direction_for_step(metric, obj_name)
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
        elif obj_name == "POSITION_BRICK":
            align_profile = learned_rules.get("ALIGN_BRICK")
            if isinstance(align_profile, dict) and isinstance(align_profile.get("calibration_profile"), dict):
                profile = dict(align_profile.get("calibration_profile") or {})

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
        slow_mm = ALIGN_SPEED_SLOW_MM
        fast_mm = ALIGN_SPEED_FAST_MM
        if obj_name != "ALIGN_BRICK":
            slow_mm = ALIGN_SPEED_SLOW_MM / 4.0
            fast_mm = ALIGN_SPEED_FAST_MM / 4.0
        if obj_name == "ALIGN_BRICK" and cmd in ("f", "b"):
            # Keep ALIGN_BRICK distance micro-adjustments identical to calibrate_align:
            # speed comes from distance error to target, not raw distance profile.
            dist_error_mm = None
            if isinstance(offsets, dict):
                dist_signed_err = _coerce_float(offsets.get("dist"), None)
                if dist_signed_err is not None:
                    dist_error_mm = abs(float(dist_signed_err))
            if dist_error_mm is None:
                dist_error_mm = _coerce_float(mm_errors.get("dist"), 0.0) or 0.0
            dist_score_float = align_brick_dist_error_speed_score(float(dist_error_mm))
            speed_score = int(round(float(dist_score_float)))
        elif obj_name == "POSITION_BRICK" and cmd in ("f", "b"):
            base_score = _score_from_mm(mm_off, slow_mm, fast_mm)
            speed_score = align_steps_dist_speed_score(dist, base_score)
        elif obj_name in ("ALIGN_BRICK", "POSITION_BRICK") and cmd in ("l", "r") and x_axis_turn_error_mm is not None:
            dist_gate_error_mm = mm_errors.get("dist")
            speed_score = align_turn_speed_score_for_step(
                obj_name,
                x_axis_turn_error_mm,
                dist_gate_error_mm=dist_gate_error_mm,
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
    # Keep ALIGN_BRICK score dynamics aligned with calibrate_align; do not apply
    # the auto hard cap to this step.
    if obj_name != "ALIGN_BRICK":
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


def _success_gates_for_step(process_rules, step):
    obj_name = _step_name(step)
    step_cfg = (process_rules or {}).get(obj_name, {}) if isinstance(process_rules, dict) else {}
    success_gates = step_cfg.get("success_gates") if isinstance(step_cfg, dict) else {}
    if not isinstance(success_gates, dict):
        success_gates = {}
    return obj_name, success_gates


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
    x_axis_mm,
    y_axis_mm=None,
    dist_mm,
    visible=True,
    angle_deg=0.0,
    duration_s=0.05,
    previous_correction_type=None,
    avoid_correction_type=None,
):
    """
    Single source selector for ALIGN_BRICK next act (cmd + speed score).

    This function encapsulates the exact approach used by the best calibrate trials:
    - Direction source: `compute_alignment_decision(...)`
    - Distance score source: `align_brick_dist_error_speed_score(...)`
    - Turn score source: `align_brick_x_axis_one_shot_score(...)`
    """
    analytics = compute_alignment_decision(
        world=None,
        step="ALIGN_BRICK",
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

    _obj_name, success_gates = _success_gates_for_step(process_rules, "ALIGN_BRICK")
    x_stats = success_gates.get("xAxis_offset_abs") if isinstance(success_gates, dict) else {}
    if not isinstance(x_stats, dict):
        x_stats = {}
    y_stats = success_gates.get("yAxis_offset_abs") if isinstance(success_gates, dict) else {}
    if not isinstance(y_stats, dict):
        y_stats = {}
    d_stats = success_gates.get("dist") if isinstance(success_gates, dict) else {}
    if not isinstance(d_stats, dict):
        d_stats = {}

    x_target = _coerce_float(x_stats.get("target"), 0.0) or 0.0
    # `offset_y` is relative to the camera vertical center, so target remains 0 by default.
    y_target = _coerce_float(y_stats.get("target"), 0.0)
    if y_target is None:
        y_target = 0.0
    dist_target = _coerce_float(d_stats.get("target"), 0.0) or 0.0

    x_tol = abs(_coerce_float(x_stats.get("tol"), 0.0) or 0.0)
    y_tol = _coerce_float(y_stats.get("tol"), None)
    if y_tol is None:
        y_tol = x_tol
    y_tol = abs(float(y_tol) or 0.0)
    dist_tol = abs(_coerce_float(d_stats.get("tol"), 0.0) or 0.0)

    x_axis_val = _coerce_float(x_axis_mm, 0.0) or 0.0
    x_err_mm = float(x_axis_val - x_target)
    y_axis_val = _coerce_float(y_axis_mm, None)
    y_err_mm = None if y_axis_val is None else float(y_axis_val - y_target)
    dist_val = _coerce_float(dist_mm, None)
    if dist_val is None:
        dist_err_mm = float("inf")
    else:
        dist_err_mm = abs(float(dist_val - dist_target))

    def _gate_ratio(abs_err_mm, tol_mm):
        try:
            err = max(0.0, float(abs_err_mm) - float(max(0.0, tol_mm)))
        except (TypeError, ValueError):
            return 0.0
        denom = max(float(tol_mm), 1.0)
        return float(err) / float(denom)

    x_ratio = _gate_ratio(abs(x_err_mm), x_tol)
    y_ratio = _gate_ratio(abs(y_err_mm), y_tol) if y_err_mm is not None else 0.0
    d_ratio = _gate_ratio(dist_err_mm, dist_tol) if dist_val is not None and dist_tol > 0.0 else 0.0

    if prod_cmd not in ("f", "b", "l", "r"):
        prod_cmd = "l" if x_err_mm <= 0.0 else "r"
        if not worst_metric:
            worst_metric = "xAxis_offset_abs"

    candidates = {}

    if x_ratio > 0.0:
        candidates["x_axis"] = {
            "cmd": "l" if x_err_mm <= 0.0 else "r",
            "correction_type": "x_axis",
            "score": int(align_brick_x_axis_one_shot_score(x_err_mm)),
            "score_float": None,
            "reason": "x_axis_alignment",
            "worst_metric": "xAxis_offset_abs",
            "ratio": float(x_ratio),
        }

    if y_err_mm is not None and y_ratio > 0.0:
        candidates["y_axis"] = {
            "cmd": "d" if float(y_err_mm) > 0.0 else "u",
            "correction_type": "y_axis",
            "score": int(align_brick_x_axis_one_shot_score(float(y_err_mm))),
            "score_float": None,
            "reason": "y_axis_alignment",
            "worst_metric": "yAxis_offset_abs",
            "ratio": float(y_ratio),
        }

    if dist_val is not None and d_ratio > 0.0:
        if prod_cmd in ("f", "b"):
            dist_cmd = str(prod_cmd)
        else:
            dist_cmd = "f" if float(dist_val) > float(dist_target) else "b"
        score_float_dist = float(align_brick_dist_error_speed_score(dist_err_mm))
        candidates["distance"] = {
            "cmd": str(dist_cmd),
            "correction_type": "distance",
            "score": int(round(score_float_dist)),
            "score_float": score_float_dist,
            "reason": "distance_alignment",
            "worst_metric": "dist",
            "ratio": float(d_ratio),
        }

    # Preserve prior behavior for the default (non-rotation) first choice.
    # Vertical camera-center alignment for step 4 (lift axis):
    # Positive y offset means marker is below center, so move mast down (`d`) to
    # bring the marker upward in the image. Negative y offset uses mast up (`u`).
    use_y_axis_correction = bool(
        "y_axis" in candidates
        and prod_cmd not in ("f", "b")   # Keep distance safety priority when depth is the main issue.
        and y_ratio >= x_ratio
    )

    if use_y_axis_correction:
        chosen = dict(candidates["y_axis"])
    elif prod_cmd in ("f", "b") and "distance" in candidates:
        chosen = dict(candidates["distance"])
    elif "x_axis" in candidates:
        chosen = dict(candidates["x_axis"])
    elif "distance" in candidates:
        chosen = dict(candidates["distance"])
    elif "y_axis" in candidates:
        chosen = dict(candidates["y_axis"])
    else:
        chosen = {
            "cmd": str(prod_cmd),
            "correction_type": "x_axis",
            "score": int(align_brick_x_axis_one_shot_score(x_err_mm)),
            "score_float": None,
            "reason": "x_axis_alignment",
            "worst_metric": worst_metric or "xAxis_offset_abs",
            "ratio": 0.0,
        }

    prev_corr = str(previous_correction_type or "").strip().lower()
    avoid_corr = str(avoid_correction_type or "").strip().lower()

    def _best_alternative(excluded_types):
        viable = [
            dict(v)
            for k, v in candidates.items()
            if str(k) not in excluded_types and float(v.get("ratio", 0.0) or 0.0) > 0.0
        ]
        if not viable:
            return None
        viable.sort(
            key=lambda row: (
                float(row.get("ratio", 0.0) or 0.0),
                1.0 if str(row.get("correction_type")) == "y_axis" else 0.0,  # slight tie-breaker toward vertical gap rotation
            ),
            reverse=True,
        )
        return viable[0]

    rotation_override = False
    chosen_type = str(chosen.get("correction_type") or "").strip().lower()
    if avoid_corr and chosen_type == avoid_corr:
        alt = _best_alternative({avoid_corr})
        if alt is not None:
            chosen = alt
            chosen_type = str(chosen.get("correction_type") or "").strip().lower()
            rotation_override = True
    elif prev_corr and chosen_type == prev_corr:
        alt = _best_alternative({prev_corr})
        if alt is not None:
            chosen = alt
            chosen_type = str(chosen.get("correction_type") or "").strip().lower()
            rotation_override = True

    correction_type = str(chosen.get("correction_type"))
    score_float = chosen.get("score_float")
    score_int = int(chosen.get("score"))
    prod_cmd = str(chosen.get("cmd"))
    reason = str(chosen.get("reason"))
    worst_metric = chosen.get("worst_metric")

    # When learned rules are provided (auto-step path), prefer the score chosen
    # by the single-source analytics pipeline so calibration_profile scaling and
    # speed caps are honored for ALIGN_BRICK.
    if learned_rules is not None and correction_type != "y_axis" and str(analytics.get("cmd")) == str(prod_cmd):
        analytics_score = analytics.get("speed_score")
        if analytics_score is not None:
            try:
                score_int = int(normalize_speed_score(analytics_score))
            except (TypeError, ValueError):
                pass

    return {
        "cmd": str(prod_cmd),
        "correction_type": str(correction_type),
        "score": int(score_int),
        "score_float": score_float,
        "reason": str(reason),
        "worst_metric": worst_metric,
        "rotation_override": bool(rotation_override),
        "x_err_mm": float(x_err_mm),
        "y_err_mm": (None if y_err_mm is None else float(y_err_mm)),
        "dist_err_mm": float(dist_err_mm),
        "dist_target_mm": float(dist_target),
        "x_tol_mm": float(x_tol),
        "y_tol_mm": float(y_tol),
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
    x_target = _coerce_float(x_stats.get("target"), 0.0)
    x_tol = abs(_coerce_float(x_stats.get("tol"), 0.0) or 0.0)
    x_err = None if x_axis is None else float(x_axis - x_target)
    x_abs_err = None if x_err is None else abs(float(x_err))
    x_within_tol = bool(x_abs_err is not None and x_abs_err <= x_tol)

    step_key = _step_name(step)
    y_stats = success_gates.get("yAxis_offset_abs") if isinstance(success_gates, dict) else {}
    if not isinstance(y_stats, dict):
        y_stats = {}
    y_target = _coerce_float(y_stats.get("target"), 0.0)
    if y_target is None:
        y_target = 0.0
    y_tol = _coerce_float(y_stats.get("tol"), None)
    if y_tol is None:
        y_tol = x_tol
    y_tol = abs(float(y_tol) or 0.0)
    y_err = None if y_axis is None else float(y_axis - y_target)
    y_abs_err = None if y_err is None else abs(float(y_err))
    y_within_tol = bool(y_abs_err is not None and y_abs_err <= y_tol)
    y_required = bool(step_key == "ALIGN_BRICK")

    d_stats = success_gates.get("dist") if isinstance(success_gates, dict) else {}
    if not isinstance(d_stats, dict):
        d_stats = {}
    d_target = _coerce_float(d_stats.get("target"), 0.0)
    d_tol = abs(_coerce_float(d_stats.get("tol"), 0.0) or 0.0)
    dist_err = None if dist_val is None else abs(float(dist_val - d_target))
    dist_within_tol = bool(dist_err is not None and dist_err <= d_tol)

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
