import math
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List

from helper_next import METRIC_DIRECTIONS, _target_tol_ok, compute_alignment_analytics, metric_direction_for_step
import helper_gate_utils as gate_utils

START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
VISIBILITY_LOST_GRACE_S = 0.5
VISIBLE_FALSE_CONFIDENT_FRAMES_REQUIRED = 3
VISIBLE_FALSE_CONFIDENT_MAX_SAMPLES = 6
BRICK_SMOOTH_FRAMES = 3
BRICK_SMOOTH_OUTLIER_MM = 12.0
BRICK_SMOOTH_OUTLIER_DEG = 6.0
VISIBLE_FALSE_GRACE_S_BY_STEP = {
    "EXIT_WALL": 1.0,
}
STACK_VIS_LITE_CONSEC_FRAMES_TRUE = 1
STACK_VIS_LITE_CONSEC_FRAMES_FALSE = 1
STACK_VIS_LITE_CONSEC_FRAMES = max(
    int(STACK_VIS_LITE_CONSEC_FRAMES_TRUE),
    int(STACK_VIS_LITE_CONSEC_FRAMES_FALSE),
)
STACK_VIS_FULL_CONSEC_FRAMES_TRUE = 3
STACK_VIS_FULL_CONSEC_FRAMES_FALSE = 3
STACK_VIS_FULL_CONSEC_FRAMES = max(
    int(STACK_VIS_FULL_CONSEC_FRAMES_TRUE),
    int(STACK_VIS_FULL_CONSEC_FRAMES_FALSE),
)
STACK_VIS_FULL_MAJ_WINDOW_TRUE = 12
STACK_VIS_FULL_MAJ_REQUIRED_TRUE = 7
STACK_VIS_FULL_MAJ_WINDOW_FALSE = 12
STACK_VIS_FULL_MAJ_REQUIRED_FALSE = 7
STACK_VIS_FULL_MAJ_WINDOW = max(
    int(STACK_VIS_FULL_MAJ_WINDOW_TRUE),
    int(STACK_VIS_FULL_MAJ_WINDOW_FALSE),
)
STACK_VIS_FULL_MAJ_REQUIRED = max(
    int(STACK_VIS_FULL_MAJ_REQUIRED_TRUE),
    int(STACK_VIS_FULL_MAJ_REQUIRED_FALSE),
)

STEP_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
    "SCOOP": "SEAT_BRICK",
    "LIFT": "ELEVATE_BRICK",
    "SEAT": "SEAT_BRICK",
    "ELEVATE": "ELEVATE_BRICK",
    "PLACE": "RETREAT",
}

METRICS_BY_STEP = {
    "FIND_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "EXIT_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "FIND_BRICK": ("angle_abs", "xAxis_offset_abs", "yAxis_offset_abs", "dist", "visible"),
    "APPROACH_VECTOR_BRICK_SUPPLY": ("angle_abs", "xAxis_offset_abs", "x_axis", "y_axis", "dist", "visible"),
    "FIND_TOPMOST_BRICK": ("yAxis_offset_abs", "dist", "brick_above", "brick_below", "visible"),
    "FIND_TOPMOST_BRICK_WALL": ("yAxis_offset_abs", "dist", "brick_above", "brick_below", "visible"),
    "BRICK_LOCK": ("xAxis_offset_abs", "dist", "brick_above", "brick_below", "visible"),
    "BRICK_LOCK_WALL": ("xAxis_offset_abs", "dist", "brick_above", "brick_below", "visible"),
    "ALIGN_BRICK": ("xAxis_offset_abs", "yAxis_offset_abs", "dist", "visible"),
    "SEAT_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "SEAT_BRICK2": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "ELEVATE_BRICK": ("visible",),
    "FIND_WALL2": ("angle_abs", "xAxis_offset_abs", "yAxis_offset_abs", "dist", "visible"),
    "APPROACH_VECTOR_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "POSITION_BRICK": ("angle_abs", "xAxis_offset_abs", "yAxis_offset_abs", "dist", "brick_above", "visible"),
    "RETREAT": ("visible", "dist"),
}

SCOOP_LIKE_STEPS = {"SEAT_BRICK", "SEAT_BRICK2"}

X_AXIS_GATE_METRICS = {"xAxis_offset_abs", "xAxis_offset", "x_axis"}
Y_AXIS_GATE_METRICS = {"yAxis_offset_abs", "yAxis_offset", "y_axis"}
VISIBILITY_REQUIRED_METRICS = {
    "angle_abs",
    "dist",
    "confidence",
    *X_AXIS_GATE_METRICS,
    *Y_AXIS_GATE_METRICS,
}

DEFAULT_ARUCO_MARKER_SIZE_MM = 20.0
BRICK_HEIGHT_MM = 44.0


def brick_height_mm():
    return float(BRICK_HEIGHT_MM)


def brick_count_to_height_mm(brick_count):
    try:
        count = int(round(float(brick_count)))
    except (TypeError, ValueError):
        return None
    if count < 0:
        count = 0
    return float(count) * float(BRICK_HEIGHT_MM)


def height_mm_to_brick_count(height_mm, *, minimum=0, maximum=None):
    try:
        mm_val = float(height_mm)
    except (TypeError, ValueError):
        return None
    mm_val = max(0.0, float(mm_val))
    brick_mm = max(1e-6, float(BRICK_HEIGHT_MM))
    count = int(round(mm_val / brick_mm))
    try:
        min_count = int(round(float(minimum)))
    except (TypeError, ValueError):
        min_count = 0
    count = max(int(min_count), int(count))
    if maximum is not None:
        try:
            max_count = int(round(float(maximum)))
        except (TypeError, ValueError):
            max_count = None
        if max_count is not None:
            count = min(int(count), int(max_count))
    return int(count)


def _crosshair_overlap_half_extents_mm():
    marker_size = DEFAULT_ARUCO_MARKER_SIZE_MM
    try:
        env_size = os.getenv("ARUCO_MARKER_SIZE_MM", "")
        if str(env_size).strip():
            parsed = float(env_size)
            if parsed > 0.0:
                marker_size = parsed
    except (TypeError, ValueError):
        marker_size = DEFAULT_ARUCO_MARKER_SIZE_MM
    half = float(marker_size) / 2.0
    return half, half


_CROSSHAIR_HALF_WIDTH_MM, _CROSSHAIR_HALF_HEIGHT_MM = _crosshair_overlap_half_extents_mm()


def _coerce_crosshair_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _in_crosshairs_center_from_step_cfg(step_cfg):
    if not isinstance(step_cfg, dict):
        return 0.0, 0.0
    center_cfg = step_cfg.get("in_crosshairs_center_mm")
    if not isinstance(center_cfg, dict):
        exception_cfg = step_cfg.get("topmost_crosshair_exception")
        if isinstance(exception_cfg, dict):
            center_cfg = exception_cfg.get("in_crosshairs_center_mm")
    if not isinstance(center_cfg, dict):
        return 0.0, 0.0

    x_raw = center_cfg.get("x")
    if x_raw is None:
        x_raw = center_cfg.get("center_x_mm")
    if x_raw is None:
        x_raw = center_cfg.get("target_x_mm")

    y_raw = center_cfg.get("y")
    if y_raw is None:
        y_raw = center_cfg.get("center_y_mm")
    if y_raw is None:
        y_raw = center_cfg.get("target_y_mm")

    return _coerce_crosshair_float(x_raw, 0.0), _coerce_crosshair_float(y_raw, 0.0)


def in_crosshairs_center_mm(world=None, step=None, process_rules=None):
    step_ref = step if step is not None else getattr(world, "step_state", None)
    if step_ref is None:
        return 0.0, 0.0
    step_key = _step_name(step_ref)
    rules = process_rules
    if not isinstance(rules, dict):
        rules = getattr(world, "process_rules", None)
    if not isinstance(rules, dict):
        return 0.0, 0.0
    step_cfg = rules.get(step_key, {}) if step_key else {}
    return _in_crosshairs_center_from_step_cfg(step_cfg)


def _compute_in_crosshairs(found, x_axis_mm, y_axis_mm, *, center_x_mm=0.0, center_y_mm=0.0):
    if not bool(found):
        return False
    try:
        x_val = float(x_axis_mm)
        y_val = float(y_axis_mm)
    except (TypeError, ValueError):
        return False
    center_x = _coerce_crosshair_float(center_x_mm, 0.0)
    center_y = _coerce_crosshair_float(center_y_mm, 0.0)
    marker_left = x_val - _CROSSHAIR_HALF_WIDTH_MM
    marker_right = x_val + _CROSSHAIR_HALF_WIDTH_MM
    marker_top = y_val - _CROSSHAIR_HALF_HEIGHT_MM
    marker_bottom = y_val + _CROSSHAIR_HALF_HEIGHT_MM
    x_overlaps = marker_left <= center_x <= marker_right
    y_overlaps = marker_top <= center_y <= marker_bottom
    return bool(x_overlaps and y_overlaps)


def compute_in_crosshairs_for_step(world, found, x_axis_mm, y_axis_mm, *, step=None, process_rules=None):
    center_x_mm, center_y_mm = in_crosshairs_center_mm(
        world,
        step=step,
        process_rules=process_rules,
    )
    return _compute_in_crosshairs(
        found,
        x_axis_mm,
        y_axis_mm,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
    )


def _average_brick_frames(frames):
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def majority(values):
        return sum(1 for v in values if v) >= (len(values) / 2.0)

    return {
        "found": majority([f["found"] for f in frames]),
        "dist": mean([f["dist"] for f in frames]),
        "angle": mean([f["angle"] for f in frames]),
        "offset_x": mean([f["offset_x"] for f in frames]),
        "offset_y": mean([f.get("offset_y", f.get("cam_h", 0.0)) for f in frames]),
        "conf": mean([f["conf"] for f in frames]),
        "cam_h": mean([f["cam_h"] for f in frames]),
        "brick_above": majority([f["brick_above"] for f in frames]),
        "brick_below": majority([f["brick_below"] for f in frames]),
    }


def _filtered_brick_frame_average(frames, min_frames=BRICK_SMOOTH_FRAMES):
    if len(frames) < min_frames:
        return None, False, None

    dist_vals = [f["dist"] for f in frames]
    offset_vals = [f["offset_x"] for f in frames]
    offset_y_vals = [f.get("offset_y", f.get("cam_h", 0.0)) for f in frames]
    angle_vals = [f["angle"] for f in frames]
    med_dist = percentile(dist_vals, 0.5)
    med_offset = percentile(offset_vals, 0.5)
    med_offset_y = percentile(offset_y_vals, 0.5)
    med_angle = percentile(angle_vals, 0.5)

    keep = []
    for frame in frames:
        if not frame["found"]:
            continue
        if (
            abs(frame["dist"] - med_dist) > BRICK_SMOOTH_OUTLIER_MM
            or abs(frame["offset_x"] - med_offset) > BRICK_SMOOTH_OUTLIER_MM
            or abs(float(frame.get("offset_y", frame.get("cam_h", 0.0)) or 0.0) - med_offset_y) > BRICK_SMOOTH_OUTLIER_MM
            or abs(frame["angle"] - med_angle) > BRICK_SMOOTH_OUTLIER_DEG
        ):
            continue
        keep.append(frame)

    if len(keep) < min_frames:
        return None, True, f"only {len(keep)}/{min_frames} frames agreed (inconsistent)"
    return _average_brick_frames(keep), True, None


def smoothed_brick_snapshot(world):
    brick = world.brick or {}
    buffer = getattr(world, "_brick_frame_buffer", None)
    if not buffer:
        return brick
    avg, _, _ = _filtered_brick_frame_average(buffer)
    if avg is None:
        return brick
    return {
        "visible": bool(avg.get("found")),
        "dist": float(avg.get("dist", 0.0)),
        "angle": float(avg.get("angle", 0.0)),
        "confidence": float(avg.get("conf", 0.0)),
        "offset_x": float(avg.get("offset_x", 0.0)),
        "x_axis": float(avg.get("offset_x", 0.0)),
        "offset_y": float(avg.get("offset_y", avg.get("cam_h", 0.0))),
        "y_axis": float(avg.get("offset_y", avg.get("cam_h", 0.0))),
        # Use the confirmed stack flags from the world state so higher-level gate checks
        # do not assert true/false until the stack-specific lite+full checks confirm.
        "brickAbove": (world.brick or {}).get("brickAbove"),
        "brickBelow": (world.brick or {}).get("brickBelow"),
        "inCrosshairs": (world.brick or {}).get("inCrosshairs"),
    }


def _recent_raw_visible(world, *, min_confidence=None, required_hits=1, max_samples=BRICK_SMOOTH_FRAMES):
    """
    Fallback visibility probe from the raw camera frame buffer.

    Start gates like `visible=true` should not fail forever just because the smoothing
    filter rejects pose values as inconsistent while the marker is plainly visible in
    the live stream. This helper checks recent raw detector hits without trusting raw
    pose metrics for numeric gates.
    """
    buffer = getattr(world, "_brick_frame_buffer", None)
    if not buffer:
        return False
    try:
        sample_limit = max(1, int(max_samples or 1))
    except (TypeError, ValueError):
        sample_limit = max(1, int(BRICK_SMOOTH_FRAMES))
    try:
        hits_needed = max(1, int(required_hits or 1))
    except (TypeError, ValueError):
        hits_needed = 1
    seen = 0
    hits = 0
    for frame in reversed(list(buffer)):
        if not isinstance(frame, dict):
            continue
        seen += 1
        found = bool(frame.get("found"))
        if found and min_confidence is not None:
            try:
                conf_val = float(frame.get("conf", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf_val = 0.0
            found = conf_val >= float(min_confidence)
        if found:
            hits += 1
            if hits >= hits_needed:
                return True
        if seen >= sample_limit:
            break
    return False


@dataclass
class GateCheck:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def reason_str(self):
        return "; ".join(self.reasons) if self.reasons else ""


def combine_gate_checks(*checks):
    ok = True
    reasons = []
    for check in checks:
        if check is None:
            continue
        if not check.ok:
            ok = False
            reasons.extend(check.reasons)
    return GateCheck(ok=ok, reasons=reasons)


def _step_name(step):
    if hasattr(step, "value"):
        name = step.value
    else:
        name = str(step)
    key = name.strip().upper()
    normalized = STEP_ALIASES.get(key, key)
    for suffix in ("_ALIGN_SETTLE", "_ALIGN", "_SETTLE"):
        if not str(normalized).endswith(suffix):
            continue
        candidate = str(normalized)[: -len(suffix)].strip().upper()
        if not candidate:
            continue
        normalized = STEP_ALIASES.get(candidate, candidate)
        break
    return normalized


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pct = max(0.0, min(1.0, pct))
    idx = int(round(pct * (len(values) - 1)))
    return values[idx]


def metric_value(brick, metric):
    if metric == "angle_abs":
        return abs(brick.get("angle", 0.0))
    if metric in X_AXIS_GATE_METRICS:
        return brick.get("x_axis", brick.get("offset_x", 0.0))
    if metric in Y_AXIS_GATE_METRICS:
        return brick.get("y_axis", brick.get("offset_y", 0.0))
    if metric == "dist":
        return brick.get("dist", 0.0)
    if metric == "visible":
        return 1.0 if brick.get("visible") else 0.0
    if metric == "confidence":
        return brick.get("confidence", 0.0)
    if metric in ("brick_above", "brickAbove"):
        if "brickAbove" in brick:
            value = brick.get("brickAbove")
            return None if value is None else bool(value)
        if "brick_above" in brick:
            value = brick.get("brick_above")
            return None if value is None else bool(value)
        return None
    if metric in ("brick_below", "brickBelow"):
        if "brickBelow" in brick:
            value = brick.get("brickBelow")
            return None if value is None else bool(value)
        if "brick_below" in brick:
            value = brick.get("brick_below")
            return None if value is None else bool(value)
        return None
    if metric in ("inCrosshairs", "in_crosshairs"):
        if "inCrosshairs" in brick:
            value = brick.get("inCrosshairs")
            return None if value is None else bool(value)
        if "in_crosshairs" in brick:
            value = brick.get("in_crosshairs")
            return None if value is None else bool(value)
        return _compute_in_crosshairs(
            bool(brick.get("visible")),
            brick.get("x_axis", brick.get("offset_x")),
            brick.get("y_axis", brick.get("offset_y")),
        )
    return None


def _bool_metric_gate_matches(value, stats):
    if value is None:
        return None
    bool_val = bool(value)
    if not isinstance(stats, dict):
        return None
    min_val = stats.get("min")
    max_val = stats.get("max")
    if isinstance(min_val, bool):
        return bool_val == min_val
    if isinstance(max_val, bool):
        return bool_val == max_val
    numeric = 1.0 if bool_val else 0.0
    ok = _target_tol_ok(numeric, stats, "band")
    if ok is not None:
        return bool(ok)
    try:
        min_num = float(min_val) if min_val is not None else None
    except (TypeError, ValueError):
        min_num = None
    try:
        max_num = float(max_val) if max_val is not None else None
    except (TypeError, ValueError):
        max_num = None
    if min_num is not None and numeric < min_num:
        return False
    if max_num is not None and numeric > max_num:
        return False
    if min_num is not None or max_num is not None:
        return True
    return None


def _stack_tracker_new(*, raw_bool=None):
    if raw_bool is True:
        consecutive_required = int(STACK_VIS_FULL_CONSEC_FRAMES_TRUE)
        majority_window = int(STACK_VIS_FULL_MAJ_WINDOW_TRUE)
        majority_required = int(STACK_VIS_FULL_MAJ_REQUIRED_TRUE)
    elif raw_bool is False:
        consecutive_required = int(STACK_VIS_FULL_CONSEC_FRAMES_FALSE)
        majority_window = int(STACK_VIS_FULL_MAJ_WINDOW_FALSE)
        majority_required = int(STACK_VIS_FULL_MAJ_REQUIRED_FALSE)
    else:
        consecutive_required = int(STACK_VIS_FULL_CONSEC_FRAMES)
        majority_window = int(STACK_VIS_FULL_MAJ_WINDOW)
        majority_required = int(STACK_VIS_FULL_MAJ_REQUIRED)
    return gate_utils.SuccessGateTracker(
        consecutive_required=consecutive_required,
        majority_window=majority_window,
        majority_required=majority_required,
    )


def _stack_tracker_snapshot(tracker):
    window_vals = list(getattr(tracker, "window", []) or [])
    window_pass = int(sum(1 for ok in window_vals if ok))
    window_size = int(len(window_vals))
    streak = int(getattr(tracker, "consecutive", 0) or 0)
    need = max(1, int(getattr(tracker, "consecutive_required", 1) or 1))
    window_total = max(1, int(getattr(tracker, "majority_window", 1) or 1))
    window_need = max(1, int(getattr(tracker, "majority_required", 1) or 1))
    truth_ok = bool(streak >= need or window_pass >= window_need)
    return {
        "checks": int(getattr(tracker, "total_checks", 0) or 0),
        "pass": int(getattr(tracker, "total_pass", 0) or 0),
        "streak": streak,
        "need": need,
        "window_pass": window_pass,
        "window_size": window_size,
        "window_total": window_total,
        "window_need": window_need,
        "truth_ok": truth_ok,
    }


def _stack_gate_state(world):
    state = getattr(world, "_stack_visibility_gate", None)
    if isinstance(state, dict):
        return state
    state = {
        "visible": False,
        "above": {
            "history": deque(maxlen=int(STACK_VIS_LITE_CONSEC_FRAMES)),
            "tracker_true": _stack_tracker_new(raw_bool=True),
            "tracker_false": _stack_tracker_new(raw_bool=False),
        },
        "below": {
            "history": deque(maxlen=int(STACK_VIS_LITE_CONSEC_FRAMES)),
            "tracker_true": _stack_tracker_new(raw_bool=True),
            "tracker_false": _stack_tracker_new(raw_bool=False),
        },
    }
    world._stack_visibility_gate = state
    return state


def _stack_gate_metric_reset(metric_state):
    metric_state["history"].clear()
    metric_state["tracker_true"] = _stack_tracker_new(raw_bool=True)
    metric_state["tracker_false"] = _stack_tracker_new(raw_bool=False)
    metric_state["raw"] = None
    metric_state["lite_count"] = 0
    metric_state["lite_need"] = int(STACK_VIS_LITE_CONSEC_FRAMES_FALSE)
    metric_state["lite_candidate"] = None
    metric_state["lite_ok"] = False
    metric_state["full_true"] = _stack_tracker_snapshot(metric_state["tracker_true"])
    metric_state["full_false"] = _stack_tracker_snapshot(metric_state["tracker_false"])
    metric_state["full_candidate"] = None
    metric_state["full_active"] = None
    metric_state["asserted"] = None
    metric_state["truth"] = None


def _stack_gate_metric_update(metric_state, raw_value, *, visible):
    if not visible or raw_value is None:
        _stack_gate_metric_reset(metric_state)
        return None

    raw_bool = bool(raw_value)
    metric_state["raw"] = raw_bool
    history = metric_state["history"]
    history.append(raw_bool)
    # Keep the first stage as a pass-through (1 frame), then use a single simple
    # confidence rule in the full tracker: assert YES/NO after either:
    # - 3 consecutive raw frames for that value, or
    # - a 7/12 majority in the rolling window.
    lite_need = int(
        STACK_VIS_LITE_CONSEC_FRAMES_TRUE if raw_bool else STACK_VIS_LITE_CONSEC_FRAMES_FALSE
    )
    lite_count = 0
    for value in reversed(list(history)):
        if bool(value) is raw_bool:
            lite_count += 1
        else:
            break
    lite_count = min(int(lite_need), int(lite_count))
    lite_candidate = None
    if len(history) >= lite_need and all(bool(v) is raw_bool for v in history):
        lite_candidate = raw_bool
    lite_ok = lite_candidate is not None

    # Only start the heavier full truth tracker after the lite consecutive gate has
    # already passed. This keeps the full gate from churning on every noisy frame.
    if lite_ok:
        metric_state["tracker_true"].update(raw_bool is True)
        metric_state["tracker_false"].update(raw_bool is False)
    else:
        metric_state["tracker_true"] = _stack_tracker_new()
        metric_state["tracker_false"] = _stack_tracker_new()
    full_true = _stack_tracker_snapshot(metric_state["tracker_true"])
    full_false = _stack_tracker_snapshot(metric_state["tracker_false"])

    asserted = None
    full_candidate = None
    full_active = None
    if lite_candidate is True and bool(full_true.get("truth_ok")):
        asserted = True
        full_candidate = True
        full_active = full_true
    elif lite_candidate is False and bool(full_false.get("truth_ok")):
        asserted = False
        full_candidate = False
        full_active = full_false
    elif lite_candidate is True:
        full_candidate = True
        full_active = full_true
    elif lite_candidate is False:
        full_candidate = False
        full_active = full_false

    metric_state["lite_count"] = int(lite_count)
    metric_state["lite_need"] = int(lite_need)
    metric_state["lite_candidate"] = lite_candidate
    metric_state["lite_ok"] = bool(lite_ok)
    metric_state["full_true"] = full_true
    metric_state["full_false"] = full_false
    metric_state["full_candidate"] = full_candidate
    metric_state["full_active"] = full_active
    metric_state["asserted"] = asserted
    metric_state["truth"] = asserted
    return asserted


def _stack_gate_log_signature(status):
    if not isinstance(status, dict):
        return None
    sig = [bool(status.get("visible", False))]
    for key in ("above", "below"):
        row = status.get(key) or {}
        full_active = row.get("full_active") if isinstance(row.get("full_active"), dict) else {}
        sig.extend(
            [
                row.get("raw"),
                row.get("lite_count"),
                row.get("lite_candidate"),
                row.get("asserted"),
                full_active.get("streak") if full_active else None,
                full_active.get("window_pass") if full_active else None,
                full_active.get("truth_ok") if full_active else None,
            ]
        )
    return tuple(sig)


def _stack_gate_row_text(label, row):
    if not isinstance(row, dict):
        return f"{label}=unknown"
    raw = row.get("raw")
    asserted = row.get("asserted")
    lite_count = int(row.get("lite_count", 0) or 0)
    lite_need = max(1, int(row.get("lite_need", STACK_VIS_LITE_CONSEC_FRAMES) or STACK_VIS_LITE_CONSEC_FRAMES))
    lite_candidate = row.get("lite_candidate")
    full_active = row.get("full_active") if isinstance(row.get("full_active"), dict) else None

    raw_txt = "-" if raw is None else ("yes" if bool(raw) else "no")
    if asserted is True:
        out_txt = "YES"
    elif asserted is False:
        out_txt = "NO"
    else:
        out_txt = "WAIT"
    lite_txt = "wait"
    if lite_candidate is True:
        lite_txt = "yes"
    elif lite_candidate is False:
        lite_txt = "no"

    if isinstance(full_active, dict):
        full_txt = (
            f"c{int(full_active.get('streak', 0) or 0)}/{int(full_active.get('need', 1) or 1)} "
            f"m{int(full_active.get('window_pass', 0) or 0)}/{int(full_active.get('window_total', 1) or 1)}"
        )
    else:
        full_txt = "c0/? m0/?"
    return (
        f"{label}: raw={raw_txt} lite={lite_txt}({lite_count}/{lite_need}) "
        f"full={full_txt} => {out_txt}"
    )


def _stack_top_candidate_log_line(status):
    if not isinstance(status, dict):
        return "[STACK GATECHECK] Top-brick candidate? status unavailable."

    visible = bool(status.get("visible", False))
    above = status.get("above") if isinstance(status.get("above"), dict) else {}
    below = status.get("below") if isinstance(status.get("below"), dict) else {}

    above_raw = above.get("raw")
    below_raw = below.get("raw")
    above_txt = "?" if above_raw is None else ("true" if bool(above_raw) else "false")
    below_txt = "?" if below_raw is None else ("true" if bool(below_raw) else "false")

    # Candidate pattern means we're looking at the uppermost brick in a stack.
    raw_candidate = bool(visible and above_raw is False and below_raw is True)

    above_lite_count = int(above.get("lite_count", 0) or 0)
    below_lite_count = int(below.get("lite_count", 0) or 0)
    above_lite_need = max(1, int(above.get("lite_need", STACK_VIS_LITE_CONSEC_FRAMES) or STACK_VIS_LITE_CONSEC_FRAMES))
    below_lite_need = max(1, int(below.get("lite_need", STACK_VIS_LITE_CONSEC_FRAMES) or STACK_VIS_LITE_CONSEC_FRAMES))
    lite_need = max(above_lite_need, below_lite_need)
    lite_consec_seen = min(above_lite_count, below_lite_count) if raw_candidate else 0

    above_false = above.get("full_false") if isinstance(above.get("full_false"), dict) else {}
    below_true = below.get("full_true") if isinstance(below.get("full_true"), dict) else {}
    full_consec_seen = min(
        int(above_false.get("streak", 0) or 0),
        int(below_true.get("streak", 0) or 0),
    )
    full_consec_need = max(
        1,
        int(above_false.get("need", STACK_VIS_FULL_CONSEC_FRAMES) or STACK_VIS_FULL_CONSEC_FRAMES),
        int(below_true.get("need", STACK_VIS_FULL_CONSEC_FRAMES) or STACK_VIS_FULL_CONSEC_FRAMES),
    )
    full_maj_seen = min(
        int(above_false.get("window_pass", 0) or 0),
        int(below_true.get("window_pass", 0) or 0),
    )
    full_maj_den = max(
        1,
        int(above_false.get("window_total", STACK_VIS_FULL_MAJ_WINDOW) or STACK_VIS_FULL_MAJ_WINDOW),
        int(below_true.get("window_total", STACK_VIS_FULL_MAJ_WINDOW) or STACK_VIS_FULL_MAJ_WINDOW),
    )

    confirmed_top_candidate = bool(visible and above.get("asserted") is False and below.get("asserted") is True)
    if not visible:
        lead = "Top-brick candidate pending"
    elif raw_candidate:
        lead = "Top-brick candidate!"
    else:
        lead = "Not top-brick candidate yet."

    verdict = "CONFIRMED" if confirmed_top_candidate else "WAIT"
    return (
        "[STACK GATECHECK] "
        f"{lead} above={above_txt}, below={below_txt}. "
        f"Lite seen {int(lite_consec_seen)}/{int(lite_need)} conseq. "
        f"Full seen {int(full_consec_seen)}/{int(full_consec_need)} conseq. "
        f"Seen {int(full_maj_seen)}/{int(full_maj_den)} maj. "
        f"=> {verdict}"
    )


def _maybe_log_stack_gatecheck(world, visible):
    return


def update_stack_flags_from_raw(world, found, brick_above=False, brick_below=False):
    if world is None:
        return None, None
    if not isinstance(getattr(world, "brick", None), dict):
        world.brick = {}
    stack_state = _stack_gate_state(world)
    if not bool(found):
        stack_state["visible"] = False
        _stack_gate_metric_update(stack_state["above"], None, visible=False)
        _stack_gate_metric_update(stack_state["below"], None, visible=False)
        confirmed_above = None
        confirmed_below = None
    else:
        stack_state["visible"] = True
        confirmed_above = _stack_gate_metric_update(
            stack_state["above"],
            bool(brick_above),
            visible=True,
        )
        confirmed_below = _stack_gate_metric_update(
            stack_state["below"],
            bool(brick_below),
            visible=True,
        )
    world.brick["brickAbove"] = confirmed_above
    world.brick["brickBelow"] = confirmed_below
    _maybe_log_stack_gatecheck(world, bool(found))
    return confirmed_above, confirmed_below


def _effective_visible(world, visible, grace_s=VISIBILITY_LOST_GRACE_S):
    if visible:
        return True
    last_seen = getattr(world, "last_visible_time", None)
    if last_seen is None:
        return False
    return (time.time() - last_seen) <= grace_s


def _recent_confident_visible_hits(
    world,
    *,
    min_confidence=gate_utils.VISIBLE_FALSE_FAIL_CONFIDENCE_MIN,
    required_frames=VISIBLE_FALSE_CONFIDENT_FRAMES_REQUIRED,
    max_samples=VISIBLE_FALSE_CONFIDENT_MAX_SAMPLES,
):
    try:
        required = max(1, int(required_frames or 1))
    except (TypeError, ValueError):
        required = int(VISIBLE_FALSE_CONFIDENT_FRAMES_REQUIRED)
    try:
        scan_limit = max(required, int(max_samples or required))
    except (TypeError, ValueError):
        scan_limit = max(required, int(VISIBLE_FALSE_CONFIDENT_MAX_SAMPLES))

    raw_history = getattr(world, "_raw_brick_visibility_history", None)
    source = raw_history if isinstance(raw_history, list) and raw_history else getattr(world, "_brick_frame_buffer", None)
    if not source:
        return False

    hits = 0
    scanned = 0
    for frame in reversed(list(source)):
        if scanned >= scan_limit:
            break
        if not isinstance(frame, dict):
            continue
        scanned += 1

        found_raw = frame.get("found")
        if found_raw is None:
            found_raw = frame.get("visible")
        found = bool(found_raw)

        conf_raw = frame.get("conf")
        if conf_raw is None:
            conf_raw = frame.get("confidence")
        try:
            conf_val = float(conf_raw or 0.0)
        except (TypeError, ValueError):
            conf_val = 0.0

        if found and conf_val >= float(min_confidence):
            hits += 1
            if hits >= required:
                return True
            continue
    return False


def visible_false_gate_confident_recent(world, stats):
    if gate_utils.bool_gate_target(stats) is not False:
        return False
    return _recent_confident_visible_hits(world)


def metric_status(value, success_stats, failure_stats, direction):
    if success_stats is None or direction is None:
        return "unknown"
    if direction == "low":
        success_max = success_stats.get("max")
        failure_max = failure_stats.get("max") if failure_stats else None
        if success_max is not None and value <= success_max:
            return "success"
        if failure_max is not None and value >= failure_max:
            return "fail"
        return "correct"
    if direction == "high":
        success_min = success_stats.get("min")
        failure_max = failure_stats.get("max") if failure_stats else None
        if success_min is not None and value >= success_min:
            return "success"
        if failure_max is not None and value <= failure_max:
            return "fail"
        return "correct"

    success_min = success_stats.get("min")
    success_max = success_stats.get("max")
    if success_min is not None and success_max is not None and success_min <= value <= success_max:
        return "success"
    if failure_stats:
        fail_min = failure_stats.get("min")
        fail_max = failure_stats.get("max")
        if fail_min is not None and fail_max is not None and (value < fail_min or value > fail_max):
            return "fail"
    return "correct"


def success_metric_bounds(stats, direction):
    if not isinstance(stats, dict) or direction is None:
        return None, None
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if direction == "low":
            return None, target + tol
        if direction == "high":
            return target - tol, None
        return target - tol, target + tol
    return stats.get("min"), stats.get("max")


def compute_brick_world_xy(world, dist, angle_deg):
    heading = math.radians(world.theta + angle_deg)
    return (
        world.x + (dist * math.cos(heading)),
        world.y + (dist * math.sin(heading)),
    )


def get_scoop_corridor_limits(world, dist):
    corridor = (
        world.learned_rules.get("SEAT_BRICK", {}).get("corridor")
        or world.learned_rules.get("SCOOP", {}).get("corridor")
    )
    if not corridor or dist is None:
        return None
    for row in corridor:
        dist_min = row.get("dist_min", 0)
        dist_max = row.get("dist_max", 0)
        if dist_min <= dist < dist_max:
            return row
    if dist < corridor[0].get("dist_min", 0):
        return corridor[0]
    return corridor[-1]


def build_envelope(process_rules, learned_rules, step):
    obj_name = _step_name(step)
    process = (process_rules or {}).get(obj_name, {})
    learned = learned_rules.get(obj_name, {}) if learned_rules else {}
    learned_gates = learned.get("gates", {})
    success = process.get("success_gates") or learned_gates.get("success", {}).get("metrics", {})
    failure = process.get("fail_gates") or learned_gates.get("failure", {}).get("metrics", {})
    return {"success": success, "failure": failure}


def success_gate_bounds(process_rules, learned_rules, step):
    envelope = build_envelope(process_rules or {}, learned_rules or {}, step)
    success_metrics = envelope.get("success") or {}
    bounds = {}
    for metric, stats in success_metrics.items():
        direction = metric_direction_for_step(metric, step)
        min_val, max_val = success_metric_bounds(stats, direction)
        bounds[metric] = {"min": min_val, "max": max_val}
    return bounds


def stream_status_lines(world, step, process_rules=None, learned_rules=None):
    obj_name = _step_name(step)
    analytics = compute_brick_analytics(
        world,
        process_rules or {},
        learned_rules or {},
        obj_name,
        duration_s=getattr(world, "_last_dt", 0.05),
    )
    gate_progress = analytics.get("gate_progress") or []
    pct = None
    for name, val in gate_progress:
        if _step_name(name) == obj_name:
            pct = val
            break
    if pct is None:
        success_check = evaluate_success_gates(world, obj_name, learned_rules or {}, process_rules or {})
        pct = 100.0 if success_check.ok else 0.0
    pct_display = int(max(0.0, min(100.0, pct)))
    suggestion = getattr(world, "_last_action_line", None) or analytics.get("suggestion") or "HOLD"
    return [
        f"OBJ: {obj_name}",
        f"SUCCESS: {pct_display}%",
        f"ACT: {suggestion}",
    ]


def compute_brick_analytics(world, process_rules, learned_rules, step, duration_s=0.05):
    process_rules = process_rules or {}
    align = compute_alignment_analytics(world, process_rules, learned_rules, step, duration_s=duration_s)
    gate_status = []
    gate_progress = []
    suggestion = None

    for obj_name, data in process_rules.items():
        if not isinstance(data, dict):
            continue
        success_gates = data.get("success_gates") or {}
        if not success_gates:
            continue
        progress_values = []
        ok = True
        for metric, stats in success_gates.items():
            value = metric_value(world.brick or {}, metric)
            if value is None or not isinstance(stats, dict):
                ok = False
                continue
            if isinstance(value, bool):
                min_val = stats.get("min")
                max_val = stats.get("max")
                if min_val is not None:
                    ok = ok and (value is bool(min_val))
                    progress_values.append(1.0 if value is bool(min_val) else 0.0)
                    continue
                if max_val is not None:
                    ok = ok and (value is bool(max_val))
                    progress_values.append(1.0 if value is bool(max_val) else 0.0)
                    continue
                ok = False
                continue
            direction = metric_direction_for_step(metric, obj_name, process_rules=process_rules)
            if direction is None:
                ok = False
                continue

            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                err = abs(value - target)
                ok = ok and (err <= tol)
                if tol > 0:
                    if err <= tol:
                        progress_values.append(1.0)
                    else:
                        progress_values.append(max(0.0, 1.0 - (err - tol) / tol))
                else:
                    progress_values.append(1.0 if err == 0 else 0.0)
                continue

            min_val = stats.get("min")
            max_val = stats.get("max")
            if min_val is not None and value < min_val:
                ok = False
            if max_val is not None and value > max_val:
                ok = False
            if min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    progress_values.append(1.0)
                else:
                    span = max(1e-3, max_val - min_val)
                    if value < min_val:
                        progress_values.append(max(0.0, 1.0 - (min_val - value) / span))
                    else:
                        progress_values.append(max(0.0, 1.0 - (value - max_val) / span))
            elif min_val is not None:
                progress_values.append(1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3)))
            elif max_val is not None:
                progress_values.append(1.0 if value <= max_val else max(0.0, 1.0 - (value - max_val) / max(max_val, 1e-3)))

        if obj_name == _step_name(step) and align.get("progress") is not None:
            gate_progress.append((obj_name, align.get("progress") * 100))
        elif progress_values:
            gate_progress.append((obj_name, sum(progress_values) / len(progress_values) * 100))
        if ok and progress_values:
            gate_status.append(obj_name)

    cmd = align.get("cmd")
    if cmd is not None:
        speed = align.get("speed") or 0.0
        speed_score = align.get("speed_score")
        suffix = f" {int(speed_score)}%" if speed_score is not None else ""
        suggestion = f"{cmd.upper()}{suffix}"

    return {
        "gate_status": gate_status,
        "gate_progress": gate_progress,
        "suggestion": suggestion,
        "align": align,
        "highlight_metric": align.get("worst_metric"),
    }


def update_from_motion(world, event, delta):
    return


def update_from_vision(world, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
    # `cam_h` is the marker's vertical offset relative to the camera center (same
    # camera-space axis used for lift calibration). Alias it into brick telemetry
    # as `offset_y` / `y_axis` for recording and step-4 alignment.
    offset_y = float(cam_h)
    world.brick["visible"] = bool(found)
    world.brick["dist"] = float(dist)
    world.brick["angle"] = float(angle)
    world.brick["confidence"] = float(conf)
    world.brick["offset_x"] = float(offset_x)
    world.brick["x_axis"] = float(offset_x)
    world.brick["offset_y"] = float(offset_y)
    world.brick["y_axis"] = float(offset_y)
    world.brick["inCrosshairs"] = compute_in_crosshairs_for_step(
        world,
        found,
        offset_x,
        offset_y,
        step=getattr(world, "step_state", None),
        process_rules=getattr(world, "process_rules", None),
    )
    update_stack_flags_from_raw(world, found, brick_above, brick_below)
    if found:
        world.last_visible_time = time.time()
        world.last_seen_angle = float(angle)
        world.last_seen_offset_x = float(offset_x)
        world.last_seen_x_axis = float(offset_x)
        world.last_seen_offset_y = float(offset_y)
        world.last_seen_y_axis = float(offset_y)
        world.last_seen_dist = float(dist)
        world.last_seen_confidence = float(conf)

    if _step_name(world.step_state) in SCOOP_LIKE_STEPS:
        world.scoop_forward_preferred = True
        world.scoop_desired_offset_x = 0.0
        world.scoop_lateral_drift = world.brick["offset_x"] - world.scoop_desired_offset_x
    else:
        world.scoop_forward_preferred = False
        world.scoop_lateral_drift = 0.0

    brick_height = None
    if found and conf > 50 and cam_h > 0:
        if world.camera_height_anchor is None:
            world.camera_height_anchor = cam_h
        brick_height = max(0.0, world.camera_height_anchor - cam_h)
        world.height_mm = brick_height
    else:
        world.height_mm = None

    if found and conf >= ALIGN_CONFIDENCE_MIN:
        align_rules = world.learned_rules.get("ALIGN_BRICK", {}) or world.learned_rules.get("ALIGN", {})
        tol_off = align_rules.get("max_offset_x", world.align_tol_offset)
        tol_ang = align_rules.get("max_angle", world.align_tol_angle)

        if _step_name(world.step_state) in SCOOP_LIKE_STEPS:
            corridor = get_scoop_corridor_limits(world, dist)
            if corridor:
                tol_off = corridor.get("max_offset_x", tol_off)
                tol_ang = corridor.get("max_angle", tol_ang)

        tol_off *= 1.1
        tol_ang *= 1.1
        if _step_name(world.step_state) in SCOOP_LIKE_STEPS:
            tol_off *= world.scoop_success_offset_factor

        angle_ok = abs(angle) <= tol_ang
        offset_ok = abs(offset_x) <= tol_off
        dist_ok = world.align_tol_dist_min <= dist <= world.align_tol_dist_max

        if angle_ok and offset_ok and dist_ok:
            world.stability_count += 1
        else:
            world.stability_count = 0
    else:
        world.stability_count = 0

    return brick_height


def evaluate_start_gates(world, step, learned_rules, process_rules=None):
    obj_name = _step_name(step)
    reasons = []
    brick = smoothed_brick_snapshot(world)
    visible = bool(brick.get("visible"))
    start_visible = bool(visible)
    if not start_visible:
        start_visible = _recent_raw_visible(
            world,
            min_confidence=START_GATE_MIN_CONFIDENCE,
            required_hits=1,
            max_samples=BRICK_SMOOTH_FRAMES,
        )
    confidence = brick.get("confidence", 0.0) or 0.0

    if obj_name in ("ALIGN_BRICK", "SEAT_BRICK", "SEAT_BRICK2", "POSITION_BRICK"):
        if not visible:
            reasons.append("brick not visible")
        elif confidence < START_GATE_MIN_CONFIDENCE:
            reasons.append(f"confidence<{START_GATE_MIN_CONFIDENCE:.0f}")

    if obj_name in SCOOP_LIKE_STEPS and visible and confidence >= START_GATE_MIN_CONFIDENCE:
        dist = brick.get("dist")
        angle = abs(brick.get("angle", 0.0))
        offset = abs(brick.get("offset_x", 0.0))

        corridor = get_scoop_corridor_limits(world, dist) if dist else None
        envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
        scoop_metrics = envelope.get("success", {})

        max_angle = None
        max_offset = None
        if corridor:
            max_angle = corridor.get("max_angle")
            max_offset = corridor.get("max_offset_x")
        else:
            max_angle = (scoop_metrics.get("angle_abs") or {}).get("max", world.align_tol_angle)
            max_offset = (scoop_metrics.get("xAxis_offset_abs") or {}).get("max", world.align_tol_offset)

        dist_min = world.align_tol_dist_min
        dist_max = (scoop_metrics.get("dist") or {}).get("max", world.align_tol_dist_max)

        if dist is None or dist <= 0:
            reasons.append("distance unknown")
        else:
            if dist_min is not None and dist < dist_min:
                reasons.append(f"dist<{dist_min:.0f}mm")
            if dist_max is not None and dist > dist_max:
                reasons.append(f"dist>{dist_max:.0f}mm")
        if max_angle is not None and angle > max_angle:
            reasons.append(f"angle>{max_angle:.1f}deg")
        if max_offset is not None and offset > max_offset:
            reasons.append(f"offset>{max_offset:.1f}mm")

    start_metrics = (process_rules or {}).get(obj_name, {}).get("start_gates") or {}
    if start_metrics:
        for metric, stats in start_metrics.items():
            if metric in VISIBILITY_REQUIRED_METRICS and not visible:
                reasons.append("brick not visible")
                continue
            min_val = stats.get("min")
            max_val = stats.get("max")
            if metric == "visible":
                if isinstance(min_val, bool):
                    if bool(start_visible) != min_val:
                        reasons.append("visible gate")
                elif isinstance(max_val, bool):
                    if bool(start_visible) != max_val:
                        reasons.append("visible gate")
                else:
                    if (1.0 if start_visible else 0.0) < (min_val or 0.0):
                        reasons.append("visible gate")
                continue
            value = metric_value(brick, metric)
            if metric in ("brick_above", "brickAbove", "brick_below", "brickBelow", "inCrosshairs", "in_crosshairs"):
                ok = _bool_metric_gate_matches(value, stats)
                if ok is not True:
                    reasons.append(f"{metric} gate")
                continue
            if value is None:
                continue
            if min_val is not None and value < min_val:
                reasons.append(f"{metric}<{min_val}")
            if max_val is not None and value > max_val:
                reasons.append(f"{metric}>{max_val}")

    return GateCheck(ok=not reasons, reasons=reasons)


def _success_gate_eval(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    obj_name = _step_name(step)
    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    success_metrics = envelope.get("success") or {}
    if obj_name not in METRICS_BY_STEP:
        supported_metrics = set(METRIC_DIRECTIONS.keys())
        success_metrics = {k: v for k, v in success_metrics.items() if k in supported_metrics}
        if not success_metrics:
            return GateCheck(ok=True), []
    elif not success_metrics:
        return GateCheck(ok=False, reasons=["no success envelope"]), []

    brick = smoothed_brick_snapshot(world)
    raw_brick = getattr(world, "brick", None) or {}
    visible = bool(brick.get("visible"))
    visible_gate = success_metrics.get("visible") or {}
    if visibility_grace_s is None:
        visible_grace_s = VISIBILITY_LOST_GRACE_S
        if isinstance(visible_gate, dict):
            if visible_gate.get("min") is True:
                visible_grace_s = 0.0
            elif visible_gate.get("min") is False:
                visible_grace_s = VISIBLE_FALSE_GRACE_S_BY_STEP.get(
                    obj_name,
                    VISIBILITY_LOST_GRACE_S,
                )
    else:
        visible_grace_s = visibility_grace_s
    effective_visible = _effective_visible(world, visible, grace_s=visible_grace_s)

    reasons = []
    entries = []

    for metric, stats in success_metrics.items():
        direction = metric_direction_for_step(metric, obj_name, process_rules=process_rules)
        entry = {
            "metric": metric,
            "stats": stats,
            "raw_visible": visible,
            "effective_visible": effective_visible,
            "visible_grace_s": visible_grace_s,
        }

        if metric in VISIBILITY_REQUIRED_METRICS and not visible:
            reasons.append("brick not visible")
            entry["value"] = None
            entry["raw_value"] = metric_value(raw_brick, metric)
            entries.append(entry)
            continue

        if metric == "angle_abs":
            angle_val = abs(brick.get("angle", 0.0))
            entry["value"] = angle_val
            entry["raw_value"] = metric_value(raw_brick, metric)
            ok = _target_tol_ok(angle_val, stats, direction)
            if ok is False:
                reasons.append("angle_abs gate")
            elif ok is None and angle_val > stats.get("max", 0.0):
                reasons.append("angle_abs gate")
        elif metric in X_AXIS_GATE_METRICS:
            offset_val = brick.get("x_axis", brick.get("offset_x", 0.0))
            entry["value"] = offset_val
            entry["raw_value"] = metric_value(raw_brick, metric)
            target = stats.get("target") if isinstance(stats, dict) else None
            tol = stats.get("tol") if isinstance(stats, dict) else None
            if target is not None and tol is not None:
                ok = abs(offset_val - target) <= tol
            else:
                ok = _target_tol_ok(offset_val, stats, direction)
            if ok is False:
                reasons.append(f"{metric} gate")
            elif ok is None and offset_val > stats.get("max", 0.0):
                reasons.append(f"{metric} gate")
        elif metric in Y_AXIS_GATE_METRICS:
            offset_val = brick.get("y_axis", brick.get("offset_y", 0.0))
            entry["value"] = offset_val
            entry["raw_value"] = metric_value(raw_brick, metric)
            target = stats.get("target") if isinstance(stats, dict) else None
            tol = stats.get("tol") if isinstance(stats, dict) else None
            if target is not None and tol is not None:
                ok = abs(offset_val - target) <= tol
            else:
                ok = _target_tol_ok(offset_val, stats, direction)
            if ok is False:
                reasons.append(f"{metric} gate")
            elif ok is None and offset_val > stats.get("max", 0.0):
                reasons.append(f"{metric} gate")
        elif metric == "dist":
            dist_val = brick.get("dist", 0.0)
            entry["value"] = dist_val
            entry["raw_value"] = metric_value(raw_brick, metric)
            target = stats.get("target") if isinstance(stats, dict) else None
            tol = stats.get("tol") if isinstance(stats, dict) else None
            if target is not None and tol is not None:
                ok = abs(dist_val - target) <= tol
            else:
                ok = _target_tol_ok(dist_val, stats, direction)
            if ok is False:
                reasons.append("dist gate")
            elif ok is None and dist_val > stats.get("max", 0.0):
                reasons.append("dist gate")
        elif metric == "confidence":
            conf_val = brick.get("confidence", 0.0)
            entry["value"] = conf_val
            entry["raw_value"] = metric_value(raw_brick, metric)
            ok = _target_tol_ok(conf_val, stats, direction)
            if ok is False:
                reasons.append("confidence gate")
            elif ok is None and conf_val < stats.get("min", 0.0):
                reasons.append("confidence gate")
        elif metric == "visible":
            min_val = stats.get("min")
            max_val = stats.get("max")
            entry["value"] = bool(effective_visible)
            confidence_val = float(brick.get("confidence", 0.0) or 0.0)
            entry["confidence"] = confidence_val
            confidence_measurement = {
                "visible": bool(visible),
                "confidence": confidence_val,
            }
            confident_recent = visible_false_gate_confident_recent(world, stats)
            entry["confident_visible_recent"] = bool(confident_recent)
            target_visible = gate_utils.bool_gate_target(stats)
            if target_visible is False:
                # Strict boolean semantics for visible=false success gates.
                if bool(effective_visible):
                    reasons.append("visible gate")
                    entries.append(entry)
                    continue
                if confident_recent or gate_utils.visible_false_gate_confidently_seen(confidence_measurement, stats):
                    reasons.append("visible gate")
                    entries.append(entry)
                    continue
                entries.append(entry)
                continue
            if isinstance(min_val, bool):
                if bool(effective_visible) != min_val:
                    reasons.append("visible gate")
            elif isinstance(max_val, bool):
                if bool(effective_visible) != max_val:
                    reasons.append("visible gate")
            else:
                if (1.0 if effective_visible else 0.0) < stats.get("min", 0.0):
                    reasons.append("visible gate")
        elif metric in ("brick_above", "brickAbove", "brick_below", "brickBelow", "inCrosshairs", "in_crosshairs"):
            bool_val = metric_value(brick, metric)
            entry["value"] = bool_val
            if _bool_metric_gate_matches(bool_val, stats) is not True:
                reasons.append(f"{metric} gate")
        entries.append(entry)

    return GateCheck(ok=not reasons, reasons=reasons), entries


def success_gate_entries(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    _, entries = _success_gate_eval(
        world,
        step,
        learned_rules,
        process_rules=process_rules,
        visibility_grace_s=visibility_grace_s,
    )
    return entries


def evaluate_success_gates(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    check, _ = _success_gate_eval(
        world,
        step,
        learned_rules,
        process_rules=process_rules,
        visibility_grace_s=visibility_grace_s,
    )
    return check


def evaluate_failure_gates(world, step, learned_rules, process_rules=None):
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    failure_metrics = envelope.get("failure") or {}
    if not failure_metrics:
        return GateCheck(ok=True)
    
    brick = smoothed_brick_snapshot(world)
    visible = bool(brick.get("visible"))
    effective_visible = _effective_visible(world, visible)
    reasons = []
    
    # Check if current state matches known failure patterns
    for metric, stats in failure_metrics.items():
        if metric in VISIBILITY_REQUIRED_METRICS and not visible:
            # Can't match failure pattern if brick isn't visible
            continue
            
        if metric == "visible":
            value = 1.0 if effective_visible else 0.0
        else:
            value = metric_value(brick, metric)
        if value is None:
            continue
        
        direction = metric_direction_for_step(metric, obj_name, process_rules=process_rules)
        
        # If we have learned failure mu/sigma, check if we're in the failure zone
        if "mu" in stats and "sigma" in stats:
            mu = stats.get("mu")
            sigma = stats.get("sigma", 0.0)
            
            # Direction-aware pattern matching
            if direction == "low":
                # For "low" metrics (angle, offset, dist), being HIGH is bad
                # Only trigger if we're near or above the failure pattern
                if value >= mu - sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
            elif direction == "high":
                # For "high" metrics (visible, confidence), being LOW is bad
                # Only trigger if we're near or below the failure pattern
                if value <= mu + sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
            else:
                # For other metrics, use simple range check
                if abs(value - mu) <= sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
        else:
            # Fallback to min/max range check
            status = metric_status(value, {}, stats, METRIC_DIRECTIONS.get(metric))
            if status == "fail":
                reasons.append(f"{metric} gate")
    
    return GateCheck(ok=not reasons, reasons=reasons)
