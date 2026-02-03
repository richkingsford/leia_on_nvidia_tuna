import math
import time
from dataclasses import dataclass, field
from typing import List

from helper_next import METRIC_DIRECTIONS, _target_tol_ok, compute_alignment_analytics, metric_direction_for_step

START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
VISIBILITY_LOST_GRACE_S = 0.5
BRICK_SMOOTH_FRAMES = 3
BRICK_SMOOTH_OUTLIER_MM = 12.0
BRICK_SMOOTH_OUTLIER_DEG = 6.0
VISIBLE_FALSE_GRACE_S_BY_STEP = {
    "EXIT_WALL": 1.0,
}

STEP_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

METRICS_BY_STEP = {
    "FIND_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "EXIT_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "FIND_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "ALIGN_BRICK": ("xAxis_offset_abs", "dist", "visible"),
    "SCOOP": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "POSITION_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
}

VISIBILITY_REQUIRED_METRICS = {"angle_abs", "xAxis_offset_abs", "dist"}


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
    angle_vals = [f["angle"] for f in frames]
    med_dist = percentile(dist_vals, 0.5)
    med_offset = percentile(offset_vals, 0.5)
    med_angle = percentile(angle_vals, 0.5)

    keep = []
    for frame in frames:
        if not frame["found"]:
            continue
        if (
            abs(frame["dist"] - med_dist) > BRICK_SMOOTH_OUTLIER_MM
            or abs(frame["offset_x"] - med_offset) > BRICK_SMOOTH_OUTLIER_MM
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
        "brickAbove": bool(avg.get("brick_above")),
        "brickBelow": bool(avg.get("brick_below")),
    }


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
    return STEP_ALIASES.get(key, key)


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
    if metric == "xAxis_offset_abs":
        return brick.get("x_axis", brick.get("offset_x", 0.0))
    if metric == "dist":
        return brick.get("dist", 0.0)
    if metric == "visible":
        return 1.0 if brick.get("visible") else 0.0
    if metric == "confidence":
        return brick.get("confidence", 0.0)
    return None


def _effective_visible(world, visible, grace_s=VISIBILITY_LOST_GRACE_S):
    if visible:
        return True
    last_seen = getattr(world, "last_visible_time", None)
    if last_seen is None:
        return False
    return (time.time() - last_seen) <= grace_s


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
    corridor = world.learned_rules.get("SCOOP", {}).get("corridor")
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
            direction = metric_direction_for_step(metric, obj_name)
            value = metric_value(world.brick or {}, metric)
            if value is None or direction is None or not isinstance(stats, dict):
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
    world.brick["visible"] = bool(found)
    world.brick["dist"] = float(dist)
    world.brick["angle"] = float(angle)
    world.brick["confidence"] = float(conf)
    world.brick["offset_x"] = float(offset_x)
    world.brick["x_axis"] = float(offset_x)
    world.brick["brickAbove"] = bool(brick_above)
    world.brick["brickBelow"] = bool(brick_below)
    if found:
        world.last_visible_time = time.time()
        world.last_seen_angle = float(angle)
        world.last_seen_offset_x = float(offset_x)
        world.last_seen_x_axis = float(offset_x)
        world.last_seen_dist = float(dist)
        world.last_seen_confidence = float(conf)

    if world.step_state.value == "SCOOP":
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

        if world.step_state.value == "SCOOP":
            corridor = get_scoop_corridor_limits(world, dist)
            if corridor:
                tol_off = corridor.get("max_offset_x", tol_off)
                tol_ang = corridor.get("max_angle", tol_ang)

        tol_off *= 1.1
        tol_ang *= 1.1
        if world.step_state.value == "SCOOP":
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
    confidence = brick.get("confidence", 0.0) or 0.0

    if obj_name in ("ALIGN_BRICK", "SCOOP", "POSITION_BRICK"):
        if not visible:
            reasons.append("brick not visible")
        elif confidence < START_GATE_MIN_CONFIDENCE:
            reasons.append(f"confidence<{START_GATE_MIN_CONFIDENCE:.0f}")

    if obj_name == "SCOOP" and visible and confidence >= START_GATE_MIN_CONFIDENCE:
        dist = brick.get("dist")
        angle = abs(brick.get("angle", 0.0))
        offset = abs(brick.get("offset_x", 0.0))

        corridor = get_scoop_corridor_limits(world, dist) if dist else None
        envelope = build_envelope(process_rules or {}, learned_rules or {}, "SCOOP")
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
            if metric in ("angle_abs", "xAxis_offset_abs", "dist", "confidence") and not visible:
                reasons.append("brick not visible")
                continue
            min_val = stats.get("min")
            max_val = stats.get("max")
            if metric == "visible":
                if isinstance(min_val, bool):
                    if bool(visible) != min_val:
                        reasons.append("visible gate")
                elif isinstance(max_val, bool):
                    if bool(visible) != max_val:
                        reasons.append("visible gate")
                else:
                    if (1.0 if visible else 0.0) < (min_val or 0.0):
                        reasons.append("visible gate")
                continue
            value = metric_value(brick, metric)
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

    # FIND_BRICK: Ensure we're finding a loose brick, not one already on the wall
    if obj_name == "FIND_BRICK" and brick.get("brickBelow"):
        reasons.append("brick already stacked")

    for metric, stats in success_metrics.items():
        direction = metric_direction_for_step(metric, obj_name)
        entry = {
            "metric": metric,
            "stats": stats,
            "raw_visible": visible,
            "effective_visible": effective_visible,
            "visible_grace_s": visible_grace_s,
        }

        if metric in ("angle_abs", "xAxis_offset_abs", "dist", "confidence") and not visible:
            reasons.append("brick not visible")
            entry["value"] = None
            entries.append(entry)
            continue

        if metric == "angle_abs":
            angle_val = abs(brick.get("angle", 0.0))
            entry["value"] = angle_val
            ok = _target_tol_ok(angle_val, stats, direction)
            if ok is False:
                reasons.append("angle_abs gate")
            elif ok is None and angle_val > stats.get("max", 0.0):
                reasons.append("angle_abs gate")
        elif metric == "xAxis_offset_abs":
            offset_val = brick.get("x_axis", brick.get("offset_x", 0.0))
            entry["value"] = offset_val
            target = stats.get("target") if isinstance(stats, dict) else None
            tol = stats.get("tol") if isinstance(stats, dict) else None
            if target is not None and tol is not None:
                ok = abs(offset_val - target) <= tol
            else:
                ok = _target_tol_ok(offset_val, stats, direction)
            if ok is False:
                reasons.append("xAxis_offset_abs gate")
            elif ok is None and offset_val > stats.get("max", 0.0):
                reasons.append("xAxis_offset_abs gate")
        elif metric == "dist":
            dist_val = brick.get("dist", 0.0)
            entry["value"] = dist_val
            ok = _target_tol_ok(dist_val, stats, direction)
            if ok is False:
                reasons.append("dist gate")
            elif ok is None and dist_val > stats.get("max", 0.0):
                reasons.append("dist gate")
        elif metric == "confidence":
            conf_val = brick.get("confidence", 0.0)
            entry["value"] = conf_val
            ok = _target_tol_ok(conf_val, stats, direction)
            if ok is False:
                reasons.append("confidence gate")
            elif ok is None and conf_val < stats.get("min", 0.0):
                reasons.append("confidence gate")
        elif metric == "visible":
            min_val = stats.get("min")
            max_val = stats.get("max")
            entry["value"] = bool(effective_visible)
            if isinstance(min_val, bool):
                if bool(effective_visible) != min_val:
                    reasons.append("visible gate")
            elif isinstance(max_val, bool):
                if bool(effective_visible) != max_val:
                    reasons.append("visible gate")
            else:
                if (1.0 if effective_visible else 0.0) < stats.get("min", 0.0):
                    reasons.append("visible gate")
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
        if metric in ("angle_abs", "xAxis_offset_abs", "dist") and not visible:
            # Can't match failure pattern if brick isn't visible
            continue
            
        if metric == "visible":
            value = 1.0 if effective_visible else 0.0
        else:
            value = metric_value(brick, metric)
        if value is None:
            continue
        
        direction = metric_direction_for_step(metric, obj_name)
        
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
