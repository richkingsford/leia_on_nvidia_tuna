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

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    "xAxis_offset_abs": "low",
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


def success_gates_visible_only(process_rules, step):
    obj_name = _step_name(step)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    if not success_metrics:
        return False
    keys = {key for key in success_metrics.keys() if key is not None}
    return keys == {"visible"}


def turn_cmd_from_signed_error(signed_error):
    if signed_error is None:
        return None
    if signed_error > 0:
        return "l"
    if signed_error < 0:
        return "r"
    return None


def compute_alignment_analytics(world, process_rules, learned_rules, step, duration_s=0.05):
    obj_name = _step_name(step)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
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
    signed_values = {
        "xAxis_offset_abs": x_axis,
        "angle_abs": angle,
        "dist": dist,
    }
    progress_values = []
    offsets = {}
    ratios = {}
    mm_errors = {}

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

    for metric, value in metrics.items():
        stats = success_metrics.get(metric) or fallback_stats(metric)
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
    x_axis_stats = success_metrics.get("xAxis_offset_abs") or fallback_stats("xAxis_offset_abs")
    x_axis_ok = metric_within_gate(x_axis_stats, x_axis)
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
            if not x_axis_ok:
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

    if worst_metric == "dist" and not x_axis_ok:
        worst_metric = "xAxis_offset_abs"
        worst_ratio = max(worst_ratio, ratios.get("xAxis_offset_abs", 1.0), 1.0)

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
        # Positive error should turn right, negative error should turn left.
        cmd = turn_cmd_from_signed_error(-signed_error)
        if abs(signed_error) < micro_offset_mm:
            speed = min(speed, micro_speed)
        worst_metric = "xAxis_offset"

    elif worst_metric == "angle":
        signed = signed_values.get("angle", 0.0)
        stats = success_metrics.get("angle") or fallback_stats("angle")
        target = stats.get("target")
        tol = stats.get("tol")
        mag = abs(signed)
        signed_error = mag - target if target is not None and tol is not None else signed
        if signed_error >= 0:
            cmd = "r" if signed > 0 else "l"
        else:
            cmd = "l" if signed > 0 else "r"
        if abs(signed) < micro_angle_deg:
            speed = min(speed, micro_speed)
        worst_metric = "angle"

    visible_only = success_gates_visible_only(process_rules, obj_name)
    if visible_only:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)
    else:
        mm_off = None
        if cmd in ("l", "r"):
            mm_off = mm_errors.get("xAxis_offset")
        elif cmd in ("f", "b"):
            mm_off = mm_errors.get("dist")
        if (mm_off is None or mm_off <= 0.0) and worst_metric == "angle":
            mm_off = None
        slow_mm = ALIGN_SPEED_SLOW_MM
        fast_mm = ALIGN_SPEED_FAST_MM
        if obj_name != "ALIGN_BRICK":
            slow_mm = ALIGN_SPEED_SLOW_MM / 4.0
            fast_mm = ALIGN_SPEED_FAST_MM / 4.0
        if obj_name == "ALIGN_BRICK" and cmd == "f":
            if mm_off is None:
                speed_score = SPEED_SCORE_DEFAULT
            elif mm_off <= slow_mm:
                speed_score = SPEED_SCORE_MIN
            else:
                speed_score = SPEED_SCORE_DEFAULT
        else:
            speed_score = _score_from_mm(mm_off, slow_mm, fast_mm)
        speed_score = normalize_speed_score(speed_score)

    if not visible_for_cmd:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)

    # Force 1% speed score (fixed 0.24 power) for all alignment turns
    if cmd in ("l", "r"):
        speed_score = SPEED_SCORE_MIN

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


def offset_cmd_from_offset_x(offset_x):
    return turn_cmd_from_signed_error(offset_x)


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


def offset_correction_cmd(measurement, gates):
    if not measurement:
        return None
    offset = measurement.get("offset_x")
    if offset is None:
        return None
    stats = (gates or {}).get("xAxis_offset_abs") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    abs_offset = abs(offset)
    if target is not None and tol is not None:
        signed_error = offset - target
        if abs(signed_error) > tol:
            return turn_cmd_from_signed_error(-signed_error)
        return None
    if max_val is not None and abs_offset > max_val:
        return turn_cmd_from_signed_error(-offset)
    if min_val is not None and abs_offset < min_val:
        return turn_cmd_from_signed_error(-offset)
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
    cmd = offset_correction_cmd(brick, success_gates)
    if cmd:
        return "turn right" if cmd == "r" else "turn left"
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
