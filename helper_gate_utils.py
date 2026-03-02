import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from helper_demo_log_utils import normalize_step_label

PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
GATE_CHECKER_MODEL_FILE = Path(__file__).resolve().parent / "world_model_gate_checker.json"

DEFAULT_GATE_CHECKER_CONFIG = {
    "consecutive_required": 12,
    "majority_window": 26,
    "majority_required": 14,
    "lite_only_aruco_experiment": False,
    "aruco_full_gatecheck_pass_scale": 1.0,
}
VISIBLE_FALSE_FAIL_CONFIDENCE_MIN = 70.0
DEFAULT_ARUCO_MARKER_SIZE_MM = 20.0


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


def _coerce_int(value, fallback, minimum=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(int(minimum), parsed)


def _coerce_float(value, fallback, minimum=0.0, maximum=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return float(parsed)


def load_gate_checker_config(path=GATE_CHECKER_MODEL_FILE, default_config=None):
    cfg = dict(default_config or DEFAULT_GATE_CHECKER_CONFIG)
    raw = {}
    file_path = Path(path)
    if file_path.exists():
        try:
            loaded = json.loads(file_path.read_text())
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            nested = loaded.get("gate_checker")
            if isinstance(nested, dict):
                raw = nested
            else:
                raw = loaded
    consecutive_required = _coerce_int(
        raw.get("consecutive_required"),
        cfg["consecutive_required"],
        minimum=1,
    )
    majority_window = _coerce_int(
        raw.get("majority_window"),
        cfg["majority_window"],
        minimum=1,
    )
    majority_required = _coerce_int(
        raw.get("majority_required"),
        cfg["majority_required"],
        minimum=1,
    )
    majority_required = min(majority_required, majority_window)
    lite_only_aruco_experiment = bool(
        raw.get("lite_only_aruco_experiment", cfg.get("lite_only_aruco_experiment", False))
    )
    aruco_full_gatecheck_pass_scale = _coerce_float(
        raw.get("aruco_full_gatecheck_pass_scale"),
        cfg.get("aruco_full_gatecheck_pass_scale", 1.0),
        minimum=0.05,
        maximum=1.0,
    )
    return {
        "consecutive_required": consecutive_required,
        "majority_window": majority_window,
        "majority_required": majority_required,
        "lite_only_aruco_experiment": lite_only_aruco_experiment,
        "aruco_full_gatecheck_pass_scale": aruco_full_gatecheck_pass_scale,
    }


def load_process_steps(path=PROCESS_MODEL_FILE):
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("steps") or {}


def metric_value_from_measurement(measurement, metric):
    if not measurement:
        return None
    if metric == "visible":
        return bool(measurement.get("visible"))
    if metric in ("inCrosshairs", "in_crosshairs"):
        if "inCrosshairs" in measurement:
            value = measurement.get("inCrosshairs")
            return None if value is None else bool(value)
        if "in_crosshairs" in measurement:
            value = measurement.get("in_crosshairs")
            return None if value is None else bool(value)
        found = bool(measurement.get("visible"))
        if not found:
            return False
        try:
            x_val = measurement.get("x_axis")
            if x_val is None:
                x_val = measurement.get("offset_x")
            y_val = measurement.get("y_axis")
            if y_val is None:
                y_val = measurement.get("offset_y")
            if y_val is None:
                y_val = measurement.get("cam_h")
            x_num = float(x_val)
            y_num = float(y_val)
            center_x = measurement.get("in_crosshairs_center_x", measurement.get("inCrosshairs_center_x", 0.0))
            center_y = measurement.get("in_crosshairs_center_y", measurement.get("inCrosshairs_center_y", 0.0))
            center_x_num = float(center_x)
            center_y_num = float(center_y)
        except (TypeError, ValueError):
            return None
        marker_left = x_num - _CROSSHAIR_HALF_WIDTH_MM
        marker_right = x_num + _CROSSHAIR_HALF_WIDTH_MM
        marker_top = y_num - _CROSSHAIR_HALF_HEIGHT_MM
        marker_bottom = y_num + _CROSSHAIR_HALF_HEIGHT_MM
        x_overlaps = marker_left <= center_x_num <= marker_right
        y_overlaps = marker_top <= center_y_num <= marker_bottom
        return bool(x_overlaps and y_overlaps)
    if metric in ("brick_above", "brickAbove"):
        if "brick_above" in measurement:
            value = measurement.get("brick_above")
            return None if value is None else bool(value)
        if "brickAbove" in measurement:
            value = measurement.get("brickAbove")
            return None if value is None else bool(value)
        return None
    if metric in ("brick_below", "brickBelow"):
        if "brick_below" in measurement:
            value = measurement.get("brick_below")
            return None if value is None else bool(value)
        if "brickBelow" in measurement:
            value = measurement.get("brickBelow")
            return None if value is None else bool(value)
        return None
    if metric == "angle_abs":
        value = measurement.get("angle")
        return abs(value) if value is not None else None
    if metric == "xAxis_offset_abs":
        value = measurement.get("x_axis")
        if value is None:
            value = measurement.get("offset_x")
        return float(value) if value is not None else None
    if metric == "yAxis_offset_abs":
        value = measurement.get("y_axis")
        if value is None:
            value = measurement.get("offset_y")
        return float(value) if value is not None else None
    if metric == "xAxis_offset":
        value = measurement.get("x_axis")
        if value is None:
            value = measurement.get("offset_x")
        return float(value) if value is not None else None
    if metric == "yAxis_offset":
        value = measurement.get("y_axis")
        if value is None:
            value = measurement.get("offset_y")
        return float(value) if value is not None else None
    if metric == "dist":
        value = measurement.get("dist")
        return float(value) if value is not None else None
    if metric == "confidence":
        value = measurement.get("confidence")
        return float(value) if value is not None else None
    if metric == "angle":
        return measurement.get("angle")
    if metric == "x_axis":
        value = measurement.get("x_axis")
        if value is None:
            value = measurement.get("offset_x")
        return value
    if metric == "y_axis":
        value = measurement.get("y_axis")
        if value is None:
            value = measurement.get("offset_y")
        return value
    if metric == "distance":
        return measurement.get("dist")
    return None


def metric_error(value, stats):
    if value is None or not isinstance(stats, dict):
        return None
    if isinstance(value, bool):
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None:
            return 0.0 if value is bool(min_val) else 1.0
        if max_val is not None:
            return 0.0 if value is bool(max_val) else 1.0
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            if isinstance(target, bool):
                try:
                    tol_num = float(tol)
                except (TypeError, ValueError):
                    tol_num = 0.0
                if tol_num <= 0.0:
                    return 0.0 if value is bool(target) else 1.0
                value_num = 1.0 if value else 0.0
                target_num = 1.0 if bool(target) else 0.0
                return max(0.0, abs(value_num - target_num) - tol_num)
        return None
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        return max(0.0, abs(value - target) - tol)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and value < min_val:
        return min_val - value
    if max_val is not None and value > max_val:
        return value - max_val
    return 0.0


def metric_progress(value, stats):
    if value is None or not isinstance(stats, dict):
        return None
    if isinstance(value, bool):
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None and isinstance(target, bool):
            try:
                tol_num = float(tol)
            except (TypeError, ValueError):
                tol_num = 0.0
            if tol_num <= 0.0:
                return 1.0 if value is bool(target) else 0.0
            value_num = 1.0 if value else 0.0
            target_num = 1.0 if bool(target) else 0.0
            distance = abs(value_num - target_num)
            if distance <= tol_num:
                return 1.0
            return max(0.0, 1.0 - (distance - tol_num) / tol_num)
        err = metric_error(value, stats)
        return 1.0 if err == 0.0 else 0.0
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if tol <= 0:
            return 1.0 if value == target else 0.0
        distance = abs(value - target)
        if distance <= tol:
            return 1.0
        return max(0.0, 1.0 - (distance - tol) / tol)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and max_val is not None:
        if min_val <= value <= max_val:
            return 1.0
        span = max(1e-3, max_val - min_val)
        if value < min_val:
            return max(0.0, 1.0 - (min_val - value) / span)
        return max(0.0, 1.0 - (value - max_val) / span)
    if min_val is not None:
        return 1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3))
    if max_val is not None:
        return 1.0 if value <= max_val else max(0.0, 1.0 - (value - max_val) / max(max_val, 1e-3))
    return None


def bool_gate_target(stats):
    if not isinstance(stats, dict):
        return None
    for key in ("min", "max", "target"):
        value = stats.get(key)
        if isinstance(value, bool):
            return bool(value)
    return None


def visible_false_gate_confidently_seen(
    measurement,
    stats,
    min_confidence=VISIBLE_FALSE_FAIL_CONFIDENCE_MIN,
):
    if bool_gate_target(stats) is not False:
        return False
    visible_now = bool((measurement or {}).get("visible"))
    if not visible_now:
        return False
    confidence = (measurement or {}).get("confidence")
    if confidence is None:
        confidence = (measurement or {}).get("conf")
    if confidence is None:
        # If confidence is unavailable but visibility is true, be conservative.
        return True
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        return True
    return conf_val >= float(min_confidence)


def step_progress(measurement, success_gates):
    if not success_gates:
        return None
    progress_values = []
    for metric, stats in success_gates.items():
        value = metric_value_from_measurement(measurement, metric)
        prog = metric_progress(value, stats)
        if prog is not None:
            progress_values.append(prog)
    if not progress_values:
        return None
    return sum(progress_values) / len(progress_values)


def gate_satisfied(measurement, gates):
    if not gates:
        return False
    saw_value = False
    for metric, stats in gates.items():
        value = metric_value_from_measurement(measurement, metric)
        if value is None:
            continue
        saw_value = True
        if metric == "visible" and bool_gate_target(stats) is False:
            # Strict boolean semantics: visible=true never satisfies visible=false.
            if bool(value):
                return False
            if visible_false_gate_confidently_seen(measurement, stats):
                return False
            continue
        err = metric_error(value, stats)
        if err is None or err > 0:
            return False
    return saw_value


def satisfied_steps(measurement, steps):
    satisfied = []
    for step_name, data in steps.items():
        success_gates = (data or {}).get("success_gates") or {}
        if gate_satisfied(measurement, success_gates):
            satisfied.append(step_name)
    return satisfied


@dataclass
class SuccessGateTracker:
    consecutive_required: int
    majority_window: int
    majority_required: int
    consecutive: int = 0
    window: list = field(default_factory=list)
    total_checks: int = 0
    total_pass: int = 0

    def update(self, success_ok):
        consecutive_pass_required = max(
            1,
            int(getattr(self, "consecutive_pass_required", self.consecutive_required)),
        )
        majority_pass_required = max(
            1,
            int(getattr(self, "majority_pass_required", self.majority_required)),
        )
        self.total_checks += 1
        if success_ok:
            self.total_pass += 1
            self.consecutive += 1
        else:
            self.consecutive = 0
        self.window.append(bool(success_ok))
        if len(self.window) > max(1, int(self.majority_window)):
            self.window.pop(0)
        if self.consecutive >= consecutive_pass_required:
            return True
        if sum(self.window) >= majority_pass_required:
            return True
        return False


def _step_key(step):
    key = normalize_step_label(step)
    return key or str(step)


def update_gatecheck(world, step, tracker, success_ok, phase=None):
    success_met = tracker.update(success_ok)
    if world is None:
        return success_met
    window_vals = list(getattr(tracker, "window", []))
    window_pass = int(sum(1 for ok in window_vals if ok))
    window_size = int(len(window_vals))
    window_total = max(1, int(getattr(tracker, "majority_window", 1)))
    window_need = max(1, int(getattr(tracker, "majority_required", 1)))
    window_need_pass = max(1, int(getattr(tracker, "majority_pass_required", window_need)))
    streak = int(getattr(tracker, "consecutive", 0))
    need = max(1, int(getattr(tracker, "consecutive_required", 1)))
    need_pass = max(1, int(getattr(tracker, "consecutive_pass_required", need)))
    consecutive_ok = streak >= need_pass
    majority_ok = window_pass >= window_need_pass
    truth_ok = consecutive_ok or majority_ok
    if consecutive_ok:
        truth_by = "consecutive"
    elif majority_ok:
        truth_by = "majority"
    else:
        truth_by = None
    mode = str(getattr(world, "_gatecheck_mode", "traditional") or "traditional").lower()
    if mode == "lite" and truth_ok:
        truth_by = "lite"
    checks = int(getattr(tracker, "total_checks", 0))
    passed = int(getattr(tracker, "total_pass", 0))
    status = {
        "step": _step_key(step),
        "phase": phase or "run",
        "mode": mode,
        "checks": checks,
        "pass": passed,
        "fail": max(0, checks - passed),
        "streak": streak,
        "need": need,
        "need_pass": need_pass,
        "consecutive_ok": bool(consecutive_ok),
        "window_pass": window_pass,
        "window_size": window_size,
        "window_total": window_total,
        "window_need": window_need,
        "window_need_pass": window_need_pass,
        "majority_ok": bool(majority_ok),
        "truth_ok": bool(truth_ok),
        "truth_by": truth_by,
        "lite_required": int(getattr(world, "_gatecheck_lite_required", 0) or 0),
        "lite_collected": int(getattr(world, "_gatecheck_lite_collected", 0) or 0),
        "frame_id": int(getattr(world, "_frame_id", 0) or 0),
        "timestamp": time.time(),
    }
    world._gatecheck_status = status
    return success_met


def record_success_gate_entry(world, step, success_ok):
    if world is None:
        return
    world._success_gate_last_step = _step_key(step)
    world._success_gate_last_ok = bool(success_ok)


def should_hold_for_success_confirmation(visible_only, tracker, success_met):
    if success_met or not visible_only:
        return False
    needed = max(1, int(getattr(tracker, "consecutive_required", 1)))
    streak = max(0, int(getattr(tracker, "consecutive", 0)))
    if 0 < streak < needed:
        return True
    window_vals = list(getattr(tracker, "window", []))
    if not window_vals:
        return False
    window_total = max(1, int(getattr(tracker, "majority_window", 1)))
    # If we have any positive evidence, pause motion long enough to confirm/refute
    # truth from a full majority window.
    return any(window_vals) and len(window_vals) < window_total


def store_gate_summary(world, tracker):
    if world is None:
        return
    checks = int(getattr(tracker, "total_checks", 0))
    passed = int(getattr(tracker, "total_pass", 0))
    world._last_gate_summary = {
        "checks": checks,
        "pass": passed,
        "fail": max(0, checks - passed),
        "streak": int(getattr(tracker, "consecutive", 0)),
        "need": int(getattr(tracker, "consecutive_required", 0)),
    }


def consume_gate_summary(world):
    if world is None:
        return None
    summary = getattr(world, "_last_gate_summary", None)
    world._last_gate_summary = None
    return summary


def format_gate_summary_line(summary, smooth_frames=1):
    if not summary:
        return None
    checks = int(summary.get("checks", 0))
    passed = int(summary.get("pass", 0))
    failed = int(summary.get("fail", 0))
    return (
        f"Required {checks} gate checks "
        f"({passed} pass, {failed} fail; {int(smooth_frames)}-frame smoothing)."
    )


def format_gatecheck_stream_line(world, step=None):
    lines = format_gatecheck_stream_lines(world, step=step)
    return lines[0] if lines else None


def format_gatecheck_stream_lines(world, step=None):
    if world is None:
        return []
    status = getattr(world, "_gatecheck_status", None)
    if not isinstance(status, dict):
        return []
    mode = str(status.get("mode") or "traditional").lower()
    checks = int(status.get("checks", 0) or 0)
    truth_ok = bool(status.get("truth_ok", False))
    state = "pass" if truth_ok else "wait"
    need = max(1, int(status.get("need", 1) or 1))
    need_pass = max(1, int(status.get("need_pass", need) or need))
    window_pass = int(status.get("window_pass", 0) or 0)
    window_size = int(status.get("window_size", 0) or 0)
    window_total = max(1, int(status.get("window_total", 1) or 1))
    window_need = max(1, int(status.get("window_need", 1) or 1))
    window_need_pass = max(1, int(status.get("window_need_pass", window_need) or window_need))
    if mode == "lite":
        lite_required = max(1, int(status.get("lite_required", 1)))
        lite_collected = max(0, int(status.get("lite_collected", 0)))
        top_line = f"LITE: {lite_collected}/{lite_required} avg-smoothed frames"
        seen_line = f"SEEN: {checks} total"
        lite_line = f"LITE-GATE: {state} ({lite_collected}/{lite_required} frames)"
        return [top_line, seen_line, lite_line]
    streak = int(status.get("streak", 0) or 0)
    top_line = f"CONSEC: {streak}/{need} ok (need:{need_pass})"
    seen_line = f"SEEN: {checks} total, win {window_size}/{window_total}"
    majority_line = f"MAJ: {window_pass}/{window_total} pass, need:{window_need_pass}"
    return [top_line, seen_line, majority_line]


def wait_for_fresh_frames(world, refresh_once, required_new_frames=1, max_cycles=80, sleep_s=0.0):
    if world is None:
        return {"required": 0, "start": 0, "end": 0, "advanced": 0, "cycles": 0}
    required = max(0, int(required_new_frames or 0))
    start_frame = int(getattr(world, "_frame_id", 0) or 0)
    target_frame = start_frame + required
    cycles = 0
    while int(getattr(world, "_frame_id", 0) or 0) < target_frame:
        refresh_once()
        cycles += 1
        if max_cycles is not None and cycles >= max(1, int(max_cycles)):
            break
        if sleep_s and int(getattr(world, "_frame_id", 0) or 0) < target_frame:
            time.sleep(max(0.0, float(sleep_s)))
    end_frame = int(getattr(world, "_frame_id", 0) or 0)
    info = {
        "required": required,
        "start": start_frame,
        "end": end_frame,
        "advanced": max(0, end_frame - start_frame),
        "cycles": cycles,
    }
    world._frame_wait_status = info
    return info


def evaluate_brick_start_gates(world, step, learned_rules, process_rules=None):
    import telemetry_brick
    return telemetry_brick.evaluate_start_gates(world, step, learned_rules, process_rules=process_rules)


def evaluate_brick_success_gates(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    import telemetry_brick
    return telemetry_brick.evaluate_success_gates(
        world,
        step,
        learned_rules,
        process_rules=process_rules,
        visibility_grace_s=visibility_grace_s,
    )


def evaluate_brick_failure_gates(world, step, learned_rules, process_rules=None):
    import telemetry_brick
    return telemetry_brick.evaluate_failure_gates(world, step, learned_rules, process_rules=process_rules)


def brick_success_gate_entries(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    import telemetry_brick
    return telemetry_brick.success_gate_entries(
        world,
        step,
        learned_rules,
        process_rules=process_rules,
        visibility_grace_s=visibility_grace_s,
    )


def brick_success_gate_bounds(process_rules, learned_rules, step):
    import telemetry_brick
    return telemetry_brick.success_gate_bounds(process_rules, learned_rules, step)


def evaluate_wall_start_gates(world, step, envelope):
    import telemetry_wall
    return telemetry_wall.evaluate_start_gates(world, step, envelope)


def evaluate_wall_success_gates(world, step, envelope):
    import telemetry_wall
    return telemetry_wall.evaluate_success_gates(world, step, envelope)


def evaluate_wall_failure_gates(world, step, envelope):
    import telemetry_wall
    return telemetry_wall.evaluate_failure_gates(world, step, envelope)


def evaluate_robot_start_gates(world, step, learned_rules, process_rules=None):
    import telemetry_robot
    return telemetry_robot.evaluate_start_gates(world, step, learned_rules, process_rules=process_rules)


def evaluate_robot_success_gates(world, step, learned_rules, process_rules=None):
    import telemetry_robot
    return telemetry_robot.evaluate_success_gates(world, step, learned_rules, process_rules=process_rules)


def evaluate_robot_failure_gates(world, step, learned_rules, process_rules=None):
    import telemetry_robot
    return telemetry_robot.evaluate_failure_gates(world, step, learned_rules, process_rules=process_rules)


def derive_start_gates(success_segments):
    import telemetry_process
    return telemetry_process.derive_start_gates(success_segments)


def derive_success_gates(success_segments, scale_by_step=None, step_rules=None):
    import telemetry_process
    return telemetry_process.derive_success_gates(
        success_segments,
        scale_by_step=scale_by_step,
        step_rules=step_rules,
    )


def refine_success_gates_with_failures(success_gates, fail_segments, step_rules=None):
    import telemetry_process
    return telemetry_process.refine_success_gates_with_failures(
        success_gates,
        fail_segments,
        step_rules=step_rules,
    )


def success_gate_metrics_for_step(metrics, step, step_rules=None):
    import telemetry_process
    return telemetry_process.success_gate_metrics_for_step(metrics, step, step_rules=step_rules)
