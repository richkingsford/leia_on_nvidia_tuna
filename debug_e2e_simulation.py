#!/usr/bin/env python3
"""
Offline end-to-end preflight for Leia's full process sequence.

The goal of this script is to catch wiring/config/demo-data issues before
running on hardware. It intentionally uses helper modules used by runtime:
  - helper_manual_config
  - helper_demo_log_utils
  - helper_gate_utils
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import helper_gate_utils
import telemetry_robot
from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_step_label
from helper_manual_config import load_manual_training_config

COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_WHITE = "\033[37m"
COLOR_RESET = "\033[0m"

ACTION_CMD_MAP = {
    "forward": "f",
    "backward": "b",
    "left_turn": "l",
    "right_turn": "r",
    "mast_up": "u",
    "mast_down": "d",
    "f": "f",
    "b": "b",
    "l": "l",
    "r": "r",
    "u": "u",
    "d": "d",
}

DEFAULT_STEP_ORDER = [
    "FIND_WALL",
    "EXIT_WALL",
    "FIND_BRICK",
    "ALIGN_BRICK",
    "SCOOP",
    "FIND_WALL2",
    "POSITION_BRICK",
    "PLACE",
    "RETREAT",
]


@dataclass
class LogEntry:
    color: str
    text: str


class Logger:
    def __init__(self, emit: bool = True) -> None:
        self.emit = emit
        self.lines: List[LogEntry] = []
        self.warning_count = 0

    def _add(self, color: str, text: str) -> None:
        self.lines.append(LogEntry(color=color, text=text))
        if self.emit:
            print(f"{color}{text}{COLOR_RESET}")

    def green(self, text: str) -> None:
        self._add(COLOR_GREEN, text)

    def red(self, text: str) -> None:
        self._add(COLOR_RED, text)

    def yellow(self, text: str) -> None:
        self.warning_count += 1
        self._add(COLOR_YELLOW, text)

    def white(self, text: str) -> None:
        self._add(COLOR_WHITE, text)


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _measurement_for_gate(stats: dict) -> Optional[float]:
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None:
        return float(target)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and max_val is not None:
        return (float(min_val) + float(max_val)) / 2.0
    if min_val is not None:
        return float(min_val)
    if max_val is not None:
        return float(max_val)
    if tol is not None:
        return float(tol)
    return None


def build_success_measurement(success_gates: Dict[str, dict]) -> Dict[str, object]:
    measurement: Dict[str, object] = {}
    for metric, stats in success_gates.items():
        if metric == "visible":
            min_val = stats.get("min")
            max_val = stats.get("max")
            if isinstance(min_val, bool):
                measurement["visible"] = bool(min_val)
            elif isinstance(max_val, bool):
                measurement["visible"] = bool(max_val)
            else:
                measurement["visible"] = True
            continue
        value = _measurement_for_gate(stats)
        if value is None:
            continue
        if metric in ("angle_abs", "angle"):
            measurement["angle"] = value
        elif metric in ("xAxis_offset_abs", "xAxis_offset", "x_axis"):
            measurement["x_axis"] = value
            measurement["offset_x"] = value
        elif metric in ("dist", "distance"):
            measurement["dist"] = value
        elif metric == "confidence":
            measurement["confidence"] = value
        elif metric == "lift_height":
            measurement["lift_height"] = value
    measurement.setdefault("visible", True)
    return measurement


def break_measurement(measurement: Dict[str, object], success_gates: Dict[str, dict]) -> Dict[str, object]:
    broken = dict(measurement)
    for metric, stats in success_gates.items():
        if metric == "visible":
            broken["visible"] = not bool(measurement.get("visible"))
            return broken
        if metric in ("dist", "distance"):
            base = _as_float(measurement.get("dist")) or 0.0
            broken["dist"] = base + max(5.0, float(stats.get("tol", 1.0)) * 2.0)
            return broken
        if metric in ("xAxis_offset_abs", "xAxis_offset", "x_axis"):
            base = _as_float(measurement.get("x_axis")) or 0.0
            broken["x_axis"] = base + max(5.0, float(stats.get("tol", 1.0)) * 2.0)
            broken["offset_x"] = broken["x_axis"]
            return broken
        if metric in ("angle_abs", "angle"):
            base = _as_float(measurement.get("angle")) or 0.0
            broken["angle"] = base + max(5.0, float(stats.get("tol", 1.0)) * 2.0)
            return broken
        if metric == "confidence":
            base = _as_float(measurement.get("confidence")) or 0.0
            broken["confidence"] = max(0.0, base - max(10.0, float(stats.get("tol", 1.0)) * 3.0))
            return broken
    broken["visible"] = False
    return broken


def measurement_from_state(state: dict) -> Dict[str, object]:
    brick = (state or {}).get("brick") or {}
    visible = bool(brick.get("visible"))
    measurement = {
        "visible": visible,
        "angle": _as_float(brick.get("angle")),
        "x_axis": _as_float(brick.get("x_axis")),
        "offset_x": _as_float(brick.get("offset_x")),
        "dist": _as_float(brick.get("dist")),
        "confidence": _as_float(brick.get("confidence")),
        "lift_height": _as_float(state.get("lift_height")),
    }
    if measurement["x_axis"] is None and measurement["offset_x"] is not None:
        measurement["x_axis"] = measurement["offset_x"]
    if measurement["offset_x"] is None and measurement["x_axis"] is not None:
        measurement["offset_x"] = measurement["x_axis"]
    if not visible:
        measurement["angle"] = None
        measurement["x_axis"] = None
        measurement["offset_x"] = None
        measurement["dist"] = None
        measurement["confidence"] = None
    return measurement


def collect_segments(logs: List[Tuple[Path, List[dict]]]) -> Tuple[Dict[str, Dict[str, List[dict]]], Dict[str, set]]:
    segments_by_obj: Dict[str, Dict[str, List[dict]]] = {}
    attempt_types: Dict[str, set] = {}
    for _, rows in logs:
        for seg in extract_attempt_segments(rows):
            step_name = normalize_step_label(seg.get("step"))
            seg_type = seg.get("type")
            if not step_name or not seg_type:
                continue
            attempt_types.setdefault(step_name, set()).add(seg_type)
            segments_by_obj.setdefault(step_name, {}).setdefault(seg_type, []).append(seg)
    return segments_by_obj, attempt_types


def select_demo_segment(
    segments_by_obj: Dict[str, Dict[str, List[dict]]],
    step_name: str,
    nominal_only: bool,
) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    obj_key = normalize_step_label(step_name)
    if not obj_key:
        return None, None, "invalid step label"

    if nominal_only:
        nominal = segments_by_obj.get(obj_key, {}).get("NOMINAL", [])
        if len(nominal) > 1:
            return None, None, f"{len(nominal)} nominal demos found; expected exactly 1"

    prefer = ["NOMINAL", "SUCCESS"] if nominal_only else ["SUCCESS", "NOMINAL"]
    candidates: List[dict] = []
    selected_type: Optional[str] = None
    for seg_type in prefer:
        bucket = segments_by_obj.get(obj_key, {}).get(seg_type, [])
        if bucket:
            candidates = list(bucket)
            selected_type = seg_type
            break

    if not candidates:
        return None, None, "no usable demo segment"

    def score(seg: dict) -> Tuple[int, float]:
        events = seg.get("events") or []
        duration = 0.0
        if seg.get("start") is not None and seg.get("end") is not None:
            duration = float(seg["end"]) - float(seg["start"])
        return len(events), duration

    candidates.sort(key=score, reverse=True)
    return candidates[0], selected_type, None


def decode_actions(events: List[dict]) -> Tuple[List[dict], List[str]]:
    actions: List[dict] = []
    unknown_commands: List[str] = []
    for evt in events or []:
        cmd_name = None
        score = None
        if evt.get("type") == "action":
            cmd_name = evt.get("command")
            score = evt.get("speedScore")
        elif evt.get("type") == "event":
            payload = evt.get("event") or {}
            cmd_name = payload.get("type")
            score = payload.get("speedScore")
        if cmd_name is None:
            continue
        cmd = ACTION_CMD_MAP.get(str(cmd_name))
        if cmd is None:
            unknown_commands.append(str(cmd_name))
            continue
        score_val = None
        if score is not None:
            try:
                score_val = int(score)
            except (TypeError, ValueError):
                score_val = None
        actions.append({"cmd": cmd, "speed_score": score_val})
    return actions, unknown_commands


def _set_world_step(world: telemetry_robot.WorldModel, step_name: str) -> None:
    key = normalize_step_label(step_name)
    if key in telemetry_robot.StepState.__members__:
        world.step_state = telemetry_robot.StepState[key]


def apply_state_to_world(world: telemetry_robot.WorldModel, step_name: str, state: dict) -> None:
    _set_world_step(world, step_name)
    pose = (state or {}).get("robot_pose") or {}
    world.x = _as_float(pose.get("x")) or 0.0
    world.y = _as_float(pose.get("y")) or 0.0
    world.theta = _as_float(pose.get("theta")) or 0.0

    brick = (state or {}).get("brick") or {}
    visible = bool(brick.get("visible"))
    dist = _as_float(brick.get("dist"))
    angle = _as_float(brick.get("angle"))
    offset = _as_float(brick.get("x_axis"))
    if offset is None:
        offset = _as_float(brick.get("offset_x"))
    conf = _as_float(brick.get("confidence"))
    world.brick["visible"] = visible
    world.brick["dist"] = dist if dist is not None else 0.0
    world.brick["angle"] = angle if angle is not None else 0.0
    world.brick["offset_x"] = offset if offset is not None else 0.0
    world.brick["x_axis"] = offset if offset is not None else 0.0
    world.brick["confidence"] = conf if conf is not None else 0.0
    world.brick["brickAbove"] = bool(brick.get("brickAbove"))
    world.brick["brickBelow"] = bool(brick.get("brickBelow"))
    world.brick["held"] = bool(brick.get("held"))
    if visible:
        world.last_visible_time = time.time()
        world.last_seen_angle = world.brick["angle"]
        world.last_seen_dist = world.brick["dist"]
        world.last_seen_offset_x = world.brick["offset_x"]
        world.last_seen_x_axis = world.brick["x_axis"]
        world.last_seen_confidence = world.brick["confidence"]

    lift = _as_float((state or {}).get("lift_height"))
    if lift is not None:
        world.lift_height = lift

    wall_origin = (state or {}).get("wall_origin")
    if isinstance(wall_origin, dict):
        wall_x = _as_float(wall_origin.get("x"))
        wall_y = _as_float(wall_origin.get("y"))
        wall_theta = _as_float(wall_origin.get("theta"))
        if wall_x is not None and wall_y is not None:
            world.wall["origin"] = {
                "x": wall_x,
                "y": wall_y,
                "theta": wall_theta if wall_theta is not None else world.wall.get("angle_deg", 0.0),
            }
            world.wall["valid"] = True
            world.wall["contradiction_reason"] = None


def _progress_pct(value: Optional[float]) -> int:
    if value is None:
        return 0
    return int(max(0.0, min(1.0, value)) * 100.0)


def validate_manual_config(logger: Logger) -> bool:
    cfg = load_manual_training_config()
    logger.white("Loaded manual training config via helper_manual_config.")
    numeric_requirements = {
        "log_rate_hz": 1,
        "command_rate_hz": 1,
        "heartbeat_timeout": 0,
        "stream_fps": 1,
        "stream_jpeg_quality": 1,
    }
    ok = True
    for key, min_value in numeric_requirements.items():
        val = cfg.get(key)
        num = _as_float(val)
        if num is None or num <= min_value:
            ok = False
            logger.red(f"[CONFIG] {key} invalid ({val}); expected > {min_value}.")
    port = cfg.get("stream_port")
    port_i = None
    try:
        port_i = int(port)
    except (TypeError, ValueError):
        pass
    if port_i is None or not (1 <= port_i <= 65535):
        ok = False
        logger.red(f"[CONFIG] stream_port invalid ({port}); expected 1..65535.")
    if ok:
        logger.green("[CONFIG] Manual training config looks valid.")
    return ok


def build_step_sequence(steps_cfg: Dict[str, dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    try:
        runtime_steps = telemetry_robot.step_sequence()
        runtime_names = [normalize_step_label(step.value) for step in runtime_steps]
    except Exception:
        runtime_names = []
    for name in runtime_names:
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in steps_cfg.keys():
        key = normalize_step_label(name)
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    for name in DEFAULT_STEP_ORDER:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def check_gate_helpers_consistency(step_name: str, success_gates: Dict[str, dict], logger: Logger) -> bool:
    if not success_gates:
        return True
    happy = build_success_measurement(success_gates)
    happy_ok = helper_gate_utils.gate_satisfied(happy, success_gates)
    broken = break_measurement(happy, success_gates)
    broken_ok = helper_gate_utils.gate_satisfied(broken, success_gates)
    if not happy_ok:
        logger.red(f"[{step_name}] helper_gate_utils rejected the synthetic happy measurement.")
        return False
    if broken_ok:
        logger.red(f"[{step_name}] helper_gate_utils accepted a deliberately broken measurement.")
        return False
    logger.green(f"[{step_name}] helper_gate_utils happy/edge gate checks passed.")
    return True


def check_step_segment(
    step_name: str,
    step_cfg: Dict[str, dict],
    segment: dict,
    segment_type: str,
    logger: Logger,
) -> bool:
    ok = True
    nominal_only = bool(step_cfg.get("nominalDemosOnly"))
    success_gates = (step_cfg or {}).get("success_gates") or {}
    start_gates = (step_cfg or {}).get("start_gates") or {}
    states = segment.get("states") or []
    events = segment.get("events") or []

    logger.white(
        f"[{step_name}] Segment {segment_type}: {len(states)} states, {len(events)} events "
        f"(nominalOnly={nominal_only})."
    )

    if nominal_only and segment_type != "NOMINAL":
        ok = False
        logger.red(f"[{step_name}] Expected NOMINAL demo but selected {segment_type}.")

    actions, unknown_commands = decode_actions(events)
    for cmd_name in sorted(set(unknown_commands)):
        ok = False
        logger.red(f"[{step_name}] Unknown logged command '{cmd_name}' (cannot map to robot cmd).")

    if not actions and events:
        logger.yellow(f"[{step_name}] Events exist but no motor actions were decoded.")

    if nominal_only and not actions:
        ok = False
        logger.red(f"[{step_name}] Nominal step is missing replayable actions.")

    valid_scores = set(telemetry_robot.SCORE_POWER_PWM.keys())
    for idx, action in enumerate(actions, start=1):
        cmd = action.get("cmd")
        score = action.get("speed_score")
        if score is None:
            if nominal_only:
                ok = False
                logger.red(f"[{step_name}] Action #{idx} ({cmd}) missing speedScore.")
            else:
                logger.yellow(f"[{step_name}] Action #{idx} ({cmd}) missing speedScore.")
            continue
        if score not in valid_scores:
            ok = False
            logger.red(f"[{step_name}] Action #{idx} ({cmd}) uses undefined speedScore={score}.")
            continue
        speed, pwm, _, duration_ms = telemetry_robot.speed_power_pwm_for_cmd(cmd, score)
        if pwm <= 0 or speed <= 0 or duration_ms <= 0:
            ok = False
            logger.red(f"[{step_name}] Invalid robot profile for cmd={cmd} score={score}.")

    if success_gates and not states:
        ok = False
        logger.red(f"[{step_name}] Success gates exist but the selected segment has no states.")

    if not nominal_only and not success_gates:
        ok = False
        logger.red(f"[{step_name}] Missing success_gates in world_model_process.json.")

    if start_gates and states:
        first_states = states[: min(3, len(states))]
        start_hits = sum(
            1 for state in first_states if helper_gate_utils.gate_satisfied(measurement_from_state(state), start_gates)
        )
        if start_hits == 0:
            logger.yellow(f"[{step_name}] Start gates were not satisfied in the first sampled states.")
        else:
            logger.green(f"[{step_name}] Start gates observed at segment start ({start_hits}/{len(first_states)}).")

    if success_gates and states:
        satisfied_count = 0
        best_progress = 0.0
        for state in states:
            measurement = measurement_from_state(state)
            progress = helper_gate_utils.step_progress(measurement, success_gates)
            if progress is not None:
                best_progress = max(best_progress, progress)
            if helper_gate_utils.gate_satisfied(measurement, success_gates):
                satisfied_count += 1

        if satisfied_count <= 0:
            ok = False
            logger.red(
                f"[{step_name}] No recorded state met success_gates "
                f"(best progress={_progress_pct(best_progress)}%)."
            )
        else:
            logger.green(
                f"[{step_name}] Recorded success_gates reached on {satisfied_count}/{len(states)} states."
            )

    if not check_gate_helpers_consistency(step_name, success_gates, logger):
        ok = False

    if success_gates and states:
        world = telemetry_robot.WorldModel()
        _set_world_step(world, step_name)
        combined_success = False
        for state in states:
            apply_state_to_world(world, step_name, state)
            brick_ok = helper_gate_utils.evaluate_brick_success_gates(
                world,
                step_name,
                {},
                process_rules=world.process_rules,
                visibility_grace_s=0.0,
            ).ok
            wall_ok = helper_gate_utils.evaluate_wall_success_gates(world, step_name, world.wall_envelope).ok
            robot_ok = helper_gate_utils.evaluate_robot_success_gates(
                world,
                step_name,
                {},
                process_rules=world.process_rules,
            ).ok
            if brick_ok and wall_ok and robot_ok:
                combined_success = True
                break
        if not combined_success:
            ok = False
            logger.red(
                f"[{step_name}] Telemetry gate evaluators never agreed on SUCCESS for recorded states."
            )
        else:
            logger.green(f"[{step_name}] Telemetry gate evaluators reached SUCCESS on recorded data.")

    return ok


def _check_learning_smoke(segments_by_obj: Dict[str, Dict[str, List[dict]]], logger: Logger) -> bool:
    try:
        import helper_learning
    except Exception as exc:
        logger.yellow(f"[LEARNING] helper_learning unavailable ({exc}). Skipping policy smoke test.")
        return True

    try:
        policy = helper_learning.BehavioralCloningPolicy(k=3)
        # helper_learning emits its own prints; keep this script's output concise.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            policy.train(segments_by_obj)
        trained_steps = sorted((policy.policy_by_obj or {}).keys())
        if trained_steps:
            logger.green(
                f"[LEARNING] helper_learning trained policy for {len(trained_steps)} step(s): "
                f"{', '.join(trained_steps)}."
            )
        else:
            logger.yellow("[LEARNING] helper_learning found no trainable state/action pairs.")
        return True
    except Exception as exc:
        logger.red(f"[LEARNING] helper_learning smoke test failed: {exc}")
        return False


def collect_simulation_logs(
    emit: bool = True,
    demos_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    strict: bool = False,
) -> List[LogEntry]:
    logger = Logger(emit=emit)
    ok = True

    root = Path(__file__).resolve().parent
    demos_path = Path(demos_dir) if demos_dir is not None else (root / "demos")

    logger.green("Starting E2E preflight simulation for Leia.")
    logger.white("Using helper_manual_config, helper_demo_log_utils, and helper_gate_utils.")

    if not validate_manual_config(logger):
        ok = False

    steps_cfg = helper_gate_utils.load_process_steps()
    if not steps_cfg:
        logger.red("[PROCESS] No steps found in world_model_process.json.")
        ok = False
        logger.red("E2E preflight failed early.")
        return logger.lines

    sequence = build_step_sequence(steps_cfg)
    logger.white(f"[PROCESS] Step sequence: {' -> '.join(sequence)}")

    logs = load_demo_logs(demos_path, session_name=session_name)
    if not logs:
        logger.red(f"[DEMOS] No demo logs found in {demos_path}.")
        ok = False
        logger.red("E2E preflight failed early.")
        return logger.lines

    logger.green(f"[DEMOS] Loaded {len(logs)} demo log(s) from {demos_path}.")
    segments_by_obj, attempt_types = collect_segments(logs)
    segment_total = sum(len(items) for per_step in segments_by_obj.values() for items in per_step.values())
    logger.white(f"[DEMOS] Extracted {segment_total} attempt segment(s).")

    success_segments = {
        step: buckets.get("SUCCESS", [])
        for step, buckets in segments_by_obj.items()
        if buckets.get("SUCCESS")
    }
    try:
        derived_start = helper_gate_utils.derive_start_gates(success_segments)
        derived_success = helper_gate_utils.derive_success_gates(success_segments, step_rules=steps_cfg)
        logger.green(
            f"[GATES] Derived start gates for {len(derived_start)} step(s), "
            f"success gates for {len(derived_success)} step(s)."
        )
    except Exception as exc:
        derived_success = {}
        logger.red(f"[GATES] Failed to derive gates from demos: {exc}")
        ok = False

    if not _check_learning_smoke(segments_by_obj, logger):
        ok = False

    sequence_set = set(sequence)
    missing_in_model = [step for step in sequence if step not in steps_cfg]
    if missing_in_model:
        ok = False
        logger.red(f"[PROCESS] Sequence step(s) missing from process config: {', '.join(missing_in_model)}")

    for idx, step_name in enumerate(sequence, start=1):
        step_cfg = (steps_cfg.get(step_name) or {})
        nominal_only = bool(step_cfg.get("nominalDemosOnly"))
        segment, seg_type, reason = select_demo_segment(segments_by_obj, step_name, nominal_only)

        logger.white(f"Step {idx}/{len(sequence)}: {step_name}")
        if segment is None:
            ok = False
            logger.red(f"[{step_name}] Missing segment: {reason or 'unknown reason'}")
            continue

        if not check_step_segment(step_name, step_cfg, segment, str(seg_type), logger):
            ok = False

        derived_step_gates = (derived_success or {}).get(step_name, {})
        configured_metrics = set((step_cfg.get("success_gates") or {}).keys())
        derived_metrics = set(derived_step_gates.keys())
        if configured_metrics and derived_metrics and configured_metrics != derived_metrics:
            logger.yellow(
                f"[{step_name}] Config success metric set {sorted(configured_metrics)} "
                f"differs from demo-derived {sorted(derived_metrics)}."
            )

        types_seen = sorted(attempt_types.get(step_name, []))
        if not types_seen:
            logger.yellow(f"[{step_name}] No attempt type markers found in demos.")
        else:
            logger.white(f"[{step_name}] Demo attempt types seen: {', '.join(types_seen)}")

    extra_demo_steps = sorted(set(segments_by_obj.keys()) - sequence_set)
    if extra_demo_steps:
        logger.yellow(
            "[DEMOS] Steps found in demos but not in sequence: "
            + ", ".join(extra_demo_steps)
        )

    if strict and logger.warning_count > 0:
        ok = False
        logger.red(f"[STRICT] Treating {logger.warning_count} warning(s) as failures.")

    if ok:
        logger.green("E2E preflight complete. All sequence steps passed checks.")
    else:
        logger.red("E2E preflight complete with failures. Fix issues before powering robot.")
    return logger.lines


def run_preflight(
    emit: bool = True,
    demos_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    strict: bool = False,
) -> bool:
    logs = collect_simulation_logs(
        emit=emit,
        demos_dir=demos_dir,
        session_name=session_name,
        strict=strict,
    )
    return bool(logs) and logs[-1].color == COLOR_GREEN


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline end-to-end preflight for Leia's full sequence."
    )
    parser.add_argument("--session", default=None, help="Optional demo session path/name.")
    parser.add_argument(
        "--demos",
        default=None,
        help="Optional demos directory. Defaults to ./demos next to this script.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Collect results without printing line-by-line logs.",
    )
    args = parser.parse_args()

    demos_dir = Path(args.demos) if args.demos else None
    success = run_preflight(
        emit=not args.quiet,
        demos_dir=demos_dir,
        session_name=args.session,
        strict=args.strict,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
