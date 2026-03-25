#!/usr/bin/env python3
"""
Debug probe: observe brick telemetry continuously while issuing low-speed movement.

Sequence:
- x axis: left for N seconds, then right for N seconds
- y axis: up for N seconds, then down for N seconds
- dist axis: forward for N seconds, then backward for N seconds

Samples are written to JSON in the same debug folder by default.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from helper_brick_detector_yolo import BrickDetector
from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from telemetry_process import send_robot_command, update_world_from_vision
from telemetry_robot import StepState, WorldModel


@dataclass(frozen=True)
class Phase:
    name: str
    axis: str
    cmd: str


DEFAULT_PHASES = [
    Phase("x_forward", "x", "l"),
    Phase("x_inverse", "x", "r"),
    Phase("y_forward", "y", "u"),
    Phase("y_inverse", "y", "d"),
    Phase("dist_forward", "dist", "f"),
    Phase("dist_inverse", "dist", "b"),
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Observe brick telemetry continuously while moving at low speed."
    )
    parser.add_argument(
        "--score",
        type=int,
        default=1,
        help="Speed score used for all commands (default: 1)",
    )
    parser.add_argument(
        "--phase-seconds",
        type=float,
        default=5.0,
        help="Seconds per movement phase before inverse (default: 5.0)",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        help="Observation samples per second (default: 20)",
    )
    parser.add_argument(
        "--command-min-interval-s",
        type=float,
        default=0.05,
        help="Minimum spacing between command sends (default: 0.05)",
    )
    parser.add_argument(
        "--vision",
        choices=("aruco", "cyan"),
        default="aruco",
        help="Vision backend for observation (default: aruco)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "debug_observing_while_moving.json",
        help="Output JSON path (default: debug/debug_observing_while_moving.json)",
    )
    return parser.parse_args()


def build_vision(vision_mode: str):
    if str(vision_mode).strip().lower() == "cyan":
        return BrickDetector(debug=True, speed_optimize=False)
    return ArucoBrickVision(debug=True)


def maybe_close(obj) -> None:
    if obj is None:
        return
    close_fn: Callable | None = getattr(obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def summarize_phase(samples: list[dict], commands: list[dict], phase_name: str) -> dict:
    phase_samples = [row for row in samples if row.get("phase") == phase_name]
    phase_commands = [row for row in commands if row.get("phase") == phase_name]
    visible_flags = [bool(row.get("visible")) for row in phase_samples]
    conf_values = [float(row.get("confidence", 0.0) or 0.0) for row in phase_samples]
    x_values = [float(row.get("x_axis", 0.0) or 0.0) for row in phase_samples]
    y_values = [float(row.get("y_axis", 0.0) or 0.0) for row in phase_samples]
    dist_values = [float(row.get("dist", 0.0) or 0.0) for row in phase_samples]

    return {
        "phase": phase_name,
        "sample_count": len(phase_samples),
        "command_count": len(phase_commands),
        "visible_count": sum(1 for flag in visible_flags if flag),
        "visible_rate": (
            float(sum(1 for flag in visible_flags if flag)) / float(len(visible_flags))
            if visible_flags
            else None
        ),
        "confidence_median": statistics.median(conf_values) if conf_values else None,
        "x_axis_median": statistics.median(x_values) if x_values else None,
        "y_axis_median": statistics.median(y_values) if y_values else None,
        "dist_median": statistics.median(dist_values) if dist_values else None,
    }


def main() -> int:
    args = parse_args()

    if args.phase_seconds <= 0:
        raise SystemExit("--phase-seconds must be > 0")
    if args.sample_hz <= 0:
        raise SystemExit("--sample-hz must be > 0")

    score = int(max(1, args.score))
    sample_period_s = 1.0 / float(args.sample_hz)

    robot = None
    vision = None
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK

    samples: list[dict] = []
    commands: list[dict] = []

    started_at = time.time()
    monotonic_start = time.monotonic()

    try:
        robot = Robot()
        vision = build_vision(args.vision)

        print(
            f"[OBSERVE-MOVE] Starting run: score={score}% phase_seconds={args.phase_seconds:.2f} "
            f"sample_hz={args.sample_hz:.1f} vision={args.vision}",
            flush=True,
        )

        for phase in DEFAULT_PHASES:
            print(
                f"[OBSERVE-MOVE] Phase {phase.name}: cmd={phase.cmd} axis={phase.axis} "
                f"for {args.phase_seconds:.2f}s",
                flush=True,
            )
            phase_start = time.monotonic()
            phase_end = phase_start + float(args.phase_seconds)
            next_cmd_time = phase_start
            next_sample_time = phase_start

            while True:
                now = time.monotonic()
                if now >= phase_end:
                    break

                update_world_from_vision(world, vision, log=False)
                brick = dict(getattr(world, "brick", {}) or {})
                elapsed_s = now - monotonic_start

                sample = {
                    "t": time.time(),
                    "elapsed_s": float(elapsed_s),
                    "phase": phase.name,
                    "axis": phase.axis,
                    "cmd": phase.cmd,
                    "visible": bool(brick.get("visible")),
                    "confidence": float(brick.get("confidence", 0.0) or 0.0),
                    "dist": float(brick.get("dist", 0.0) or 0.0),
                    "raw_dist": float(brick.get("raw_dist", 0.0) or 0.0),
                    "x_axis": float(brick.get("x_axis", 0.0) or 0.0),
                    "offset_x": float(brick.get("offset_x", 0.0) or 0.0),
                    "y_axis": float(brick.get("y_axis", 0.0) or 0.0),
                    "offset_y": float(brick.get("offset_y", 0.0) or 0.0),
                    "angle": float(brick.get("angle", 0.0) or 0.0),
                    "brick_above": bool(brick.get("brickAbove")),
                    "brick_below": bool(brick.get("brickBelow")),
                    "camera_fps_est": float(getattr(world, "_camera_fps", 0.0) or 0.0),
                    "pose_source": str(brick.get("pose_source") or ""),
                    "vision_backend": str(getattr(world, "_vision_backend", "") or ""),
                }
                samples.append(sample)

                if now >= next_cmd_time:
                    action_meta = send_robot_command(
                        robot,
                        world,
                        world.step_state,
                        phase.cmd,
                        speed=0.0,
                        speed_score=score,
                    )
                    command_entry = {
                        "t": time.time(),
                        "elapsed_s": float(elapsed_s),
                        "phase": phase.name,
                        "axis": phase.axis,
                        "cmd": phase.cmd,
                        "speed_score_requested": int(score),
                        "meta": action_meta,
                    }
                    commands.append(command_entry)

                    duration_ms = None
                    if isinstance(action_meta, dict):
                        try:
                            duration_ms = int(action_meta.get("duration_ms"))
                        except (TypeError, ValueError):
                            duration_ms = None
                    if duration_ms is not None and duration_ms > 0:
                        next_cmd_time = now + max(
                            float(args.command_min_interval_s),
                            (float(duration_ms) / 1000.0) * 0.9,
                        )
                    else:
                        next_cmd_time = now + float(args.command_min_interval_s)

                next_sample_time += sample_period_s
                sleep_s = max(0.0, next_sample_time - time.monotonic())
                if sleep_s > 0:
                    time.sleep(sleep_s)

            robot.stop()
            time.sleep(0.1)

        robot.stop()

    except KeyboardInterrupt:
        print("[OBSERVE-MOVE] Interrupted by user.", flush=True)
    finally:
        maybe_close(robot)
        maybe_close(vision)

    summaries = [summarize_phase(samples, commands, phase.name) for phase in DEFAULT_PHASES]
    payload = {
        "meta": {
            "script": str(Path(__file__).name),
            "started_epoch_s": float(started_at),
            "ended_epoch_s": float(time.time()),
            "total_elapsed_s": float(time.monotonic() - monotonic_start),
            "score": int(score),
            "phase_seconds": float(args.phase_seconds),
            "sample_hz": float(args.sample_hz),
            "vision": str(args.vision),
            "phase_order": [phase.name for phase in DEFAULT_PHASES],
        },
        "phase_summaries": summaries,
        "commands": commands,
        "samples": samples,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"[OBSERVE-MOVE] Wrote log: {output_path}", flush=True)
    for summary in summaries:
        phase_name = summary.get("phase")
        visible_rate = summary.get("visible_rate")
        visible_rate_text = "n/a" if visible_rate is None else f"{(100.0 * float(visible_rate)):.1f}%"
        print(
            f"[OBSERVE-MOVE] {phase_name}: samples={summary.get('sample_count')} "
            f"cmds={summary.get('command_count')} visible={visible_rate_text} "
            f"conf_med={summary.get('confidence_median')}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
