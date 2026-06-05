#!/usr/bin/env python3
"""Run Leia's straight-line speed ramp test.

Dry-run is the default. Add --execute only when the robot is clear to move.
"""

from __future__ import annotations

import argparse
import os

from helper_robot_control import Robot
from helper_speed_test import (
    DEFAULT_CEILING_PWM_SCALE,
    DEFAULT_END_SCORE,
    DEFAULT_FLOOR_PWM_SCALE,
    DEFAULT_INTERVAL_MS,
    DEFAULT_LOG_DIR,
    DEFAULT_PHASE_DURATION_S,
    build_speed_test_sequence,
    run_speed_test,
    write_speed_test_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ramp forward speed, then run the reverse-direction ramp.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send motor commands. Without this flag, only print the planned table.",
    )
    parser.add_argument(
        "--port",
        help="Optional serial device path. Sets LEIA_SERIAL_PORT before connecting.",
    )
    parser.add_argument(
        "--phase-s",
        type=float,
        default=DEFAULT_PHASE_DURATION_S,
        help="Approximate seconds for each ramp phase.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Cadence for sending the next speed command.",
    )
    parser.add_argument(
        "--reverse-mode",
        choices=("backward", "down"),
        default="backward",
        help="Use backward for the second ramp, or down for a forward ramp-down.",
    )
    parser.add_argument(
        "--end-score",
        type=int,
        default=DEFAULT_END_SCORE,
        help="Upper speed-score ceiling for each ramp.",
    )
    parser.add_argument(
        "--start-pwm-scale",
        type=float,
        default=None,
        help="Deprecated alias for --floor-pwm-scale.",
    )
    parser.add_argument(
        "--floor-pwm-scale",
        type=float,
        default=DEFAULT_FLOOR_PWM_SCALE,
        help="Multiplier for the low end of the PWM ramp.",
    )
    parser.add_argument(
        "--ceiling-pwm-scale",
        type=float,
        default=DEFAULT_CEILING_PWM_SCALE,
        help="Multiplier for the high end of the PWM ramp.",
    )
    parser.add_argument(
        "--sample-dist",
        dest="sample_dist",
        action="store_true",
        default=None,
        help="Sample brick distance at every pulse. Default: on for --execute, off for dry-run.",
    )
    parser.add_argument(
        "--no-sample-dist",
        dest="sample_dist",
        action="store_false",
        help="Do not sample brick distance.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for JSON logs and PWM chart HTML.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional short label included in the saved log/chart filenames.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not save JSON logs or chart HTML for this run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.port:
        os.environ["LEIA_SERIAL_PORT"] = str(args.port)
    floor_pwm_scale = (
        float(args.start_pwm_scale)
        if args.start_pwm_scale is not None
        else float(args.floor_pwm_scale)
    )

    sequence = build_speed_test_sequence(
        phase_duration_s=float(args.phase_s),
        interval_ms=int(args.interval_ms),
        reverse_mode=str(args.reverse_mode),
        end_score=int(args.end_score),
        floor_pwm_scale=float(floor_pwm_scale),
        ceiling_pwm_scale=float(args.ceiling_pwm_scale),
    )

    robot = None
    vision = None
    try:
        if args.execute:
            robot = Robot()
        sample_dist = bool(args.execute) if args.sample_dist is None else bool(args.sample_dist)
        vision_sampler = None
        if sample_dist:
            from helper_brick_detector_native_oak import BrickDetector

            vision = BrickDetector(debug=False)

            def vision_sampler():
                found, _angle, dist, offset_x, conf, cam_h, _above, _below = vision.read()
                return {
                    "found": bool(found),
                    "dist_mm": float(dist) if bool(found) else None,
                    "x_mm": float(offset_x) if bool(found) else None,
                    "y_mm": float(cam_h) if bool(found) else None,
                    "conf": float(conf) if bool(found) else None,
                    "status": getattr(vision, "last_status", None),
                    "source": getattr(vision, "last_bbox_distance_source", None),
                }

        result = run_speed_test(
            robot=robot,
            execute=bool(args.execute),
            sequence=sequence,
            vision_sampler=vision_sampler,
        )
        if not bool(args.no_log):
            artifacts = write_speed_test_artifacts(
                result,
                log_dir=str(args.log_dir),
                run_label=args.run_label or ("execute" if args.execute else "dry_run"),
                metadata={
                    "phase_s": float(args.phase_s),
                    "interval_ms": int(args.interval_ms),
                    "reverse_mode": str(args.reverse_mode),
                    "end_score": int(args.end_score),
                    "floor_pwm_scale": float(floor_pwm_scale),
                    "ceiling_pwm_scale": float(args.ceiling_pwm_scale),
                    "sample_dist": bool(sample_dist),
                },
            )
            print(f"[SPEED] JSON log saved: {artifacts['json_path']}")
            print(f"[SPEED] PWM chart saved: {artifacts['html_path']}")
        if not bool(result.get("execute")):
            print("[DRY-RUN] No motor commands were sent. Re-run with --execute when Leia is clear.")
        return 0
    finally:
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
