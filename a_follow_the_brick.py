#!/usr/bin/env python3
"""Follow-the-brick: continuously tracks a brick, never pausing.

At startup the brick's position is captured as the reference ('happy place').
Within ±HAPPY_TOL_MM of that reference the robot holds still.  Beyond that it
arc-turns and drives to close the gaps using short overlapping motor pulses so
motion is smooth and continuous rather than stop-start.

Press Ctrl-C to stop.
"""

from __future__ import annotations

import logging
import time

from helper_brick_detector_yolo import (
    BrickDetector,
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
)
from helper_robot_control import Robot
from helper_turn_drive_motion import build_turn_drive_motion_plan
import telemetry_robot as _telemetry_robot

# ── tuning ────────────────────────────────────────────────────────────────────
# Success-gate values from world_model_process.json → steps → ALIGN_BRICK → success_gates
TARGET_DIST_MM = 149.26  # dist.target
DIST_TOL_MM    = 5.0     # dist.tol
X_TOL_MM       = 5.0     # xAxis_offset_abs.tol  (centred at x=0)

SPEED_SCORE        = 1    # minimum motor speed score
CURVE_INNER_RATIO  = 0.7  # inner-wheel ratio during a forward curve (0.0 = arc-assist,
                          # 1.0 = straight); 0.7 means both wheels drive forward but
                          # the inner one runs at 70% speed, curving toward the brick
PULSE_MS       = 100     # motor pulse duration — kept short so pulses overlap
                         # and the robot never fully stops between loop ticks
LOOP_S         = 0.05    # control loop interval (20 Hz)
WARMUP_READS   = 16      # reads to warm the camera pipeline before capture

CROWN_PROFILE_TUNING = {
    "confidence": 0.25,
    "smoothing_alpha": 0.15,
    "hsv_enabled": True,
    "hsv_erode_iterations": 1,
    "hsv_lower": list(CYAN_HSV_BALANCED_LOWER),
    "hsv_upper": list(CYAN_HSV_BALANCED_UPPER),
    "hsv_cyan_coverage_min": 0.12,
    "hsv_min_area_ratio": 0.07,
    "shape_gate_mode": "negative_cutouts",
    "negative_cutout_cyan_fill_max": 0.20,
    "negative_cutout_ring_cyan_min": 0.58,
    "negative_cutout_ring_dilate_px": 4,
    "negative_cutout_min_area_px": 24.0,
    "negative_cutout_triangle_side_ratio_max": 1.75,
    "negative_cutout_triangle_angle_spread_max_deg": 60.0,
    "negative_cutout_triangle_overlap_min": 0.75,
    "negative_cutout_pair_x_axis_max_angle_deg": 10.0,
    "conf_gate_pct": 75.0,
    "trust_detector_boxes": False,
    "require_cyan_shape": True,
    "depth_source_mode": "pinhole",  # use apparent brick size, not stereo depth
}
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("follow_the_brick")


def _curve_forward(robot: Robot, cmd: str) -> None:
    """Both wheels drive forward; inner wheel at CURVE_INNER_RATIO speed.

    This closes the distance gap and corrects x-axis offset in a single movement —
    outer tread full speed, inner tread at CURVE_INNER_RATIO, curving toward the brick.
    """
    plan = build_turn_drive_motion_plan(
        cmd=cmd,
        score=SPEED_SCORE,
        hold_duration_ms=PULSE_MS,
        profile_override={
            "drive_mode": "forward",
            "inner_ratio": CURVE_INNER_RATIO,
            "outer_ratio": 1.0,
        },
    )
    if not isinstance(plan, dict):
        return
    robot.send_custom_actions_pwm(
        str(plan.get("cmd") or cmd),
        plan.get("actions") or [],
        duration_ms=int(plan.get("duration_ms") or PULSE_MS),
    )


def _drive(robot: Robot, direction: str) -> None:
    """Straight drive: 'f' forward, 'b' backward."""
    _, pwm, _, _ = _telemetry_robot.speed_power_pwm_for_cmd(direction, SPEED_SCORE)
    robot.send_command_pwm(direction, int(pwm), duration_ms=PULSE_MS)


def _warmup(vision: BrickDetector) -> None:
    log.info("Warming up camera pipeline (%d reads)...", WARMUP_READS)
    for _ in range(WARMUP_READS):
        try:
            vision.read()
        except Exception:
            pass
        time.sleep(0.06)
    log.info("Camera ready.")


INVISIBLE_HOLD_FRAMES = 2   # ignore up to this many consecutive not-visible frames
INVISIBLE_STOP_FRAMES = 3   # stop motors after this many consecutive not-visible frames


def _follow_loop(vision: BrickDetector, robot: Robot, duration_s: float = 15.0) -> None:
    last_action = ""
    print_ticker = 0
    miss_count = 0
    last_result = None
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        loop_start = time.monotonic()

        result = None
        try:
            result = vision.read()
        except Exception as exc:
            log.warning("Vision read error: %s", exc)

        found = isinstance(result, tuple) and len(result) >= 1 and bool(result[0])

        if found:
            miss_count = 0
            last_result = result
        else:
            miss_count += 1
            if miss_count <= INVISIBLE_HOLD_FRAMES and last_result is not None:
                result = last_result   # coast on last good reading
                found = True
            elif miss_count == INVISIBLE_STOP_FRAMES:
                print("[FOLLOW] NOT VISIBLE x3 — stopping", flush=True)
                try:
                    robot.stop()
                except Exception:
                    pass
            else:
                if last_action != "NO_VIS":
                    print("[FOLLOW] NOT VISIBLE", flush=True)
                last_action = "NO_VIS"
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue

        if found:
            _, _angle, dist_mm, x_mm, conf, _cam_h, _above, _below = result[:8]
            dist_mm = float(dist_mm)
            x_mm    = float(x_mm)

            dist_err = dist_mm - TARGET_DIST_MM  # + = too far, - = too close
            x_err    = x_mm                       # + = brick right of centre

            x_ok    = abs(x_err)    <= X_TOL_MM
            dist_ok = abs(dist_err) <= DIST_TOL_MM

            if x_ok and dist_ok:
                action = "HAPPY"
                # No motor command — last PULSE_MS pulse expires and robot coasts to stop
            elif dist_err < -DIST_TOL_MM:
                # Brick too close — back up straight; re-centre x once there's room.
                action = "BCK"
                _drive(robot, "b")
            elif not x_ok:
                # x off-centre (and not too close): curve forward — both wheels drive,
                # inner at CURVE_INNER_RATIO, simultaneously closing dist + x gaps.
                cmd    = "r" if x_err > 0 else "l"
                action = f"CURVE_{cmd.upper()}"
                _curve_forward(robot, cmd)
            else:
                # x centred, just too far — drive straight forward.
                action = "FWD"
                _drive(robot, "f")

            # Print on state change or every 20 ticks (~1 s) to avoid flooding
            print_ticker += 1
            if action != last_action or print_ticker >= 20:
                print(
                    f"[FOLLOW] {action:<10} dist_err={dist_err:+.1f}mm  "
                    f"x_err={x_err:+.1f}mm  conf={conf:.0f}%",
                    flush=True,
                )
                print_ticker = 0
            last_action = action
        elapsed = time.monotonic() - loop_start
        remaining = LOOP_S - elapsed
        if remaining > 0:
            time.sleep(remaining)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    vision = BrickDetector(debug=True)
    vision.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))

    robot = Robot()

    print(
        f"[FOLLOW] Target: dist={TARGET_DIST_MM:.0f}mm ±{DIST_TOL_MM:.0f}mm  "
        f"x=0 ±{X_TOL_MM:.0f}mm  |  pulse: {PULSE_MS}ms  |  loop: {int(1/LOOP_S)}Hz  |  score: {SPEED_SCORE}",
        flush=True,
    )
    print("[FOLLOW] Press Ctrl-C to stop.", flush=True)

    _warmup(vision)

    try:
        _follow_loop(vision, robot, duration_s=15.0)
        print("[FOLLOW] 15 s elapsed — done.", flush=True)
    except KeyboardInterrupt:
        print("\n[FOLLOW] Stopped.", flush=True)
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        try:
            vision.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
