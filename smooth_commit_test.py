#!/usr/bin/env python3
"""Prototype + test a SMOOTH forward glide commit to a target distance.

Forward-only (no reversals -> no oscillation jitter), uses the stable green-edge
close-range distance as live feedback, decelerates near the target, and eases to a
stop. Safety: per-pulse timed (firmware auto-stops -> bounded runaway), hard timeout,
stop on brick loss, and never drives if already at/inside target.
"""
from __future__ import annotations

import argparse
import time

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot

FWD_PWM = 103          # SPEED_SCORE=1 wheel floor (hard rule: x/z moves at score 1)
POLL_S = 0.05


def _read(vision):
    r = follow._read_brick_measurement(vision, jump_guard=False)
    return r if r.get("confident") else None


def smooth_forward_glide(vision, robot, target_dist_mm, *,
                         stop_lead_mm=6.0, max_s=14.0, max_travel_mm=220.0,
                         far_ms=300, mid_ms=200, near_ms=120,
                         mid_gap_mm=40.0, near_gap_mm=18.0):
    start = _read(vision)
    if start is None:
        print("[GLIDE] brick not visible at start; aborting (no motion).", flush=True)
        return {"ok": False, "reason": "no_brick_at_start"}
    start_dist = float(start["dist_mm"])
    print(f"[GLIDE] start dist={start_dist:.1f}mm  target={target_dist_mm:.1f}mm  "
          f"(forward {start_dist-target_dist_mm:.0f}mm)", flush=True)
    if start_dist <= target_dist_mm + stop_lead_mm:
        print("[GLIDE] already at/inside target; no motion.", flush=True)
        return {"ok": True, "reason": "already_there", "final_dist": start_dist}

    deadline = time.monotonic() + max_s
    last_dist = start_dist
    try:
        while time.monotonic() < deadline:
            r = _read(vision)
            if r is None:
                follow._stop_robot(robot)
                print("[GLIDE] brick lost mid-commit; STOPPED.", flush=True)
                return {"ok": False, "reason": "brick_lost", "final_dist": last_dist}
            dist = float(r["dist_mm"]); x = float(r["x_mm"]); y = float(r["y_mm"])
            last_dist = dist
            gap = dist - float(target_dist_mm)
            # safety: never travel more than max_travel from start
            if (start_dist - dist) > max_travel_mm:
                follow._stop_robot(robot)
                print(f"[GLIDE] travel safety stop ({start_dist-dist:.0f}mm).", flush=True)
                return {"ok": False, "reason": "travel_safety", "final_dist": dist}
            if gap <= stop_lead_mm:
                follow._stop_robot(robot)
                print(f"[GLIDE] reached target: dist={dist:.1f} x={x:+.1f} y={y:+.1f}", flush=True)
                return {"ok": True, "reason": "target_reached", "final_dist": dist,
                        "final_x": x, "final_y": y}
            # decelerating, forward-only pulse sized to remaining gap
            if gap > mid_gap_mm:
                pulse = far_ms
            elif gap > near_gap_mm:
                pulse = mid_ms
            else:
                pulse = near_ms
            print(f"[GLIDE] FWD {pulse}ms  dist={dist:.1f} (gap {gap:+.1f}) x={x:+.1f} y={y:+.1f}", flush=True)
            # forward only; never reverse -> no oscillation jitter
            follow.guarded_send_command_pwm(
                robot, "f", FWD_PWM, duration_ms=int(pulse), reading=r,
                context="smooth_glide_commit",
            )
            time.sleep(pulse / 1000.0 + POLL_S)
        follow._stop_robot(robot)
        print(f"[GLIDE] timeout; STOPPED at dist={last_dist:.1f}", flush=True)
        return {"ok": False, "reason": "timeout", "final_dist": last_dist}
    finally:
        follow._stop_robot(robot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=76.0, help="target dist mm")
    ap.add_argument("--stop-lead", type=float, default=6.0)
    args = ap.parse_args()
    follow._set_game_profile("empty")
    vision = BrickDetector(debug=True)
    robot = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        for _ in range(6):  # let vision filter settle
            _read(vision); time.sleep(0.08)
        res = smooth_forward_glide(vision, robot, float(args.target), stop_lead_mm=float(args.stop_lead))
        print(f"[GLIDE] result: {res}", flush=True)
        return 0 if res.get("ok") else 1
    finally:
        follow._stop_robot(robot)
        try: robot.close()
        except Exception: pass
        try: vision.close()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
