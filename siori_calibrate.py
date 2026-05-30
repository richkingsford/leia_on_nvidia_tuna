#!/usr/bin/env python3
"""
siori_calibrate.py — Measure Leia's kinematics in LOGICAL vision coordinates
at the real operating distance (~217mm), at the PWM the controller uses.

Outputs siori_calibration.json with mm-per-ms for every control primitive:
  - drive f/b      → dist_mm per ms   (and incidental x drift)
  - bias l/r       → x_mm per ms AND dist_mm per ms  (the coupled turn)
  - mast u/d       → y_mm per ms

These are the constants the deliberate-move controller needs to compute
"how long do I pulse to close an N-mm gap" in each axis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot

OUT = Path("siori_calibration.json")
DRIVE_PWM = 103
MAST_PWM = 255
SETTLE_S = 0.45          # let motion + vision settle before reading
READS = 3                # median of N reads per measurement


def _read(vision) -> dict | None:
    vals = []
    last = None
    for _ in range(READS):
        r = follow._read_brick_measurement(vision, jump_guard=False)
        if r.get("confident"):
            last = r
            vals.append((float(r["dist_mm"]), float(r["x_mm"]), float(r["y_mm"])))
        time.sleep(0.05)
    if not vals:
        return None
    vals.sort(key=lambda t: t[0])
    mid = vals[len(vals) // 2]
    return {"dist_mm": mid[0], "x_mm": mid[1], "y_mm": mid[2]}


def _drive(robot, cmd, ms):
    robot.send_command_pwm(cmd, DRIVE_PWM, duration_ms=ms)
    time.sleep(ms / 1000.0 + SETTLE_S)
    follow._stop_robot(robot)


def _mast(robot, cmd, ms):
    robot.send_command_pwm(cmd, MAST_PWM, duration_ms=ms)
    time.sleep(ms / 1000.0 + SETTLE_S)
    follow._stop_robot(robot)


def _bias(robot, turn_cmd, drive_mode, strength, ms, reading):
    follow._send_drive_bias(
        robot, turn_cmd=turn_cmd, drive_mode=drive_mode, strength=strength,
        duration_ms=ms, reading=reading, context="siori_calib",
    )
    time.sleep(ms / 1000.0 + SETTLE_S)
    follow._stop_robot(robot)


def _measure_pair(vision, robot, do_move, undo_move, ms, label):
    """Do a move, measure delta, then undo to recenter. Returns deltas dict or None."""
    before = _read(vision)
    if before is None:
        print(f"[CAL] {label}: brick lost before move", flush=True)
        return None
    do_move(before)
    after = _read(vision)
    if after is None:
        print(f"[CAL] {label}: brick lost after move", flush=True)
        return None
    d = {
        "d_dist": after["dist_mm"] - before["dist_mm"],
        "d_x": after["x_mm"] - before["x_mm"],
        "d_y": after["y_mm"] - before["y_mm"],
        "ms": ms,
    }
    print(f"[CAL] {label} {ms}ms: Δdist={d['d_dist']:+.1f} Δx={d['d_x']:+.1f} "
          f"Δy={d['d_y']:+.1f}  (rate dist={d['d_dist']/ms:+.4f} x={d['d_x']/ms:+.4f} "
          f"y={d['d_y']/ms:+.4f} mm/ms)", flush=True)
    # recenter
    if undo_move is not None:
        after2 = _read(vision) or after
        undo_move(after2)
    return d


def main() -> int:
    follow._set_game_profile("empty")
    vision = BrickDetector(debug=True)
    robot = Robot()
    samples: dict[str, list[dict]] = {}
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)

        # Ensure brick visible
        r = _read(vision)
        if r is None:
            print("[CAL] brick not visible — raising mast 1500ms", flush=True)
            _mast(robot, "u", 1500)
            r = _read(vision)
        if r is None:
            print("[CAL] FAILED: brick not visible. Position Leia and retry.", flush=True)
            return 2
        print(f"[CAL] start: dist={r['dist_mm']:.0f} x={r['x_mm']:+.0f} y={r['y_mm']:+.0f}", flush=True)

        def add(key, d):
            if d is not None:
                samples.setdefault(key, []).append(d)

        REPS = 2
        for _ in range(REPS):
            # ── DRIVE (dist axis) ──
            add("drive_f", _measure_pair(
                vision, robot,
                lambda b: _drive(robot, "f", 220),
                lambda b: _drive(robot, "b", 220),
                220, "DRIVE_F"))
            add("drive_b", _measure_pair(
                vision, robot,
                lambda b: _drive(robot, "b", 220),
                lambda b: _drive(robot, "f", 220),
                220, "DRIVE_B"))

            # ── MAST (y axis) ──
            add("mast_d", _measure_pair(
                vision, robot,
                lambda b: _mast(robot, "d", 400),
                lambda b: _mast(robot, "u", 400),
                400, "MAST_D"))
            add("mast_u", _measure_pair(
                vision, robot,
                lambda b: _mast(robot, "u", 400),
                lambda b: _mast(robot, "d", 400),
                400, "MAST_U"))

            # ── BIAS STRONG (coupled x+dist) ──
            add("bias_r_strong", _measure_pair(
                vision, robot,
                lambda b: _bias(robot, "r", "forward", "strong", 220, b),
                lambda b: _bias(robot, "l", "forward", "strong", 220, b),
                220, "BIAS_R_STRONG"))
            add("bias_l_strong", _measure_pair(
                vision, robot,
                lambda b: _bias(robot, "l", "forward", "strong", 220, b),
                lambda b: _bias(robot, "r", "forward", "strong", 220, b),
                220, "BIAS_L_STRONG"))

            # ── BIAS GENTLE (coupled x+dist, wider) ──
            add("bias_r_gentle", _measure_pair(
                vision, robot,
                lambda b: _bias(robot, "r", "forward", "gentle", 220, b),
                lambda b: _bias(robot, "l", "forward", "gentle", 220, b),
                220, "BIAS_R_GENTLE"))
            add("bias_l_gentle", _measure_pair(
                vision, robot,
                lambda b: _bias(robot, "l", "forward", "gentle", 220, b),
                lambda b: _bias(robot, "r", "forward", "gentle", 220, b),
                220, "BIAS_L_GENTLE"))

        # ── Aggregate: average rates ──
        result = {"operating_dist_mm": r["dist_mm"], "drive_pwm": DRIVE_PWM, "mast_pwm": MAST_PWM, "rates": {}}
        for key, lst in samples.items():
            n = len(lst)
            avg = {
                "d_dist_per_ms": sum(s["d_dist"] / s["ms"] for s in lst) / n,
                "d_x_per_ms": sum(s["d_x"] / s["ms"] for s in lst) / n,
                "d_y_per_ms": sum(s["d_y"] / s["ms"] for s in lst) / n,
                "n": n,
            }
            result["rates"][key] = avg
        OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
        print("\n[CAL] ===== CALIBRATION SUMMARY (mm per ms) =====", flush=True)
        for key, a in result["rates"].items():
            print(f"  {key:16s} dist={a['d_dist_per_ms']:+.4f}  x={a['d_x_per_ms']:+.4f}  "
                  f"y={a['d_y_per_ms']:+.4f}  (n={a['n']})", flush=True)
        print(f"[CAL] saved → {OUT}", flush=True)
        return 0
    finally:
        follow._stop_robot(robot)
        try: robot.close()
        except Exception: pass
        try: vision.close()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
