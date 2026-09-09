#!/usr/bin/env python3
"""One-second live smoke test for the new direct-serial visual servo path."""

from __future__ import annotations

import argparse
import csv
import time

import serial

from direct_visual_servo import BrickMeasurement, ServoConfig, VisualServoController
from helper_brick_detector_yolo import BrickDetector


CONTROL_HZ = 20.0
TEST_DURATION_S = 4.0


def wire_packet(left: float, right: float) -> bytes:
    left_i = max(-100, min(100, round(left)))
    right_i = max(-100, min(100, round(right)))
    return f"<{left_i},{right_i},0>".encode("ascii")


def send(port: serial.Serial, left: float, right: float) -> None:
    port.write(wire_packet(left, right))
    port.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-second direct visual-servo test.")
    parser.add_argument("--execute", action="store_true", help="Enable physical tread commands.")
    parser.add_argument("--port", default="/dev/leia-uno")
    parser.add_argument("--log", default="direct_visual_servo_1s.csv")
    args = parser.parse_args()

    controller = VisualServoController(
        ServoConfig(
            left_min_moving_pct=30.0,
            right_min_moving_pct=30.0,
            max_tread_pct=42.0,
            sharp_inner_pct=30.0,
            semi_gentle_inner_pct=36.0,
            gentle_inner_pct=39.0,
            super_gentle_inner_pct=42.0,
        )
    )
    port = None
    detector = None
    started = None
    rows = []
    try:
        detector = BrickDetector(debug=False, speed_optimize=True)
        if args.execute:
            port = serial.Serial(args.port, 115200, timeout=1)
            time.sleep(2.0)
        started = time.monotonic()
        print("Direct visual-servo test: 4.0 s, mast permanently zero.", flush=True)
        while time.monotonic() - started < TEST_DURATION_S:
            cycle_start = time.monotonic()
            camera_start = time.monotonic()
            found, _angle, dist, x, confidence, *_ = detector.read()
            camera_end = time.monotonic()
            measurement = BrickMeasurement(x, dist, confidence / 100.0, camera_end) if found else None
            control_start = time.monotonic()
            now = control_start
            command = controller.update(measurement, now)
            control_end = time.monotonic()
            serial_start = time.monotonic()
            if port is not None:
                send(port, command.left_pct, command.right_pct)
            serial_end = time.monotonic()
            print(f"{command.state}: x={x:.1f} dist={dist:.1f} conf={confidence:.1f} -> <{command.left_pct:.0f},{command.right_pct:.0f},0>", flush=True)
            rows.append({
                "timestamp": now,
                "cycle_start": cycle_start,
                "camera_start": camera_start,
                "camera_end": camera_end,
                "control_start": control_start,
                "control_end": control_end,
                "serial_start": serial_start,
                "serial_end": serial_end,
                "loop_period_s": now - getattr(main, "_last_tick", now),
                "camera_duration_s": camera_end - camera_start,
                "control_duration_s": control_end - control_start,
                "serial_duration_s": serial_end - serial_start,
                "measurement_age_s": now - measurement.timestamp if measurement else "",
                "raw_x_mm": x,
                "raw_dist_mm": dist,
                "confidence": confidence,
                "state": command.state,
                "left_final": command.left_pct,
                "right_final": command.right_pct,
                "valid": command.valid,
                "rejection_reason": command.rejection_reason,
            })
            main._last_tick = now
            time.sleep(max(0.0, (1.0 / CONTROL_HZ) - (time.monotonic() - cycle_start)))
        return 0
    finally:
        if port is not None:
            send(port, 0, 0)
            port.close()
        with open(args.log, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["timestamp"])
            writer.writeheader()
            writer.writerows(rows)
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
