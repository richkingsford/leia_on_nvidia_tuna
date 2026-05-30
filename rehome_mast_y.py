#!/usr/bin/env python3
"""One-time mast re-home: lower the mast until the brick's y sits near the
empty step-1 target (-42.7mm). Mast-down is slow (~0.2mm/100ms) and far slower
than mast-up, so repeated full-power 'u' vision recoveries ratchet the mast up
and it can't recover within a trial. This re-homes it once before a trial batch.
Reads y the same way the game does: vision.read()[5]."""
import statistics
import time

from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot

TARGET_Y_MM = -42.7
STOP_AT_Y_MM = -40.0          # stop once y has come down to the target band
HARD_FLOOR_Y_MM = -46.0       # never drive below this
MAX_ITERS = 30
DOWN_CHUNK_MS = 1500


def read_y(vision, samples=5):
    ys = []
    for _ in range(samples):
        try:
            r = vision.read()
        except Exception:
            r = None
        if isinstance(r, tuple) and len(r) >= 6 and bool(r[0]):
            try:
                ys.append(float(r[5]))
            except (TypeError, ValueError):
                pass
        time.sleep(0.06)
    if not ys:
        return None
    return float(statistics.median(ys))


def main():
    vision = BrickDetector(debug=False)
    robot = Robot(exit_on_failure=False)

    # Warm up / wait for the camera + a confident y read.
    y = None
    t0 = time.time()
    while time.time() - t0 < 45.0:
        y = read_y(vision, samples=3)
        if y is not None:
            break
        time.sleep(0.5)
    print(f"[REHOME] initial y={y}", flush=True)
    if y is None:
        print("[REHOME] no visible brick; aborting (no motion).", flush=True)
        robot.stop(); robot.close()
        return

    # Bidirectional re-home: up if too low (fast), down if too high (slow).
    BAND_LOW = -45.0   # below this y is too low -> raise (mast 'u')
    BAND_HIGH = -40.0  # above this y is too high -> lower (mast 'd')
    UP_CHUNK_MS = 400  # mast-up is fast (~2.0mm/100ms); small chunks
    for i in range(MAX_ITERS):
        y = read_y(vision, samples=5)
        print(f"[REHOME] iter {i}: y={y}", flush=True)
        if y is None:
            print("[REHOME] lost brick; stopping.", flush=True)
            break
        if BAND_LOW <= y <= BAND_HIGH:
            print(f"[REHOME] in band (y={y:.1f}).", flush=True)
            break
        if y < BAND_LOW:
            robot.send_command('u', 1.0, duration_ms=UP_CHUNK_MS)   # too low -> raise
        else:
            robot.send_command('d', 1.0, duration_ms=DOWN_CHUNK_MS)  # too high -> lower
        time.sleep(0.25)

    robot.stop()
    final_y = read_y(vision, samples=7)
    print(f"[REHOME] final y={final_y} (target {TARGET_Y_MM})", flush=True)
    robot.close()
    try:
        vision.release()
    except Exception:
        try:
            vision.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
