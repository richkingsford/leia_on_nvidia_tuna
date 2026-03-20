#!/usr/bin/env python3
"""
Continuous 6x6 ArUco marker logger.

Logs one JSON line per frame (unless --detections-only is set) using Leia's
runtime 6x6 detection path from helper_vision_aruco.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import cv2

from helper_vision_aruco import ArucoBrickVision


def _open_camera_for_leia(
    vision: ArucoBrickVision,
    *,
    width: int,
    height: int,
    preferred_index: Optional[int],
):
    tried = []
    for idx in vision._candidate_camera_indices(preferred_index):
        tried.append(int(idx))
        cap = vision._open_camera(int(idx), int(width), int(height))
        if cap is not None and cap.isOpened():
            return cap, int(idx), tried
    return None, None, tried


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log ArUco 6x6 marker detections as fast as possible (JSONL)."
    )
    parser.add_argument("--camera-index", type=int, default=None, help="Preferred camera index (default: auto).")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480).")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Optional max runtime in seconds; 0 means run until Ctrl+C (default: 0).",
    )
    parser.add_argument(
        "--summary-interval",
        type=float,
        default=1.0,
        help="Summary cadence in seconds to stderr; 0 disables summaries (default: 1.0).",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Optional marker ID to track explicitly (for example: 11).",
    )
    parser.add_argument(
        "--detections-only",
        action="store_true",
        help="Only emit frame JSON when one or more IDs are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is unavailable in this environment.")

    vision = ArucoBrickVision(debug=False)
    cap, active_camera_index, tried_indices = _open_camera_for_leia(
        vision,
        width=int(args.width),
        height=int(args.height),
        preferred_index=(int(args.camera_index) if args.camera_index is not None else None),
    )
    if cap is None or active_camera_index is None:
        raise RuntimeError(f"Unable to open camera. Tried indices: {tried_indices}")
    # Reuse the same capture handle inside the helper so process() doesn't try to re-open.
    vision.cap = cap
    vision.camera_index = int(active_camera_index)
    vision.init_camera(
        width=int(args.width),
        height=int(args.height),
        camera_index=int(active_camera_index),
    )

    print(
        f"[ARUCO-6X6] Camera opened on index {active_camera_index} "
        f"({int(args.width)}x{int(args.height)}).",
        file=sys.stderr,
        flush=True,
    )
    print("[ARUCO-6X6] Logging started. Press Ctrl+C to stop.", file=sys.stderr, flush=True)

    start_mono = time.monotonic()
    last_summary = start_mono
    frames_processed = 0
    frames_detected = 0
    frames_target_detected = 0

    try:
        while True:
            if float(args.max_seconds) > 0:
                if (time.monotonic() - start_mono) >= float(args.max_seconds):
                    break

            ok, frame = cap.read()
            if (not bool(ok)) or frame is None:
                continue

            frames_processed += 1
            pose = vision.process(frame)
            id_values = [int(pose.marker_id)] if bool(pose.found) and int(pose.marker_id) >= 0 else []

            found = bool(len(id_values) > 0)
            if found:
                frames_detected += 1

            target_seen = bool(
                args.target_id is not None and int(args.target_id) in set(int(v) for v in id_values)
            )
            if target_seen:
                frames_target_detected += 1

            if (not bool(args.detections_only)) or found:
                event = {
                    "type": "aruco_6x6_frame",
                    "timestamp_s": float(time.time()),
                    "frame_index": int(frames_processed),
                    "camera_index": int(active_camera_index),
                    "found": bool(found),
                    "ids": [int(v) for v in id_values],
                    "detect_pass": str(getattr(vision, "last_detect_pass", "unknown")),
                    "confidence": (float(pose.confidence) if bool(pose.found) else 0.0),
                    "x_mm": (float(pose.position[0]) if bool(pose.found) else 0.0),
                    "dist_mm": (float(pose.position[2]) if bool(pose.found) else 0.0),
                }
                if args.target_id is not None:
                    event["target_id"] = int(args.target_id)
                    event["target_id_seen"] = bool(target_seen)
                print(json.dumps(event), flush=True)

            if float(args.summary_interval) > 0:
                now_mono = time.monotonic()
                if (now_mono - last_summary) >= float(args.summary_interval):
                    elapsed = max(1e-6, now_mono - start_mono)
                    summary = {
                        "type": "aruco_6x6_summary",
                        "elapsed_s": float(elapsed),
                        "frames_processed": int(frames_processed),
                        "fps": float(frames_processed / elapsed),
                        "frames_detected": int(frames_detected),
                        "detection_rate": float(
                            (frames_detected / frames_processed) if frames_processed > 0 else 0.0
                        ),
                        "frames_target_detected": int(frames_target_detected),
                        "target_detection_rate": float(
                            (frames_target_detected / frames_processed) if frames_processed > 0 else 0.0
                        ),
                        "target_id": (int(args.target_id) if args.target_id is not None else None),
                    }
                    print(f"[ARUCO-6X6] {json.dumps(summary)}", file=sys.stderr, flush=True)
                    last_summary = now_mono
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    elapsed = max(1e-6, time.monotonic() - start_mono)
    final_summary = {
        "type": "aruco_6x6_final_summary",
        "elapsed_s": float(elapsed),
        "frames_processed": int(frames_processed),
        "fps": float(frames_processed / elapsed),
        "frames_detected": int(frames_detected),
        "detection_rate": float((frames_detected / frames_processed) if frames_processed > 0 else 0.0),
        "frames_target_detected": int(frames_target_detected),
        "target_detection_rate": float(
            (frames_target_detected / frames_processed) if frames_processed > 0 else 0.0
        ),
        "target_id": (int(args.target_id) if args.target_id is not None else None),
        "camera_index": int(active_camera_index),
    }
    print(f"[ARUCO-6X6] {json.dumps(final_summary)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
