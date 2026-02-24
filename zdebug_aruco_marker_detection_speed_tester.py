#!/usr/bin/env python3
"""
Manual ArUco detection reliability tester for Leia's camera.

Procedure:
1) Show a 4x4 marker on your phone and run N short capture runs.
2) Pause, switch phone to a 6x6 marker, press Enter, run N short capture runs.
3) Repeat for multiple rounds.

Each run prints a JSON object with frame counts and detection rate.
At the end, a summary compares average detection rate for 4x4 vs 6x6.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import cv2


DEFAULT_DICT_4X4 = "DICT_4X4_50"
DEFAULT_DICT_6X6 = "DICT_6X6_50"


@dataclass(frozen=True)
class DictionaryConfig:
    name: str
    cv2_value: int
    family: str  # "4x4", "6x6", or "other"


def _utc_iso(timestamp_s: float) -> str:
    return datetime.fromtimestamp(float(timestamp_s), tz=timezone.utc).isoformat()


def _detect_family(dict_name: str) -> str:
    upper = str(dict_name or "").upper()
    if "4X4" in upper:
        return "4x4"
    if "6X6" in upper:
        return "6x6"
    return "other"


def _candidate_camera_indices(preferred_index: Optional[int]) -> list[int]:
    candidates: list[int] = []
    if preferred_index is not None:
        candidates.append(int(preferred_index))

    env_index = os.getenv("LEIA_CAMERA_INDEX")
    if env_index:
        try:
            candidates.append(int(env_index))
        except ValueError:
            pass

    candidates.extend([0, 1, 2, 3])
    deduped: list[int] = []
    for idx in candidates:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _open_camera(index: int, width: int, height: int):
    backend_pref = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
    for backend in (backend_pref, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            return cap
        if cap is not None:
            cap.release()
        if backend == cv2.CAP_ANY:
            break
    return None


def open_camera_for_leia(width: int, height: int, preferred_index: Optional[int]):
    tried: list[int] = []
    for idx in _candidate_camera_indices(preferred_index):
        tried.append(int(idx))
        cap = _open_camera(index=int(idx), width=int(width), height=int(height))
        if cap is not None and cap.isOpened():
            return cap, int(idx), tried
    return None, None, tried


def parse_dictionary(name: str) -> DictionaryConfig:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is unavailable in this environment.")
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Dictionary name cannot be empty.")
    if not hasattr(cv2.aruco, raw):
        raise ValueError(
            f"Unknown ArUco dictionary '{raw}'. Example values: "
            "DICT_4X4_50, DICT_4X4_100, DICT_6X6_50, DICT_6X6_100."
        )
    return DictionaryConfig(
        name=raw,
        cv2_value=int(getattr(cv2.aruco, raw)),
        family=_detect_family(raw),
    )


def build_detector(dict_cfg: DictionaryConfig) -> Callable[[object], bool]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(int(dict_cfg.cv2_value))
    params = cv2.aruco.DetectorParameters()

    # Match Leia's practical tuning used in helper_vision_aruco.
    params.minMarkerPerimeterRate = 0.005
    params.polygonalApproxAccuracyRate = 0.1
    params.minCornerDistanceRate = 0.05
    params.minSideLengthCanonicalImg = 4

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        def detect_marker(gray_frame) -> bool:
            _, ids, _ = detector.detectMarkers(gray_frame)
            return bool(ids is not None and len(ids) > 0)

        return detect_marker

    def detect_marker(gray_frame) -> bool:
        _, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=params)
        return bool(ids is not None and len(ids) > 0)

    return detect_marker


def run_single_detection_window(
    *,
    cap,
    detector: Callable[[object], bool],
    run_seconds: float,
    dictionary: DictionaryConfig,
    round_idx: int,
    run_in_block: int,
    run_global: int,
    camera_index: int,
) -> dict:
    start_ts = time.time()
    deadline = time.monotonic() + float(run_seconds)

    frames_processed = 0
    frames_detected = 0

    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if (not bool(ok)) or frame is None:
            continue
        frames_processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bool(detector(gray)):
            frames_detected += 1

    end_ts = time.time()
    detection_rate = (
        float(frames_detected) / float(frames_processed)
        if int(frames_processed) > 0
        else 0.0
    )

    return {
        "type": "aruco_detection_speed_run",
        "round": int(round_idx),
        "run_in_dictionary_block": int(run_in_block),
        "global_run_index": int(run_global),
        "dictionary": str(dictionary.name),
        "dictionary_family": str(dictionary.family),
        "camera_index": int(camera_index),
        "run_seconds": float(run_seconds),
        "start_timestamp_s": float(start_ts),
        "end_timestamp_s": float(end_ts),
        "start_utc": _utc_iso(start_ts),
        "end_utc": _utc_iso(end_ts),
        "frames_processed": int(frames_processed),
        "frames_detected": int(frames_detected),
        "detection_rate": float(detection_rate),
    }


def average_detection_rate(results: list[dict], dictionary_name: str) -> float:
    rates = [
        float(row.get("detection_rate", 0.0))
        for row in results
        if str(row.get("dictionary")) == str(dictionary_name)
    ]
    if not rates:
        return 0.0
    return float(sum(rates) / float(len(rates)))


def build_summary(*, results: list[dict], dict_a: DictionaryConfig, dict_b: DictionaryConfig) -> dict:
    avg_a = average_detection_rate(results, dict_a.name)
    avg_b = average_detection_rate(results, dict_b.name)

    avg_4x4 = None
    avg_6x6 = None
    if dict_a.family == "4x4":
        avg_4x4 = avg_a
    elif dict_b.family == "4x4":
        avg_4x4 = avg_b
    if dict_a.family == "6x6":
        avg_6x6 = avg_a
    elif dict_b.family == "6x6":
        avg_6x6 = avg_b

    if avg_a > avg_b:
        easier = dict_a.name
    elif avg_b > avg_a:
        easier = dict_b.name
    else:
        easier = "tie"

    return {
        "type": "aruco_detection_speed_summary",
        "dictionary_a": dict_a.name,
        "dictionary_b": dict_b.name,
        "average_detection_rate_dictionary_a": float(avg_a),
        "average_detection_rate_dictionary_b": float(avg_b),
        "average_detection_rate_4x4": (float(avg_4x4) if avg_4x4 is not None else None),
        "average_detection_rate_6x6": (float(avg_6x6) if avg_6x6 is not None else None),
        "easier_marker_dictionary": str(easier),
        "total_runs": int(len(results)),
    }


def _prompt_enter(message: str) -> None:
    input(f"{message}\nPress Enter to continue...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ArUco detection reliability for two dictionaries using Leia's camera."
    )
    parser.add_argument("--camera-index", type=int, default=None, help="Preferred camera index (default: auto).")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480).")
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=2.5,
        help="Duration of each detection run in seconds; use 2.0-3.0 (default: 2.5).",
    )
    parser.add_argument(
        "--runs-per-dict-per-round",
        type=int,
        default=5,
        help="Runs per dictionary before switching markers (default: 5).",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Number of full rounds (default: 2).")
    parser.add_argument(
        "--dict-4x4",
        type=str,
        default=DEFAULT_DICT_4X4,
        help=f"4x4 dictionary name (default: {DEFAULT_DICT_4X4}).",
    )
    parser.add_argument(
        "--dict-6x6",
        type=str,
        default=DEFAULT_DICT_6X6,
        help=f"6x6 dictionary name (default: {DEFAULT_DICT_6X6}).",
    )
    parser.add_argument("--warmup-frames", type=int, default=15, help="Warm-up frames before first run (default: 15).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_seconds = float(args.run_seconds)
    if run_seconds < 2.0 or run_seconds > 3.0:
        raise ValueError("--run-seconds must be between 2.0 and 3.0 seconds for this controlled test.")

    rounds = max(1, int(args.rounds))
    runs_per_block = max(1, int(args.runs_per_dict_per_round))
    dict_4x4 = parse_dictionary(args.dict_4x4)
    dict_6x6 = parse_dictionary(args.dict_6x6)
    detector_4x4 = build_detector(dict_4x4)
    detector_6x6 = build_detector(dict_6x6)

    cap, active_camera_index, tried_indices = open_camera_for_leia(
        width=int(args.width),
        height=int(args.height),
        preferred_index=(int(args.camera_index) if args.camera_index is not None else None),
    )
    if cap is None or active_camera_index is None:
        raise RuntimeError(f"Unable to open camera. Tried indices: {tried_indices}")

    print(
        f"[ARUCO-TEST] Camera opened on index {active_camera_index} "
        f"({int(args.width)}x{int(args.height)}).",
        flush=True,
    )
    print(
        f"[ARUCO-TEST] Plan: {rounds} rounds, {runs_per_block} runs per dictionary per round, "
        f"{run_seconds:.2f}s/run.",
        flush=True,
    )
    print(
        f"[ARUCO-TEST] Dictionaries: 4x4={dict_4x4.name}, 6x6={dict_6x6.name}",
        flush=True,
    )

    warmup_frames = max(0, int(args.warmup_frames))
    for _ in range(warmup_frames):
        cap.read()

    results: list[dict] = []
    global_run_idx = 1

    try:
        for round_idx in range(1, rounds + 1):
            round_label = f"{round_idx}/{rounds}"
            _prompt_enter(
                f"\n[ARUCO-TEST] Round {round_label}: show {dict_4x4.name} marker on your phone."
            )

            for run_in_block in range(1, runs_per_block + 1):
                run_result = run_single_detection_window(
                    cap=cap,
                    detector=detector_4x4,
                    run_seconds=run_seconds,
                    dictionary=dict_4x4,
                    round_idx=round_idx,
                    run_in_block=run_in_block,
                    run_global=global_run_idx,
                    camera_index=active_camera_index,
                )
                print(json.dumps(run_result, sort_keys=True), flush=True)
                results.append(run_result)
                global_run_idx += 1

            _prompt_enter(
                f"\n[ARUCO-TEST] Round {round_label}: switch phone display to {dict_6x6.name} marker."
            )

            for run_in_block in range(1, runs_per_block + 1):
                run_result = run_single_detection_window(
                    cap=cap,
                    detector=detector_6x6,
                    run_seconds=run_seconds,
                    dictionary=dict_6x6,
                    round_idx=round_idx,
                    run_in_block=run_in_block,
                    run_global=global_run_idx,
                    camera_index=active_camera_index,
                )
                print(json.dumps(run_result, sort_keys=True), flush=True)
                results.append(run_result)
                global_run_idx += 1

        summary = build_summary(results=results, dict_a=dict_4x4, dict_b=dict_6x6)
        print(json.dumps(summary, sort_keys=True), flush=True)

    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
