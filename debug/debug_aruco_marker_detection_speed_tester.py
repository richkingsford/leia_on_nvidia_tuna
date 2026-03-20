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
import numpy as np


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


def build_detector(dict_cfg: DictionaryConfig) -> Callable[[object], tuple[bool, list[int], str]]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(int(dict_cfg.cv2_value))
    clahe_loose_max_rejected = int(os.getenv("ARUCO_CLAHE_LOOSE_MAX_REJECTED", "600") or 600)

    def _build_aruco_params(loose: bool):
        params = cv2.aruco.DetectorParameters()

        # Keep this aligned with helper_vision_aruco so runtime and tester semantics match.
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        if hasattr(params, "cornerRefinementMinAccuracy"):
            params.cornerRefinementMinAccuracy = 0.01

        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 4
        params.perspectiveRemovePixelPerCell = 6
        params.minCornerDistanceRate = 0.05
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 1

        if loose:
            params.minMarkerPerimeterRate = 0.01
            params.polygonalApproxAccuracyRate = 0.06
            params.minSideLengthCanonicalImg = 8
        else:
            params.minMarkerPerimeterRate = 0.008
            params.polygonalApproxAccuracyRate = 0.05
            params.minSideLengthCanonicalImg = 16
        return params

    def _quad_bbox(quad):
        try:
            pts = np.asarray(quad, dtype=np.float32)
        except Exception:
            return None
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
            return None
        x, y, w, h = cv2.boundingRect(np.round(pts[:4]).astype(np.int32))
        return (float(x), float(y), float(max(1, w)), float(max(1, h)))

    def _bbox_overlap_frac(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = float(iw * ih)
        area_a = max(1.0, float(aw * ah))
        return inter / area_a

    def _boxes_same_marker(box_a, box_b):
        if box_a is None or box_b is None:
            return False
        ov = max(_bbox_overlap_frac(box_a, box_b), _bbox_overlap_frac(box_b, box_a))
        if ov >= 0.35:
            return True
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        acx = ax + (aw * 0.5)
        acy = ay + (ah * 0.5)
        bcx = bx + (bw * 0.5)
        bcy = by + (bh * 0.5)
        center_dist = float(np.hypot(acx - bcx, acy - bcy))
        size_ref = max(8.0, min(max(aw, ah), max(bw, bh)))
        return center_dist <= (0.35 * float(size_ref))

    strict_params = _build_aruco_params(loose=False)
    loose_params = _build_aruco_params(loose=True)

    if hasattr(cv2.aruco, "ArucoDetector"):
        strict_detector = cv2.aruco.ArucoDetector(aruco_dict, strict_params)
        loose_detector = cv2.aruco.ArucoDetector(aruco_dict, loose_params)
    else:
        strict_detector = None
        loose_detector = None

    def _run_detect(detector_obj, gray_img, params_obj):
        if detector_obj is not None:
            return detector_obj.detectMarkers(gray_img)
        return cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=params_obj)

    def detect_marker(gray_frame):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray_frame)

        pass_results = []
        for label, detector_obj, params_obj, img in (
            ("raw", strict_detector, strict_params, gray_frame),
            ("clahe", strict_detector, strict_params, clahe_gray),
            ("raw_loose", loose_detector, loose_params, gray_frame),
        ):
            corners_i, ids_i, rejected_i = _run_detect(detector_obj, img, params_obj)
            pass_results.append((label, corners_i, ids_i, rejected_i))

        merged_corner_boxes = []
        merged_ids = []
        primary_label = None
        merged_from_later = False

        for label, corners_i, ids_i, _rejected_i in pass_results:
            if ids_i is None or corners_i is None or len(ids_i) <= 0:
                continue
            if primary_label is None:
                primary_label = str(label)
            for idx in range(min(len(corners_i), len(ids_i))):
                quad = corners_i[idx]
                bbox = _quad_bbox(quad)
                if bbox is None:
                    continue
                duplicate = any(_boxes_same_marker(bbox, existing) for existing in merged_corner_boxes)
                if duplicate:
                    continue
                try:
                    marker_id = int(ids_i[idx][0])
                except Exception:
                    try:
                        marker_id = int(ids_i[idx])
                    except Exception:
                        marker_id = -1
                merged_corner_boxes.append(bbox)
                merged_ids.append(marker_id)
                if primary_label is not None and str(label) != str(primary_label):
                    merged_from_later = True

        if merged_ids:
            pass_name = f"{primary_label}+merge" if merged_from_later else str(primary_label or "raw")
            return True, sorted(set(int(v) for v in merged_ids if int(v) >= 0)), pass_name

        raw_loose_rejected = 0
        for label, _corners, _ids, rejected_i in pass_results:
            if str(label) == "raw_loose":
                raw_loose_rejected = int(len(rejected_i) if rejected_i is not None else 0)
                break
        if clahe_loose_max_rejected <= 0 or raw_loose_rejected < clahe_loose_max_rejected:
            corners_i, ids_i, rejected_i = _run_detect(loose_detector, clahe_gray, loose_params)
            pass_results.append(("clahe_loose", corners_i, ids_i, rejected_i))
            if ids_i is not None and corners_i is not None and len(ids_i) > 0:
                ids_list = sorted(set(int(v) for v in ids_i.flatten().tolist()))
                return True, ids_list, "clahe_loose"

        # Keep pass semantics consistent with runtime helper fallback.
        best = max(
            pass_results,
            key=lambda item: len(item[3]) if item[3] is not None else 0,
        )
        return False, [], str(best[0])

    return detect_marker


def run_single_detection_window(
    *,
    cap,
    detector: Callable[[object], tuple[bool, list[int], str]],
    run_seconds: float,
    dictionary: DictionaryConfig,
    round_idx: int,
    run_in_block: int,
    run_global: int,
    camera_index: int,
    emit_frame_events: bool = False,
    target_id: Optional[int] = None,
) -> dict:
    start_ts = time.time()
    deadline = time.monotonic() + float(run_seconds)

    frames_processed = 0
    frames_detected = 0
    frames_target_detected = 0
    pass_counts: dict[str, int] = {}
    ids_seen_counts: dict[int, int] = {}

    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if (not bool(ok)) or frame is None:
            continue
        frames_processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected, ids_seen, pass_name = detector(gray)
        pass_counts[str(pass_name)] = int(pass_counts.get(str(pass_name), 0) + 1)
        if bool(detected):
            frames_detected += 1
            for marker_id in ids_seen:
                ids_seen_counts[int(marker_id)] = int(ids_seen_counts.get(int(marker_id), 0) + 1)
            target_seen = bool(target_id is not None and int(target_id) in set(int(v) for v in ids_seen))
            if target_seen:
                frames_target_detected += 1
            if emit_frame_events:
                event = {
                    "type": "aruco_detection_frame",
                    "timestamp_s": float(time.time()),
                    "round": int(round_idx),
                    "run_in_dictionary_block": int(run_in_block),
                    "global_run_index": int(run_global),
                    "dictionary": str(dictionary.name),
                    "frame_index_in_run": int(frames_processed),
                    "ids": [int(v) for v in sorted(set(ids_seen))],
                    "detect_pass": str(pass_name),
                }
                if target_id is not None:
                    event["target_id"] = int(target_id)
                    event["target_id_seen"] = bool(target_seen)
                print(json.dumps(event, sort_keys=True), flush=True)

    end_ts = time.time()
    detection_rate = (
        float(frames_detected) / float(frames_processed)
        if int(frames_processed) > 0
        else 0.0
    )
    target_detection_rate = (
        float(frames_target_detected) / float(frames_processed)
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
        "frames_target_detected": int(frames_target_detected),
        "target_detection_rate": float(target_detection_rate),
        "target_id": (int(target_id) if target_id is not None else None),
        "ids_seen_counts": {str(int(k)): int(v) for k, v in sorted(ids_seen_counts.items(), key=lambda kv: kv[0])},
        "detect_pass_counts": {str(k): int(v) for k, v in sorted(pass_counts.items(), key=lambda kv: kv[0])},
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
    parser.add_argument(
        "--single-dict",
        type=str,
        default=None,
        help="Run only one dictionary (non-interactive), e.g. DICT_6X6_50.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip Enter prompts in two-dictionary mode.",
    )
    parser.add_argument(
        "--emit-frame-events",
        action="store_true",
        help="Emit JSON log lines for every frame where one or more markers are detected.",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Optional marker ID to track explicitly (for example 11).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_seconds = float(args.run_seconds)
    if run_seconds < 2.0 or run_seconds > 3.0:
        raise ValueError("--run-seconds must be between 2.0 and 3.0 seconds for this controlled test.")

    rounds = max(1, int(args.rounds))
    runs_per_block = max(1, int(args.runs_per_dict_per_round))
    single_dict = str(args.single_dict).strip() if args.single_dict else None
    dict_4x4 = parse_dictionary(args.dict_4x4)
    dict_6x6 = parse_dictionary(args.dict_6x6)
    detector_4x4 = build_detector(dict_4x4)
    detector_6x6 = build_detector(dict_6x6)
    detector_single = build_detector(parse_dictionary(single_dict)) if single_dict else None

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
    if single_dict:
        print(f"[ARUCO-TEST] Single-dictionary mode: {single_dict}", flush=True)
    if args.target_id is not None:
        print(f"[ARUCO-TEST] Target marker ID: {int(args.target_id)}", flush=True)
    if bool(args.emit_frame_events):
        print("[ARUCO-TEST] Frame-event logging enabled.", flush=True)

    warmup_frames = max(0, int(args.warmup_frames))
    for _ in range(warmup_frames):
        cap.read()

    results: list[dict] = []
    global_run_idx = 1

    try:
        if single_dict:
            dict_single = parse_dictionary(single_dict)
            assert detector_single is not None
            for round_idx in range(1, rounds + 1):
                for run_in_block in range(1, runs_per_block + 1):
                    run_result = run_single_detection_window(
                        cap=cap,
                        detector=detector_single,
                        run_seconds=run_seconds,
                        dictionary=dict_single,
                        round_idx=round_idx,
                        run_in_block=run_in_block,
                        run_global=global_run_idx,
                        camera_index=active_camera_index,
                        emit_frame_events=bool(args.emit_frame_events),
                        target_id=(int(args.target_id) if args.target_id is not None else None),
                    )
                    print(json.dumps(run_result, sort_keys=True), flush=True)
                    results.append(run_result)
                    global_run_idx += 1

            avg = average_detection_rate(results, dict_single.name)
            target_avg = 0.0
            if results:
                target_avg = float(
                    sum(float(row.get("target_detection_rate", 0.0)) for row in results) / float(len(results))
                )
            summary = {
                "type": "aruco_detection_speed_summary_single",
                "dictionary": dict_single.name,
                "average_detection_rate": float(avg),
                "average_target_detection_rate": float(target_avg),
                "target_id": (int(args.target_id) if args.target_id is not None else None),
                "total_runs": int(len(results)),
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
        else:
            for round_idx in range(1, rounds + 1):
                round_label = f"{round_idx}/{rounds}"
                if not bool(args.no_prompt):
                    _prompt_enter(
                        f"\n[ARUCO-TEST] Round {round_label}: show {dict_4x4.name} marker on your phone."
                    )
                else:
                    print(f"[ARUCO-TEST] Round {round_label}: starting {dict_4x4.name} block.", flush=True)

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
                        emit_frame_events=bool(args.emit_frame_events),
                        target_id=(int(args.target_id) if args.target_id is not None else None),
                    )
                    print(json.dumps(run_result, sort_keys=True), flush=True)
                    results.append(run_result)
                    global_run_idx += 1

                if not bool(args.no_prompt):
                    _prompt_enter(
                        f"\n[ARUCO-TEST] Round {round_label}: switch phone display to {dict_6x6.name} marker."
                    )
                else:
                    print(f"[ARUCO-TEST] Round {round_label}: starting {dict_6x6.name} block.", flush=True)

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
                        emit_frame_events=bool(args.emit_frame_events),
                        target_id=(int(args.target_id) if args.target_id is not None else None),
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
