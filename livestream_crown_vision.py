#!/usr/bin/env python3
"""Standalone livestream for the current crown-brick vision model."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import cv2

from helper_brick_detector_yolo import (
    BRICK_HEIGHT_MM,
    BRICK_WIDTH_MM,
    BrickDetector,
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
    CYAN_HSV_TIGHT_LOWER,
    CYAN_HSV_TIGHT_UPPER,
    CYAN_SHADE_HEXES,
)
from helper_manual_config import load_manual_training_config
from helper_stream_server import format_stream_url
from helper_streaming import start_stream_server
import helper_xyz_coords


# Frames to hold the last good detection before declaring invisible.
# At 15 Hz camera this is ~1 s — enough to bridge YOLO misses without
# masking real disappearances.
HOLD_FRAMES = 15

CROWN_PROFILE_KEY = "tight_color"

CROWN_PROFILE_BASE_TUNING = {
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
    "closeup_full_frame_hsv_enabled": True,
    "depth_source_mode": "pinhole",
    "stereo_config_mode": "standard",
}

TIGHT_COLOR_TUNING = {
    **CROWN_PROFILE_BASE_TUNING,
    "label": "2 Tight Color",
    "confidence": 0.35,
    "hsv_lower": list(CYAN_HSV_TIGHT_LOWER),
    "hsv_upper": list(CYAN_HSV_TIGHT_UPPER),
    "hsv_cyan_coverage_min": 0.18,
    "hsv_min_area_ratio": 0.10,
    "negative_cutout_min_area_px": 42.0,
    "conf_gate_pct": 85.0,
}

CROWN_PROFILE_TUNINGS = {
    "blue_reject": {
        **CROWN_PROFILE_BASE_TUNING,
        "label": "1 Strict Blue Reject",
        "confidence": 0.40,
        "hsv_lower": [88, 95, 45],
        "hsv_upper": [99, 255, 245],
        "hsv_cyan_coverage_min": 0.22,
        "hsv_min_area_ratio": 0.12,
        "negative_cutout_min_area_px": 54.0,
        "negative_cutout_ring_cyan_min": 0.66,
        "conf_gate_pct": 90.0,
        "closeup_full_frame_hsv_enabled": False,
    },
    "tight_color": TIGHT_COLOR_TUNING,
    "tight_far_slots": {
        **TIGHT_COLOR_TUNING,
        "label": "2A Far: Smaller Slots",
        "confidence": 0.32,
        "hsv_cyan_coverage_min": 0.16,
        "hsv_min_area_ratio": 0.07,
        "negative_cutout_min_area_px": 24.0,
        "conf_gate_pct": 82.0,
    },
    "tight_far_conf": {
        **TIGHT_COLOR_TUNING,
        "label": "2B Far: Lower YOLO",
        "confidence": 0.25,
        "hsv_cyan_coverage_min": 0.15,
        "hsv_min_area_ratio": 0.06,
        "negative_cutout_min_area_px": 18.0,
        "conf_gate_pct": 76.0,
    },
    "tight_far_no_erode": {
        **TIGHT_COLOR_TUNING,
        "label": "2C Far: No Erode",
        "confidence": 0.25,
        "hsv_erode_iterations": 0,
        "hsv_cyan_coverage_min": 0.14,
        "hsv_min_area_ratio": 0.05,
        "negative_cutout_min_area_px": 12.0,
        "conf_gate_pct": 76.0,
    },
    "tight_far_dim": {
        **TIGHT_COLOR_TUNING,
        "label": "2D Far: Dim Cyan",
        "confidence": 0.22,
        "hsv_erode_iterations": 0,
        "hsv_lower": [88, 60, 35],
        "hsv_upper": [100, 255, 255],
        "hsv_cyan_coverage_min": 0.12,
        "hsv_min_area_ratio": 0.045,
        "negative_cutout_ring_cyan_min": 0.60,
        "negative_cutout_min_area_px": 10.0,
        "conf_gate_pct": 72.0,
    },
    "balanced_far_guard": {
        **CROWN_PROFILE_BASE_TUNING,
        "label": "2E Far: Wider Hue Guard",
        "confidence": 0.22,
        "hsv_lower": [85, 70, 35],
        "hsv_upper": list(CYAN_HSV_BALANCED_UPPER),
        "hsv_cyan_coverage_min": 0.14,
        "hsv_min_area_ratio": 0.055,
        "negative_cutout_ring_cyan_min": 0.64,
        "negative_cutout_min_area_px": 14.0,
        "conf_gate_pct": 78.0,
        "closeup_full_frame_hsv_enabled": False,
    },
}
CROWN_PROFILE_OPTIONS = [
    (key, str(settings.get("label", key)))
    for key, settings in CROWN_PROFILE_TUNINGS.items()
]
CROWN_PROFILE_LABELS = dict(CROWN_PROFILE_OPTIONS)
CROWN_PROFILE_TUNING = {
    key: value
    for key, value in CROWN_PROFILE_TUNINGS[CROWN_PROFILE_KEY].items()
    if key != "label"
}


def _profile_settings(profile_key: str) -> tuple[str, dict]:
    key = str(profile_key or "").strip().lower()
    if key not in CROWN_PROFILE_TUNINGS:
        key = CROWN_PROFILE_KEY
    settings = {
        setting_key: value
        for setting_key, value in CROWN_PROFILE_TUNINGS[key].items()
        if setting_key != "label"
    }
    return key, settings


def _profile_label(profile_key: str) -> str:
    key, _settings = _profile_settings(profile_key)
    return CROWN_PROFILE_LABELS.get(key, key)


def _pct(value) -> str:
    try:
        return f"{max(0.0, min(1.0, float(value))) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _pct_precise(value) -> str:
    try:
        return f"{max(0.0, min(1.0, float(value))) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _int_text(value) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _num_text(value, digits: int = 0, suffix: str = "") -> str:
    try:
        return f"{float(value):.{max(0, int(digits))}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _build_footer_html() -> str:
    swatch_items = "".join(
        f"<div style='display:inline-flex;flex-direction:column;align-items:center;gap:2px;margin-right:2px;'>"
        f"<div style='background:#{h};width:28px;height:22px;border-radius:3px;border:1px solid #444;'></div>"
        f"<div style='font-size:9px;color:#9cc;font-family:monospace;'>#{h}</div>"
        f"</div>"
        for h in CYAN_SHADE_HEXES
    )
    lower = list(CROWN_PROFILE_TUNING["hsv_lower"])
    upper = list(CROWN_PROFILE_TUNING["hsv_upper"])
    hsv_label = f"H {lower[0]}–{upper[0]} · S {lower[1]}–{upper[1]} · V {lower[2]}–{upper[2]}"
    return (
        "<div class='footer-sections'>"
        "<div class='footer-section'>"
        "<div class='footer-title'>Crown Brick Vision</div>"
        f"<div>TensorRT · trapezoid slot gate · HOLD_FRAMES={HOLD_FRAMES}</div>"
        "</div>"
        "<div class='footer-section'>"
        "<div class='footer-title'>Cyan Calibration Palette</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;'>{swatch_items}</div>"
        f"<div style='margin-top:5px;font-size:10px;color:#9cc;font-family:monospace;'>{hsv_label}</div>"
        "<div style='margin-top:2px;font-size:10px;color:#777;'>Profile dropdown controls HSV range, gate strictness, and close-up rescue.</div>"
        "</div>"
        "</div>"
    )


def _draw_center_guides(frame) -> None:
    if frame is None:
        return
    h, w = frame.shape[:2]
    cx = int(w / 2)
    cy = int(h / 2)
    color = (215, 245, 255)
    cv2.line(frame, (cx, 0), (cx, h), color, 1, cv2.LINE_AA)
    cv2.line(frame, (0, cy), (w, cy), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 9, color, 1, cv2.LINE_AA)


class CrownVisionLivestream:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()
        self.last_read = None
        self.vision = None
        self.server = None
        self.url = format_stream_url(args.stream_host, args.stream_port)
        self.state = {
            "frame": None,
            "text_lines": [],
            "lock": threading.Lock(),
            "show_center_line": True,
            "vision_mode": "cyan",
            "cyan_profile": CROWN_PROFILE_KEY,
            "xyz_workspace": helper_xyz_coords.build_live_position_workspace(
                None,
                visible=False,
                target_name="brick_supply",
                step_name="LIVE_CAMERA",
            ),
        }
        self.thread = None
        self._held_result = None
        self._held_frame = None
        self._miss_count = 0

    def start(self) -> str:
        # Start Flask immediately so the URL is accessible within ~1 s.
        # Camera + TensorRT init happens in the background; the stream server
        # shows a "loading" placeholder until the first frame arrives.
        self.server, self.url = start_stream_server(
            self.state,
            title="Crown Brick Vision Livestream",
            header="",
            footer=_build_footer_html(),
            host=str(self.args.stream_host),
            port=int(self.args.stream_port),
            fps=max(1, int(self.args.stream_fps)),
            jpeg_quality=max(1, min(100, int(self.args.stream_jpeg_quality))),
            img_width=max(320, int(self.args.stream_img_width)),
            sharpen=bool(self.args.sharpen),
            port_tries=max(1, int(self.args.port_tries)),
            cyan_profile_options=CROWN_PROFILE_OPTIONS,
            xyz_workspace_getter=self._get_xyz_workspace,
        )
        self.thread = threading.Thread(target=self._init_and_run, daemon=True)
        self.thread.start()
        return self.url

    def _init_and_run(self) -> None:
        try:
            self.vision = BrickDetector(debug=True)
            with self.state["lock"]:
                requested_profile = str(self.state.get("cyan_profile", CROWN_PROFILE_KEY) or CROWN_PROFILE_KEY)
            profile_key, settings = _profile_settings(requested_profile)
            with self.state["lock"]:
                self.state["cyan_profile"] = profile_key
            self.vision.set_runtime_tuning(**dict(settings))
            self._camera_loop()
        except Exception:
            logging.getLogger("CrownVisionLivestream").exception("Camera init/loop crashed")

    def _get_xyz_workspace(self):
        with self.state["lock"]:
            return self.state.get("xyz_workspace")

    def stop(self) -> None:
        self.running.clear()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.server is not None:
            try:
                self.server.stop()
            except Exception:
                pass
        if self.vision is not None:
            close_fn = getattr(self.vision, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def _camera_loop(self) -> None:
        interval_s = 1.0 / max(1, int(self.args.camera_fps))
        active_profile = None
        while self.running.is_set():
            started = time.monotonic()
            with self.state["lock"]:
                requested_profile = str(self.state.get("cyan_profile", CROWN_PROFILE_KEY) or CROWN_PROFILE_KEY)
            profile_key, settings = _profile_settings(requested_profile)
            if profile_key != active_profile:
                active_profile = profile_key
                with self.state["lock"]:
                    self.state["cyan_profile"] = profile_key
                try:
                    self.vision.set_runtime_tuning(**dict(settings))
                except Exception:
                    logging.getLogger("CrownVisionLivestream").exception(
                        "Failed to apply vision profile %s",
                        profile_key,
                    )
            result = None
            try:
                result = self.vision.read()
            except Exception as exc:
                logging.getLogger("CrownVisionLivestream").exception("Vision read failed: %s", exc)
            self._publish(result)
            elapsed = time.monotonic() - started
            if elapsed < interval_s:
                time.sleep(interval_s - elapsed)

    def _publish(self, result) -> None:
        frame = None
        if self.vision is not None:
            frame = getattr(self.vision, "current_frame", None)
            if frame is None:
                frame = getattr(self.vision, "raw_frame", None)
        if frame is not None:
            frame = frame.copy()

        found = isinstance(result, tuple) and len(result) >= 1 and bool(result[0])
        if found:
            self._held_result = result
            self._held_frame = frame
            self._miss_count = 0
        else:
            self._miss_count += 1
            if self._miss_count <= HOLD_FRAMES and self._held_result is not None:
                result = self._held_result
                frame = self._held_frame

        with self.state["lock"]:
            show_center_line = bool(self.state.get("show_center_line", True))
        if show_center_line and frame is not None:
            _draw_center_guides(frame)

        lines = self._text_lines(result)
        workspace = self._build_workspace(result)
        with self.state["lock"]:
            self.state["frame"] = frame
            self.state["text_lines"] = lines
            self.state["xyz_workspace"] = workspace

    def _build_workspace(self, result):
        found = False
        dist = offset_x = conf_pct = cam_height = None
        if isinstance(result, tuple) and len(result) >= 8:
            found = bool(result[0])
            dist = result[2]
            offset_x = result[3]
            conf_pct = result[4]
            cam_height = result[5]

        with self.state["lock"]:
            previous = self.state.get("xyz_workspace")
        return helper_xyz_coords.build_live_position_workspace(
            previous,
            dist_mm=dist if found else None,
            x_axis_mm=offset_x if found else None,
            y_axis_mm=cam_height if found else None,
            confidence=conf_pct if found else None,
            visible=found,
            target_name="brick_supply",
            step_name="LIVE_CAMERA",
        )

    def _text_lines(self, result) -> list[dict]:
        found = False
        angle = dist = offset_x = conf_pct = cam_height = 0.0
        brick_above = brick_below = False
        if isinstance(result, tuple) and len(result) >= 8:
            found, angle, dist, offset_x, conf_pct, cam_height, brick_above, brick_below = result[:8]

        vision = self.vision
        backend = str(getattr(vision, "inference_backend", "") or "-")
        model_path = getattr(vision, "model_path", None)
        model_name = Path(str(model_path)).name if model_path else "-"
        trust = "model" if bool(getattr(vision, "_trust_detector_boxes", False)) else "shape-gate"
        status = str(getattr(vision, "last_status", "starting") or "starting")
        raw_count = getattr(vision, "last_raw_prediction_count", None)
        candidate_count = getattr(vision, "last_candidate_count", None)
        nms_count = getattr(vision, "last_nms_count", None)
        top_conf = getattr(vision, "last_primary_confidence", None)
        raw_max_conf = getattr(vision, "last_max_confidence", None)
        threshold = getattr(vision, "conf_threshold", None)
        smooth = getattr(vision, "_smooth_alpha", None)
        input_size = getattr(vision, "input_size", None)
        geometry_source = str(getattr(vision, "last_geometry_source", "") or "-")
        focal_px = getattr(vision, "focal_px", None)
        bbox_w_px = getattr(vision, "last_bbox_w_px", None)
        bbox_h_px = getattr(vision, "last_bbox_h_px", None)
        bbox_width_dist = getattr(vision, "last_bbox_width_dist", None)
        bbox_height_dist = getattr(vision, "last_bbox_height_dist", None)
        bbox_cal_dist = getattr(vision, "last_bbox_calibrated_height_dist", None)
        bbox_dist = getattr(vision, "last_bbox_dist", None)
        pre_depth_dist = getattr(vision, "last_pre_depth_dist", None)
        depth_dist = getattr(vision, "last_depth_dist", None)
        raw_dist = getattr(vision, "last_raw_dist", None)
        tri_span_px = getattr(vision, "last_tri_span_px", None)
        tri_span_dist = getattr(vision, "last_tri_span_dist", None)
        depth_stats = getattr(vision, "last_depth_stats", {}) or {}
        if not isinstance(depth_stats, dict):
            depth_stats = {}
        with self.state["lock"]:
            profile_label = _profile_label(self.state.get("cyan_profile", CROWN_PROFILE_KEY))

        lines = [
            {
                "text": f"[ML] PROFILE: {profile_label} | BACKEND: {backend} | TRUST: {trust} | MODEL: {model_name}",
                "color": "#ffffff",
            },
            {
                "text": (
                    f"[ML] SEARCH: {status} | RAW:{_int_text(raw_count)} "
                    f">THR:{_int_text(candidate_count)} NMS:{_int_text(nms_count)}"
                ),
                "color": "#ffffff",
            },
            {
                "text": (
                    f"[ML] TOP CONF: {_pct(top_conf)} | MAX RAW: {_pct_precise(raw_max_conf)} "
                    f"| MIN CONF: {_pct_precise(threshold)} | SMOOTH: {float(smooth):.2f} "
                    f"| INPUT: {_int_text(input_size)}px"
                ),
                "color": "#ffffff",
            },
            {
                "text": f"[ML] GEOMETRY: {geometry_source}",
                "color": "#ffffff",
            },
            {
                "text": f"VISIBLE: {str(bool(found)).lower()}",
                "color": "#00ff00" if bool(found) else "#ff5555",
            },
        ]
        if bool(found):
            bbox_dims = (
                f"{_num_text(bbox_w_px, 0)}x{_num_text(bbox_h_px, 0)}px"
                if bbox_w_px is not None and bbox_h_px is not None
                else "-"
            )
            bbox_calc = (
                f"W={_num_text(bbox_width_dist, 0, 'mm')} "
                f"H={_num_text(bbox_height_dist, 0, 'mm')} "
                f"cal={_num_text(bbox_cal_dist, 0, 'mm')} "
                f"used={_num_text(bbox_dist, 0, 'mm')}"
            )
            span_str = (
                f"{_num_text(tri_span_px, 0)}px -> {_num_text(tri_span_dist, 0, 'mm')}"
                if tri_span_px is not None and tri_span_dist is not None
                else "n/a"
            )
            depth_valid = depth_stats.get("valid_px")
            depth_str = (
                f"{_num_text(depth_dist, 0, 'mm')} ({_int_text(depth_valid)}px)"
                if depth_dist is not None
                else "n/a"
            )
            lines.extend(
                [
                    {"text": f"X-AXIS: {float(offset_x):.1f} mm", "color": "#ffffff"},
                    {"text": f"Y-AXIS: {float(cam_height):.1f} mm", "color": "#ffffff"},
                    {"text": f"DIST:   {float(dist):.0f} mm  (smoothed, {geometry_source})", "color": "#ffffff"},
                    {"text": f"  raw: {_num_text(raw_dist, 0, 'mm')} | pre-depth: {_num_text(pre_depth_dist, 0, 'mm')} | depth: {depth_str}", "color": "#aaaaaa"},
                    {"text": f"  bbox: {bbox_dims} | f={_num_text(focal_px, 0)}px", "color": "#aaaaaa"},
                    {"text": f"  pinhole: {BRICK_WIDTH_MM:.1f}*f/w, {BRICK_HEIGHT_MM:.1f}*f/h -> {bbox_calc}", "color": "#aaaaaa"},
                    {"text": f"  span: {span_str}", "color": "#aaaaaa"},
                    {"text": f"ANGLE:  {float(angle):.1f} deg", "color": "#ffffff"},
                    {"text": f"CONF:   {float(conf_pct):.0f}%", "color": "#ffffff"},
                ]
            )
            stack = []
            if bool(brick_above):
                stack.append("ABOVE")
            if bool(brick_below):
                stack.append("BELOW")
            if stack:
                lines.append({"text": "STACK: " + " + ".join(stack), "color": "#ffd166"})
        return lines


def parse_args(argv=None):
    cfg = load_manual_training_config()
    parser = argparse.ArgumentParser(description="Run the current crown-brick vision livestream.")
    parser.add_argument("--stream-host", default=str(cfg.get("stream_host", "127.0.0.1")))
    parser.add_argument("--stream-port", type=int, default=int(cfg.get("stream_port", 5000)))
    parser.add_argument("--stream-fps", type=int, default=int(cfg.get("stream_fps", 10)))
    parser.add_argument("--stream-jpeg-quality", type=int, default=int(cfg.get("stream_jpeg_quality", 85)))
    parser.add_argument("--stream-img-width", type=int, default=int(cfg.get("stream_img_width", 1600)))
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--port-tries", type=int, default=10)
    parser.add_argument("--sharpen", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _wait_for_port(host, port, timeout=30.0):
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main(argv=None) -> int:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    args = parse_args(argv)
    app = CrownVisionLivestream(args)
    url = app.start()
    _wait_for_port(str(args.stream_host), int(args.stream_port), timeout=10.0)
    print(f"[CROWN] READY — open: {url}", flush=True)
    print("[CROWN] Camera initialising in background (~15 s until live feed).", flush=True)
    print("[CROWN] Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
