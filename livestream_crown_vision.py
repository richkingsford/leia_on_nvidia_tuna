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
    BrickDetector,
    CYAN_HSV_WIDE_LOWER,
    CYAN_HSV_WIDE_UPPER,
    CYAN_SHADE_HEXES,
)
from helper_manual_config import load_manual_training_config
from helper_stream_server import format_stream_url
from helper_streaming import start_stream_server
import helper_xyz_coords


# Frames to hold the last good detection before declaring invisible.
# At 15 Hz camera this is ~0.5 s — enough to bridge YOLO misses without
# masking real disappearances.
HOLD_FRAMES = 8

CROWN_PROFILE_KEY = "config9"
CROWN_PROFILE_LABEL = "Config 9"
CROWN_PROFILE_TUNING = {
    "confidence": 0.10,
    "smoothing_alpha": 0.15,
    "hsv_enabled": True,
    "hsv_erode_iterations": 0,
    "hsv_lower": list(CYAN_HSV_WIDE_LOWER),
    "hsv_upper": list(CYAN_HSV_WIDE_UPPER),
    "shape_gate_mode": "shape_match",
}


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


def _build_footer_html() -> str:
    swatch_items = "".join(
        f"<div style='display:inline-flex;flex-direction:column;align-items:center;gap:2px;margin-right:2px;'>"
        f"<div style='background:#{h};width:28px;height:22px;border-radius:3px;border:1px solid #444;'></div>"
        f"<div style='font-size:9px;color:#9cc;font-family:monospace;'>#{h}</div>"
        f"</div>"
        for h in CYAN_SHADE_HEXES
    )
    lower = list(CYAN_HSV_WIDE_LOWER)
    upper = list(CYAN_HSV_WIDE_UPPER)
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
        "<div style='margin-top:2px;font-size:10px;color:#777;'>WIDE range active</div>"
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
        self.vision = BrickDetector(debug=True)
        self.vision.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))

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
            vision_mode_options=[("cyan", "Crown Bricks")],
            cyan_profile_options=[(CROWN_PROFILE_KEY, CROWN_PROFILE_LABEL)],
            xyz_workspace_getter=self._get_xyz_workspace,
        )
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        return self.url

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
        while self.running.is_set():
            started = time.monotonic()
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

        lines = [
            {
                "text": f"[ML] PROFILE: {CROWN_PROFILE_LABEL} | BACKEND: {backend} | TRUST: {trust} | MODEL: {model_name}",
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
                "text": f"VISIBLE: {str(bool(found)).lower()}",
                "color": "#00ff00" if bool(found) else "#ff5555",
            },
        ]
        if bool(found):
            lines.extend(
                [
                    {"text": f"X-AXIS: {float(offset_x):.1f} mm", "color": "#ffffff"},
                    {"text": f"Y-AXIS: {float(cam_height):.1f} mm", "color": "#ffffff"},
                    {"text": f"DIST:   {float(dist):.0f} mm", "color": "#ffffff"},
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


def main(argv=None) -> int:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    args = parse_args(argv)
    app = CrownVisionLivestream(args)
    url = app.start()
    print(f"[CROWN] Livestream URL: {url}", flush=True)
    print("[CROWN] Current brick vision model active. Press Ctrl-C to stop.", flush=True)
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
