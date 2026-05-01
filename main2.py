"""Camera-only livestream for Leia using the Jetson/NVIDIA camera path."""

from __future__ import annotations

import argparse
import math
import signal
import socket
import subprocess
import threading
import time
from copy import deepcopy
from typing import Sequence

import cv2

from helper_brick_detector_yolo import (
    BRICK_HEIGHT_MM,
    BRICK_WIDTH_MM,
    FOCAL_PX_REF,
    FOCAL_REF_WIDTH,
    build_negative_cutout_shape_detector,
    detect_single_negative_cutout_brick,
    draw_brick_with_id,
)
from helper_camera_sources import (
    build_nvidia_v4l2_gstreamer_pipeline,
    candidate_camera_sources,
    open_opencv_camera_source,
)
from helper_streaming import start_stream_server
from helper_xyz_coords import build_live_position_workspace


DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_STREAM_FPS = 10
DEFAULT_JPEG_QUALITY = 85
DEFAULT_IMG_WIDTH = 1600
METRIC_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
METRIC_LABEL_SCALE = 0.46
METRIC_LABEL_THICKNESS = 1
METRIC_LABEL_LINE_GAP_PX = 5
METRIC_LABEL_PAD_X_PX = 7
METRIC_LABEL_PAD_Y_PX = 6
METRIC_LABEL_BLUE_BGR = (255, 96, 0)
METRIC_LABEL_BG_BGR = (8, 16, 28)
METRIC_LABEL_TEXT_BGR = (255, 255, 255)


def _coerce_finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _coerce_positive_float(value) -> float | None:
    number = _coerce_finite_float(value)
    if number is None:
        return None
    if number <= 0.0:
        return None
    return float(number)


def _candidate_bbox(candidate: dict) -> tuple[float, float, float, float] | None:
    if not isinstance(candidate, dict):
        return None
    bbox = candidate.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return float(x), float(y), float(w), float(h)


def _candidate_center(candidate: dict) -> tuple[float, float] | None:
    if not isinstance(candidate, dict):
        return None
    cx = _coerce_finite_float(candidate.get("center_x"))
    cy = _coerce_finite_float(candidate.get("center_y"))
    if cx is not None and cy is not None:
        return float(cx), float(cy)
    bbox = _candidate_bbox(candidate)
    if bbox is None:
        return None
    x, y, w, h = bbox
    return float(x + (w * 0.5)), float(y + (h * 0.5))


def _candidate_scale_px_per_mm(candidate: dict) -> float | None:
    scale = _coerce_positive_float((candidate or {}).get("scale_px_per_mm"))
    if scale is not None:
        return float(scale)

    bbox = _candidate_bbox(candidate)
    if bbox is None:
        return None
    _x, _y, w, h = bbox
    scale_candidates = [
        float(w) / float(BRICK_WIDTH_MM),
        float(h) / float(BRICK_HEIGHT_MM),
    ]
    scale_candidates = [value for value in scale_candidates if value > 0.0]
    if not scale_candidates:
        return None
    return sum(scale_candidates) / float(len(scale_candidates))


def _candidate_metrics(
    candidate: dict,
    *,
    frame_w: int,
    frame_h: int,
    focal_px: float,
) -> dict | None:
    center = _candidate_center(candidate)
    scale = _candidate_scale_px_per_mm(candidate)
    if center is None or scale is None or scale <= 0.0:
        return None
    cx_center = float(frame_w) / 2.0
    cy_center = float(frame_h) / 2.0
    center_x, center_y = center
    return {
        "dist_mm": float(focal_px) / float(scale),
        "x_mm": (float(center_x) - cx_center) / float(scale),
        "y_mm": (float(center_y) - cy_center) / float(scale),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "scale_px_per_mm": float(scale),
    }


def _format_metric_lines(metrics: dict) -> list[str]:
    return [
        f"x {float(metrics['x_mm']):+.0f} mm",
        f"y {float(metrics['y_mm']):+.0f} mm",
        f"dist {float(metrics['dist_mm']):.0f} mm",
    ]


def _draw_brick_metric_label(frame, candidate: dict, metrics: dict) -> None:
    if frame is None or not isinstance(metrics, dict):
        return
    frame_h, frame_w = frame.shape[:2]
    lines = _format_metric_lines(metrics)
    text_sizes = [
        cv2.getTextSize(line, METRIC_LABEL_FONT, METRIC_LABEL_SCALE, METRIC_LABEL_THICKNESS)[0]
        for line in lines
    ]
    line_height = max(1, max(height for _width, height in text_sizes))
    text_width = max(1, max(width for width, _height in text_sizes))
    box_w = int(text_width + (METRIC_LABEL_PAD_X_PX * 2))
    box_h = int((line_height * len(lines)) + (METRIC_LABEL_LINE_GAP_PX * (len(lines) - 1)) + (METRIC_LABEL_PAD_Y_PX * 2))

    bbox = _candidate_bbox(candidate)
    if bbox is not None:
        x, y, w, _h = bbox
        box_x = int(round(x + w + 8.0))
        box_y = int(round(y))
    else:
        center = _candidate_center(candidate)
        if center is None:
            return
        center_x, center_y = center
        box_x = int(round(center_x + 14.0))
        box_y = int(round(center_y - (box_h * 0.5)))

    if box_x + box_w > frame_w - 2:
        if bbox is not None:
            x, _y, _w, _h = bbox
            box_x = int(round(x - box_w - 8.0))
        else:
            box_x = frame_w - box_w - 2
    box_x = max(2, min(frame_w - box_w - 2, box_x))
    box_y = max(2, min(frame_h - box_h - 2, box_y))

    top_left = (int(box_x), int(box_y))
    bottom_right = (int(box_x + box_w), int(box_y + box_h))
    cv2.rectangle(frame, top_left, bottom_right, METRIC_LABEL_BG_BGR, cv2.FILLED)
    cv2.rectangle(frame, top_left, bottom_right, METRIC_LABEL_BLUE_BGR, 1)

    y_cursor = int(box_y + METRIC_LABEL_PAD_Y_PX + line_height)
    x_text = int(box_x + METRIC_LABEL_PAD_X_PX)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x_text, y_cursor),
            METRIC_LABEL_FONT,
            METRIC_LABEL_SCALE,
            (0, 0, 0),
            METRIC_LABEL_THICKNESS + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x_text, y_cursor),
            METRIC_LABEL_FONT,
            METRIC_LABEL_SCALE,
            METRIC_LABEL_TEXT_BGR,
            METRIC_LABEL_THICKNESS,
            cv2.LINE_AA,
        )
        y_cursor += int(line_height + METRIC_LABEL_LINE_GAP_PX)


class NvidiaCameraLivestream:
    def __init__(
        self,
        *,
        device: str = DEFAULT_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        robot=None,
    ):
        self.device = str(device or DEFAULT_DEVICE)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.pipeline = build_nvidia_v4l2_gstreamer_pipeline(
            self.device,
            width=self.width,
            height=self.height,
        )
        self.state = {
            "frame": None,
            "lock": threading.Lock(),
            "text_lines": [],
            "show_center_line": True,
            "step_success_seq": 0,
            "step_success_step": None,
            "step_success_at": 0.0,
            "xyz_workspace": build_live_position_workspace(),
        }
        self._bricks_telemetry: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._frame_count = 0
        self._last_fps_ts = time.monotonic()
        self._last_fps_count = 0
        self._fps = 0.0
        self._last_error = ""
        self._backend = ""
        self._brick_shape_detector = build_negative_cutout_shape_detector()
        self._brick_highlight_locked = False
        self._brick_candidate_count = 0
        self._robot = robot
        self._bt_tree = None
        self._bt_writer = None
        self._bt_status: str = ""
        if robot is not None:
            import py_trees
            from helper_bt_align import build_x_align_tree, _BB_NAMESPACE, _BB_KEY, ALIGN_TURN_DURATION_MS
            self._bt_tree = build_x_align_tree(robot)
            self._bt_writer = py_trees.blackboard.Client(
                name="leia_camera", namespace=_BB_NAMESPACE
            )
            self._bt_writer.register_key(_BB_KEY, access=py_trees.common.Access.WRITE)
            self._bt_tick_interval = ALIGN_TURN_DURATION_MS / 1000.0
        else:
            self._bt_tick_interval = 0.0
        self._bt_last_tick_at = 0.0

    def start(self) -> None:
        self._cap = self._open_camera()
        self._thread = threading.Thread(target=self._loop, name="leia-nvidia-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _open_camera(self):
        candidates = []

        def add_candidate(source) -> None:
            if source is None:
                return
            if source not in candidates:
                candidates.append(source)

        add_candidate(self.pipeline)
        add_candidate(self.device)
        for source in candidate_camera_sources(width=self.width, height=self.height):
            add_candidate(source)

        attempted = []
        for source in candidates:
            attempted.append(str(source))
            cap = open_opencv_camera_source(
                source,
                cv2,
                width=self.width,
                height=self.height,
            )
            if cap is not None and cap.isOpened():
                try:
                    self._backend = str(cap.getBackendName())
                except Exception:
                    self._backend = "unknown"
                return cap
            if cap is not None:
                cap.release()
        raise RuntimeError(
            "Unable to open Leia camera. Tried sources: "
            + "; ".join(attempted)
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                try:
                    self._cap = self._open_camera()
                    self._last_error = ""
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_text()
                    time.sleep(1.0)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._last_error = "camera read returned no frame"
                self._refresh_text()
                time.sleep(0.05)
                continue

            self._last_error = ""
            self._frame_count += 1
            self._update_fps()
            frame = self._highlight_single_brick(frame)
            self._tick_bt()
            with self.state["lock"]:
                self._update_xyz_workspace_locked()
                self.state["frame"] = frame
                self.state["text_lines"] = self._text_lines_locked()

    def _tick_bt(self) -> None:
        """Tick the alignment BT at most once per turn-duration interval."""
        if self._bt_tree is None or self._bt_writer is None:
            return
        now = time.monotonic()
        if now - self._bt_last_tick_at < self._bt_tick_interval:
            return  # previous command still executing
        x_mm = self._bricks_telemetry[0]["x_mm"] if self._bricks_telemetry else None
        self._bt_writer.x_mm = x_mm
        self._bt_tree.tick()
        self._bt_status = self._bt_tree.root.status.name
        self._bt_last_tick_at = now

    def xyz_workspace_snapshot(self) -> dict | None:
        with self.state["lock"]:
            workspace = self.state.get("xyz_workspace")
            return deepcopy(workspace) if isinstance(workspace, dict) else None

    def _highlight_single_brick(self, frame):
        display = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        focal_px = FOCAL_PX_REF * (frame_w / FOCAL_REF_WIDTH)
        cx_center = frame_w / 2.0
        cy_center = frame_h / 2.0

        try:
            _primary, candidates = detect_single_negative_cutout_brick(
                self._brick_shape_detector,
                display,
            )
        except Exception as exc:
            self._brick_highlight_locked = False
            self._brick_candidate_count = 0
            self._bricks_telemetry = []
            self._last_error = f"brick highlight error: {exc}"
            return display

        self._brick_candidate_count = len(candidates)
        self._brick_highlight_locked = bool(candidates)

        # Sort top-to-bottom by vertical center: ID 0 = topmost brick
        sorted_bricks = sorted(candidates, key=lambda c: c.get("center_y", 0))
        brick_metrics: list[tuple[dict, dict]] = []
        for brick_id, candidate in enumerate(sorted_bricks):
            draw_brick_with_id(self._brick_shape_detector, display, candidate, brick_id)
            metrics = _candidate_metrics(
                candidate,
                frame_w=frame_w,
                frame_h=frame_h,
                focal_px=focal_px,
            )
            if metrics is not None:
                brick_metrics.append((candidate, metrics))
                _draw_brick_metric_label(display, candidate, metrics)

        # Telemetry: brick closest to camera center
        def _dist_to_center(row):
            _candidate, metrics = row
            cand_x = float(metrics["center_x"])
            cand_y = float(metrics["center_y"])
            return (cand_x - cx_center) ** 2 + (cand_y - cy_center) ** 2

        closest = min(brick_metrics, key=_dist_to_center) if brick_metrics else None
        if closest is not None:
            _candidate, closest_metrics = closest
            self._bricks_telemetry = [{
                "dist_mm": float(closest_metrics["dist_mm"]),
                "x_mm": float(closest_metrics["x_mm"]),
                "y_mm": float(closest_metrics["y_mm"]),
            }]
        else:
            self._bricks_telemetry = []
        return display

    def _update_xyz_workspace_locked(self) -> None:
        telemetry = self._bricks_telemetry[0] if self._bricks_telemetry else None
        previous = self.state.get("xyz_workspace")
        if isinstance(telemetry, dict):
            self.state["xyz_workspace"] = build_live_position_workspace(
                previous,
                dist_mm=telemetry.get("dist_mm"),
                x_axis_mm=telemetry.get("x_mm"),
                y_axis_mm=telemetry.get("y_mm"),
                confidence=1.0,
                visible=True,
            )
            return
        self.state["xyz_workspace"] = build_live_position_workspace(
            previous,
            dist_mm=None,
            x_axis_mm=None,
            y_axis_mm=None,
            confidence=None,
            visible=False,
        )

    def _update_fps(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_fps_ts
        if elapsed < 1.0:
            return
        frames = self._frame_count - self._last_fps_count
        self._fps = frames / max(elapsed, 0.001)
        self._last_fps_count = self._frame_count
        self._last_fps_ts = now

    def _refresh_text(self) -> None:
        with self.state["lock"]:
            self.state["text_lines"] = self._text_lines_locked()

    def _text_lines_locked(self) -> list[str]:
        if self._bricks_telemetry:
            t = self._bricks_telemetry[0]
            x_sign = "+" if t["x_mm"] >= 0 else ""
            y_sign = "+" if t["y_mm"] >= 0 else ""
            lines = [
                f"x   {x_sign}{t['x_mm']:.0f} mm",
                f"y   {y_sign}{t['y_mm']:.0f} mm",
                f"dist   {t['dist_mm']:.0f} mm",
            ]
        else:
            lines = [
                "x   --",
                "y   --",
                "dist   --",
            ]
        lines += [
            f"FPS: {self._fps:.1f}",
            (
                f"Bricks: {self._brick_candidate_count} "
                f"({'locked' if self._brick_highlight_locked else 'searching'})"
            ),
        ]
        if self._bt_tree is not None:
            lines.append(f"Nav: {self._bt_status or 'idle'}")
        if self._last_error:
            lines.append(f"Error: {self._last_error}")
        return lines


def _open_robot_or_none(
    *,
    enabled: bool,
    require_robot: bool,
    serial_port: str | None = None,
    robot_factory=None,
):
    """Try to open the robot serial connection.

    Returns (robot, status_message).  On failure, returns (None, message)
    unless require_robot=True, in which case raises RuntimeError.

    When robot_factory is None the real Robot class is used in nonfatal mode,
    so a missing Uno does not kill the camera stream.
    """
    if not enabled:
        return None, "Robot disabled"
    if robot_factory is None:
        from helper_robot_control import Robot as _RobotCls
        _real = _RobotCls
        def robot_factory(*, exit_on_failure=True, serial_port=None):  # noqa: E306
            return _real(exit_on_failure=exit_on_failure, serial_port=serial_port)
    try:
        robot = robot_factory(exit_on_failure=False, serial_port=serial_port)
        port = getattr(robot, "SERIAL_PORT", serial_port or "unknown")
        return robot, f"Robot connected on {port}"
    except (Exception, SystemExit) as exc:
        msg = str(exc) if not isinstance(exc, SystemExit) else "serial port not found (see logs above)"
        if require_robot:
            raise RuntimeError(f"Robot required but unavailable: {msg}") from exc
        return None, f"Robot unavailable ({msg}), continuing camera-only"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run Leia's old livestream page with the NVIDIA-backed camera pane."
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="V4L2 camera device path.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Camera capture height.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Livestream bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Livestream starting port.")
    parser.add_argument("--stream-fps", type=int, default=DEFAULT_STREAM_FPS, help="MJPEG stream FPS.")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="MJPEG JPEG quality.")
    parser.add_argument("--img-width", type=int, default=DEFAULT_IMG_WIDTH, help="Displayed camera pane width.")
    parser.add_argument("--no-sharpen", action="store_true", help="Disable stream sharpening.")
    parser.add_argument(
        "--robot",
        dest="robot",
        action="store_true",
        default=True,
        help="Connect to robot for x-axis BT alignment (default).",
    )
    parser.add_argument(
        "--no-robot",
        dest="robot",
        action="store_false",
        help="Skip the robot connection and run camera-only.",
    )
    parser.add_argument("--serial-port", default=None, help="Override robot serial port (e.g. /dev/ttyCH341USB0).")
    return parser.parse_args(argv)


def _lan_urls(port: int) -> list[str]:
    urls: list[str] = []
    candidates: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            candidates.append(str(probe.getsockname()[0]))
        finally:
            probe.close()
    except Exception:
        pass
    try:
        names = {socket.gethostname(), socket.getfqdn()}
        for name in names:
            for info in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM):
                candidates.append(str(info[4][0]))
    except Exception:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=1.0)
        candidates.extend(output.split())
    except Exception:
        pass
    for addr in candidates:
        if addr.startswith("127.") or addr.startswith("169.254.") or addr.startswith("172.17."):
            continue
        url = f"http://{addr}:{int(port)}"
        if url not in urls:
            urls.append(url)
    return urls


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    robot, robot_status = _open_robot_or_none(
        enabled=bool(args.robot),
        require_robot=False,
        serial_port=args.serial_port or None,
    )
    print(f"[LEIA] {robot_status}", flush=True)
    camera = NvidiaCameraLivestream(
        device=args.device, width=args.width, height=args.height, robot=robot
    )
    camera.start()

    server, url = start_stream_server(
        camera.state,
        title="Keyboard Training Livestream",
        header="",
        footer="",
        host=str(args.host),
        port=int(args.port),
        fps=int(args.stream_fps),
        jpeg_quality=int(args.jpeg_quality),
        img_width=int(args.img_width),
        sharpen=not bool(args.no_sharpen),
        port_tries=10,
        ready_timeout_s=3.0,
        xyz_workspace_getter=camera.xyz_workspace_snapshot,
    )

    stop_event = threading.Event()

    def _stop(_signum=None, _frame=None):
        stop_event.set()

    prior_sigint = signal.getsignal(signal.SIGINT)
    prior_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        actual_port = getattr(server, "port", int(args.port))
        if int(actual_port) != int(args.port):
            print(f"[LEIA] Livestream port {args.port} busy; using {actual_port}", flush=True)
        print(f"[LEIA] Livestream URL: {url}", flush=True)
        if str(args.host).strip() in {"0.0.0.0", "::"}:
            for lan_url in _lan_urls(int(actual_port)):
                print(f"[LEIA] LAN URL: {lan_url}", flush=True)
        print(
            f"[LEIA] Camera pane: NVIDIA V4L2/GStreamer {camera.width}x{camera.height} via nvvidconv.",
            flush=True,
        )
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        signal.signal(signal.SIGINT, prior_sigint)
        signal.signal(signal.SIGTERM, prior_sigterm)
        try:
            server.stop()
        finally:
            camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
