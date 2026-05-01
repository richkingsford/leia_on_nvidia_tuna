"""Camera-only livestream for Leia using the Jetson/NVIDIA camera path."""

from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import threading
import time
from typing import Sequence

import cv2

from helper_brick_detector_yolo import (
    build_negative_cutout_shape_detector,
    detect_single_negative_cutout_brick,
    draw_brick_with_id,
)
from helper_camera_sources import (
    build_nvidia_v4l2_gstreamer_pipeline,
    open_opencv_camera_source,
)
from helper_streaming import start_stream_server


DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_STREAM_FPS = 10
DEFAULT_JPEG_QUALITY = 85
DEFAULT_IMG_WIDTH = 1600


class NvidiaCameraLivestream:
    def __init__(
        self,
        *,
        device: str = DEFAULT_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
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
            "xyz_workspace": None,
        }
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
        cap = open_opencv_camera_source(
            self.pipeline,
            cv2,
            width=self.width,
            height=self.height,
        )
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise RuntimeError(
                "Unable to open Leia camera with NVIDIA V4L2/GStreamer pipeline: "
                f"{self.pipeline}"
            )
        try:
            self._backend = str(cap.getBackendName())
        except Exception:
            self._backend = "GSTREAMER"
        return cap

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
            with self.state["lock"]:
                self.state["frame"] = frame
                self.state["text_lines"] = self._text_lines_locked()

    def _highlight_single_brick(self, frame):
        display = frame.copy()
        try:
            _primary, candidates = detect_single_negative_cutout_brick(
                self._brick_shape_detector,
                display,
            )
        except Exception as exc:
            self._brick_highlight_locked = False
            self._brick_candidate_count = 0
            self._last_error = f"brick highlight error: {exc}"
            return display

        self._brick_candidate_count = len(candidates)
        self._brick_highlight_locked = bool(candidates)

        # Sort top-to-bottom by vertical center: ID 0 = topmost brick
        sorted_bricks = sorted(candidates, key=lambda c: c.get("center_y", 0))
        for brick_id, candidate in enumerate(sorted_bricks):
            draw_brick_with_id(self._brick_shape_detector, display, candidate, brick_id)
        return display

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
        lines = [
            "Leia camera livestream",
            "Camera path: NVIDIA V4L2/GStreamer",
            f"Device: {self.device}",
            f"Resolution: {self.width}x{self.height}",
            f"OpenCV backend: {self._backend or 'GSTREAMER'}",
            f"Frames: {self._frame_count}",
            f"FPS: {self._fps:.1f}",
            (
                f"Bricks detected: {self._brick_candidate_count} "
                f"({'locked' if self._brick_highlight_locked else 'searching'})"
            ),
        ]
        if self._last_error:
            lines.append(f"Camera error: {self._last_error}")
        return lines


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
    camera = NvidiaCameraLivestream(device=args.device, width=args.width, height=args.height)
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
        xyz_workspace_getter=lambda: None,
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
