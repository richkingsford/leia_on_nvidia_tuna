from __future__ import annotations

import glob
import os
import re
import time
from typing import Iterable, Optional, Union


CameraSource = Union[int, str]

_VIDEO_NODE_RE = re.compile(r"^/dev/video(\d+)$")
DEPTHAI_CAMERA_SOURCE = "depthai"


def is_depthai_source(source: CameraSource) -> bool:
    text = str(source or "").strip().lower()
    return text in {DEPTHAI_CAMERA_SOURCE, "oak", "oak-d"} or text.startswith("depthai:")


def _depthai_socket_name(source: CameraSource) -> str:
    text = str(source or "").strip()
    if ":" not in text:
        return "CAM_A"
    _prefix, suffix = text.split(":", 1)
    socket = str(suffix or "").strip().upper()
    if socket in {"RGB", "COLOR"}:
        return "CAM_A"
    return socket or "CAM_A"


def _import_depthai():
    try:
        import depthai as dai  # type: ignore
    except Exception:
        return None
    return dai


def depthai_camera_available() -> bool:
    dai = _import_depthai()
    if dai is None:
        return False
    try:
        return bool(dai.Device.getAllAvailableDevices())
    except Exception:
        return False


def _positive_float(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(1.0, parsed)


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


class DepthAICapture:
    """Small OpenCV VideoCapture-like wrapper for DepthAI/OAK RGB frames."""

    def __init__(
        self,
        cv2_module,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | float | None = None,
        source: CameraSource = DEPTHAI_CAMERA_SOURCE,
    ) -> None:
        self._width = max(1, int(width or 640))
        self._height = max(1, int(height or 480))
        self._fps = _positive_float(fps or os.getenv("LEIA_DEPTHAI_FPS"), 30.0)
        self._source = source
        self._opened = False
        self._device = None
        self._pipeline = None
        self._queue = None
        self._depth_queue = None
        self._stereo_input_config_queue = None
        self._depth_enabled = _env_enabled("LEIA_DEPTHAI_DEPTH", True)
        self._latest_depth_frame = None
        self._latest_depth_stats = {}
        self._stereo_config_mode = "standard"
        self.camera_intrinsics = None
        self._prop_frame_width = getattr(cv2_module, "CAP_PROP_FRAME_WIDTH", 3)
        self._prop_frame_height = getattr(cv2_module, "CAP_PROP_FRAME_HEIGHT", 4)
        self._prop_fps = getattr(cv2_module, "CAP_PROP_FPS", 5)
        self._open()

    def _open(self) -> None:
        dai = _import_depthai()
        if dai is None:
            return
        try:
            self._device = dai.Device()
            self._pipeline = dai.Pipeline(self._device)
            socket_name = _depthai_socket_name(self._source)
            board_socket = getattr(
                dai.CameraBoardSocket,
                socket_name,
                dai.CameraBoardSocket.CAM_A,
            )
            cam = self._pipeline.create(dai.node.Camera).build(board_socket)
            output = cam.requestOutput((self._width, self._height), fps=self._fps)
            self._queue = output.createOutputQueue(maxSize=4, blocking=False)
            self._configure_intrinsics(dai, board_socket)
            if self._depth_enabled:
                self._configure_stereo_depth(dai, board_socket)
            self._pipeline.start()
            self._opened = True
        except Exception:
            self.release()

    def _configure_intrinsics(self, dai, board_socket) -> None:
        if self._device is None:
            return
        try:
            calib = self._device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                board_socket,
                int(self._width),
                int(self._height),
            )
            fx = float(intrinsics[0][0])
            fy = float(intrinsics[1][1])
            cx = float(intrinsics[0][2])
            cy = float(intrinsics[1][2])
            if fx > 0.0 and fy > 0.0:
                self.camera_intrinsics = {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "width": int(self._width),
                    "height": int(self._height),
                }
        except Exception:
            self.camera_intrinsics = None

    def _configure_stereo_depth(self, dai, align_socket) -> None:
        if self._pipeline is None:
            return
        try:
            left = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = self._pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(align_socket)
            stereo.setOutputSize(int(self._width), int(self._height))
            left.requestFullResolutionOutput().link(stereo.left)
            right.requestFullResolutionOutput().link(stereo.right)
            self._depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
            try:
                self._stereo_input_config_queue = stereo.inputConfig.createInputQueue()
            except Exception:
                self._stereo_input_config_queue = None
        except Exception:
            self._depth_queue = None
            self._stereo_input_config_queue = None

    STEREO_CONFIG_MODES = ["standard", "smooth", "stable", "dense", "fast"]

    def set_stereo_config(self, mode: str) -> bool:
        """Send a runtime StereoDepthConfig to the OAK-D stereo node.

        Modes
        -----
        standard  Current default: LR-check on, subpixel on, no extra filtering.
        smooth    Adds median 7×7 + temporal filter — best for reducing x-offset jitter.
        stable    Adds median 7×7 + temporal + speckle filter — most stable depth.
        dense     LR-check off + median 5×5 — more pixel coverage, slightly less accurate.
        fast      LR-check off, subpixel off, no median — lowest latency.
        """
        mode = str(mode or "").strip().lower()
        if mode not in self.STEREO_CONFIG_MODES:
            return False
        queue = self._stereo_input_config_queue
        if queue is None:
            self._stereo_config_mode = mode
            return False
        dai = _import_depthai()
        if dai is None:
            return False
        try:
            cfg = dai.StereoDepthConfig()
            ac = cfg.algorithmControl
            if mode == "standard":
                ac.enableLeftRightCheck = True
                ac.enableSubpixel = True
                cfg.postProcessing.median = dai.StereoDepthConfig.MedianFilter.MEDIAN_OFF
            elif mode == "smooth":
                ac.enableLeftRightCheck = True
                ac.enableSubpixel = True
                cfg.postProcessing.median = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7
                cfg.postProcessing.temporalFilter.enable = True
                cfg.postProcessing.temporalFilter.alpha = 0.4
                cfg.postProcessing.temporalFilter.delta = 20
            elif mode == "stable":
                ac.enableLeftRightCheck = True
                ac.enableSubpixel = True
                cfg.postProcessing.median = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7
                cfg.postProcessing.temporalFilter.enable = True
                cfg.postProcessing.temporalFilter.alpha = 0.4
                cfg.postProcessing.temporalFilter.delta = 20
                cfg.postProcessing.speckleFilter.enable = True
                cfg.postProcessing.speckleFilter.speckleRange = 50
            elif mode == "dense":
                ac.enableLeftRightCheck = False
                ac.enableSubpixel = True
                cfg.postProcessing.median = dai.StereoDepthConfig.MedianFilter.KERNEL_5x5
            elif mode == "fast":
                ac.enableLeftRightCheck = False
                ac.enableSubpixel = False
                cfg.postProcessing.median = dai.StereoDepthConfig.MedianFilter.MEDIAN_OFF
            queue.send(cfg)
            self._stereo_config_mode = mode
            return True
        except Exception:
            return False

    def isOpened(self) -> bool:
        return bool(self._opened and self._queue is not None)

    def read(self):
        if not self.isOpened():
            return False, None
        packet = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                candidate = self._queue.tryGet()
            except Exception:
                self.release()
                return False, None
            if candidate is not None:
                packet = candidate
                break
            time.sleep(0.005)
        if packet is None:
            return False, None
        try:
            frame = packet.getCvFrame()
        except Exception:
            return False, None
        if frame is None:
            return False, None
        self._update_depth_frame()
        return True, frame

    @property
    def latest_depth_frame(self):
        return self._latest_depth_frame

    @property
    def latest_depth_stats(self) -> dict:
        return dict(self._latest_depth_stats)

    def _update_depth_frame(self) -> None:
        if self._depth_queue is None:
            return
        latest = None
        try:
            while True:
                candidate = self._depth_queue.tryGet()
                if candidate is None:
                    break
                latest = candidate
        except Exception:
            self._depth_queue = None
            self._latest_depth_frame = None
            return
        if latest is None:
            return
        try:
            frame = latest.getFrame()
        except Exception:
            return
        if frame is None:
            return
        self._latest_depth_frame = frame

    def depth_at_region(self, center_x, center_y, bbox=None, *, radius_px: int = 18):
        frame = self._latest_depth_frame
        if frame is None:
            return None
        try:
            import numpy as np
        except Exception:
            return None
        h, w = frame.shape[:2]
        try:
            cx = int(round(float(center_x)))
            cy = int(round(float(center_y)))
        except (TypeError, ValueError):
            return None
        if bbox is not None:
            try:
                bx, by, bw, bh = [float(v) for v in bbox]
                shrink_x = max(3.0, bw * 0.25)
                shrink_y = max(3.0, bh * 0.25)
                x0 = int(max(0, round(bx + shrink_x)))
                y0 = int(max(0, round(by + shrink_y)))
                x1 = int(min(w, round(bx + bw - shrink_x)))
                y1 = int(min(h, round(by + bh - shrink_y)))
            except (TypeError, ValueError):
                x0 = max(0, cx - int(radius_px))
                y0 = max(0, cy - int(radius_px))
                x1 = min(w, cx + int(radius_px) + 1)
                y1 = min(h, cy + int(radius_px) + 1)
        else:
            radius = max(2, int(radius_px))
            x0 = max(0, cx - radius)
            y0 = max(0, cy - radius)
            x1 = min(w, cx + radius + 1)
            y1 = min(h, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        search_boxes = [(x0, y0, x1, y1)]
        if bbox is not None:
            try:
                bx, by, bw, bh = [float(v) for v in bbox]
                search_boxes.append(
                    (
                        int(max(0, round(bx))),
                        int(max(0, round(by))),
                        int(min(w, round(bx + bw))),
                        int(min(h, round(by + bh))),
                    )
                )
            except (TypeError, ValueError):
                pass
        for radius in (int(radius_px) * 2, int(radius_px) * 4):
            radius = max(3, int(radius))
            search_boxes.append(
                (
                    max(0, cx - radius),
                    max(0, cy - radius),
                    min(w, cx + radius + 1),
                    min(h, cy + radius + 1),
                )
            )
        valid = np.asarray([], dtype=frame.dtype)
        used_box = (x0, y0, x1, y1)
        for sx0, sy0, sx1, sy1 in search_boxes:
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            roi = np.asarray(frame[sy0:sy1, sx0:sx1])
            candidate_valid = roi[(roi > 0) & (roi < 5000)]
            if candidate_valid.size >= 3:
                valid = candidate_valid
                used_box = (sx0, sy0, sx1, sy1)
                break
            if candidate_valid.size > valid.size:
                valid = candidate_valid
                used_box = (sx0, sy0, sx1, sy1)
        if valid.size < 3:
            self._latest_depth_stats = {
                "valid_px": int(valid.size),
                "bbox": [int(v) for v in used_box],
            }
            return None
        median_mm = float(np.median(valid))
        self._latest_depth_stats = {
            "valid_px": int(valid.size),
            "median_mm": median_mm,
            "bbox": [int(v) for v in used_box],
        }
        return median_mm

    def get(self, prop_id) -> float:
        if prop_id == self._prop_frame_width:
            return float(self._width)
        if prop_id == self._prop_frame_height:
            return float(self._height)
        if prop_id == self._prop_fps:
            return float(self._fps)
        return 0.0

    def set(self, prop_id, value) -> bool:
        if prop_id == self._prop_frame_width:
            self._width = max(1, int(value))
            return True
        if prop_id == self._prop_frame_height:
            self._height = max(1, int(value))
            return True
        if prop_id == self._prop_fps:
            self._fps = max(1, float(value))
            return True
        return False

    def getBackendName(self) -> str:
        return "DEPTHAI"

    def release(self) -> None:
        self._opened = False
        stereo_config_queue = self._stereo_input_config_queue
        self._stereo_input_config_queue = None
        if stereo_config_queue is not None:
            close = getattr(stereo_config_queue, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        depth_queue = self._depth_queue
        self._depth_queue = None
        if depth_queue is not None:
            close = getattr(depth_queue, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        queue = self._queue
        self._queue = None
        if queue is not None:
            close = getattr(queue, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            stop = getattr(pipeline, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        device = self._device
        self._device = None
        if device is not None:
            close = getattr(device, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def is_gstreamer_pipeline(source: CameraSource) -> bool:
    return isinstance(source, str) and "!" in source


def build_nvidia_v4l2_gstreamer_pipeline(
    device: str,
    *,
    width: int = 640,
    height: int = 480,
) -> str:
    """Build the Jetson USB-camera pipeline used by OpenCV appsink readers."""
    w = max(1, int(width))
    h = max(1, int(height))
    dev = str(device or "/dev/video0").replace('"', '\\"')
    return (
        f'v4l2src device="{dev}" io-mode=2 do-timestamp=true ! '
        f"video/x-raw,format=YUY2,width={w},height={h} ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_nvidia_argus_gstreamer_pipeline(
    *,
    sensor_id: int = 0,
    width: int = 640,
    height: int = 480,
    framerate: int = 30,
) -> str:
    """Build the Jetson CSI/Argus pipeline used by OpenCV appsink readers."""
    sid = max(0, int(sensor_id))
    w = max(1, int(width))
    h = max(1, int(height))
    fps = max(1, int(framerate))
    return (
        f"nvarguscamerasrc sensor-id={sid} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},format=NV12,framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def opencv_backends_for_source(source: CameraSource, cv2_module) -> list[int]:
    if is_gstreamer_pipeline(source):
        if hasattr(cv2_module, "CAP_GSTREAMER"):
            return [cv2_module.CAP_GSTREAMER]
        return [cv2_module.CAP_ANY]

    backends: list[int] = []
    if hasattr(cv2_module, "CAP_V4L2"):
        backends.append(cv2_module.CAP_V4L2)
    if hasattr(cv2_module, "CAP_GSTREAMER"):
        backends.append(cv2_module.CAP_GSTREAMER)
    backends.append(cv2_module.CAP_ANY)
    return backends


def open_opencv_camera_source(
    source: CameraSource,
    cv2_module,
    *,
    width: int | None = None,
    height: int | None = None,
):
    if is_depthai_source(source):
        cap = DepthAICapture(cv2_module, width=width, height=height, source=source)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    for backend in opencv_backends_for_source(source, cv2_module):
        cap = cv2_module.VideoCapture(source, backend)
        if cap is not None and cap.isOpened():
            pipeline_source = is_gstreamer_pipeline(source)
            if not pipeline_source and hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
            if not pipeline_source:
                if width is not None:
                    cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, int(width))
                if height is not None:
                    cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, int(height))
            return cap
        if cap is not None:
            cap.release()
    return None


def _coerce_camera_source(value) -> CameraSource | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _camera_node_index(path: str) -> int | None:
    match = _VIDEO_NODE_RE.fullmatch(str(path or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def existing_camera_nodes() -> list[str]:
    numbered_nodes: list[tuple[int, str]] = []
    for raw_path in glob.glob("/dev/video*"):
        path = str(raw_path or "").strip()
        index = _camera_node_index(path)
        if index is None:
            continue
        numbered_nodes.append((int(index), path))
    numbered_nodes.sort(key=lambda item: (int(item[0]), str(item[1])))
    return [str(path) for _index, path in numbered_nodes]


def existing_camera_indices() -> list[int]:
    indices: list[int] = []
    for path in existing_camera_nodes():
        index = _camera_node_index(path)
        if index is None:
            continue
        indices.append(int(index))
    return indices


def candidate_camera_sources(
    preferred_index: Optional[int] = None,
    *,
    env_source_var: str = "LEIA_CAMERA_SOURCE",
    env_index_var: str = "LEIA_CAMERA_INDEX",
    fallback_indices: Iterable[int] = (0, 1, 2, 3),
    width: int = 640,
    height: int = 480,
    include_nvidia_pipelines: bool = True,
) -> list[CameraSource]:
    candidates: list[CameraSource] = []
    known_nodes = existing_camera_nodes()
    node_by_index = {
        int(index): str(path)
        for index, path in (
            (_camera_node_index(path), path)
            for path in known_nodes
        )
        if index is not None
    }

    def _append(source) -> None:
        value = _coerce_camera_source(source)
        if value is None:
            return
        if value not in candidates:
            candidates.append(value)

    def _append_source(source) -> None:
        value = _coerce_camera_source(source)
        if value is None:
            return
        if isinstance(value, int):
            _append_index(value)
            return
        if include_nvidia_pipelines and not is_gstreamer_pipeline(value):
            node_index = _camera_node_index(value)
            if node_index is not None:
                _append(build_nvidia_v4l2_gstreamer_pipeline(value, width=width, height=height))
        _append(value)

    def _append_index(index_value) -> None:
        try:
            idx = int(index_value)
        except (TypeError, ValueError):
            return
        node_path = node_by_index.get(int(idx))
        if include_nvidia_pipelines:
            pipeline_node = node_path or f"/dev/video{idx}"
            _append(build_nvidia_v4l2_gstreamer_pipeline(pipeline_node, width=width, height=height))
        if node_path:
            _append(node_path)
        _append(int(idx))

    if preferred_index is not None:
        _append_index(preferred_index)

    env_source = _coerce_camera_source(os.getenv(env_source_var))
    _append_source(env_source)

    env_index = os.getenv(env_index_var)
    if env_index is not None and str(env_index).strip():
        _append_index(env_index)

    if depthai_camera_available():
        _append(DEPTHAI_CAMERA_SOURCE)

    if known_nodes:
        for path in known_nodes:
            _append_source(path)
            node_index = _camera_node_index(path)
            if node_index is not None:
                _append(int(node_index))
    else:
        if include_nvidia_pipelines:
            _append(build_nvidia_argus_gstreamer_pipeline(width=width, height=height))
        for index in fallback_indices:
            _append_index(index)

    return candidates
