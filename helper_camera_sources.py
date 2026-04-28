from __future__ import annotations

import glob
import os
import re
from typing import Iterable, Optional, Union


CameraSource = Union[int, str]

_VIDEO_NODE_RE = re.compile(r"^/dev/video(\d+)$")


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
