from __future__ import annotations

import glob
import os
import re
from typing import Iterable, Optional, Union


CameraSource = Union[int, str]

_VIDEO_NODE_RE = re.compile(r"^/dev/video(\d+)$")


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

    def _append_index(index_value) -> None:
        try:
            idx = int(index_value)
        except (TypeError, ValueError):
            return
        node_path = node_by_index.get(int(idx))
        if node_path:
            _append(node_path)
        _append(int(idx))

    if preferred_index is not None:
        _append_index(preferred_index)

    env_source = _coerce_camera_source(os.getenv(env_source_var))
    if isinstance(env_source, int):
        _append_index(env_source)
    else:
        _append(env_source)

    env_index = os.getenv(env_index_var)
    if env_index is not None and str(env_index).strip():
        _append_index(env_index)

    if known_nodes:
        for path in known_nodes:
            _append(path)
            node_index = _camera_node_index(path)
            if node_index is not None:
                _append(int(node_index))
    else:
        for index in fallback_indices:
            _append_index(index)

    return candidates
