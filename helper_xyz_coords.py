#!/usr/bin/env python3
"""Workspace xyz tracker and live top-down snapshot renderer."""

from __future__ import annotations

import html
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

XYZ_LAYOUT_DIR = Path(__file__).resolve().parent / "trials"
LIVE_HTML_PATH = XYZ_LAYOUT_DIR / "run_view.html"
LIVE_SVG_PATH = XYZ_LAYOUT_DIR / "workspace.svg"
LIVE_MAST_SVG_PATH = XYZ_LAYOUT_DIR / "mast_view.svg"
LIVE_JSON_PATH = XYZ_LAYOUT_DIR / "workspace.json"
RUN_VIEWS_DIR = XYZ_LAYOUT_DIR / "run_views"
RUN_VIEWS_INDEX_PATH = RUN_VIEWS_DIR / "index.html"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
SCHEMA_VERSION = 1

DEFAULT_CAMERA_Z_MM = 145.0
DEFAULT_BRICK_SUPPLY_POS_MM = {"x_mm": 0.0, "y_mm": -180.0, "z_mm": 0.0}
DEFAULT_WALL_POS_MM = {"x_mm": 180.0, "y_mm": 0.0, "z_mm": 0.0, "theta_deg": 180.0}
DEFAULT_WALL_RENDER_LENGTH_MM = 320.0
DEFAULT_STACK_RENDER_FOOTPRINT_MM = 54.0
DEFAULT_STACK_HEIGHT_MM = 44.0
DEFAULT_BRICK_MM = {"length_mm": 44.0, "width_mm": 22.0, "height_mm": 22.0}
BIRD_HELD_BRICK_RENDER_SCALE = 0.5
DEFAULT_WORKSPACE_STEP_TARGETS = (
    (1, 2, "wall"),
    (3, 9, "brick_supply"),
    (10, 16, "wall"),
)
ROBOT_NOSE_TO_TAIL_MM = 68.0 * 1.7  # Made 1.7x longer for better visibility
ROBOT_TAIL_CENTER_MM = 58.0 * 1.7
ROBOT_HALF_WIDTH_MM = 22.0
ROBOT_OBJECT_MARGIN_MM = 6.0
BIRDSEYE_VIEW_MARGIN_MM = 6.0
BIRDSEYE_MIN_SPAN_X_MM = 68.0
BIRDSEYE_MIN_SPAN_Y_MM = 52.0
BIRD_STACK_SHIFT_X_PX = 200.0
BIRD_SUPPLY_RENDER_SHIFT_X_PX = -36.0
BIRD_HISTORY_MAX_POINTS = 18
BIRD_HISTORY_COLOR_OLDER = "#63d7ff"
BIRD_HISTORY_COLOR_NEWEST = "#0b3d91"
CURRENT_POSITION_COLOR = "#2563eb"
CURRENT_POSITION_HALO_COLOR = "#60a5fa"
CURRENT_POSITION_LABEL_COLOR = "#0b3d91"
STEP_HISTORY_COLOR_CLOSER = "#2f9e44"
STEP_HISTORY_COLOR_FURTHER = "#d94841"
STEP_HISTORY_COLOR_NEUTRAL = "#c59d2a"
STEP_HISTORY_COLOR_UNKNOWN = "#7b7668"
BIRD_HISTORY_DOT_RADIUS_PX = 6.0
BIRD_DIST_TRACK_HALF_RANGE_MM = 60.0
BIRD_DIST_TRACK_MARGIN_PX = 42.0
BIRD_DIST_TRACK_STACK_GAP_PX = 16.0
BIRD_X_AXIS_TRACK_DEFAULT_EXTENT_MM = 10.0
BIRD_X_AXIS_TRACK_EXPANDED_EXTENT_MM = 20.0
BIRD_X_AXIS_TRACK_EXPAND_THRESHOLD_MM = 10.0
BIRD_X_AXIS_TRACK_HALF_SPAN_PX = 54.0
BIRD_DIST_TARGET_MM = 149.26
BIRD_DIST_TOL_MM = 5.0
BIRD_X_AXIS_TARGET_MM = 6.03
BIRD_X_AXIS_TOL_MM = 5.0
MAST_Y_AXIS_TARGET_MM = 4.13
MAST_Y_AXIS_TOL_MM = 2.3

SVG_WIDTH = 980
BIRD_SVG_HEIGHT = 228
BIRD_SVG_PADDING = 28
BIRD_STACK_CENTER_Y_PX = 118.0
SVG_PADDING = 80
MAST_SVG_HEIGHT = 360
MAST_VIEW_MIN_MM = 0.0
MAST_VIEW_MAX_MM = 20.0
MAST_Y_AXIS_DEFAULT_EXTENT_MM = 10.0
MAST_Y_AXIS_EXPANDED_EXTENT_MM = 20.0
MAST_Y_AXIS_EXPAND_THRESHOLD_MM = 10.0
MAST_HISTORY_LINE_CURRENT = {"recency": "current", "stroke": "#0b3d91", "stroke_width": "1"}
MAST_HISTORY_LINE_PREVIOUS = {"recency": "n-1", "stroke": "#2563eb", "stroke_width": "1"}
MAST_HISTORY_LINE_PREVIOUS_2 = {"recency": "n-2", "stroke": "#63d7ff", "stroke_width": "1"}
MAST_HISTORY_LINE_OLDER = {"recency": "older", "stroke": "#d3d7dd", "stroke_width": "1"}
_PROCESS_MODEL_CACHE = {"mtime_ns": None, "payload": None}


def _coerce_float(value, fallback=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value, fallback=None):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _norm_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _deepcopy(value):
    return deepcopy(value)


def _normalize_step_key(step) -> str | None:
    value = getattr(step, "value", step)
    text = str(value or "").strip().upper()
    return text or None


def _load_process_model_config() -> dict:
    try:
        stat = PROCESS_MODEL_FILE.stat()
        mtime_ns = int(getattr(stat, "st_mtime_ns", 0))
    except OSError:
        _PROCESS_MODEL_CACHE["mtime_ns"] = None
        _PROCESS_MODEL_CACHE["payload"] = {}
        return {}
    if _PROCESS_MODEL_CACHE["mtime_ns"] == mtime_ns and isinstance(_PROCESS_MODEL_CACHE["payload"], dict):
        return _PROCESS_MODEL_CACHE["payload"]
    try:
        with open(PROCESS_MODEL_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    _PROCESS_MODEL_CACHE["mtime_ns"] = mtime_ns
    _PROCESS_MODEL_CACHE["payload"] = payload
    return payload


def _process_step_rules(world) -> dict:
    rules = getattr(world, "process_rules", None)
    if isinstance(rules, dict) and rules:
        return rules
    steps = (_load_process_model_config().get("steps") or {})
    return steps if isinstance(steps, dict) else {}


def _process_step_order(world) -> list[str]:
    order = []
    for raw_key in _process_step_rules(world).keys():
        key = _normalize_step_key(raw_key)
        if key and key not in order:
            order.append(key)
    return order


def _step_process_cfg(world, step_key: str | None) -> dict:
    if not step_key:
        return {}
    for raw_key, cfg in _process_step_rules(world).items():
        if _normalize_step_key(raw_key) == step_key and isinstance(cfg, dict):
            return cfg
    return {}


def _step_number_for_key(world, step_key: str | None) -> int | None:
    if not step_key:
        return None
    for idx, key in enumerate(_process_step_order(world), start=1):
        if key == step_key:
            return int(idx)
    return None


def _workspace_step_target_ranges() -> list[tuple[int, int, str]]:
    cfg = (_load_process_model_config().get("workspace") or {})
    raw_ranges = cfg.get("step_target_ranges")
    parsed = []
    if isinstance(raw_ranges, (list, tuple)):
        for item in raw_ranges:
            if not isinstance(item, dict):
                continue
            min_step = _coerce_int(item.get("min_step"), None)
            max_step = _coerce_int(item.get("max_step"), None)
            target = str(item.get("target") or "").strip().lower()
            if min_step is None or max_step is None or target not in {"wall", "brick_supply"}:
                continue
            parsed.append((int(min_step), int(max_step), target))
    if parsed:
        return parsed
    return list(DEFAULT_WORKSPACE_STEP_TARGETS)


def _target_label(target: str | None) -> str | None:
    if target == "wall":
        return "Wall"
    if target == "brick_supply":
        return "Brick Supply"
    return None


def _workspace_target_for_step(world, step_key: str | None, step_number: int | None) -> str | None:
    step_cfg = _step_process_cfg(world, step_key)
    explicit = str(step_cfg.get("workspace_target") or "").strip().lower()
    if explicit in {"wall", "brick_supply"}:
        return explicit
    if step_number is not None:
        for min_step, max_step, target in _workspace_step_target_ranges():
            if int(min_step) <= int(step_number) <= int(max_step):
                return target
    if step_key in {"FIND_WALL", "EXIT_WALL", "FIND_WALL2", "APPROACH_VECTOR_WALL", "FIND_TOPMOST_BRICK_WALL", "BRICK_LOCK_WALL", "POSITION_BRICK", "SEAT_BRICK2", "RETREAT"}:
        return "wall"
    if step_key in {"FIND_BRICK", "APPROACH_VECTOR_BRICK_SUPPLY", "FIND_TOPMOST_BRICK", "BRICK_LOCK", "ALIGN_BRICK", "SEAT_BRICK", "ELEVATE_BRICK"}:
        return "brick_supply"
    return None


def _default_object_state() -> dict:
    return {
        "brick_supply": {
            "name": "brick_supply",
            "label": "Brick Supply",
            "x_mm": float(DEFAULT_BRICK_SUPPLY_POS_MM["x_mm"]),
            "y_mm": float(DEFAULT_BRICK_SUPPLY_POS_MM["y_mm"]),
            "z_mm": float(DEFAULT_BRICK_SUPPLY_POS_MM["z_mm"]),
            "theta_deg": 0.0,
            "count": None,
            "height_mm": None,
            "confidence": None,
            "visible": False,
            "fixed": True,
            "render_footprint_mm": float(DEFAULT_STACK_RENDER_FOOTPRINT_MM),
            "source": "default_layout_seed",
            "last_seen_ts": None,
        },
        "wall": {
            "name": "wall",
            "label": "Wall",
            "x_mm": float(DEFAULT_WALL_POS_MM["x_mm"]),
            "y_mm": float(DEFAULT_WALL_POS_MM["y_mm"]),
            "z_mm": float(DEFAULT_WALL_POS_MM["z_mm"]),
            "theta_deg": float(DEFAULT_WALL_POS_MM["theta_deg"]),
            "length_mm": float(DEFAULT_WALL_RENDER_LENGTH_MM),
            "render_footprint_mm": float(DEFAULT_STACK_RENDER_FOOTPRINT_MM),
            "height_mm": None,
            "confidence": None,
            "visible": False,
            "valid": False,
            "fixed": True,
            "source": "default_layout_seed",
            "last_seen_ts": None,
        },
    }


def _default_workspace_state(*, render_enabled: bool) -> dict:
    return {
        "schema_version": int(SCHEMA_VERSION),
        "updated_at": time.time(),
        "robot": {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 0.0,
            "theta_deg": 0.0,
            "lift_mm": 0.0,
        },
        "raw_robot": {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 0.0,
            "theta_deg": 0.0,
            "lift_mm": 0.0,
        },
        "leia": {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": float(DEFAULT_CAMERA_Z_MM),
            "theta_deg": 0.0,
            "label": "Leia",
        },
        "objects": _default_object_state(),
        "active_target": {
            "step_name": None,
            "step_number": None,
            "object_name": None,
            "label": None,
            "history_step_seq": 0,
        },
        "held_brick": {
            "held": False,
            **dict(DEFAULT_BRICK_MM),
        },
        "last_visible_brick": {
            "visible": False,
            "dist_mm": None,
            "x_axis_mm": None,
            "y_axis_mm": None,
            "confidence": None,
            "angle_deg": None,
        },
        "history": [],
        "history_step_seq": 0,
        "history_step_name": None,
        "micro_adjust_phase": False,
    }


def _normalized_render_snapshot(state: dict | None) -> dict:
    snapshot = _default_workspace_state(render_enabled=False)
    if not isinstance(state, dict):
        return snapshot

    for key in (
        "schema_version",
        "updated_at",
        "history",
        "history_step_seq",
        "history_step_name",
        "micro_adjust_phase",
        "run_replay",
        "run_log_path",
    ):
        if key in state:
            snapshot[key] = _deepcopy(state.get(key))

    for key in (
        "robot",
        "raw_robot",
        "leia",
        "active_target",
        "held_brick",
        "last_visible_brick",
    ):
        override = state.get(key)
        if isinstance(override, dict):
            merged = dict(snapshot.get(key) or {})
            merged.update(_deepcopy(override))
            snapshot[key] = merged

    raw_objects = state.get("objects")
    if isinstance(raw_objects, dict):
        merged_objects = dict(snapshot.get("objects") or {})
        for obj_name, default_obj in list(merged_objects.items()):
            override = raw_objects.get(obj_name)
            if isinstance(default_obj, dict) and isinstance(override, dict):
                obj_merged = dict(default_obj)
                obj_merged.update(_deepcopy(override))
                merged_objects[obj_name] = obj_merged
        for obj_name, override in raw_objects.items():
            if obj_name not in merged_objects:
                merged_objects[obj_name] = _deepcopy(override)
        snapshot["objects"] = merged_objects

    return snapshot


def ensure_workspace(world, *, render_enabled: bool = True) -> dict:
    state = getattr(world, "_xyz_workspace", None)
    if isinstance(state, dict):
        return state
    state = _default_workspace_state(render_enabled=bool(render_enabled))
    setattr(world, "_xyz_workspace", state)
    return state


def workspace_snapshot(world) -> dict | None:
    state = getattr(world, "_xyz_workspace", None)
    if not isinstance(state, dict):
        return None
    return _deepcopy(state)


def build_live_position_workspace(
    previous_state: dict | None = None,
    *,
    dist_mm=None,
    x_axis_mm=None,
    y_axis_mm=None,
    confidence=None,
    visible: bool | None = None,
    target_name: str = "brick_supply",
    step_name: str = "LIVE_CAMERA",
    camera_height_mm=None,
    lift_mm=None,
    history_min_interval_s: float = 0.25,
    history_maxlen: int = 60,
) -> dict:
    """Build/update a workspace snapshot from live camera position metrics."""
    previous = previous_state if isinstance(previous_state, dict) else None
    state = (
        _normalized_render_snapshot(previous)
        if previous is not None
        else _default_workspace_state(render_enabled=False)
    )
    now = time.time()

    target_key = str(target_name or "").strip().lower()
    if target_key not in {"wall", "brick_supply"}:
        target_key = "brick_supply"
    step_key = _normalize_step_key(step_name) or "LIVE_CAMERA"
    step_seq = int(_coerce_int(state.get("history_step_seq"), 0) or 0)
    previous_step = _normalize_step_key(state.get("history_step_name"))
    if step_seq <= 0 or previous_step != step_key:
        step_seq = max(1, step_seq + 1)
    state["history_step_name"] = step_key
    state["history_step_seq"] = int(step_seq)

    active = state.setdefault("active_target", {})
    active["step_name"] = step_key
    active["step_number"] = None
    active["object_name"] = target_key
    active["label"] = _target_label(target_key)
    active["history_step_seq"] = int(step_seq)

    dist_val = _coerce_float(dist_mm, None)
    x_val = _coerce_float(x_axis_mm, None)
    y_val = _coerce_float(y_axis_mm, None)
    confidence_val = _coerce_float(confidence, None)
    visible_val = bool(visible) if visible is not None else dist_val is not None

    robot = state.setdefault("robot", {})
    robot["lift_mm"] = float(_coerce_float(lift_mm, robot.get("lift_mm", 0.0)) or 0.0)
    robot["z_mm"] = 0.0
    obj = (state.get("objects") or {}).get(target_key)
    if isinstance(obj, dict):
        origin_robot = {"x_mm": 0.0, "y_mm": 0.0, "theta_deg": 0.0}
        robot["theta_deg"] = _norm_deg(_heading_toward_object_deg(origin_robot, obj))
        if visible_val and dist_val is not None:
            _reconcile_robot_pose_from_fixed_object(
                state,
                target_key,
                distance_mm=float(dist_val),
                bearing_deg=_observation_bearing_deg(
                    {"dist_mm": dist_val, "x_axis_mm": x_val if x_val is not None else 0.0}
                ),
            )
            obj["visible"] = True
            obj["last_seen_ts"] = now

    leia = state.setdefault("leia", {})
    leia["x_mm"] = float(_coerce_float(robot.get("x_mm"), 0.0) or 0.0)
    leia["y_mm"] = float(_coerce_float(robot.get("y_mm"), 0.0) or 0.0)
    leia["theta_deg"] = float(_coerce_float(robot.get("theta_deg"), 0.0) or 0.0)
    if camera_height_mm is None:
        camera_height_val = float(DEFAULT_CAMERA_Z_MM) + float(
            _coerce_float(robot.get("lift_mm"), 0.0) or 0.0
        )
    else:
        camera_height_val = float(
            _coerce_float(camera_height_mm, DEFAULT_CAMERA_Z_MM) or DEFAULT_CAMERA_Z_MM
        )
    leia["z_mm"] = float(camera_height_val)

    state["last_visible_brick"] = {
        "visible": bool(visible_val),
        "dist_mm": dist_val,
        "x_axis_mm": x_val,
        "y_axis_mm": y_val,
        "confidence": confidence_val,
        "angle_deg": None,
    }
    state["updated_at"] = now

    last_history_ts = _coerce_float((previous or {}).get("_live_position_last_history_ts"), None)
    interval_s = max(0.0, float(_coerce_float(history_min_interval_s, 0.25) or 0.0))
    should_append = (
        bool(visible_val)
        and dist_val is not None
        and x_val is not None
        and (last_history_ts is None or (now - float(last_history_ts)) >= interval_s)
    )
    if should_append:
        entry = {
            "type": "observation",
            "reason": "live_position",
            "ts": now,
            "step_name": step_key,
            "step_seq": int(step_seq),
            "target_name": target_key,
            "target_visible": True,
            "dist_mm": float(dist_val),
            "target_range_mm": float(dist_val),
            "x_axis_mm": float(x_val),
            "y_axis_mm": None if y_val is None else float(y_val),
            "camera_height_mm": float(camera_height_val),
            "current_lift_mm": float(_coerce_float(robot.get("lift_mm"), 0.0) or 0.0),
        }
        _append_history(state, entry, maxlen=max(1, int(history_maxlen)))
        state["_live_position_last_history_ts"] = now
    else:
        state["_live_position_last_history_ts"] = last_history_ts

    return state


def _append_history(state: dict, entry: dict, *, maxlen: int = 60) -> None:
    history = state.setdefault("history", [])
    row = dict(entry)
    active = state.get("active_target") or {}
    robot = state.get("robot") or {}
    leia = state.get("leia") or {}
    last_visible = state.get("last_visible_brick") or {}
    row.setdefault("step_name", active.get("step_name"))
    row.setdefault("step_number", active.get("step_number"))
    row.setdefault("step_seq", active.get("history_step_seq"))
    row.setdefault("target_name", active.get("object_name"))
    row.setdefault("camera_height_mm", float(_coerce_float(leia.get("z_mm"), DEFAULT_CAMERA_Z_MM) or DEFAULT_CAMERA_Z_MM))
    row.setdefault("current_lift_mm", float(_coerce_float(robot.get("lift_mm"), 0.0) or 0.0))
    row.setdefault("target_visible", bool(last_visible.get("visible", False)))
    dist_mm = _coerce_float(last_visible.get("dist_mm"), None)
    if dist_mm is not None:
        row.setdefault("dist_mm", float(dist_mm))
        row.setdefault("target_range_mm", float(dist_mm))
    x_axis_mm = _coerce_float(last_visible.get("x_axis_mm"), None)
    row.setdefault("x_axis_mm", x_axis_mm)
    if x_axis_mm is not None:
        row.setdefault("x_axis_abs_mm", abs(float(x_axis_mm)))
    y_axis_mm = _coerce_float(last_visible.get("y_axis_mm"), None)
    row.setdefault("y_axis_mm", y_axis_mm)
    if y_axis_mm is not None:
        row.setdefault("y_axis_abs_mm", abs(float(y_axis_mm)))
    if row.get("type") == "motion":
        row.setdefault("x_mm", float(_coerce_float(robot.get("x_mm"), 0.0) or 0.0))
        row.setdefault("y_mm", float(_coerce_float(robot.get("y_mm"), 0.0) or 0.0))
        row.setdefault("theta_deg", float(_coerce_float(robot.get("theta_deg"), 0.0) or 0.0))
        target_name = str(row.get("target_name") or "").strip().lower()
        target_obj = ((state.get("objects") or {}).get(target_name) if target_name else None)
        if isinstance(target_obj, dict):
            row.setdefault("target_range_mm", float(_relative_pose(robot, target_obj)["range_mm"]))
    history.append(row)
    if len(history) > int(maxlen):
        del history[:-int(maxlen)]


def _heading_vector(theta_deg: float) -> tuple[float, float]:
    rad = math.radians(float(theta_deg))
    return math.cos(rad), math.sin(rad)


def _relative_pose(robot: dict, obj: dict) -> dict:
    dx = float(obj.get("x_mm", 0.0)) - float(robot.get("x_mm", 0.0))
    dy = float(obj.get("y_mm", 0.0)) - float(robot.get("y_mm", 0.0))
    hx, hy = _heading_vector(float(robot.get("theta_deg", 0.0)))
    forward_mm = dx * hx + dy * hy
    lateral_mm = -dx * hy + dy * hx
    bearing_deg = math.degrees(math.atan2(lateral_mm, forward_mm))
    return {
        "forward_mm": float(forward_mm),
        "lateral_mm": float(lateral_mm),
        "range_mm": float(math.hypot(dx, dy)),
        "bearing_deg": float(bearing_deg),
    }


def _world_robot_pose(world) -> dict:
    return {
        "x_mm": float(_coerce_float(getattr(world, "x", 0.0), 0.0) or 0.0),
        "y_mm": float(_coerce_float(getattr(world, "y", 0.0), 0.0) or 0.0),
        "theta_deg": float(_coerce_float(getattr(world, "theta", 0.0), 0.0) or 0.0),
        "lift_mm": float(_coerce_float(getattr(world, "lift_height", 0.0), 0.0) or 0.0),
        "z_mm": 0.0,
    }


def _sync_leia_pose(state: dict, world) -> None:
    robot = state["robot"]
    leia = state["leia"]
    leia["x_mm"] = float(robot["x_mm"])
    leia["y_mm"] = float(robot["y_mm"])
    leia["theta_deg"] = float(robot["theta_deg"])
    camera_height = _coerce_float(getattr(world, "height_mm", None), None)
    if camera_height is None:
        camera_height = float(DEFAULT_CAMERA_Z_MM) + float(robot["lift_mm"])
    leia["z_mm"] = float(camera_height)


def _sync_robot_pose(state: dict, world, *, reset_pose: bool = False) -> None:
    raw_pose = _world_robot_pose(world)
    state["raw_robot"].update(raw_pose)
    robot = state["robot"]
    if bool(reset_pose) or not bool(state.get("robot_pose_initialized", False)):
        robot.update(raw_pose)
        state["robot_pose_initialized"] = True
    else:
        robot["lift_mm"] = float(raw_pose["lift_mm"])
        robot["z_mm"] = float(raw_pose["z_mm"])
    _sync_leia_pose(state, world)


def _sync_active_target(state: dict, world) -> None:
    step_key = _normalize_step_key(getattr(world, "step_state", None))
    step_number = _step_number_for_key(world, step_key)
    object_name = _workspace_target_for_step(world, step_key, step_number)
    prev_step_name = _normalize_step_key(state.get("history_step_name"))
    if step_key != prev_step_name:
        state["history_step_name"] = step_key
        if step_key:
            state["history_step_seq"] = int(_coerce_int(state.get("history_step_seq"), 0) or 0) + 1
    active = state["active_target"]
    active["step_name"] = step_key
    active["step_number"] = step_number
    active["object_name"] = object_name
    active["label"] = _target_label(object_name)
    active["history_step_seq"] = int(_coerce_int(state.get("history_step_seq"), 0) or 0)


def _heading_toward_object_deg(robot: dict, obj: dict) -> float:
    dx = float(obj.get("x_mm", 0.0)) - float(robot.get("x_mm", 0.0))
    dy = float(obj.get("y_mm", 0.0)) - float(robot.get("y_mm", 0.0))
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(robot.get("theta_deg", 0.0))
    return float(math.degrees(math.atan2(dy, dx)))


def _maybe_align_heading_to_active_target(state: dict, *, reason: str) -> None:
    active = state.get("active_target") or {}
    active_name = active.get("object_name")
    obj = (state.get("objects") or {}).get(active_name)
    if not isinstance(obj, dict):
        return
    prev_target = state.get("heading_target_name")
    prev_step = state.get("heading_target_step_name")
    step_name = active.get("step_name")
    target_changed = active_name != prev_target or step_name != prev_step
    raw_theta = float(_coerce_float((state.get("raw_robot") or {}).get("theta_deg"), 0.0) or 0.0)
    should_snap = False
    if bool(target_changed) and prev_target is not None:
        should_snap = True
    elif not bool(state.get("heading_initialized", False)) and abs(raw_theta) < 1e-6:
        should_snap = True
    elif str(reason or "").strip().lower() in {"init", "reset_mission"} and abs(raw_theta) < 1e-6:
        should_snap = True
    if should_snap:
        state["robot"]["theta_deg"] = _norm_deg(_heading_toward_object_deg(state["robot"], obj))
        state["heading_initialized"] = True
    state["heading_target_name"] = active_name
    state["heading_target_step_name"] = step_name


def _sync_known_objects(state: dict, world) -> None:
    objects = state["objects"]
    supply = objects["brick_supply"]
    wall = objects["wall"]

    supply_count = _coerce_int(getattr(world, "brick_supply_height_bricks", None), None)
    if supply_count is not None:
        supply["count"] = int(supply_count)
    supply_height_mm = _coerce_float(getattr(world, "brick_supply_height_mm", None), None)
    if supply_height_mm is None and supply_count is not None:
        supply_height_mm = float(supply_count) * float(DEFAULT_STACK_HEIGHT_MM)
    supply["height_mm"] = None if supply_height_mm is None else float(max(0.0, supply_height_mm))

    brick = getattr(world, "brick", None) or {}
    held = bool(brick.get("held", False))
    state["held_brick"]["held"] = held
    state["last_visible_brick"] = {
        "visible": bool(brick.get("visible", False)),
        "dist_mm": _coerce_float(brick.get("dist"), None),
        "x_axis_mm": _coerce_float(brick.get("x_axis", brick.get("offset_x")), None),
        "y_axis_mm": _coerce_float(brick.get("y_axis", brick.get("offset_y")), None),
        "confidence": _coerce_float(brick.get("confidence"), None),
        "angle_deg": _coerce_float(brick.get("angle"), None),
    }

    wall_state = getattr(world, "wall", None) or {}
    origin = wall_state.get("origin") if isinstance(wall_state, dict) else None
    if isinstance(origin, dict):
        x_mm = _coerce_float(origin.get("x"), None)
        y_mm = _coerce_float(origin.get("y"), None)
        if x_mm is not None and y_mm is not None:
            wall["observed_world_x_mm"] = float(x_mm)
            wall["observed_world_y_mm"] = float(y_mm)
            wall["observed_world_theta_deg"] = float(_coerce_float(origin.get("theta"), wall.get("theta_deg", 180.0)) or wall.get("theta_deg", 180.0))
            wall["visible"] = bool(wall_state.get("last_seen"))
            wall["valid"] = bool(wall_state.get("valid", True))
            wall["source"] = wall_state.get("source") or wall.get("source")
            wall["last_seen_ts"] = _coerce_float(wall_state.get("last_seen"), wall.get("last_seen_ts"))
    else:
        wall["valid"] = bool(wall_state.get("valid", False))
    
    wall_count = _coerce_int(getattr(world, "wall_height_bricks", None), None)
    if wall_count is not None:
        wall["count"] = int(wall_count)
    wall_height_mm = _coerce_float(getattr(world, "wall_height_mm", None), None)
    if wall_height_mm is None and wall_count is not None:
        wall_height_mm = float(wall_count) * float(DEFAULT_STACK_HEIGHT_MM)
    wall["height_mm"] = None if wall_height_mm is None else float(max(0.0, wall_height_mm))
    state["micro_adjust_phase"] = bool(getattr(world, "_xyz_micro_adjust_phase", False))


def _mark_active_target_visible(state: dict) -> None:
    active_name = (state.get("active_target") or {}).get("object_name")
    if active_name not in {"wall", "brick_supply"}:
        return
    last_visible = state.get("last_visible_brick") or {}
    if not bool(last_visible.get("visible", False)):
        return
    obj = (state.get("objects") or {}).get(active_name)
    if not isinstance(obj, dict):
        return
    obj["visible"] = True
    obj["last_seen_ts"] = time.time()


def _object_footprint_mm(obj: dict) -> tuple[float, float]:
    footprint = _coerce_float(obj.get("render_footprint_mm"), None)
    if footprint is None or float(footprint) <= 0.0:
        footprint = float(DEFAULT_STACK_RENDER_FOOTPRINT_MM)
    return float(footprint), float(footprint)


def _point_to_object_local(x_mm: float, y_mm: float, obj: dict) -> tuple[float, float]:
    dx = float(x_mm) - float(obj.get("x_mm", 0.0))
    dy = float(y_mm) - float(obj.get("y_mm", 0.0))
    theta = math.radians(float(obj.get("theta_deg", 0.0)))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t


def _point_from_object_local(local_x: float, local_y: float, obj: dict) -> tuple[float, float]:
    theta = math.radians(float(obj.get("theta_deg", 0.0)))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = float(local_x) * cos_t - float(local_y) * sin_t
    dy = float(local_x) * sin_t + float(local_y) * cos_t
    return float(obj.get("x_mm", 0.0)) + dx, float(obj.get("y_mm", 0.0)) + dy


def _point_inside_object(x_mm: float, y_mm: float, obj: dict, *, margin_mm: float = 0.0) -> bool:
    length_mm, width_mm = _object_footprint_mm(obj)
    local_x, local_y = _point_to_object_local(x_mm, y_mm, obj)
    return abs(local_x) <= (length_mm * 0.5 + float(margin_mm)) and abs(local_y) <= (width_mm * 0.5 + float(margin_mm))


def _clamp_robot_clear_of_objects(state: dict) -> None:
    robot = state["robot"]
    for _ in range(2):
        moved = False
        for obj in (state.get("objects") or {}).values():
            if not isinstance(obj, dict):
                continue
            if not _point_inside_object(robot["x_mm"], robot["y_mm"], obj, margin_mm=ROBOT_OBJECT_MARGIN_MM):
                continue
            length_mm, width_mm = _object_footprint_mm(obj)
            local_x, local_y = _point_to_object_local(robot["x_mm"], robot["y_mm"], obj)
            half_l = length_mm * 0.5 + float(ROBOT_OBJECT_MARGIN_MM)
            half_w = width_mm * 0.5 + float(ROBOT_OBJECT_MARGIN_MM)
            gap_x = half_l - abs(local_x)
            gap_y = half_w - abs(local_y)
            if gap_x <= gap_y:
                sign = 1.0 if local_x >= 0.0 else -1.0
                if abs(local_x) < 1e-6:
                    sign = 1.0
                local_x = sign * half_l
            else:
                sign = 1.0 if local_y >= 0.0 else -1.0
                if abs(local_y) < 1e-6:
                    sign = 1.0
                local_y = sign * half_w
            robot["x_mm"], robot["y_mm"] = _point_from_object_local(local_x, local_y, obj)
            moved = True
        if not moved:
            break


def _apply_motion_delta_to_robot(robot: dict, event, delta) -> None:
    if event is None or delta is None:
        return
    action_type = getattr(event, "action_type", None)
    dist_mm = float(_coerce_float(getattr(delta, "dist_mm", None), 0.0) or 0.0)
    rot_deg = float(_coerce_float(getattr(delta, "rot_deg", None), 0.0) or 0.0)
    theta_deg = float(robot.get("theta_deg", 0.0))
    rad = math.radians(theta_deg)
    if action_type == "forward":
        robot["x_mm"] = float(robot["x_mm"]) + dist_mm * math.cos(rad)
        robot["y_mm"] = float(robot["y_mm"]) + dist_mm * math.sin(rad)
    elif action_type == "backward":
        robot["x_mm"] = float(robot["x_mm"]) - dist_mm * math.cos(rad)
        robot["y_mm"] = float(robot["y_mm"]) - dist_mm * math.sin(rad)
    elif action_type == "left_turn":
        robot["x_mm"] = float(robot["x_mm"]) + dist_mm * math.cos(rad)
        robot["y_mm"] = float(robot["y_mm"]) + dist_mm * math.sin(rad)
        robot["theta_deg"] = _norm_deg(theta_deg + rot_deg)
    elif action_type == "right_turn":
        robot["x_mm"] = float(robot["x_mm"]) + dist_mm * math.cos(rad)
        robot["y_mm"] = float(robot["y_mm"]) + dist_mm * math.sin(rad)
        robot["theta_deg"] = _norm_deg(theta_deg - rot_deg)


def _observation_bearing_deg(observation: dict | None) -> float:
    data = observation or {}
    dist_mm = _coerce_float(data.get("dist_mm"), None)
    lateral_mm = _coerce_float(data.get("x_axis_mm"), 0.0) or 0.0
    if dist_mm is None or abs(float(dist_mm)) < 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(lateral_mm, dist_mm)))


def _reconcile_robot_pose_from_fixed_object(
    state: dict,
    object_name: str,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
) -> None:
    obj = (state.get("objects") or {}).get(str(object_name or "").strip().lower())
    robot = state.get("robot") or {}
    if not isinstance(obj, dict) or not isinstance(robot, dict):
        return
    heading_deg = float(robot.get("theta_deg", 0.0)) + float(_coerce_float(bearing_deg, 0.0) or 0.0)
    hx, hy = _heading_vector(heading_deg)
    distance_val = max(0.0, float(_coerce_float(distance_mm, 0.0) or 0.0))
    robot["x_mm"] = float(obj.get("x_mm", 0.0)) - distance_val * hx
    robot["y_mm"] = float(obj.get("y_mm", 0.0)) - distance_val * hy


def _apply_vision_reconciliation(state: dict, world, *, reason: str) -> None:
    if str(reason or "").strip().lower() != "vision":
        return
    active_name = (state.get("active_target") or {}).get("object_name")
    if active_name not in {"wall", "brick_supply"}:
        return
    last_visible = state.get("last_visible_brick") or {}
    if bool(last_visible.get("visible", False)) and last_visible.get("dist_mm") is not None:
        _reconcile_robot_pose_from_fixed_object(
            state,
            active_name,
            distance_mm=float(last_visible["dist_mm"]),
            bearing_deg=_observation_bearing_deg(last_visible),
        )
        return
    if active_name != "wall":
        return
    wall = ((state.get("objects") or {}).get("wall") or {})
    raw_x = _coerce_float(wall.get("observed_world_x_mm"), None)
    raw_y = _coerce_float(wall.get("observed_world_y_mm"), None)
    if raw_x is None or raw_y is None or not bool(wall.get("valid", False)):
        return
    raw_robot = state.get("raw_robot") or {}
    robot = state.get("robot") or {}
    robot["x_mm"] = float(_coerce_float(raw_robot.get("x_mm"), robot.get("x_mm", 0.0)) or robot.get("x_mm", 0.0)) + (
        float(wall.get("x_mm", 0.0)) - float(raw_x)
    )
    robot["y_mm"] = float(_coerce_float(raw_robot.get("y_mm"), robot.get("y_mm", 0.0)) or robot.get("y_mm", 0.0)) + (
        float(wall.get("y_mm", 0.0)) - float(raw_y)
    )


def sync_from_world(world, *, reason: str = "sync", render: bool = True) -> dict:
    state = ensure_workspace(world)
    _sync_robot_pose(state, world, reset_pose=str(reason or "").strip().lower() in {"init", "reset_mission"})
    _sync_known_objects(state, world)
    _sync_active_target(state, world)
    _maybe_align_heading_to_active_target(state, reason=reason)
    _mark_active_target_visible(state)
    _apply_vision_reconciliation(state, world, reason=reason)
    _clamp_robot_clear_of_objects(state)
    _sync_leia_pose(state, world)
    state["updated_at"] = time.time()
    sync_entry: dict = {"type": "sync", "reason": str(reason), "ts": state["updated_at"]}
    last_action_line = getattr(world, "_last_action_line", None)
    if last_action_line:
        sync_entry["note_after"] = str(last_action_line)
    _append_history(state, sync_entry)
    if bool(render):
        _write_live_assets(state)
    return state


def update_from_motion(world, *, event=None, delta=None, render: bool = True) -> dict:
    state = ensure_workspace(world)
    had_pose = bool(state.get("robot_pose_initialized", False))
    _sync_robot_pose(state, world, reset_pose=not had_pose)
    if had_pose:
        _apply_motion_delta_to_robot(state["robot"], event, delta)
    _sync_known_objects(state, world)
    _sync_active_target(state, world)
    _maybe_align_heading_to_active_target(state, reason="motion")
    _mark_active_target_visible(state)
    _clamp_robot_clear_of_objects(state)
    _sync_leia_pose(state, world)
    state["updated_at"] = time.time()
    entry = {"type": "motion", "ts": time.time()}
    if event is not None:
        entry["action_type"] = getattr(event, "action_type", None)
        entry["duration_ms"] = _coerce_int(getattr(event, "duration_ms", None), None)
        entry["speed_score"] = _coerce_int(getattr(event, "speed_score", None), None)
    if delta is not None:
        entry["dist_mm"] = _coerce_float(getattr(delta, "dist_mm", None), None)
        entry["rot_deg"] = _coerce_float(getattr(delta, "rot_deg", None), None)
        entry["lift_mm"] = _coerce_float(getattr(delta, "lift_mm", None), None)
    _append_history(state, entry)
    if bool(render):
        _write_live_assets(state)
    return state


def observe_object(
    world,
    object_name: str,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
    count: int | None = None,
    confidence: float | None = None,
    theta_deg: float | None = None,
    source: str = "manual_observation",
    render: bool = True,
) -> dict:
    state = sync_from_world(world, reason=f"observe_{object_name}", render=False)
    objects = state["objects"]
    key = str(object_name or "").strip().lower()
    if key not in objects:
        raise KeyError(f"Unknown workspace object '{object_name}'")
    obj = objects[key]
    distance_val = max(0.0, float(_coerce_float(distance_mm, 0.0) or 0.0))
    if theta_deg is not None:
        obj["theta_deg"] = float(theta_deg)
    if count is not None and key == "brick_supply":
        obj["count"] = int(count)
    obj["confidence"] = _coerce_float(confidence, obj.get("confidence"))
    obj["visible"] = True
    obj["source"] = str(source)
    obj["last_seen_ts"] = time.time()
    _reconcile_robot_pose_from_fixed_object(
        state,
        key,
        distance_mm=float(distance_val),
        bearing_deg=float(_coerce_float(bearing_deg, 0.0) or 0.0),
    )
    _clamp_robot_clear_of_objects(state)
    _sync_leia_pose(state, world)
    state["updated_at"] = time.time()
    _append_history(
        state,
        {
            "type": "observation",
            "object": key,
            "distance_mm": float(distance_val),
            "bearing_deg": float(_coerce_float(bearing_deg, 0.0) or 0.0),
            "confidence": obj.get("confidence"),
            "ts": obj["last_seen_ts"],
        },
    )
    if bool(render):
        _write_live_assets(state)
    return state


def observe_brick_supply(
    world,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
    count: int | None = None,
    confidence: float | None = None,
    source: str = "brick_supply_observation",
    render: bool = True,
) -> dict:
    return observe_object(
        world,
        "brick_supply",
        distance_mm=distance_mm,
        bearing_deg=bearing_deg,
        count=count,
        confidence=confidence,
        source=source,
        render=render,
    )


def observe_wall(
    world,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
    theta_deg: float | None = None,
    confidence: float | None = None,
    source: str = "wall_observation",
    render: bool = True,
) -> dict:
    return observe_object(
        world,
        "wall",
        distance_mm=distance_mm,
        bearing_deg=bearing_deg,
        theta_deg=theta_deg,
        confidence=confidence,
        source=source,
        render=render,
    )


def set_brick_supply_count(world, count: int | None, *, source: str = "manual_count", render: bool = True) -> dict:
    state = sync_from_world(world, reason="set_brick_supply_count", render=False)
    supply = state["objects"]["brick_supply"]
    supply["count"] = _coerce_int(count, None)
    supply["source"] = str(source)
    state["updated_at"] = time.time()
    _append_history(
        state,
        {"type": "brick_supply_count", "count": supply["count"], "source": str(source), "ts": state["updated_at"]},
    )
    if bool(render):
        _write_live_assets(state)
    return state


def set_holding_brick(world, held: bool, *, source: str = "manual_hold_state", render: bool = True) -> dict:
    state = sync_from_world(world, reason="set_holding_brick", render=False)
    state["held_brick"]["held"] = bool(held)
    brick = getattr(world, "brick", None)
    if isinstance(brick, dict):
        brick["held"] = bool(held)
    state["updated_at"] = time.time()
    _append_history(
        state,
        {"type": "held_brick", "held": bool(held), "source": str(source), "ts": state["updated_at"]},
    )
    if bool(render):
        _write_live_assets(state)
    return state


def reconcile_object_distance(
    world,
    object_name: str,
    observed_dist_mm: float,
    *,
    bearing_deg: float | None = None,
    source: str = "distance_reconciliation",
    render: bool = True,
) -> dict:
    state = sync_from_world(world, reason=f"reconcile_{object_name}", render=False)
    key = str(object_name or "").strip().lower()
    obj = state["objects"].get(key)
    if not isinstance(obj, dict):
        raise KeyError(f"Unknown workspace object '{object_name}'")
    if bearing_deg is None:
        bearing_deg = _relative_pose(state["robot"], obj)["bearing_deg"]
    return observe_object(
        world,
        key,
        distance_mm=observed_dist_mm,
        bearing_deg=float(bearing_deg),
        count=obj.get("count"),
        confidence=obj.get("confidence"),
        theta_deg=obj.get("theta_deg"),
        source=source,
        render=render,
    )


def relative_object_pose(world, object_name: str) -> dict | None:
    state = ensure_workspace(world)
    obj = state["objects"].get(str(object_name or "").strip().lower())
    if not isinstance(obj, dict):
        return None
    return _relative_pose(state["robot"], obj)


def _plan_reverse_then_turn_from_state(
    state: dict,
    *,
    turn_cmd: str = "l",
    reverse_step_mm: float = 30.0,
    turn_when_wall_within_mm: float = 45.0,
    max_reverse_acts: int = 8,
) -> dict:
    wall = (state.get("objects") or {}).get("wall")
    robot = state.get("robot") or {}
    if not isinstance(wall, dict) or not isinstance(robot, dict):
        return {"ok": False, "reason": "wall_unknown", "actions": []}
    wall_pose = _relative_pose(robot, wall)
    behind_mm = max(0.0, -float(wall_pose.get("forward_mm", 0.0)))
    if behind_mm <= 0.0:
        return {
            "ok": True,
            "reason": "wall_not_behind",
            "wall_behind_mm": 0.0,
            "reverse_acts": 0,
            "actions": [{"cmd": str(turn_cmd), "reason": "turn_now"}],
        }
    reverse_gap_mm = max(0.0, behind_mm - float(turn_when_wall_within_mm))
    reverse_step = max(1.0, float(reverse_step_mm))
    reverse_acts = min(int(max_reverse_acts), int(math.ceil(reverse_gap_mm / reverse_step)))
    actions = [{"cmd": "b", "distance_mm": float(reverse_step), "reason": "close_wall_gap"} for _ in range(reverse_acts)]
    actions.append({"cmd": str(turn_cmd), "reason": "begin_turn_after_reverse"})
    return {
        "ok": True,
        "reason": "planned",
        "wall_behind_mm": float(behind_mm),
        "target_wall_distance_mm": float(turn_when_wall_within_mm),
        "reverse_gap_mm": float(reverse_gap_mm),
        "reverse_acts": int(reverse_acts),
        "actions": actions,
    }


def plan_reverse_then_turn_for_wall(
    world,
    *,
    turn_cmd: str = "l",
    reverse_step_mm: float = 30.0,
    turn_when_wall_within_mm: float = 45.0,
    max_reverse_acts: int = 8,
    render: bool = True,
) -> dict:
    state = sync_from_world(world, reason="plan_reverse_then_turn", render=render)
    return _plan_reverse_then_turn_from_state(
        state,
        turn_cmd=turn_cmd,
        reverse_step_mm=reverse_step_mm,
        turn_when_wall_within_mm=turn_when_wall_within_mm,
        max_reverse_acts=max_reverse_acts,
    )


def _viewport_points(state: dict) -> list[tuple[float, float]]:
    points = []
    robot = state["robot"]
    points.append((float(robot["x_mm"]), float(robot["y_mm"])))
    for entry in _current_step_motion_history(state):
        points.append((float(entry["x_mm"]), float(entry["y_mm"])))
    active = state.get("active_target") or {}
    active_name = str(active.get("object_name") or "").strip().lower()
    object_names = ["wall", "brick_supply"]
    if active_name in {"wall", "brick_supply"}:
        object_names = [active_name]
    for obj_name in object_names:
        obj = (state.get("objects") or {}).get(obj_name) or {}
        points.append((float(obj.get("x_mm", 0.0)), float(obj.get("y_mm", 0.0))))
        if obj.get("name") == "wall":
            points.extend(_wall_render_points(obj))
    return points


def _view_coords(x_mm: float, y_mm: float) -> tuple[float, float]:
    # Rotate the rendered workspace 90 degrees clockwise without changing
    # the underlying world-frame math.
    return float(y_mm), -float(x_mm)


def _build_viewbox(state: dict) -> tuple[float, float, float, float]:
    points = [_view_coords(x_mm, y_mm) for x_mm, y_mm in _viewport_points(state)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs) - float(BIRDSEYE_VIEW_MARGIN_MM)
    max_x = max(xs) + float(BIRDSEYE_VIEW_MARGIN_MM)
    min_y = min(ys) - float(BIRDSEYE_VIEW_MARGIN_MM)
    max_y = max(ys) + float(BIRDSEYE_VIEW_MARGIN_MM)
    if (max_x - min_x) < float(BIRDSEYE_MIN_SPAN_X_MM):
        mid_x = (max_x + min_x) * 0.5
        min_x = mid_x - float(BIRDSEYE_MIN_SPAN_X_MM) * 0.5
        max_x = mid_x + float(BIRDSEYE_MIN_SPAN_X_MM) * 0.5
    if (max_y - min_y) < float(BIRDSEYE_MIN_SPAN_Y_MM):
        mid_y = (max_y + min_y) * 0.5
        min_y = mid_y - float(BIRDSEYE_MIN_SPAN_Y_MM) * 0.5
        max_y = mid_y + float(BIRDSEYE_MIN_SPAN_Y_MM) * 0.5
    return min_x, max_x, min_y, max_y


def _project_fn(state: dict):
    min_x, max_x, min_y, max_y = _build_viewbox(state)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min((SVG_WIDTH - 2 * BIRD_SVG_PADDING) / span_x, (BIRD_SVG_HEIGHT - 2 * BIRD_SVG_PADDING) / span_y)

    def view_project(view_x_mm: float, view_y_mm: float) -> tuple[float, float]:
        sx = BIRD_SVG_PADDING + (float(view_x_mm) - min_x) * scale
        sy = BIRD_SVG_HEIGHT - BIRD_SVG_PADDING - (float(view_y_mm) - min_y) * scale
        return sx, sy

    def project(x_mm: float, y_mm: float) -> tuple[float, float]:
        view_x_mm, view_y_mm = _view_coords(x_mm, y_mm)
        return view_project(view_x_mm, view_y_mm)

    return project, view_project, scale


def _polygon_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _screen_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _anchor_stack_screen_points(
    points: list[tuple[float, float]],
    *,
    center_y_px: float = BIRD_STACK_CENTER_Y_PX,
) -> list[tuple[float, float]]:
    if not points:
        return []
    _, _, min_y, max_y = _screen_bounds(points)
    current_center_y = (float(min_y) + float(max_y)) * 0.5
    shift_y = float(center_y_px) - float(current_center_y)
    return [(float(x), float(y) + float(shift_y)) for x, y in points]


def _bird_distance_track_ratio(dist_mm: float | None) -> float:
    distance_val = _coerce_float(dist_mm, None)
    if distance_val is None:
        distance_val = _bird_dist_track_far_mm()
    near_mm = _bird_dist_track_near_mm()
    far_mm = max(float(near_mm) + 1.0, _bird_dist_track_far_mm())
    return max(0.0, min(1.0, (float(distance_val) - float(near_mm)) / (float(far_mm) - float(near_mm))))


def _bird_dist_track_near_mm() -> float:
    return float(BIRD_DIST_TARGET_MM) - float(BIRD_DIST_TRACK_HALF_RANGE_MM)


def _bird_dist_track_far_mm() -> float:
    return float(BIRD_DIST_TARGET_MM) + float(BIRD_DIST_TRACK_HALF_RANGE_MM)


def _rotated_rect(center_x: float, center_y: float, theta_deg: float, length_mm: float, width_mm: float) -> list[tuple[float, float]]:
    hx, hy = _heading_vector(theta_deg)
    px, py = -hy, hx
    half_l = float(length_mm) * 0.5
    half_w = float(width_mm) * 0.5
    return [
        (center_x + hx * half_l + px * half_w, center_y + hy * half_l + py * half_w),
        (center_x + hx * half_l - px * half_w, center_y + hy * half_l - py * half_w),
        (center_x - hx * half_l - px * half_w, center_y - hy * half_l - py * half_w),
        (center_x - hx * half_l + px * half_w, center_y - hy * half_l + py * half_w),
    ]


def _wall_render_footprint_mm(wall: dict) -> float:
    if not isinstance(wall, dict):
        return float(DEFAULT_STACK_RENDER_FOOTPRINT_MM)
    footprint_mm = _coerce_float(wall.get("render_footprint_mm"), None)
    if footprint_mm is None or float(footprint_mm) <= 0.0:
        footprint_mm = float(DEFAULT_STACK_RENDER_FOOTPRINT_MM)
    return float(footprint_mm)


def _wall_render_points(wall: dict) -> list[tuple[float, float]]:
    if not isinstance(wall, dict):
        wall = {}
    footprint_mm = _wall_render_footprint_mm(wall)
    return _rotated_rect(
        float(wall.get("x_mm", 0.0)),
        float(wall.get("y_mm", 0.0)),
        float(wall.get("theta_deg", DEFAULT_WALL_POS_MM["theta_deg"])),
        footprint_mm,
        footprint_mm,
    )


def _robot_points(robot: dict) -> list[tuple[float, float]]:
    x_mm = float(robot.get("x_mm", 0.0))
    y_mm = float(robot.get("y_mm", 0.0))
    theta = float(robot.get("theta_deg", 0.0))
    hx, hy = _heading_vector(theta)
    px, py = -hy, hx
    nose = (x_mm, y_mm)
    back_left = (
        x_mm - hx * ROBOT_NOSE_TO_TAIL_MM + px * ROBOT_HALF_WIDTH_MM,
        y_mm - hy * ROBOT_NOSE_TO_TAIL_MM + py * ROBOT_HALF_WIDTH_MM,
    )
    back_center = (x_mm - hx * ROBOT_TAIL_CENTER_MM, y_mm - hy * ROBOT_TAIL_CENTER_MM)
    back_right = (
        x_mm - hx * ROBOT_NOSE_TO_TAIL_MM - px * ROBOT_HALF_WIDTH_MM,
        y_mm - hy * ROBOT_NOSE_TO_TAIL_MM - py * ROBOT_HALF_WIDTH_MM,
    )
    return [nose, back_left, back_center, back_right]


def _grid_lines(min_v: float, max_v: float, step: float = 50.0) -> list[float]:
    start = math.floor(min_v / step) * step
    end = math.ceil(max_v / step) * step
    values = []
    cur = start
    while cur <= end + 0.001:
        values.append(float(cur))
        cur += float(step)
    return values


def _current_step_motion_history(state: dict) -> list[dict]:
    history = state.get("history") or []
    active = state.get("active_target") or {}
    current_seq = _coerce_int(active.get("history_step_seq"), None)
    current_step = _normalize_step_key(active.get("step_name"))
    filtered = []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("type") != "motion":
            continue
        action_type = str(entry.get("action_type") or "").strip().lower()
        if action_type in {"mast_up", "mast_down"}:
            continue
        x_mm = _coerce_float(entry.get("x_mm"), None)
        y_mm = _coerce_float(entry.get("y_mm"), None)
        if x_mm is None or y_mm is None:
            continue
        entry_seq = _coerce_int(entry.get("step_seq"), None)
        entry_step = _normalize_step_key(entry.get("step_name"))
        if current_seq is not None and entry_seq is not None:
            if int(entry_seq) != int(current_seq):
                continue
        elif current_step and entry_step != current_step:
            continue
        filtered.append(entry)
    return filtered


def _step_history_trend(previous_entry: dict | None, entry: dict) -> str:
    if not isinstance(previous_entry, dict):
        return "unknown"
    if str(previous_entry.get("target_name") or "") != str(entry.get("target_name") or ""):
        return "unknown"
    prev_range = _coerce_float(previous_entry.get("target_range_mm"), None)
    current_range = _coerce_float(entry.get("target_range_mm"), None)
    if prev_range is None or current_range is None:
        return "unknown"
    delta_mm = float(current_range) - float(prev_range)
    if delta_mm <= -0.5:
        return "closer"
    if delta_mm >= 0.5:
        return "further"
    return "neutral"


def _step_history_color(trend: str) -> str:
    if trend == "closer":
        return STEP_HISTORY_COLOR_CLOSER
    if trend == "further":
        return STEP_HISTORY_COLOR_FURTHER
    if trend == "neutral":
        return STEP_HISTORY_COLOR_NEUTRAL
    return STEP_HISTORY_COLOR_UNKNOWN


def _bird_history_fill_color(index: int, history_count: int) -> str:
    is_most_recent = int(index) >= max(0, int(history_count) - 1)
    if is_most_recent:
        return str(BIRD_HISTORY_COLOR_NEWEST)
    return str(BIRD_HISTORY_COLOR_OLDER)


def _current_step_observation_history(state: dict) -> list[dict]:
    history = state.get("history") or []
    active = state.get("active_target") or {}
    current_seq = _coerce_int(active.get("history_step_seq"), None)
    current_step = _normalize_step_key(active.get("step_name"))
    filtered = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        action_type = str(entry.get("action_type") or "").strip().lower()
        if action_type in {"mast_up", "mast_down"}:
            continue
        entry_type = str(entry.get("type") or "").strip().lower()
        if entry_type not in {"sync", "observation"}:
            continue
        if entry_type == "sync" and not bool(entry.get("target_visible", False)):
            continue
        entry_seq = _coerce_int(entry.get("step_seq"), None)
        entry_step = _normalize_step_key(entry.get("step_name"))
        if current_seq is not None and entry_seq is not None:
            if int(entry_seq) != int(current_seq):
                continue
        elif current_step and entry_step != current_step:
            continue
        dist_mm = _coerce_float(entry.get("dist_mm", entry.get("target_range_mm")), None)
        x_axis_mm = _coerce_float(entry.get("x_axis_mm"), None)
        if dist_mm is None or x_axis_mm is None:
            continue
        filtered.append(entry)
    return filtered


def _current_step_bird_history(state: dict) -> list[dict]:
    observed = _current_step_observation_history(state)
    if bool((state or {}).get("run_replay")):
        return observed
    return observed[-int(BIRD_HISTORY_MAX_POINTS):]


def _current_step_mast_history(state: dict) -> list[dict]:
    history = state.get("history") or []
    active = state.get("active_target") or {}
    current_seq = _coerce_int(active.get("history_step_seq"), None)
    current_step = _normalize_step_key(active.get("step_name"))
    filtered = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        entry_seq = _coerce_int(entry.get("step_seq"), None)
        entry_step = _normalize_step_key(entry.get("step_name"))
        if current_seq is not None and entry_seq is not None:
            if int(entry_seq) != int(current_seq):
                continue
        elif current_step and entry_step != current_step:
            continue
        action_type = str(entry.get("action_type") or "").strip().lower()
        y_axis_mm = _coerce_float(entry.get("y_axis_mm"), None)
        camera_height_mm = _coerce_float(entry.get("camera_height_mm"), None)
        current_lift_mm = _coerce_float(entry.get("current_lift_mm"), None)
        if camera_height_mm is None and current_lift_mm is None:
            continue
        if action_type not in {"mast_up", "mast_down"} and y_axis_mm is None:
            continue
        filtered.append(entry)
    return filtered


def _all_motion_history(state: dict) -> list[dict]:
    history = state.get("history") or []
    filtered = []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("type") != "motion":
            continue
        action_type = str(entry.get("action_type") or "").strip().lower()
        if action_type in {"mast_up", "mast_down"}:
            continue
        x_mm = _coerce_float(entry.get("x_mm"), None)
        y_mm = _coerce_float(entry.get("y_mm"), None)
        if x_mm is None or y_mm is None:
            continue
        filtered.append(entry)
    return filtered


def _all_mast_history(state: dict) -> list[dict]:
    history = state.get("history") or []
    filtered = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        action_type = str(entry.get("action_type") or "").strip().lower()
        y_axis_mm = _coerce_float(entry.get("y_axis_mm"), None)
        camera_height_mm = _coerce_float(entry.get("camera_height_mm"), None)
        current_lift_mm = _coerce_float(entry.get("current_lift_mm"), None)
        if camera_height_mm is None and current_lift_mm is None:
            continue
        if action_type not in {"mast_up", "mast_down"} and y_axis_mm is None:
            continue
        filtered.append(entry)
    return filtered


def _mast_history_trend(previous_entry: dict | None, entry: dict) -> str:
    if not isinstance(previous_entry, dict):
        return "unknown"
    prev_abs = _coerce_float(previous_entry.get("y_axis_abs_mm"), None)
    current_abs = _coerce_float(entry.get("y_axis_abs_mm"), None)
    if prev_abs is None or current_abs is None:
        return "unknown"
    delta_mm = float(current_abs) - float(prev_abs)
    if delta_mm <= -0.5:
        return "closer"
    if delta_mm >= 0.5:
        return "further"
    return "neutral"


def _mast_history_line_style(segment_from_end: int) -> dict:
    if int(segment_from_end) <= 0:
        return dict(MAST_HISTORY_LINE_CURRENT)
    if int(segment_from_end) == 1:
        return dict(MAST_HISTORY_LINE_PREVIOUS)
    if int(segment_from_end) == 2:
        return dict(MAST_HISTORY_LINE_PREVIOUS_2)
    return dict(MAST_HISTORY_LINE_OLDER)


def _mast_history_screen_points(
    y_values: list[float],
    *,
    history_left_x: float,
    history_right_x: float,
    project_y,
) -> list[tuple[float, float]]:
    if not y_values:
        return []
    if len(y_values) == 1:
        return [(float(history_left_x), float(project_y(y_values[0])))]
    span_x = max(1.0, float(history_right_x) - float(history_left_x))
    step_x = float(span_x) / float(len(y_values) - 1)
    return [
        (
            float(history_right_x) - float(step_x) * float(idx),
            float(project_y(value)),
        )
        for idx, value in enumerate(y_values)
    ]


def _autoscaled_axis_extent_mm(
    values,
    *,
    default_extent_mm: float,
    expanded_extent_mm: float,
    expand_threshold_mm: float,
) -> float:
    default_extent = max(1.0, float(_coerce_float(default_extent_mm, 10.0) or 10.0))
    expanded_extent = max(default_extent, float(_coerce_float(expanded_extent_mm, default_extent) or default_extent))
    expand_threshold = max(0.0, float(_coerce_float(expand_threshold_mm, default_extent) or default_extent))
    for value in values or ():
        numeric = _coerce_float(value, None)
        if numeric is None or not math.isfinite(float(numeric)):
            continue
        if abs(float(numeric)) > float(expand_threshold):
            return float(expanded_extent)
    return float(default_extent)


def _bird_x_axis_track_extent_mm(state: dict) -> float:
    bird_history = _current_step_bird_history(state)
    values = [
        float(_coerce_float(entry.get("x_axis_mm"), None))
        for entry in bird_history
        if _coerce_float(entry.get("x_axis_mm"), None) is not None
    ]
    current_x_axis_mm = _coerce_float((state.get("last_visible_brick") or {}).get("x_axis_mm"), None)
    if current_x_axis_mm is not None:
        values.append(float(current_x_axis_mm))
    return _autoscaled_axis_extent_mm(
        values,
        default_extent_mm=BIRD_X_AXIS_TRACK_DEFAULT_EXTENT_MM,
        expanded_extent_mm=BIRD_X_AXIS_TRACK_EXPANDED_EXTENT_MM,
        expand_threshold_mm=BIRD_X_AXIS_TRACK_EXPAND_THRESHOLD_MM,
    )


def _mast_y_axis_extent_mm(state: dict) -> float:
    mast_history = _current_step_mast_history(state)
    values = [
        float(_coerce_float(entry.get("y_axis_mm"), None))
        for entry in mast_history
        if _coerce_float(entry.get("y_axis_mm"), None) is not None
    ]
    current_y_axis_mm = _coerce_float((state.get("last_visible_brick") or {}).get("y_axis_mm"), None)
    if current_y_axis_mm is not None:
        values.append(float(current_y_axis_mm))
    return _autoscaled_axis_extent_mm(
        values,
        default_extent_mm=MAST_Y_AXIS_DEFAULT_EXTENT_MM,
        expanded_extent_mm=MAST_Y_AXIS_EXPANDED_EXTENT_MM,
        expand_threshold_mm=MAST_Y_AXIS_EXPAND_THRESHOLD_MM,
    )


def _stack_visual_style(stack_name: str | None) -> dict:
    key = str(stack_name or "").strip().lower()
    if key == "wall":
        return {
            "fill": "#cc3a2b",
            "stroke": "#7d241b",
            "label": "Wall",
        }
    return {
        "fill": "#1f4b8f",
        "stroke": "#122d57",
        "label": "Supply",
    }


def _stack_card_svg(
    *,
    center_x: float,
    center_y: float,
    size_px: float,
    label: str,
    fill: str,
    stroke: str,
    count_text: str | None = None,
    label_gap_px: float = 14.0,
    corner_radius_px: float = 12.0,
    count_font_size: float = 36.0,
    count_y_nudge: float = 16.0,
) -> list[str]:
    top = float(center_y) - (float(size_px) * 0.5)
    left = float(center_x) - (float(size_px) * 0.5)
    svg_parts = [
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{float(size_px):.1f}" height="{float(size_px):.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3" rx="{float(corner_radius_px):.1f}" />',
        f'<text x="{float(center_x):.1f}" y="{top - float(label_gap_px):.1f}" text-anchor="middle" font-size="28" font-weight="600" fill="#1d2830">{html.escape(str(label))}</text>',
    ]
    if count_text not in (None, ""):
        svg_parts.append(
            f'<text x="{float(center_x):.1f}" y="{float(center_y) + float(count_y_nudge):.1f}" text-anchor="middle" font-size="{float(count_font_size):.1f}" font-weight="700" fill="#ffffff">{html.escape(str(count_text))}</text>'
        )
    return svg_parts


def render_workspace_svg(state: dict | None) -> str:
    snapshot = _normalized_render_snapshot(state)
    project, view_project, scale = _project_fn(snapshot)
    min_x, max_x, min_y, max_y = _build_viewbox(snapshot)
    active_target = snapshot.get("active_target") or {}
    active_name = active_target.get("object_name")
    bird_history = _current_step_bird_history(snapshot)
    bird_x_axis_extent_mm = _bird_x_axis_track_extent_mm(snapshot)
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{BIRD_SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {BIRD_SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        f'<rect x="22" y="12" width="936" height="{BIRD_SVG_HEIGHT - 24}" rx="18" fill="#f9f6ef" stroke="#d7d1c4" stroke-width="1.5" />',
    ]

    # Add title at top
    tx = SVG_WIDTH * 0.5
    ty = 42.0
    svg_parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="28" font-weight="900" fill="#233843">Bird View</text>')

    show_stack = active_name if active_name in {"wall", "brick_supply"} else None
    if show_stack is None:
        step_key = _normalize_step_key(active_target.get("step_name"))
        step_number = _coerce_int(active_target.get("step_number"), None)
        inferred = _workspace_target_for_step(snapshot, step_key, step_number)
        if inferred in {"wall", "brick_supply"}:
            show_stack = inferred
    if show_stack not in {"wall", "brick_supply"}:
        show_stack = "brick_supply"
    stack_screen = None
    stack_center = None

    wall = snapshot["objects"]["wall"]
    if show_stack == "wall":
        wall_cx = float(wall.get("x_mm", 0.0))
        wall_cy = float(wall.get("y_mm", 0.0))
        wall_rect = _wall_render_points(wall)
        wall_screen = [project(x, y) for x, y in wall_rect]
        wall_screen = [(sx + float(BIRD_STACK_SHIFT_X_PX), sy) for sx, sy in wall_screen]
        wall_screen = _anchor_stack_screen_points(wall_screen)
        stack_screen = list(wall_screen)
        w_tx, w_ty = project(wall_cx, wall_cy)
        w_tx = float(w_tx) + float(BIRD_STACK_SHIFT_X_PX)
        stack_center = (float(w_tx), float(BIRD_STACK_CENTER_Y_PX))
        wall_style = _stack_visual_style("wall")
        wall_count_text = None if wall.get("count") is None else str(int(wall["count"]))
        min_x, max_x, min_y, max_y = _screen_bounds(wall_screen)
        wall_size_px = max(float(max_x) - float(min_x), float(max_y) - float(min_y))
        svg_parts.extend(
            _stack_card_svg(
                center_x=float(w_tx),
                center_y=float(BIRD_STACK_CENTER_Y_PX),
                size_px=float(wall_size_px),
                label=str(wall_style["label"]),
                fill=str(wall_style["fill"]),
                stroke=str(wall_style["stroke"]),
                count_text=wall_count_text,
                count_font_size=24.0,
                count_y_nudge=10.0,
            )
        )

    supply = snapshot["objects"]["brick_supply"]
    if show_stack == "brick_supply":
        supply_rect = _rotated_rect(
            float(supply.get("x_mm", 0.0)),
            float(supply.get("y_mm", 0.0)),
            float(supply.get("theta_deg", 0.0)),
            54.0,
            54.0,
        )
        supply_screen = [project(x, y) for x, y in supply_rect]
        supply_screen = [
            (sx + float(BIRD_STACK_SHIFT_X_PX) + float(BIRD_SUPPLY_RENDER_SHIFT_X_PX), sy)
            for sx, sy in supply_screen
        ]
        supply_screen = _anchor_stack_screen_points(supply_screen)
        stack_screen = list(supply_screen)
        supply_tx, supply_ty = project(float(supply.get("x_mm", 0.0)), float(supply.get("y_mm", 0.0)))
        supply_tx = (
            float(supply_tx)
            + float(BIRD_STACK_SHIFT_X_PX)
            + float(BIRD_SUPPLY_RENDER_SHIFT_X_PX)
        )
        stack_center = (float(supply_tx), float(BIRD_STACK_CENTER_Y_PX))
        supply_style = _stack_visual_style("brick_supply")
        supply_count_text = None if supply.get("count") is None else str(int(supply["count"]))
        min_x, max_x, min_y, max_y = _screen_bounds(supply_screen)
        supply_size_px = max(float(max_x) - float(min_x), float(max_y) - float(min_y))
        svg_parts.extend(
            _stack_card_svg(
                center_x=float(supply_tx),
                center_y=float(BIRD_STACK_CENTER_Y_PX),
                size_px=float(supply_size_px),
                label=str(supply_style["label"]),
                fill=str(supply_style["fill"]),
                stroke=str(supply_style["stroke"]),
                count_text=supply_count_text,
            )
        )

    robot = snapshot["robot"]
    target_obj = (snapshot.get("objects") or {}).get(active_name) if active_name in {"wall", "brick_supply"} else None

    # Build a dedicated observation chart so distance maps left/right and
    # x-axis maps up/down relative to the stack-centered baseline.
    def _bird_history_screen_point(entry: dict) -> tuple[float, float]:
        try:
            if not isinstance(entry, dict):
                raise ValueError("entry")
            if not isinstance(stack_screen, list) or len(stack_screen) < 3:
                raise ValueError("stack")

            min_x, max_x, min_y, max_y = _screen_bounds(stack_screen)
            center_x = (min_x + max_x) * 0.5

            track_y = None
            if isinstance(stack_center, tuple) and len(stack_center) == 2:
                track_y = float(stack_center[1])
            if track_y is None:
                track_y = (float(min_y) + float(max_y)) * 0.5
            track_y = max(58.0, min(float(BIRD_SVG_HEIGHT) - 26.0, float(track_y)))

            if center_x <= (float(SVG_WIDTH) * 0.5):
                near_x = min(float(SVG_WIDTH) - float(BIRD_DIST_TRACK_MARGIN_PX), max_x + float(BIRD_DIST_TRACK_STACK_GAP_PX))
                far_x = float(SVG_WIDTH) - float(BIRD_DIST_TRACK_MARGIN_PX)
            else:
                near_x = max(float(BIRD_DIST_TRACK_MARGIN_PX), min_x - float(BIRD_DIST_TRACK_STACK_GAP_PX))
                far_x = float(BIRD_DIST_TRACK_MARGIN_PX)

            if abs(float(far_x) - float(near_x)) < 10.0:
                near_x = float(SVG_WIDTH) * 0.5
                far_x = near_x + 120.0

            dist_mm = _coerce_float(entry.get("dist_mm", entry.get("target_range_mm")), None)
            if dist_mm is None and isinstance(target_obj, dict):
                ex = _coerce_float(entry.get("x_mm"), None)
                ey = _coerce_float(entry.get("y_mm"), None)
                tx_obj = _coerce_float(target_obj.get("x_mm"), None)
                ty_obj = _coerce_float(target_obj.get("y_mm"), None)
                if ex is not None and ey is not None and tx_obj is not None and ty_obj is not None:
                    dist_mm = math.hypot(float(tx_obj) - float(ex), float(ty_obj) - float(ey))
            if dist_mm is None:
                dist_mm = _bird_dist_track_far_mm()

            dist_ratio = _bird_distance_track_ratio(dist_mm)
            px = float(near_x) + (float(far_x) - float(near_x)) * dist_ratio
            x_axis_mm = _coerce_float(entry.get("x_axis_mm"), None)
            if x_axis_mm is None:
                return float(px), float(track_y)
            x_ratio_raw = (float(x_axis_mm) - float(BIRD_X_AXIS_TARGET_MM)) / 15.0
            x_ratio = max(-1.0, min(1.0, float(x_ratio_raw)))
            py = float(track_y) - (x_ratio * float(BIRD_X_AXIS_TRACK_HALF_SPAN_PX))
            py = max(58.0, min(float(BIRD_SVG_HEIGHT) - 26.0, float(py)))
            return float(px), float(py)
        except Exception:
            return project(float(entry.get("x_mm", 0.0)), float(entry.get("y_mm", 0.0)))

    if isinstance(stack_screen, list) and len(stack_screen) >= 3:
        guide_start_dist_mm = _bird_dist_track_near_mm()
        guide_end_dist_mm = _bird_dist_track_far_mm()
        guide_start = _bird_history_screen_point({"target_range_mm": guide_start_dist_mm})
        guide_end = _bird_history_screen_point({"target_range_mm": guide_end_dist_mm})
        svg_parts.append(
            f'<line x1="{guide_start[0]:.1f}" y1="{guide_start[1]:.1f}" x2="{guide_end[0]:.1f}" y2="{guide_end[1]:.1f}" stroke="#b7aea0" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.7" />'
        )
        svg_parts.append(
            f'<text x="{guide_start[0]:.1f}" y="{guide_start[1] - 10:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#5d676e">{guide_start_dist_mm:.0f}mm</text>'
        )
        svg_parts.append(
            f'<text x="{guide_end[0]:.1f}" y="{guide_end[1] - 10:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#5d676e">{guide_end_dist_mm:.0f}mm</text>'
        )
        svg_parts.append(
            f'<text x="{guide_start[0] - 14.0:.1f}" y="{guide_start[1] + 4.0:.1f}" text-anchor="end" font-size="12" font-weight="700" fill="#5d676e">x=6</text>'
        )
        dist_low = float(BIRD_DIST_TARGET_MM) - float(BIRD_DIST_TOL_MM)
        dist_high = float(BIRD_DIST_TARGET_MM) + float(BIRD_DIST_TOL_MM)
        x_low = float(BIRD_X_AXIS_TARGET_MM) - float(BIRD_X_AXIS_TOL_MM)
        x_high = float(BIRD_X_AXIS_TARGET_MM) + float(BIRD_X_AXIS_TOL_MM)
        gate_tl = _bird_history_screen_point({"dist_mm": dist_low, "x_axis_mm": x_high})
        gate_br = _bird_history_screen_point({"dist_mm": dist_high, "x_axis_mm": x_low})
        gate_center = _bird_history_screen_point(
            {"dist_mm": float(BIRD_DIST_TARGET_MM), "x_axis_mm": float(BIRD_X_AXIS_TARGET_MM)}
        )
        gate_rx = min(gate_tl[0], gate_br[0])
        gate_ry = min(gate_tl[1], gate_br[1])
        gate_rw = abs(gate_br[0] - gate_tl[0])
        gate_rh = abs(gate_br[1] - gate_tl[1])
        svg_parts.append(
            f'<rect x="{gate_rx:.1f}" y="{gate_ry:.1f}" width="{gate_rw:.1f}" height="{gate_rh:.1f}" '
            'fill="#2aae6c" fill-opacity="0.13" stroke="#2aae6c" stroke-width="1.8" '
            'stroke-dasharray="5 3" rx="3" />'
        )
        svg_parts.append(
            f'<line x1="{gate_center[0]:.1f}" y1="{gate_ry - 10.0:.1f}" x2="{gate_center[0]:.1f}" y2="{gate_ry + gate_rh + 10.0:.1f}" '
            'stroke="#1f8f58" stroke-width="1.8" stroke-linecap="round" />'
        )
        svg_parts.append(
            f'<line x1="{gate_rx - 10.0:.1f}" y1="{gate_center[1]:.1f}" x2="{gate_rx + gate_rw + 10.0:.1f}" y2="{gate_center[1]:.1f}" '
            'stroke="#1f8f58" stroke-width="1.8" stroke-linecap="round" />'
        )
        svg_parts.append(
            f'<text x="{gate_center[0]:.1f}" y="{gate_ry - 16.0:.1f}" text-anchor="middle" font-size="11" font-weight="800" fill="#1f8f58">dist {BIRD_DIST_TARGET_MM:.1f} ±{BIRD_DIST_TOL_MM:.1f}</text>'
        )
        svg_parts.append(
            f'<text x="{gate_rx - 12.0:.1f}" y="{gate_center[1] - 5.0:.1f}" text-anchor="end" font-size="11" font-weight="800" fill="#1f8f58">x {BIRD_X_AXIS_TARGET_MM:.1f}</text>'
        )
        svg_parts.append(
            f'<text x="{gate_rx + gate_rw + 12.0:.1f}" y="{gate_center[1] + 14.0:.1f}" text-anchor="start" font-size="11" font-weight="700" fill="#1f8f58">±{BIRD_X_AXIS_TOL_MM:.1f}</text>'
        )

    if bird_history:
        previous_entry = None
        history_count = len(bird_history)
        history_rows: list[dict] = []
        for idx, entry in enumerate(bird_history):
            trend = _step_history_trend(previous_entry, entry)
            color = _bird_history_fill_color(idx, history_count)
            hx, hy = _bird_history_screen_point(entry)
            entry_data = html.escape(json.dumps({
                "view": "bird",
                "idx": idx,
                "dist": round(float(entry["dist_mm"]), 1) if entry.get("dist_mm") is not None else None,
                "x": round(float(entry["x_axis_mm"]), 2) if entry.get("x_axis_mm") is not None else None,
                "y": round(float(entry["y_axis_mm"]), 2) if entry.get("y_axis_mm") is not None else None,
                "act": entry.get("action_after"),
                "score": entry.get("score_after"),
                "note": entry.get("note_after"),
                "trend": trend,
            }), quote=True)
            history_rows.append(
                {
                    "x": float(hx),
                    "y": float(hy),
                    "trend": trend,
                    "color": color,
                    "entry_data": entry_data,
                    "is_first_after_action": bool(entry.get("is_first_after_action")),
                }
            )
            previous_entry = entry

        duplicate_groups: dict[tuple[float, float], list[int]] = {}
        for idx, row in enumerate(history_rows):
            key = (round(float(row["x"]), 1), round(float(row["y"]), 1))
            duplicate_groups.setdefault(key, []).append(idx)
        for indices in duplicate_groups.values():
            if len(indices) <= 1:
                continue
            cols = int(math.ceil(math.sqrt(len(indices))))
            rows = int(math.ceil(len(indices) / max(1, cols)))
            spacing_px = 9.0
            base_x = float(history_rows[indices[0]]["x"])
            base_y = float(history_rows[indices[0]]["y"])
            for spread_idx, row_idx in enumerate(indices):
                row_num = int(spread_idx // cols)
                col_num = int(spread_idx % cols)
                hx = base_x + (float(col_num) - (float(cols) - 1.0) * 0.5) * spacing_px
                hy = base_y + (float(row_num) - (float(rows) - 1.0) * 0.5) * spacing_px
                history_rows[row_idx]["x"] = max(36.0, min(float(SVG_WIDTH) - 36.0, float(hx)))
                history_rows[row_idx]["y"] = max(50.0, min(float(BIRD_SVG_HEIGHT) - 22.0, float(hy)))

        history_points = [(float(row["x"]), float(row["y"])) for row in history_rows]
        for row in history_rows:
            dot_opacity = "1.0" if row.get("is_first_after_action") else "0.4"
            svg_parts.append(
                f'<circle class="step-history-dot" data-trend="{row["trend"]}" data-entry="{row["entry_data"]}" '
                f'cx="{float(row["x"]):.1f}" cy="{float(row["y"]):.1f}" '
                f'r="{float(BIRD_HISTORY_DOT_RADIUS_PX):.1f}" fill="{row["color"]}" fill-opacity="{dot_opacity}" '
                'stroke="#f9f6ef" stroke-width="2" style="cursor:pointer" />'
            )
        if len(history_points) >= 2:
            svg_parts.insert(
                len(svg_parts) - history_count,
                f'<polyline class="step-history-path" points="{_polygon_points(history_points)}" '
                'fill="none" stroke="#7a95bb" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round" opacity="0.82" />',
            )

    current_visible = snapshot.get("last_visible_brick") or {}
    current_dist_mm = _coerce_float(current_visible.get("dist_mm"), None)
    current_x_axis_mm = _coerce_float(current_visible.get("x_axis_mm"), None)
    if bool(current_visible.get("visible", False)) and current_dist_mm is not None and current_x_axis_mm is not None:
        current_entry = {
            "dist_mm": float(current_dist_mm),
            "target_range_mm": float(current_dist_mm),
            "x_axis_mm": float(current_x_axis_mm),
            "y_axis_mm": _coerce_float(current_visible.get("y_axis_mm"), None),
            "target_name": active_name,
        }
        current_x, current_y = _bird_history_screen_point(current_entry)
        current_data = html.escape(json.dumps({
            "view": "bird",
            "idx": "current",
            "dist": round(float(current_dist_mm), 1),
            "x": round(float(current_x_axis_mm), 2),
            "y": (
                None
                if current_entry.get("y_axis_mm") is None
                else round(float(current_entry["y_axis_mm"]), 2)
            ),
        }), quote=True)
        label_y = float(current_y) - 15.0
        if label_y < 62.0:
            label_y = float(current_y) + 25.0
        svg_parts.append(
            '<g class="current-position" data-view="bird">'
            f'<circle class="current-position-halo" data-view="bird" cx="{current_x:.1f}" cy="{current_y:.1f}" r="18.0" fill="{CURRENT_POSITION_HALO_COLOR}" fill-opacity="0.28" />'
            f'<circle class="current-position-dot" data-view="bird" data-entry="{current_data}" cx="{current_x:.1f}" cy="{current_y:.1f}" r="9.0" fill="{CURRENT_POSITION_COLOR}" stroke="#f9f6ef" stroke-width="3.0" />'
            f'<text class="current-position-label" data-view="bird" x="{current_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="11" font-weight="900" fill="{CURRENT_POSITION_LABEL_COLOR}" stroke="#f9f6ef" stroke-width="2.8" paint-order="stroke">now {current_dist_mm:.0f}mm x {current_x_axis_mm:+.0f}</text>'
            "</g>"
        )

    # Add distance displays
    robot_x = float(robot.get("x_mm", 0.0))
    robot_y = float(robot.get("y_mm", 0.0))
    
    # Distance to active target (brick)
    if isinstance(target_obj, dict):
        target_x = float(target_obj.get("x_mm", 0.0))
        target_y = float(target_obj.get("y_mm", 0.0))
        dist_to_target = math.hypot(target_x - robot_x, target_y - robot_y)
        
        # Position text near the dashed line, closer to robot
        mid_x = robot_x + (target_x - robot_x) * 0.3
        mid_y = robot_y + (target_y - robot_y) * 0.3
        text_x, text_y = project(mid_x, mid_y)
        
        svg_parts.append(
            f'<text x="{text_x:.1f}" y="{text_y - 8:.1f}" text-anchor="middle" font-size="16" font-weight="600" fill="#233843" stroke="#f9f6ef" stroke-width="2" paint-order="stroke">{dist_to_target:.0f}</text>'
        )
    
    # Do not render distance text for hidden stacks in compact single-stack mode.

    if bool(snapshot["held_brick"].get("held", False)):
        hx, hy = _heading_vector(float(robot.get("theta_deg", 0.0)))
        brick_center_x = float(robot.get("x_mm", 0.0)) + hx * 20.0
        brick_center_y = float(robot.get("y_mm", 0.0)) + hy * 20.0
        held_length_mm = (
            float(snapshot["held_brick"].get("length_mm", DEFAULT_BRICK_MM["length_mm"]))
            * float(BIRD_HELD_BRICK_RENDER_SCALE)
        )
        held_width_mm = (
            float(snapshot["held_brick"].get("width_mm", DEFAULT_BRICK_MM["width_mm"]))
            * float(BIRD_HELD_BRICK_RENDER_SCALE)
        )
        held_rect = _rotated_rect(
            brick_center_x,
            brick_center_y,
            float(robot.get("theta_deg", 0.0)),
            float(held_length_mm),
            float(held_width_mm),
        )
        held_screen = [project(x, y) for x, y in held_rect]
        svg_parts.append(
            f'<polygon points="{_polygon_points(held_screen)}" fill="#63f2f7" stroke="#18494e" stroke-width="2.5" />'
        )
        htx, hty = project(brick_center_x, brick_center_y)
        svg_parts.append(f'<text x="{htx:.1f}" y="{hty - 18:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#18494e">Held Brick</text>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_mast_svg(state: dict | None) -> str:
    snapshot = _normalized_render_snapshot(state)
    mast_history = _current_step_mast_history(snapshot)
    active_target = snapshot.get("active_target") or {}
    active_name = str(active_target.get("object_name") or "").strip().lower()
    objects = snapshot.get("objects") or {}
    supply = objects.get("brick_supply") or {}
    wall = objects.get("wall") or {}
    last_visible = snapshot.get("last_visible_brick") or {}

    show_stack = active_name if active_name in {"wall", "brick_supply"} else None
    if show_stack is None:
        step_key = _normalize_step_key(active_target.get("step_name"))
        step_number = _coerce_int(active_target.get("step_number"), None)
        inferred = _workspace_target_for_step(snapshot, step_key, step_number)
        if inferred in {"wall", "brick_supply"}:
            show_stack = inferred
    if show_stack not in {"wall", "brick_supply"}:
        show_stack = "brick_supply"

    plot_left = 86.0
    plot_right = float(SVG_WIDTH) - 86.0
    plot_top = 72.0
    plot_bottom = float(MAST_SVG_HEIGHT) - 44.0
    plot_height = max(1.0, plot_bottom - plot_top)
    stack_x = 216.0
    stack_size_px = 92.0
    zero_y = plot_top + (plot_height * 0.5)
    stack_right_x = float(stack_x) + (float(stack_size_px) * 0.5)
    history_left_x = max(plot_left + 18.0, stack_right_x + 10.0)
    history_right_x = plot_right - 24.0
    axis_extent_mm = _mast_y_axis_extent_mm(snapshot)

    def project_y_axis_to_y(y_axis_mm: float | None) -> float:
        if y_axis_mm is None:
            return float(zero_y)
        val = max(-axis_extent_mm, min(axis_extent_mm, float(y_axis_mm)))
        usable_half_span = max(16.0, (plot_height * 0.42))
        return float(zero_y) - (val / axis_extent_mm) * usable_half_span

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{MAST_SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {MAST_SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        f'<rect x="22" y="22" width="936" height="{MAST_SVG_HEIGHT - 44}" rx="24" fill="#f9f6ef" stroke="#d7d1c4" stroke-width="1.5" />',
    ]
    title_x = SVG_WIDTH * 0.5
    svg_parts.append(f'<text x="{title_x:.1f}" y="52.0" text-anchor="middle" font-size="28" font-weight="900" fill="#233843">Mast View</text>')
    # Single-brick reference: 0mm is at the square center.
    stack_style = _stack_visual_style(show_stack)
    square_top = float(zero_y) - (stack_size_px * 0.5)
    zero_guide_right_x = max(plot_left + 24.0, stack_x - (stack_size_px * 0.5) - 8.0)
    svg_parts.extend(
        _stack_card_svg(
            center_x=float(stack_x),
            center_y=float(zero_y),
            size_px=float(stack_size_px),
            label=str(stack_style["label"]),
            fill=str(stack_style["fill"]),
            stroke=str(stack_style["stroke"]),
            count_text=None,
        )
    )
    svg_parts.append(
        f'<line class="mast-zero-guide" x1="{plot_left:.1f}" y1="{zero_y:.1f}" '
        f'x2="{zero_guide_right_x:.1f}" y2="{zero_y:.1f}" '
        'stroke="#c8c2b5" stroke-width="2.6" stroke-linecap="round" />'
    )
    svg_parts.append(
        f'<text x="{plot_left - 10:.1f}" y="{zero_y + 5:.1f}" text-anchor="end" font-size="12" font-weight="700" fill="#5d676e">0mm</text>'
    )
    top_label_y = project_y_axis_to_y(axis_extent_mm)
    bottom_label_y = project_y_axis_to_y(-axis_extent_mm)
    svg_parts.append(
        f'<text x="{plot_left - 10:.1f}" y="{top_label_y + 4:.1f}" text-anchor="end" font-size="11" font-weight="600" fill="#5d676e">+{int(axis_extent_mm)}mm</text>'
    )
    svg_parts.append(
        f'<text x="{plot_left - 10:.1f}" y="{bottom_label_y + 4:.1f}" text-anchor="end" font-size="11" font-weight="600" fill="#5d676e">-{int(axis_extent_mm)}mm</text>'
    )
    mast_gate_top_y = project_y_axis_to_y(float(MAST_Y_AXIS_TARGET_MM) + float(MAST_Y_AXIS_TOL_MM))
    mast_gate_bottom_y = project_y_axis_to_y(float(MAST_Y_AXIS_TARGET_MM) - float(MAST_Y_AXIS_TOL_MM))
    mast_gate_center_y = project_y_axis_to_y(float(MAST_Y_AXIS_TARGET_MM))
    svg_parts.append(
        f'<rect x="{history_left_x:.1f}" y="{mast_gate_top_y:.1f}" '
        f'width="{history_right_x - history_left_x:.1f}" height="{mast_gate_bottom_y - mast_gate_top_y:.1f}" '
        'fill="#2aae6c" fill-opacity="0.13" stroke="#2aae6c" stroke-width="1.8" '
        'stroke-dasharray="5 3" rx="3" />'
    )
    svg_parts.append(
        f'<line x1="{history_left_x - 10.0:.1f}" y1="{mast_gate_center_y:.1f}" '
        f'x2="{history_right_x + 10.0:.1f}" y2="{mast_gate_center_y:.1f}" '
        'stroke="#1f8f58" stroke-width="1.8" stroke-linecap="round" />'
    )
    svg_parts.append(
        f'<text x="{history_left_x - 14.0:.1f}" y="{mast_gate_center_y + 4.0:.1f}" text-anchor="end" font-size="11" font-weight="800" fill="#1f8f58">target {MAST_Y_AXIS_TARGET_MM:.1f}</text>'
    )
    svg_parts.append(
        f'<text x="{history_left_x - 14.0:.1f}" y="{mast_gate_top_y + 4.0:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="#1f8f58">+{MAST_Y_AXIS_TOL_MM:.1f}</text>'
    )
    svg_parts.append(
        f'<text x="{history_left_x - 14.0:.1f}" y="{mast_gate_bottom_y + 4.0:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="#1f8f58">-{MAST_Y_AXIS_TOL_MM:.1f}</text>'
    )

    if mast_history:
        svg_parts.append('<g id="mast-history">')
        history_y_values = [
            float(_coerce_float(entry.get("y_axis_mm"), None))
            for entry in mast_history
            if _coerce_float(entry.get("y_axis_mm"), None) is not None
        ]
        history_points = _mast_history_screen_points(
            history_y_values,
            history_left_x=history_left_x,
            history_right_x=history_right_x,
            project_y=project_y_axis_to_y,
        )
        if history_points:
            latest_x, latest_y = history_points[-1]
            lead_start_x = stack_right_x + 2.0
            if float(lead_start_x) < float(latest_x):
                svg_parts.append(
                    f'<line class="mast-history-lead" x1="{lead_start_x:.1f}" y1="{latest_y:.1f}" '
                    f'x2="{latest_x:.1f}" y2="{latest_y:.1f}" '
                    'stroke="#0b3d91" stroke-width="1.4" stroke-linecap="round" opacity="0.95" />'
                )
        if len(history_points) >= 2:
            trace_points = list(reversed(history_points))
            svg_parts.append(
                f'<polyline class="mast-history-trace" points="{_polygon_points(trace_points)}" '
                'fill="none" stroke="#9db4da" stroke-width="1.4" stroke-linecap="round" '
                'stroke-linejoin="round" opacity="0.55" />'
            )
            segment_count = len(history_points) - 1
            for segment_idx in range(1, len(history_points)):
                start_x, start_y = history_points[segment_idx - 1]
                end_x, end_y = history_points[segment_idx]
                style = _mast_history_line_style(segment_count - segment_idx)
                svg_parts.append(
                    f'<line class="mast-history-segment" data-recency="{style["recency"]}" '
                    f'x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
                    f'stroke="{style["stroke"]}" stroke-width="{style["stroke_width"]}" '
                    'stroke-linecap="round" />'
                )
        for pt_idx, (px, py) in enumerate(history_points):
            entry = mast_history[pt_idx] if pt_idx < len(mast_history) else {}
            entry_data = html.escape(json.dumps({
                "view": "mast",
                "idx": pt_idx,
                "dist": round(float(entry["dist_mm"]), 1) if entry.get("dist_mm") is not None else None,
                "x": round(float(entry["x_axis_mm"]), 2) if entry.get("x_axis_mm") is not None else None,
                "y": round(float(entry["y_axis_mm"]), 2) if entry.get("y_axis_mm") is not None else None,
                "lift": round(float(entry["current_lift_mm"]), 1) if entry.get("current_lift_mm") is not None else None,
                "act": entry.get("action_type"),
                "score": entry.get("speed_score"),
                "note": entry.get("action_note"),
            }), quote=True)
            is_latest = (pt_idx == len(history_points) - 1)
            dot_fill = "#0b3d91" if is_latest else "#9db4da"
            svg_parts.append(
                f'<circle class="mast-history-dot" data-entry="{entry_data}" '
                f'cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{dot_fill}" fill-opacity="0.9" '
                'stroke="#f9f6ef" stroke-width="1.5" style="cursor:pointer" />'
            )
        svg_parts.append("</g>")

    current_y_axis_mm = _coerce_float(last_visible.get("y_axis_mm"), None)
    if bool(last_visible.get("visible", False)) and current_y_axis_mm is not None:
        current_x = float(history_left_x)
        current_y = float(project_y_axis_to_y(current_y_axis_mm))
        current_dist_mm = _coerce_float(last_visible.get("dist_mm"), None)
        current_x_axis_mm = _coerce_float(last_visible.get("x_axis_mm"), None)
        current_data = html.escape(json.dumps({
            "view": "mast",
            "idx": "current",
            "dist": None if current_dist_mm is None else round(float(current_dist_mm), 1),
            "x": None if current_x_axis_mm is None else round(float(current_x_axis_mm), 2),
            "y": round(float(current_y_axis_mm), 2),
            "lift": round(
                float(_coerce_float((snapshot.get("robot") or {}).get("lift_mm"), 0.0) or 0.0),
                1,
            ),
        }), quote=True)
        label_y = float(current_y) - 14.0
        if label_y < 76.0:
            label_y = float(current_y) + 25.0
        svg_parts.append(
            '<g class="current-position" data-view="mast">'
            f'<line class="current-position-lead" data-view="mast" x1="{stack_right_x + 2.0:.1f}" y1="{current_y:.1f}" x2="{current_x:.1f}" y2="{current_y:.1f}" stroke="{CURRENT_POSITION_COLOR}" stroke-width="2.2" stroke-linecap="round" opacity="0.9" />'
            f'<circle class="current-position-halo" data-view="mast" cx="{current_x:.1f}" cy="{current_y:.1f}" r="18.0" fill="{CURRENT_POSITION_HALO_COLOR}" fill-opacity="0.28" />'
            f'<circle class="current-position-dot" data-view="mast" data-entry="{current_data}" cx="{current_x:.1f}" cy="{current_y:.1f}" r="9.0" fill="{CURRENT_POSITION_COLOR}" stroke="#f9f6ef" stroke-width="3.0" />'
            f'<text class="current-position-label" data-view="mast" x="{current_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="11" font-weight="900" fill="{CURRENT_POSITION_LABEL_COLOR}" stroke="#f9f6ef" stroke-width="2.8" paint-order="stroke">now y {current_y_axis_mm:+.0f}mm</text>'
            "</g>"
        )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _summary_payload(state: dict) -> dict:
    plan = _plan_reverse_then_turn_from_state(state, turn_cmd="l")
    payload = {
        "updated_at": float(state.get("updated_at", time.time())),
        "robot": _deepcopy(state.get("robot", {})),
        "raw_robot": _deepcopy(state.get("raw_robot", {})),
        "leia": _deepcopy(state.get("leia", {})),
        "objects": _deepcopy(state.get("objects", {})),
        "active_target": _deepcopy(state.get("active_target", {})),
        "held_brick": _deepcopy(state.get("held_brick", {})),
        "wall_reverse_plan": plan,
        "last_visible_brick": _deepcopy(state.get("last_visible_brick", {})),
    }
    if bool(state.get("run_replay")):
        payload["run_replay"] = True
        payload["run_log_path"] = str(state.get("run_log_path") or "")
        payload["history"] = _deepcopy(state.get("history", []))
    return payload


def render_workspace_html(state: dict | None) -> str:
    if isinstance(state, dict) and state.get("run_log") and not state.get("history"):
        state = build_state_from_run_log(state["run_log"])
    snapshot = _normalized_render_snapshot(state)
    bird_svg = render_workspace_svg(snapshot)
    mast_svg = render_mast_svg(snapshot)
    active = snapshot.get("active_target") or {}
    step_name = str(active.get("step_name") or "").upper() or "RUN"
    dist_target = 149.26
    dist_tol = 5.0
    x_target = 6.03
    x_tol = 5.0
    y_target = 4.13
    y_tol = 2.3
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(step_name)} Run View</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      display: flex;
      min-height: 100vh;
      background: #ede8df;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: #10202b;
    }}
    .charts {{
      flex: 1;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
    }}
    .chart-card {{
      background: #faf7f2;
      border: 1px solid #d9d3c8;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 18px rgba(16,32,43,0.07);
    }}
    .chart-card svg {{ display: block; width: 100%; height: auto; }}
    .sidebar {{
      width: 320px;
      flex-shrink: 0;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      border-left: 1px solid #ccc8bf;
      background: #f5f1ea;
      overflow-y: auto;
    }}
    .sidebar h2 {{
      font-size: 15px;
      font-weight: 700;
      color: #233843;
    }}
    .prompt {{
      font-size: 13px;
      color: #7a7368;
      line-height: 1.55;
    }}
    .entry-card {{
      background: #fff;
      border: 1px solid #d9d3c8;
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .entry-card h3 {{
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: #5d676e;
    }}
    .kv-table {{ width: 100%; border-collapse: collapse; }}
    .kv-table td {{ padding: 3px 0; font-size: 13px; vertical-align: top; }}
    .kv-table td:first-child {{ color: #7a7368; white-space: nowrap; padding-right: 10px; width: 40%; }}
    .kv-table td:last-child {{ font-weight: 600; color: #10202b; }}
    .err-ok {{ color: #2f9e44; }}
    .err-bad {{ color: #d94841; }}
    .err-warn {{ color: #c59d2a; }}
    .trend-closer {{ background: #d3f9d8; color: #2f9e44; }}
    .trend-further {{ background: #ffe3e3; color: #d94841; }}
    .trend-neutral {{ background: #fff3cd; color: #c59d2a; }}
    .trend-unknown {{ background: #f1f3f5; color: #868e96; }}
    .trend-badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 99px;
      font-size: 12px;
      font-weight: 700;
    }}
    .act-badge {{
      display: inline-block;
      background: #233843;
      color: #fff;
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .03em;
    }}
    [data-entry] {{ cursor: pointer; transition: opacity .1s; }}
    [data-entry]:hover {{ opacity: .75; }}
    .dot-selected {{ stroke: #ff7a00 !important; stroke-width: 3.5px !important; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: #5d676e; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  </style>
</head>
<body>
  <div class="charts">
    <div class="chart-card">{bird_svg}</div>
    <div class="chart-card">{mast_svg}</div>
  </div>
  <div class="sidebar">
    <div>
      <h2>{html.escape(step_name)}</h2>
      <p class="prompt" style="margin-top:6px">Click any dot to see what the robot was thinking at that moment.</p>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#2f9e44"></div> Closer</div>
      <div class="legend-item"><div class="legend-dot" style="background:#d94841"></div> Further</div>
      <div class="legend-item"><div class="legend-dot" style="background:#c59d2a"></div> Neutral</div>
      <div class="legend-item"><div class="legend-dot" style="background:#0b3d91"></div> Latest</div>
    </div>
    <div id="detail" style="display:none"></div>
    <p class="prompt" id="no-selection">No dot selected yet.</p>
  </div>

  <script>
    const DIST_TARGET = {dist_target}, DIST_TOL = {dist_tol};
    const X_TARGET = {x_target}, X_TOL = {x_tol};
    const Y_TARGET = {y_target}, Y_TOL = {y_tol};

    function fmt(v, dec) {{
      return v === null || v === undefined ? '—' : Number(v).toFixed(dec);
    }}
    function errClass(err, tol) {{
      if (err === null || err === undefined) return '';
      return Math.abs(err) <= tol ? 'err-ok' : 'err-bad';
    }}
    function errSign(v) {{
      if (v === null || v === undefined) return '—';
      return (v >= 0 ? '+' : '') + Number(v).toFixed(1) + 'mm';
    }}
    function trendClass(t) {{ return 'trend-' + (t || 'unknown'); }}

    let selectedDot = null;

    function spreadOverlappingBirdDots() {{
      const dots = Array.from(document.querySelectorAll('.step-history-dot[data-entry]'));
      if (!dots.length) return;
      const groups = new Map();
      dots.forEach(dot => {{
        const key = `${{dot.getAttribute('cx')}},${{dot.getAttribute('cy')}}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(dot);
      }});
      groups.forEach(group => {{
        if (group.length <= 1) return;
        const baseX = Number(group[0].getAttribute('cx'));
        const baseY = Number(group[0].getAttribute('cy'));
        const cols = Math.ceil(Math.sqrt(group.length));
        const spacing = 9;
        group.forEach((dot, idx) => {{
          const row = Math.floor(idx / cols);
          const col = idx % cols;
          const px = baseX + (col - (cols - 1) / 2) * spacing;
          const py = baseY + (row - (Math.ceil(group.length / cols) - 1) / 2) * spacing;
          dot.setAttribute('cx', px.toFixed(1));
          dot.setAttribute('cy', py.toFixed(1));
        }});
      }});
      const points = dots.map(dot => `${{dot.getAttribute('cx')}},${{dot.getAttribute('cy')}}`);
      const path = document.querySelector('.step-history-path');
      if (path && points.length >= 2) path.setAttribute('points', points.join(' '));
    }}

    document.addEventListener('click', function(e) {{
      const dot = e.target.closest('[data-entry]');
      if (!dot) return;
      selectDot(dot);
    }});

    function selectDot(dot) {{
      if (!dot) return;
      if (selectedDot) selectedDot.classList.remove('dot-selected');
      dot.classList.add('dot-selected');
      selectedDot = dot;
      const entry = JSON.parse(dot.dataset.entry);
      showEntry(entry);
    }}

    window.addEventListener('DOMContentLoaded', function() {{
      spreadOverlappingBirdDots();
      const dots = Array.from(document.querySelectorAll('[data-entry]'));
      if (dots.length) selectDot(dots[dots.length - 1]);
    }});

    function showEntry(e) {{
      const distErr = e.dist !== null && e.dist !== undefined ? e.dist - DIST_TARGET : null;
      const xErr = e.x !== null && e.x !== undefined ? e.x - X_TARGET : null;
      const yErr = e.y !== null && e.y !== undefined ? e.y - Y_TARGET : null;

      const trendHtml = e.trend
        ? `<span class="trend-badge ${{trendClass(e.trend)}}">${{e.trend}}</span>`
        : '';

      // Infer which gap was targeted from the decision sentence or error magnitudes
      function gapTarget() {{
        if (!e.note) {{
          // Fallback: infer from largest absolute error
          const gaps = [
            {{label:'dist', err: distErr, tol: DIST_TOL}},
            {{label:'x_axis', err: xErr, tol: X_TOL}},
            {{label:'y_axis', err: yErr, tol: Y_TOL}},
          ].filter(g => g.err !== null && Math.abs(g.err) > g.tol);
          if (!gaps.length) return 'all in gate — holding';
          gaps.sort((a,b) => Math.abs(b.err) - Math.abs(a.err));
          return `closing ${{gaps[0].label}} gap (${{errSign(gaps[0].err)}})`;
        }}
        return null;
      }}

      const decisionNote = e.note || null;
      const decisionFallback = gapTarget();
      const hasDecision = decisionNote || decisionFallback;

      document.getElementById('no-selection').style.display = 'none';
      document.getElementById('detail').style.display = '';
      document.getElementById('detail').innerHTML = `
        <div class="entry-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <h3>Position</h3>
            ${{trendHtml}}
          </div>
          <table class="kv-table">
            <tr><td>dist</td><td class="${{errClass(distErr, DIST_TOL)}}">${{fmt(e.dist,1)}}mm</td></tr>
            <tr><td>x_axis</td><td class="${{errClass(xErr, X_TOL)}}">${{fmt(e.x,2)}}mm</td></tr>
            <tr><td>y_axis</td><td class="${{errClass(yErr, Y_TOL)}}">${{fmt(e.y,2)}}mm</td></tr>
            ${{e.lift !== undefined && e.lift !== null ? `<tr><td>lift</td><td>${{fmt(e.lift,1)}}mm</td></tr>` : ''}}
          </table>
        </div>
        <div class="entry-card">
          <h3>Error from goal</h3>
          <table class="kv-table">
            <tr><td>dist (±${{DIST_TOL}}mm)</td><td class="${{errClass(distErr, DIST_TOL)}}">${{errSign(distErr)}}</td></tr>
            <tr><td>x_axis (±${{X_TOL}}mm)</td><td class="${{errClass(xErr, X_TOL)}}">${{errSign(xErr)}}</td></tr>
            <tr><td>y_axis (±${{Y_TOL}}mm)</td><td class="${{errClass(yErr, Y_TOL)}}">${{errSign(yErr)}}</td></tr>
          </table>
        </div>
        ${{hasDecision ? `
        <div class="entry-card">
          <h3>Previous act</h3>
          ${{decisionNote
            ? `<p style="margin:0;font-size:13px;line-height:1.5;color:#1a2430;font-family:monospace;white-space:pre-wrap">${{decisionNote}}</p>`
            : `<p style="margin:0;font-size:13px;color:#5d676e">${{decisionFallback}}</p>`
          }}
        </div>` : ''}}
      `;
    }}
  </script>
</body>
</html>
"""
def build_state_from_run_log(log_path: str | Path) -> dict:
    """Parse a run log JSON file and build a state dict for rendering."""
    try:
        with open(log_path, encoding="utf-8") as f:
            content = f.read().rstrip()
    except Exception:
        return {}
    if not content.endswith("]"):
        content = content.rstrip().rstrip(",") + "]"
    try:
        events = json.loads(content)
    except Exception:
        return {}
    if not isinstance(events, list):
        return {}

    step_name = "ALIGN_BRICK"
    for evt in events:
        if isinstance(evt, dict) and evt.get("type") == "keyframe":
            s = str(evt.get("step") or "").strip()
            if s:
                step_name = s
                break

    history = []
    step_seq = 1
    act_idx = 0
    last_action = None
    last_action_first_state_seen = None
    for evt in events:
        if not isinstance(evt, dict):
            continue
        evt_type = str(evt.get("type") or "").strip().lower()
        if evt_type == "action":
            last_action = evt
            act_idx += 1
        elif evt_type == "state":
            brick = evt.get("brick") or {}
            pose = evt.get("robot_pose") or {}
            visible = bool(brick.get("visible", False))
            dist = brick.get("dist")
            x_axis = brick.get("x_axis")
            y_axis = brick.get("y_axis")
            if visible and dist is not None and x_axis is not None:
                lift_height = _coerce_float(evt.get("lift_height"), None)
                if lift_height is None:
                    lift_height = _coerce_float(pose.get("height_mm"), 0.0) or 0.0
                entry = {
                    "type": "sync",
                    "target_visible": True,
                    "dist_mm": float(dist),
                    "target_range_mm": float(dist),
                    "x_axis_mm": float(x_axis),
                    "y_axis_mm": (None if y_axis is None else float(y_axis)),
                    "step_name": step_name,
                    "step_seq": step_seq,
                    "x_mm": float(pose.get("x", 0.0) or 0.0),
                    "y_mm": float(pose.get("y", 0.0) or 0.0),
                    "theta_deg": float(pose.get("theta", 0.0) or 0.0),
                    "current_lift_mm": float(lift_height),
                    "camera_height_mm": float(lift_height),
                    "act_idx": act_idx,
                }
                is_first_after_action = False
                if last_action is not None:
                    action_cmd = str(last_action.get("command") or "")
                    action_score = last_action.get("speedScore")
                    action_note = str(last_action.get("actionNote") or "").strip()
                    action_display = str(last_action.get("actionDisplay") or "").strip()
                    entry["action_cmd"] = action_cmd
                    entry["action_score"] = action_score
                    entry["action_after"] = action_cmd
                    entry["score_after"] = action_score
                    entry["action_type"] = action_cmd
                    entry["speed_score"] = action_score
                    if action_note:
                        entry["action_note"] = action_note
                        entry["note_after"] = action_note
                    if action_display:
                        entry["action_display"] = action_display
                        entry["action_after"] = action_display
                    custom_specs = last_action.get("customActionSpecs")
                    if isinstance(custom_specs, list) and custom_specs:
                        entry["custom_action_specs"] = [
                            dict(spec) for spec in custom_specs if isinstance(spec, dict)
                        ]
                    if last_action is not last_action_first_state_seen:
                        is_first_after_action = True
                        last_action_first_state_seen = last_action
                entry["is_first_after_action"] = is_first_after_action
                history.append(entry)

    state = _default_workspace_state(render_enabled=False)
    state["run_replay"] = True
    state["run_log_path"] = str(log_path)
    state["history"] = history
    state["active_target"] = {
        "step_name": step_name,
        "history_step_seq": step_seq,
    }
    if history:
        last = history[-1]
        state["last_visible_brick"] = {
            "visible": True,
            "dist_mm": last.get("dist_mm"),
            "x_axis_mm": last.get("x_axis_mm"),
            "y_axis_mm": last.get("y_axis_mm"),
        }
    return state


_RUN_VIEW_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _run_view_slug_from_log_path(log_path: str | Path) -> str:
    stem = Path(log_path).stem if log_path else f"run_{int(time.time())}"
    slug = re.sub(r"[^a-z0-9]+", "_", str(stem).strip().lower()).strip("_")
    return slug or f"run_{int(time.time())}"


def _run_view_js_string(value) -> str:
    return str(value or "").replace("\\", "\\\\").replace("'", "\\'")


def _run_view_entry_for_file(path: Path) -> tuple[tuple, dict]:
    stem = path.stem.lower()
    m = re.match(r"^([a-z]{3})_(\d{1,2})_(\d{1,2})(\d{2})(am|pm)$", stem)
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    if m:
        month_key, day_s, hour_s, minute_s, suffix = m.groups()
        month_num = int(_RUN_VIEW_MONTHS.get(month_key, 0) or 0)
        day = int(day_s)
        hour_display = int(hour_s)
        minute = int(minute_s)
        hour_24 = int(hour_display % 12) + (12 if suffix == "pm" else 0)
        try:
            year = int(time.localtime(mtime).tm_year)
        except Exception:
            year = int(time.localtime().tm_year)
        month_label = month_key.title()
        entry = {
            "file": path.name,
            "label": f"{month_label} {day} {hour_display}:{minute:02d} {suffix.upper()}",
            "group": f"{month_label} {day}",
        }
        return (year, month_num, day, hour_24, minute, mtime), entry
    label = path.stem.replace("_", " ").strip().title() or path.name
    entry = {"file": path.name, "label": label, "group": "Runs"}
    return (0, 0, 0, 0, 0, mtime), entry


def _run_view_entries_newest_first() -> list[dict]:
    if not RUN_VIEWS_DIR.exists():
        return []
    rows = []
    for path in RUN_VIEWS_DIR.glob("*.html"):
        if path.name == RUN_VIEWS_INDEX_PATH.name:
            continue
        rows.append(_run_view_entry_for_file(path))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [entry for _sort_key, entry in rows]


def _render_run_views_block(entries: list[dict]) -> str:
    lines = ["    const RUNS = ["]
    for idx, entry in enumerate(entries):
        comma = "," if idx < len(entries) - 1 else ""
        lines.append(
            "      { file: '"
            + _run_view_js_string(entry.get("file"))
            + "', label: '"
            + _run_view_js_string(entry.get("label"))
            + "', group: '"
            + _run_view_js_string(entry.get("group"))
            + f"' }}{comma}"
        )
    lines.append("    ];")
    return "\n".join(lines)


def _refresh_run_views_index() -> None:
    if not RUN_VIEWS_INDEX_PATH.exists():
        return
    entries = _run_view_entries_newest_first()
    try:
        text = RUN_VIEWS_INDEX_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    block = _render_run_views_block(entries)
    new_text, count = re.subn(
        r"    const RUNS = \[\n.*?\n    \];",
        block,
        text,
        count=1,
        flags=re.S,
    )
    if count and new_text != text:
        RUN_VIEWS_INDEX_PATH.write_text(new_text, encoding="utf-8")


def write_run_view_from_log(log_path: str | Path | None = None) -> None:
    """Regenerate run_view.html from a run log file (latest if not specified)."""
    if log_path is None:
        runs_dir = Path(__file__).resolve().parent / "Runs - cyan"
        logs = sorted(runs_dir.glob("*.json"))
        if not logs:
            return
        log_path = logs[-1]
    state = build_state_from_run_log(log_path)
    if not state:
        return
    _write_live_assets(state)


def _write_live_assets(state: dict) -> None:
    try:
        XYZ_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. HTML
        html_content = render_workspace_html(state)
        with open(LIVE_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
        run_log_path = state.get("run_log_path") if isinstance(state, dict) else None
        if run_log_path:
            try:
                RUN_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
                saved_path = RUN_VIEWS_DIR / f"{_run_view_slug_from_log_path(run_log_path)}.html"
                saved_path.write_text(html_content, encoding="utf-8")
                _refresh_run_views_index()
            except Exception as index_error:
                print(f"[XYZ-LAYOUT-ERROR] Failed to update saved run view: {index_error}", file=sys.stderr)
            
        # 2. SVG
        svg_content = render_workspace_svg(state)
        with open(LIVE_SVG_PATH, "w", encoding="utf-8") as f:
            f.write(svg_content)

        # 3. Mast SVG
        mast_svg_content = render_mast_svg(state)
        with open(LIVE_MAST_SVG_PATH, "w", encoding="utf-8") as f:
            f.write(mast_svg_content)

        # 4. JSON snapshot
        payload = _summary_payload(state)
        with open(LIVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
    except Exception as e:
        # Avoid crashing the robot logic if visualization writing fails
        print(f"[XYZ-LAYOUT-ERROR] Failed to write live assets: {e}", file=sys.stderr)


def get_workspace_snapshot(world) -> dict:
    state = ensure_workspace(world)
    state["updated_at"] = time.time()
    snapshot = _summary_payload(state)
    return snapshot
