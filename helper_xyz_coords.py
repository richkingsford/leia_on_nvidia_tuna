#!/usr/bin/env python3
"""Workspace xyz tracker and live top-down snapshot renderer."""

from __future__ import annotations

import html
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

XYZ_LAYOUT_DIR = Path(__file__).resolve().parent / "xyz layout"
LIVE_HTML_PATH = XYZ_LAYOUT_DIR / "index.html"
LIVE_SVG_PATH = XYZ_LAYOUT_DIR / "workspace.svg"
LIVE_MAST_SVG_PATH = XYZ_LAYOUT_DIR / "mast_view.svg"
LIVE_JSON_PATH = XYZ_LAYOUT_DIR / "workspace.json"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
SCHEMA_VERSION = 1

DEFAULT_CAMERA_Z_MM = 145.0
DEFAULT_BRICK_SUPPLY_POS_MM = {"x_mm": 0.0, "y_mm": -180.0, "z_mm": 0.0}
DEFAULT_WALL_POS_MM = {"x_mm": 180.0, "y_mm": 0.0, "z_mm": 0.0, "theta_deg": 180.0}
DEFAULT_WALL_RENDER_LENGTH_MM = 320.0
DEFAULT_STACK_RENDER_FOOTPRINT_MM = 54.0
DEFAULT_STACK_HEIGHT_MM = 44.0
DEFAULT_BRICK_MM = {"length_mm": 44.0, "width_mm": 22.0, "height_mm": 22.0}
DEFAULT_WORKSPACE_STEP_TARGETS = (
    (1, 2, "wall"),
    (3, 9, "brick_supply"),
    (10, 16, "wall"),
)
ROBOT_NOSE_TO_TAIL_MM = 68.0 * 1.7  # Made 1.7x longer for better visibility
ROBOT_TAIL_CENTER_MM = 58.0 * 1.7
ROBOT_HALF_WIDTH_MM = 22.0
ROBOT_OBJECT_MARGIN_MM = 6.0
BIRDSEYE_VIEW_MARGIN_MM = 35.0
BIRDSEYE_MIN_SPAN_X_MM = 220.0
BIRDSEYE_MIN_SPAN_Y_MM = 170.0
STEP_HISTORY_COLOR_CLOSER = "#2f9e44"
STEP_HISTORY_COLOR_FURTHER = "#d94841"
STEP_HISTORY_COLOR_NEUTRAL = "#c59d2a"
STEP_HISTORY_COLOR_UNKNOWN = "#7b7668"
STEP_HISTORY_RECENT_MEDIUM_DOTS = 3
STEP_HISTORY_DOT_RADIUS_TINY_PX = 2.0
STEP_HISTORY_DOT_RADIUS_MEDIUM_PX = 6.0

SVG_WIDTH = 980
SVG_HEIGHT = 680
SVG_PADDING = 80
MAST_SVG_HEIGHT = 360
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
    }


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
    _append_history(state, {"type": "sync", "reason": str(reason), "ts": state["updated_at"]})
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
    for obj in state["objects"].values():
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
    scale = min((SVG_WIDTH - 2 * SVG_PADDING) / span_x, (SVG_HEIGHT - 2 * SVG_PADDING) / span_y)

    def view_project(view_x_mm: float, view_y_mm: float) -> tuple[float, float]:
        sx = SVG_PADDING + (float(view_x_mm) - min_x) * scale
        sy = SVG_HEIGHT - SVG_PADDING - (float(view_y_mm) - min_y) * scale
        return sx, sy

    def project(x_mm: float, y_mm: float) -> tuple[float, float]:
        view_x_mm, view_y_mm = _view_coords(x_mm, y_mm)
        return view_project(view_x_mm, view_y_mm)

    return project, view_project, scale


def _polygon_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


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


def _history_dot_radius(index: int, history_count: int) -> float:
    is_recent = int(index) >= max(0, int(history_count) - int(STEP_HISTORY_RECENT_MEDIUM_DOTS))
    return (
        float(STEP_HISTORY_DOT_RADIUS_MEDIUM_PX)
        if is_recent
        else float(STEP_HISTORY_DOT_RADIUS_TINY_PX)
    )


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
        if camera_height_mm is None:
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


def render_workspace_svg(state: dict | None) -> str:
    snapshot = _deepcopy(state or _default_workspace_state(render_enabled=False))
    project, view_project, scale = _project_fn(snapshot)
    min_x, max_x, min_y, max_y = _build_viewbox(snapshot)
    active_target = snapshot.get("active_target") or {}
    active_name = active_target.get("object_name")
    step_history = _current_step_motion_history(snapshot)
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        '<rect x="22" y="22" width="936" height="636" rx="24" fill="#f9f6ef" stroke="#d7d1c4" stroke-width="1.5" />',
    ]

    # Add title at top
    tx = SVG_WIDTH * 0.5
    ty = 54.0
    svg_parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="28" font-weight="900" fill="#233843">Bird\'s Eye View</text>')

    wall = snapshot["objects"]["wall"]
    wall_cx = float(wall.get("x_mm", 0.0))
    wall_cy = float(wall.get("y_mm", 0.0))
    wall_rect = _wall_render_points(wall)
    wall_screen = [project(x, y) for x, y in wall_rect]
    wall_fill = "#c2523c" if bool(wall.get("valid", False)) else "#d4a090"
    wall_stroke = "#f3c548" if active_name == "wall" else "#7b2e1e"
    wall_stroke_width = "4.5" if active_name == "wall" else "2.5"
    svg_parts.append(
        f'<polygon points="{_polygon_points(wall_screen)}" fill="{wall_fill}" stroke="{wall_stroke}" stroke-width="{wall_stroke_width}" />'
    )
    w_tx, w_ty = project(wall_cx, wall_cy)
    svg_parts.append(
        f'<text x="{w_tx:.1f}" y="{w_ty - 46:.1f}" text-anchor="middle" font-size="28" font-weight="600" fill="#7b2e1e">Wall</text>'
    )
    wall_count = "?" if wall.get("count") is None else str(int(wall["count"]))
    svg_parts.append(
        f'<text x="{w_tx:.1f}" y="{w_ty + 8:.1f}" text-anchor="middle" font-size="22" font-weight="700" fill="#ffffff">{wall_count}</text>'
    )

    supply = snapshot["objects"]["brick_supply"]
    supply_rect = _rotated_rect(
        float(supply.get("x_mm", 0.0)),
        float(supply.get("y_mm", 0.0)),
        float(supply.get("theta_deg", 0.0)),
        54.0,
        54.0,
    )
    supply_screen = [project(x, y) for x, y in supply_rect]
    supply_stroke = "#f3c548" if active_name == "brick_supply" else "#18494e"
    supply_stroke_width = "4.5" if active_name == "brick_supply" else "3"
    svg_parts.append(
        f'<polygon points="{_polygon_points(supply_screen)}" fill="#63f2f7" stroke="{supply_stroke}" stroke-width="{supply_stroke_width}" />'
    )
    supply_tx, supply_ty = project(float(supply.get("x_mm", 0.0)), float(supply.get("y_mm", 0.0)))
    svg_parts.append(f'<text x="{supply_tx:.1f}" y="{supply_ty - 46:.1f}" text-anchor="middle" font-size="36" font-weight="600" fill="#18494e">Supply</text>')
    count_text = "?" if supply.get("count") is None else str(int(supply["count"]))
    svg_parts.append(
        f'<text x="{supply_tx:.1f}" y="{supply_ty + 16:.1f}" text-anchor="middle" font-size="44" font-weight="700" fill="#0d2b32">{count_text}</text>'
    )

    robot = snapshot["robot"]
    target_obj = (snapshot.get("objects") or {}).get(active_name) if active_name in {"wall", "brick_supply"} else None
    if isinstance(target_obj, dict):
        ax, ay = project(float(robot.get("x_mm", 0.0)), float(robot.get("y_mm", 0.0)))
        bx, by = project(float(target_obj.get("x_mm", 0.0)), float(target_obj.get("y_mm", 0.0)))
        svg_parts.append(
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#7b7668" stroke-width="1.8" stroke-dasharray="8 7" />'
        )
    current_trend = "unknown"
    current_dot_color = "#2c6fbb"
    if step_history:
        svg_parts.append('<g id="step-history">')
        history_points = [
            project(float(entry.get("x_mm", 0.0)), float(entry.get("y_mm", 0.0)))
            for entry in step_history
        ]
        if len(history_points) >= 2:
            svg_parts.append(
                f'<polyline class="step-history-path" points="{_polygon_points(history_points)}" '
                'fill="none" stroke="#b7aea0" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round" opacity="0.85" />'
            )
        previous_entry = None
        history_count = len(step_history)
        for idx, entry in enumerate(step_history):
            trend = _step_history_trend(previous_entry, entry)
            color = _step_history_color(trend)
            hx, hy = project(float(entry.get("x_mm", 0.0)), float(entry.get("y_mm", 0.0)))
            radius = _history_dot_radius(idx, history_count)
            opacity = 0.5 + 0.4 * ((float(idx) + 1.0) / float(history_count))
            svg_parts.append(
                f'<circle class="step-history-dot" data-trend="{trend}" cx="{hx:.1f}" cy="{hy:.1f}" '
                f'r="{radius:.1f}" fill="{color}" fill-opacity="{opacity:.2f}" '
                'stroke="#f9f6ef" stroke-width="2" />'
            )
            previous_entry = entry
            current_trend = trend
            current_dot_color = color
        svg_parts.append("</g>")
    hx, hy = _heading_vector(float(robot.get("theta_deg", 0.0)))
    camera_x = float(robot.get("x_mm", 0.0))
    camera_y = float(robot.get("y_mm", 0.0))
    stem_tail_x = camera_x - hx * float(ROBOT_NOSE_TO_TAIL_MM)
    stem_tail_y = camera_y - hy * float(ROBOT_NOSE_TO_TAIL_MM)
    stem_x2, stem_y2 = project(camera_x, camera_y)
    dot_radius = max(6.0, 13.0 * scale / 100.0)
    svg_parts.append('<g id="camera-glyph">')
    svg_parts.append(
        f'<circle class="camera-dot" data-trend="{current_trend}" cx="{stem_x2:.1f}" cy="{stem_y2:.1f}" '
        f'r="{dot_radius:.1f}" fill="{current_dot_color}" stroke="#233843" stroke-width="2.2" />'
    )
    svg_parts.append("</g>")
    
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
    
    # Distance to the other stack (non-active target)
    other_stack_name = "wall" if active_name == "brick_supply" else "brick_supply"
    other_stack = snapshot["objects"].get(other_stack_name)
    if isinstance(other_stack, dict):
        other_x = float(other_stack.get("x_mm", 0.0))
        other_y = float(other_stack.get("y_mm", 0.0))
        dist_to_other = math.hypot(other_x - robot_x, other_y - robot_y)
        
        # Position text near the other stack
        other_text_x, other_text_y = project(other_x, other_y)
        other_label = "Wall" if other_stack_name == "wall" else "Supply"
        
        svg_parts.append(
            f'<text x="{other_text_x:.1f}" y="{other_text_y + 58:.1f}" text-anchor="middle" font-size="14" font-weight="500" fill="#5d676e" stroke="#f9f6ef" stroke-width="2" paint-order="stroke">~{dist_to_other:.0f} to {other_label}</text>'
        )

    if bool(snapshot["held_brick"].get("held", False)):
        hx, hy = _heading_vector(float(robot.get("theta_deg", 0.0)))
        brick_center_x = float(robot.get("x_mm", 0.0)) + hx * 20.0
        brick_center_y = float(robot.get("y_mm", 0.0)) + hy * 20.0
        held_rect = _rotated_rect(
            brick_center_x,
            brick_center_y,
            float(robot.get("theta_deg", 0.0)),
            float(snapshot["held_brick"].get("length_mm", DEFAULT_BRICK_MM["length_mm"])),
            float(snapshot["held_brick"].get("width_mm", DEFAULT_BRICK_MM["width_mm"])),
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
    snapshot = _deepcopy(state or _default_workspace_state(render_enabled=False))
    mast_history = _current_step_mast_history(snapshot)
    objects = snapshot.get("objects") or {}
    supply = objects.get("brick_supply") or {}
    wall = objects.get("wall") or {}
    leia = snapshot.get("leia") or {}
    last_visible = snapshot.get("last_visible_brick") or {}

    history_heights = [
        float(_coerce_float(entry.get("camera_height_mm"), 0.0) or 0.0)
        for entry in mast_history
    ]
    max_height_mm = max(
        160.0,
        float(_coerce_float(leia.get("z_mm"), DEFAULT_CAMERA_Z_MM) or DEFAULT_CAMERA_Z_MM),
        float(_coerce_float(supply.get("height_mm"), 0.0) or 0.0),
        float(_coerce_float(wall.get("height_mm"), 0.0) or 0.0),
        *(history_heights or [0.0]),
    ) + 28.0
    plot_left = 86.0
    plot_right = float(SVG_WIDTH) - 86.0
    plot_top = 72.0
    plot_bottom = float(MAST_SVG_HEIGHT) - 44.0
    plot_height = max(1.0, plot_bottom - plot_top)
    supply_x = 216.0
    camera_x = float(SVG_WIDTH) * 0.5
    wall_x = float(SVG_WIDTH) - 216.0
    stack_width = 92.0
    trail_half_span_px = 108.0
    y_axis_values = [
        abs(float(_coerce_float(entry.get("y_axis_mm"), 0.0) or 0.0))
        for entry in mast_history
        if _coerce_float(entry.get("y_axis_mm"), None) is not None
    ]
    current_y_axis_mm = _coerce_float(last_visible.get("y_axis_mm"), None)
    if current_y_axis_mm is not None:
        y_axis_values.append(abs(float(current_y_axis_mm)))
    max_y_axis_mm = max(16.0, *(y_axis_values or [0.0]))

    def project_height(height_mm: float) -> float:
        clamped = max(0.0, float(_coerce_float(height_mm, 0.0) or 0.0))
        return plot_bottom - (clamped / float(max_height_mm)) * plot_height

    def project_y_axis_mm(y_axis_mm: float | None) -> float:
        if y_axis_mm is None:
            return camera_x
        ratio = max(-1.0, min(1.0, float(y_axis_mm) / float(max_y_axis_mm)))
        return camera_x + ratio * trail_half_span_px

    def stack_rect(x_center: float, height_mm: float, fill: str, stroke: str, label: str, count: int | None) -> list[str]:
        top_y = project_height(height_mm)
        rect_height = max(6.0, plot_bottom - top_y)
        count_text = "?" if count is None else str(int(count))
        return [
            f'<rect x="{x_center - stack_width * 0.5:.1f}" y="{top_y:.1f}" width="{stack_width:.1f}" height="{rect_height:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="3" rx="16" />',
            f'<text x="{x_center:.1f}" y="{top_y - 16.0:.1f}" text-anchor="middle" font-size="28" font-weight="600" fill="{stroke}">{label}</text>',
            f'<text x="{x_center:.1f}" y="{top_y + rect_height * 0.5 + 8.0:.1f}" text-anchor="middle" font-size="26" font-weight="700" fill="#ffffff">{count_text}</text>',
        ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{MAST_SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {MAST_SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        f'<rect x="22" y="22" width="936" height="{MAST_SVG_HEIGHT - 44}" rx="24" fill="#f9f6ef" stroke="#d7d1c4" stroke-width="1.5" />',
    ]
    title_x = SVG_WIDTH * 0.5
    svg_parts.append(f'<text x="{title_x:.1f}" y="52.0" text-anchor="middle" font-size="28" font-weight="900" fill="#233843">Mast View</text>')
    svg_parts.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{plot_bottom:.1f}" stroke="#c8c2b5" stroke-width="3" stroke-linecap="round" />'
    )
    svg_parts.extend(
        stack_rect(
            supply_x,
            float(_coerce_float(supply.get("height_mm"), 0.0) or 0.0),
            "#63f2f7",
            "#18494e",
            "Supply",
            _coerce_int(supply.get("count"), None),
        )
    )
    svg_parts.extend(
        stack_rect(
            wall_x,
            float(_coerce_float(wall.get("height_mm"), 0.0) or 0.0),
            "#c2523c",
            "#7b2e1e",
            "Wall",
            _coerce_int(wall.get("count"), None),
        )
    )

    current_trend = "unknown"
    current_dot_color = "#2c6fbb"
    if mast_history:
        svg_parts.append('<g id="mast-history">')
        history_points = [
            (
                project_y_axis_mm(_coerce_float(entry.get("y_axis_mm"), None)),
                project_height(float(_coerce_float(entry.get("camera_height_mm"), 0.0) or 0.0)),
            )
            for entry in mast_history
        ]
        if len(history_points) >= 2:
            svg_parts.append(
                f'<polyline class="mast-history-path" points="{_polygon_points(history_points)}" '
                'fill="none" stroke="#b7aea0" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round" opacity="0.85" />'
            )
        previous_entry = None
        history_count = len(mast_history)
        for idx, entry in enumerate(mast_history):
            trend = _mast_history_trend(previous_entry, entry)
            color = _step_history_color(trend)
            hx = project_y_axis_mm(_coerce_float(entry.get("y_axis_mm"), None))
            hy = project_height(float(_coerce_float(entry.get("camera_height_mm"), 0.0) or 0.0))
            radius = _history_dot_radius(idx, history_count)
            opacity = 0.5 + 0.4 * ((float(idx) + 1.0) / float(history_count))
            svg_parts.append(
                f'<circle class="mast-history-dot" data-trend="{trend}" cx="{hx:.1f}" cy="{hy:.1f}" '
                f'r="{radius:.1f}" fill="{color}" fill-opacity="{opacity:.2f}" '
                'stroke="#f9f6ef" stroke-width="2" />'
            )
            previous_entry = entry
            current_trend = trend
            current_dot_color = color
        svg_parts.append("</g>")

    current_camera_x = project_y_axis_mm(current_y_axis_mm)
    current_camera_y = project_height(float(_coerce_float(leia.get("z_mm"), DEFAULT_CAMERA_Z_MM) or DEFAULT_CAMERA_Z_MM))
    current_dot_radius = 8.0
    svg_parts.append(
        f'<circle class="mast-camera-dot" data-trend="{current_trend}" cx="{current_camera_x:.1f}" cy="{current_camera_y:.1f}" '
        f'r="{current_dot_radius:.1f}" fill="{current_dot_color}" stroke="#233843" stroke-width="2.2" />'
    )
    if current_y_axis_mm is not None:
        svg_parts.append(
            f'<text x="{current_camera_x:.1f}" y="{current_camera_y - 14.0:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#233843">y={float(current_y_axis_mm):+.0f}mm</text>'
        )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _summary_payload(state: dict) -> dict:
    plan = _plan_reverse_then_turn_from_state(state, turn_cmd="l")
    return {
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


def render_workspace_html(state: dict | None) -> str:
    snapshot = _deepcopy(state or _default_workspace_state(render_enabled=False))
    svg = render_workspace_svg(snapshot)
    mast_svg = render_mast_svg(snapshot)
    summary_json = json.dumps(_summary_payload(snapshot), indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leia Workspace XYZ</title>
  <style>
    :root {{
      --bg: #efe8d9;
      --panel: rgba(255,255,255,0.9);
      --ink: #10202b;
      --muted: #5d676e;
      --line: rgba(16,32,43,0.12);
    }}
    html, body {{
      margin: 0;
      min-height: 100%;
      background: radial-gradient(circle at top, #f8f2e5 0%, var(--bg) 72%);
      color: var(--ink);
      font-family: "Segoe UI", sans-serif;
    }}
    body {{
      display: grid;
      place-items: center;
      padding: 18px;
      box-sizing: border-box;
    }}
    .shell {{
      width: min(1320px, 100%);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 18px;
    }}
    .stage-stack {{
      display: grid;
      gap: 18px;
    }}
    .stage, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 20px 48px rgba(16,32,43,0.09);
      overflow: hidden;
    }}
    .stage {{
      padding: 12px;
    }}
    .panel {{
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    pre {{
      margin: 0;
      max-height: 560px;
      overflow: auto;
      padding: 12px;
      border-radius: 14px;
      background: rgba(16,32,43,0.05);
      border: 1px solid rgba(16,32,43,0.08);
      font-size: 12px;
      line-height: 1.38;
    }}
    @media (max-width: 1080px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="stage-stack">
      <div class="stage">{svg}</div>
      <div class="stage">{mast_svg}</div>
    </div>
    <div class="panel">
      <div>
        <h1>Leia Workspace XYZ</h1>
        <p>Live workspace snapshot. Open this file once and it will refresh every 2 seconds while the helper rewrites it.</p>
      </div>
      <div>
        <p><strong>Files</strong></p>
        <p>{html.escape(str(LIVE_HTML_PATH))}</p>
        <p>{html.escape(str(LIVE_SVG_PATH))}</p>
        <p>{html.escape(str(LIVE_MAST_SVG_PATH))}</p>
        <p>{html.escape(str(LIVE_JSON_PATH))}</p>
      </div>
      <pre>{html.escape(summary_json)}</pre>
    </div>
  </div>
</body>
</html>
"""
def _write_live_assets(state: dict) -> None:
    try:
        XYZ_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. HTML
        html_content = render_workspace_html(state)
        with open(LIVE_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
            
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
