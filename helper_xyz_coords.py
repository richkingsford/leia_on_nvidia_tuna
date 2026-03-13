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
LIVE_HTML_PATH = XYZ_LAYOUT_DIR / "xyz_workspace_live.html"
LIVE_JSON_PATH = XYZ_LAYOUT_DIR / "xyz_workspace_live.json"
LIVE_SVG_PATH = XYZ_LAYOUT_DIR / "xyz_workspace_live.svg"
SCHEMA_VERSION = 1

DEFAULT_CAMERA_Z_MM = 145.0
DEFAULT_BRICK_SUPPLY_POS_MM = {"x_mm": 140.0, "y_mm": 25.0, "z_mm": 0.0}
DEFAULT_WALL_POS_MM = {"x_mm": -130.0, "y_mm": 0.0, "z_mm": 0.0, "theta_deg": 180.0}
DEFAULT_WALL_RENDER_LENGTH_MM = 320.0
DEFAULT_BRICK_MM = {"length_mm": 44.0, "width_mm": 22.0, "height_mm": 22.0}

SVG_WIDTH = 980
SVG_HEIGHT = 680
SVG_PADDING = 80

_NO_WRITE = object()


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


def default_live_render_enabled() -> bool:
    env_value = os.environ.get("LEIA_XYZ_RENDER")
    if env_value is not None:
        return str(env_value).strip().lower() not in {"0", "false", "no", "off"}
    if "unittest" in sys.modules or "pytest" in sys.modules:
        return False
    return XYZ_LAYOUT_DIR.exists()


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
            "confidence": None,
            "visible": False,
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
            "confidence": None,
            "visible": False,
            "valid": False,
            "source": "default_layout_seed",
            "last_seen_ts": None,
        },
    }


def _default_workspace_state(*, render_enabled: bool) -> dict:
    return {
        "schema_version": int(SCHEMA_VERSION),
        "updated_at": time.time(),
        "render_enabled": bool(render_enabled),
        "live_paths": {
            "html": str(LIVE_HTML_PATH),
            "json": str(LIVE_JSON_PATH),
            "svg": str(LIVE_SVG_PATH),
        },
        "robot": {
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
        },
        "history": [],
    }


def ensure_workspace(world, *, render_enabled: bool | None = None) -> dict:
    state = getattr(world, "_xyz_workspace", None)
    if isinstance(state, dict):
        if render_enabled is not None:
            state["render_enabled"] = bool(render_enabled)
        return state
    state = _default_workspace_state(
        render_enabled=default_live_render_enabled() if render_enabled is None else bool(render_enabled)
    )
    setattr(world, "_xyz_workspace", state)
    return state


def workspace_snapshot(world) -> dict | None:
    state = getattr(world, "_xyz_workspace", None)
    if not isinstance(state, dict):
        return None
    return _deepcopy(state)


def _append_history(state: dict, entry: dict, *, maxlen: int = 60) -> None:
    history = state.setdefault("history", [])
    history.append(dict(entry))
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


def _sync_robot_pose(state: dict, world) -> None:
    robot = state["robot"]
    robot["x_mm"] = float(_coerce_float(getattr(world, "x", 0.0), 0.0) or 0.0)
    robot["y_mm"] = float(_coerce_float(getattr(world, "y", 0.0), 0.0) or 0.0)
    robot["theta_deg"] = float(_coerce_float(getattr(world, "theta", 0.0), 0.0) or 0.0)
    robot["lift_mm"] = float(_coerce_float(getattr(world, "lift_height", 0.0), 0.0) or 0.0)
    robot["z_mm"] = 0.0

    leia = state["leia"]
    leia["x_mm"] = float(robot["x_mm"])
    leia["y_mm"] = float(robot["y_mm"])
    leia["theta_deg"] = float(robot["theta_deg"])
    camera_height = _coerce_float(getattr(world, "height_mm", None), None)
    if camera_height is None:
        camera_height = float(DEFAULT_CAMERA_Z_MM) + float(robot["lift_mm"])
    leia["z_mm"] = float(camera_height)


def _sync_known_objects(state: dict, world) -> None:
    objects = state["objects"]
    supply = objects["brick_supply"]
    wall = objects["wall"]

    supply_count = _coerce_int(getattr(world, "brick_supply_height_bricks", None), None)
    if supply_count is not None:
        supply["count"] = int(supply_count)

    brick = getattr(world, "brick", None) or {}
    held = bool(brick.get("held", False))
    state["held_brick"]["held"] = held
    state["last_visible_brick"] = {
        "visible": bool(brick.get("visible", False)),
        "dist_mm": _coerce_float(brick.get("dist"), None),
        "x_axis_mm": _coerce_float(brick.get("x_axis", brick.get("offset_x")), None),
        "y_axis_mm": _coerce_float(brick.get("y_axis", brick.get("offset_y")), None),
        "confidence": _coerce_float(brick.get("confidence"), None),
    }

    wall_state = getattr(world, "wall", None) or {}
    origin = wall_state.get("origin") if isinstance(wall_state, dict) else None
    if isinstance(origin, dict):
        x_mm = _coerce_float(origin.get("x"), None)
        y_mm = _coerce_float(origin.get("y"), None)
        if x_mm is not None and y_mm is not None:
            wall["x_mm"] = float(x_mm)
            wall["y_mm"] = float(y_mm)
            wall["theta_deg"] = float(_coerce_float(origin.get("theta"), wall.get("theta_deg", 180.0)) or wall.get("theta_deg", 180.0))
            wall["visible"] = bool(wall_state.get("last_seen"))
            wall["valid"] = bool(wall_state.get("valid", True))
            wall["source"] = wall_state.get("source") or wall.get("source")
            wall["last_seen_ts"] = _coerce_float(wall_state.get("last_seen"), wall.get("last_seen_ts"))
    else:
        wall["valid"] = bool(wall_state.get("valid", False))


def sync_from_world(world, *, reason: str = "sync", render=_NO_WRITE) -> dict:
    state = ensure_workspace(world)
    _sync_robot_pose(state, world)
    _sync_known_objects(state, world)
    state["updated_at"] = time.time()
    _append_history(state, {"type": "sync", "reason": str(reason), "ts": state["updated_at"]})
    if render is _NO_WRITE:
        render = bool(state.get("render_enabled", False))
    if bool(render):
        write_live_files(world)
    return state


def update_from_motion(world, *, event=None, delta=None, render=_NO_WRITE) -> dict:
    state = sync_from_world(world, reason="motion", render=False)
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
    if render is _NO_WRITE:
        render = bool(state.get("render_enabled", False))
    if bool(render):
        write_live_files(world)
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
    render=_NO_WRITE,
    source: str = "manual_observation",
) -> dict:
    state = sync_from_world(world, reason=f"observe_{object_name}", render=False)
    objects = state["objects"]
    key = str(object_name or "").strip().lower()
    if key not in objects:
        raise KeyError(f"Unknown workspace object '{object_name}'")
    obj = objects[key]
    robot = state["robot"]
    heading_deg = float(robot.get("theta_deg", 0.0)) + float(_coerce_float(bearing_deg, 0.0) or 0.0)
    hx, hy = _heading_vector(heading_deg)
    distance_val = max(0.0, float(_coerce_float(distance_mm, 0.0) or 0.0))
    obj["x_mm"] = float(robot["x_mm"]) + distance_val * hx
    obj["y_mm"] = float(robot["y_mm"]) + distance_val * hy
    if theta_deg is not None:
        obj["theta_deg"] = float(theta_deg)
    if count is not None and key == "brick_supply":
        obj["count"] = int(count)
    obj["confidence"] = _coerce_float(confidence, obj.get("confidence"))
    obj["visible"] = True
    obj["source"] = str(source)
    obj["last_seen_ts"] = time.time()
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
    if render is _NO_WRITE:
        render = bool(state.get("render_enabled", False))
    if bool(render):
        write_live_files(world)
    return state


def observe_brick_supply(
    world,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
    count: int | None = None,
    confidence: float | None = None,
    render=_NO_WRITE,
    source: str = "brick_supply_observation",
) -> dict:
    return observe_object(
        world,
        "brick_supply",
        distance_mm=distance_mm,
        bearing_deg=bearing_deg,
        count=count,
        confidence=confidence,
        render=render,
        source=source,
    )


def observe_wall(
    world,
    *,
    distance_mm: float,
    bearing_deg: float = 0.0,
    theta_deg: float | None = None,
    confidence: float | None = None,
    render=_NO_WRITE,
    source: str = "wall_observation",
) -> dict:
    return observe_object(
        world,
        "wall",
        distance_mm=distance_mm,
        bearing_deg=bearing_deg,
        theta_deg=theta_deg,
        confidence=confidence,
        render=render,
        source=source,
    )


def set_brick_supply_count(world, count: int | None, *, render=_NO_WRITE, source: str = "manual_count") -> dict:
    state = sync_from_world(world, reason="set_brick_supply_count", render=False)
    supply = state["objects"]["brick_supply"]
    supply["count"] = _coerce_int(count, None)
    supply["source"] = str(source)
    state["updated_at"] = time.time()
    _append_history(
        state,
        {"type": "brick_supply_count", "count": supply["count"], "source": str(source), "ts": state["updated_at"]},
    )
    if render is _NO_WRITE:
        render = bool(state.get("render_enabled", False))
    if bool(render):
        write_live_files(world)
    return state


def set_holding_brick(world, held: bool, *, render=_NO_WRITE, source: str = "manual_hold_state") -> dict:
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
    if render is _NO_WRITE:
        render = bool(state.get("render_enabled", False))
    if bool(render):
        write_live_files(world)
    return state


def reconcile_object_distance(
    world,
    object_name: str,
    observed_dist_mm: float,
    *,
    bearing_deg: float | None = None,
    render=_NO_WRITE,
    source: str = "distance_reconciliation",
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
        render=render,
        source=source,
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
) -> dict:
    state = sync_from_world(world, reason="plan_reverse_then_turn", render=False)
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
    for obj in state["objects"].values():
        points.append((float(obj.get("x_mm", 0.0)), float(obj.get("y_mm", 0.0))))
        if obj.get("name") == "wall":
            theta = float(obj.get("theta_deg", 180.0)) + 90.0
            dx = math.cos(math.radians(theta)) * float(obj.get("length_mm", DEFAULT_WALL_RENDER_LENGTH_MM)) * 0.5
            dy = math.sin(math.radians(theta)) * float(obj.get("length_mm", DEFAULT_WALL_RENDER_LENGTH_MM)) * 0.5
            cx = float(obj.get("x_mm", 0.0))
            cy = float(obj.get("y_mm", 0.0))
            points.extend([(cx - dx, cy - dy), (cx + dx, cy + dy)])
    return points


def _build_viewbox(state: dict) -> tuple[float, float, float, float]:
    points = _viewport_points(state)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs) - 120.0
    max_x = max(xs) + 120.0
    min_y = min(ys) - 120.0
    max_y = max(ys) + 120.0
    if (max_x - min_x) < 320.0:
        mid_x = (max_x + min_x) * 0.5
        min_x = mid_x - 160.0
        max_x = mid_x + 160.0
    if (max_y - min_y) < 260.0:
        mid_y = (max_y + min_y) * 0.5
        min_y = mid_y - 130.0
        max_y = mid_y + 130.0
    return min_x, max_x, min_y, max_y


def _project_fn(state: dict):
    min_x, max_x, min_y, max_y = _build_viewbox(state)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min((SVG_WIDTH - 2 * SVG_PADDING) / span_x, (SVG_HEIGHT - 2 * SVG_PADDING) / span_y)

    def project(x_mm: float, y_mm: float) -> tuple[float, float]:
        sx = SVG_PADDING + (float(x_mm) - min_x) * scale
        sy = SVG_HEIGHT - SVG_PADDING - (float(y_mm) - min_y) * scale
        return sx, sy

    return project, scale


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


def _robot_points(robot: dict) -> list[tuple[float, float]]:
    x_mm = float(robot.get("x_mm", 0.0))
    y_mm = float(robot.get("y_mm", 0.0))
    theta = float(robot.get("theta_deg", 0.0))
    hx, hy = _heading_vector(theta)
    px, py = -hy, hx
    nose = (x_mm + hx * 40.0, y_mm + hy * 40.0)
    back_left = (x_mm - hx * 28.0 + px * 22.0, y_mm - hy * 28.0 + py * 22.0)
    back_center = (x_mm - hx * 18.0, y_mm - hy * 18.0)
    back_right = (x_mm - hx * 28.0 - px * 22.0, y_mm - hy * 28.0 - py * 22.0)
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


def render_workspace_svg(state: dict | None) -> str:
    snapshot = _deepcopy(state or _default_workspace_state(render_enabled=False))
    project, scale = _project_fn(snapshot)
    min_x, max_x, min_y, max_y = _build_viewbox(snapshot)
    grid_x = _grid_lines(min_x, max_x, 50.0)
    grid_y = _grid_lines(min_y, max_y, 50.0)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        '<rect x="22" y="22" width="936" height="636" rx="24" fill="#f9f6ef" stroke="#d7d1c4" stroke-width="1.5" />',
        '<g id="grid">',
    ]
    for x_mm in grid_x:
        x1, y1 = project(x_mm, min_y)
        x2, y2 = project(x_mm, max_y)
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d9d6cf" stroke-width="1" />'
        )
    for y_mm in grid_y:
        x1, y1 = project(min_x, y_mm)
        x2, y2 = project(max_x, y_mm)
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d9d6cf" stroke-width="1" />'
        )
    svg_parts.append("</g>")

    ox1, oy1 = project(0.0, min_y)
    ox2, oy2 = project(0.0, max_y)
    oyx1, oyy1 = project(min_x, 0.0)
    oyx2, oyy2 = project(max_x, 0.0)
    svg_parts.append(f'<line x1="{ox1:.1f}" y1="{oy1:.1f}" x2="{ox2:.1f}" y2="{oy2:.1f}" stroke="#8c877b" stroke-width="1.8" />')
    svg_parts.append(f'<line x1="{oyx1:.1f}" y1="{oyy1:.1f}" x2="{oyx2:.1f}" y2="{oyy2:.1f}" stroke="#8c877b" stroke-width="1.8" />')

    wall = snapshot["objects"]["wall"]
    wall_cx = float(wall.get("x_mm", 0.0))
    wall_cy = float(wall.get("y_mm", 0.0))
    wall_dir_deg = float(wall.get("theta_deg", 180.0)) + 90.0
    wall_dx = math.cos(math.radians(wall_dir_deg)) * float(wall.get("length_mm", DEFAULT_WALL_RENDER_LENGTH_MM)) * 0.5
    wall_dy = math.sin(math.radians(wall_dir_deg)) * float(wall.get("length_mm", DEFAULT_WALL_RENDER_LENGTH_MM)) * 0.5
    wx1, wy1 = project(wall_cx - wall_dx, wall_cy - wall_dy)
    wx2, wy2 = project(wall_cx + wall_dx, wall_cy + wall_dy)
    wall_stroke = "#c2523c" if bool(wall.get("valid", False)) else "#d4a090"
    svg_parts.append(
        f'<line x1="{wx1:.1f}" y1="{wy1:.1f}" x2="{wx2:.1f}" y2="{wy2:.1f}" stroke="{wall_stroke}" stroke-width="8" stroke-linecap="round" />'
    )
    wall_tx, wall_ty = project(wall_cx, wall_cy)
    svg_parts.append(
        f'<text x="{wall_tx:.1f}" y="{wall_ty - 16:.1f}" text-anchor="middle" font-size="20" font-weight="700" fill="#8a2f22">Wall</text>'
    )

    supply = snapshot["objects"]["brick_supply"]
    supply_rect = _rotated_rect(
        float(supply.get("x_mm", 0.0)),
        float(supply.get("y_mm", 0.0)),
        0.0,
        54.0,
        54.0,
    )
    supply_screen = [project(x, y) for x, y in supply_rect]
    svg_parts.append(
        f'<polygon points="{_polygon_points(supply_screen)}" fill="#63f2f7" stroke="#18494e" stroke-width="3" />'
    )
    supply_tx, supply_ty = project(float(supply.get("x_mm", 0.0)), float(supply.get("y_mm", 0.0)))
    svg_parts.append(
        f'<text x="{supply_tx:.1f}" y="{supply_ty + 8:.1f}" text-anchor="middle" font-size="22" font-weight="700" fill="#0d2b32">{int(supply["count"]) if supply.get("count") is not None else "?"}</text>'
    )
    svg_parts.append(
        f'<text x="{supply_tx:.1f}" y="{supply_ty + 42:.1f}" text-anchor="middle" font-size="18" font-weight="700" fill="#18494e">Brick Supply</text>'
    )

    robot = snapshot["robot"]
    robot_screen = [project(x, y) for x, y in _robot_points(robot)]
    svg_parts.append(
        f'<polygon points="{_polygon_points(robot_screen)}" fill="#f0a43e" stroke="#233843" stroke-width="3.5" />'
    )
    rx, ry = project(float(robot.get("x_mm", 0.0)), float(robot.get("y_mm", 0.0)))
    svg_parts.append(f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="{max(3.5, 9.0 * scale / 100.0):.1f}" fill="#233843" />')

    leia = snapshot["leia"]
    lx, ly = project(float(leia.get("x_mm", 0.0)), float(leia.get("y_mm", 0.0)))
    svg_parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6.5" fill="#2c6fbb" stroke="#0d2b32" stroke-width="1.5" />')
    svg_parts.append(
        f'<text x="{lx + 12:.1f}" y="{ly - 12:.1f}" font-size="18" font-weight="700" fill="#1f4d80">Leia</text>'
    )

    if bool(snapshot["held_brick"].get("held", False)):
        hx, hy = _heading_vector(float(robot.get("theta_deg", 0.0)))
        brick_center_x = float(robot.get("x_mm", 0.0)) + hx * 48.0
        brick_center_y = float(robot.get("y_mm", 0.0)) + hy * 48.0
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
        held_tx, held_ty = project(brick_center_x, brick_center_y)
        svg_parts.append(
            f'<text x="{held_tx:.1f}" y="{held_ty - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#18494e">Held Brick</text>'
        )

    plan = _plan_reverse_then_turn_from_state(snapshot, turn_cmd="l")
    legend_x = 655
    legend_y = 64
    svg_parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="270" height="250" rx="18" fill="rgba(255,255,255,0.88)" stroke="#d7d1c4" stroke-width="1.5" />',
            f'<text x="{legend_x + 18}" y="{legend_y + 28}" font-size="22" font-weight="700" fill="#10202b">Workspace</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 58}" font-size="15" fill="#10202b">Robot: x={robot["x_mm"]:.1f} y={robot["y_mm"]:.1f} theta={robot["theta_deg"]:.1f}</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 82}" font-size="15" fill="#10202b">Leia z={leia["z_mm"]:.1f}mm</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 106}" font-size="15" fill="#10202b">Brick Supply count={supply.get("count") if supply.get("count") is not None else "?"}</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 130}" font-size="15" fill="#10202b">Held Brick={"YES" if snapshot["held_brick"].get("held") else "NO"}</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 154}" font-size="15" fill="#10202b">Wall behind={max(0.0, -_relative_pose(snapshot["robot"], snapshot["objects"]["wall"]).get("forward_mm", 0.0)):.1f}mm</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 186}" font-size="15" font-weight="700" fill="#8a2f22">Wall Plan</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 210}" font-size="15" fill="#10202b">reverse acts={int(plan.get("reverse_acts", 0) or 0)} then {str((plan.get("actions") or [{}])[-1].get("cmd", "-")).upper()}</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 234}" font-size="13" fill="#55626b">Auto-refresh HTML rewrites from motion/vision syncs.</text>',
        ]
    )

    svg_parts.append(
        '<text x="60" y="48" font-size="26" font-weight="700" fill="#10202b">Leia Workspace XYZ</text>'
    )
    svg_parts.append(
        '<text x="60" y="72" font-size="14" fill="#55626b">Top-down mm map. +X points right, +Y points up. Z is shown in labels.</text>'
    )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _summary_payload(state: dict) -> dict:
    plan = _plan_reverse_then_turn_from_state(state, turn_cmd="l")
    return {
        "updated_at": float(state.get("updated_at", time.time())),
        "robot": _deepcopy(state.get("robot", {})),
        "leia": _deepcopy(state.get("leia", {})),
        "objects": _deepcopy(state.get("objects", {})),
        "held_brick": _deepcopy(state.get("held_brick", {})),
        "wall_reverse_plan": plan,
        "last_visible_brick": _deepcopy(state.get("last_visible_brick", {})),
    }


def render_workspace_html(state: dict | None) -> str:
    snapshot = _deepcopy(state or _default_workspace_state(render_enabled=False))
    svg = render_workspace_svg(snapshot)
    summary_json = json.dumps(_summary_payload(snapshot), indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="2">
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
    <div class="stage">{svg}</div>
    <div class="panel">
      <div>
        <h1>Leia Workspace XYZ</h1>
        <p>Live workspace snapshot. Open this file once and it will refresh every 2 seconds while the helper rewrites it.</p>
      </div>
      <div>
        <p><strong>Files</strong></p>
        <p>{html.escape(str(LIVE_HTML_PATH))}</p>
        <p>{html.escape(str(LIVE_SVG_PATH))}</p>
        <p>{html.escape(str(LIVE_JSON_PATH))}</p>
      </div>
      <pre>{html.escape(summary_json)}</pre>
    </div>
  </div>
</body>
</html>
"""


def write_live_files(world) -> dict:
    state = ensure_workspace(world)
    state["updated_at"] = time.time()
    XYZ_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _summary_payload(state)
    LIVE_JSON_PATH.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    LIVE_SVG_PATH.write_text(render_workspace_svg(state), encoding="utf-8")
    LIVE_HTML_PATH.write_text(render_workspace_html(state), encoding="utf-8")
    return snapshot
