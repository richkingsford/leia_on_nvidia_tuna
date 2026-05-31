#!/usr/bin/env python3
"""Run mast-frozen Step 1/Step 2 real-robot trials with a static progress site."""

from __future__ import annotations

import argparse
import html
import json
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np

import a_follow_the_brick as follow
from helper_brick_detector_yolo import BrickDetector
from helper_robot_control import Robot


DEFAULT_SITE_DIR = Path("runs/frozen_step12_site")
AXES = ("dist", "x", "y")
Y_LOCK_TARGET_MM: float | None = None
Y_LOCK_TOL_MM = 5.0
TARGET_GUARD_MAX_DIST_MM = 350.0
TARGET_GUARD_MAX_ABS_X_MM = 120.0
WHEEL_TURN_ACTION_CAP_MS: int | None = None
DIRECT_X_PULSE_MS = 35
DIRECT_X_PWM = 92
DIRECT_DIST_PULSE_MS = 55
DIRECT_RESET_DIST_MAX_PULSES = 8
DIRECT_STEP2_DIST_PULSE_MS = 55
DIRECT_STEP2_DIST_PWM = 98
DIRECT_STEP2_DIST_MAX_PULSES = 10
DIRECT_USE_TURN_CURVE = False
DIRECT_X_PRIMITIVE = "command"
DIRECT_X_ADAPTIVE_PULSE = False
DIRECT_STEP1_DIST_TOL_MM = 50.0
DIRECT_STEP1_X_TOL_MM = 7.0
HONEST_RESET_DIST_OFFSET_MIN_MM = 1.0
HONEST_RESET_DIST_OFFSET_MAX_MM = 5.0
HONEST_RESET_X_GAP_MIN_MM = 5.0
HONEST_RESET_X_GAP_MAX_MM = 20.0
HONEST_RESET_X_TARGET_GAP_MM = 12.0
HONEST_RESET_X_SNAP_MIN_MS = 210
HONEST_RESET_X_OPEN_MS = 120
HONEST_RESET_X_CONTRACT_MS = 90
DIRECT_X_SNAP_MIN_MS = 210
DIRECT_X_DIST_GUARD_MM = 14.0
DIRECT_X_MAX_DIST_DRIFT_MM = 16.0
DIRECT_X_MIN_IMPROVEMENT_MM = 0.35
DIRECT_X_STALL_RETRY_SLACK_MM = 1.0
DIRECT_X_SNAP_OUTSIDE_MAX_MM = 2.0
DIRECT_X_MAX_ABS_ERR_MM = 22.0
DIRECT_X_MIN_SAFE_CX_PX = 95.0
DIRECT_X_MAX_SAFE_CX_PX = 560.0
DIRECT_X_MAX_CX_JUMP_PX = 65.0
DIRECT_X_MAX_BOX_SCALE_JUMP = 0.40
GREEN_CONTOUR_DIST_SCALE_MM_PX = 19000.0
DIRECT_STABLE_READS = 1
DIRECT_POST_PULSE_EXTRA_SETTLE_S = 0.45
DIRECT_LIVE_OBSERVE_MOVES = True
DIRECT_LIVE_SAMPLE_S = 0.08
DIRECT_LIVE_FINAL_SETTLE_S = 0.0
DIRECT_LIVE_DIST_CRAWL_MS = 1000
DIRECT_LIVE_RESET_DIST_CRAWL_MS = 450
DIRECT_LIVE_STEP2_DIST_CRAWL_MS = 80
DIRECT_LIVE_X_CRAWL_MS = 700
DIRECT_STEP2_DIST_EARLY_STOP_MARGIN_MM = 10.0
DIRECT_STEP2_DIST_TARGET_MM = 185.0
DIRECT_STEP2_DIST_TOL_MM = 10.0
DIRECT_STEP2_DIST_LIVE_MAX_X_ERR_MM = 22.0
DIRECT_STEP2_FORWARD_X_PREBIAS_MM = 12.0
DIRECT_STEP1_PREALIGN_DIST_TOL_MM = 8.0

EXPERIMENT_LINES = {
    "strict-y": "Original Step 1/2 gates with the mast command physically blocked.",
    "short-reset": "Short reset reverse so Leia does not back far away before trying the same gates.",
    "balanced-short-reset": "Short reset plus wider detector tuning, with the same movement gates.",
    "balanced-visible-reset": "Reset only to the visible pose about 30mm behind Step 1.",
    "ignore-y-gates": "Mast frozen and Y ignored; kept only as a historical comparison.",
    "balanced-visible-yfree-soft": "Movement-only reset nudges with mast frozen, but using old relaxed gates.",
    "honest-locked-y": "Mast/Y frozen; reset near Step 1, then Step 1 and Step 2 must close real dist and X gaps.",
    "honest-locked-y-xflip": "Mast/Y frozen; positive X closes left, and reset starts nearer the Step 1 X target.",
    "honest-locked-y-xflip-resetloose": "Mast/Y frozen; positive X closes left, while reset only returns to the visible distance band.",
    "honest-locked-y-xfirst": "Mast/Y frozen; positive X closes left using X-first turns before distance polish.",
    "honest-locked-y-forward-xfirst": "Mast/Y frozen; X-first corrections are forward-only, and any reverse pursuit is stopped.",
    "honest-locked-y-forward-xfirst-distreset": "Mast/Y frozen; reset only stages distance, then forward-only X-first pursuit closes the gaps.",
    "honest-locked-y-forward-xfirst-preflight": "Mast/Y frozen; first creeps forward until vision is confident, then runs one forward-only X-first trial.",
    "honest-locked-y-micro-xfirst": "Mast/Y frozen; forward-only X-first pursuit uses tiny turn pulses to avoid overcommitting.",
    "honest-locked-y-micro-xfirst-gapguard": "Mast/Y frozen; tiny X-first pulses plus reset stops immediately on wrong-way distance changes.",
    "honest-locked-y-guarded-micro": "Mast/Y frozen; tiny forward-only X moves, with target-continuity and adjacent-stack guards.",
    "honest-locked-y-single-contour": "Mast/Y frozen; one visible green stack may drive a transparent contour reading, but adjacent stacks stop the trial.",
    "honest-locked-y-single-contour-right": "Mast/Y frozen; one visible green stack, hard-capped rightward X probes, and no reverse pursuit.",
    "honest-locked-y-single-contour-right-actioncap": "Mast/Y frozen; one visible green stack, rightward X probes capped at the robot action layer.",
    "honest-locked-y-direct-right": "Mast/Y frozen; direct controller uses one tiny right-turn pulse only if each read proves the gap shrank.",
    "honest-locked-y-direct-left": "Mast/Y frozen; direct controller uses one tiny left-turn pulse only if each read proves the gap shrank.",
    "honest-locked-y-direct-right-80": "Mast/Y frozen; direct controller uses 80ms right-turn probes with immediate wrong-way failure.",
    "honest-locked-y-direct-right-150": "Mast/Y frozen; direct controller uses 150ms right-turn probes with immediate wrong-way failure.",
    "honest-locked-y-direct-curve-right": "Mast/Y frozen; direct controller uses capped right turn-curve probes with immediate wrong-way failure.",
    "honest-locked-y-direct-curve-left": "Mast/Y frozen; direct controller uses capped left turn-curve probes with immediate wrong-way failure.",
    "honest-locked-y-direct-strict-spin-right-240": "Mast/Y frozen; calibrated contour distance must be tight before X steering; one 240ms right spin probe is allowed only while every read proves the gaps shrink.",
    "honest-locked-y-direct-contour19000-spin-right-240": "Mast/Y frozen; contour distance uses the calibrated 19000/height scale; one 240ms right spin probe is allowed only while every read proves the gaps shrink.",
    "honest-locked-y-direct-contour19000-spin-left-240": "Mast/Y frozen; contour distance uses the calibrated 19000/height scale; one 240ms left spin probe is allowed only while every read proves the gaps shrink.",
    "honest-locked-y-direct-contour19000-onewheel-right-120": "Mast/Y frozen; calibrated contour distance, then a one-tread right pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-contour19000-onewheel-left-120": "Mast/Y frozen; calibrated contour distance, then a one-tread left pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-contour19000-onewheel-left-240": "Mast/Y frozen; calibrated contour distance, then a stronger one-tread left pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-stablecontour-onewheel-left-240": "Mast/Y frozen; stable contour-only reads, then a stronger one-tread left pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-stablecontour-onewheel-right-240": "Mast/Y frozen; stable contour-only reads, then a stronger one-tread right pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-stablecontour-onewheel-right-360": "Mast/Y frozen; stable contour-only reads, then a 360ms one-tread right pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140": "Mast/Y frozen; stable contour-only reads, then a stronger-PWM one-tread right pivot probe is allowed only if it measurably shrinks X.",
    "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140": "Mast/Y frozen; stable contour-only reads ignore far-edge artifacts, then the stronger-PWM one-tread right pivot is guarded by real gap shrinkage.",
    "honest-locked-y-direct-stablecontour-adaptive-right-pwm140": "Mast/Y frozen; stable contour-only reads with edge filtering; X pulse length scales down near target to avoid overshoot.",
    "honest-locked-y-direct-adaptive-right-step2-strongdist": "Mast/Y frozen; keeps the adaptive X solution and gives Step 2 stronger forward distance pulses with wrong-way proof.",
    "honest-locked-y-direct-adaptive-right-resetlong-step2strong": "Mast/Y frozen; keeps adaptive X, gives reset more distance pulses, and gives Step 2 stronger forward distance pulses.",
    "honest-locked-y-direct-adaptive-right-resetstrong-step2strong": "Mast/Y frozen; honest reset, Step 1 distance polish, tiny repeated Step 2 crawls, capped X nudges, and drift recentering.",
    "honest-locked-y-direct-adaptive-right-step2-toggle": "Mast/Y frozen; each attempt starts from a centered Step 1 win pose, captures the prior image, pursues only Step 2 with adaptive polish, then returns to centered Step 1 for the next toggle.",
}


class MastFrozenRobot:
    """Robot proxy that guarantees mast commands are never sent to hardware."""

    def __init__(self, robot: Robot):
        self._robot = robot
        self.mast_attempts: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        cmd_key = str(cmd or "").strip().lower()
        if cmd_key in {"u", "d"}:
            event = {
                "kind": "send_command_pwm",
                "cmd": cmd_key,
                "pwm": int(pwm or 0),
                "duration_ms": int(duration_ms or 0),
                "blocked_reason": "mast_frozen",
            }
            self.mast_attempts.append(event)
            return {
                "cmd_sent": cmd_key,
                "pwm": 0,
                "duration_ms": 0,
                "skipped": True,
                "mast_frozen": True,
                "blocked": False,
                "reason": "mast_frozen",
            }
        send_duration = duration_ms
        if WHEEL_TURN_ACTION_CAP_MS is not None and cmd_key in {"l", "r"} and duration_ms is not None:
            send_duration = min(int(duration_ms), int(WHEEL_TURN_ACTION_CAP_MS))
        return self._robot.send_command_pwm(cmd, pwm, duration_ms=send_duration)

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        original = list(actions or [])
        filtered = []
        removed = []
        for action in original:
            target = str((action or {}).get("target") or "").strip().lower()
            act = str((action or {}).get("action") or "").strip().lower()
            if target == "m" or act in {"u", "d"}:
                removed.append(dict(action or {}))
            else:
                next_action = dict(action or {})
                if WHEEL_TURN_ACTION_CAP_MS is not None and target in {"l", "r"}:
                    try:
                        next_action["duration_ms"] = min(
                            int(next_action.get("duration_ms") or WHEEL_TURN_ACTION_CAP_MS),
                            int(WHEEL_TURN_ACTION_CAP_MS),
                        )
                    except (TypeError, ValueError):
                        next_action["duration_ms"] = int(WHEEL_TURN_ACTION_CAP_MS)
                filtered.append(next_action)
        if removed:
            self.mast_attempts.append(
                {
                    "kind": "send_custom_actions_pwm",
                    "cmd": str(cmd or ""),
                    "duration_ms": int(duration_ms or 0),
                    "removed": removed,
                }
            )
        if not filtered:
            return {
                "cmd_sent": cmd,
                "actions": [],
                "duration_ms": 0,
                "skipped": True,
                "mast_frozen": True,
                "blocked": False,
                "reason": "mast_frozen",
            }
        send_duration = duration_ms
        if WHEEL_TURN_ACTION_CAP_MS is not None:
            wheel_actions = [
                action
                for action in filtered
                if str((action or {}).get("target") or "").strip().lower() in {"l", "r"}
            ]
            if wheel_actions and duration_ms is not None:
                send_duration = min(int(duration_ms), int(WHEEL_TURN_ACTION_CAP_MS))
        return self._robot.send_custom_actions_pwm(cmd, filtered, duration_ms=send_duration)


def _now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _reading_summary(reading: dict | None) -> dict:
    if not isinstance(reading, dict):
        return {}
    out = {}
    for key in (
        "visible",
        "confident",
        "conf",
        "dist_mm",
        "x_mm",
        "y_mm",
        "reason",
        "target_guard_blocked",
        "target_guard_reason",
        "green_stack_candidate_count",
        "green_contour_fallback",
        "vision_candidate_count",
        "vision_geometry_source",
    ):
        if key in reading:
            value = reading.get(key)
            if isinstance(value, float):
                out[key] = round(value, 3)
            else:
                out[key] = value
    return out


def _fmt_mm(value, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if signed:
        return f"{number:+.1f}"
    return f"{number:.1f}"


def _float_or_none(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_entry(target, tol, *, locked: bool = False) -> dict:
    return {
        "target_mm": _float_or_none(target),
        "tol_mm": _float_or_none(tol),
        "locked": bool(locked),
    }


def _current_targets() -> dict:
    cfg = follow._follow_motion_config()
    y_cfg = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
    step2_targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
    locked_y = _float_or_none(Y_LOCK_TARGET_MM)
    locked_tol = _float_or_none(Y_LOCK_TOL_MM)
    return {
        "step1": {
            "dist": _target_entry(follow._dist_target_mm(), follow._dist_tol_mm()),
            "x": _target_entry(follow._x_target_mm(), DIRECT_STEP1_X_TOL_MM),
            "y": _target_entry(
                locked_y if locked_y is not None else y_cfg.get("win_target_mm"),
                locked_tol if locked_y is not None else y_cfg.get("win_tol_mm"),
                locked=locked_y is not None,
            ),
        },
        "step2": {
            "dist": _target_entry(step2_targets.get("dist_mm"), step2_targets.get("dist_tol_mm")),
            "x": _target_entry(step2_targets.get("x_mm"), step2_targets.get("x_tol_mm")),
            "y": _target_entry(
                locked_y if locked_y is not None else step2_targets.get("y_mm"),
                locked_tol if locked_y is not None else step2_targets.get("y_tol_mm"),
                locked=locked_y is not None,
            ),
        },
    }


def _evaluate_reading(reading: dict | None, step: str | None) -> dict:
    if step not in {"step1", "step2"}:
        return {}
    targets = _current_targets().get(str(step), {})
    summary = _reading_summary(reading)
    axes = {}
    all_ok = True
    any_axis = False
    for axis in AXES:
        spec = targets.get(axis) if isinstance(targets, dict) else {}
        target = _float_or_none((spec or {}).get("target_mm"))
        tol = _float_or_none((spec or {}).get("tol_mm"))
        value = _float_or_none(summary.get(f"{axis}_mm"))
        locked = bool((spec or {}).get("locked"))
        axis_ok = False
        closeness = 0.0
        err = None
        if target is not None and tol is not None and value is not None and tol > 0.0:
            err = float(value) - float(target)
            closeness = max(0.0, min(100.0, 100.0 * (1.0 - (abs(err) / float(tol)))))
            axis_ok = abs(err) <= float(tol)
            if not locked:
                any_axis = True
        else:
            if not locked:
                all_ok = False
        if not locked and not axis_ok:
            all_ok = False
        axes[axis] = {
            "value_mm": value,
            "target_mm": target,
            "tol_mm": tol,
            "err_mm": err,
            "closeness_pct": round(closeness, 1),
            "ok": bool(axis_ok),
            "locked": bool(locked),
            "gated": not bool(locked),
        }
    return {"step": step, "axes": axes, "target_met": bool(any_axis and all_ok)}


def _green_stack_candidates(frame) -> list[dict]:
    if frame is None:
        return []
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except Exception:
        return []
    mask = cv2.inRange(
        hsv,
        np.array([45, 45, 25], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    mask[:60, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((25, 45), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_h, frame_w = frame.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 700.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = float(x) + float(w) / 2.0
        if (
            y < 40
            or y > int(frame_h * 0.55)
            or x < 20
            or x + w > int(frame_w - 20)
            or cx < float(DIRECT_X_MIN_SAFE_CX_PX)
            or cx > float(DIRECT_X_MAX_SAFE_CX_PX)
            or w < 28
            or h < 45
            or w > int(frame_w * 0.45)
        ):
            continue
        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": round(cx, 1),
                "area": round(area, 1),
            }
        )
    return sorted(candidates, key=lambda box: float(box["cx"]))


def _target_guard_frame(vision: BrickDetector):
    frame = getattr(vision, "raw_frame", None)
    if frame is None:
        frame = getattr(vision, "current_frame", None)
    return frame


def _annotate_target_guard(vision: BrickDetector, reading: dict) -> dict:
    if not isinstance(reading, dict):
        return reading
    out = dict(reading)
    candidates = _green_stack_candidates(_target_guard_frame(vision))
    out["green_stack_candidate_count"] = int(len(candidates))
    if candidates:
        out["green_stack_candidates"] = candidates[:4]
    if len(candidates) > 1:
        out["confident"] = False
        out["target_guard_blocked"] = True
        out["target_guard_reason"] = "adjacent_stack_ambiguous"
        out["reason"] = "target_guard_adjacent_stack_ambiguous"
        return out
    dist = _float_or_none(out.get("dist_mm"))
    x_mm = _float_or_none(out.get("x_mm"))
    if dist is not None and dist > TARGET_GUARD_MAX_DIST_MM:
        out["confident"] = False
        out["target_guard_blocked"] = True
        out["target_guard_reason"] = "target_identity_depth_jump"
        out["reason"] = "target_guard_depth_jump"
        return out
    if x_mm is not None and abs(x_mm) > TARGET_GUARD_MAX_ABS_X_MM:
        out["confident"] = False
        out["target_guard_blocked"] = True
        out["target_guard_reason"] = "target_identity_lateral_jump"
        out["reason"] = "target_guard_lateral_jump"
        return out
    return out


def _install_target_continuity_guard() -> None:
    if getattr(follow, "_frozen_step12_target_guard_installed", False):
        return
    original = follow._read_brick_measurement

    def _guarded_reading(vision: BrickDetector, *, jump_guard: bool = False) -> dict:
        reading = original(vision, jump_guard=jump_guard)
        return _annotate_target_guard(vision, reading)

    follow._read_brick_measurement = _guarded_reading
    follow._frozen_step12_target_guard_installed = True


def _green_contour_measurement_from_frame(frame, base: dict | None = None) -> dict:
    candidates = _green_stack_candidates(frame)
    out = dict(base) if isinstance(base, dict) else {}
    out["green_stack_candidate_count"] = int(len(candidates))
    if len(candidates) != 1:
        out["visible"] = bool(candidates)
        out["confident"] = False
        out["reason"] = "green_contour_ambiguous" if candidates else "green_contour_not_visible"
        return out
    box = candidates[0]
    dist_mm = float(GREEN_CONTOUR_DIST_SCALE_MM_PX) / max(1.0, float(box["h"]))
    x_mm = ((float(box["cx"]) - 320.0) * float(dist_mm)) / 620.0
    out.update(
        {
            "visible": True,
            "confident": True,
            "conf": 80.0,
            "reason": "green_stack_contour_single",
            "dist_mm": float(dist_mm),
            "x_mm": float(x_mm),
            "y_mm": -0.22 * float(dist_mm),
            "green_contour_fallback": True,
            "green_stack_candidates": candidates,
        }
    )
    return out


def _install_single_green_contour_fallback() -> None:
    if getattr(follow, "_frozen_step12_green_contour_fallback_installed", False):
        return
    original = follow._read_brick_measurement

    def _green_contour_reading(vision: BrickDetector, *, jump_guard: bool = False) -> dict:
        reading = original(vision, jump_guard=jump_guard)
        if bool(reading.get("confident")) and not bool(reading.get("ghost_dist_implausible")):
            return reading
        frame = _target_guard_frame(vision)
        return _green_contour_measurement_from_frame(frame, reading)

    follow._read_brick_measurement = _green_contour_reading
    follow._frozen_step12_green_contour_fallback_installed = True


def _install_green_contour_override() -> None:
    if getattr(follow, "_frozen_step12_green_contour_override_installed", False):
        return
    original = follow._read_brick_measurement

    def _green_contour_override_reading(vision: BrickDetector, *, jump_guard: bool = False) -> dict:
        base = original(vision, jump_guard=jump_guard)
        frame = _target_guard_frame(vision)
        reading = _green_contour_measurement_from_frame(frame, base)
        reading["green_contour_override"] = True
        return reading

    follow._read_brick_measurement = _green_contour_override_reading
    follow._frozen_step12_green_contour_override_installed = True


class ProgressSite:
    def __init__(self, site_dir: Path, title: str):
        self.site_dir = Path(site_dir)
        self.images_dir = self.site_dir / "images"
        self.state_path = self.site_dir / "state.json"
        self.index_path = self.site_dir / "index.html"
        self.title = title
        self.state = self._load_state(title)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._write()

    def _load_state(self, title: str) -> dict:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
            if isinstance(state, dict):
                state.setdefault("title", title)
                state.setdefault("started_at", _now_label())
                state.setdefault("rows", [])
                state.setdefault("targets", {})
                state.setdefault("experiment_line", "")
                state["updated_at"] = _now_label()
                return state
        return {
            "title": title,
            "started_at": _now_label(),
            "updated_at": _now_label(),
            "experiment": "",
            "experiment_line": "",
            "summary": "Starting",
            "targets": {},
            "rows": [],
        }

    def set_experiment(self, name: str, summary: str) -> None:
        self.state["experiment"] = str(name)
        self.state["experiment_line"] = EXPERIMENT_LINES.get(str(name), str(summary))
        self.state["summary"] = str(summary)
        self._write()

    def set_targets(self) -> None:
        self.state["targets"] = _current_targets()
        self.state["updated_at"] = _now_label()
        self._write()

    def add_row(self, row: dict) -> None:
        row = dict(row)
        row.setdefault("time", _now_label())
        row.setdefault("experiment", self.state.get("experiment", ""))
        row.setdefault("experiment_line", self.state.get("experiment_line", ""))
        self.state.setdefault("rows", []).append(row)
        self.state["updated_at"] = _now_label()
        self._write()

    def update_summary(self, summary: str) -> None:
        self.state["summary"] = str(summary)
        self.state["updated_at"] = _now_label()
        self._write()

    def capture(self, vision: BrickDetector, label: str, reading: dict | None = None) -> tuple[str | None, dict]:
        reading = reading if isinstance(reading, dict) else follow._read_brick_measurement(vision)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
        image_name = f"{int(time.time())}_{safe_label}.png"
        image_path = self.images_dir / image_name
        frame = getattr(vision, "current_frame", None)
        if frame is None:
            frame = getattr(vision, "raw_frame", None)
        rel = None
        if frame is not None:
            cv2.imwrite(str(image_path), frame)
            rel = f"images/{image_name}"
        return rel, _reading_summary(reading)

    def _write(self) -> None:
        self.site_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        self.index_path.write_text(self._render_html(), encoding="utf-8")

    def _render_html(self) -> str:
        rows = list(self.state.get("rows") or [])
        last_failure_index = -1
        for index, row in enumerate(rows):
            if row.get("status") == "fail":
                last_failure_index = index
        streak_rows = rows[last_failure_index + 1 :]
        honest_trial_rows = [
            row for row in streak_rows if row.get("phase") == "trial" and row.get("honest_trial") is not None
        ]
        historical_trial_rows = [
            row for row in rows if row.get("phase") == "trial" and row.get("honest_trial") is not None
        ]
        wins = sum(1 for row in honest_trial_rows if row.get("status") == "win" and bool(row.get("honest_trial")))
        trials = len(honest_trial_rows)
        historical_wins = sum(
            1 for row in historical_trial_rows if row.get("status") == "win" and bool(row.get("honest_trial"))
        )
        step1_rows = [row for row in rows if row.get("step") == "step1"]
        step2_rows = [row for row in rows if row.get("step") == "step2"]
        end_rows = {}
        for row in rows:
            if row.get("phase") == "trial" and row.get("honest_trial") is not None:
                try:
                    trial_key = int(row.get("trial"))
                except (TypeError, ValueError):
                    continue
                try:
                    attempt_key = int(row.get("attempt"))
                except (TypeError, ValueError):
                    attempt_key = 0
                end_rows[(attempt_key, trial_key)] = row
        iteration_groups: dict[int, list[dict]] = {}
        for row in rows:
            try:
                attempt_key = int(row.get("attempt"))
            except (TypeError, ValueError):
                continue
            iteration_groups.setdefault(attempt_key, []).append(row)

        def iteration_card(attempt_key: int, attempt_rows: list[dict]) -> str:
            experiment_name = ""
            experiment_line = ""
            for row in attempt_rows:
                if row.get("experiment"):
                    experiment_name = str(row.get("experiment") or "")
                if row.get("experiment_line"):
                    experiment_line = str(row.get("experiment_line") or "")
            if not experiment_line:
                experiment_line = EXPERIMENT_LINES.get(experiment_name, "")
            trial_rows = [
                row for row in attempt_rows if row.get("phase") == "trial" and row.get("honest_trial") is not None
            ]
            wins_count = sum(
                1 for row in trial_rows if row.get("status") == "win" and bool(row.get("honest_trial"))
            )
            fail_rows = [row for row in attempt_rows if row.get("status") == "fail"]
            if wins_count >= 5 and not fail_rows:
                grade = "A 5/5"
                grade_class = "win"
            elif fail_rows:
                grade = f"Stop {wins_count}/5"
                grade_class = "fail"
            else:
                grade = f"{wins_count}/5"
                grade_class = "info"
            stop_reason = ""
            if fail_rows:
                stop_reason = str(fail_rows[-1].get("reason") or "")
            elif trial_rows:
                stop_reason = str(trial_rows[-1].get("reason") or "")
            prior_rows = [
                row for row in attempt_rows if row.get("phase") == "prior" and row.get("trial") not in {None, 0}
            ]

            def thumb_strip(selected_rows: list[dict], *, empty: str, alt: str) -> str:
                thumbs = []
                for row in selected_rows[-5:]:
                    image = row.get("image")
                    status = str(row.get("status", ""))
                    if image:
                        thumbs.append(
                            f'<a href="{html.escape(str(image))}" title="trial {html.escape(str(row.get("trial", "")))}">'
                            f'<img src="{html.escape(str(image))}" alt="{html.escape(alt)}"></a>'
                        )
                    else:
                        thumbs.append(f'<span class="status {html.escape(status)}">{html.escape(status)}</span>')
                return "".join(thumbs) if thumbs else f'<span class="muted">{html.escape(empty)}</span>'

            prior_html = thumb_strip(prior_rows, empty="No prior images yet", alt="prior to attempt")
            end_html = thumb_strip(trial_rows, empty="No trial end image yet", alt="trial end")
            return (
                f'<article class="iteration-card">'
                f'<div class="iteration-head"><b>Iteration {attempt_key}</b>'
                f'<span class="status {html.escape(grade_class)}">{html.escape(grade)}</span></div>'
                f'<div class="experiment-name">{html.escape(experiment_name or "unknown")}</div>'
                f'<div class="experiment-line">{html.escape(experiment_line or "No description recorded.")}</div>'
                f'<div class="iteration-meta">Trial wins {wins_count}/5'
                f'{(" - " + html.escape(stop_reason)) if stop_reason else ""}</div>'
                f'<div class="iteration-label">Prior to attempt</div>'
                f'<div class="iteration-images">{prior_html}</div>'
                f'<div class="iteration-label">Trial end</div>'
                f'<div class="iteration-images">{end_html}</div>'
                f'</article>'
            )

        iteration_cards = [
            iteration_card(attempt_key, iteration_groups[attempt_key])
            for attempt_key in sorted(iteration_groups.keys(), reverse=True)
        ]
        iterations_html = (
            "".join(iteration_cards) if iteration_cards else '<p class="empty">No iterations yet.</p>'
        )

        def axis_cell(row: dict, axis: str) -> str:
            evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
            axes = evaluation.get("axes") if isinstance(evaluation.get("axes"), dict) else {}
            data = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
            pct = max(0.0, min(100.0, _float_or_none(data.get("closeness_pct")) or 0.0))
            ok_class = " axis-ok" if bool(data.get("ok")) else ""
            locked = " locked" if bool(data.get("locked")) else ""
            value = _fmt_mm(data.get("value_mm"), signed=axis in {"x", "y"})
            target = _fmt_mm(data.get("target_mm"), signed=axis in {"x", "y"})
            tol = _fmt_mm(data.get("tol_mm"))
            err = _fmt_mm(data.get("err_mm"), signed=True)
            return (
                f'<div class="axis{ok_class}{locked}">'
                f'<div class="axis-top"><b>{html.escape(axis.upper())}</b><span>{value} / {target} +/- {tol}</span></div>'
                f'<div class="bar"><i style="width:{pct:.1f}%"></i></div>'
                f'<div class="axis-sub">err {err} mm, close {pct:.0f}%</div>'
                "</div>"
            )

        def step_table(step_rows_for_table: list[dict]) -> str:
            cards = []
            for row in step_rows_for_table:
                image = row.get("image")
                thumb = ""
                if image:
                    thumb = f'<a href="{html.escape(str(image))}"><img src="{html.escape(str(image))}" alt="camera"></a>'
                status = str(row.get("status", ""))
                cards.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('trial', '')))}</td>"
                    f"<td>{html.escape(str(row.get('attempt', '')))}</td>"
                    f"<td>{html.escape(str(row.get('phase', '')))}</td>"
                    f"<td><span class=\"status {html.escape(status)}\">{html.escape(status)}</span></td>"
                    f"<td>{html.escape(str(row.get('reason', '')))}</td>"
                    f"<td>{axis_cell(row, 'dist')}</td>"
                    f"<td>{axis_cell(row, 'x')}</td>"
                    f"<td>{axis_cell(row, 'y')}</td>"
                    f"<td>{html.escape(str(row.get('mast_attempts', 0)))}</td>"
                    f"<td>{thumb}</td>"
                    "</tr>"
                )
            if not cards:
                return '<p class="empty">No rows yet.</p>'
            return (
                "<table>"
                "<thead><tr><th>Trial</th><th>Iteration</th><th>Phase</th><th>Status</th><th>Reason</th>"
                "<th>Dist</th><th>X</th><th>Y</th><th>Mast Plans</th><th>Image</th></tr></thead>"
                f"<tbody>{''.join(cards)}</tbody></table>"
            )

        gallery_cards = []
        for (attempt_key, trial), row in sorted(end_rows.items()):
            image = row.get("image")
            if not image:
                continue
            status = str(row.get("status", ""))
            gallery_cards.append(
                f'<figure><a href="{html.escape(str(image))}"><img src="{html.escape(str(image))}" alt="trial {trial} end"></a>'
                f'<figcaption>Iteration {attempt_key}, trial {trial}: <span class="status {html.escape(status)}">{html.escape(status)}</span></figcaption></figure>'
            )
        gallery = ''.join(gallery_cards) if gallery_cards else '<p class="empty">No end-of-attempt images yet.</p>'
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(self.title)}</title>
  <style>
    body {{ margin: 0; font: 14px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; background: #f5f7f8; color: #17202a; }}
    header {{ padding: 18px 22px; background: #193549; color: white; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 700; }}
    main {{ padding: 18px 22px 36px; }}
    .summary {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px; }}
    .metric {{ background: white; border: 1px solid #d9e0e8; border-radius: 6px; padding: 10px 12px; min-width: 160px; }}
    .metric b {{ display: block; font-size: 20px; }}
    .metric.wide {{ min-width: 360px; max-width: 720px; }}
    h2 {{ margin: 20px 0 10px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e0e8; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid #e8edf3; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    img {{ width: 180px; max-height: 130px; object-fit: contain; border: 1px solid #d9e0e8; border-radius: 4px; background: #111; }}
    .status {{ display: inline-block; min-width: 52px; padding: 2px 7px; border-radius: 999px; text-align: center; font-weight: 700; }}
    .win {{ background: #dff6e8; color: #17633a; }}
    .fail {{ background: #ffe3df; color: #8c261e; }}
    .info {{ background: #e4eefc; color: #1d4f8f; }}
    .running {{ background: #fff2c7; color: #7a5600; }}
    .tabs {{ margin-top: 12px; }}
    .tabs input {{ position: absolute; opacity: 0; pointer-events: none; }}
    .tab-labels {{ display: flex; gap: 8px; margin-bottom: 10px; }}
    .tab-labels label {{ display: inline-flex; align-items: center; justify-content: center; min-width: 110px; padding: 8px 12px; border: 1px solid #bdc8d4; border-radius: 6px; background: #fff; font-weight: 700; cursor: pointer; }}
    #tab-step1:checked ~ .tab-labels label[for="tab-step1"], #tab-step2:checked ~ .tab-labels label[for="tab-step2"] {{ background: #193549; color: #fff; border-color: #193549; }}
    .panel {{ display: none; }}
    #tab-step1:checked ~ .panels .panel-step1, #tab-step2:checked ~ .panels .panel-step2 {{ display: block; }}
    .axis {{ min-width: 180px; }}
    .axis-top {{ display: flex; gap: 8px; justify-content: space-between; font-size: 12px; }}
    .bar {{ height: 10px; margin: 4px 0; background: #edf1f4; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: #bd5a45; }}
    .axis-ok .bar i {{ background: #2b8a58; }}
    .axis-sub {{ color: #566574; font-size: 12px; }}
    .locked .axis-top b::after {{ content: " locked"; color: #5d6b78; font-weight: 600; text-transform: none; }}
    .iteration-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }}
    .iteration-card {{ background: white; border: 1px solid #d9e0e8; border-radius: 6px; padding: 10px; }}
    .iteration-head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .experiment-name {{ margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; color: #33485b; overflow-wrap: anywhere; }}
    .experiment-line {{ margin-top: 4px; color: #253341; }}
    .iteration-meta {{ margin-top: 6px; color: #566574; font-size: 12px; }}
    .iteration-label {{ margin-top: 8px; color: #33485b; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }}
    .iteration-images {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .iteration-images img {{ width: 76px; height: 54px; object-fit: cover; }}
    .muted {{ color: #6b7785; font-size: 12px; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 10px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d9e0e8; border-radius: 6px; padding: 8px; }}
    figcaption {{ margin-top: 6px; font-size: 13px; }}
    .empty {{ padding: 14px; background: white; border: 1px solid #d9e0e8; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(self.title)}</h1>
    <div>Experiment: {html.escape(str(self.state.get('experiment') or ''))}</div>
    <div>{html.escape(str(self.state.get('experiment_line') or ''))}</div>
    <div>Updated: {html.escape(str(self.state.get('updated_at') or ''))}</div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>Current honest streak</span><b>{wins}/5</b></div>
      <div class="metric"><span>Current trial attempts</span><b>{trials}</b></div>
      <div class="metric"><span>Historical honest rows</span><b>{historical_wins}/{len(historical_trial_rows)}</b></div>
      <div class="metric wide"><span>Status</span><b>{html.escape(str(self.state.get('summary') or ''))}</b></div>
    </section>
    <h2>Iterations</h2>
    <section class="iteration-grid">{iterations_html}</section>
    <section class="tabs">
      <input id="tab-step1" name="tabs" type="radio" checked>
      <input id="tab-step2" name="tabs" type="radio">
      <div class="tab-labels">
        <label for="tab-step1">Step 1</label>
        <label for="tab-step2">Step 2</label>
      </div>
      <div class="panels">
        <section class="panel panel-step1">{step_table(step1_rows)}</section>
        <section class="panel panel-step2">{step_table(step2_rows)}</section>
      </div>
    </section>
    <h2>End Images</h2>
    <section class="gallery">{gallery}</section>
  </main>
</body>
</html>
"""


def _local_url(port: int) -> str:
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        host_ip = "plainOrin"
    return f"http://{host_ip}:{int(port)}/"


def _apply_visible_reset(*, target_abs_x_mm: float = 19.0, x_offset_max_mm: float = 30.0) -> None:
    reset_cfg = follow._reset_motion_config()
    reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
    step1_dist = float(follow._dist_target_mm())
    reverse["dist_target_mm"] = step1_dist + 30.0
    reverse["dist_tol_mm"] = 10.0
    reverse["x_offset_min_mm"] = 0.0
    reverse["x_offset_max_mm"] = float(x_offset_max_mm)
    reverse["target_abs_x_mm"] = float(target_abs_x_mm)
    reverse["post_pause_s"] = 0.0
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    straight["enabled"] = False
    straight["duration_ms"] = 0
    straight["duration_min_ms"] = 0
    straight["duration_max_ms"] = 0
    reverse["straight_back_first"] = straight
    adjustment = reverse.get("adjustment") if isinstance(reverse.get("adjustment"), dict) else {}
    adjustment["enabled"] = False
    adjustment["max_attempts"] = 0
    reverse["adjustment"] = adjustment
    reset_cfg["reverse_turn"] = reverse
    profiles = reset_cfg.get("game_profiles") if isinstance(reset_cfg.get("game_profiles"), dict) else {}
    empty_profile = profiles.get("empty") if isinstance(profiles.get("empty"), dict) else {}
    empty_profile["reverse_turn"] = dict(reverse)
    profiles["empty"] = empty_profile
    reset_cfg["game_profiles"] = profiles


def _apply_distance_only_reset(*, dist_offset_mm: float = 20.0, dist_tol_mm: float = 10.0) -> None:
    reset_cfg = follow._reset_motion_config()
    reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
    step1_dist = float(follow._dist_target_mm())
    reverse["dist_target_mm"] = step1_dist + float(dist_offset_mm)
    reverse["dist_tol_mm"] = float(dist_tol_mm)
    reverse["x_offset_min_mm"] = 0.0
    reverse["x_offset_max_mm"] = 999.0
    reverse["target_abs_x_mm"] = 0.0
    reverse["post_pause_s"] = 0.0
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    straight["enabled"] = False
    straight["duration_ms"] = 0
    straight["duration_min_ms"] = 0
    straight["duration_max_ms"] = 0
    reverse["straight_back_first"] = straight
    adjustment = reverse.get("adjustment") if isinstance(reverse.get("adjustment"), dict) else {}
    adjustment["enabled"] = False
    adjustment["max_attempts"] = 0
    reverse["adjustment"] = adjustment
    reset_cfg["reverse_turn"] = reverse
    profiles = reset_cfg.get("game_profiles") if isinstance(reset_cfg.get("game_profiles"), dict) else {}
    empty_profile = profiles.get("empty") if isinstance(profiles.get("empty"), dict) else {}
    empty_profile["reverse_turn"] = dict(reverse)
    profiles["empty"] = empty_profile
    reset_cfg["game_profiles"] = profiles


def _disable_y_gates(cfg: dict, step2: dict) -> None:
    y_axis = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    y_axis["enabled"] = False
    cfg["y_axis"] = y_axis
    targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
    targets["y_mm"] = None
    targets["y_tol_mm"] = None
    step2["targets"] = targets


def _apply_soft_simultaneous_tracking(cfg: dict) -> None:
    cfg["min_confidence_pct"] = 45.0
    follow.load_brick_visibility_motion_safety_config = lambda path=None: {"min_confidence_pct": 45.0}

    dist_axis = cfg.get("dist_axis") if isinstance(cfg.get("dist_axis"), dict) else {}
    dist_axis["win_target_mm"] = 190.0
    dist_axis["win_tol_mm"] = 18.0
    cfg["dist_axis"] = dist_axis

    x_axis = cfg.get("x_axis") if isinstance(cfg.get("x_axis"), dict) else {}
    x_axis["win_target_mm"] = 19.0
    x_axis["win_tol_mm"] = 8.0
    x_axis["positive_error_turn_cmd"] = "l"
    cfg["x_axis"] = x_axis

    win_confirmation = cfg.get("win_confirmation") if isinstance(cfg.get("win_confirmation"), dict) else {}
    win_confirmation["min_confidence_pct"] = 50.0
    cfg["win_confirmation"] = win_confirmation

    cautious = cfg.get("cautious_visibility") if isinstance(cfg.get("cautious_visibility"), dict) else {}
    cautious["motion_min_confidence_pct"] = 45.0
    cfg["cautious_visibility"] = cautious

    x_priority = cfg.get("x_priority_policy") if isinstance(cfg.get("x_priority_policy"), dict) else {}
    x_priority["simultaneous_x_abs_max_mm"] = 999.0
    x_priority["x_first_turn_strength"] = "adaptive"
    x_priority["adaptive_outer_pwm_scale"] = 0.8
    x_priority["turn_settle_s"] = 0.15
    x_priority["tiny_x_outside_no_turn_mm"] = 1.5
    cfg["x_priority_policy"] = x_priority

    x_dist = cfg.get("x_dist_curve_policy") if isinstance(cfg.get("x_dist_curve_policy"), dict) else {}
    x_dist["combined_bias_max_pulse_ms"] = 280
    x_dist["near_wide_x_strength"] = "gentle"
    cfg["x_dist_curve_policy"] = x_dist

    dist_approach = cfg.get("dist_approach_policy") if isinstance(cfg.get("dist_approach_policy"), dict) else {}
    dist_approach["near_target_max_pulse_ms"] = 260
    dist_approach["min_effective_drive_pulse_ms"] = 240
    dist_approach["micro_nudge_min_effective_pulse_ms"] = 220
    cfg["dist_approach_policy"] = dist_approach

    stall = cfg.get("act_stall_guard") if isinstance(cfg.get("act_stall_guard"), dict) else {}
    stall["max_no_change_tries"] = 6
    stall["close_trap_max_no_change_tries"] = 4
    stall["min_axis_delta_mm"] = 0.4
    cfg["act_stall_guard"] = stall

    x_only = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    x_only["sharp_max_abs_dist_err_mm"] = 0.0
    cfg["x_only_turn"] = x_only

    original_x_only_turn_plan = follow._x_only_turn_plan

    def _simultaneous_x_turn_plan(**kwargs):
        dist_err = float(kwargs.get("dist_err", 0.0))
        drive_mode = follow._drive_mode_for_dist_error(dist_err)
        return follow._x_dist_drive_bias_plan(
            turn_cmd=str(kwargs.get("turn_cmd") or "r"),
            drive_mode=drive_mode,
            dist_err=dist_err,
            x_err=float(kwargs.get("x_err", 0.0)),
            x_outside=float(kwargs.get("x_outside", 0.0)),
            dist_outside=float(kwargs.get("dist_outside", 0.0)),
            y_plan=kwargs.get("y_plan"),
            reason="soft_simultaneous_x_dist",
        )

    _simultaneous_x_turn_plan._original = original_x_only_turn_plan
    follow._x_only_turn_plan = _simultaneous_x_turn_plan


def _apply_simultaneous_gap_closure(cfg: dict) -> None:
    x_priority = cfg.get("x_priority_policy") if isinstance(cfg.get("x_priority_policy"), dict) else {}
    x_priority["simultaneous_x_abs_max_mm"] = 999.0
    x_priority["x_first_turn_strength"] = "adaptive"
    x_priority["adaptive_outer_pwm_scale"] = 0.75
    x_priority["turn_settle_s"] = 0.12
    x_priority["tiny_x_outside_no_turn_mm"] = 0.8
    cfg["x_priority_policy"] = x_priority

    x_dist = cfg.get("x_dist_curve_policy") if isinstance(cfg.get("x_dist_curve_policy"), dict) else {}
    x_dist["combined_bias_max_pulse_ms"] = 240
    x_dist["near_wide_x_strength"] = "gentle"
    cfg["x_dist_curve_policy"] = x_dist

    dist_approach = cfg.get("dist_approach_policy") if isinstance(cfg.get("dist_approach_policy"), dict) else {}
    dist_approach["near_target_max_pulse_ms"] = 220
    dist_approach["min_effective_drive_pulse_ms"] = 180
    dist_approach["micro_nudge_min_effective_pulse_ms"] = 160
    cfg["dist_approach_policy"] = dist_approach

    stall = cfg.get("act_stall_guard") if isinstance(cfg.get("act_stall_guard"), dict) else {}
    stall["max_no_change_tries"] = 4
    stall["close_trap_max_no_change_tries"] = 3
    stall["min_axis_delta_mm"] = 0.6
    cfg["act_stall_guard"] = stall

    x_only = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    x_only["sharp_max_abs_dist_err_mm"] = 0.0
    cfg["x_only_turn"] = x_only

    original_x_only_turn_plan = getattr(follow._x_only_turn_plan, "_original", follow._x_only_turn_plan)

    def _simultaneous_x_turn_plan(**kwargs):
        dist_err = float(kwargs.get("dist_err", 0.0))
        drive_mode = follow._drive_mode_for_dist_error(dist_err)
        return follow._x_dist_drive_bias_plan(
            turn_cmd=str(kwargs.get("turn_cmd") or "r"),
            drive_mode=drive_mode,
            dist_err=dist_err,
            x_err=float(kwargs.get("x_err", 0.0)),
            x_outside=float(kwargs.get("x_outside", 0.0)),
            dist_outside=float(kwargs.get("dist_outside", 0.0)),
            y_plan=None,
            reason="honest_simultaneous_dist_x",
        )

    _simultaneous_x_turn_plan._original = original_x_only_turn_plan
    follow._x_only_turn_plan = _simultaneous_x_turn_plan


def _apply_step2_safe_targets(step2: dict) -> None:
    targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
    targets["dist_mm"] = 175.0
    targets["dist_tol_mm"] = 20.0
    targets["x_mm"] = 19.0
    targets["x_tol_mm"] = 8.0
    targets["y_mm"] = None
    targets["y_tol_mm"] = None
    step2["targets"] = targets
    step2["precision_drive_min_pulse_ms"] = 45
    step2["precision_drive_max_pulse_ms"] = 90
    step2["precision_max_attempts"] = 20
    step2["precision_hard_max_attempts"] = 120
    step2["step_timeout_s"] = 30.0


def _apply_step2_strict_movement(step2: dict) -> None:
    targets = step2.get("targets") if isinstance(step2.get("targets"), dict) else {}
    targets["dist_mm"] = float(DIRECT_STEP2_DIST_TARGET_MM)
    targets["dist_tol_mm"] = float(DIRECT_STEP2_DIST_TOL_MM)
    targets["x_mm"] = 5.9
    targets["x_tol_mm"] = 3.0
    targets["y_mm"] = None
    targets["y_tol_mm"] = None
    step2["targets"] = targets
    step2["semi_happy_targets"] = {
        "dist_mm": float(DIRECT_STEP2_DIST_TARGET_MM),
        "dist_tol_mm": 12.0,
        "x_mm": 5.9,
        "x_tol_mm": 8.0,
        "y_mm": None,
        "y_tol_mm": None,
    }
    step2["seat_mast_duration_ms"] = 0
    step2["recovery_creep_enabled"] = False
    step2["recovery_creep_max_attempts"] = 0
    step2["visibility_recovery_creep_enabled"] = False
    step2["visibility_recovery_creep_max_attempts"] = 0
    step2["precision_drive_min_pulse_ms"] = 55
    step2["precision_drive_max_pulse_ms"] = 105
    step2["precision_max_attempts"] = 18
    step2["precision_hard_max_attempts"] = 60
    step2["step_timeout_s"] = 24.0


def _flip_positive_x_left(cfg: dict) -> None:
    x_axis = cfg.get("x_axis") if isinstance(cfg.get("x_axis"), dict) else {}
    x_axis["positive_error_turn_cmd"] = "l"
    cfg["x_axis"] = x_axis


def _apply_x_first_policy(cfg: dict) -> None:
    x_priority = cfg.get("x_priority_policy") if isinstance(cfg.get("x_priority_policy"), dict) else {}
    x_priority["polish_abs_x_mm"] = 3.0
    x_priority["simultaneous_dist_outside_min_mm"] = 999.0
    x_priority["x_first_turn_strength"] = "strong"
    x_priority["adaptive_outer_pwm_scale"] = 1.0
    x_priority["turn_settle_s"] = 0.25
    x_priority["tiny_x_outside_no_turn_mm"] = 0.0
    cfg["x_priority_policy"] = x_priority
    x_only = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    x_only["drive_mode"] = "forward"
    x_only["far_drive_mode"] = "forward"
    x_only["forward_min_dist_err_mm"] = 0.0
    x_only["sharp_max_abs_dist_err_mm"] = 999.0
    cfg["x_only_turn"] = x_only


def _install_no_reverse_pursuit_guard() -> None:
    if getattr(follow, "_frozen_step12_no_reverse_guard_installed", False):
        return
    original = follow._follow_action_plan

    def _no_reverse_follow_action_plan(reading: dict) -> dict:
        plan = original(reading)
        if not isinstance(plan, dict):
            return plan
        kind = str(plan.get("kind") or "").strip().lower()
        cmd = str(plan.get("cmd") or "").strip().lower()
        drive_mode = str(plan.get("drive_mode") or "").strip().lower()
        reverse = (kind in {"drive", "drive_bias", "drive_nudge"} and cmd == "b") or (
            kind == "turn" and drive_mode == "backward"
        )
        if not reverse:
            return plan
        return {
            "kind": "wait",
            "action": "NO_REVERSE_STOP",
            "dist_err": _float_or_none(plan.get("dist_err")) or 0.0,
            "x_err": _float_or_none(plan.get("x_err")) or 0.0,
            "duration_ms": 0,
            "reason": f"reverse_pursuit_blocked:{plan.get('reason') or plan.get('action') or kind}",
            "blocked_plan": {
                "kind": kind,
                "cmd": cmd,
                "drive_mode": drive_mode,
                "action": plan.get("action"),
                "duration_ms": plan.get("duration_ms"),
            },
        }

    follow._follow_action_plan = _no_reverse_follow_action_plan
    follow._frozen_step12_no_reverse_guard_installed = True


def _install_micro_x_turns(max_ms: int = 110) -> None:
    original = getattr(follow._x_turn_duration_ms, "_original", follow._x_turn_duration_ms)

    def _micro_x_turn_duration_ms(*, dist_err: float, turn_cmd: str) -> int:
        return int(max(70, min(int(max_ms), int(original(dist_err=dist_err, turn_cmd=turn_cmd)))))

    _micro_x_turn_duration_ms._original = original
    follow._x_turn_duration_ms = _micro_x_turn_duration_ms
    original_send = getattr(follow._send_turn_curve, "_original", follow._send_turn_curve)

    def _capped_send_turn_curve(robot: Robot, *, duration_ms: int, **kwargs):
        capped_ms = int(max(40, min(int(max_ms), int(duration_ms or max_ms))))
        return original_send(robot, duration_ms=capped_ms, **kwargs)

    _capped_send_turn_curve._original = original_send
    follow._send_turn_curve = _capped_send_turn_curve


def _install_wheel_turn_action_cap(max_ms: int) -> None:
    global WHEEL_TURN_ACTION_CAP_MS
    WHEEL_TURN_ACTION_CAP_MS = int(max(20, int(max_ms)))


def _set_direct_x_pulse_ms(duration_ms: int) -> None:
    global DIRECT_X_PULSE_MS
    DIRECT_X_PULSE_MS = int(max(20, int(duration_ms)))


def _set_direct_x_pwm(pwm: int) -> None:
    global DIRECT_X_PWM
    DIRECT_X_PWM = int(max(1, min(255, int(pwm))))


def _set_direct_step2_dist(pulse_ms: int, max_pulses: int) -> None:
    global DIRECT_STEP2_DIST_PULSE_MS, DIRECT_STEP2_DIST_MAX_PULSES
    DIRECT_STEP2_DIST_PULSE_MS = int(max(20, int(pulse_ms)))
    DIRECT_STEP2_DIST_MAX_PULSES = int(max(1, int(max_pulses)))


def _set_direct_reset_dist(pulse_ms: int, max_pulses: int) -> None:
    global DIRECT_DIST_PULSE_MS, DIRECT_RESET_DIST_MAX_PULSES
    DIRECT_DIST_PULSE_MS = int(max(20, int(pulse_ms)))
    DIRECT_RESET_DIST_MAX_PULSES = int(max(1, int(max_pulses)))


def _set_direct_turn_curve_enabled(enabled: bool) -> None:
    global DIRECT_USE_TURN_CURVE
    DIRECT_USE_TURN_CURVE = bool(enabled)


def _set_direct_x_primitive(name: str) -> None:
    global DIRECT_X_PRIMITIVE
    key = str(name or "command").strip().lower()
    DIRECT_X_PRIMITIVE = key if key in {"command", "onewheel_forward"} else "command"


def _set_direct_x_adaptive_pulse(enabled: bool) -> None:
    global DIRECT_X_ADAPTIVE_PULSE
    DIRECT_X_ADAPTIVE_PULSE = bool(enabled)


def _set_direct_stable_reads(count: int) -> None:
    global DIRECT_STABLE_READS
    DIRECT_STABLE_READS = int(max(1, int(count)))


def _lock_y_target_from_current(vision: BrickDetector) -> dict:
    global Y_LOCK_TARGET_MM
    if int(DIRECT_STABLE_READS) > 1:
        reading = _direct_read(vision, timeout_s=6.0)
    else:
        reading = follow._wait_for_confident_brick(vision, timeout_s=6.0, sample_s=0.12)
    y_mm = _float_or_none(reading.get("y_mm") if isinstance(reading, dict) else None)
    if y_mm is not None:
        Y_LOCK_TARGET_MM = float(y_mm)
    return reading


def _lock_y_target_from_reading(reading: dict | None) -> bool:
    global Y_LOCK_TARGET_MM
    y_mm = _float_or_none(reading.get("y_mm") if isinstance(reading, dict) else None)
    if y_mm is None:
        return False
    Y_LOCK_TARGET_MM = float(y_mm)
    return True


def _install_simple_hsv_fallback() -> None:
    if getattr(follow, "_frozen_step12_simple_hsv_installed", False):
        return
    original = follow._read_brick_measurement

    def _simple_hsv_reading(vision: BrickDetector, *, jump_guard: bool = False) -> dict:
        reading = original(vision, jump_guard=jump_guard)
        if bool(reading.get("confident")):
            return reading
        frame = getattr(vision, "current_frame", None)
        if frame is None:
            frame = getattr(vision, "raw_frame", None)
        if frame is None:
            return reading
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([50, 70, 25], dtype=np.uint8),
            np.array([100, 255, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 800.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if y > 230 or w < 35 or h < 35 or w > 150 or h > 130:
                continue
            aspect = float(w) / max(1.0, float(h))
            if aspect < 0.65 or aspect > 1.6:
                continue
            candidates.append((area, x, y, w, h))
        if not candidates:
            return reading
        _area, x, y, w, h = max(candidates, key=lambda row: row[0])
        cx = float(x) + (float(w) / 2.0)
        dist_mm = 14000.0 / max(1.0, float(h))
        x_mm = ((cx - 320.0) * float(dist_mm)) / 620.0
        fallback = dict(reading)
        fallback.update(
            {
                "visible": True,
                "confident": True,
                "conf": 85.0,
                "min_confidence_pct": 45.0,
                "reason": "simple_hsv_fallback_visible",
                "dist_mm": float(dist_mm),
                "x_mm": float(x_mm),
                "y_mm": -0.22 * float(dist_mm),
                "brick_above": False,
                "brick_below": False,
                "simple_hsv_bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            }
        )
        return fallback

    follow._read_brick_measurement = _simple_hsv_reading
    follow._frozen_step12_simple_hsv_installed = True


def _apply_experiment(name: str) -> str:
    follow._set_game_profile("empty")
    cfg = follow._follow_motion_config()
    step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
    step2["seat_mast_duration_ms"] = 0
    if isinstance(step2.get("semi_happy_targets"), dict):
        step2["semi_happy_targets"]["y_mm"] = None
        step2["semi_happy_targets"]["y_tol_mm"] = None
    visibility = cfg.get("visibility_recovery") if isinstance(cfg.get("visibility_recovery"), dict) else {}
    visibility["mast_down_duration_ms"] = 0
    cfg["visibility_recovery"] = visibility
    if name == "strict-y":
        return "Mast frozen; Step 1 and Step 2 gates unchanged except the blind Step 2 mast seat is disabled."
    if name == "short-reset":
        reset_cfg = follow._reset_motion_config()
        reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
        straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
        straight["duration_ms"] = 650
        straight["duration_min_ms"] = 450
        straight["duration_max_ms"] = 800
        reverse["straight_back_first"] = straight
        reset_cfg["reverse_turn"] = reverse
        return (
            "Mast frozen; strict Step 1/Step 2 gates; blind Step 2 mast seat disabled; "
            "reset straight-back shortened to avoid losing detector confidence."
        )
    if name == "balanced-short-reset":
        reset_cfg = follow._reset_motion_config()
        reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
        straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
        straight["duration_ms"] = 650
        straight["duration_min_ms"] = 450
        straight["duration_max_ms"] = 800
        reverse["straight_back_first"] = straight
        reset_cfg["reverse_turn"] = reverse
        return (
            "Mast frozen; strict Step 1/Step 2 gates; blind Step 2 mast seat disabled; "
            "reset straight-back shortened; balanced HSV vision profile for the frozen mast pose."
        )
    if name == "balanced-visible-reset":
        _apply_visible_reset()
        return (
            "Mast frozen; strict Step 1/Step 2 gates; balanced HSV vision; "
            "reset target changed to the visible pose, long straight-back disabled, and reset pause removed."
        )
    if name == "ignore-y-gates":
        _disable_y_gates(cfg, step2)
        return "Mast frozen; Y planning disabled and Step 2 Y gate disabled for this experiment."
    if name == "balanced-visible-yfree-soft":
        _disable_y_gates(cfg, step2)
        _apply_step2_safe_targets(step2)
        _apply_soft_simultaneous_tracking(cfg)
        _apply_visible_reset()
        _install_simple_hsv_fallback()
        return (
            "Mast frozen; balanced HSV; long reset reverse disabled; Y gates disabled; "
            "Step 1 X gate widened for handoff; Step 2 target moved 10mm safer; "
            "X corrections forced into soft simultaneous distance/X bias."
        )
    if name == "honest-locked-y":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_simultaneous_gap_closure(cfg)
        _apply_visible_reset()
        return (
            "Mast/Y frozen; no HSV fallback; reset stays close to Step 1; "
            "Step 1 and Step 2 wins require strict dist and X closure, with Y shown against a locked reference."
        )
    if name == "honest-locked-y-xflip":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_simultaneous_gap_closure(cfg)
        _flip_positive_x_left(cfg)
        _apply_visible_reset(target_abs_x_mm=6.0, x_offset_max_mm=12.0)
        return (
            "Mast/Y frozen; positive X correction turns left; reset must start within 12mm of X center; "
            "Step 1 and Step 2 wins require strict dist and X closure, with Y shown against a locked reference."
        )
    if name == "honest-locked-y-xflip-resetloose":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_simultaneous_gap_closure(cfg)
        _flip_positive_x_left(cfg)
        _apply_visible_reset()
        return (
            "Mast/Y frozen; positive X correction turns left; reset uses the visible distance band; "
            "Step 1 and Step 2 wins require strict dist and X closure, with Y shown against a locked reference."
        )
    if name == "honest-locked-y-xfirst":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _apply_visible_reset()
        return (
            "Mast/Y frozen; positive X correction turns left; Step 1 uses X-first turns instead of blended drive-turns; "
            "Step 1 and Step 2 wins require strict dist and X closure, with Y shown against a locked reference."
        )
    if name == "honest-locked-y-forward-xfirst":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_no_reverse_pursuit_guard()
        _apply_visible_reset()
        return (
            "Mast/Y frozen; positive X correction turns left; pursuit never sends reverse/backward actions; "
            "if Leia is too close, the attempt stops instead of widening the gap."
        )
    if name == "honest-locked-y-forward-xfirst-distreset":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_no_reverse_pursuit_guard()
        _apply_distance_only_reset(dist_offset_mm=20.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; reset only stages distance about 20mm behind Step 1; "
            "pursuit uses forward-only X-first actions and stops instead of reversing when too close."
        )
    if name == "honest-locked-y-forward-xfirst-preflight":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_no_reverse_pursuit_guard()
        _apply_distance_only_reset(dist_offset_mm=20.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; preflight uses only small forward pulses until official vision is confident; "
            "then one forward-only X-first Step 1/2 trial is attempted."
        )
    if name == "honest-locked-y-micro-xfirst":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=110)
        _apply_distance_only_reset(dist_offset_mm=10.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; preflight forward recovery allowed; reset stages only 10mm behind Step 1; "
            "X-first turns are capped to 110ms and pursuit never reverses."
        )
    if name == "honest-locked-y-micro-xfirst-gapguard":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=90)
        _apply_distance_only_reset(dist_offset_mm=10.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; reset uses distance-only nudges with a wrong-way stop; "
            "X-first turns are capped to 90ms and pursuit never reverses."
        )
    if name == "honest-locked-y-guarded-micro":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_target_continuity_guard()
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=80)
        _apply_distance_only_reset(dist_offset_mm=8.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; target-continuity guard rejects adjacent stacks and depth/lateral jumps; "
            "X-first turns are capped to 80ms and pursuit never reverses."
        )
    if name == "honest-locked-y-single-contour":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _flip_positive_x_left(cfg)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=70)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; if official vision fails, one and only one green stack contour may provide transparent motion readings; "
            "reset stages at Step 1 distance, X turns are capped to 70ms, and pursuit never reverses."
        )
    if name == "honest-locked-y-single-contour-right":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=50)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; a single green-stack contour may provide transparent motion readings; "
            "positive X uses right turns capped at the actual send layer to 50ms, and pursuit never reverses."
        )
    if name == "honest-locked-y-single-contour-right-actioncap":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _apply_x_first_policy(cfg)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _install_no_reverse_pursuit_guard()
        _install_micro_x_turns(max_ms=40)
        _install_wheel_turn_action_cap(max_ms=40)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; one green-stack contour may drive transparent readings; "
            "right-turn X probes are capped to 40ms at both planner and robot action layers, and pursuit never reverses."
        )
    if name == "honest-locked-y-direct-right":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller sends one 35ms right-turn pulse at a time and fails immediately "
            "if dist or X gap widens after a pulse."
        )
    if name == "honest-locked-y-direct-left":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller sends one 35ms left-turn pulse at a time and fails immediately "
            "if dist or X gap widens after a pulse."
        )
    if name == "honest-locked-y-direct-right-80":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(80)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller sends one 80ms right-turn pulse at a time and fails immediately "
            "if dist or X gap widens after a pulse."
        )
    if name == "honest-locked-y-direct-right-150":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(150)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller sends one 150ms right-turn pulse at a time and fails immediately "
            "if dist or X gap widens after a pulse."
        )
    if name == "honest-locked-y-direct-curve-right":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _install_wheel_turn_action_cap(max_ms=80)
        _set_direct_x_pulse_ms(80)
        _set_direct_turn_curve_enabled(True)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller uses the turn-curve path with right probes capped to 80ms at the robot layer "
            "and fails immediately if X widens after a pulse."
        )
    if name == "honest-locked-y-direct-curve-left":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _install_wheel_turn_action_cap(max_ms=80)
        _set_direct_x_pulse_ms(80)
        _set_direct_turn_curve_enabled(True)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=10.0)
        return (
            "Mast/Y frozen; direct controller uses the turn-curve path with left probes capped to 80ms at the robot layer "
            "and fails immediately if X widens after a pulse."
        )
    if name == "honest-locked-y-direct-strict-spin-right-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; calibrated contour distance must be within 10mm before X is touched; "
            "X uses a single direct right-spin probe and stops on any X, distance, or target-lock regression."
        )
    if name == "honest-locked-y-direct-contour19000-spin-right-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; contour distance uses the calibrated 19000/height scale and must be within 10mm before X is touched; "
            "X uses a single direct right-spin probe and stops on any X, distance, or target-lock regression."
        )
    if name == "honest-locked-y-direct-contour19000-spin-left-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; contour distance uses the calibrated 19000/height scale and must be within 10mm before X is touched; "
            "X uses a single direct left-spin probe and stops on any X, distance, or target-lock regression."
        )
    if name == "honest-locked-y-direct-contour19000-onewheel-right-120":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(120)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; contour distance uses the calibrated 19000/height scale and must be within 10mm before X is touched; "
            "X uses one left-tread-only right pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-contour19000-onewheel-left-120":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(120)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; contour distance uses the calibrated 19000/height scale and must be within 10mm before X is touched; "
            "X uses one right-tread-only left pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-contour19000-onewheel-left-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_single_green_contour_fallback()
        _install_target_continuity_guard()
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; contour distance uses the calibrated 19000/height scale and must be within 10mm before X is touched; "
            "X uses a stronger right-tread-only left pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-onewheel-left-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(1)
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; uses four stable contour-only reads so distance and X are not mixed between detectors; "
            "X uses a stronger right-tread-only left pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-onewheel-right-240":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(1)
        _set_direct_x_pulse_ms(240)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; uses four stable contour-only reads so distance and X are not mixed between detectors; "
            "X uses a stronger left-tread-only right pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-onewheel-right-360":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; uses four stable contour-only reads so distance and X are not mixed between detectors; "
            "X uses a 360ms left-tread-only right pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_x_pwm(140)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; uses four stable contour-only reads so distance and X are not mixed between detectors; "
            "X uses a 360ms, PWM 140 left-tread-only right pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_x_pwm(140)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; stable contour-only reads ignore far-edge green artifacts; "
            "X uses a 360ms, PWM 140 left-tread-only right pivot and stops unless X improves by at least 1mm."
        )
    if name == "honest-locked-y-direct-stablecontour-adaptive-right-pwm140":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_x_pwm(140)
        _set_direct_x_adaptive_pulse(True)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; stable contour-only reads ignore far-edge green artifacts; "
            "X uses PWM 140 one-wheel pivots with pulse length reduced near target to avoid overshoot."
        )
    if name == "honest-locked-y-direct-adaptive-right-step2-strongdist":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_x_pwm(140)
        _set_direct_x_adaptive_pulse(True)
        _set_direct_step2_dist(pulse_ms=160, max_pulses=30)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; keeps the stable adaptive one-wheel X controller; "
            "Step 2 uses 240ms forward distance pulses and stops on any wrong-way distance change."
        )
    if name == "honest-locked-y-direct-adaptive-right-resetlong-step2strong":
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(4)
        _set_direct_x_pulse_ms(360)
        _set_direct_x_pwm(140)
        _set_direct_x_adaptive_pulse(True)
        _set_direct_reset_dist(pulse_ms=55, max_pulses=20)
        _set_direct_step2_dist(pulse_ms=45, max_pulses=60)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        return (
            "Mast/Y frozen; keeps the stable adaptive one-wheel X controller; "
            "reset gets up to 20 proven distance pulses, and Step 2 uses 240ms forward distance pulses."
        )
    if name in {
        "honest-locked-y-direct-adaptive-right-resetstrong-step2strong",
        "honest-locked-y-direct-adaptive-right-step2-toggle",
    }:
        _disable_y_gates(cfg, step2)
        _apply_step2_strict_movement(step2)
        _install_green_contour_override()
        _install_target_continuity_guard()
        _set_direct_stable_reads(1)
        _set_direct_x_pulse_ms(180)
        _set_direct_x_pwm(140)
        _set_direct_x_adaptive_pulse(True)
        _set_direct_reset_dist(pulse_ms=240, max_pulses=24)
        _set_direct_step2_dist(pulse_ms=240, max_pulses=22)
        _set_direct_turn_curve_enabled(False)
        _set_direct_x_primitive("onewheel_forward")
        _apply_distance_only_reset(dist_offset_mm=0.0, dist_tol_mm=float(DIRECT_STEP1_DIST_TOL_MM))
        if name == "honest-locked-y-direct-adaptive-right-step2-toggle":
            return (
                "Mast/Y frozen; each attempt first proves a centered Step 1 start pose, captures the prior image, "
                "pursues only Step 2 with adaptive distance polish and guarded adaptive PWM 140 X pivots, "
                "then returns to centered Step 1 before the next toggle."
            )
        return (
            "Mast/Y frozen; locked Y is display-only; centered contour filter ignores far-left artifacts; "
            "quick live contour reads replace stopped multi-frame settling; "
            "reset uses the visually dialed closer band plus a direct 5-20mm X-gap opener; "
            "distance and X moves are live-observed slow crawls that stop as soon as the camera sees the band, "
            "with Step 1 X accepting 7mm and Step 2 using adaptive forward polish."
        )
    raise ValueError(f"unknown experiment: {name}")


def _vision_tuning_for_experiment(name: str) -> dict:
    if name in {"balanced-short-reset", "balanced-visible-reset", "balanced-visible-yfree-soft"}:
        tuning = dict(follow.CROWN_PROFILE_TUNING)
        tuning["hsv_lower"] = [62, 97, 33]
        tuning["hsv_upper"] = [87, 255, 255]
        if name == "balanced-visible-yfree-soft":
            tuning["hsv_lower"] = [50, 70, 25]
            tuning["hsv_upper"] = [95, 255, 255]
            tuning["hsv_cyan_coverage_min"] = 0.03
            tuning["full_frame_hsv_cyan_coverage_min"] = 0.03
            tuning["hsv_min_area_ratio"] = 0.015
            tuning["full_frame_hsv_min_area_ratio"] = 0.015
            tuning["conf_gate_pct"] = 45.0
        return tuning
    return dict(follow.CROWN_PROFILE_TUNING)


def _capture_phase(
    site: ProgressSite,
    vision: BrickDetector,
    trial: int,
    attempt: int,
    phase: str,
    status: str,
    reason: str,
    reading: dict | None,
    mast_count: int,
    *,
    step: str | None = None,
    honest_trial: bool | None = None,
) -> None:
    image, summary = site.capture(vision, f"trial{trial}_{attempt}_{phase}", reading)
    evaluation = _evaluate_reading(reading, step)
    if step in {"step1", "step2"}:
        status = "win" if bool(evaluation.get("target_met")) and status == "win" else "fail"
        if status == "fail" and reason:
            reason = str(reason)
    site.add_row(
        {
            "trial": trial,
            "attempt": attempt,
            "phase": phase,
            "step": step,
            "status": status,
            "reason": reason,
            "reading": summary,
            "evaluation": evaluation,
            "target_met": bool(evaluation.get("target_met")) if evaluation else None,
            "honest_trial": honest_trial,
            "mast_attempts": mast_count,
            "image": image,
        }
    )


def _capture_trial_failure(
    site: ProgressSite,
    vision: BrickDetector,
    trial: int,
    attempt: int,
    reason: str,
    reading: dict | None,
    mast_count: int,
) -> None:
    _capture_phase(
        site,
        vision,
        trial,
        attempt,
        "trial",
        "fail",
        reason,
        reading,
        mast_count,
        honest_trial=False,
    )


def _forward_recover_confidence(
    site: ProgressSite,
    vision: BrickDetector,
    robot: MastFrozenRobot,
    attempt: int,
    *,
    max_pulses: int = 8,
) -> dict:
    reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
    _capture_phase(
        site,
        vision,
        0,
        attempt,
        "preflight",
        "info" if bool(reading.get("confident")) else "fail",
        str(reading.get("reason") or "preflight_read"),
        reading,
        len(robot.mast_attempts),
    )
    if bool(reading.get("confident")):
        return reading
    for pulse in range(1, int(max_pulses) + 1):
        reason = str(reading.get("reason") or "")
        if not bool(reading.get("visible")):
            return reading
        if reason not in {"ghost_dist_implausible", "reacquiring_stable_brick_lock"}:
            return reading
        try:
            robot.send_command_pwm("f", 105, duration_ms=160)
            time.sleep(0.24)
            follow._stop_robot(robot)
        finally:
            time.sleep(0.12)
        follow._reset_follow_reading_history(vision)
        reading = follow._wait_for_confident_brick(vision, timeout_s=2.5, sample_s=0.12)
        _capture_phase(
            site,
            vision,
            0,
            attempt,
            f"preflight-fwd-{pulse}",
            "info" if bool(reading.get("confident")) else "fail",
            str(reading.get("reason") or "preflight_forward_recovery"),
            reading,
            len(robot.mast_attempts),
        )
        if bool(reading.get("confident")):
            return reading
    return reading


def _reset_dist_band() -> tuple[float, float]:
    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
    return float(reset_cfg.get("dist_target_mm", follow._dist_target_mm() + 30.0)), float(reset_cfg.get("dist_tol_mm", 10.0))


def _nudge_to_reset_dist_band(vision: BrickDetector, robot: MastFrozenRobot) -> tuple[dict, str, int]:
    target, tol = _reset_dist_band()
    reading = follow._wait_for_confident_brick(vision, timeout_s=6.0, sample_s=0.12)
    acts = 0
    if not bool(reading.get("confident")):
        return reading, "reset_dist_not_confident", acts
    for _ in range(10):
        try:
            dist = float(reading.get("dist_mm"))
        except (TypeError, ValueError):
            return reading, "reset_dist_invalid_reading", acts
        if (target - tol) <= dist <= (target + tol):
            return reading, "reset_dist_band_hit", acts
        err = dist - target
        cmd = "f" if err > 0.0 else "b"
        gap = max(0.0, abs(err) - tol)
        duration_ms = int(max(70, min(130, 55 + gap * 2.0)))
        send_result = follow._drive(
            robot,
            cmd,
            reading,
            pwm=103,
            duration_ms=duration_ms,
        )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            return reading, f"reset_dist_nudge_blocked_{cmd}", acts
        acts += 1
        time.sleep((duration_ms / 1000.0) + 0.18)
        follow._stop_robot(robot)
        follow._reset_follow_reading_history(vision)
        reading = follow._read_brick_measurement(vision)
        if not bool(reading.get("confident")):
            reading = follow._wait_for_visibility_recovery(
                vision,
                robot,
                reading,
                context="reset_dist_nudge",
            )
        if not bool(reading.get("confident")):
            return reading, "reset_dist_lost_visibility", acts
        new_dist = _float_or_none(reading.get("dist_mm"))
        if new_dist is not None:
            new_gap = max(0.0, abs(float(new_dist) - target) - tol)
            if new_gap > gap + 8.0:
                return reading, f"reset_dist_wrong_way_{cmd}", acts
    return reading, "reset_dist_nudge_limit", acts


def _direct_send_motion(robot: MastFrozenRobot, cmd: str, duration_ms: int, *, pwm: int = 100, reading: dict | None = None) -> None:
    cmd_key = str(cmd or "").strip().lower()
    if DIRECT_X_PRIMITIVE == "onewheel_forward" and cmd_key in {"l", "r"}:
        if cmd_key == "r":
            actions = [
                {"target": "l", "action": "b", "pwm": int(pwm), "duration_ms": int(duration_ms)},
                {"target": "r", "action": "s", "duration_ms": int(duration_ms)},
            ]
        else:
            actions = [
                {"target": "l", "action": "s", "duration_ms": int(duration_ms)},
                {"target": "r", "action": "f", "pwm": int(pwm), "duration_ms": int(duration_ms)},
            ]
        robot.send_custom_actions_pwm(cmd_key, actions, duration_ms=int(duration_ms))
    elif DIRECT_USE_TURN_CURVE and cmd_key in {"l", "r"}:
        follow._send_turn_curve(
            robot,
            cmd=cmd_key,
            drive_mode="forward",
            strength="gentle",
            duration_ms=int(duration_ms),
            reading=reading if isinstance(reading, dict) else {},
            context="direct_probe_turn_curve",
            use_production_curve=False,
        )
    else:
        robot.send_command_pwm(cmd, pwm, duration_ms=int(duration_ms))


def _direct_pulse(robot: MastFrozenRobot, cmd: str, duration_ms: int, *, pwm: int = 100, reading: dict | None = None) -> None:
    _direct_send_motion(robot, cmd, duration_ms, pwm=pwm, reading=reading)
    time.sleep((int(duration_ms) / 1000.0) + 0.18 + float(DIRECT_POST_PULSE_EXTRA_SETTLE_S))
    follow._stop_robot(robot)
    time.sleep(0.10)


def _median_float(values: list[float]) -> float | None:
    cleaned = sorted(float(v) for v in values if v is not None)
    if not cleaned:
        return None
    mid = len(cleaned) // 2
    if len(cleaned) % 2:
        return float(cleaned[mid])
    return float((cleaned[mid - 1] + cleaned[mid]) / 2.0)


def _direct_contour_read(vision: BrickDetector) -> dict:
    try:
        vision.read()
    except Exception:
        pass
    frame = _target_guard_frame(vision)
    reading = _green_contour_measurement_from_frame(frame)
    reading["green_contour_override"] = True
    reading["vision_geometry_source"] = "direct_contour"
    return reading


def _direct_read(vision: BrickDetector, *, timeout_s: float = 2.5) -> dict:
    follow._reset_follow_reading_history(vision)
    if int(DIRECT_STABLE_READS) <= 1:
        return follow._wait_for_confident_brick(vision, timeout_s=float(timeout_s), sample_s=0.12)
    samples: list[dict] = []
    last: dict = {}
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline and len(samples) < int(DIRECT_STABLE_READS):
        reading = _direct_contour_read(vision)
        if isinstance(reading, dict):
            last = reading
            if bool(reading.get("confident")):
                samples.append(reading)
        time.sleep(0.12)
    if len(samples) < min(3, int(DIRECT_STABLE_READS)):
        out = dict(last)
        out["confident"] = False
        out["reason"] = "direct_read_insufficient_stable_samples"
        out["direct_stable_read_samples"] = int(len(samples))
        return out
    out = dict(samples[-1])
    out["direct_stable_read_samples"] = int(len(samples))
    for key in ("dist_mm", "x_mm", "y_mm", "conf"):
        vals = [_float_or_none(sample.get(key)) for sample in samples]
        med = _median_float([float(v) for v in vals if v is not None])
        if med is not None:
            out[key] = float(med)
    dist_vals = [_float_or_none(sample.get("dist_mm")) for sample in samples]
    x_vals = [_float_or_none(sample.get("x_mm")) for sample in samples]
    dist_vals = [float(v) for v in dist_vals if v is not None]
    x_vals = [float(v) for v in x_vals if v is not None]
    dist_range = (max(dist_vals) - min(dist_vals)) if dist_vals else 0.0
    x_range = (max(x_vals) - min(x_vals)) if x_vals else 0.0
    out["direct_stable_dist_range_mm"] = float(dist_range)
    out["direct_stable_x_range_mm"] = float(x_range)
    if dist_range > 24.0 or x_range > 8.0:
        out["confident"] = False
        out["reason"] = "direct_read_unstable"
    return out


def _direct_live_observed_move(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    cmd: str,
    duration_ms: int,
    *,
    pwm: int,
    reading: dict,
    stop_when,
) -> tuple[dict, bool]:
    if not bool(DIRECT_LIVE_OBSERVE_MOVES):
        _direct_pulse(robot, cmd, duration_ms, pwm=pwm, reading=reading)
        return _direct_read(vision), False
    _direct_send_motion(robot, cmd, duration_ms, pwm=pwm, reading=reading)
    deadline = time.time() + max(0.05, float(duration_ms) / 1000.0)
    latest = reading if isinstance(reading, dict) else {}
    hit = False
    while time.time() < deadline:
        time.sleep(float(DIRECT_LIVE_SAMPLE_S))
        live = _direct_contour_read(vision)
        if not isinstance(live, dict) or not bool(live.get("confident")):
            continue
        latest = live
        try:
            if bool(stop_when(live)):
                hit = True
                break
        except Exception:
            pass
    follow._stop_robot(robot)
    if float(DIRECT_LIVE_FINAL_SETTLE_S) > 0.0:
        time.sleep(float(DIRECT_LIVE_FINAL_SETTLE_S))
    if hit and bool(latest.get("confident")):
        return latest, True
    final = _direct_contour_read(vision)
    if isinstance(final, dict) and bool(final.get("confident")):
        try:
            if bool(stop_when(final)):
                return final, True
        except Exception:
            pass
        return final, False
    if isinstance(latest, dict) and bool(latest.get("confident")):
        return latest, False
    return final if isinstance(final, dict) else latest, False


def _direct_step1_dist_target() -> float:
    return float(follow._dist_target_mm())


def _honest_reset_dist_target() -> float:
    return float(follow._dist_target_mm()) + (
        (float(HONEST_RESET_DIST_OFFSET_MIN_MM) + float(HONEST_RESET_DIST_OFFSET_MAX_MM)) / 2.0
    )


def _honest_reset_dist_tol() -> float:
    return min(20.0, (float(HONEST_RESET_DIST_OFFSET_MAX_MM) - float(HONEST_RESET_DIST_OFFSET_MIN_MM)) / 2.0)


def _honest_reset_x_target() -> float:
    return float(follow._x_target_mm()) + float(HONEST_RESET_X_TARGET_GAP_MM)


def _honest_reset_x_tol() -> float:
    return min(
        float(HONEST_RESET_X_TARGET_GAP_MM) - float(HONEST_RESET_X_GAP_MIN_MM),
        float(HONEST_RESET_X_GAP_MAX_MM) - float(HONEST_RESET_X_TARGET_GAP_MM),
    )


def _honest_reset_target_met(reading: dict | None) -> bool:
    dist = _float_or_none((reading or {}).get("dist_mm") if isinstance(reading, dict) else None)
    x_val = _float_or_none((reading or {}).get("x_mm") if isinstance(reading, dict) else None)
    if dist is None or x_val is None:
        return False
    step1_dist = float(follow._dist_target_mm())
    dist_offset = float(dist) - step1_dist
    x_offset = float(x_val) - float(follow._x_target_mm())
    return (
        float(HONEST_RESET_DIST_OFFSET_MIN_MM) <= dist_offset <= float(HONEST_RESET_DIST_OFFSET_MAX_MM)
        and float(HONEST_RESET_X_GAP_MIN_MM) <= x_offset <= float(HONEST_RESET_X_GAP_MAX_MM)
    )


def _honest_reset_x_gap(reading: dict | None) -> float | None:
    if not isinstance(reading, dict):
        return None
    x_val = _float_or_none(reading.get("x_mm"))
    if x_val is None:
        return None
    return abs(float(x_val) - float(follow._x_target_mm()))


def _honest_reset_x_offset(reading: dict | None) -> float | None:
    if not isinstance(reading, dict):
        return None
    x_val = _float_or_none(reading.get("x_mm"))
    if x_val is None:
        return None
    return float(x_val) - float(follow._x_target_mm())


def _honest_reset_reason(reading: dict | None) -> str:
    dist = _float_or_none((reading or {}).get("dist_mm") if isinstance(reading, dict) else None)
    x_val = _float_or_none((reading or {}).get("x_mm") if isinstance(reading, dict) else None)
    if dist is None or x_val is None:
        return "honest_reset_reading_invalid"
    dist_offset = float(dist) - float(follow._dist_target_mm())
    x_offset = float(x_val) - float(follow._x_target_mm())
    return f"honest_reset_dist_offset_{dist_offset:.1f}_x_offset_{x_offset:.1f}"


def _direct_open_honest_reset_x_gap(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    reading: dict,
    *,
    positive_cmd: str,
    dist_target: float,
    dist_tol: float,
    label: str,
) -> tuple[bool, str, dict]:
    current = (
        reading
        if isinstance(reading, dict) and _float_or_none(reading.get("x_mm")) is not None
        else _direct_read(vision)
    )
    last_cmd = "l"
    for attempt in range(8):
        offset = _honest_reset_x_offset(current)
        if offset is None:
            return False, f"{label}_reset_x_gap_invalid", current
        if float(HONEST_RESET_X_GAP_MIN_MM) <= float(offset) <= float(HONEST_RESET_X_GAP_MAX_MM):
            return True, f"{label}_reset_x_gap_hit", current
        if float(offset) > float(HONEST_RESET_X_GAP_MAX_MM):
            x_val = _float_or_none(current.get("x_mm"))
            if x_val is None:
                return False, f"{label}_reset_x_gap_too_large_{float(offset):.1f}", current
            center = float(follow._x_target_mm())
            ok, reason, current = _direct_close_x(
                vision,
                robot,
                positive_cmd=positive_cmd,
                dist_target=dist_target,
                dist_tol=dist_tol,
                x_target=center + float(HONEST_RESET_X_TARGET_GAP_MM),
                x_tol=_honest_reset_x_tol(),
                min_pulse_ms=int(HONEST_RESET_X_SNAP_MIN_MS),
                pwm=150,
                max_pulses=8,
                label=f"{label}_reset_x_contract",
                initial_reading=current,
            )
            if not ok:
                return False, reason, current
            continue

        old_offset = float(offset)
        cmd = "l" if old_offset < -5.0 else last_cmd

        def gap_open(live: dict) -> bool:
            live_offset = _honest_reset_x_offset(live)
            return live_offset is not None and float(live_offset) >= float(HONEST_RESET_X_GAP_MIN_MM)

        current, live_hit = _direct_live_observed_move(
            vision,
            robot,
            cmd,
            int(HONEST_RESET_X_OPEN_MS),
            pwm=150,
            reading=current,
            stop_when=gap_open,
        )
        if not bool(current.get("confident")):
            current = _direct_read(vision, timeout_s=4.0)
            if not bool(current.get("confident")) and _float_or_none(current.get("x_mm") if isinstance(current, dict) else None) is None:
                return False, f"{label}_reset_x_gap_lost_confidence_after_{cmd}", current

        new_offset = _honest_reset_x_offset(current)
        if new_offset is None:
            return False, f"{label}_reset_x_gap_invalid_after_{cmd}", current
        if float(HONEST_RESET_X_GAP_MIN_MM) <= float(new_offset) <= float(HONEST_RESET_X_GAP_MAX_MM):
            suffix = "live_hit" if live_hit else "hit"
            return True, f"{label}_reset_x_gap_{suffix}", current
        if float(new_offset) > float(HONEST_RESET_X_GAP_MAX_MM):
            continue
        if float(new_offset) <= old_offset + 0.35:
            last_cmd = "r" if cmd == "l" else "l"
        else:
            last_cmd = cmd
    return False, f"{label}_reset_x_gap_limit", current


def _direct_dist_pulse_ms(label: str, old_gap: float, tol: float) -> int:
    pulse_ms = int(DIRECT_STEP2_DIST_PULSE_MS) if str(label).startswith("step2") else int(DIRECT_DIST_PULSE_MS)
    outside = max(0.0, float(old_gap) - float(tol))
    if bool(DIRECT_LIVE_OBSERVE_MOVES):
        if "reset" in str(label):
            if outside > 12.0:
                pulse_ms = max(pulse_ms, int(DIRECT_LIVE_RESET_DIST_CRAWL_MS))
            elif outside > 6.0:
                pulse_ms = max(pulse_ms, 300)
            elif outside > 2.0:
                pulse_ms = max(pulse_ms, 220)
        elif str(label).startswith("step2"):
            if outside > 12.0:
                pulse_ms = max(pulse_ms, int(DIRECT_LIVE_STEP2_DIST_CRAWL_MS))
            elif outside > 6.0:
                pulse_ms = max(pulse_ms, 60)
            elif outside > 2.0:
                pulse_ms = max(pulse_ms, 45)
        elif outside > 20.0:
            pulse_ms = max(pulse_ms, int(DIRECT_LIVE_DIST_CRAWL_MS))
        elif outside > 6.0:
            pulse_ms = max(pulse_ms, 650)
        elif outside > 2.0:
            pulse_ms = max(pulse_ms, 360)
    elif str(label).startswith("step2") or "reset" in str(label):
        if outside <= 2.0:
            pulse_ms = min(pulse_ms, 80)
        elif outside <= 6.0:
            pulse_ms = min(pulse_ms, 140)
        elif "reset" in str(label) and outside <= 20.0:
            pulse_ms = min(pulse_ms, 180)
        elif "reset" in str(label) and outside <= 35.0:
            pulse_ms = min(pulse_ms, 260)
    return int(pulse_ms)


def _direct_dist_live_stop_margin(label: str, cmd: str) -> float:
    if str(label).startswith("step2") and str(cmd).strip().lower() == "f":
        return float(DIRECT_STEP2_DIST_EARLY_STOP_MARGIN_MM)
    return 0.0


def _direct_step2_dist_x_unsafe(reading: dict | None) -> bool:
    if not isinstance(reading, dict):
        return False
    x_val = _float_or_none(reading.get("x_mm"))
    if x_val is None:
        return False
    return abs(float(x_val) - float(follow._x_target_mm())) > float(DIRECT_STEP2_DIST_LIVE_MAX_X_ERR_MM)


def _reading_green_box(reading: dict | None) -> dict | None:
    if not isinstance(reading, dict):
        return None
    candidates = reading.get("green_stack_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return None
    box = candidates[0]
    return box if isinstance(box, dict) else None


def _box_float(box: dict | None, key: str) -> float | None:
    if not isinstance(box, dict):
        return None
    try:
        return float(box.get(key))
    except (TypeError, ValueError):
        return None


def _direct_x_box_guard(reading: dict, *, label: str, previous_reading: dict | None = None) -> str | None:
    box = _reading_green_box(reading)
    if box is None:
        return f"{label}_x_box_unavailable"
    cx = _box_float(box, "cx")
    if cx is None:
        return f"{label}_x_box_cx_invalid"
    if cx < float(DIRECT_X_MIN_SAFE_CX_PX) or cx > float(DIRECT_X_MAX_SAFE_CX_PX):
        return f"{label}_x_box_adjacent_risk_cx_{cx:.1f}"
    prev_box = _reading_green_box(previous_reading)
    if prev_box is None:
        return None
    prev_cx = _box_float(prev_box, "cx")
    if prev_cx is not None and abs(float(cx) - float(prev_cx)) > float(DIRECT_X_MAX_CX_JUMP_PX):
        return f"{label}_x_box_identity_jump_cx_{prev_cx:.1f}_to_{cx:.1f}"
    for key in ("w", "h"):
        prev_val = _box_float(prev_box, key)
        cur_val = _box_float(box, key)
        if prev_val is None or cur_val is None or prev_val <= 0:
            continue
        scale_delta = abs(float(cur_val) - float(prev_val)) / float(prev_val)
        if scale_delta > float(DIRECT_X_MAX_BOX_SCALE_JUMP):
            return f"{label}_x_box_identity_jump_{key}_{prev_val:.1f}_to_{cur_val:.1f}"
    return None


def _direct_x_pose_guard(
    reading: dict,
    *,
    label: str,
    dist_target: float,
    dist_tol: float,
    previous_reading: dict | None = None,
) -> str | None:
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return f"{label}_x_not_confident"
    if bool(reading.get("target_guard_blocked")):
        return f"{label}_x_target_guard_{reading.get('target_guard_reason') or 'blocked'}"
    candidate_count = reading.get("green_stack_candidate_count")
    if candidate_count is not None:
        try:
            if int(candidate_count) != 1:
                return f"{label}_x_candidate_count_{int(candidate_count)}"
        except (TypeError, ValueError):
            return f"{label}_x_candidate_count_invalid"
    dist = _float_or_none(reading.get("dist_mm"))
    if dist is None:
        return f"{label}_x_dist_invalid"
    guard_tol = float(dist_tol) if str(label).startswith("step2") else max(float(dist_tol), float(DIRECT_X_DIST_GUARD_MM))
    if abs(float(dist) - float(dist_target)) > guard_tol:
        return f"{label}_x_unsafe_dist_{float(dist):.1f}"
    x_val = _float_or_none(reading.get("x_mm"))
    if x_val is None:
        return f"{label}_x_invalid"
    max_abs_err = 80.0 if "reset_x" in str(label) else float(DIRECT_X_MAX_ABS_ERR_MM)
    if abs(float(x_val) - float(follow._x_target_mm())) > float(max_abs_err):
        return f"{label}_x_too_far_for_direct_{float(x_val):.1f}"
    box_reason = _direct_x_box_guard(reading, label=label, previous_reading=previous_reading)
    if box_reason is not None:
        return box_reason
    return None


def _direct_reading_inside_dist_x(reading: dict, *, dist_target: float, dist_tol: float, x_target: float, x_tol: float) -> bool:
    dist = _float_or_none(reading.get("dist_mm") if isinstance(reading, dict) else None)
    x_val = _float_or_none(reading.get("x_mm") if isinstance(reading, dict) else None)
    if dist is None or x_val is None:
        return False
    return abs(float(dist) - float(dist_target)) <= float(dist_tol) and abs(float(x_val) - float(x_target)) <= float(x_tol)


def _direct_close_dist(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    target: float,
    tol: float,
    *,
    allow_back: bool,
    max_pulses: int,
    label: str,
    initial_reading: dict | None = None,
) -> tuple[bool, str, dict]:
    initial_dist = _float_or_none(initial_reading.get("dist_mm") if isinstance(initial_reading, dict) else None)
    use_initial = isinstance(initial_reading, dict) and (
        bool(initial_reading.get("confident")) or ("reset" in str(label) and initial_dist is not None)
    )
    reading = initial_reading if use_initial else _direct_read(vision)
    if not bool(reading.get("confident")) and not ("reset" in str(label) and _float_or_none(reading.get("dist_mm")) is not None):
        return False, f"{label}_not_confident", reading
    for _ in range(int(max_pulses)):
        dist = _float_or_none(reading.get("dist_mm"))
        if dist is None:
            return False, f"{label}_dist_invalid", reading
        old_gap = abs(float(dist) - float(target))
        if old_gap <= float(tol):
            return True, f"{label}_dist_hit", reading
        cmd = "f" if float(dist) > float(target) else "b"
        if cmd == "b" and not bool(allow_back):
            return False, f"{label}_reverse_blocked", reading
        pulse_ms = _direct_dist_pulse_ms(label, old_gap, float(tol))
        def dist_hit(live: dict) -> bool:
            live_dist = _float_or_none(live.get("dist_mm"))
            if live_dist is None:
                return False
            if str(label).startswith("step2") and _direct_step2_dist_x_unsafe(live):
                return True
            if abs(float(live_dist) - float(target)) <= float(tol):
                return True
            margin = _direct_dist_live_stop_margin(label, cmd)
            if margin <= 0.0:
                return False
            if cmd == "f":
                return float(live_dist) <= float(target) + float(tol) + float(margin)
            if cmd == "b":
                return float(live_dist) >= float(target) - float(tol) - float(margin)
            return False

        next_reading, live_hit = _direct_live_observed_move(
            vision,
            robot,
            cmd,
            pulse_ms,
            pwm=int(DIRECT_STEP2_DIST_PWM) if str(label).startswith("step2") else 98,
            reading=reading,
            stop_when=dist_hit,
        )
        if not bool(next_reading.get("confident")):
            time.sleep(0.8)
            retry_reading = _direct_read(vision, timeout_s=8.0)
            if not bool(retry_reading.get("confident")):
                retry_dist = _float_or_none(retry_reading.get("dist_mm") if isinstance(retry_reading, dict) else None)
                if "reset" in str(label) and retry_dist is not None:
                    next_reading = retry_reading
                else:
                    return False, f"{label}_lost_confidence_after_{cmd}", retry_reading
            else:
                next_reading = retry_reading
        next_dist = _float_or_none(next_reading.get("dist_mm"))
        if next_dist is None:
            return False, f"{label}_dist_invalid_after_{cmd}", next_reading
        if str(label).startswith("step2") and _direct_step2_dist_x_unsafe(next_reading):
            next_x = _float_or_none(next_reading.get("x_mm"))
            x_text = "invalid" if next_x is None else f"{float(next_x):.1f}"
            return False, f"{label}_dist_x_unsafe_{x_text}_after_{cmd}", next_reading
        new_gap = abs(float(next_dist) - float(target))
        if live_hit and new_gap <= float(tol):
            return True, f"{label}_dist_live_hit", next_reading
        wrong_way_slack = 12.0 if "reset" in str(label) else 4.0
        if new_gap > old_gap + wrong_way_slack:
            confirm = _direct_read(vision, timeout_s=2.0)
            confirm_dist = _float_or_none(confirm.get("dist_mm") if isinstance(confirm, dict) else None)
            if confirm_dist is not None:
                confirm_gap = abs(float(confirm_dist) - float(target))
                if confirm_gap <= old_gap + wrong_way_slack:
                    reading = confirm
                    continue
                next_reading = confirm
            return False, f"{label}_dist_wrong_way_{cmd}", next_reading
        reading = next_reading
    return False, f"{label}_dist_pulse_limit", reading


def _direct_close_x(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    positive_cmd: str,
    dist_target: float,
    dist_tol: float,
    x_target: float | None = None,
    x_tol: float | None = None,
    min_pulse_ms: int | None = None,
    pwm: int | None = None,
    max_pulses: int,
    label: str,
    initial_reading: dict | None = None,
) -> tuple[bool, str, dict]:
    target = float(follow._x_target_mm() if x_target is None else x_target)
    tol = float(follow._x_tol_mm() if x_tol is None else x_tol)
    initial_x = _float_or_none(initial_reading.get("x_mm") if isinstance(initial_reading, dict) else None)
    initial_dist = _float_or_none(initial_reading.get("dist_mm") if isinstance(initial_reading, dict) else None)
    numeric_reset_contract = "reset_x_contract" in str(label) and initial_x is not None and initial_dist is not None
    reading = initial_reading if isinstance(initial_reading, dict) and (bool(initial_reading.get("confident")) or numeric_reset_contract) else _direct_read(vision)
    if not bool(reading.get("confident")) and not numeric_reset_contract:
        return False, f"{label}_not_confident", reading
    if bool(reading.get("confident")):
        guard_reason = _direct_x_pose_guard(reading, label=label, dist_target=dist_target, dist_tol=dist_tol)
        if guard_reason is not None:
            return False, guard_reason, reading
    positive = str(positive_cmd or "r").strip().lower()
    if positive not in {"l", "r"}:
        positive = "r"
    negative = "l" if positive == "r" else "r"
    stall_retries = 0
    wrong_way_recoveries = 0
    override_cmd: str | None = None
    for _ in range(int(max_pulses)):
        x_val = _float_or_none(reading.get("x_mm"))
        if x_val is None:
            return False, f"{label}_x_invalid", reading
        old_err = float(x_val) - target
        if abs(old_err) <= tol:
            return True, f"{label}_x_hit", reading
        cmd = override_cmd or (positive if old_err > 0.0 else negative)
        override_cmd = None
        pulse_ms = int(DIRECT_X_PULSE_MS)
        if bool(DIRECT_X_ADAPTIVE_PULSE):
            outside = max(0.0, abs(float(old_err)) - float(tol))
            if outside <= 2.0:
                pulse_ms = min(pulse_ms, 80)
            elif outside <= 6.0:
                pulse_ms = min(pulse_ms, 140)
            elif outside <= 12.0:
                pulse_ms = min(pulse_ms, 220)
        if min_pulse_ms is not None:
            outside_for_snap = max(0.0, abs(float(old_err)) - float(tol))
            if outside_for_snap <= float(DIRECT_X_SNAP_OUTSIDE_MAX_MM):
                pulse_ms = max(int(pulse_ms), int(min_pulse_ms))
        if "reset_x_contract" in str(label):
            pulse_ms = min(int(pulse_ms), int(HONEST_RESET_X_CONTRACT_MS))
        elif "step2_prex" in str(label):
            pulse_ms = min(max(int(pulse_ms), 60), 110)
        elif str(label).startswith(("step1", "step2")) and bool(DIRECT_LIVE_OBSERVE_MOVES):
            outside = max(0.0, abs(float(old_err)) - float(tol))
            if outside <= 4.0:
                pulse_ms = max(int(pulse_ms), 120)
            elif outside <= 12.0:
                pulse_ms = max(int(pulse_ms), 180)
            elif outside <= 20.0:
                pulse_ms = max(int(pulse_ms), 240)
            else:
                pulse_ms = max(int(pulse_ms), int(DIRECT_LIVE_X_CRAWL_MS))
        elif bool(DIRECT_LIVE_OBSERVE_MOVES):
            outside = max(0.0, abs(float(old_err)) - float(tol))
            if outside > 12.0:
                pulse_ms = max(int(pulse_ms), int(DIRECT_LIVE_X_CRAWL_MS))
            elif outside > 6.0:
                pulse_ms = max(int(pulse_ms), 500)
            elif outside > 2.0:
                pulse_ms = max(int(pulse_ms), 300)
        pulse_pwm = int(DIRECT_X_PWM if pwm is None else pwm)
        def x_hit(live: dict) -> bool:
            live_x = _float_or_none(live.get("x_mm"))
            return live_x is not None and abs(float(live_x) - float(target)) <= float(tol)

        next_reading, live_hit = _direct_live_observed_move(
            vision,
            robot,
            cmd,
            int(pulse_ms),
            pwm=pulse_pwm,
            reading=reading,
            stop_when=x_hit,
        )
        if not bool(next_reading.get("confident")):
            time.sleep(0.8)
            retry_reading = _direct_read(vision, timeout_s=8.0)
            retry_numeric_contract = (
                "reset_x_contract" in str(label)
                and _float_or_none(retry_reading.get("x_mm") if isinstance(retry_reading, dict) else None) is not None
                and _float_or_none(retry_reading.get("dist_mm") if isinstance(retry_reading, dict) else None) is not None
            )
            if not bool(retry_reading.get("confident")) and not retry_numeric_contract:
                return False, f"{label}_lost_confidence_after_{cmd}", retry_reading
            next_reading = retry_reading
        if bool(next_reading.get("confident")):
            guard_reason = _direct_x_pose_guard(
                next_reading,
                label=label,
                dist_target=dist_target,
                dist_tol=dist_tol,
                previous_reading=reading,
            )
        if guard_reason is not None and not _direct_reading_inside_dist_x(
            next_reading,
            dist_target=dist_target,
            dist_tol=dist_tol,
            x_target=target,
            x_tol=tol,
        ):
            if str(label).startswith("step1") and "too_far_for_direct" in str(guard_reason) and wrong_way_recoveries < 1:
                wrong_way_recoveries += 1
                reading = next_reading
                override_cmd = "l" if cmd == "r" else "r"
                continue
            return False, f"{guard_reason}_after_{cmd}", next_reading
        next_x = _float_or_none(next_reading.get("x_mm"))
        if next_x is None:
            return False, f"{label}_x_invalid_after_{cmd}", next_reading
        next_dist = _float_or_none(next_reading.get("dist_mm"))
        old_dist = _float_or_none(reading.get("dist_mm"))
        if next_dist is None or old_dist is None:
            return False, f"{label}_dist_invalid_after_{cmd}", next_reading
        if _direct_reading_inside_dist_x(
            next_reading,
            dist_target=dist_target,
            dist_tol=dist_tol,
            x_target=target,
            x_tol=tol,
        ):
            return True, f"{label}_x_hit", next_reading
        if abs(float(next_dist) - float(old_dist)) > float(DIRECT_X_MAX_DIST_DRIFT_MM):
            return False, f"{label}_x_dist_drift_after_{cmd}", next_reading
        new_err = float(next_x) - target
        if live_hit and abs(new_err) <= float(tol):
            return True, f"{label}_x_live_hit", next_reading
        can_flip_once = str(label).startswith(("step1", "step2")) or "reset_x_contract" in str(label)
        if abs(new_err) > abs(old_err) + 2.0:
            if can_flip_once and wrong_way_recoveries < 1:
                wrong_way_recoveries += 1
                reading = next_reading
                override_cmd = "l" if cmd == "r" else "r"
                continue
            return False, f"{label}_x_wrong_way_{cmd}", next_reading
        if abs(new_err) > abs(old_err) - float(DIRECT_X_MIN_IMPROVEMENT_MM):
            if can_flip_once and wrong_way_recoveries < 1:
                wrong_way_recoveries += 1
                reading = next_reading
                override_cmd = "l" if cmd == "r" else "r"
                continue
            if abs(new_err) <= abs(old_err) + float(DIRECT_X_STALL_RETRY_SLACK_MM) and stall_retries < 2:
                stall_retries += 1
                reading = next_reading
                override_cmd = "l" if cmd == "r" else "r"
                continue
            return False, f"{label}_x_no_progress_{cmd}", next_reading
        stall_retries = 0
        reading = next_reading
    return False, f"{label}_x_pulse_limit", reading


def _direct_step2_target_values() -> tuple[float, float, float, float]:
    step2_targets = follow._follow_step2_config().get("targets")
    step2_targets = step2_targets if isinstance(step2_targets, dict) else {}
    return (
        float(step2_targets.get("dist_mm", 149.0)),
        float(step2_targets.get("dist_tol_mm", 5.0)),
        float(step2_targets.get("x_mm", follow._x_target_mm())),
        float(step2_targets.get("x_tol_mm", follow._x_tol_mm())),
    )


def _direct_close_step1_pose(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    positive_cmd: str,
    label: str,
    dist_tol: float | None = None,
    initial_reading: dict | None = None,
) -> tuple[bool, str, dict]:
    step1_dist_target = _direct_step1_dist_target()
    step1_dist_tol = float(DIRECT_STEP1_DIST_TOL_MM if dist_tol is None else dist_tol)
    ok, reason, reading = _direct_close_dist(
        vision,
        robot,
        step1_dist_target,
        step1_dist_tol,
        allow_back=True,
        max_pulses=int(DIRECT_RESET_DIST_MAX_PULSES),
        label=f"{label}_dist",
        initial_reading=initial_reading,
    )
    if ok and bool(_evaluate_reading(reading, "step1").get("target_met")):
        return True, f"{label}_hit", reading
    if ok:
        ok, reason, reading = _direct_close_x(
            vision,
            robot,
            positive_cmd=positive_cmd,
            dist_target=step1_dist_target,
            dist_tol=step1_dist_tol,
            x_target=float(follow._x_target_mm()),
            x_tol=float(DIRECT_STEP1_X_TOL_MM),
            min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
            max_pulses=6,
            label=f"{label}_x",
            initial_reading=reading,
        )
    return ok, reason, reading


def _direct_close_honest_reset_pose(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    positive_cmd: str,
    label: str,
    initial_reading: dict | None = None,
) -> tuple[bool, str, dict]:
    reset_dist_target = _honest_reset_dist_target()
    reset_dist_tol = _honest_reset_dist_tol()
    ok, reason, reading = _direct_close_dist(
        vision,
        robot,
        reset_dist_target,
        reset_dist_tol,
        allow_back=True,
        max_pulses=int(DIRECT_RESET_DIST_MAX_PULSES),
        label=f"{label}_reset_dist",
        initial_reading=initial_reading,
    )
    if ok:
        ok, reason, reading = _direct_open_honest_reset_x_gap(
            vision,
            robot,
            reading,
            positive_cmd=positive_cmd,
            dist_target=reset_dist_target,
            dist_tol=reset_dist_tol,
            label=label,
        )
    if not _honest_reset_target_met(reading):
        x_offset = _honest_reset_x_offset(reading)
        if x_offset is not None and (
            float(x_offset) < float(HONEST_RESET_X_GAP_MIN_MM)
            or float(x_offset) > float(HONEST_RESET_X_GAP_MAX_MM)
        ):
            ok, reason, reading = _direct_close_x(
                vision,
                robot,
                positive_cmd=positive_cmd,
                dist_target=reset_dist_target,
                dist_tol=reset_dist_tol,
                x_target=_honest_reset_x_target(),
                x_tol=_honest_reset_x_tol(),
                min_pulse_ms=int(HONEST_RESET_X_SNAP_MIN_MS),
                pwm=150,
                max_pulses=16,
                label=f"{label}_reset_x_contract_final",
                initial_reading=reading,
            )
    if not _honest_reset_target_met(reading):
        x_offset = _honest_reset_x_offset(reading)
        if x_offset is not None and float(x_offset) < float(HONEST_RESET_X_GAP_MIN_MM):
            ok, reason, reading = _direct_open_honest_reset_x_gap(
                vision,
                robot,
                reading,
                positive_cmd=positive_cmd,
                dist_target=reset_dist_target,
                dist_tol=reset_dist_tol,
                label=f"{label}_reset_x_reopen",
            )
    if not _honest_reset_target_met(reading):
        dist = _float_or_none(reading.get("dist_mm") if isinstance(reading, dict) else None)
        x_offset = _honest_reset_x_offset(reading)
        if dist is not None and x_offset is not None and float(HONEST_RESET_X_GAP_MIN_MM) <= x_offset <= float(HONEST_RESET_X_GAP_MAX_MM):
            ok, reason, reading = _direct_close_dist(
                vision,
                robot,
                reset_dist_target,
                reset_dist_tol,
                allow_back=True,
                max_pulses=8,
                label=f"{label}_reset_dist_polish",
                initial_reading=reading,
            )
    if not _honest_reset_target_met(reading):
        x_offset = _honest_reset_x_offset(reading)
        if x_offset is not None and float(x_offset) < float(HONEST_RESET_X_GAP_MIN_MM):
            ok, reason, reading = _direct_open_honest_reset_x_gap(
                vision,
                robot,
                reading,
                positive_cmd=positive_cmd,
                dist_target=reset_dist_target,
                dist_tol=reset_dist_tol,
                label=f"{label}_reset_x_final_reopen",
            )
        elif x_offset is not None and float(x_offset) > float(HONEST_RESET_X_GAP_MAX_MM):
            ok, reason, reading = _direct_close_x(
                vision,
                robot,
                positive_cmd=positive_cmd,
                dist_target=reset_dist_target,
                dist_tol=reset_dist_tol,
                x_target=_honest_reset_x_target(),
                x_tol=_honest_reset_x_tol(),
                min_pulse_ms=int(HONEST_RESET_X_SNAP_MIN_MS),
                pwm=150,
                max_pulses=8,
                label=f"{label}_reset_x_final_contract",
                initial_reading=reading,
            )
    if _honest_reset_target_met(reading):
        return True, _honest_reset_reason(reading), reading
    return False, _honest_reset_reason(reading) if ok else reason, reading


def _direct_close_step2_pose(
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    positive_cmd: str,
    label: str = "step2",
    initial_reading: dict | None = None,
) -> tuple[bool, str, dict]:
    step2_dist_target, step2_dist_tol, step2_x_target, step2_x_tol = _direct_step2_target_values()
    reading = initial_reading if isinstance(initial_reading, dict) and bool(initial_reading.get("confident")) else _direct_read(vision)
    x_now = _float_or_none(reading.get("x_mm") if isinstance(reading, dict) else None)
    dist_now = _float_or_none(reading.get("dist_mm") if isinstance(reading, dict) else None)
    x_prealign_target = step2_x_target
    x_prealign_tol = step2_x_tol
    if dist_now is not None and float(dist_now) > float(step2_dist_target) + float(step2_dist_tol):
        x_prealign_target = float(DIRECT_STEP2_FORWARD_X_PREBIAS_MM)
        x_prealign_tol = 6.0
    if x_now is not None and dist_now is not None and abs(float(x_now) - float(x_prealign_target)) > float(x_prealign_tol):
        ok, reason, reading = _direct_close_x(
            vision,
            robot,
            positive_cmd=positive_cmd,
            dist_target=float(dist_now),
            dist_tol=35.0,
            x_target=x_prealign_target,
            x_tol=x_prealign_tol,
            min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
            max_pulses=3,
            label=f"{label}_prex",
            initial_reading=reading,
        )
        if not ok:
            return ok, reason, reading
    ok, reason, reading = _direct_close_dist(
        vision,
        robot,
        step2_dist_target,
        step2_dist_tol,
        allow_back=False,
        max_pulses=int(DIRECT_STEP2_DIST_MAX_PULSES),
        label=label,
        initial_reading=reading,
    )
    if not ok and "dist_x_unsafe" in str(reason):
        rescue_dist = _float_or_none(reading.get("dist_mm") if isinstance(reading, dict) else None)
        rescue_ok, rescue_reason, rescue_reading = _direct_close_x(
            vision,
            robot,
            positive_cmd=positive_cmd,
            dist_target=float(rescue_dist) if rescue_dist is not None else step2_dist_target,
            dist_tol=45.0,
            x_target=float(DIRECT_STEP2_FORWARD_X_PREBIAS_MM),
            x_tol=6.0,
            min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
            max_pulses=4,
            label=f"{label}_dist_x_rescue",
            initial_reading=reading,
        )
        reading = rescue_reading
        if rescue_ok:
            ok, reason, reading = _direct_close_dist(
                vision,
                robot,
                step2_dist_target,
                step2_dist_tol,
                allow_back=False,
                max_pulses=max(2, int(DIRECT_STEP2_DIST_MAX_PULSES) // 2),
                label=f"{label}_dist_retry",
                initial_reading=reading,
            )
        else:
            reason = f"{reason};{rescue_reason}"
    if ok:
        ok, reason, reading = _direct_close_x(
            vision,
            robot,
            positive_cmd=positive_cmd,
            dist_target=step2_dist_target,
            dist_tol=step2_dist_tol,
            x_target=step2_x_target,
            x_tol=step2_x_tol,
            min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
            max_pulses=4,
            label=label,
            initial_reading=reading,
        )
        if not ok and ("x_wrong_way" in str(reason) or "x_no_progress" in str(reason)):
            rescue_positive = "l" if str(positive_cmd).strip().lower() == "r" else "r"
            rescue_ok, rescue_reason, rescue_reading = _direct_close_x(
                vision,
                robot,
                positive_cmd=rescue_positive,
                dist_target=step2_dist_target,
                dist_tol=step2_dist_tol,
                x_target=step2_x_target,
                x_tol=step2_x_tol,
                min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
                max_pulses=3,
                label=f"{label}_x_reverse_rescue",
                initial_reading=reading,
            )
            if rescue_ok:
                ok, reason, reading = rescue_ok, rescue_reason, rescue_reading
            else:
                reading = rescue_reading
                reason = f"{reason};{rescue_reason}"
    if ok:
        for polish_round in range(2):
            step2_eval_now = _evaluate_reading(reading, "step2")
            if bool(step2_eval_now.get("target_met")):
                break
            axes = step2_eval_now.get("axes") if isinstance(step2_eval_now, dict) else {}
            dist_axis = axes.get("dist") if isinstance(axes, dict) else {}
            x_axis = axes.get("x") if isinstance(axes, dict) else {}
            if isinstance(dist_axis, dict) and not bool(dist_axis.get("ok")):
                ok, reason, reading = _direct_close_dist(
                    vision,
                    robot,
                    step2_dist_target,
                    step2_dist_tol,
                    allow_back=False,
                    max_pulses=2,
                    label=f"{label}_polish{polish_round + 1}",
                    initial_reading=reading,
                )
                if not ok:
                    break
                continue
            if isinstance(x_axis, dict) and not bool(x_axis.get("ok")):
                ok, reason, reading = _direct_close_x(
                    vision,
                    robot,
                    positive_cmd=positive_cmd,
                    dist_target=step2_dist_target,
                    dist_tol=step2_dist_tol,
                    x_target=step2_x_target,
                    x_tol=step2_x_tol,
                    min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
                    max_pulses=2,
                    label=f"{label}_polish{polish_round + 1}",
                    initial_reading=reading,
                )
                if not ok:
                    break
                continue
            break
    return ok, reason, reading


def _run_direct_single_trial(
    site: ProgressSite,
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    trial: int,
    attempt: int,
    positive_cmd: str,
    initial_reading: dict | None = None,
    prefer_current_reset: bool = False,
) -> int:
    ok = False
    reason = "prior_reset_pending"
    reading = initial_reading if isinstance(initial_reading, dict) else None
    if bool(prefer_current_reset):
        reading = _direct_read(vision, timeout_s=3.0)
        if _honest_reset_target_met(reading):
            ok = True
            reason = _honest_reset_reason(reading)
    if not ok:
        ok, reason, reading = _direct_close_honest_reset_pose(
            vision,
            robot,
            positive_cmd=positive_cmd,
            label="prior",
            initial_reading=reading,
        )
    _capture_phase(site, vision, trial, attempt, "prior", "win" if ok else "fail", reason, reading, len(robot.mast_attempts))
    if not ok:
        _capture_trial_failure(site, vision, trial, attempt, f"trial incomplete: {reason}", reading, len(robot.mast_attempts))
        site.update_summary(f"Stopped after trial {trial} honest reset failure")
        return 2
    step1_dist_target = _direct_step1_dist_target()
    step1_dist_tol = float(DIRECT_STEP1_DIST_TOL_MM)
    ok, reason, reading = _direct_close_dist(
        vision,
        robot,
        step1_dist_target,
        min(step1_dist_tol, float(DIRECT_STEP1_PREALIGN_DIST_TOL_MM)),
        allow_back=False,
        max_pulses=int(DIRECT_RESET_DIST_MAX_PULSES),
        label="step1",
        initial_reading=reading,
    )
    if not ok:
        _capture_phase(site, vision, trial, attempt, "step1", "fail", reason, reading, len(robot.mast_attempts), step="step1")
        _capture_trial_failure(site, vision, trial, attempt, f"trial incomplete: {reason}", reading, len(robot.mast_attempts))
        site.update_summary(f"Stopped after trial {trial} direct Step 1 distance failure")
        return 3
    ok, reason, reading = _direct_close_x(
        vision,
        robot,
        positive_cmd=positive_cmd,
        dist_target=step1_dist_target,
        dist_tol=step1_dist_tol,
        x_target=float(follow._x_target_mm()),
        x_tol=float(DIRECT_STEP1_X_TOL_MM),
        min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
        max_pulses=6,
        label="step1",
        initial_reading=reading,
    )
    if ok:
        dist_after_x = _float_or_none(reading.get("dist_mm") if isinstance(reading, dict) else None)
        if dist_after_x is not None and float(dist_after_x) > step1_dist_target + float(DIRECT_STEP1_PREALIGN_DIST_TOL_MM):
            ok, reason, reading = _direct_close_dist(
                vision,
                robot,
                step1_dist_target,
                float(DIRECT_STEP1_PREALIGN_DIST_TOL_MM),
                allow_back=False,
                max_pulses=4,
                label="step1_dist_polish",
                initial_reading=reading,
            )
            if ok and not bool(_evaluate_reading(reading, "step1").get("target_met")):
                ok, reason, reading = _direct_close_x(
                    vision,
                    robot,
                    positive_cmd=positive_cmd,
                    dist_target=step1_dist_target,
                    dist_tol=step1_dist_tol,
                    x_target=float(follow._x_target_mm()),
                    x_tol=float(DIRECT_STEP1_X_TOL_MM),
                    min_pulse_ms=int(DIRECT_X_SNAP_MIN_MS),
                    max_pulses=4,
                    label="step1_x_after_dist_polish",
                    initial_reading=reading,
                )
    step1_eval = _evaluate_reading(reading, "step1")
    step1_ok = bool(ok and step1_eval.get("target_met"))
    _capture_phase(site, vision, trial, attempt, "step1", "win" if step1_ok else "fail", reason, reading, len(robot.mast_attempts), step="step1")
    if not step1_ok:
        _capture_trial_failure(site, vision, trial, attempt, f"trial incomplete: {reason}", reading, len(robot.mast_attempts))
        site.update_summary(f"Stopped after trial {trial} direct Step 1 failure")
        return 3
    ok, reason, reading = _direct_close_step2_pose(
        vision,
        robot,
        positive_cmd=positive_cmd,
        label="step2",
        initial_reading=reading,
    )
    step2_eval = _evaluate_reading(reading, "step2")
    step2_ok = bool(ok and step2_eval.get("target_met"))
    _capture_phase(site, vision, trial, attempt, "step2", "win" if step2_ok else "fail", reason, reading, len(robot.mast_attempts), step="step2")
    post_reset_ok = False
    post_reset_reason = "post_reset_skipped"
    post_reset_reading = reading
    if step2_ok:
        post_reset_ok, post_reset_reason, post_reset_reading = _direct_close_honest_reset_pose(
            vision,
            robot,
            positive_cmd=positive_cmd,
            label="post_reset",
            initial_reading=reading,
        )
        _capture_phase(
            site,
            vision,
            trial,
            attempt,
            "post-reset",
            "win" if post_reset_ok else "fail",
            post_reset_reason,
            post_reset_reading,
            len(robot.mast_attempts),
        )
    honest_win = bool(step2_ok and post_reset_ok)
    _capture_phase(
        site,
        vision,
        trial,
        attempt,
        "trial",
        "win" if honest_win else "fail",
        "honest reset -> step1 -> step2 -> honest reset" if honest_win else f"trial incomplete: {post_reset_reason if step2_ok else reason}",
        post_reset_reading if step2_ok else reading,
        len(robot.mast_attempts),
        honest_trial=honest_win,
    )
    site.update_summary("Victory: 1/1 direct trial won" if honest_win else f"Stopped after trial {trial} direct {'post-reset' if step2_ok else 'Step 2'} failure")
    return 0 if honest_win else 3


def _run_direct_trial_iteration(
    site: ProgressSite,
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    total_trials: int,
    attempt: int,
    positive_cmd: str,
    initial_reading: dict | None = None,
) -> int:
    wins = 0
    total = max(1, int(total_trials))
    for trial in range(1, total + 1):
        rc = _run_direct_single_trial(
            site,
            vision,
            robot,
            trial=trial,
            attempt=attempt,
            positive_cmd=positive_cmd,
            initial_reading=initial_reading if trial == 1 else None,
            prefer_current_reset=trial > 1,
        )
        if rc != 0:
            site.update_summary(f"Iteration {attempt}: stopped after {wins}/{total} direct wins")
            return rc
        wins += 1
        site.update_summary(f"Iteration {attempt}: {wins}/{total} direct wins")
    site.update_summary(f"Victory: iteration {attempt} won {wins}/{total} direct trials")
    return 0


def _run_direct_step2_toggle_single_trial(
    site: ProgressSite,
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    trial: int,
    attempt: int,
    positive_cmd: str,
) -> int:
    ok, reason, reading = _direct_close_step1_pose(
        vision,
        robot,
        positive_cmd=positive_cmd,
        label="s1_start",
        dist_tol=40.0,
    )
    step1_eval = _evaluate_reading(reading, "step1")
    start_ok = bool(ok and step1_eval.get("target_met"))
    _capture_phase(
        site,
        vision,
        trial,
        attempt,
        "prior",
        "win" if start_ok else "fail",
        reason,
        reading,
        len(robot.mast_attempts),
        step="step1",
    )
    if not start_ok:
        _capture_trial_failure(site, vision, trial, attempt, f"toggle incomplete: start {reason}", reading, len(robot.mast_attempts))
        site.update_summary(f"Stopped after trial {trial} Step 2 toggle start failure")
        return 3

    ok, reason, reading = _direct_close_step2_pose(
        vision,
        robot,
        positive_cmd=positive_cmd,
        label="step2_toggle",
    )
    step2_eval = _evaluate_reading(reading, "step2")
    step2_ok = bool(ok and step2_eval.get("target_met"))
    _capture_phase(
        site,
        vision,
        trial,
        attempt,
        "step2",
        "win" if step2_ok else "fail",
        reason,
        reading,
        len(robot.mast_attempts),
        step="step2",
    )

    return_ok = False
    return_reason = "return_to_step1_skipped"
    return_reading = reading
    if step2_ok:
        return_ok, return_reason, return_reading = _direct_close_step1_pose(
            vision,
            robot,
            positive_cmd=positive_cmd,
            label="s1_return",
            dist_tol=40.0,
        )
        return_eval = _evaluate_reading(return_reading, "step1")
        return_ok = bool(return_ok and return_eval.get("target_met"))
        _capture_phase(
            site,
            vision,
            trial,
            attempt,
            "post-step1",
            "win" if return_ok else "fail",
            return_reason,
            return_reading,
            len(robot.mast_attempts),
            step="step1",
        )

    honest_win = bool(start_ok and step2_ok and return_ok)
    _capture_phase(
        site,
        vision,
        trial,
        attempt,
        "trial",
        "win" if honest_win else "fail",
        "step1 -> step2 -> step1" if honest_win else f"toggle incomplete: {return_reason if step2_ok else reason}",
        return_reading if step2_ok else reading,
        len(robot.mast_attempts),
        honest_trial=honest_win,
    )
    site.update_summary(
        "Victory: 1/1 Step 2 toggle trial won"
        if honest_win
        else f"Stopped after trial {trial} Step 2 toggle {'return' if step2_ok else 'Step 2'} failure"
    )
    return 0 if honest_win else 3


def _run_direct_step2_toggle_iteration(
    site: ProgressSite,
    vision: BrickDetector,
    robot: MastFrozenRobot,
    *,
    total_trials: int,
    attempt: int,
    positive_cmd: str,
) -> int:
    wins = 0
    total = max(1, int(total_trials))
    for trial in range(1, total + 1):
        rc = _run_direct_step2_toggle_single_trial(
            site,
            vision,
            robot,
            trial=trial,
            attempt=attempt,
            positive_cmd=positive_cmd,
        )
        if rc != 0:
            site.update_summary(f"Iteration {attempt}: stopped after {wins}/{total} Step 2 toggle wins")
            return rc
        wins += 1
        site.update_summary(f"Iteration {attempt}: {wins}/{total} Step 2 toggle wins")
    site.update_summary(f"Victory: iteration {attempt} won {wins}/{total} Step 2 toggle trials")
    return 0


def _run_frozen_reset(vision: BrickDetector, robot: MastFrozenRobot) -> dict:
    pre_reading, nudge_reason, nudge_acts = _nudge_to_reset_dist_band(vision, robot)
    if not bool(pre_reading.get("confident")):
        return {
            "success": False,
            "target_met": False,
            "reason": nudge_reason,
            "reading": pre_reading,
            "reset_dist_nudge_acts": int(nudge_acts),
        }
    cfg = follow._reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    if follow._reset_xy_target_ready(pre_reading, cfg):
        return {
            "success": True,
            "target_met": True,
            "phase": "dist_band",
            "reason": "target_hit_after_dist_nudge",
            "reading": pre_reading,
            "reset_dist_nudge_reason": nudge_reason,
            "reset_dist_nudge_acts": int(nudge_acts),
        }
    result = follow._run_reset_sequence(vision, robot)
    if isinstance(result, dict):
        result = dict(result)
        result["reset_dist_nudge_reason"] = nudge_reason
        result["reset_dist_nudge_acts"] = int(nudge_acts)
    else:
        result = {"success": False, "target_met": False, "reason": "reset_failed", "reading": pre_reading}
    if bool(result.get("success")) and not bool(result.get("target_met")):
        reading, retry_reason, retry_acts = _nudge_to_reset_dist_band(vision, robot)
        result["reading"] = reading
        result["reset_dist_nudge_reason"] = f"{nudge_reason};{retry_reason}"
        result["reset_dist_nudge_acts"] = int(nudge_acts) + int(retry_acts)
        if bool(reading.get("confident")):
            cfg = follow._reset_motion_config().get("reverse_turn")
            result["target_met"] = bool(follow._reset_xy_target_ready(reading, cfg if isinstance(cfg, dict) else {}))
            result["reason"] = "target_hit_after_dist_nudge" if bool(result["target_met"]) else retry_reason
    return result


def run_trials(args: argparse.Namespace) -> int:
    site = ProgressSite(Path(args.site_dir), "Leia Frozen Mast Step 1/2 Trials")
    summary = _apply_experiment(str(args.experiment))
    site.set_experiment(str(args.experiment), summary)
    print(f"[SITE] {Path(args.site_dir).resolve()}", flush=True)
    print(f"[SITE] Suggested URL: {_local_url(int(args.port))}", flush=True)
    vision = None
    base_robot = None
    robot = None
    existing_rows = list(site.state.get("rows") or [])
    last_failure_index = -1
    for index, row in enumerate(existing_rows):
        if row.get("status") == "fail":
            last_failure_index = index
    streak_rows = existing_rows[last_failure_index + 1 :]
    wins = sum(
        1
        for row in streak_rows
        if row.get("phase") == "trial" and row.get("status") == "win" and bool(row.get("honest_trial"))
    )
    attempts = []
    for row in existing_rows:
        try:
            attempts.append(int(row.get("attempt")))
        except (TypeError, ValueError):
            pass
    attempt = (max(attempts) + 1) if attempts else 1
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**_vision_tuning_for_experiment(str(args.experiment)))
        follow._warmup(vision)
        base_robot = Robot()
        robot = MastFrozenRobot(base_robot)
        preflight_initial_reading = None
        if str(args.experiment) == "honest-locked-y-forward-xfirst-preflight":
            site.update_summary("Preflight: forward-only recovery to regain confident brick geometry")
            preflight_reading = _forward_recover_confidence(site, vision, robot, attempt)
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: preflight could not regain confident brick geometry")
                return 6
        if str(args.experiment) == "honest-locked-y-micro-xfirst":
            site.update_summary("Preflight: forward-only recovery before micro X-first trial")
            preflight_reading = _forward_recover_confidence(site, vision, robot, attempt, max_pulses=6)
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: micro X-first preflight could not regain confident brick geometry")
                return 6
        if str(args.experiment) == "honest-locked-y-micro-xfirst-gapguard":
            site.update_summary("Preflight: forward-only recovery before gap-guarded micro X-first trial")
            preflight_reading = _forward_recover_confidence(site, vision, robot, attempt, max_pulses=4)
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: gap-guarded micro X-first preflight could not regain confident brick geometry")
                return 6
        if str(args.experiment) == "honest-locked-y-guarded-micro":
            site.update_summary("Preflight: target-continuity guarded recovery before micro X-first trial")
            preflight_reading = _forward_recover_confidence(site, vision, robot, attempt, max_pulses=3)
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: guarded micro preflight could not regain an unambiguous target")
                return 6
        if str(args.experiment) == "honest-locked-y-single-contour":
            site.update_summary("Preflight: single-contour target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "single_contour_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: single-contour target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-single-contour-right":
            site.update_summary("Preflight: single-contour target check before right-turn micro trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "single_contour_right_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: right-turn contour target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-single-contour-right-actioncap":
            site.update_summary("Preflight: single-contour target check before action-capped right-turn trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "single_contour_right_actioncap_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: action-capped contour target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-right":
            site.update_summary("Preflight: direct right-pulse target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_right_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct right target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-left":
            site.update_summary("Preflight: direct left-pulse target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_left_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct left target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-right-80":
            site.update_summary("Preflight: direct 80ms right-pulse target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_right_80_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct right-80 target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-right-150":
            site.update_summary("Preflight: direct 150ms right-pulse target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_right_150_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct right-150 target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-curve-right":
            site.update_summary("Preflight: direct capped turn-curve target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_curve_right_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct curve-right target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-curve-left":
            site.update_summary("Preflight: direct capped left turn-curve target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_curve_left_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: direct curve-left target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-strict-spin-right-240":
            site.update_summary("Preflight: strict-distance direct spin target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_strict_spin_right_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: strict spin target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-contour19000-spin-right-240":
            site.update_summary("Preflight: calibrated-contour direct spin target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_contour19000_spin_right_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: calibrated contour target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-contour19000-spin-left-240":
            site.update_summary("Preflight: calibrated-contour direct left-spin target check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_contour19000_spin_left_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: calibrated contour left-spin target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-right-120":
            site.update_summary("Preflight: calibrated-contour one-wheel right pivot check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_contour19000_onewheel_right_120_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: calibrated contour one-wheel target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-left-120":
            site.update_summary("Preflight: calibrated-contour one-wheel left pivot check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_contour19000_onewheel_left_120_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: calibrated contour one-wheel-left target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-left-240":
            site.update_summary("Preflight: calibrated-contour strong one-wheel left pivot check before one Step 1/2 trial")
            preflight_reading = follow._wait_for_confident_brick(vision, timeout_s=3.0, sample_s=0.12)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_contour19000_onewheel_left_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: calibrated contour strong one-wheel-left target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-left-240":
            site.update_summary("Preflight: stable contour-only one-wheel left pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_onewheel_left_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-240":
            site.update_summary("Preflight: stable contour-only one-wheel right pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=8.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_onewheel_right_240_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour right-pivot target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-360":
            site.update_summary("Preflight: stable contour-only 360ms one-wheel right pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_onewheel_right_360_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour right-360 target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140":
            site.update_summary("Preflight: stable contour-only PWM140 one-wheel right pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_onewheel_right_360_pwm140_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour right-360-pwm140 target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140":
            site.update_summary("Preflight: stable contour-only edge-filtered right pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_edgefilter_right_360_pwm140_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour edge-filter target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-adaptive-right-pwm140":
            site.update_summary("Preflight: stable contour-only adaptive right pivot check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_stablecontour_adaptive_right_pwm140_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: stable contour adaptive target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-step2-strongdist":
            site.update_summary("Preflight: adaptive X plus stronger Step 2 distance check before one Step 1/2 trial")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_adaptive_right_step2_strongdist_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: adaptive strongdist target was not unambiguous")
                return 6
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-resetlong-step2strong":
            site.update_summary("Preflight: adaptive X plus longer reset and stronger Step 2 distance check")
            preflight_reading = _direct_read(vision, timeout_s=4.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or "direct_adaptive_right_resetlong_step2strong_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                site.update_summary("Stopped before trial: adaptive resetlong target was not unambiguous")
                return 6
        if str(args.experiment) in {
            "honest-locked-y-direct-adaptive-right-resetstrong-step2strong",
            "honest-locked-y-direct-adaptive-right-step2-toggle",
        }:
            site.update_summary("Preflight: adaptive X plus strong reset and adaptive Step 2 distance check")
            preflight_initial_reading = None
            preflight_reading = _direct_read(vision, timeout_s=8.0)
            _capture_phase(
                site,
                vision,
                0,
                attempt,
                "preflight",
                "info" if bool(preflight_reading.get("confident")) else "fail",
                str(preflight_reading.get("reason") or f"{str(args.experiment)}_preflight"),
                preflight_reading,
                len(robot.mast_attempts),
            )
            if not bool(preflight_reading.get("confident")):
                recovery_reading = follow._wait_for_confident_brick(vision, timeout_s=2.5, sample_s=0.12)
                _capture_phase(
                    site,
                    vision,
                    0,
                    attempt,
                    "preflight-recovery",
                    "info" if bool(recovery_reading.get("confident")) else "fail",
                    str(recovery_reading.get("reason") or "hsv_preflight_recovery"),
                    recovery_reading,
                    len(robot.mast_attempts),
                )
                if not bool(recovery_reading.get("confident")):
                    recovery_dist = _float_or_none(recovery_reading.get("dist_mm") if isinstance(recovery_reading, dict) else None)
                    recovery_x = _float_or_none(recovery_reading.get("x_mm") if isinstance(recovery_reading, dict) else None)
                    if recovery_dist is None or recovery_x is None:
                        site.update_summary("Stopped before trial: adaptive target was not unambiguous")
                        return 6
                preflight_initial_reading = recovery_reading
        if str(args.experiment) in {
            "honest-locked-y",
            "honest-locked-y-xflip",
            "honest-locked-y-xflip-resetloose",
            "honest-locked-y-xfirst",
            "honest-locked-y-forward-xfirst",
            "honest-locked-y-forward-xfirst-distreset",
            "honest-locked-y-forward-xfirst-preflight",
            "honest-locked-y-micro-xfirst",
            "honest-locked-y-micro-xfirst-gapguard",
            "honest-locked-y-guarded-micro",
            "honest-locked-y-single-contour",
            "honest-locked-y-single-contour-right",
            "honest-locked-y-single-contour-right-actioncap",
            "honest-locked-y-direct-right",
            "honest-locked-y-direct-left",
            "honest-locked-y-direct-right-80",
            "honest-locked-y-direct-right-150",
            "honest-locked-y-direct-curve-right",
            "honest-locked-y-direct-curve-left",
            "honest-locked-y-direct-strict-spin-right-240",
            "honest-locked-y-direct-contour19000-spin-right-240",
            "honest-locked-y-direct-contour19000-spin-left-240",
            "honest-locked-y-direct-contour19000-onewheel-right-120",
            "honest-locked-y-direct-contour19000-onewheel-left-120",
            "honest-locked-y-direct-contour19000-onewheel-left-240",
            "honest-locked-y-direct-stablecontour-onewheel-left-240",
            "honest-locked-y-direct-stablecontour-onewheel-right-240",
            "honest-locked-y-direct-stablecontour-onewheel-right-360",
            "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140",
            "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140",
            "honest-locked-y-direct-stablecontour-adaptive-right-pwm140",
            "honest-locked-y-direct-adaptive-right-step2-strongdist",
            "honest-locked-y-direct-adaptive-right-resetlong-step2strong",
            "honest-locked-y-direct-adaptive-right-resetstrong-step2strong",
            "honest-locked-y-direct-adaptive-right-step2-toggle",
        }:
            lock_reading = _lock_y_target_from_current(vision)
            if not bool(lock_reading.get("confident")) or _float_or_none(lock_reading.get("y_mm")) is None:
                if _lock_y_target_from_reading(preflight_initial_reading):
                    lock_reading = dict(preflight_initial_reading)
                    lock_reading["y_lock_from_preflight_recovery"] = True
                    _capture_phase(
                        site,
                        vision,
                        0,
                        attempt,
                        "y-lock",
                        "info",
                        "locked y from preflight recovery reading",
                        lock_reading,
                        0,
                    )
                else:
                    site.update_summary("Stopped before motion: could not lock a confident Y reference")
                    _capture_phase(
                        site,
                        vision,
                        0,
                        attempt,
                        "y-lock",
                        "fail",
                        "could not lock confident Y reference",
                        lock_reading,
                        0,
                    )
                    return 5
            site.set_targets()
            site.update_summary(f"Locked Y reference at {_fmt_mm(Y_LOCK_TARGET_MM, signed=True)} mm")
        else:
            site.set_targets()
        total_trials = int(args.trials)
        direct_positive_cmds = {
            "honest-locked-y-direct-right": "r",
            "honest-locked-y-direct-left": "l",
            "honest-locked-y-direct-right-80": "r",
            "honest-locked-y-direct-right-150": "r",
            "honest-locked-y-direct-curve-right": "r",
            "honest-locked-y-direct-curve-left": "l",
            "honest-locked-y-direct-strict-spin-right-240": "r",
            "honest-locked-y-direct-contour19000-spin-right-240": "r",
            "honest-locked-y-direct-contour19000-spin-left-240": "l",
            "honest-locked-y-direct-contour19000-onewheel-right-120": "r",
            "honest-locked-y-direct-contour19000-onewheel-left-120": "l",
            "honest-locked-y-direct-contour19000-onewheel-left-240": "l",
            "honest-locked-y-direct-stablecontour-onewheel-left-240": "l",
            "honest-locked-y-direct-stablecontour-onewheel-right-240": "r",
            "honest-locked-y-direct-stablecontour-onewheel-right-360": "r",
            "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140": "r",
            "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140": "r",
            "honest-locked-y-direct-stablecontour-adaptive-right-pwm140": "r",
            "honest-locked-y-direct-adaptive-right-step2-strongdist": "r",
            "honest-locked-y-direct-adaptive-right-resetlong-step2strong": "r",
            "honest-locked-y-direct-adaptive-right-resetstrong-step2strong": "r",
        }
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-step2-toggle":
            return _run_direct_step2_toggle_iteration(
                site,
                vision,
                robot,
                total_trials=total_trials,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) in direct_positive_cmds:
            return _run_direct_trial_iteration(
                site,
                vision,
                robot,
                total_trials=total_trials,
                attempt=attempt,
                positive_cmd=direct_positive_cmds[str(args.experiment)],
                initial_reading=preflight_initial_reading,
            )
        if str(args.experiment) == "honest-locked-y-direct-right":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-left":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-right-80":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-right-150":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-curve-right":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-curve-left":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-strict-spin-right-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-contour19000-spin-right-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-contour19000-spin-left-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-right-120":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-left-120":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-contour19000-onewheel-left-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-left-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="l",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-240":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-360":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-stablecontour-adaptive-right-pwm140":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-step2-strongdist":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-resetlong-step2strong":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        if str(args.experiment) == "honest-locked-y-direct-adaptive-right-resetstrong-step2strong":
            return _run_direct_single_trial(
                site,
                vision,
                robot,
                trial=1,
                attempt=attempt,
                positive_cmd="r",
            )
        for trial in range(wins + 1, total_trials + 1):
            site.update_summary(f"Running trial {trial}/{total_trials}")
            before = follow._wait_for_confident_brick(vision, timeout_s=6.0, sample_s=0.12)
            _capture_phase(site, vision, trial, attempt, "start", "info", "pre-trial observation", before, len(robot.mast_attempts))

            pre_reset = _run_frozen_reset(vision, robot)
            pre_reading = pre_reset.get("reading") if isinstance(pre_reset, dict) else None
            pre_ok = bool(pre_reset.get("success")) and bool(pre_reset.get("target_met"))
            _capture_phase(
                site,
                vision,
                trial,
                attempt,
                "pre-reset",
                "win" if pre_ok else "fail",
                str(pre_reset.get("reason") if isinstance(pre_reset, dict) else "reset_failed"),
                pre_reading,
                len(robot.mast_attempts),
            )
            if not pre_ok:
                _capture_trial_failure(
                    site,
                    vision,
                    trial,
                    attempt,
                    f"trial incomplete: pre-reset {pre_reset.get('reason') if isinstance(pre_reset, dict) else 'reset_failed'}",
                    pre_reading,
                    len(robot.mast_attempts),
                )
                site.update_summary(f"Stopped after trial {trial} pre-reset failure")
                return 2

            stats = follow._follow_loop(
                vision,
                robot,
                duration_s=float(args.duration_s),
                reset_after_win=False,
                stop_after_win=True,
                stop_after_step2=False,
                debug_mode=False,
            )
            step1_reading = follow._read_brick_measurement(vision)
            step1_eval = _evaluate_reading(step1_reading, "step1")
            step1_ok = int(stats.get("win_count", 0) or 0) >= 1 and bool(step1_eval.get("target_met"))
            _capture_phase(
                site,
                vision,
                trial,
                attempt,
                "step1",
                "win" if step1_ok else "fail",
                str(stats.get("last_action") or "step1_follow_loop_done"),
                step1_reading,
                len(robot.mast_attempts),
                step="step1",
            )
            if not step1_ok:
                _capture_trial_failure(
                    site,
                    vision,
                    trial,
                    attempt,
                    "trial incomplete: Step 1 target not met",
                    step1_reading,
                    len(robot.mast_attempts),
                )
                site.update_summary(f"Stopped after trial {trial} Step 1 failure")
                return 3

            step2_result = follow._run_step2_seat_sequence(vision, robot)
            step2_reading = step2_result.get("reading") if isinstance(step2_result, dict) else None
            step2_eval = _evaluate_reading(step2_reading, "step2")
            step2_ok = bool(step2_result.get("success")) and bool(step2_eval.get("target_met"))
            _capture_phase(
                site,
                vision,
                trial,
                attempt,
                "step2",
                "win" if step2_ok else "fail",
                str(step2_result.get("reason") if isinstance(step2_result, dict) else "step2_failed"),
                step2_reading,
                len(robot.mast_attempts),
                step="step2",
            )
            if not step2_ok:
                _capture_trial_failure(
                    site,
                    vision,
                    trial,
                    attempt,
                    "trial incomplete: Step 2 target not met",
                    step2_reading,
                    len(robot.mast_attempts),
                )
                site.update_summary(f"Stopped after trial {trial} Step 2 failure")
                return 3

            post_reset = _run_frozen_reset(vision, robot)
            post_reading = post_reset.get("reading") if isinstance(post_reset, dict) else None
            post_ok = bool(post_reset.get("success")) and bool(post_reset.get("target_met"))
            _capture_phase(
                site,
                vision,
                trial,
                attempt,
                "post-reset",
                "win" if post_ok else "fail",
                str(post_reset.get("reason") if isinstance(post_reset, dict) else "reset_failed"),
                post_reading,
                len(robot.mast_attempts),
            )
            trial_ok = bool(step1_ok and step2_ok and post_ok)
            _capture_phase(
                site,
                vision,
                trial,
                attempt,
                "trial",
                "win" if trial_ok else "fail",
                "reset -> step1 -> step2 -> reset" if trial_ok else "trial incomplete",
                post_reading,
                len(robot.mast_attempts),
                honest_trial=trial_ok,
            )
            if not trial_ok:
                site.update_summary(f"Stopped after trial {trial} final reset failure")
                return 4
            wins += 1
            site.update_summary(f"{wins}/{total_trials} trials won")
        site.update_summary(f"Victory: {wins}/{int(args.trials)} trials won")
        return 0
    finally:
        try:
            if robot is not None:
                follow._stop_robot(robot)
        finally:
            close_robot = getattr(base_robot, "close", None)
            if callable(close_robot):
                close_robot()
        close_vision = getattr(vision, "close", None)
        if callable(close_vision):
            close_vision()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--site-dir", default=str(DEFAULT_SITE_DIR))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--experiment",
        choices=(
            "strict-y",
            "short-reset",
            "balanced-short-reset",
            "balanced-visible-reset",
            "ignore-y-gates",
            "balanced-visible-yfree-soft",
            "honest-locked-y",
            "honest-locked-y-xflip",
            "honest-locked-y-xflip-resetloose",
            "honest-locked-y-xfirst",
            "honest-locked-y-forward-xfirst",
            "honest-locked-y-forward-xfirst-distreset",
            "honest-locked-y-forward-xfirst-preflight",
            "honest-locked-y-micro-xfirst",
            "honest-locked-y-micro-xfirst-gapguard",
            "honest-locked-y-guarded-micro",
            "honest-locked-y-single-contour",
            "honest-locked-y-single-contour-right",
            "honest-locked-y-single-contour-right-actioncap",
            "honest-locked-y-direct-right",
            "honest-locked-y-direct-left",
            "honest-locked-y-direct-right-80",
            "honest-locked-y-direct-right-150",
            "honest-locked-y-direct-curve-right",
            "honest-locked-y-direct-curve-left",
            "honest-locked-y-direct-strict-spin-right-240",
            "honest-locked-y-direct-contour19000-spin-right-240",
            "honest-locked-y-direct-contour19000-spin-left-240",
            "honest-locked-y-direct-contour19000-onewheel-right-120",
            "honest-locked-y-direct-contour19000-onewheel-left-120",
            "honest-locked-y-direct-contour19000-onewheel-left-240",
            "honest-locked-y-direct-stablecontour-onewheel-left-240",
            "honest-locked-y-direct-stablecontour-onewheel-right-240",
            "honest-locked-y-direct-stablecontour-onewheel-right-360",
            "honest-locked-y-direct-stablecontour-onewheel-right-360-pwm140",
            "honest-locked-y-direct-stablecontour-edgefilter-right-360-pwm140",
            "honest-locked-y-direct-stablecontour-adaptive-right-pwm140",
            "honest-locked-y-direct-adaptive-right-step2-strongdist",
            "honest-locked-y-direct-adaptive-right-resetlong-step2strong",
            "honest-locked-y-direct-adaptive-right-resetstrong-step2strong",
            "honest-locked-y-direct-adaptive-right-step2-toggle",
        ),
        default="honest-locked-y",
    )
    return parser.parse_args()


def main() -> int:
    return run_trials(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
