#!/usr/bin/env python3
"""
helper_mini_align_calibrate.py
------------------------------
Mini alignment turn calibration helper used before auto ALIGN_BRICK / POSITION_BRICK
steps. Currently focuses on X-axis (left/right) micro-turn discovery.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import cv2

import telemetry_robot as telemetry_robot_module
from helper_gate_utils import load_process_steps
from helper_robot_control import Robot
from helper_stream_server import StreamServer, format_stream_url
from helper_vision_aruco import ArucoBrickVision
from telemetry_process import send_robot_command_pwm
from telemetry_robot import WorldModel, StepState, draw_telemetry_overlay


NUM_TRIALS_DEFAULT = 10
STREAM_PORT_DEFAULT = 5000
MINI_TRIALS_DEFAULT = 1

RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent.parent / "world_model_x_axis.json"
ROBOT_MODEL_FILE_DEFAULT = Path(
    getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", Path(__file__).resolve().parent / "world_model_robot.json")
)
WORLD_MODEL_X_AXIS_SECTION_KEY = "x_axis_turn_calibration"
WORLD_MODEL_SPEED_SECONDS_TURN_LEFT_KEY = "speed_score_seconds_turn_left"
WORLD_MODEL_SPEED_SECONDS_TURN_RIGHT_KEY = "speed_score_seconds_turn_right"
WORLD_MODEL_TURN_INTENSITY_POSTS_LEFT_KEY = str(
    getattr(telemetry_robot_module, "TURN_INTENSITY_KEY_TURN_LEFT", "turn_intensity_posts_turn_left")
)
WORLD_MODEL_TURN_INTENSITY_POSTS_RIGHT_KEY = str(
    getattr(telemetry_robot_module, "TURN_INTENSITY_KEY_TURN_RIGHT", "turn_intensity_posts_turn_right")
)
WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY = "speed_score_seconds_turn_meta"
WORLD_MODEL_AUTO_MINI_LAST_RUN_TS_KEY = "mini_x_axis_last_run_epoch_s"
WORLD_MODEL_AUTO_MINI_LAST_RUN_ISO_KEY = "mini_x_axis_last_run_iso_utc"
WORLD_MODEL_AUTO_MINI_LAST_RUN_STEP_KEY = "mini_x_axis_last_run_step"
WORLD_MODEL_AUTO_MINI_LAST_RUN_OK_KEY = "mini_x_axis_last_run_ok"
WORLD_MODEL_AUTO_MINI_LAST_RUN_CONFIDENT_KEY = "mini_x_axis_last_run_confident"
WORLD_MODEL_AUTO_MINI_LAST_RUN_DURATION_S_KEY = "mini_x_axis_last_run_seconds"
WORLD_MODEL_AUTO_MINI_MIN_INTERVAL_HOURS_KEY = "mini_x_axis_min_interval_hours"
AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS_DEFAULT = 12.0

# Learning / control knobs
ALPHA = 0.25
MAX_COARSE_SCORE = 50
OBSERVE_SLEEP_S = 0.02
CONTROL_SLEEP_S = 0.04
REQUIRED_SUCCESS_FRAMES = 3
TRIAL_TIMEOUT_S = 10.0
MAX_ACTS_PER_TRIAL = 180

# Reset (random starting offset away from gate target)
RESET_MIN_AWAY_MM = 15.0
RESET_MAX_AWAY_MM = 35.0
RESET_SCORE = 8
RESTORE_TO_START_TOL_MM = 1.5
RESTORE_TO_START_MAX_ACTS = 100

# Overshoot / reward shaping
CROSS_DEADZONE_MM = 0.3
OVERSHOOT_PENALTY = 25.0
ACT_PENALTY = 0.05

# If left/right feel backwards, set --x-axis-sign -1.
X_AXIS_SIGN_DEFAULT = 1.0

# 1% turn-step calibration goals (near target).
STEP_TARGET_MIN_MM = 0.30
STEP_TARGET_MAX_MM = 0.50
STEP_TARGET_MID_MM = (STEP_TARGET_MIN_MM + STEP_TARGET_MAX_MM) / 2.0
STEP_NEAR_TARGET_MM = 8.0
STEP_MOVEMENT_DETECT_MM = 0.12
STEP_REWARD_PENALTY_PER_MM = 3.0
STEP_CLOSE_TIERS_MM = (0.50, 0.70, 1.00, 1.50)
# Adaptive step-size targets by |x_err|. Closer to target stays tight; farther
# from target gradually relaxes the acceptable range.
STEP_TARGET_BANDS_MM = (
    (8.0, 0.30, 0.50),
    (20.0, 0.30, 0.70),
    (35.0, 0.30, 1.00),
    (1e9, 0.30, 1.50),
)

# "Below 1%" notch model: negative notches keep score=1 and shorten duration.
TURN_NOTCH_MIN = -9
TURN_NOTCH_MAX = 7
TURN_NOTCH_DURATION_STEP = 0.10
TARGET_MICRO_INTENSITY_PCT = 0.01
DISCOVERY_START_BELOW_CURRENT_NOTCHES = 5
DISCOVERY_CONFIRM_FRAMES = 3
DISCOVERY_SAMPLE_TIMEOUT_S = 1.5

# Consume first-turn asymmetry by sending one sacrificial tiny pulse whenever
# direction changes, then measure/control with the steady-state pulse.
TURN_DIR_WARMUP_ENABLED = True
TURN_DIR_WARMUP_NOTCH = -2
# Extra settle to reduce physical slide when quickly reversing turn direction.
TURN_REVERSAL_SETTLE_S = 0.5

# Confidence gate for concluding calibration quality.
CONF_MIN_TRIALS = 6
CONF_MIN_OBS_PER_DIR = 6
CONF_SUBTLE_HIT_RATE = 0.85

# Vision recovery: replay inverse of recent acts and check vision after each.
RECOVERY_MAX_INVERSE_ACTS = 3
RECOVERY_CHECK_TIMEOUT_S = 1.0

_SESSION_TINY_TURN_DISCOVERY = {
    "done": False,
    "by_cmd": {},
    "persist_result": {},
}


def _coerce_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _sign(value, deadzone=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if abs(v) <= float(deadzone):
        return 0
    return 1 if v > 0 else -1


def _load_x_gate_from_process():
    steps = load_process_steps()
    align_gates = (steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}
    metric = "xAxis_offset_abs"
    gate = align_gates.get(metric)
    if not isinstance(gate, dict):
        metric = "xAxis_offset"
        gate = align_gates.get(metric)
    if not isinstance(gate, dict):
        # Fallback to defaults if process model is missing or schema changed.
        metric = "xAxis_offset_abs"
        gate = {"target": -2.24, "tol": 1.5}
    target = _coerce_float(gate.get("target"), -2.24)
    tol = abs(_coerce_float(gate.get("tol"), 1.5))
    return metric, target, tol


@dataclass
class PendingAct:
    trial: int
    phase: str
    act: int
    timestamp: float
    cmd: str
    score: int
    epsilon: float
    offset_x: float
    x_err_mm: float
    abs_err_mm: float
    bin_idx: int
    duration_scale: float = 1.0
    notch: int | None = None
    micro_mode: bool = False
    warmup_applied: bool = False


class JsonEventLog:
    def __init__(self, path: Path, *, flush_every=50):
        self.path = Path(path)
        self.flush_every = int(flush_every)
        self.events: list[dict] = []
        self._unsaved = 0

    def wipe(self):
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass
        self.events = []
        self._unsaved = 0

    def append(self, event: dict, *, force=False):
        if not isinstance(event, dict):
            return
        self.events.append(event)
        self._unsaved += 1
        self.flush(force=force)

    def flush(self, *, force=False):
        if not force and self._unsaved < self.flush_every:
            return
        try:
            self.path.write_text(json.dumps(self.events, indent=(2 if force else None)))
            self._unsaved = 0
        except Exception as e:
            print(f"[LOG] Failed to write {self.path}: {e}")


class XAxisSpeedPolicy:
    """
    Epsilon-greedy bandit per (cmd, |x_err| bin) over discrete speed scores.
    Reward is "mm closer to target" with heavy penalties for overshoot.
    """

    ERROR_BINS_MM = (2.0, 4.0, 8.0, 12.0, 20.0, 35.0, 60.0)  # open-ended last bin
    SCORE_CANDIDATES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50)

    # Safety caps near the target to reduce overshoot during exploration.
    MAX_SCORE_BY_BIN = (6, 10, 15, 20, 26, 35, 45, 50)

    # Defaults for tie-breaking / cold start.
    DEFAULT_SCORE_BY_BIN = (1, 2, 5, 8, 12, 20, 35, 50)

    def __init__(self, *, alpha=ALPHA):
        self.alpha = float(alpha)
        self.q: dict[str, dict[int, dict[int, float]]] = {}
        self.n: dict[str, dict[int, dict[int, int]]] = {}
        for cmd in ("l", "r"):
            self.q[cmd] = {}
            self.n[cmd] = {}
            for bin_idx in range(self.num_bins()):
                self.q[cmd][bin_idx] = {score: 0.0 for score in self.SCORE_CANDIDATES}
                self.n[cmd][bin_idx] = {score: 0 for score in self.SCORE_CANDIDATES}

    @classmethod
    def num_bins(cls) -> int:
        return len(cls.ERROR_BINS_MM) + 1

    @classmethod
    def bin_idx(cls, abs_err_mm: float) -> int:
        try:
            v = abs(float(abs_err_mm))
        except (TypeError, ValueError):
            return cls.num_bins() - 1
        for idx, edge in enumerate(cls.ERROR_BINS_MM):
            if v < float(edge):
                return idx
        return cls.num_bins() - 1

    @classmethod
    def bin_label(cls, idx: int) -> str:
        edges = (0.0,) + tuple(cls.ERROR_BINS_MM) + (None,)
        lo = edges[int(idx)]
        hi = edges[int(idx) + 1]
        if hi is None:
            return f">={lo:.0f}mm"
        return f"{lo:.0f}-{hi:.0f}mm"

    def _allowed_scores(self, bin_idx: int) -> list[int]:
        cap = int(self.MAX_SCORE_BY_BIN[int(bin_idx)])
        return [s for s in self.SCORE_CANDIDATES if int(s) <= cap]

    def default_score(self, bin_idx: int) -> int:
        return int(self.DEFAULT_SCORE_BY_BIN[int(bin_idx)])

    def choose_score(self, cmd: str, abs_err_mm: float, *, epsilon: float) -> tuple[int, int, float]:
        bin_idx = self.bin_idx(abs_err_mm)
        allowed = self._allowed_scores(bin_idx)
        default = self.default_score(bin_idx)

        explore = random.random() < max(0.0, min(1.0, float(epsilon)))
        if explore:
            return int(random.choice(allowed)), int(bin_idx), float(epsilon)

        q_bin = self.q.get(cmd, {}).get(bin_idx, {})
        best_q = None
        best_scores: list[int] = []
        for score in allowed:
            qv = q_bin.get(int(score), 0.0)
            if best_q is None or qv > best_q:
                best_q = float(qv)
                best_scores = [int(score)]
            elif qv == best_q:
                best_scores.append(int(score))

        if not best_scores:
            return int(default), int(bin_idx), float(epsilon)

        # Tie-break toward the default score, then toward smaller scores (safer near target).
        best_scores.sort(key=lambda s: (abs(int(s) - int(default)), int(s)))
        return int(best_scores[0]), int(bin_idx), float(epsilon)

    def update(self, cmd: str, bin_idx: int, score: int, reward: float) -> tuple[float, float]:
        cmd = str(cmd)
        bin_idx = int(bin_idx)
        score = int(score)
        reward = float(reward)
        q_prev = float(self.q[cmd][bin_idx].get(score, 0.0))
        q_next = (1.0 - self.alpha) * q_prev + self.alpha * reward
        self.q[cmd][bin_idx][score] = float(q_next)
        self.n[cmd][bin_idx][score] = int(self.n[cmd][bin_idx].get(score, 0) + 1)
        return q_prev, q_next

    def snapshot_best_scores(self) -> dict:
        out = {}
        for cmd in ("l", "r"):
            cmd_out = {}
            for bin_idx in range(self.num_bins()):
                allowed = self._allowed_scores(bin_idx)
                q_bin = self.q.get(cmd, {}).get(bin_idx, {})
                default = self.default_score(bin_idx)
                best = None
                best_scores = []
                for score in allowed:
                    qv = q_bin.get(int(score), 0.0)
                    if best is None or qv > best:
                        best = float(qv)
                        best_scores = [int(score)]
                    elif qv == best:
                        best_scores.append(int(score))
                if not best_scores:
                    best_scores = [int(default)]
                best_scores.sort(key=lambda s: (abs(int(s) - int(default)), int(s)))
                cmd_out[self.bin_label(bin_idx)] = {
                    "score": int(best_scores[0]),
                    "q": float(best if best is not None else 0.0),
                }
            out[cmd] = cmd_out
        return out


class XAxisCalibrator:
    def __init__(
        self,
        *,
        num_trials: int,
        stream_port: int,
        log_path: Path,
        x_axis_sign: float,
        robot=None,
        vision=None,
        enable_stream: bool = True,
    ):
        requested_trials = int(num_trials)
        self.num_trials = int(max(1, requested_trials))
        if requested_trials < 1:
            print(f"[X-AXIS] Requested trials {requested_trials} is invalid; using {self.num_trials}.")
        self.x_axis_sign = float(x_axis_sign)

        self.x_gate_metric, self.x_target_mm, self.x_tol_mm = _load_x_gate_from_process()

        self._owns_robot = robot is None
        self._owns_vision = vision is None
        self.robot = robot if robot is not None else Robot()
        self.vision = vision if vision is not None else ArucoBrickVision(debug=False)
        self.world = WorldModel()
        self.world.step_state = StepState.ALIGN_BRICK

        self.run_id = int(time.time())
        self.current_trial = 0
        self.current_phase = "init"
        self.act_idx = 0

        # Near-target micro-turn state (per turn direction).
        self.turn_notch_by_cmd = self._load_current_tiny_notch_profiles()
        self.turn_step_mm_by_cmd = {"l": deque(maxlen=20), "r": deque(maxlen=20)}
        self._last_turn_cmd_for_trial = None
        self._last_turn_cmd_sent = None
        self.recent_turn_acts = deque(maxlen=24)
        self.tiny_discovery_by_cmd = {}
        self.tiny_discovery_persist_result = {}
        self.tiny_discovery_ran_this_instance = False

        self._last_obs_offset_x = None
        self._last_obs_delta_offset_x = None

        self.policy = XAxisSpeedPolicy(alpha=ALPHA)

        self.log = JsonEventLog(Path(log_path), flush_every=50)
        self.log.wipe()

        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True

        self.server = None
        if bool(enable_stream):
            self.server = StreamServer(self._get_frame, port=int(stream_port))
            self.server.start()
            print(f"[X-AXIS] Livestream: {format_stream_url('127.0.0.1', int(stream_port))}")
        print(
            f"[X-AXIS] Gate ({self.x_gate_metric}): target {self.x_target_mm:+.2f}mm tol {self.x_tol_mm:.2f}mm "
            f"(sign {self.x_axis_sign:+.0f})"
        )
        print(
            f"[X-AXIS] 1% goal: {STEP_TARGET_MIN_MM:.2f}-{STEP_TARGET_MAX_MM:.2f}mm/turn "
            f"(notch range {TURN_NOTCH_MIN}..{TURN_NOTCH_MAX}; session discovery target={TARGET_MICRO_INTENSITY_PCT:.2f}%)"
        )
        print(f"[X-AXIS] Log: {Path(log_path)}")

    def _get_frame(self):
        with self.lock:
            return self.current_frame

    def _x_err_mm(self, offset_x: float) -> float:
        try:
            return (float(offset_x) * float(self.x_axis_sign)) - float(self.x_target_mm)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _score_candidates() -> list[int]:
        return [int(s) for s in XAxisSpeedPolicy.SCORE_CANDIDATES if int(s) <= int(MAX_COARSE_SCORE)]

    def _score_duration_for_notch(self, notch: int) -> tuple[int, float, int]:
        """
        Convert an abstract notch into (score, duration_scale).
        Negative notches are "below 1%" by shortening the 1% pulse duration.
        """
        notch_clamped = max(int(TURN_NOTCH_MIN), min(int(TURN_NOTCH_MAX), int(notch)))
        if notch_clamped < 0:
            duration_scale = max(0.05, 1.0 + (float(notch_clamped) * float(TURN_NOTCH_DURATION_STEP)))
            return 1, float(duration_scale), int(notch_clamped)
        candidates = self._score_candidates()
        if not candidates:
            return 1, 1.0, int(notch_clamped)
        idx = max(0, min(len(candidates) - 1, int(notch_clamped)))
        return int(candidates[idx]), 1.0, int(notch_clamped)

    @staticmethod
    def _intensity_entry_for_target(intensity_map: dict, target_pct: float):
        if not isinstance(intensity_map, dict):
            return None
        target = float(target_pct)
        for raw_key, value in intensity_map.items():
            try:
                key_pct = float(raw_key)
            except (TypeError, ValueError):
                continue
            if abs(float(key_pct) - target) <= 1e-6 and isinstance(value, dict):
                return value
        return None

    def _notch_from_profile_entry(self, entry: dict | None) -> int:
        if not isinstance(entry, dict):
            return 0

        try:
            explicit_notch = int(entry.get("notch"))
            return int(max(TURN_NOTCH_MIN, min(TURN_NOTCH_MAX, explicit_notch)))
        except (TypeError, ValueError):
            pass

        try:
            score = int(entry.get("score", 1))
        except (TypeError, ValueError):
            score = 1
        try:
            duration_scale = float(entry.get("duration_scale", 1.0))
        except (TypeError, ValueError):
            duration_scale = 1.0

        if score <= 1:
            try:
                notch = int(round((float(duration_scale) - 1.0) / float(TURN_NOTCH_DURATION_STEP)))
            except (TypeError, ValueError, ZeroDivisionError):
                notch = 0
            return int(max(TURN_NOTCH_MIN, min(TURN_NOTCH_MAX, notch)))

        candidates = self._score_candidates()
        if not candidates:
            return 0
        nearest_idx = min(range(len(candidates)), key=lambda idx: abs(int(candidates[idx]) - int(score)))
        return int(max(TURN_NOTCH_MIN, min(TURN_NOTCH_MAX, nearest_idx)))

    def _load_current_tiny_notch_profiles(self) -> dict:
        model = _world_model_load(ROBOT_MODEL_FILE_DEFAULT)
        out = {"l": 0, "r": 0}
        for cmd, intensity_key in (
            ("l", WORLD_MODEL_TURN_INTENSITY_POSTS_LEFT_KEY),
            ("r", WORLD_MODEL_TURN_INTENSITY_POSTS_RIGHT_KEY),
        ):
            intensity_map = model.get(intensity_key)
            entry = self._intensity_entry_for_target(intensity_map, TARGET_MICRO_INTENSITY_PCT)
            out[cmd] = int(self._notch_from_profile_entry(entry))
        return out

    def _collect_offset_samples(self, *, samples=1, timeout_s=1.5) -> list[float]:
        offsets: list[float] = []
        start_t = time.time()
        requested = max(1, int(samples))
        while len(offsets) < requested and (time.time() - start_t) < float(timeout_s) and self.running:
            found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
            self._update_world(found, angle, dist, offset_x, conf, cam_h, above, below, self.vision.current_frame)
            if found:
                offsets.append(float(offset_x))
                if len(offsets) < requested:
                    time.sleep(OBSERVE_SLEEP_S)
            else:
                time.sleep(OBSERVE_SLEEP_S)
        return offsets

    def _persist_tiny_turn_profile_to_world_model(self, discovered_by_cmd: dict) -> dict:
        model_path = ROBOT_MODEL_FILE_DEFAULT
        try:
            model = _world_model_load(model_path)
            meta = model.get(WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY)
            if not isinstance(meta, dict):
                meta = {}
                model[WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY] = meta

            applied = {}
            intensity_key_text = f"{float(TARGET_MICRO_INTENSITY_PCT):.2f}"
            for cmd, intensity_key in (
                ("l", WORLD_MODEL_TURN_INTENSITY_POSTS_LEFT_KEY),
                ("r", WORLD_MODEL_TURN_INTENSITY_POSTS_RIGHT_KEY),
            ):
                row = discovered_by_cmd.get(cmd) if isinstance(discovered_by_cmd, dict) else None
                if not isinstance(row, dict):
                    continue
                try:
                    notch = int(row.get("selected_notch"))
                except (TypeError, ValueError):
                    continue
                score, duration_scale, _ = self._score_duration_for_notch(notch)
                intensity_map = model.get(intensity_key)
                if not isinstance(intensity_map, dict):
                    intensity_map = {}
                    model[intensity_key] = intensity_map
                intensity_map[intensity_key_text] = {
                    "score": int(score),
                    "duration_scale": round(float(duration_scale), 3),
                    "notch": int(notch),
                    "source": "mini_x_axis_discovery",
                }
                applied[cmd] = {
                    "notch": int(notch),
                    "score": int(score),
                    "duration_scale": round(float(duration_scale), 3),
                }

            now_s = float(time.time())
            meta["mini_x_axis_tiny_profile_pct"] = float(TARGET_MICRO_INTENSITY_PCT)
            meta["mini_x_axis_tiny_profile_last_updated_epoch_s"] = round(now_s, 3)
            meta["mini_x_axis_tiny_profile_last_updated_iso_utc"] = _format_utc_timestamp(now_s)
            meta["mini_x_axis_tiny_profile_notches"] = {
                "l": int(self.turn_notch_by_cmd.get("l", 0)),
                "r": int(self.turn_notch_by_cmd.get("r", 0)),
            }

            model_path.write_text(json.dumps(model, indent=2) + "\n")
            return {
                "ok": True,
                "model_path": str(model_path),
                "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                "applied": applied,
            }
        except OSError as exc:
            return {
                "ok": False,
                "model_path": str(model_path),
                "error": str(exc),
            }

    def _discover_tiny_turn_profiles_once(self) -> dict:
        global _SESSION_TINY_TURN_DISCOVERY

        if bool(_SESSION_TINY_TURN_DISCOVERY.get("done")):
            cached = _SESSION_TINY_TURN_DISCOVERY.get("by_cmd")
            if isinstance(cached, dict):
                for cmd in ("l", "r"):
                    row = cached.get(cmd) if isinstance(cached.get(cmd), dict) else None
                    if isinstance(row, dict):
                        try:
                            self.turn_notch_by_cmd[cmd] = int(row.get("selected_notch", self.turn_notch_by_cmd.get(cmd, 0)))
                        except (TypeError, ValueError):
                            pass
                self.tiny_discovery_by_cmd = {k: dict(v) for k, v in cached.items() if isinstance(v, dict)}
            cached_persist = _SESSION_TINY_TURN_DISCOVERY.get("persist_result")
            self.tiny_discovery_persist_result = dict(cached_persist) if isinstance(cached_persist, dict) else {}
            self.tiny_discovery_ran_this_instance = False
            self.log.append(
                {
                    "type": "turn_tiny_profile_discovery_session_reuse",
                    "run_id": self.run_id,
                    "timestamp": time.time(),
                    "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                    "turn_notch_by_cmd": dict(self.turn_notch_by_cmd),
                    "cached_discovery": self.tiny_discovery_by_cmd,
                }
            )
            print(
                f"[X-AXIS] Reusing session 0.01% profile: "
                f"L notch {int(self.turn_notch_by_cmd.get('l', 0)):+d}, "
                f"R notch {int(self.turn_notch_by_cmd.get('r', 0)):+d}"
            )
            return self.tiny_discovery_by_cmd

        self.tiny_discovery_ran_this_instance = True
        prev_phase = self.current_phase
        self.current_phase = "probe_tiny"
        discovered: dict[str, dict] = {}
        self.log.append(
            {
                "type": "turn_tiny_profile_discovery_start",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                "start_below_notches": int(DISCOVERY_START_BELOW_CURRENT_NOTCHES),
                "confirm_frames": int(DISCOVERY_CONFIRM_FRAMES),
                "movement_threshold_mm": float(STEP_MOVEMENT_DETECT_MM),
                "starting_notches": dict(self.turn_notch_by_cmd),
            }
        )

        try:
            for cmd in ("l", "r"):
                current_notch = int(self.turn_notch_by_cmd.get(cmd, 0))
                start_notch = int(max(TURN_NOTCH_MIN, current_notch - int(DISCOVERY_START_BELOW_CURRENT_NOTCHES)))
                selected_row = None
                probe_count = 0

                for notch in range(int(start_notch), int(TURN_NOTCH_MAX) + 1):
                    probe_count += 1
                    warmup_applied = self._direction_change_warmup(cmd, phase=self.current_phase)
                    if warmup_applied:
                        self._collect_offset_samples(samples=1, timeout_s=1.0)

                    before_offsets = self._collect_offset_samples(
                        samples=int(DISCOVERY_CONFIRM_FRAMES),
                        timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                    )
                    if len(before_offsets) < int(DISCOVERY_CONFIRM_FRAMES):
                        recovered = self._recover_from_lost_vision(source_phase=self.current_phase)
                        if not recovered:
                            print(
                                f"[X-AXIS][0.01%] {cmd.upper()} probe ended early: insufficient pre-move vision samples."
                            )
                            break
                        before_offsets = [float(recovered["offset_x"])] * int(DISCOVERY_CONFIRM_FRAMES)
                    baseline_offset = float(statistics.mean(before_offsets))

                    score, duration_scale, notch_used = self._score_duration_for_notch(notch)
                    sent = self._send_turn(cmd, score, duration_scale=duration_scale, phase=self.current_phase)
                    time.sleep(CONTROL_SLEEP_S)

                    after_offsets = self._collect_offset_samples(
                        samples=int(DISCOVERY_CONFIRM_FRAMES),
                        timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                    )
                    if len(after_offsets) < int(DISCOVERY_CONFIRM_FRAMES):
                        recovered = self._recover_from_lost_vision(source_phase=self.current_phase)
                        if recovered is None:
                            print(
                                f"[X-AXIS][0.01%] {cmd.upper()} probe ended early: insufficient post-move vision samples."
                            )
                            break
                        after_offsets = [float(recovered["offset_x"])] * int(DISCOVERY_CONFIRM_FRAMES)

                    deltas = [abs(float(val) - float(baseline_offset)) for val in after_offsets]
                    moved_frames = int(
                        sum(1 for delta in deltas if float(delta) >= float(STEP_MOVEMENT_DETECT_MM))
                    )
                    movement_detected = bool(moved_frames >= int(DISCOVERY_CONFIRM_FRAMES))
                    step_mm = float(max(deltas) if deltas else 0.0)

                    row = {
                        "cmd": str(cmd),
                        "current_notch_before_scan": int(current_notch),
                        "start_notch": int(start_notch),
                        "probe_index": int(probe_count),
                        "notch": int(notch_used),
                        "score": int(score),
                        "duration_scale": float(duration_scale),
                        "before_offsets": [float(v) for v in before_offsets],
                        "after_offsets": [float(v) for v in after_offsets],
                        "baseline_offset": float(baseline_offset),
                        "step_mm": float(step_mm),
                        "moved_frames": int(moved_frames),
                        "confirm_frames_required": int(DISCOVERY_CONFIRM_FRAMES),
                        "movement_detected": bool(movement_detected),
                        "warmup_applied": bool(warmup_applied),
                        "sent": sent if isinstance(sent, dict) else None,
                    }
                    print(
                        f"[X-AXIS][0.01%] Probe {cmd.upper()} #{probe_count}: "
                        f"notch {int(notch_used):+d} -> {int(score)}% @ {float(duration_scale):.2f}x, "
                        f"step {float(step_mm):.3f}mm, moved_frames {int(moved_frames)}/{int(DISCOVERY_CONFIRM_FRAMES)} "
                        f"{'DETECTED' if movement_detected else 'no-detect'}"
                    )
                    self.log.append(
                        {
                            "type": "turn_tiny_profile_probe",
                            "run_id": self.run_id,
                            "trial": self.current_trial,
                            "phase": self.current_phase,
                            "timestamp": time.time(),
                            **row,
                        }
                    )

                    if movement_detected:
                        selected_row = row
                        break

                if selected_row is None:
                    selected_notch = int(max(TURN_NOTCH_MIN, min(TURN_NOTCH_MAX, current_notch)))
                    score, duration_scale, _ = self._score_duration_for_notch(selected_notch)
                    selected_row = {
                        "cmd": str(cmd),
                        "current_notch_before_scan": int(current_notch),
                        "start_notch": int(start_notch),
                        "selected_notch": int(selected_notch),
                        "score": int(score),
                        "duration_scale": float(duration_scale),
                        "step_mm": 0.0,
                        "movement_detected": False,
                        "probes_recorded": int(probe_count),
                    }
                else:
                    selected_row = {
                        "cmd": str(cmd),
                        "current_notch_before_scan": int(current_notch),
                        "start_notch": int(start_notch),
                        "selected_notch": int(selected_row.get("notch", current_notch)),
                        "score": int(selected_row.get("score", 1)),
                        "duration_scale": float(selected_row.get("duration_scale", 1.0)),
                        "step_mm": float(selected_row.get("step_mm", 0.0)),
                        "movement_detected": bool(selected_row.get("movement_detected")),
                        "probes_recorded": int(probe_count),
                    }

                self.turn_notch_by_cmd[cmd] = int(selected_row["selected_notch"])
                discovered[cmd] = dict(selected_row)
                cmd_persist = self._persist_tiny_turn_profile_to_world_model({cmd: discovered[cmd]})
                self.log.append(
                    {
                        "type": "turn_tiny_profile_cmd_persist",
                        "run_id": self.run_id,
                        "timestamp": time.time(),
                        "cmd": str(cmd),
                        "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                        "selected_notch": int(selected_row["selected_notch"]),
                        "persist_result": cmd_persist,
                    }
                )
                print(
                    f"[X-AXIS] 0.01% discovery {cmd.upper()}: "
                    f"start notch {start_notch:+d} -> selected {int(selected_row['selected_notch']):+d} "
                    f"({int(selected_row['score'])}% @ {float(selected_row['duration_scale']):.2f}x, "
                    f"step {float(selected_row['step_mm']):.3f}mm)"
                )

            self.tiny_discovery_by_cmd = {k: dict(v) for k, v in discovered.items()}
            self.tiny_discovery_persist_result = self._persist_tiny_turn_profile_to_world_model(self.tiny_discovery_by_cmd)
            self.log.append(
                {
                    "type": "turn_tiny_profile_discovery_end",
                    "run_id": self.run_id,
                    "timestamp": time.time(),
                    "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                    "discovered_by_cmd": self.tiny_discovery_by_cmd,
                    "turn_notch_by_cmd": dict(self.turn_notch_by_cmd),
                    "persist_result": self.tiny_discovery_persist_result,
                },
                force=True,
            )

            discovery_complete = all(cmd in self.tiny_discovery_by_cmd for cmd in ("l", "r"))
            _SESSION_TINY_TURN_DISCOVERY = {
                "done": bool(discovery_complete),
                "by_cmd": {k: dict(v) for k, v in self.tiny_discovery_by_cmd.items()},
                "persist_result": dict(self.tiny_discovery_persist_result),
            }
            return self.tiny_discovery_by_cmd
        finally:
            self.current_phase = prev_phase

    @staticmethod
    def _step_target_band(abs_err_mm: float) -> tuple[float, float, float]:
        """
        Return (target_min_mm, target_max_mm, max_err_for_band_mm) for the
        current absolute x-axis error.
        """
        try:
            err = abs(float(abs_err_mm))
        except (TypeError, ValueError):
            err = 0.0
        for max_err, target_min, target_max in STEP_TARGET_BANDS_MM:
            if err <= float(max_err):
                lo = float(target_min)
                hi = float(target_max)
                if hi < lo:
                    lo, hi = hi, lo
                return float(lo), float(hi), float(max_err)
        max_err, target_min, target_max = STEP_TARGET_BANDS_MM[-1]
        lo = float(target_min)
        hi = float(target_max)
        if hi < lo:
            lo, hi = hi, lo
        return float(lo), float(hi), float(max_err)

    @staticmethod
    def _step_distance_to_band(step_mm: float, target_min_mm: float, target_max_mm: float) -> float:
        try:
            step = float(step_mm)
            lo = float(target_min_mm)
            hi = float(target_max_mm)
        except (TypeError, ValueError):
            return 0.0
        if hi < lo:
            lo, hi = hi, lo
        if step < lo:
            return float(lo - step)
        if step > hi:
            return float(step - hi)
        return 0.0

    def _send_turn(self, cmd: str, score: int, *, duration_scale=1.0, phase=None, record_history=True):
        cmd = str(cmd)
        score = int(max(1, min(MAX_COARSE_SCORE, int(score))))
        try:
            dur_scale = float(duration_scale)
        except (TypeError, ValueError):
            dur_scale = 1.0
        dur_scale = max(0.05, min(3.0, dur_scale))

        if cmd in ("l", "r"):
            prev_cmd = self._last_turn_cmd_sent
            if prev_cmd in ("l", "r") and prev_cmd != cmd and float(TURN_REVERSAL_SETTLE_S) > 0.0:
                self.log.append(
                    {
                        "type": "turn_reversal_settle",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "phase": str(phase) if phase else str(self.current_phase),
                        "timestamp": time.time(),
                        "cmd_from": str(prev_cmd),
                        "cmd_to": str(cmd),
                        "settle_seconds": float(TURN_REVERSAL_SETTLE_S),
                    }
                )
                time.sleep(float(TURN_REVERSAL_SETTLE_S))

        power, pwm, score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
        duration_req_ms = max(1, int(round(float(duration_model_ms) * dur_scale)))

        sent = send_robot_command_pwm(
            self.robot,
            self.world,
            StepState.ALIGN_BRICK,
            cmd,
            power,
            pwm,
            duration_req_ms,
            speed_score=score_used,
            auto_mode=False,
        )
        if not isinstance(sent, dict):
            return None
        try:
            sent["duration_model_ms"] = int(duration_model_ms)
        except (TypeError, ValueError):
            pass
        sent["duration_scale"] = float(dur_scale)
        if cmd in ("l", "r"):
            active_notch = int(self.turn_notch_by_cmd.get(cmd, 0))
            active_score, active_scale, _ = self._score_duration_for_notch(active_notch)
            sent["tiny_profile_target_pct"] = float(TARGET_MICRO_INTENSITY_PCT)
            sent["tiny_profile_active_notch"] = int(active_notch)
            sent["tiny_profile_active_score"] = int(active_score)
            sent["tiny_profile_active_duration_scale"] = float(active_scale)
            sent["tiny_profile_discovery_done_this_session"] = bool(_SESSION_TINY_TURN_DISCOVERY.get("done"))
            sent["tiny_profile_discovery_ran_this_instance"] = bool(self.tiny_discovery_ran_this_instance)
            self._last_turn_cmd_sent = str(cmd)
            if bool(record_history):
                self.recent_turn_acts.append(
                    {
                        "cmd": str(cmd),
                        "score": int(score_used),
                        "duration_scale": float(dur_scale),
                        "timestamp": time.time(),
                        "phase": str(phase) if phase else str(self.current_phase),
                    }
                )
            self.log.append(
                {
                    "type": "turn_runtime_profile",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "phase": str(phase) if phase else str(self.current_phase),
                    "timestamp": time.time(),
                    "cmd": str(cmd),
                    "score_requested": int(score),
                    "duration_scale_requested": float(dur_scale),
                    "tiny_profile_target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                    "tiny_profile_active_notch": int(active_notch),
                    "tiny_profile_active_score": int(active_score),
                    "tiny_profile_active_duration_scale": float(active_scale),
                    "tiny_profile_discovery_done_this_session": bool(_SESSION_TINY_TURN_DISCOVERY.get("done")),
                    "tiny_profile_discovery_ran_this_instance": bool(self.tiny_discovery_ran_this_instance),
                }
            )
        else:
            self._last_turn_cmd_sent = None
        return sent

    def _direction_change_warmup(self, cmd: str, *, phase: str):
        if not TURN_DIR_WARMUP_ENABLED or cmd not in ("l", "r"):
            return False
        if self._last_turn_cmd_for_trial == cmd:
            return False

        warm_score, warm_scale, warm_notch = self._score_duration_for_notch(TURN_DIR_WARMUP_NOTCH)
        sent = self._send_turn(
            cmd,
            warm_score,
            duration_scale=warm_scale,
            phase=phase,
            record_history=False,
        )
        self.log.append(
            {
                "type": "turn_warmup",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": str(phase),
                "timestamp": time.time(),
                "cmd": str(cmd),
                "notch": int(warm_notch),
                "score": int(warm_score),
                "duration_scale": float(warm_scale),
                "sent": sent if isinstance(sent, dict) else None,
            }
        )
        time.sleep(max(OBSERVE_SLEEP_S, CONTROL_SLEEP_S * 0.75))
        self._last_turn_cmd_for_trial = cmd
        return True

    @staticmethod
    def _inverse_turn_cmd(cmd: str):
        if cmd == "l":
            return "r"
        if cmd == "r":
            return "l"
        return None

    def _recover_from_lost_vision(self, *, source_phase: str):
        """
        Recover vision by replaying the inverse of the most recent turn acts.
        After each inverse act, immediately check whether the brick is visible.
        """
        history = list(self.recent_turn_acts)
        if not history:
            return None
        plan = []
        for act in reversed(history[-int(RECOVERY_MAX_INVERSE_ACTS) :]):
            cmd = self._inverse_turn_cmd(str(act.get("cmd")))
            if cmd not in ("l", "r"):
                continue
            score = int(max(1, min(MAX_COARSE_SCORE, int(act.get("score", 1)))))
            try:
                duration_scale = float(act.get("duration_scale", 1.0))
            except (TypeError, ValueError):
                duration_scale = 1.0
            duration_scale = max(0.05, min(3.0, duration_scale))
            plan.append(
                {
                    "cmd": cmd,
                    "score": int(score),
                    "duration_scale": float(duration_scale),
                    "origin_cmd": str(act.get("cmd")),
                    "origin_phase": str(act.get("phase")),
                    "origin_timestamp": act.get("timestamp"),
                }
            )
        if not plan:
            return None

        prev_phase = self.current_phase
        self.current_phase = "recovery"
        self.log.append(
            {
                "type": "recovery_start",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "source_phase": str(source_phase),
                "plan": plan,
            }
        )
        recovered_pose = None
        for idx, step in enumerate(plan, start=1):
            sent = self._send_turn(
                step["cmd"],
                int(step["score"]),
                duration_scale=float(step["duration_scale"]),
                phase="recovery",
                record_history=False,
            )
            time.sleep(CONTROL_SLEEP_S)
            pose = self._get_pose(samples=1, timeout_s=float(RECOVERY_CHECK_TIMEOUT_S))
            found = pose is not None
            self.log.append(
                {
                    "type": "recovery_act",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "index": int(idx),
                    "total": int(len(plan)),
                    "cmd": str(step["cmd"]),
                    "score": int(step["score"]),
                    "duration_scale": float(step["duration_scale"]),
                    "origin_cmd": str(step["origin_cmd"]),
                    "origin_phase": str(step["origin_phase"]),
                    "found_after_act": bool(found),
                    "offset_x": (float(pose["offset_x"]) if found else None),
                    "sent": sent if isinstance(sent, dict) else None,
                }
            )
            if found:
                recovered_pose = pose
                break

        self.log.append(
            {
                "type": "recovery_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "source_phase": str(source_phase),
                "recovered": bool(recovered_pose is not None),
                "acts_attempted": int(len(plan)),
            }
        )
        self.current_phase = prev_phase
        return recovered_pose

    def _probe_turn_notch(self, cmd: str):
        """
        Before each trial, start a few notches below 1% and ramp upward until
        movement is detected. Pick the notch closest to the adaptive step-size
        target band for the current |x_err|.
        """
        best = None
        first_detected = None
        probe_records = 0
        selected_target_min = float(STEP_TARGET_MIN_MM)
        selected_target_max = float(STEP_TARGET_MAX_MM)
        selected_target_err_max = float(STEP_TARGET_BANDS_MM[0][0])

        for notch in range(int(TURN_NOTCH_MIN), int(TURN_NOTCH_MAX) + 1):
            warmup_applied = self._direction_change_warmup(cmd, phase="probe")
            if warmup_applied:
                # Discard warmup movement; take a fresh baseline pose.
                self._get_pose(samples=1, timeout_s=1.0)
            pose_before = self._get_pose(samples=1, timeout_s=1.3)
            if not pose_before:
                pose_before = self._recover_from_lost_vision(source_phase="probe")
                if not pose_before:
                    break
            offset_before = float(pose_before["offset_x"])
            abs_err_before = abs(float(self._x_err_mm(offset_before)))
            target_min_mm, target_max_mm, target_err_max_mm = self._step_target_band(abs_err_before)
            selected_target_min = float(target_min_mm)
            selected_target_max = float(target_max_mm)
            selected_target_err_max = float(target_err_max_mm)

            score, duration_scale, notch_used = self._score_duration_for_notch(notch)
            sent = self._send_turn(cmd, score, duration_scale=duration_scale, phase="probe")
            time.sleep(CONTROL_SLEEP_S)

            pose_after = self._get_pose(samples=1, timeout_s=1.3)
            if not pose_after:
                pose_after = self._recover_from_lost_vision(source_phase="probe")
                if not pose_after:
                    break
            offset_after = float(pose_after["offset_x"])
            step_mm = abs(offset_after - offset_before)
            moved = bool(step_mm >= float(STEP_MOVEMENT_DETECT_MM))
            in_target = bool(float(target_min_mm) <= step_mm <= float(target_max_mm))
            probe_records += 1

            if moved and first_detected is None:
                first_detected = {
                    "notch": int(notch_used),
                    "step_mm": float(step_mm),
                    "target_min_mm": float(target_min_mm),
                    "target_max_mm": float(target_max_mm),
                    "target_err_max_mm": float(target_err_max_mm),
                }
            if moved:
                quality = self._step_distance_to_band(step_mm, target_min_mm, target_max_mm)
                if (
                    best is None
                    or (in_target and not best["in_target"])
                    or (in_target == best["in_target"] and quality < best["quality"])
                ):
                    best = {
                        "notch": int(notch_used),
                        "step_mm": float(step_mm),
                        "quality": float(quality),
                        "in_target": bool(in_target),
                        "target_min_mm": float(target_min_mm),
                        "target_max_mm": float(target_max_mm),
                        "target_err_max_mm": float(target_err_max_mm),
                    }
                if in_target:
                    # Early stop once we are in the active target band.
                    self.log.append(
                        {
                            "type": "turn_probe",
                            "run_id": self.run_id,
                            "trial": self.current_trial,
                            "phase": "probe",
                            "timestamp": time.time(),
                            "cmd": str(cmd),
                            "notch": int(notch_used),
                            "score": int(score),
                            "duration_scale": float(duration_scale),
                            "warmup_applied": bool(warmup_applied),
                            "offset_before": float(offset_before),
                            "offset_after": float(offset_after),
                            "abs_err_before_mm": float(abs_err_before),
                            "step_mm": float(step_mm),
                            "step_gap_to_target_mm": float(quality),
                            "movement_detected": bool(moved),
                            "in_target_band": bool(in_target),
                            "target_min_mm": float(target_min_mm),
                            "target_max_mm": float(target_max_mm),
                            "target_err_max_mm": float(target_err_max_mm),
                            "selected": True,
                            "sent": sent if isinstance(sent, dict) else None,
                        }
                    )
                    break

            self.log.append(
                {
                    "type": "turn_probe",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "phase": "probe",
                    "timestamp": time.time(),
                    "cmd": str(cmd),
                    "notch": int(notch_used),
                    "score": int(score),
                    "duration_scale": float(duration_scale),
                    "warmup_applied": bool(warmup_applied),
                    "offset_before": float(offset_before),
                    "offset_after": float(offset_after),
                    "abs_err_before_mm": float(abs_err_before),
                    "step_mm": float(step_mm),
                    "step_gap_to_target_mm": float(self._step_distance_to_band(step_mm, target_min_mm, target_max_mm)),
                    "movement_detected": bool(moved),
                    "in_target_band": bool(in_target),
                    "target_min_mm": float(target_min_mm),
                    "target_max_mm": float(target_max_mm),
                    "target_err_max_mm": float(target_err_max_mm),
                    "selected": False,
                    "sent": sent if isinstance(sent, dict) else None,
                }
            )

        if best is not None:
            selected_notch = int(best["notch"])
            selected_step = float(best["step_mm"])
            selected_target_min = float(best.get("target_min_mm", selected_target_min))
            selected_target_max = float(best.get("target_max_mm", selected_target_max))
            selected_target_err_max = float(best.get("target_err_max_mm", selected_target_err_max))
        elif first_detected is not None:
            selected_notch = int(first_detected["notch"])
            selected_step = float(first_detected["step_mm"])
            selected_target_min = float(first_detected.get("target_min_mm", selected_target_min))
            selected_target_max = float(first_detected.get("target_max_mm", selected_target_max))
            selected_target_err_max = float(first_detected.get("target_err_max_mm", selected_target_err_max))
        else:
            # If still no movement detected, bias one notch higher next trial.
            selected_notch = int(min(TURN_NOTCH_MAX, max(0, self.turn_notch_by_cmd.get(cmd, 0) + 1)))
            selected_step = 0.0

        self.turn_notch_by_cmd[cmd] = int(selected_notch)
        self.turn_step_mm_by_cmd[cmd].append(float(selected_step))

        score, duration_scale, notch_used = self._score_duration_for_notch(selected_notch)
        self.log.append(
            {
                "type": "turn_probe_summary",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": "probe",
                "timestamp": time.time(),
                "cmd": str(cmd),
                "selected_notch": int(notch_used),
                "selected_score": int(score),
                "selected_duration_scale": float(duration_scale),
                "selected_step_mm": float(selected_step),
                "probes_recorded": int(probe_records),
                "movement_threshold_mm": float(STEP_MOVEMENT_DETECT_MM),
                "target_step_min_mm": float(selected_target_min),
                "target_step_max_mm": float(selected_target_max),
                "target_err_max_mm": float(selected_target_err_max),
            }
        )
        return (
            int(notch_used),
            float(selected_step),
            int(score),
            float(duration_scale),
            float(selected_target_min),
            float(selected_target_max),
            float(selected_target_err_max),
        )

    @staticmethod
    def _step_in_target(step_mm, target_min_mm=STEP_TARGET_MIN_MM, target_max_mm=STEP_TARGET_MAX_MM) -> bool:
        try:
            value = float(step_mm)
            lo = float(target_min_mm)
            hi = float(target_max_mm)
        except (TypeError, ValueError):
            return False
        if hi < lo:
            lo, hi = hi, lo
        return float(lo) <= value <= float(hi)

    def _confidence_summary(self, trial_records: list[dict]) -> dict:
        total_trials = len(trial_records)
        obs_l = 0
        obs_r = 0
        subtle_hits_l = 0
        subtle_hits_r = 0
        for rec in trial_records:
            probe = rec.get("probe_summary") if isinstance(rec.get("probe_summary"), dict) else {}
            l_entry = probe.get("l") if isinstance(probe.get("l"), dict) else None
            r_entry = probe.get("r") if isinstance(probe.get("r"), dict) else None

            if isinstance(l_entry, dict) and l_entry.get("step_mm") is not None:
                obs_l += 1
                if self._step_in_target(
                    l_entry.get("step_mm"),
                    l_entry.get("target_min_mm", STEP_TARGET_MIN_MM),
                    l_entry.get("target_max_mm", STEP_TARGET_MAX_MM),
                ):
                    subtle_hits_l += 1

            if isinstance(r_entry, dict) and r_entry.get("step_mm") is not None:
                obs_r += 1
                if self._step_in_target(
                    r_entry.get("step_mm"),
                    r_entry.get("target_min_mm", STEP_TARGET_MIN_MM),
                    r_entry.get("target_max_mm", STEP_TARGET_MAX_MM),
                ):
                    subtle_hits_r += 1

        subtle_rate_l = float((subtle_hits_l / obs_l) if obs_l > 0 else 0.0)
        subtle_rate_r = float((subtle_hits_r / obs_r) if obs_r > 0 else 0.0)
        confident = bool(
            total_trials >= int(CONF_MIN_TRIALS)
            and obs_l >= int(CONF_MIN_OBS_PER_DIR)
            and obs_r >= int(CONF_MIN_OBS_PER_DIR)
            and subtle_rate_l >= float(CONF_SUBTLE_HIT_RATE)
            and subtle_rate_r >= float(CONF_SUBTLE_HIT_RATE)
        )
        return {
            "confident": bool(confident),
            "trials": int(total_trials),
            "obs_l": int(obs_l),
            "obs_r": int(obs_r),
            "subtle_hits_l": int(subtle_hits_l),
            "subtle_hits_r": int(subtle_hits_r),
            "subtle_rate_l": float(subtle_rate_l),
            "subtle_rate_r": float(subtle_rate_r),
            "criteria": {
                "min_trials": int(CONF_MIN_TRIALS),
                "min_obs_per_dir": int(CONF_MIN_OBS_PER_DIR),
                "min_subtle_rate": float(CONF_SUBTLE_HIT_RATE),
                "step_target_bands_mm": [list(row) for row in STEP_TARGET_BANDS_MM],
            },
        }

    def _closeness_summary(self, trial_records: list[dict]) -> dict:
        summary = {
            "tiers_mm": [float(t) for t in STEP_CLOSE_TIERS_MM],
            "adaptive_bands_mm": [list(row) for row in STEP_TARGET_BANDS_MM],
            "by_cmd": {},
        }
        for cmd in ("l", "r"):
            steps: list[float] = []
            adaptive_hits = 0
            adaptive_total = 0
            tier_counts = {float(t): 0 for t in STEP_CLOSE_TIERS_MM}
            for rec in trial_records:
                probe = rec.get("probe_summary") if isinstance(rec.get("probe_summary"), dict) else {}
                entry = probe.get(cmd) if isinstance(probe.get(cmd), dict) else None
                if not isinstance(entry, dict):
                    continue
                try:
                    step = float(entry.get("step_mm"))
                except (TypeError, ValueError):
                    continue
                steps.append(step)
                for tier in STEP_CLOSE_TIERS_MM:
                    if step <= float(tier):
                        tier_counts[float(tier)] += 1
                try:
                    target_min = float(entry.get("target_min_mm", STEP_TARGET_MIN_MM))
                    target_max = float(entry.get("target_max_mm", STEP_TARGET_MAX_MM))
                except (TypeError, ValueError):
                    target_min = float(STEP_TARGET_MIN_MM)
                    target_max = float(STEP_TARGET_MAX_MM)
                adaptive_total += 1
                if self._step_in_target(step, target_min, target_max):
                    adaptive_hits += 1

            if not steps:
                summary["by_cmd"][cmd] = {
                    "n": 0,
                    "mean_mm": None,
                    "median_mm": None,
                    "best_mm": None,
                    "worst_mm": None,
                    "adaptive_hit_count": int(adaptive_hits),
                    "adaptive_total": int(adaptive_total),
                    "adaptive_hit_rate": 0.0,
                    "tier_counts": {f"<={float(t):.2f}": 0 for t in STEP_CLOSE_TIERS_MM},
                    "tier_rates": {f"<={float(t):.2f}": 0.0 for t in STEP_CLOSE_TIERS_MM},
                }
                continue

            n = len(steps)
            tier_counts_fmt = {f"<={float(t):.2f}": int(tier_counts[float(t)]) for t in STEP_CLOSE_TIERS_MM}
            tier_rates_fmt = {
                f"<={float(t):.2f}": float(tier_counts[float(t)] / float(n))
                for t in STEP_CLOSE_TIERS_MM
            }
            summary["by_cmd"][cmd] = {
                "n": int(n),
                "mean_mm": float(statistics.mean(steps)),
                "median_mm": float(statistics.median(steps)),
                "best_mm": float(min(steps)),
                "worst_mm": float(max(steps)),
                "adaptive_hit_count": int(adaptive_hits),
                "adaptive_total": int(adaptive_total),
                "adaptive_hit_rate": float((adaptive_hits / adaptive_total) if adaptive_total > 0 else 0.0),
                "tier_counts": tier_counts_fmt,
                "tier_rates": tier_rates_fmt,
            }
        return summary

    def _print_closeness_summary(self, closeness: dict):
        print("[X-AXIS] Closeness summary (selected probe steps):")
        by_cmd = closeness.get("by_cmd") if isinstance(closeness, dict) else {}
        for cmd in ("l", "r"):
            label = "L" if cmd == "l" else "R"
            row = by_cmd.get(cmd) if isinstance(by_cmd, dict) else None
            if not isinstance(row, dict) or int(row.get("n") or 0) <= 0:
                print(f"[X-AXIS]  {label}: no probe data")
                continue
            n = int(row.get("n") or 0)
            mean_mm = float(row.get("mean_mm") or 0.0)
            median_mm = float(row.get("median_mm") or 0.0)
            best_mm = float(row.get("best_mm") or 0.0)
            adaptive_hit_count = int(row.get("adaptive_hit_count") or 0)
            adaptive_total = int(row.get("adaptive_total") or 0)
            tier_counts = row.get("tier_counts") if isinstance(row.get("tier_counts"), dict) else {}
            tier_bits = []
            for tier in STEP_CLOSE_TIERS_MM:
                key = f"<={float(tier):.2f}"
                tier_bits.append(f"{key}mm {int(tier_counts.get(key, 0))}/{n}")
            print(
                f"[X-AXIS]  {label}: median {median_mm:.2f}mm, mean {mean_mm:.2f}mm, best {best_mm:.2f}mm | "
                f"adaptive-hit {adaptive_hit_count}/{adaptive_total} | "
                + ", ".join(tier_bits)
            )

    def _probe_rows_for_cmd(self, trial_records: list[dict], cmd: str) -> list[dict]:
        rows = []
        for rec in trial_records:
            probe = rec.get("probe_summary") if isinstance(rec.get("probe_summary"), dict) else {}
            entry = probe.get(cmd) if isinstance(probe.get(cmd), dict) else None
            if not isinstance(entry, dict):
                continue
            try:
                step_mm = float(entry.get("step_mm"))
            except (TypeError, ValueError):
                continue
            if step_mm <= 0.0:
                continue
            try:
                duration_scale = float(entry.get("duration_scale", 1.0))
            except (TypeError, ValueError):
                duration_scale = 1.0
            duration_scale = max(0.01, min(3.0, duration_scale))
            try:
                target_min = float(entry.get("target_min_mm", STEP_TARGET_MIN_MM))
            except (TypeError, ValueError):
                target_min = float(STEP_TARGET_MIN_MM)
            try:
                target_max = float(entry.get("target_max_mm", STEP_TARGET_MAX_MM))
            except (TypeError, ValueError):
                target_max = float(STEP_TARGET_MAX_MM)
            if target_max < target_min:
                target_min, target_max = target_max, target_min
            try:
                notch = int(entry.get("notch", 0))
            except (TypeError, ValueError):
                notch = 0
            try:
                score = int(entry.get("score", 1))
            except (TypeError, ValueError):
                score = 1
            rows.append(
                {
                    "step_mm": float(step_mm),
                    "duration_scale": float(duration_scale),
                    "score": int(max(1, min(MAX_COARSE_SCORE, score))),
                    "notch": int(max(TURN_NOTCH_MIN, min(TURN_NOTCH_MAX, notch))),
                    "target_min_mm": float(target_min),
                    "target_max_mm": float(target_max),
                    "target_mid_mm": float((target_min + target_max) / 2.0),
                    "adaptive_hit": bool(self._step_in_target(step_mm, target_min, target_max)),
                    "adaptive_distance_mm": float(self._step_distance_to_band(step_mm, target_min, target_max)),
                }
            )
        return rows

    def _build_conclusion(self, trial_records: list[dict], confidence: dict, closeness: dict) -> dict:
        by_cmd = {}
        by_cmd_closeness = closeness.get("by_cmd") if isinstance(closeness.get("by_cmd"), dict) else {}

        for cmd in ("l", "r"):
            rows = self._probe_rows_for_cmd(trial_records, cmd)
            if not rows:
                by_cmd[cmd] = {
                    "available": False,
                    "samples": 0,
                }
                continue

            adaptive_hits = [row for row in rows if bool(row.get("adaptive_hit"))]
            selected_rows = adaptive_hits if adaptive_hits else rows

            steps = [float(row["step_mm"]) for row in selected_rows]
            target_mids = [float(row["target_mid_mm"]) for row in selected_rows]
            duration_scales = [float(row["duration_scale"]) for row in selected_rows]
            notches = [int(row["notch"]) for row in selected_rows]
            scores = [int(row["score"]) for row in selected_rows]
            median_step = float(statistics.median(steps))
            median_target_mid = float(statistics.median(target_mids)) if target_mids else float(STEP_TARGET_MID_MM)
            median_duration_scale = float(statistics.median(duration_scales)) if duration_scales else 1.0

            recommended_duration_scale = float(median_duration_scale)
            if median_step > 1e-6:
                recommended_duration_scale = float(median_duration_scale * (median_target_mid / median_step))
            recommended_duration_scale = max(0.20, min(3.0, float(recommended_duration_scale)))

            try:
                _, _, _, current_1pct_duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                    cmd,
                    int(getattr(telemetry_robot_module, "SPEED_SCORE_MIN", 1)),
                )
                current_1pct_duration_ms = int(max(1, int(round(float(current_1pct_duration_ms)))))
            except Exception:
                current_1pct_duration_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 250) or 250)
            recommended_1pct_duration_ms = int(
                max(1, int(round(float(current_1pct_duration_ms) * float(recommended_duration_scale))))
            )

            by_cmd[cmd] = {
                "available": True,
                "samples": int(len(rows)),
                "adaptive_hits": int(len(adaptive_hits)),
                "adaptive_hit_rate": float((len(adaptive_hits) / len(rows)) if rows else 0.0),
                "median_step_mm": float(median_step),
                "median_target_step_mm": float(median_target_mid),
                "median_notch": int(round(statistics.median(notches))) if notches else 0,
                "median_score": int(round(statistics.median(scores))) if scores else 1,
                "recommended_duration_scale_1pct": float(recommended_duration_scale),
                "current_duration_ms_1pct": int(current_1pct_duration_ms),
                "recommended_duration_ms_1pct": int(recommended_1pct_duration_ms),
                "recommended_seconds_1pct": round(float(recommended_1pct_duration_ms) / 1000.0, 3),
                "closeness": (
                    dict(by_cmd_closeness.get(cmd))
                    if isinstance(by_cmd_closeness.get(cmd), dict)
                    else None
                ),
            }

        return {
            "schema_version": 1,
            "run_id": int(self.run_id),
            "timestamp": float(time.time()),
            "x_gate_metric": str(self.x_gate_metric),
            "x_gate_target_mm": float(self.x_target_mm),
            "x_gate_tol_mm": float(self.x_tol_mm),
            "x_axis_sign": float(self.x_axis_sign),
            "confidence": dict(confidence) if isinstance(confidence, dict) else {},
            "recommended_by_cmd": by_cmd,
        }

    def _persist_conclusion_to_world_model(self, conclusion: dict, *, allow_persist: bool = True) -> dict:
        model_path = ROBOT_MODEL_FILE_DEFAULT
        if not bool(allow_persist):
            return {
                "ok": False,
                "skipped": True,
                "reason": "low_confidence",
                "model_path": str(model_path),
            }
        try:
            if model_path.exists():
                try:
                    model = json.loads(model_path.read_text())
                except (OSError, json.JSONDecodeError):
                    model = {}
            else:
                model = {}
            if not isinstance(model, dict):
                model = {}

            model[WORLD_MODEL_X_AXIS_SECTION_KEY] = dict(conclusion)

            applied_seconds = {}
            rec = conclusion.get("recommended_by_cmd") if isinstance(conclusion.get("recommended_by_cmd"), dict) else {}
            for cmd, seconds_key in (
                ("l", WORLD_MODEL_SPEED_SECONDS_TURN_LEFT_KEY),
                ("r", WORLD_MODEL_SPEED_SECONDS_TURN_RIGHT_KEY),
            ):
                cmd_row = rec.get(cmd) if isinstance(rec.get(cmd), dict) else None
                if not isinstance(cmd_row, dict) or not bool(cmd_row.get("available")):
                    continue
                seconds_1pct = cmd_row.get("recommended_seconds_1pct")
                try:
                    seconds_val = float(seconds_1pct)
                except (TypeError, ValueError):
                    continue
                if seconds_val <= 0.0:
                    continue
                sec_map = model.get(seconds_key)
                if not isinstance(sec_map, dict):
                    sec_map = {}
                    model[seconds_key] = sec_map
                sec_map["1"] = round(float(seconds_val), 3)
                applied_seconds[cmd] = round(float(seconds_val), 3)

                # Also persist an explicit tiny turn-intensity post
                # so turn interpolation can request tiny L/R pulses below score=1.
                intensity_key = (
                    WORLD_MODEL_TURN_INTENSITY_POSTS_LEFT_KEY
                    if cmd == "l"
                    else WORLD_MODEL_TURN_INTENSITY_POSTS_RIGHT_KEY
                )
                intensity_map = model.get(intensity_key)
                if not isinstance(intensity_map, dict):
                    intensity_map = {}
                    model[intensity_key] = intensity_map
                try:
                    base_scale_1pct = float(cmd_row.get("recommended_duration_scale_1pct", 1.0))
                except (TypeError, ValueError):
                    base_scale_1pct = 1.0
                duration_scale_01 = max(
                    0.01,
                    min(3.0, float(base_scale_1pct) * (float(TARGET_MICRO_INTENSITY_PCT) / 1.0)),
                )
                tiny_key = f"{float(TARGET_MICRO_INTENSITY_PCT):.2f}"
                existing = intensity_map.get(tiny_key)
                if not (isinstance(existing, dict) and str(existing.get("source")) == "mini_x_axis_discovery"):
                    intensity_map[tiny_key] = {
                        "score": 1,
                        "duration_scale": round(float(duration_scale_01), 3),
                    }

            model_path.write_text(json.dumps(model, indent=2) + "\n")
            return {
                "ok": True,
                "model_path": str(model_path),
                "applied_seconds_1pct": applied_seconds,
            }
        except OSError as exc:
            return {
                "ok": False,
                "model_path": str(model_path),
                "error": str(exc),
            }

    def _print_conclusion_summary(self, conclusion: dict, persist_result: dict):
        print("[X-AXIS] Conclusion (applied to world model 1% turn durations):")
        rec = conclusion.get("recommended_by_cmd") if isinstance(conclusion.get("recommended_by_cmd"), dict) else {}
        for cmd in ("l", "r"):
            label = "L" if cmd == "l" else "R"
            row = rec.get(cmd) if isinstance(rec.get(cmd), dict) else None
            if not isinstance(row, dict) or not bool(row.get("available")):
                print(f"[X-AXIS]  {label}: no conclusion (insufficient probe data)")
                continue
            print(
                f"[X-AXIS]  {label}: median step {float(row.get('median_step_mm', 0.0)):.2f}mm -> "
                f"recommend 1% {float(row.get('recommended_seconds_1pct', 0.0)):.3f}s "
                f"({int(row.get('recommended_duration_ms_1pct', 0))}ms, scale {float(row.get('recommended_duration_scale_1pct', 1.0)):.2f}x)"
            )
        if bool(persist_result.get("skipped")):
            print("[X-AXIS] World model update skipped (confidence gate not reached).")
            return
        if bool(persist_result.get("ok")):
            print(f"[X-AXIS] World model updated: {persist_result.get('model_path')}")
        else:
            print(f"[X-AXIS] World model update failed: {persist_result.get('error')}")

    def _update_world(self, found, angle, dist, offset_x, conf, cam_h, above, below, frame):
        self.world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
        try:
            if self._last_obs_offset_x is None:
                self._last_obs_delta_offset_x = None
            else:
                self._last_obs_delta_offset_x = float(offset_x) - float(self._last_obs_offset_x)
        except (TypeError, ValueError):
            self._last_obs_delta_offset_x = None
        self._last_obs_offset_x = offset_x

        if frame is None:
            return
        x_err = self._x_err_mm(offset_x)
        status = "OK" if abs(x_err) <= self.x_tol_mm else "PENDING"
        target_min_mm, target_max_mm, target_err_max_mm = self._step_target_band(abs(x_err))
        l_notch = int(self.turn_notch_by_cmd.get("l", 0))
        r_notch = int(self.turn_notch_by_cmd.get("r", 0))
        l_score, l_dur, _ = self._score_duration_for_notch(l_notch)
        r_score, r_dur, _ = self._score_duration_for_notch(r_notch)
        header = [
            f"TRIAL: {self.current_trial}/{self.num_trials}  PHASE: {self.current_phase}",
            f"X_ERR: {x_err:+.1f}mm  (tgt {self.x_target_mm:+.1f} tol {self.x_tol_mm:.1f})  {status}",
        ]
        display = frame.copy()
        draw_telemetry_overlay(
            display,
            self.world,
            header_lines=header,
            extra_messages=[
                f"X_OFF: {offset_x:+.1f}mm",
                (
                    f"target step: {target_min_mm:.2f}-{target_max_mm:.2f}mm "
                    f"(|x_err|<={target_err_max_mm:.0f}mm band)"
                ),
                f"L notch {l_notch:+d} -> {l_score}% @ {l_dur:.2f}x | R notch {r_notch:+d} -> {r_score}% @ {r_dur:.2f}x",
            ],
            highlight_metric=self.x_gate_metric,
        )
        with self.lock:
            self.current_frame = display

    def _get_pose(self, *, samples=1, timeout_s=1.5):
        poses = self._collect_offset_samples(samples=samples, timeout_s=timeout_s)
        if not poses or not self.running:
            return None
        return {"offset_x": float(statistics.mean(poses))}

    def _append_meta(self):
        self.log.append(
            {
                "type": "meta",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "num_trials": self.num_trials,
                "x_gate_metric": self.x_gate_metric,
                "x_gate_target_mm": self.x_target_mm,
                "x_gate_tol_mm": self.x_tol_mm,
                "x_axis_sign": self.x_axis_sign,
                "robot_model_file": str(ROBOT_MODEL_FILE_DEFAULT),
                "policy": {
                    "alpha": ALPHA,
                    "error_bins_mm": list(XAxisSpeedPolicy.ERROR_BINS_MM),
                    "score_candidates": list(XAxisSpeedPolicy.SCORE_CANDIDATES),
                    "default_score_by_bin": list(XAxisSpeedPolicy.DEFAULT_SCORE_BY_BIN),
                    "max_score_by_bin": list(XAxisSpeedPolicy.MAX_SCORE_BY_BIN),
                    "overshoot_penalty": OVERSHOOT_PENALTY,
                    "act_penalty": ACT_PENALTY,
                    "cross_deadzone_mm": CROSS_DEADZONE_MM,
                    "step_target_min_mm": STEP_TARGET_MIN_MM,
                    "step_target_max_mm": STEP_TARGET_MAX_MM,
                    "step_target_bands_mm": [list(row) for row in STEP_TARGET_BANDS_MM],
                    "step_close_tiers_mm": list(STEP_CLOSE_TIERS_MM),
                    "step_near_target_mm": STEP_NEAR_TARGET_MM,
                    "step_movement_detect_mm": STEP_MOVEMENT_DETECT_MM,
                    "step_reward_penalty_per_mm": STEP_REWARD_PENALTY_PER_MM,
                    "turn_notch_min": TURN_NOTCH_MIN,
                    "turn_notch_max": TURN_NOTCH_MAX,
                    "turn_notch_duration_step": TURN_NOTCH_DURATION_STEP,
                    "target_micro_intensity_pct": TARGET_MICRO_INTENSITY_PCT,
                    "discovery_start_below_current_notches": DISCOVERY_START_BELOW_CURRENT_NOTCHES,
                    "discovery_confirm_frames": DISCOVERY_CONFIRM_FRAMES,
                    "discovery_sample_timeout_s": DISCOVERY_SAMPLE_TIMEOUT_S,
                    "turn_dir_warmup_enabled": TURN_DIR_WARMUP_ENABLED,
                    "turn_dir_warmup_notch": TURN_DIR_WARMUP_NOTCH,
                    "turn_reversal_settle_s": TURN_REVERSAL_SETTLE_S,
                    "recovery_max_inverse_acts": RECOVERY_MAX_INVERSE_ACTS,
                    "recovery_check_timeout_s": RECOVERY_CHECK_TIMEOUT_S,
                    "confidence_min_trials": CONF_MIN_TRIALS,
                    "confidence_min_obs_per_dir": CONF_MIN_OBS_PER_DIR,
                    "confidence_min_subtle_rate": CONF_SUBTLE_HIT_RATE,
                },
            },
            force=True,
        )

    def _log_act(self, pending: PendingAct, sent: dict | None):
        hw_cmd = pending.cmd
        event = {
            "type": "act",
            "run_id": self.run_id,
            "trial": pending.trial,
            "phase": pending.phase,
            "act": pending.act,
            "timestamp": pending.timestamp,
            "offset_x": pending.offset_x,
            "delta_offset_x": self._last_obs_delta_offset_x,
            "x_err_mm": pending.x_err_mm,
            "abs_err_mm": pending.abs_err_mm,
            "target_mm": self.x_target_mm,
            "tol_mm": self.x_tol_mm,
            "cmd": pending.cmd,
            "hw_cmd": hw_cmd,
            "score_requested": int(pending.score),
            "epsilon": float(pending.epsilon),
            "bin": int(pending.bin_idx),
            "bin_label": XAxisSpeedPolicy.bin_label(int(pending.bin_idx)),
            "duration_scale": float(pending.duration_scale),
            "turn_notch": None if pending.notch is None else int(pending.notch),
            "micro_mode": bool(pending.micro_mode),
            "warmup_applied": bool(pending.warmup_applied),
        }
        if isinstance(sent, dict):
            event.update(sent)
        self.log.append(event)

    def _log_learn(
        self,
        *,
        trial: int,
        phase: str,
        act: int,
        prev: PendingAct,
        new_offset_x: float,
        new_x_err_mm: float,
        crossed: bool,
        overshoot: bool,
        reward: float,
        q_prev: float,
        q_next: float,
        delta_step_mm: float | None = None,
        step_penalty: float = 0.0,
        notch_before: int | None = None,
        notch_after: int | None = None,
    ):
        self.log.append(
            {
                "type": "learn",
                "run_id": self.run_id,
                "trial": int(trial),
                "phase": str(phase),
                "act": int(act),
                "timestamp": time.time(),
                "cmd": prev.cmd,
                "score": int(prev.score),
                "bin": int(prev.bin_idx),
                "bin_label": XAxisSpeedPolicy.bin_label(int(prev.bin_idx)),
                "prev_offset_x": float(prev.offset_x),
                "prev_x_err_mm": float(prev.x_err_mm),
                "prev_abs_err_mm": float(prev.abs_err_mm),
                "new_offset_x": float(new_offset_x),
                "new_x_err_mm": float(new_x_err_mm),
                "new_abs_err_mm": float(abs(new_x_err_mm)),
                "improvement_mm": float(prev.abs_err_mm - abs(new_x_err_mm)),
                "crossed_target": bool(crossed),
                "overshoot": bool(overshoot),
                "reward": float(reward),
                "delta_step_mm": None if delta_step_mm is None else float(delta_step_mm),
                "step_penalty": float(step_penalty),
                "notch_before": None if notch_before is None else int(notch_before),
                "notch_after": None if notch_after is None else int(notch_after),
                "q_prev": float(q_prev),
                "q_next": float(q_next),
            }
        )

    def _reset_to_random_offset(self):
        self.current_phase = "reset"
        pose = self._get_pose(samples=1, timeout_s=2.0)
        if not pose:
            pose = self._recover_from_lost_vision(source_phase="reset_init")
            if not pose:
                return None
        start_offset = float(pose["offset_x"])
        start_err = float(self._x_err_mm(start_offset))
        desired_abs_err = float(random.uniform(RESET_MIN_AWAY_MM, RESET_MAX_AWAY_MM))
        reset_dir = random.choice(["l", "r"])
        self.log.append(
            {
                "type": "trial_reset",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "start_offset_x": start_offset,
                "start_x_err_mm": start_err,
                "start_abs_err_mm": abs(start_err),
                "desired_abs_err_mm": desired_abs_err,
                "reset_dir_initial": reset_dir,
                "reset_score": int(RESET_SCORE),
            }
        )
        prev_abs_err = abs(start_err)
        max_reset_acts = 80
        reset_acts = 0
        last_offset = start_offset
        last_err = start_err
        while self.running and reset_acts < int(max_reset_acts):
            if prev_abs_err >= desired_abs_err:
                break
            self._send_turn(reset_dir, RESET_SCORE, phase=self.current_phase)
            reset_acts += 1
            time.sleep(CONTROL_SLEEP_S)

            pose = self._get_pose(samples=1, timeout_s=1.5)
            if not pose:
                pose = self._recover_from_lost_vision(source_phase="reset")
                if not pose:
                    break
            last_offset = float(pose["offset_x"])
            last_err = float(self._x_err_mm(last_offset))
            abs_err = abs(last_err)

            # Greedy "away from the gate": if we got closer, switch directions.
            if abs_err < prev_abs_err:
                reset_dir = "r" if reset_dir == "l" else "l"
            prev_abs_err = abs_err

        self.log.append(
            {
                "type": "trial_reset_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "reset_acts": int(reset_acts),
                "end_offset_x": float(last_offset),
                "end_x_err_mm": float(last_err),
                "end_abs_err_mm": float(prev_abs_err),
                "reset_dir_final": str(reset_dir),
            }
        )
        return self._get_pose(samples=1, timeout_s=2.0)

    def restore_to_offset(self, target_offset_x: float):
        """
        After calibration, return to the measured pre-run offset so callers keep a
        stable physical starting pose.
        """
        try:
            target = float(target_offset_x)
        except (TypeError, ValueError):
            return {"ok": False, "skipped": True, "reason": "invalid_target_offset"}

        tol_mm = float(max(0.1, RESTORE_TO_START_TOL_MM))
        max_acts = int(max(1, RESTORE_TO_START_MAX_ACTS))
        prev_phase = str(self.current_phase)
        self.current_phase = "restore_start_offset"

        pose = self._get_pose(samples=1, timeout_s=2.0)
        if not pose:
            pose = self._recover_from_lost_vision(source_phase="restore_start_offset_init")
        if not pose:
            self.current_phase = prev_phase
            return {"ok": False, "reason": "pose_unavailable_before_restore"}

        current = float(pose["offset_x"])
        delta = float(target - current)
        abs_delta = abs(delta)
        cmd = "r" if delta >= 0.0 else "l"
        acts = 0
        reason = "max_acts"
        success = bool(abs_delta <= tol_mm)

        self.log.append(
            {
                "type": "restore_start_offset_begin",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "target_offset_x": float(target),
                "start_offset_x": float(current),
                "start_abs_delta_mm": float(abs_delta),
                "tol_mm": float(tol_mm),
                "max_acts": int(max_acts),
            }
        )

        while self.running and acts < max_acts and not success:
            if abs_delta > float(STEP_NEAR_TARGET_MM):
                score = int(RESET_SCORE)
                duration_scale = 1.0
            else:
                notch = int(self.turn_notch_by_cmd.get(cmd, 0))
                score, duration_scale, _ = self._score_duration_for_notch(notch)

            sent = self._send_turn(
                cmd,
                int(score),
                duration_scale=float(duration_scale),
                phase=self.current_phase,
                record_history=False,
            )
            acts += 1
            time.sleep(CONTROL_SLEEP_S)

            pose = self._get_pose(samples=1, timeout_s=1.5)
            if not pose:
                pose = self._recover_from_lost_vision(source_phase="restore_start_offset")
                if not pose:
                    reason = "pose_lost_during_restore"
                    break

            next_current = float(pose["offset_x"])
            next_delta = float(target - next_current)
            next_abs_delta = abs(next_delta)
            improved = bool(next_abs_delta < abs_delta)

            self.log.append(
                {
                    "type": "restore_start_offset_act",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "act": int(acts),
                    "cmd": str(cmd),
                    "score": int(score),
                    "duration_scale": float(duration_scale),
                    "offset_x": float(next_current),
                    "abs_delta_mm": float(next_abs_delta),
                    "improved": bool(improved),
                    "sent": sent if isinstance(sent, dict) else None,
                }
            )

            current = float(next_current)
            delta = float(next_delta)
            abs_delta = float(next_abs_delta)
            if abs_delta <= tol_mm:
                success = True
                reason = "within_tolerance"
                break

            desired_cmd = "r" if delta >= 0.0 else "l"
            if not improved:
                cmd = self._inverse_turn_cmd(cmd) or desired_cmd
            else:
                cmd = desired_cmd

        if success and reason != "within_tolerance":
            reason = "within_tolerance"
        result = {
            "ok": bool(success),
            "reason": str(reason),
            "acts": int(acts),
            "target_offset_x": float(target),
            "final_offset_x": float(current),
            "remaining_mm": float(abs_delta),
            "tol_mm": float(tol_mm),
        }
        self.log.append(
            {
                "type": "restore_start_offset_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                **result,
            }
        )
        self.current_phase = prev_phase
        return result

    def run(self):
        self._append_meta()
        self._discover_tiny_turn_profiles_once()
        confidence_trials: list[dict] = []

        for trial in range(1, self.num_trials + 1):
            if not self.running:
                break
            self.current_trial = int(trial)
            self.act_idx = 0
            self._last_turn_cmd_for_trial = None
            self._last_turn_cmd_sent = None
            self.recent_turn_acts.clear()
            for history in self.turn_step_mm_by_cmd.values():
                history.clear()
            probe_summary = {}

            epsilon = max(0.08, 0.45 * (0.85 ** float(trial - 1)))
            self.log.append(
                {
                    "type": "trial_start",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "epsilon": float(epsilon),
                    "policy_best_scores": self.policy.snapshot_best_scores(),
                }
            )

            pose_after_reset = self._reset_to_random_offset()
            if not pose_after_reset:
                self.log.append(
                    {
                        "type": "trial_end",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "timestamp": time.time(),
                        "success": False,
                        "reason": "reset_failed_vision",
                    },
                    force=True,
                )
                print(
                    f"[X-AXIS] Trial {trial}/{self.num_trials} FAIL(reset_failed_vision) "
                    f"(acts={self.act_idx})"
                )
                confidence_trials.append(
                    {
                        "trial": int(self.current_trial),
                        "success": False,
                        "reason": "reset_failed_vision",
                        "probe_summary": {},
                    }
                )
                continue

            self.current_phase = "probe"
            for probe_cmd in ("l", "r"):
                notch_locked = int(self.turn_notch_by_cmd.get(probe_cmd, 0))
                score, duration_scale, notch = self._score_duration_for_notch(notch_locked)
                discovery_row = (
                    self.tiny_discovery_by_cmd.get(probe_cmd)
                    if isinstance(self.tiny_discovery_by_cmd.get(probe_cmd), dict)
                    else {}
                )
                step_mm = float(discovery_row.get("step_mm", 0.0))
                target_min_mm = float(STEP_TARGET_MIN_MM)
                target_max_mm = float(STEP_TARGET_MAX_MM)
                target_err_max_mm = float(STEP_TARGET_BANDS_MM[0][0])
                probe_summary[probe_cmd] = {
                    "notch": int(notch),
                    "step_mm": float(step_mm),
                    "score": int(score),
                    "duration_scale": float(duration_scale),
                    "target_min_mm": float(target_min_mm),
                    "target_max_mm": float(target_max_mm),
                    "target_err_max_mm": float(target_err_max_mm),
                    "locked_from_session_discovery": True,
                    "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
                }
                print(
                    f"[X-AXIS] Trial {trial} probe {probe_cmd.upper()}: "
                    f"locked 0.01% notch {notch:+d} -> {score}% @ {duration_scale:.2f}x "
                    f"(discovered step {step_mm:.3f}mm)"
                )
            self.log.append(
                {
                    "type": "trial_probe_end",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "probe_summary": probe_summary,
                }
            )

            self.current_phase = "align"
            self._last_turn_cmd_for_trial = None
            trial_start_t = time.time()
            success_frames = 0
            overshoot_events = 0
            pending: PendingAct | None = None
            end_reason = None

            while self.running:
                if (time.time() - trial_start_t) > float(TRIAL_TIMEOUT_S):
                    end_reason = "timeout"
                    break
                if self.act_idx >= int(MAX_ACTS_PER_TRIAL):
                    end_reason = "max_acts"
                    break

                pose = self._get_pose(samples=1, timeout_s=1.5)
                if not pose:
                    recovered_pose = self._recover_from_lost_vision(source_phase=self.current_phase)
                    if recovered_pose is not None:
                        pending = None
                        self.current_phase = "align"
                        continue
                    self.current_phase = "lost"
                    end_reason = "lost_vision"
                    break

                offset_x = float(pose["offset_x"])
                x_err = float(self._x_err_mm(offset_x))
                abs_err = float(abs(x_err))

                # Learn from the previous act.
                if pending is not None:
                    crossed = (
                        _sign(pending.x_err_mm, deadzone=CROSS_DEADZONE_MM) != 0
                        and _sign(x_err, deadzone=CROSS_DEADZONE_MM) != 0
                        and _sign(pending.x_err_mm, deadzone=CROSS_DEADZONE_MM) != _sign(x_err, deadzone=CROSS_DEADZONE_MM)
                    )
                    overshoot = bool(crossed)
                    if overshoot:
                        overshoot_events += 1

                    improvement = float(pending.abs_err_mm - abs_err)
                    reward = float(improvement) - float(ACT_PENALTY)
                    if overshoot:
                        reward -= float(OVERSHOOT_PENALTY)

                    delta_step_mm = abs(float(offset_x) - float(pending.offset_x))
                    step_penalty = 0.0
                    notch_before = pending.notch if pending.micro_mode else None
                    notch_after = notch_before
                    target_min_step, target_max_step, _ = self._step_target_band(pending.abs_err_mm)
                    if pending.micro_mode:
                        if delta_step_mm < float(target_min_step):
                            step_penalty = (float(target_min_step) - float(delta_step_mm)) * float(
                                STEP_REWARD_PENALTY_PER_MM
                            )
                        elif delta_step_mm > float(target_max_step):
                            step_penalty = (float(delta_step_mm) - float(target_max_step)) * float(
                                STEP_REWARD_PENALTY_PER_MM
                            )
                        reward -= float(step_penalty)
                        if pending.cmd in self.turn_step_mm_by_cmd:
                            self.turn_step_mm_by_cmd[pending.cmd].append(float(delta_step_mm))

                        if pending.notch is not None and not bool(self.tiny_discovery_by_cmd):
                            next_notch = int(pending.notch)
                            if delta_step_mm < float(target_min_step):
                                next_notch += 1
                            elif delta_step_mm > float(target_max_step):
                                next_notch -= 1
                            next_notch = max(int(TURN_NOTCH_MIN), min(int(TURN_NOTCH_MAX), int(next_notch)))
                            notch_after = int(next_notch)
                            if next_notch != int(pending.notch):
                                self.turn_notch_by_cmd[pending.cmd] = int(next_notch)
                                self.log.append(
                                    {
                                        "type": "turn_notch_adjust",
                                        "run_id": self.run_id,
                                        "trial": self.current_trial,
                                        "phase": self.current_phase,
                                        "act": pending.act,
                                        "timestamp": time.time(),
                                        "cmd": pending.cmd,
                                        "delta_step_mm": float(delta_step_mm),
                                        "target_min_mm": float(target_min_step),
                                        "target_max_mm": float(target_max_step),
                                        "notch_before": int(pending.notch),
                                        "notch_after": int(next_notch),
                                        "reason": "step_small" if delta_step_mm < float(target_min_step) else "step_large",
                                    }
                                )
                        elif pending.notch is not None and bool(self.tiny_discovery_by_cmd):
                            self.log.append(
                                {
                                    "type": "turn_notch_adjust_skipped_locked_profile",
                                    "run_id": self.run_id,
                                    "trial": self.current_trial,
                                    "phase": self.current_phase,
                                    "act": pending.act,
                                    "timestamp": time.time(),
                                    "cmd": pending.cmd,
                                    "notch_locked": int(pending.notch),
                                    "reason": "session_0.01_profile_locked",
                                }
                            )

                    q_prev, q_next = self.policy.update(pending.cmd, pending.bin_idx, pending.score, reward)
                    self._log_learn(
                        trial=self.current_trial,
                        phase=self.current_phase,
                        act=pending.act,
                        prev=pending,
                        new_offset_x=offset_x,
                        new_x_err_mm=x_err,
                        crossed=crossed,
                        overshoot=overshoot,
                        reward=reward,
                        q_prev=q_prev,
                        q_next=q_next,
                        delta_step_mm=delta_step_mm,
                        step_penalty=step_penalty,
                        notch_before=notch_before,
                        notch_after=notch_after,
                    )

                    if overshoot:
                        # Trial objective is zero overshoot: end immediately.
                        end_reason = "overshoot"
                        break

                # Success gate: require a few stable frames.
                if abs_err <= float(self.x_tol_mm):
                    success_frames += 1
                    if success_frames >= int(REQUIRED_SUCCESS_FRAMES):
                        end_reason = "success"
                        break
                else:
                    success_frames = 0

                cmd = "r" if x_err > 0.0 else "l"
                warmup_applied = self._direction_change_warmup(cmd, phase=self.current_phase)
                if warmup_applied:
                    pose_after_warmup = self._get_pose(samples=1, timeout_s=1.3)
                    if pose_after_warmup:
                        offset_x = float(pose_after_warmup["offset_x"])
                        x_err = float(self._x_err_mm(offset_x))
                        abs_err = float(abs(x_err))
                        cmd = "r" if x_err > 0.0 else "l"
                        if cmd != self._last_turn_cmd_for_trial:
                            warmup_applied = self._direction_change_warmup(cmd, phase=self.current_phase) or warmup_applied
                            pose_after_warmup = self._get_pose(samples=1, timeout_s=1.0)
                            if pose_after_warmup:
                                offset_x = float(pose_after_warmup["offset_x"])
                                x_err = float(self._x_err_mm(offset_x))
                                abs_err = float(abs(x_err))
                                cmd = "r" if x_err > 0.0 else "l"

                micro_mode = bool(abs_err <= float(STEP_NEAR_TARGET_MM))
                if micro_mode:
                    notch = int(self.turn_notch_by_cmd.get(cmd, 0))
                    score, duration_scale, notch = self._score_duration_for_notch(notch)
                    bin_idx = self.policy.bin_idx(abs_err)
                    eps_used = 0.0
                else:
                    score, bin_idx, eps_used = self.policy.choose_score(cmd, abs_err, epsilon=epsilon)
                    duration_scale = 1.0
                    notch = None

                sent = self._send_turn(cmd, score, duration_scale=duration_scale, phase=self.current_phase)

                pending = PendingAct(
                    trial=self.current_trial,
                    phase=self.current_phase,
                    act=self.act_idx,
                    timestamp=time.time(),
                    cmd=cmd,
                    score=int(score),
                    epsilon=float(eps_used),
                    offset_x=offset_x,
                    x_err_mm=x_err,
                    abs_err_mm=abs_err,
                    bin_idx=int(bin_idx),
                    duration_scale=float(duration_scale),
                    notch=(None if notch is None else int(notch)),
                    micro_mode=bool(micro_mode),
                    warmup_applied=bool(warmup_applied),
                )
                self._log_act(pending, sent)
                self.act_idx += 1

                time.sleep(CONTROL_SLEEP_S)

            # Trial summary
            final_pose = self._get_pose(samples=1, timeout_s=2.0)
            final_offset = float(final_pose["offset_x"]) if final_pose else None
            final_err = float(self._x_err_mm(final_offset)) if final_offset is not None else None
            success = bool(final_err is not None and abs(float(final_err)) <= float(self.x_tol_mm) and overshoot_events == 0)
            if end_reason is None:
                end_reason = "success" if success else "unknown"
            avg_step_l = (
                float(statistics.mean(self.turn_step_mm_by_cmd["l"])) if self.turn_step_mm_by_cmd.get("l") else None
            )
            avg_step_r = (
                float(statistics.mean(self.turn_step_mm_by_cmd["r"])) if self.turn_step_mm_by_cmd.get("r") else None
            )
            self.log.append(
                {
                    "type": "trial_end",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "success": bool(success),
                    "reason": str(end_reason),
                    "seconds": float(time.time() - trial_start_t),
                    "acts": int(self.act_idx),
                    "overshoot_events": int(overshoot_events),
                    "final_offset_x": final_offset,
                    "final_x_err_mm": final_err,
                    "turn_notch_l": int(self.turn_notch_by_cmd.get("l", 0)),
                    "turn_notch_r": int(self.turn_notch_by_cmd.get("r", 0)),
                    "avg_turn_step_l_mm": avg_step_l,
                    "avg_turn_step_r_mm": avg_step_r,
                    "policy_best_scores": self.policy.snapshot_best_scores(),
                },
                force=True,
            )

            status = "SUCCESS" if success else "FAIL"
            if not success and end_reason:
                status = f"FAIL({end_reason})"
            print(
                f"[X-AXIS] Trial {trial}/{self.num_trials} {status} "
                f"({time.time() - trial_start_t:.2f}s, acts={self.act_idx}, overshoots={overshoot_events}) "
                f"| notch L:{int(self.turn_notch_by_cmd.get('l', 0)):+d} "
                f"R:{int(self.turn_notch_by_cmd.get('r', 0)):+d}"
            )
            confidence_trials.append(
                {
                    "trial": int(self.current_trial),
                    "success": bool(success),
                    "reason": str(end_reason),
                    "probe_summary": dict(probe_summary),
                }
            )

        confidence = self._confidence_summary(confidence_trials)
        closeness = self._closeness_summary(confidence_trials)
        self.log.append(
            {
                "type": "confidence_summary",
                "run_id": self.run_id,
                "timestamp": time.time(),
                **confidence,
            },
            force=True,
        )
        self.log.append(
            {
                "type": "closeness_summary",
                "run_id": self.run_id,
                "timestamp": time.time(),
                **closeness,
            },
            force=True,
        )
        self._print_closeness_summary(closeness)
        conclusion = self._build_conclusion(confidence_trials, confidence, closeness)
        persist_result = self._persist_conclusion_to_world_model(
            conclusion,
            allow_persist=bool(confidence.get("confident")),
        )
        self.log.append(
            {
                "type": "conclusion",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "data": conclusion,
                "persist_result": persist_result,
            },
            force=True,
        )
        self._print_conclusion_summary(conclusion, persist_result)
        criteria = confidence.get("criteria") if isinstance(confidence.get("criteria"), dict) else {}
        subtle_rate_l = float(confidence.get("subtle_rate_l", 0.0)) * 100.0
        subtle_rate_r = float(confidence.get("subtle_rate_r", 0.0)) * 100.0
        if confidence.get("confident"):
            print(
                "[X-AXIS] Confidence reached: "
                f"subtle L {subtle_rate_l:.1f}%/{float(criteria.get('min_subtle_rate', CONF_SUBTLE_HIT_RATE)) * 100.0:.1f}%, "
                f"R {subtle_rate_r:.1f}%/{float(criteria.get('min_subtle_rate', CONF_SUBTLE_HIT_RATE)) * 100.0:.1f}%."
            )
        else:
            print(
                "[X-AXIS] Confidence not reached yet: "
                f"trials {confidence.get('trials')}/{criteria.get('min_trials', CONF_MIN_TRIALS)}, "
                f"obs L {confidence.get('obs_l')}/{criteria.get('min_obs_per_dir', CONF_MIN_OBS_PER_DIR)}, "
                f"R {confidence.get('obs_r')}/{criteria.get('min_obs_per_dir', CONF_MIN_OBS_PER_DIR)}, "
                f"subtle L {subtle_rate_l:.1f}%/{float(criteria.get('min_subtle_rate', CONF_SUBTLE_HIT_RATE)) * 100.0:.1f}%, "
                f"R {subtle_rate_r:.1f}%/{float(criteria.get('min_subtle_rate', CONF_SUBTLE_HIT_RATE)) * 100.0:.1f}%."
            )
        print("[X-AXIS] Done.")

    def run_tiny_turn_discovery_only(self) -> dict:
        self._append_meta()
        print(
            f"[X-AXIS] Running 0.01% L/R tiny-turn discovery only "
            f"(target={float(TARGET_MICRO_INTENSITY_PCT):.2f}%)."
        )
        discovered = self._discover_tiny_turn_profiles_once()
        summary = {
            "ok": True,
            "run_id": int(self.run_id),
            "target_pct": float(TARGET_MICRO_INTENSITY_PCT),
            "turn_notch_by_cmd": dict(self.turn_notch_by_cmd),
            "tiny_discovery_by_cmd": (
                dict(discovered)
                if isinstance(discovered, dict)
                else {}
            ),
            "tiny_discovery_persist_result": (
                dict(self.tiny_discovery_persist_result)
                if isinstance(self.tiny_discovery_persist_result, dict)
                else {}
            ),
            "tiny_discovery_ran_this_instance": bool(self.tiny_discovery_ran_this_instance),
        }
        self.log.append(
            {
                "type": "turn_tiny_profile_discovery_only_summary",
                "run_id": self.run_id,
                "timestamp": time.time(),
                **summary,
            },
            force=True,
        )
        for cmd in ("l", "r"):
            row = discovered.get(cmd) if isinstance(discovered, dict) else None
            if not isinstance(row, dict):
                continue
            print(
                f"[X-AXIS] 0.01% {cmd.upper()} summary: "
                f"selected notch {int(row.get('selected_notch', self.turn_notch_by_cmd.get(cmd, 0))):+d}, "
                f"step {float(row.get('step_mm', 0.0)):.3f}mm, "
                f"detected={bool(row.get('movement_detected'))}."
            )
        print(f"[X-AXIS] 0.01% profile persist: {summary['tiny_discovery_persist_result']}")
        print("[X-AXIS] Tiny-turn discovery only complete.")
        return summary

    def close(self):
        self.running = False
        try:
            self.log.flush(force=True)
        except Exception:
            pass
        try:
            self.robot.stop()
        except Exception:
            pass
        if bool(self._owns_robot):
            try:
                self.robot.close()
            except Exception:
                pass
        if bool(self._owns_vision):
            try:
                self.vision.close()
            except Exception:
                pass
        try:
            if self.server is not None:
                self.server.stop()
        except Exception:
            pass


def _latest_event(events, event_type):
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if isinstance(event, dict) and event.get("type") == event_type:
            return event
    return None


def _world_model_load(path: Path) -> dict:
    try:
        if path.exists():
            try:
                model = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                model = {}
        else:
            model = {}
    except Exception:
        model = {}
    if not isinstance(model, dict):
        return {}
    return model


def _auto_mini_last_run_epoch_s_from_model(model: dict) -> float | None:
    if not isinstance(model, dict):
        return None

    meta = model.get(WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY)
    if isinstance(meta, dict):
        try:
            ts = float(meta.get(WORLD_MODEL_AUTO_MINI_LAST_RUN_TS_KEY))
        except (TypeError, ValueError):
            ts = None
        if ts is not None and ts > 0.0:
            return float(ts)

    # Backward-compatible fallback: use the latest full/mini x-axis calibration timestamp.
    x_axis = model.get(WORLD_MODEL_X_AXIS_SECTION_KEY)
    if isinstance(x_axis, dict):
        try:
            ts = float(x_axis.get("timestamp"))
        except (TypeError, ValueError):
            ts = None
        if ts is not None and ts > 0.0:
            return float(ts)
    return None


def _format_utc_timestamp(epoch_s: float) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch_s)))
    except Exception:
        return ""


def _persist_auto_mini_run_meta(
    *,
    step_name: str,
    started_s: float,
    ended_s: float,
    min_interval_hours: float,
    ok: bool,
    confident: bool | None,
) -> dict:
    model_path = ROBOT_MODEL_FILE_DEFAULT
    try:
        model = _world_model_load(model_path)
        meta = model.get(WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY)
        if not isinstance(meta, dict):
            meta = {}
            model[WORLD_MODEL_SPEED_SECONDS_TURN_META_KEY] = meta

        try:
            started_val = float(started_s)
        except (TypeError, ValueError):
            started_val = float(time.time())
        try:
            ended_val = float(ended_s)
        except (TypeError, ValueError):
            ended_val = float(time.time())
        ended_val = max(started_val, ended_val)
        duration_s = max(0.0, float(ended_val - started_val))

        meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_TS_KEY] = round(float(ended_val), 3)
        meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_ISO_KEY] = _format_utc_timestamp(ended_val)
        meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_STEP_KEY] = str(step_name or "ALIGN_BRICK")
        meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_OK_KEY] = bool(ok)
        if confident is not None:
            meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_CONFIDENT_KEY] = bool(confident)
        meta[WORLD_MODEL_AUTO_MINI_LAST_RUN_DURATION_S_KEY] = round(float(duration_s), 3)
        meta[WORLD_MODEL_AUTO_MINI_MIN_INTERVAL_HOURS_KEY] = round(max(0.0, float(min_interval_hours)), 3)

        model_path.write_text(json.dumps(model, indent=2) + "\n")
        return {
            "ok": True,
            "model_path": str(model_path),
            WORLD_MODEL_AUTO_MINI_LAST_RUN_TS_KEY: float(meta.get(WORLD_MODEL_AUTO_MINI_LAST_RUN_TS_KEY)),
            WORLD_MODEL_AUTO_MINI_MIN_INTERVAL_HOURS_KEY: float(
                meta.get(WORLD_MODEL_AUTO_MINI_MIN_INTERVAL_HOURS_KEY)
            ),
        }
    except OSError as exc:
        return {
            "ok": False,
            "model_path": str(model_path),
            "error": str(exc),
        }


def run_mini_x_axis_calibration(
    *,
    robot,
    vision,
    step_key=None,
    trials=MINI_TRIALS_DEFAULT,
    log_path=RUN_LOG_FILE_DEFAULT,
    x_axis_sign=X_AXIS_SIGN_DEFAULT,
    log_fn=None,
    min_interval_hours=AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS_DEFAULT,
):
    """
    Run a quick in-process X-axis calibration using the caller's live robot/vision.
    Returns a summary dict and persists conclusion updates to world_model_robot.json.
    """
    logger = log_fn if callable(log_fn) else print
    step_name = str(step_key or "ALIGN_BRICK")
    try:
        cooldown_hours = max(0.0, float(min_interval_hours))
    except (TypeError, ValueError):
        cooldown_hours = float(AUTO_MINI_X_AXIS_MIN_INTERVAL_HOURS_DEFAULT)
    cooldown_seconds = float(cooldown_hours) * 3600.0

    if cooldown_seconds > 0.0:
        model = _world_model_load(ROBOT_MODEL_FILE_DEFAULT)
        last_run_epoch_s = _auto_mini_last_run_epoch_s_from_model(model)
        if last_run_epoch_s is not None:
            now_s = float(time.time())
            elapsed_s = max(0.0, float(now_s - float(last_run_epoch_s)))
            if elapsed_s < cooldown_seconds:
                remaining_s = float(cooldown_seconds - elapsed_s)
                next_allowed_s = float(last_run_epoch_s + cooldown_seconds)
                logger(
                    f"[AUTO] {step_name} mini x-axis calibration skipped "
                    f"({cooldown_hours:.1f}h cooldown active; {remaining_s / 3600.0:.2f}h remaining; "
                    f"last run {_format_utc_timestamp(last_run_epoch_s)})."
                )
                return {
                    "ok": True,
                    "skipped": True,
                    "skip_reason": "cooldown",
                    "cooldown_hours": float(cooldown_hours),
                    "last_run_epoch_s": float(last_run_epoch_s),
                    "next_allowed_epoch_s": float(next_allowed_s),
                    "remaining_seconds": float(remaining_s),
                }

    logger(f"[AUTO] {step_name} preflight mini x-axis calibration ({int(max(1, int(trials)))} trial).")
    calibrator = XAxisCalibrator(
        num_trials=int(max(1, int(trials))),
        stream_port=int(STREAM_PORT_DEFAULT),
        log_path=Path(log_path),
        x_axis_sign=float(x_axis_sign),
        robot=robot,
        vision=vision,
        enable_stream=False,
    )
    start_pose = calibrator._get_pose(samples=1, timeout_s=2.0)
    if not start_pose:
        start_pose = calibrator._recover_from_lost_vision(source_phase="preflight_start_pose")
    try:
        start_offset_x = float(start_pose["offset_x"]) if start_pose else None
    except (TypeError, ValueError, KeyError):
        start_offset_x = None
    started = time.time()
    try:
        try:
            calibrator.run()
        except Exception as exc:
            if start_offset_x is not None:
                try:
                    restore_result = calibrator.restore_to_offset(start_offset_x)
                except Exception as restore_exc:
                    restore_result = {
                        "ok": False,
                        "reason": "restore_exception",
                        "error": str(restore_exc),
                    }
            else:
                restore_result = {
                    "ok": False,
                    "skipped": True,
                    "reason": "start_pose_unavailable",
                }
            ended = float(time.time())
            meta_result = _persist_auto_mini_run_meta(
                step_name=step_name,
                started_s=float(started),
                ended_s=float(ended),
                min_interval_hours=float(cooldown_hours),
                ok=False,
                confident=None,
            )
            summary = {
                "ok": False,
                "error": str(exc),
                "seconds": float(ended - started),
                "run_meta_persist_result": meta_result,
                "restore_start_pose_result": restore_result,
            }
            logger(f"[AUTO] {step_name} mini x-axis calibration failed: {exc}")
            if bool(restore_result.get("ok")):
                logger(
                    f"[AUTO] {step_name} mini x-axis reset to start pose "
                    f"(remaining={float(restore_result.get('remaining_mm', 0.0)):.2f}mm, "
                    f"acts={int(restore_result.get('acts', 0))})."
                )
            elif not bool(restore_result.get("skipped")):
                logger(
                    f"[AUTO] {step_name} mini x-axis could not fully reset to start pose "
                    f"({restore_result.get('reason')})."
                )
            if not bool(meta_result.get("ok")):
                logger(f"[AUTO] {step_name} mini x-axis run meta update failed: {meta_result.get('error')}")
            return summary

        events = calibrator.log.events if hasattr(calibrator, "log") else []
        confidence = _latest_event(events, "confidence_summary") or {}
        conclusion = _latest_event(events, "conclusion") or {}
        persist = conclusion.get("persist_result") if isinstance(conclusion, dict) else {}
        data = conclusion.get("data") if isinstance(conclusion, dict) else {}

        summary = {
            "ok": True,
            "seconds": float(time.time() - started),
            "confidence": confidence,
            "conclusion": data if isinstance(data, dict) else {},
            "persist_result": persist if isinstance(persist, dict) else {},
            "tiny_discovery_by_cmd": (
                dict(calibrator.tiny_discovery_by_cmd)
                if isinstance(getattr(calibrator, "tiny_discovery_by_cmd", None), dict)
                else {}
            ),
            "tiny_discovery_persist_result": (
                dict(calibrator.tiny_discovery_persist_result)
                if isinstance(getattr(calibrator, "tiny_discovery_persist_result", None), dict)
                else {}
            ),
            "tiny_discovery_ran_this_instance": bool(getattr(calibrator, "tiny_discovery_ran_this_instance", False)),
        }
        if start_offset_x is not None:
            try:
                restore_result = calibrator.restore_to_offset(start_offset_x)
            except Exception as restore_exc:
                restore_result = {
                    "ok": False,
                    "reason": "restore_exception",
                    "error": str(restore_exc),
                }
        else:
            restore_result = {
                "ok": False,
                "skipped": True,
                "reason": "start_pose_unavailable",
            }
        summary["restore_start_pose_result"] = restore_result
        confidence_dict = summary["confidence"] if isinstance(summary["confidence"], dict) else {}
        confident = bool(confidence_dict.get("confident")) if confidence_dict else None
        meta_result = _persist_auto_mini_run_meta(
            step_name=step_name,
            started_s=float(started),
            ended_s=float(time.time()),
            min_interval_hours=float(cooldown_hours),
            ok=True,
            confident=confident,
        )
        summary["run_meta_persist_result"] = meta_result
        if bool(summary["persist_result"].get("ok")):
            logger(
                f"[AUTO] {step_name} mini x-axis calibration applied "
                f"(L/R 1% secs: {summary['persist_result'].get('applied_seconds_1pct', {})})."
            )
        elif bool(summary["persist_result"].get("skipped")):
            logger(
                f"[AUTO] {step_name} mini x-axis calibration did not apply "
                f"(confidence gate not reached)."
            )
        else:
            logger(
                f"[AUTO] {step_name} mini x-axis calibration completed but world model update failed: "
                f"{summary['persist_result'].get('error')}"
            )
        tiny_persist = summary.get("tiny_discovery_persist_result")
        if isinstance(tiny_persist, dict):
            if bool(tiny_persist.get("ok")):
                logger(
                    f"[AUTO] {step_name} 0.01% turn profile updated "
                    f"({tiny_persist.get('applied')})."
                )
            elif tiny_persist:
                logger(
                    f"[AUTO] {step_name} 0.01% turn profile update failed: "
                    f"{tiny_persist.get('error')}"
                )
        if bool(restore_result.get("ok")):
            logger(
                f"[AUTO] {step_name} mini x-axis reset to start pose "
                f"(remaining={float(restore_result.get('remaining_mm', 0.0)):.2f}mm, "
                f"acts={int(restore_result.get('acts', 0))})."
            )
        elif not bool(restore_result.get("skipped")):
            logger(
                f"[AUTO] {step_name} mini x-axis could not fully reset to start pose "
                f"({restore_result.get('reason')})."
            )
        if not bool(meta_result.get("ok")):
            logger(f"[AUTO] {step_name} mini x-axis run meta update failed: {meta_result.get('error')}")
        return summary
    finally:
        calibrator.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate X-axis alignment turn scores (ALIGN_BRICK).")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS_DEFAULT)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT_DEFAULT)
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    parser.add_argument("--x-axis-sign", type=float, default=X_AXIS_SIGN_DEFAULT)
    parser.add_argument(
        "--discover-tiny-only",
        action="store_true",
        help="Run only 0.01% L/R tiny-turn discovery and persist profile updates.",
    )
    parser.add_argument("--stream", dest="stream", action="store_true", help="Enable mini calibration livestream")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable mini calibration livestream")
    parser.set_defaults(stream=False)
    args = parser.parse_args()

    calibrator = XAxisCalibrator(
        num_trials=args.trials,
        stream_port=args.stream_port,
        log_path=Path(args.log),
        x_axis_sign=args.x_axis_sign,
        enable_stream=bool(args.stream),
    )
    try:
        if bool(args.discover_tiny_only):
            summary = calibrator.run_tiny_turn_discovery_only()
            print(json.dumps(summary, indent=2))
        else:
            calibrator.run()
    except KeyboardInterrupt:
        print("\n[X-AXIS] Interrupted.")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
