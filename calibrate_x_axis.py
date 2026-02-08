#!/usr/bin/env python3
"""
calibrate_x_axis.py
-------------------
Compartmentalized experiment to learn turn speed-scores that close the brick X-axis gap
for ALIGN_BRICK without overshoot events.

Overshoot definition (experiment-local):
  - The X error crosses the target (sign flips) beyond a small deadzone.
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

import cv2

import telemetry_robot as telemetry_robot_module
from helper_gate_utils import load_process_steps
from helper_robot_control import Robot
from helper_stream_server import StreamServer, format_stream_url
from helper_vision_aruco import ArucoBrickVision
from telemetry_process import send_robot_command
from telemetry_robot import WorldModel, StepState, draw_telemetry_overlay


NUM_TRIALS_DEFAULT = 10
STREAM_PORT_DEFAULT = 5000

RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent.parent / "world_model_x_axis.json"

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

# Overshoot / reward shaping
CROSS_DEADZONE_MM = 0.3
OVERSHOOT_PENALTY = 25.0
ACT_PENALTY = 0.05

# If left/right feel backwards, set --x-axis-sign -1.
X_AXIS_SIGN_DEFAULT = 1.0


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
    ):
        self.num_trials = int(num_trials)
        self.x_axis_sign = float(x_axis_sign)

        self.x_gate_metric, self.x_target_mm, self.x_tol_mm = _load_x_gate_from_process()

        self.robot = Robot()
        self.vision = ArucoBrickVision(debug=False)
        self.world = WorldModel()
        self.world.step_state = StepState.ALIGN_BRICK

        self.run_id = int(time.time())
        self.current_trial = 0
        self.current_phase = "init"
        self.act_idx = 0

        self._last_obs_offset_x = None
        self._last_obs_delta_offset_x = None

        self.policy = XAxisSpeedPolicy(alpha=ALPHA)

        self.log = JsonEventLog(Path(log_path), flush_every=50)
        self.log.wipe()

        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True

        self.server = StreamServer(self._get_frame, port=int(stream_port))
        self.server.start()

        print(f"[X-AXIS] Livestream: {format_stream_url('127.0.0.1', int(stream_port))}")
        print(
            f"[X-AXIS] Gate ({self.x_gate_metric}): target {self.x_target_mm:+.2f}mm tol {self.x_tol_mm:.2f}mm "
            f"(sign {self.x_axis_sign:+.0f})"
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
            ],
            highlight_metric=self.x_gate_metric,
        )
        with self.lock:
            self.current_frame = display

    def _get_pose(self, *, samples=1, timeout_s=1.5):
        poses = []
        start_t = time.time()
        while len(poses) < int(samples) and (time.time() - start_t) < float(timeout_s) and self.running:
            found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
            self._update_world(found, angle, dist, offset_x, conf, cam_h, above, below, self.vision.current_frame)
            if found:
                poses.append(float(offset_x))
                if len(poses) < int(samples):
                    time.sleep(OBSERVE_SLEEP_S)
            else:
                time.sleep(OBSERVE_SLEEP_S)
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
                "policy": {
                    "alpha": ALPHA,
                    "error_bins_mm": list(XAxisSpeedPolicy.ERROR_BINS_MM),
                    "score_candidates": list(XAxisSpeedPolicy.SCORE_CANDIDATES),
                    "default_score_by_bin": list(XAxisSpeedPolicy.DEFAULT_SCORE_BY_BIN),
                    "max_score_by_bin": list(XAxisSpeedPolicy.MAX_SCORE_BY_BIN),
                    "overshoot_penalty": OVERSHOOT_PENALTY,
                    "act_penalty": ACT_PENALTY,
                    "cross_deadzone_mm": CROSS_DEADZONE_MM,
                },
            },
            force=True,
        )

    def _send_turn(self, cmd: str, score: int):
        cmd = str(cmd)
        score = int(max(1, min(MAX_COARSE_SCORE, int(score))))
        sent = send_robot_command(
            self.robot,
            self.world,
            StepState.ALIGN_BRICK,
            cmd,
            0.0,
            speed_score=score,
        )
        return sent if isinstance(sent, dict) else None

    def _log_act(self, pending: PendingAct, sent: dict | None):
        hw_cmd = None
        try:
            hw_cmd = (getattr(telemetry_robot_module, "COMMAND_REMAP", {}) or {}).get(pending.cmd, pending.cmd)
        except Exception:
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
                "q_prev": float(q_prev),
                "q_next": float(q_next),
            }
        )

    def _reset_to_random_offset(self):
        self.current_phase = "reset"
        pose = self._get_pose(samples=1, timeout_s=2.0)
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
            self._send_turn(reset_dir, RESET_SCORE)
            reset_acts += 1
            time.sleep(CONTROL_SLEEP_S)

            pose = self._get_pose(samples=1, timeout_s=1.5)
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

    def run(self):
        self._append_meta()

        for trial in range(1, self.num_trials + 1):
            if not self.running:
                break
            self.current_trial = int(trial)
            self.act_idx = 0

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
                continue

            self.current_phase = "align"
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
                score, bin_idx, eps_used = self.policy.choose_score(cmd, abs_err, epsilon=epsilon)
                sent = self._send_turn(cmd, score)

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
                    "policy_best_scores": self.policy.snapshot_best_scores(),
                },
                force=True,
            )

            status = "SUCCESS" if success else "FAIL"
            if not success and end_reason:
                status = f"FAIL({end_reason})"
            print(
                f"[X-AXIS] Trial {trial}/{self.num_trials} {status} "
                f"({time.time() - trial_start_t:.2f}s, acts={self.act_idx}, overshoots={overshoot_events})"
            )

        print("[X-AXIS] Done.")

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
        try:
            self.robot.close()
        except Exception:
            pass
        try:
            self.vision.close()
        except Exception:
            pass
        try:
            self.server.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Calibrate X-axis alignment turn scores (ALIGN_BRICK).")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS_DEFAULT)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT_DEFAULT)
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    parser.add_argument("--x-axis-sign", type=float, default=X_AXIS_SIGN_DEFAULT)
    args = parser.parse_args()

    calibrator = XAxisCalibrator(
        num_trials=args.trials,
        stream_port=args.stream_port,
        log_path=Path(args.log),
        x_axis_sign=args.x_axis_sign,
    )
    try:
        calibrator.run()
    except KeyboardInterrupt:
        print("\n[X-AXIS] Interrupted.")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
