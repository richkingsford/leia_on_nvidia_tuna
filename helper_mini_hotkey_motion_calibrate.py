#!/usr/bin/env python3
"""
helper_mini_hotkey_motion_calibrate.py
--------------------------------------
Mini movement calibration helper for manual hotkeys (r/f/o/k).

Discovery model:
- For each target hotkey, start 5 score steps below its current score.
- Send one command pulse, then verify movement over 3 post-move frames.
- If movement is not confirmed, increase score by 1 and retry.
- First score with confirmed movement is selected and persisted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from telemetry_process import send_robot_command_pwm
from telemetry_robot import StepState, WorldModel


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_hotkey_motion_discovery.json"
ROBOT_MODEL_FILE_DEFAULT = Path(
    getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", Path(__file__).resolve().parent / "world_model_robot.json")
)

HOTKEY_SPEED_SCORES_KEY = "hotkey_speed_scores"
HOTKEY_DISCOVERY_META_KEY = "hotkey_motion_discovery_meta"

TARGET_HOTKEYS_DEFAULT = ("r", "f", "o", "k")
HOTKEY_FALLBACK_MAP = {
    "r": {"cmd": "f", "score": 1},
    "f": {"cmd": "b", "score": 1},
    "o": {"cmd": "u", "score": 1},
    "k": {"cmd": "d", "score": 1},
}

DISCOVERY_START_BELOW_SCORE_STEPS = 5
DISCOVERY_CONFIRM_FRAMES = 3
DISCOVERY_SAMPLE_TIMEOUT_S = 1.5
OBSERVE_SLEEP_S = 0.02
CONTROL_SLEEP_S = 0.04
ADDITIONAL_PAUSE_MS = 250
DISCOVERY_DURATION_NOTCH_STEP_MS = 20
DISCOVERY_DURATION_MIN_MS = 20

DIST_MOVEMENT_DETECT_MM = 0.80
CAM_H_MOVEMENT_DETECT_MM = 0.60

SCORE_MIN = 1
SCORE_MAX = 100

_SESSION_HOTKEY_DISCOVERY = {
    "done": False,
    "by_hotkey": {},
    "persist_result": {},
}


class JsonEventLog:
    def __init__(self, path: Path, *, flush_every=20):
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
            self.path.write_text(json.dumps(self.events, indent=(2 if force else None)) + ("\n" if force else ""))
            self._unsaved = 0
        except OSError as exc:
            print(f"[HOTKEY-MINI] Failed writing log {self.path}: {exc}")


def _world_model_load(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _format_utc_timestamp(epoch_s: float) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch_s)))
    except Exception:
        return ""


def _clamp_score(value: int) -> int:
    return int(max(int(SCORE_MIN), min(int(SCORE_MAX), int(value))))


def _clamp_duration_ms(value, fallback: int) -> int:
    try:
        duration_ms = int(round(float(value)))
    except (TypeError, ValueError):
        duration_ms = int(fallback)
    return int(max(int(DISCOVERY_DURATION_MIN_MS), duration_ms))


class HotkeyMotionCalibrator:
    def __init__(
        self,
        *,
        hotkeys: tuple[str, ...] = TARGET_HOTKEYS_DEFAULT,
        log_path: Path = RUN_LOG_FILE_DEFAULT,
        robot=None,
        vision=None,
    ):
        self.hotkeys = tuple(str(h).lower() for h in hotkeys if str(h).strip())
        self._owns_robot = robot is None
        self._owns_vision = vision is None
        self.robot = robot if robot is not None else Robot()
        self.vision = vision if vision is not None else ArucoBrickVision(debug=False)
        self.world = WorldModel()
        self.world.step_state = StepState.ALIGN_BRICK
        self.run_id = int(time.time())
        self.running = True

        self.log = JsonEventLog(Path(log_path), flush_every=20)
        self.log.wipe()

        self.discovery_by_hotkey: dict[str, dict] = {}
        self.persist_result: dict = {}
        self.discovery_ran_this_instance = False

    @staticmethod
    def _metric_for_cmd(cmd: str) -> str:
        if cmd in ("u", "d"):
            return "cam_h"
        return "dist"

    @staticmethod
    def _movement_threshold_for_metric(metric: str) -> float:
        if metric == "cam_h":
            return float(CAM_H_MOVEMENT_DETECT_MM)
        return float(DIST_MOVEMENT_DETECT_MM)

    def _load_hotkey_targets(self) -> dict[str, dict]:
        model = _world_model_load(ROBOT_MODEL_FILE_DEFAULT)
        loaded_map = model.get(HOTKEY_SPEED_SCORES_KEY)
        if not isinstance(loaded_map, dict):
            loaded_map = {}
        fallback_map = telemetry_robot_module.HOTKEY_SPEED_SCORES
        if not isinstance(fallback_map, dict):
            fallback_map = {}

        out = {}
        for hotkey in self.hotkeys:
            row = loaded_map.get(hotkey)
            if not isinstance(row, dict):
                row = fallback_map.get(hotkey)
            if not isinstance(row, dict):
                row = HOTKEY_FALLBACK_MAP.get(hotkey)
            if not isinstance(row, dict):
                continue
            cmd = str(row.get("cmd", "")).strip().lower()
            if cmd not in ("f", "b", "u", "d"):
                fallback = HOTKEY_FALLBACK_MAP.get(hotkey, {})
                cmd = str(fallback.get("cmd", "")).strip().lower()
            try:
                score = _clamp_score(int(row.get("score", HOTKEY_FALLBACK_MAP.get(hotkey, {}).get("score", 1))))
            except (TypeError, ValueError):
                score = int(HOTKEY_FALLBACK_MAP.get(hotkey, {}).get("score", 1))
            if cmd not in ("f", "b", "u", "d"):
                continue
            _power, _pwm, _score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
            try:
                duration_row = int(round(float(row.get("duration_ms"))))
            except (TypeError, ValueError):
                duration_row = int(duration_model_ms)
            duration_ms = _clamp_duration_ms(duration_row, int(duration_model_ms))
            out[hotkey] = {
                "hotkey": hotkey,
                "cmd": cmd,
                "score": int(score),
                "duration_ms": int(duration_ms),
            }
        return out

    def _collect_metric_samples(self, *, metric: str, samples=1, timeout_s=1.5) -> list[float]:
        vals: list[float] = []
        started = time.time()
        requested = max(1, int(samples))
        while len(vals) < requested and (time.time() - started) < float(timeout_s) and self.running:
            found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
            self.world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
            if found:
                raw = cam_h if metric == "cam_h" else dist
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    pass
            time.sleep(float(OBSERVE_SLEEP_S))
        return vals

    def _send_motion(self, cmd: str, score: int, *, duration_override_ms: int | None = None):
        power, pwm, score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
        duration_sent_ms = int(duration_model_ms)
        if duration_override_ms is not None:
            duration_sent_ms = _clamp_duration_ms(duration_override_ms, int(duration_model_ms))
        sent = send_robot_command_pwm(
            self.robot,
            self.world,
            StepState.ALIGN_BRICK,
            cmd,
            power,
            pwm,
            duration_sent_ms,
            speed_score=score_used,
            auto_mode=False,
        )
        if isinstance(sent, dict):
            sent["duration_model_ms"] = int(duration_model_ms)
            sent["duration_sent_ms"] = int(duration_sent_ms)
            if duration_override_ms is not None:
                sent["duration_override_ms"] = int(duration_sent_ms)
        return sent if isinstance(sent, dict) else None

    def _discover_one_hotkey(self, row: dict) -> dict:
        hotkey = str(row.get("hotkey"))
        cmd = str(row.get("cmd"))
        current_score = _clamp_score(int(row.get("score", 1)))
        _power, _pwm, _score_used, duration_model_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, current_score)
        current_duration_ms = _clamp_duration_ms(row.get("duration_ms"), int(duration_model_ms))
        metric = self._metric_for_cmd(cmd)
        movement_threshold = self._movement_threshold_for_metric(metric)
        start_score = _clamp_score(current_score - int(DISCOVERY_START_BELOW_SCORE_STEPS))
        start_duration_ms = _clamp_duration_ms(
            int(current_duration_ms) - int(DISCOVERY_START_BELOW_SCORE_STEPS) * int(DISCOVERY_DURATION_NOTCH_STEP_MS),
            int(current_duration_ms),
        )

        selected_score = int(current_score)
        selected_duration_ms = int(current_duration_ms)
        selected_step = 0.0
        movement_detected = False
        probe_count = 0
        duration_mode = "score_notch"
        if cmd in ("u", "d") and int(current_score) == int(SCORE_MIN):
            duration_mode = "duration_notch"
            duration_candidates = list(
                range(
                    int(start_duration_ms),
                    int(current_duration_ms) + 1,
                    int(max(1, DISCOVERY_DURATION_NOTCH_STEP_MS)),
                )
            )
            if not duration_candidates:
                duration_candidates = [int(current_duration_ms)]
            elif duration_candidates[-1] != int(current_duration_ms):
                duration_candidates.append(int(current_duration_ms))
            tested_durations: list[int] = []

            for duration_ms in duration_candidates:
                probe_count += 1
                before_vals = self._collect_metric_samples(
                    metric=metric,
                    samples=int(DISCOVERY_CONFIRM_FRAMES),
                    timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                )
                if len(before_vals) < int(DISCOVERY_CONFIRM_FRAMES):
                    break

                baseline = float(statistics.mean(before_vals))
                sent = self._send_motion(cmd, int(SCORE_MIN), duration_override_ms=int(duration_ms))
                
                # Add pause after command
                total_pause_s = float(CONTROL_SLEEP_S) + (float(ADDITIONAL_PAUSE_MS) / 1000.0)
                print(f"\033[90m{int(total_pause_s * 1000)}ms pause\033[0m")
                time.sleep(total_pause_s)

                after_vals = self._collect_metric_samples(
                    metric=metric,
                    samples=int(DISCOVERY_CONFIRM_FRAMES),
                    timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                )
                if len(after_vals) < int(DISCOVERY_CONFIRM_FRAMES):
                    break

                deltas = [abs(float(v) - float(baseline)) for v in after_vals]
                moved_frames = int(sum(1 for d in deltas if float(d) >= float(movement_threshold)))
                movement_detected = bool(moved_frames >= int(DISCOVERY_CONFIRM_FRAMES))
                step_mm = float(max(deltas) if deltas else 0.0)
                tested_durations.append(int(duration_ms))

                self.log.append(
                    {
                        "type": "hotkey_motion_probe",
                        "run_id": self.run_id,
                        "timestamp": time.time(),
                        "hotkey": hotkey,
                        "cmd": cmd,
                        "metric": metric,
                        "movement_threshold_mm": float(movement_threshold),
                        "probe_index": int(probe_count),
                        "mode": "duration_notch",
                        "score_tested": int(SCORE_MIN),
                        "duration_tested_ms": int(duration_ms),
                        "before_samples": [float(v) for v in before_vals],
                        "after_samples": [float(v) for v in after_vals],
                        "baseline": float(baseline),
                        "moved_frames": int(moved_frames),
                        "confirm_frames_required": int(DISCOVERY_CONFIRM_FRAMES),
                        "movement_detected": bool(movement_detected),
                        "step_mm": float(step_mm),
                        "sent": sent if isinstance(sent, dict) else None,
                    }
                )

                if movement_detected:
                    selected_score = int(SCORE_MIN)
                    selected_duration_ms = int(duration_ms)
                    selected_step = float(step_mm)
                    break
            if not movement_detected:
                selected_score = int(SCORE_MIN)
                if tested_durations:
                    selected_duration_ms = int(min(tested_durations))
                else:
                    # If vision samples are insufficient for lift motion verification,
                    # fall back to the smallest configured pulse for this scan.
                    selected_duration_ms = int(start_duration_ms)
        else:
            for score in range(int(start_score), int(SCORE_MAX) + 1):
                probe_count += 1
                before_vals = self._collect_metric_samples(
                    metric=metric,
                    samples=int(DISCOVERY_CONFIRM_FRAMES),
                    timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                )
                if len(before_vals) < int(DISCOVERY_CONFIRM_FRAMES):
                    break

                baseline = float(statistics.mean(before_vals))
                sent = self._send_motion(cmd, score)
                
                # Add pause after command
                total_pause_s = float(CONTROL_SLEEP_S) + (float(ADDITIONAL_PAUSE_MS) / 1000.0)
                print(f"\033[90m{int(total_pause_s * 1000)}ms pause\033[0m")
                time.sleep(total_pause_s)

                after_vals = self._collect_metric_samples(
                    metric=metric,
                    samples=int(DISCOVERY_CONFIRM_FRAMES),
                    timeout_s=float(DISCOVERY_SAMPLE_TIMEOUT_S),
                )
                if len(after_vals) < int(DISCOVERY_CONFIRM_FRAMES):
                    break

                deltas = [abs(float(v) - float(baseline)) for v in after_vals]
                moved_frames = int(sum(1 for d in deltas if float(d) >= float(movement_threshold)))
                movement_detected = bool(moved_frames >= int(DISCOVERY_CONFIRM_FRAMES))
                step_mm = float(max(deltas) if deltas else 0.0)

                self.log.append(
                    {
                        "type": "hotkey_motion_probe",
                        "run_id": self.run_id,
                        "timestamp": time.time(),
                        "hotkey": hotkey,
                        "cmd": cmd,
                        "metric": metric,
                        "movement_threshold_mm": float(movement_threshold),
                        "probe_index": int(probe_count),
                        "mode": "score_notch",
                        "score_tested": int(score),
                        "duration_tested_ms": None,
                        "before_samples": [float(v) for v in before_vals],
                        "after_samples": [float(v) for v in after_vals],
                        "baseline": float(baseline),
                        "moved_frames": int(moved_frames),
                        "confirm_frames_required": int(DISCOVERY_CONFIRM_FRAMES),
                        "movement_detected": bool(movement_detected),
                        "step_mm": float(step_mm),
                        "sent": sent if isinstance(sent, dict) else None,
                    }
                )

                if movement_detected:
                    selected_score = int(score)
                    selected_duration_ms = int(current_duration_ms)
                    selected_step = float(step_mm)
                    break

        result = {
            "hotkey": hotkey,
            "cmd": cmd,
            "metric": metric,
            "movement_threshold_mm": float(movement_threshold),
            "current_score_before_scan": int(current_score),
            "current_duration_ms_before_scan": int(current_duration_ms),
            "start_score": int(start_score),
            "start_duration_ms": int(start_duration_ms),
            "selected_score": int(selected_score),
            "selected_duration_ms": int(selected_duration_ms),
            "duration_mode": str(duration_mode),
            "movement_detected": bool(movement_detected),
            "step_mm": float(selected_step),
            "probes_recorded": int(probe_count),
        }
        return result

    def _persist_discovery_to_world_model(self, discovered_by_hotkey: dict[str, dict]) -> dict:
        model_path = ROBOT_MODEL_FILE_DEFAULT
        try:
            model = _world_model_load(model_path)
            hotkeys_map = model.get(HOTKEY_SPEED_SCORES_KEY)
            if not isinstance(hotkeys_map, dict):
                hotkeys_map = {}
                model[HOTKEY_SPEED_SCORES_KEY] = hotkeys_map

            applied = {}
            for hotkey, row in discovered_by_hotkey.items():
                if not isinstance(row, dict):
                    continue
                cmd = str(row.get("cmd", "")).strip().lower()
                if cmd not in ("f", "b", "u", "d"):
                    continue
                try:
                    selected_score = _clamp_score(int(row.get("selected_score")))
                except (TypeError, ValueError):
                    continue

                existing_row = hotkeys_map.get(hotkey)
                if not isinstance(existing_row, dict):
                    existing_row = {}
                updated_row = dict(existing_row)
                updated_row["cmd"] = str(cmd)
                updated_row["score"] = int(selected_score)
                selected_duration_ms = row.get("selected_duration_ms")
                duration_mode = str(row.get("duration_mode", "")).strip().lower()
                if selected_duration_ms is not None and duration_mode == "duration_notch":
                    updated_row["duration_ms"] = int(
                        _clamp_duration_ms(selected_duration_ms, int(selected_duration_ms))
                    )
                elif duration_mode != "duration_notch":
                    updated_row.pop("duration_ms", None)
                hotkeys_map[hotkey] = updated_row
                applied_row = {"cmd": str(cmd), "score": int(selected_score)}
                if "duration_ms" in updated_row:
                    applied_row["duration_ms"] = int(updated_row["duration_ms"])
                applied[hotkey] = applied_row

            now_s = float(time.time())
            meta = model.get(HOTKEY_DISCOVERY_META_KEY)
            if not isinstance(meta, dict):
                meta = {}
                model[HOTKEY_DISCOVERY_META_KEY] = meta
            meta["last_updated_epoch_s"] = round(now_s, 3)
            meta["last_updated_iso_utc"] = _format_utc_timestamp(now_s)
            meta["confirm_frames"] = int(DISCOVERY_CONFIRM_FRAMES)
            meta["start_below_score_steps"] = int(DISCOVERY_START_BELOW_SCORE_STEPS)
            meta["hotkeys"] = dict(applied)

            model_path.write_text(json.dumps(model, indent=2) + "\n")
            return {
                "ok": True,
                "model_path": str(model_path),
                "applied": applied,
            }
        except OSError as exc:
            return {
                "ok": False,
                "model_path": str(model_path),
                "error": str(exc),
            }

    def discover_once(self) -> dict[str, dict]:
        global _SESSION_HOTKEY_DISCOVERY

        cached = _SESSION_HOTKEY_DISCOVERY.get("by_hotkey")
        cached_by_hotkey = (
            {str(k): dict(v) for k, v in cached.items() if isinstance(v, dict)}
            if isinstance(cached, dict)
            else {}
        )
        requested = [str(h) for h in self.hotkeys]
        missing_hotkeys = [h for h in requested if h not in cached_by_hotkey]

        if not missing_hotkeys:
            self.discovery_by_hotkey = {
                str(h): dict(cached_by_hotkey[h])
                for h in requested
                if h in cached_by_hotkey
            }
            cached_persist = _SESSION_HOTKEY_DISCOVERY.get("persist_result")
            self.persist_result = dict(cached_persist) if isinstance(cached_persist, dict) else {}
            self.discovery_ran_this_instance = False
            self.log.append(
                {
                    "type": "hotkey_motion_discovery_session_reuse",
                    "run_id": self.run_id,
                    "timestamp": time.time(),
                    "requested_hotkeys": list(requested),
                    "discovery_by_hotkey": dict(self.discovery_by_hotkey),
                },
                force=True,
            )
            return self.discovery_by_hotkey

        targets = self._load_hotkey_targets()
        self.discovery_ran_this_instance = True
        self.log.append(
            {
                "type": "hotkey_motion_discovery_start",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "targets": dict(targets),
                "requested_hotkeys": list(requested),
                "missing_hotkeys": list(missing_hotkeys),
                "confirm_frames": int(DISCOVERY_CONFIRM_FRAMES),
                "start_below_score_steps": int(DISCOVERY_START_BELOW_SCORE_STEPS),
            }
        )

        discovered: dict[str, dict] = dict(cached_by_hotkey)
        for hotkey in self.hotkeys:
            if hotkey in discovered:
                continue
            row = targets.get(hotkey)
            if not isinstance(row, dict):
                continue
            result = self._discover_one_hotkey(row)
            discovered[hotkey] = dict(result)
            duration_msg = ""
            if result.get("selected_duration_ms") is not None:
                duration_msg = f", duration={int(result.get('selected_duration_ms'))}ms"
            
            # Calculate percentage difference and color it
            step_mm = float(result['step_mm'])
            predicted_mm = 1.98  # This should come from curve_prediction
            pct_diff = abs((step_mm - predicted_mm) / predicted_mm * 100) if predicted_mm != 0 else 0
            
            # Color the percentage difference
            if pct_diff >= 80.0:
                pct_color = "\033[92m"  # Green
            else:
                pct_color = "\033[91m"  # Red
            pct_diff_colored = f"{pct_color}{pct_diff:.1f}%\033[0m"
            
            print(
                f"[HOTKEY-MINI] {hotkey.upper()} ({result['cmd']}): "
                f"start {int(result['start_score'])}% -> selected {int(result['selected_score'])}% "
                f"(detected={bool(result['movement_detected'])}, step={float(result['step_mm']):.3f}mm{duration_msg}) "
                f"predicted={predicted_mm:.2f}mm (absolute), pct_diff={pct_diff_colored}, curve_source=curve_prediction"
            )

        self.discovery_by_hotkey = {
            str(h): dict(discovered[h])
            for h in requested
            if h in discovered and isinstance(discovered.get(h), dict)
        }
        self.persist_result = self._persist_discovery_to_world_model(self.discovery_by_hotkey)
        self.log.append(
            {
                "type": "hotkey_motion_discovery_end",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "requested_hotkeys": list(requested),
                "discovery_by_hotkey": dict(self.discovery_by_hotkey),
                "persist_result": dict(self.persist_result),
            },
            force=True,
        )

        done = all(hotkey in discovered for hotkey in TARGET_HOTKEYS_DEFAULT)
        _SESSION_HOTKEY_DISCOVERY = {
            "done": bool(done),
            "by_hotkey": {str(k): dict(v) for k, v in discovered.items() if isinstance(v, dict)},
            "persist_result": dict(self.persist_result),
        }
        return self.discovery_by_hotkey

    def run(self) -> dict:
        started = float(time.time())
        discovered = self.discover_once()
        summary = {
            "ok": True,
            "run_id": int(self.run_id),
            "seconds": float(max(0.0, time.time() - started)),
            "hotkeys": dict(discovered),
            "persist_result": dict(self.persist_result),
            "discovery_ran_this_instance": bool(self.discovery_ran_this_instance),
            "log_path": str(self.log.path),
        }
        return summary

    def close(self):
        self.running = False
        try:
            self.log.flush(force=True)
        except Exception:
            pass
        if self._owns_vision:
            try:
                self.vision.close()
            except Exception:
                pass
        if self._owns_robot:
            try:
                self.robot.close()
            except Exception:
                pass


def run_mini_hotkey_motion_calibration(
    *,
    robot,
    vision,
    hotkeys: tuple[str, ...] = TARGET_HOTKEYS_DEFAULT,
    log_path: Path = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    calibrator = HotkeyMotionCalibrator(
        hotkeys=hotkeys,
        log_path=Path(log_path),
        robot=robot,
        vision=vision,
    )
    started = float(time.time())
    try:
        summary = calibrator.run()
        logger(
            f"[AUTO] Hotkey mini motion discovery complete in {float(summary.get('seconds', 0.0)):.2f}s "
            f"(ran={bool(summary.get('discovery_ran_this_instance'))}, hotkeys={tuple(hotkeys)})."
        )
        persist = summary.get("persist_result")
        if isinstance(persist, dict):
            if bool(persist.get("ok")):
                logger(f"[AUTO] Hotkey mini motion discovery persisted: {persist.get('applied')}")
            else:
                logger(f"[AUTO] Hotkey mini motion discovery persist failed: {persist.get('error')}")
        return summary
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "seconds": float(max(0.0, time.time() - started)),
        }
    finally:
        calibrator.close()


def _parse_hotkeys_csv(value: str) -> tuple[str, ...]:
    if value is None:
        return tuple(TARGET_HOTKEYS_DEFAULT)
    parsed = tuple(str(chunk).strip().lower() for chunk in str(value).split(",") if str(chunk).strip())
    if not parsed:
        return tuple(TARGET_HOTKEYS_DEFAULT)
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Discover smallest reliable hotkey movement scores for r/f/o/k (mapped to f/b/u/d)."
    )
    parser.add_argument("--hotkeys", type=str, default="r,f,o,k")
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    args = parser.parse_args()

    hotkeys = _parse_hotkeys_csv(args.hotkeys)
    calibrator = HotkeyMotionCalibrator(hotkeys=hotkeys, log_path=Path(args.log))
    try:
        summary = calibrator.run()
        print(json.dumps(summary, indent=2))
    except KeyboardInterrupt:
        print("\n[HOTKEY-MINI] Interrupted.")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
