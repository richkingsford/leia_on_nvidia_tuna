"""Reusable manual hotkey controls for Leia.

The movement mapping comes from ``world_model_robot.json`` via
``telemetry_robot.HOTKEY_SPEED_SCORES``. This helper owns keyboard input and
manual pulse dispatch so runners such as ``main2.py`` can stay thin.
"""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from typing import Callable

import telemetry_robot as telemetry_robot_module
from helper_manual_config import load_manual_training_config
from helper_manual_drive_assist import (
    build_manual_drive_assist_plan,
    execute_manual_drive_assist_plan,
    format_manual_drive_assist_line,
)
from helper_manual_turn_arc_assist import (
    build_manual_turn_arc_plan,
    execute_manual_turn_arc_plan,
    format_manual_turn_arc_assist_line,
)
from telemetry_process import send_robot_command, send_robot_command_pwm
from telemetry_robot import HOTKEY_SPEED_SCORES, MotionEvent, StepState, WorldModel


DEFAULT_COMMAND_RATE_HZ = 30.0
DEFAULT_HEARTBEAT_TIMEOUT = 0.3
HOTKEY_EASE_IN_OUT_KEYS = frozenset({"t", "g", "z", "c"})
STOP_KEYS = frozenset({" ", "\x1b"})
QUIT_KEY = "Q"


def _coerce_positive_int(value, default=None):
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    if result <= 0:
        return default
    return int(result)


def hotkey_uses_ease_in_out(hotkey) -> bool:
    return str(hotkey or "").strip().lower() in HOTKEY_EASE_IN_OUT_KEYS


def hotkey_action_from_key(key, hotkeys=None) -> dict | None:
    """Return the model-backed movement action for a keyboard key."""

    key_norm = str(key or "").strip().lower()
    if not key_norm:
        return None
    rows = hotkeys if isinstance(hotkeys, dict) else HOTKEY_SPEED_SCORES
    entry = rows.get(key_norm) if isinstance(rows, dict) else None
    if not isinstance(entry, dict):
        return None

    cmd = str(entry.get("cmd", "")).strip().lower()
    if cmd not in {"f", "b", "l", "r", "u", "d"}:
        return None
    try:
        score = telemetry_robot_module.normalize_speed_score(entry.get("score"))
    except Exception:
        return None

    return {
        "hotkey": key_norm,
        "cmd": cmd,
        "score": int(score),
        "duration_ms": _coerce_positive_int(entry.get("duration_ms")),
        "pwm_override": _coerce_positive_int(entry.get("pwm")),
    }


def _score_text(key, hotkeys):
    action = hotkey_action_from_key(key, hotkeys=hotkeys)
    if not action:
        return None
    return f"{str(key).upper()} {int(action['score'])}%"


def _paired_score_text(left_key, right_key, hotkeys):
    left = hotkey_action_from_key(left_key, hotkeys=hotkeys)
    right = hotkey_action_from_key(right_key, hotkeys=hotkeys)
    if not left or not right:
        return None
    if int(left["score"]) == int(right["score"]):
        return f"{left_key.upper()}/{right_key.upper()} {int(left['score'])}%"
    return f"{left_key.upper()} {int(left['score'])}%, {right_key.upper()} {int(right['score'])}%"


def format_hotkey_help(hotkeys=None) -> str:
    rows = hotkeys if isinstance(hotkeys, dict) else HOTKEY_SPEED_SCORES
    drive_parts = [
        _paired_score_text("w", "s", rows),
        _paired_score_text("r", "f", rows),
        _paired_score_text("t", "g", rows),
    ]
    turn_parts = [
        _paired_score_text("q", "e", rows),
        _paired_score_text("a", "d", rows),
        _paired_score_text("z", "c", rows),
    ]
    lift_up_parts = [_score_text(key, rows) for key in ("o", "u", "p")]
    lift_down_parts = [_score_text(key, rows) for key in ("k", "l")]
    sections = []
    for label, parts in (
        ("Drive", drive_parts),
        ("Turn", turn_parts),
        ("Lift up", lift_up_parts),
        ("Lift down", lift_down_parts),
    ):
        clean = [part for part in parts if part]
        if clean:
            sections.append(f"{label}: {', '.join(clean)}")
    suffix = "Space/Esc stop, uppercase Q quits."
    if not sections:
        return f"[CTRL] No movement hotkeys loaded. {suffix}"
    return "[CTRL] " + ". ".join(sections) + f". {suffix}"


def _cmd_to_motion_action_type(cmd):
    return {
        "f": "forward",
        "b": "backward",
        "l": "left_turn",
        "r": "right_turn",
        "u": "mast_up",
        "d": "mast_down",
    }.get(str(cmd or "").strip().lower(), "unknown")


def _cmd_action_label(cmd):
    return {
        "f": "move forward",
        "b": "move backward",
        "l": "turn left",
        "r": "turn right",
        "u": "lift mast",
        "d": "lower mast",
    }.get(str(cmd or "").strip().lower(), "move")


class ManualHotkeyController:
    """One-shot keyboard hotkey controller for Leia motion."""

    def __init__(
        self,
        *,
        robot,
        world: WorldModel | None = None,
        step_state=None,
        hotkeys=None,
        command_rate_hz: float | None = None,
        heartbeat_timeout: float | None = None,
        logger: Callable[[str], None] | None = None,
        stop_callback: Callable[[], None] | None = None,
        enable_terminal: bool = True,
    ):
        cfg = load_manual_training_config()
        self.robot = robot
        self.world = world if world is not None else WorldModel()
        if step_state is not None:
            self.world.step_state = step_state
        elif getattr(self.world, "step_state", None) is None:
            self.world.step_state = StepState.ALIGN_BRICK
        self.hotkeys = hotkeys if isinstance(hotkeys, dict) else HOTKEY_SPEED_SCORES
        try:
            rate = float(command_rate_hz if command_rate_hz is not None else cfg.get("command_rate_hz"))
        except (TypeError, ValueError):
            rate = DEFAULT_COMMAND_RATE_HZ
        self.command_rate_hz = max(1.0, float(rate))
        try:
            timeout = float(
                heartbeat_timeout if heartbeat_timeout is not None else cfg.get("heartbeat_timeout")
            )
        except (TypeError, ValueError):
            timeout = DEFAULT_HEARTBEAT_TIMEOUT
        self.heartbeat_timeout = max(0.05, float(timeout))
        self.logger = logger if callable(logger) else print
        self.stop_callback = stop_callback
        self.enable_terminal = bool(enable_terminal)

        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._command_thread: threading.Thread | None = None
        self._terminal_enabled = False

        self.active_command = None
        self.active_hotkey = None
        self.active_speed = 0.0
        self.active_speed_score = None
        self.active_duration_ms = None
        self.active_pwm_override = None
        self.last_key_time = 0.0
        self.last_key_line = ""
        self.last_action_line = ""
        self.last_error = ""

    def start(self) -> None:
        if self._command_thread is None or not self._command_thread.is_alive():
            self._command_thread = threading.Thread(
                target=self._command_loop,
                name="leia-manual-hotkey-command",
                daemon=True,
            )
            self._command_thread.start()

        self._terminal_enabled = bool(self.enable_terminal and sys.stdin is not None and sys.stdin.isatty())
        if self._terminal_enabled and (self._keyboard_thread is None or not self._keyboard_thread.is_alive()):
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_loop,
                name="leia-manual-hotkey-keyboard",
                daemon=True,
            )
            self._keyboard_thread.start()
            self.logger("[CTRL] Terminal hotkeys armed. Press ? for help.")
            self.logger(format_hotkey_help(self.hotkeys))
        elif self.enable_terminal:
            self.logger("[CTRL] Terminal hotkeys disabled because stdin is not a TTY.")

    def stop(self, *, stop_robot: bool = True) -> None:
        self._stop_event.set()
        with self.lock:
            self._clear_active_locked()
        if stop_robot and self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass
        thread = self._command_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def status_lines(self) -> list[str]:
        with self.lock:
            active = self.active_command
            active_hotkey = self.active_hotkey
            active_score = self.active_speed_score
            last_key = str(self.last_key_line or "").strip()
            last_action = str(self.last_action_line or "").strip()
            last_error = str(self.last_error or "").strip()
        mode = "ready" if self.is_running() else "stopped"
        input_mode = "terminal" if self._terminal_enabled else "programmatic"
        lines = [format_hotkey_help(self.hotkeys), f"Manual controls: {mode} ({input_mode} input)"]
        if active and active_score is not None:
            lines.append(
                f"Active hotkey: {str(active_hotkey or '?').upper()} -> "
                f"{_cmd_action_label(active)} {int(active_score)}%"
            )
        if last_key:
            lines.append(last_key)
        if last_action:
            lines.append(last_action)
        if last_error:
            lines.append(f"[CTRL ERROR] {last_error}")
        return lines

    def queue_key(self, ch) -> dict | str | None:
        if ch == QUIT_KEY:
            self._request_shutdown("uppercase Q")
            return "quit"
        if ch in STOP_KEYS:
            self._stop_motion("operator stop")
            return "stop"
        action = hotkey_action_from_key(ch, hotkeys=self.hotkeys)
        if not action:
            if str(ch or "") == "?":
                self.logger(format_hotkey_help(self.hotkeys))
            return None
        with self.lock:
            self.last_key_time = time.time()
            self.active_command = action["cmd"]
            self.active_hotkey = action["hotkey"]
            self.active_speed_score = int(action["score"])
            self.active_duration_ms = action.get("duration_ms")
            self.active_pwm_override = action.get("pwm_override")
            self.active_speed = 0.0
            self.last_key_line = (
                f"[CTRL] Key {str(action['hotkey']).upper()} received -> "
                f"{_cmd_action_label(action['cmd'])} {int(action['score'])}%."
            )
            self.last_error = ""
        self.logger(self.last_key_line)
        return action

    def process_once(self, *, now: float | None = None):
        now_val = time.time() if now is None else float(now)
        with self.lock:
            if now_val - float(self.last_key_time or 0.0) > float(self.heartbeat_timeout):
                self._clear_active_locked()
            cmd = self.active_command
            hotkey = self.active_hotkey
            score = self.active_speed_score
            duration_override_ms = self.active_duration_ms
            pwm_override = self.active_pwm_override

        if not cmd or score is None:
            return None

        try:
            quantized_speed, score_used = telemetry_robot_module.quantize_speed(cmd, score=score)
            speed = self._normalize_robot_speed(cmd, quantized_speed)
            with self.lock:
                self.active_speed = float(speed)
                self.active_speed_score = int(score_used) if score_used is not None else None
            if score_used is None:
                return None
            send_result = self._send_hotkey_action(
                cmd=cmd,
                hotkey=hotkey,
                speed=float(speed),
                score_used=int(score_used),
                duration_override_ms=duration_override_ms,
                pwm_override=pwm_override,
            )
            self._apply_motion_telemetry(
                cmd=cmd,
                speed=float(speed),
                score=int(score_used),
                send_result=send_result,
            )
            action_line = self._format_hotkey_operator_log(
                hotkey=hotkey,
                cmd=cmd,
                score=int(score_used),
                send_result=send_result,
            )
            with self.lock:
                self.last_action_line = action_line
                self.last_error = ""
            self.logger(action_line)
            return send_result
        except Exception as exc:
            self._handle_send_error(exc)
            return None
        finally:
            with self.lock:
                if self.active_command == cmd and self.active_hotkey == hotkey:
                    self._clear_active_locked()

    def _keyboard_loop(self) -> None:
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
        while not self._stop_event.is_set():
            try:
                ch = self._getch()
            except Exception as exc:
                with self.lock:
                    self.last_error = f"keyboard input stopped ({exc})"
                self.logger(f"[CTRL ERROR] {self.last_error}")
                return
            if not ch:
                continue
            self.queue_key(ch)

    def _command_loop(self) -> None:
        dt = 1.0 / max(float(self.command_rate_hz), 1.0)
        while not self._stop_event.is_set():
            loop_start = time.time()
            self.process_once(now=loop_start)
            elapsed = time.time() - loop_start
            sleep_s = max(0.0, float(dt) - float(elapsed))
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _clear_active_locked(self) -> None:
        self.active_command = None
        self.active_hotkey = None
        self.active_speed = 0.0
        self.active_speed_score = None
        self.active_duration_ms = None
        self.active_pwm_override = None

    def _normalize_robot_speed(self, cmd, speed) -> float:
        if self.robot is not None and hasattr(self.robot, "normalize_speed"):
            try:
                normalized, _pwm = self.robot.normalize_speed(cmd, speed)
                return float(normalized or 0.0)
            except Exception:
                pass
        try:
            return max(0.0, min(1.0, float(speed or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def _send_hotkey_action(
        self,
        *,
        cmd,
        hotkey,
        speed,
        score_used,
        duration_override_ms,
        pwm_override,
    ):
        send_result = None
        if hotkey and cmd in ("l", "r"):
            send_result = self._send_turn_hotkey_with_arc(
                cmd=cmd,
                score_used=score_used,
                hotkey=hotkey,
                duration_override_ms=duration_override_ms,
                pwm_override=pwm_override,
            )
        if send_result is None and hotkey and cmd in ("f", "b"):
            send_result = self._send_drive_hotkey_with_assist(
                cmd=cmd,
                score_used=score_used,
                hotkey=hotkey,
                duration_override_ms=duration_override_ms,
                pwm_override=pwm_override,
            )
        if send_result is None and pwm_override is not None:
            send_result = self._send_pwm_override(
                cmd=cmd,
                score_used=score_used,
                speed=speed,
                duration_override_ms=duration_override_ms,
                pwm_override=pwm_override,
            )
        if send_result is None:
            send_result = send_robot_command(
                self.robot,
                self.world,
                self.world.step_state,
                cmd,
                speed,
                speed_score=score_used,
                duration_override_ms=duration_override_ms,
                ease_in_out_enabled=hotkey_uses_ease_in_out(hotkey),
            )
        return send_result

    def _send_drive_hotkey_with_assist(
        self,
        *,
        cmd,
        score_used,
        hotkey,
        duration_override_ms,
        pwm_override,
    ):
        plan = build_manual_drive_assist_plan(
            hotkey=hotkey,
            cmd=cmd,
            score=score_used,
            hold_duration_ms=duration_override_ms,
            pwm_override=pwm_override,
        )
        if not isinstance(plan, dict):
            return None
        line = format_manual_drive_assist_line(plan)
        if line:
            self.logger(line)
        return execute_manual_drive_assist_plan(
            robot=self.robot,
            world=self.world,
            step_state=self.world.step_state,
            hotkey=hotkey,
            cmd=cmd,
            score=score_used,
            hold_duration_ms=plan.get("hold_duration_ms"),
            pwm_override=pwm_override,
            send_robot_command_fn=send_robot_command,
            send_robot_command_pwm_fn=send_robot_command_pwm,
            sleep_fn=time.sleep,
        )

    def _send_turn_hotkey_with_arc(
        self,
        *,
        cmd,
        score_used,
        hotkey,
        duration_override_ms,
        pwm_override,
    ):
        plan = build_manual_turn_arc_plan(
            hotkey=hotkey,
            cmd=cmd,
            score=score_used,
            hold_duration_ms=duration_override_ms,
            pwm_override=pwm_override,
        )
        if not isinstance(plan, dict):
            return None
        line = format_manual_turn_arc_assist_line(plan)
        if line:
            self.logger(line)
        return execute_manual_turn_arc_plan(
            robot=self.robot,
            hotkey=hotkey,
            cmd=cmd,
            score=score_used,
            hold_duration_ms=plan.get("duration_ms"),
            pwm_override=pwm_override,
        )

    def _send_pwm_override(
        self,
        *,
        cmd,
        score_used,
        speed,
        duration_override_ms,
        pwm_override,
    ):
        pwm_val = telemetry_robot_module.clamp_pwm(int(pwm_override))
        if pwm_val <= 0:
            return None
        power_for_pwm = telemetry_robot_module.pwm_to_power(pwm_val)
        if power_for_pwm is None:
            power_for_pwm = float(speed or 0.0)
        duration_ms = _coerce_positive_int(duration_override_ms)
        if duration_ms is None:
            try:
                _power, _pwm, _score, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                    cmd,
                    score_used,
                )
                duration_ms = max(1, int(duration_ms))
            except Exception:
                duration_ms = int(getattr(telemetry_robot_module, "ACT_DURATION_MS", 250) or 250)
        return send_robot_command_pwm(
            self.robot,
            self.world,
            self.world.step_state,
            cmd,
            float(power_for_pwm or 0.0),
            int(pwm_val),
            int(duration_ms),
            speed_score=score_used,
            auto_mode=False,
            ease_in_out_enabled=hotkey_uses_ease_in_out(self.active_hotkey),
        )

    def _apply_motion_telemetry(self, *, cmd, speed, score, send_result=None) -> None:
        action_type = _cmd_to_motion_action_type(cmd)
        if action_type == "unknown":
            return
        result = send_result if isinstance(send_result, dict) else {}
        try:
            power_used = float(result.get("power", speed))
        except (TypeError, ValueError):
            power_used = float(speed or 0.0)
        power_used = max(0.0, min(1.0, float(power_used)))
        duration_ms = _coerce_positive_int(result.get("duration_ms"))
        if duration_ms is None:
            duration_ms = _coerce_positive_int(self.active_duration_ms, default=100)
        evt = MotionEvent(
            action_type,
            int(round(power_used * 255.0)),
            max(1, int(duration_ms)),
            speed_score=score,
        )
        if hasattr(self.world, "update_from_motion"):
            self.world.update_from_motion(evt)

    def _format_hotkey_operator_log(self, *, hotkey, cmd, score, send_result=None) -> str:
        result = send_result if isinstance(send_result, dict) else {}
        pwm = _coerce_positive_int(result.get("pwm"), default=0)
        duration_ms = _coerce_positive_int(result.get("duration_ms"), default=0)
        try:
            power = float(result.get("power", 0.0) or 0.0)
        except (TypeError, ValueError):
            power = 0.0
        assist = ""
        if isinstance(result.get("manual_turn_arc_assist"), dict):
            assist = "; turn arc"
        elif isinstance(result.get("manual_drive_assist"), dict):
            assist = "; drive assist"
        return (
            f"[HOTKEY] {str(hotkey or '?').upper()} -> {_cmd_action_label(cmd)} "
            f"({str(cmd or '').upper()} {int(score)}%, pwm={int(pwm)}, "
            f"pwr={float(power):.3f}, t={int(duration_ms)}ms{assist})"
        )

    def _handle_send_error(self, exc: Exception) -> None:
        msg = str(exc).strip() or exc.__class__.__name__
        with self.lock:
            self.last_error = msg
            self._clear_active_locked()
        try:
            if self.robot is not None:
                self.robot.stop()
        except Exception:
            pass
        self.logger(f"[CTRL ERROR] {msg}")

    def _stop_motion(self, reason: str) -> None:
        with self.lock:
            self.last_key_time = time.time()
            self._clear_active_locked()
            self.last_action_line = f"[CTRL] Stop ({reason})."
        try:
            if self.robot is not None:
                self.robot.stop()
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
        self.logger(self.last_action_line)

    def _request_shutdown(self, reason: str) -> None:
        with self.lock:
            self._clear_active_locked()
            self.last_action_line = f"[CTRL] Shutdown requested ({reason})."
        self._stop_event.set()
        try:
            if self.robot is not None:
                self.robot.stop()
        except Exception:
            pass
        self.logger(self.last_action_line)
        if callable(self.stop_callback):
            try:
                self.stop_callback()
            except Exception:
                pass
