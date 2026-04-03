"""
helper_robot_control.py
----------------
The Translator.
Converts autonomous decisions into the specific command strings
that the Uno firmware expects (e.g., "l.b.40.250,r.b.40.250").
"""
import glob
import os
import serial
import serial.tools.list_ports
import time
import sys
from telemetry_robot import (
    MIN_PWM,
    MAX_PWM,
    MIN_TURN_POWER,
    ACT_DURATION_MS,
    power_to_pwm,
    pwm_to_power,
    turn_pwm_floor,
)


VALID_MOTION_COMMANDS = frozenset({"f", "b", "l", "r", "u", "d"})
UNO_MAX_PERCENT = 100
DEFAULT_SERIAL_PORT = "/dev/ttyCH341USB0"
SERIAL_PORT_ENV_VARS = (
    "LEIA_SERIAL_PORT",
    "ROBOT_SERIAL_PORT",
    "ARDUINO_SERIAL_PORT",
    "SERIAL_PORT",
)
SERIAL_PORT_GLOB_PATTERNS = (
    "/dev/ttyCH341USB*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/cu.usbserial*",
    "/dev/cu.usbmodem*",
)
# The current robot has the left tread polarity inverted relative to the right
# tread, while the mast actuator remains inverted relative to logical up/down.
UNO_MOTION_MAP = {
    "f": (("l", "b"), ("r", "f")),
    "b": (("l", "f"), ("r", "b")),
    "l": (("l", "f"), ("r", "f")),
    "r": (("l", "b"), ("r", "b")),
    "u": (("m", "d"),),
    "d": (("m", "u"),),
}
UNO_STOP_TARGETS = {
    "f": ("l", "r"),
    "b": ("l", "r"),
    "l": ("l", "r"),
    "r": ("l", "r"),
    "u": ("m",),
    "d": ("m",),
}


class Robot:
    def __init__(self):
        self.SERIAL_PORT = DEFAULT_SERIAL_PORT
        self.BAUD_RATE = 115200
        self.ser = None
        self._last_turn_cmd = None
        # Timed serial commands are fire-and-forget; this transport does not queue
        # multiple timed pulses for guaranteed sequential execution.
        self.supports_timed_command_queue = False
        
        # --- PHYSICAL CONSTANTS (Single source: telemetry_robot) ---
        self.MIN_PWM = MIN_PWM
        self.MAX_PWM = MAX_PWM
        self.MIN_TURN_POWER = MIN_TURN_POWER
        self.CMD_DURATION = int(ACT_DURATION_MS) # ms (single source: world_model_robot.json)
        
        self.connect()

    def _serial_env_override(self):
        for key in SERIAL_PORT_ENV_VARS:
            value = str(os.environ.get(key, "")).strip()
            if value:
                return value
        return None

    def _rank_serial_port_info(self, port_info):
        device = str(getattr(port_info, "device", "") or "").strip()
        description = str(getattr(port_info, "description", "") or "").strip()
        manufacturer = str(getattr(port_info, "manufacturer", "") or "").strip()
        hwid = str(getattr(port_info, "hwid", "") or "").strip()
        haystack = " ".join((device, description, manufacturer, hwid)).lower()

        score = 100
        if device == DEFAULT_SERIAL_PORT:
            score -= 80
        if "ch340" in haystack or "ch341" in haystack:
            score -= 60
        if "arduino" in haystack or "uno" in haystack:
            score -= 50
        if "usb serial" in haystack or "wch" in haystack:
            score -= 40
        if "/dev/ttyusb" in device.lower():
            score -= 20
        if "/dev/ttyacm" in device.lower():
            score -= 10
        return score, device

    def _available_serial_ports(self):
        try:
            port_infos = list(serial.tools.list_ports.comports())
        except Exception:
            return []
        return sorted(port_infos, key=self._rank_serial_port_info)

    def _serial_port_candidates(self):
        candidates = []
        seen = set()

        def add_candidate(port):
            port_name = str(port or "").strip()
            if not port_name or port_name in seen:
                return
            seen.add(port_name)
            candidates.append(port_name)

        add_candidate(self._serial_env_override())
        add_candidate(DEFAULT_SERIAL_PORT)

        for port_info in self._available_serial_ports():
            add_candidate(getattr(port_info, "device", None))

        for pattern in SERIAL_PORT_GLOB_PATTERNS:
            for port_name in sorted(glob.glob(pattern)):
                add_candidate(port_name)

        return candidates

    def _print_available_serial_ports(self):
        port_infos = self._available_serial_ports()
        if not port_infos:
            print("[ROBOT] No serial ports were detected on this machine.")
            return

        print("[ROBOT] Available serial ports detected:")
        for port_info in port_infos:
            device = str(getattr(port_info, "device", "") or "").strip()
            description = str(getattr(port_info, "description", "") or "").strip()
            manufacturer = str(getattr(port_info, "manufacturer", "") or "").strip()
            details = [item for item in (description, manufacturer) if item]
            if details:
                print(f"[ROBOT]   {device} ({'; '.join(details)})")
            else:
                print(f"[ROBOT]   {device}")

    def connect(self):
        candidates = self._serial_port_candidates()
        last_error = None

        for port_name in candidates:
            try:
                print(f"[ROBOT] Connecting to Arduino on {port_name}...")
                self.ser = serial.Serial(port_name, self.BAUD_RATE, timeout=1)
                self.SERIAL_PORT = port_name
                time.sleep(2)
                self.ser.reset_input_buffer()
                return
            except Exception as exc:
                last_error = exc
                self.ser = None

        attempted = ", ".join(candidates) if candidates else "none"
        self._print_available_serial_ports()
        print(
            "[ROBOT] Set LEIA_SERIAL_PORT (or ROBOT_SERIAL_PORT) to force a known device path."
        )
        print(
            f"[ROBOT] ERROR: Could not open any candidate serial port ({attempted}). "
            f"Last error: {last_error}"
        )
        sys.exit(1)

    def _send(self, command_str):
        """Internal helper to write the string to Serial"""
        self.last_command = str(command_str).strip()
        if self.ser:
            try:
                # The Arduino expects bytes
                self.ser.write(command_str.encode('utf-8'))
            except Exception as e:
                print(f"[ROBOT] Write Error: {e}")

    def _percent_to_pwm(self, percent):
        try:
            percent_val = int(round(float(percent)))
        except (TypeError, ValueError):
            percent_val = 0
        percent_val = max(0, min(int(UNO_MAX_PERCENT), int(percent_val)))
        return max(0, min(int(self.MAX_PWM), int((percent_val * int(self.MAX_PWM)) / int(UNO_MAX_PERCENT))))

    def _pwm_to_percent(self, pwm):
        try:
            pwm_val = int(round(float(pwm)))
        except (TypeError, ValueError):
            pwm_val = 0
        pwm_val = max(0, min(int(self.MAX_PWM), int(pwm_val)))

        best_percent = 0
        best_diff = None
        for percent in range(int(UNO_MAX_PERCENT) + 1):
            candidate_pwm = self._percent_to_pwm(percent)
            diff = abs(int(candidate_pwm) - int(pwm_val))
            if best_diff is None or diff < best_diff:
                best_percent = int(percent)
                best_diff = int(diff)
        return int(best_percent)

    def _format_uno_token(self, target, action, *, percent=None, duration_ms=None):
        target_key = str(target or "").strip().lower()
        action_key = str(action or "").strip().lower()
        if action_key == "s":
            return f"{target_key}.s"
        percent_val = max(0, min(int(UNO_MAX_PERCENT), int(round(float(percent or 0)))))
        if duration_ms is None:
            return f"{target_key}.{action_key}.{int(percent_val)}"
        return f"{target_key}.{action_key}.{int(percent_val)}.{int(duration_ms)}"

    def _build_motion_payload(self, cmd_char, *, pwm, duration_ms):
        logical_cmd = str(cmd_char or "").strip().lower()
        if logical_cmd not in VALID_MOTION_COMMANDS:
            return None

        try:
            pwm_val = int(round(float(pwm)))
        except (TypeError, ValueError):
            pwm_val = 0
        pwm_val = max(0, min(int(self.MAX_PWM), int(pwm_val)))

        try:
            duration_val = int(round(float(duration_ms)))
        except (TypeError, ValueError):
            duration_val = 0
        duration_val = max(0, int(duration_val))
        timed = duration_val > 0

        if pwm_val <= 0:
            stop_targets = UNO_STOP_TARGETS.get(logical_cmd, ())
            if not stop_targets:
                wire_text = "s"
            else:
                wire_text = ",".join(self._format_uno_token(target, "s") for target in stop_targets)
            return {
                "cmd_sent": logical_cmd,
                "pwm": 0,
                "power": 0.0,
                "percent": 0,
                "duration_ms": int(duration_val),
                "wire_text": str(wire_text),
            }

        actions = UNO_MOTION_MAP.get(logical_cmd)
        if not actions:
            return None

        percent_val = int(self._pwm_to_percent(pwm_val))
        effective_pwm = int(self._percent_to_pwm(percent_val))
        effective_power = pwm_to_power(effective_pwm) or 0.0
        tokens = [
            self._format_uno_token(
                target,
                direction,
                percent=percent_val,
                duration_ms=duration_val if timed else None,
            )
            for target, direction in actions
        ]
        return {
            "cmd_sent": logical_cmd,
            "pwm": int(effective_pwm),
            "power": float(effective_power),
            "percent": int(percent_val),
            "duration_ms": int(duration_val),
            "wire_text": ",".join(tokens),
        }

    def _build_custom_action_payload(self, cmd_char, *, action_specs, duration_ms):
        logical_cmd = str(cmd_char or "").strip().lower()
        if logical_cmd not in VALID_MOTION_COMMANDS:
            return None
        if not isinstance(action_specs, (list, tuple)):
            return None

        try:
            duration_val = int(round(float(duration_ms)))
        except (TypeError, ValueError):
            duration_val = 0
        duration_val = max(0, int(duration_val))
        timed = duration_val > 0

        tokens = []
        normalized_actions = []
        peak_pwm = 0
        peak_percent = 0

        for spec in action_specs:
            if not isinstance(spec, dict):
                continue
            target_key = str(spec.get("target") or "").strip().lower()
            action_key = str(spec.get("action") or "").strip().lower()
            if target_key not in ("l", "r", "m"):
                continue
            if target_key in ("l", "r") and action_key not in ("f", "b", "s"):
                continue
            if target_key == "m" and action_key not in ("u", "d", "s"):
                continue
            if action_key == "s":
                tokens.append(self._format_uno_token(target_key, "s"))
                normalized_actions.append({"target": target_key, "action": "s", "pwm": 0, "percent": 0, "power": 0.0})
                continue

            pwm_raw = spec.get("pwm")
            pwm_val = None
            if pwm_raw is not None:
                try:
                    pwm_val = int(round(float(pwm_raw)))
                except (TypeError, ValueError):
                    pwm_val = None
            if pwm_val is None:
                percent_raw = spec.get("percent")
                try:
                    percent_val = max(0, min(int(UNO_MAX_PERCENT), int(round(float(percent_raw)))))
                except (TypeError, ValueError):
                    percent_val = None
                if percent_val is None:
                    continue
                pwm_val = self._percent_to_pwm(percent_val)
            pwm_val = max(0, min(int(self.MAX_PWM), int(pwm_val)))
            if pwm_val <= 0:
                tokens.append(self._format_uno_token(target_key, "s"))
                normalized_actions.append({"target": target_key, "action": "s", "pwm": 0, "percent": 0, "power": 0.0})
                continue

            percent_val = int(self._pwm_to_percent(pwm_val))
            effective_pwm = int(self._percent_to_pwm(percent_val))
            effective_power = pwm_to_power(effective_pwm) or 0.0
            tokens.append(
                self._format_uno_token(
                    target_key,
                    action_key,
                    percent=percent_val,
                    duration_ms=duration_val if timed else None,
                )
            )
            normalized_actions.append(
                {
                    "target": target_key,
                    "action": action_key,
                    "pwm": int(effective_pwm),
                    "percent": int(percent_val),
                    "power": float(effective_power),
                }
            )
            peak_pwm = max(int(peak_pwm), int(effective_pwm))
            peak_percent = max(int(peak_percent), int(percent_val))

        if not tokens:
            return None

        peak_power = pwm_to_power(peak_pwm) or 0.0
        return {
            "cmd_sent": logical_cmd,
            "pwm": int(peak_pwm),
            "power": float(peak_power),
            "percent": int(peak_percent),
            "duration_ms": int(duration_val),
            "wire_text": ",".join(tokens),
            "actions": normalized_actions,
        }

    def normalize_speed(self, cmd_char, speed):
        try:
            speed = abs(float(speed))
        except (TypeError, ValueError):
            return 0.0, 0
        if cmd_char in ("l", "r") and 0.0 < speed < self.MIN_TURN_POWER:
            speed = float(self.MIN_TURN_POWER)
        if speed < 0.05:
            return 0.0, 0
        pwm = power_to_pwm(speed)
        if pwm is None:
            return 0.0, 0
        pwm = int(pwm)
        if cmd_char in ("l", "r") and pwm > 0:
            pwm = max(int(turn_pwm_floor()), pwm)
        power = pwm_to_power(pwm) or 0.0
        return float(power), int(pwm)

    def send_command(self, cmd_char, speed, duration_ms=None):
        """
        Sends a high-level command using the Uno's per-motor protocol.
        cmd_char: f, b, l, r, u, d
        speed: 0.0 to 1.0
        """
        logical_cmd = str(cmd_char or "").strip().lower()
        speed, pwm = self.normalize_speed(cmd_char, speed)
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        payload = self._build_motion_payload(logical_cmd, pwm=pwm if speed > 0.0 else 0, duration_ms=duration)
        if payload is None:
            return {"cmd_sent": logical_cmd, "pwm": 0, "power": 0.0, "duration_ms": int(duration)}
        self._send(f"{payload['wire_text']}\n")
        return payload

    def send_command_pwm(self, cmd_char, pwm, duration_ms=None):
        """Send a command using a precomputed PWM value from world_model_robot."""
        logical_cmd = str(cmd_char or "").strip().lower()
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        payload = self._build_motion_payload(logical_cmd, pwm=pwm, duration_ms=duration)
        if payload is None:
            return {"cmd_sent": logical_cmd, "pwm": 0, "power": 0.0, "duration_ms": int(duration)}
        self._send(f"{payload['wire_text']}\n")
        return payload

    def send_custom_actions_pwm(self, cmd_char, action_specs, duration_ms=None):
        """Send an explicit per-target Uno action list while preserving the logical cmd label."""
        logical_cmd = str(cmd_char or "").strip().lower()
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        payload = self._build_custom_action_payload(logical_cmd, action_specs=action_specs, duration_ms=duration)
        if payload is None:
            return {"cmd_sent": logical_cmd, "pwm": 0, "power": 0.0, "duration_ms": int(duration)}
        self._send(f"{payload['wire_text']}\n")
        return payload

    def drive(self, speed):
        """Wrapper for BACKWARD COMPATIBILITY with maneuvers.py"""
        if speed > 0:
            self.send_command('f', speed)
        elif speed < 0:
            self.send_command('b', abs(speed))
        else:
            self.stop()

    def spin(self, speed):
        """Wrapper: speed>0 -> Right, speed<0 -> Left"""
        if speed > 0:
            self.send_command('r', speed)
        elif speed < 0:
            self.send_command('l', abs(speed))
        else:
            self.stop() # Or just stop drive?
            
    def set_lift_motor(self, speed):
        """Wrapper: speed>0 -> Up, speed<0 -> Down"""
        if speed > 0:
            self.send_command('u', speed)
        elif speed < 0:
            self.send_command('d', abs(speed))
        else:
            self.send_command('u', 0)

    def stop(self):
        # The Uno firmware supports a global stop token.
        self._last_turn_cmd = None
        self._send("s\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()
