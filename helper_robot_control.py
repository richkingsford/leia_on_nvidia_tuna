"""
helper_robot_control.py
----------------
The Translator.
Converts autonomous decisions into the specific command strings 
that your Arduino firmware expects (e.g., "f 200 50").
"""
import serial
import time
import sys
import json
from pathlib import Path
import telemetry_robot as telemetry_robot_module
from telemetry_robot import (
    MIN_PWM,
    MAX_PWM,
    MIN_TURN_POWER,
    ACT_DURATION_MS,
    power_to_pwm,
    pwm_to_power,
    turn_pwm_floor,
)


class Robot:
    def __init__(self):
        self.SERIAL_PORT = '/dev/ttyCH341USB0' 
        self.BAUD_RATE = 115200
        self.ser = None
        self._last_turn_cmd = None
        self._command_remap_cache = None
        self._command_remap_mtime = None
        
        # --- PHYSICAL CONSTANTS (Single source: telemetry_robot) ---
        self.MIN_PWM = MIN_PWM
        self.MAX_PWM = MAX_PWM
        self.MIN_TURN_POWER = MIN_TURN_POWER
        self.CMD_DURATION = int(ACT_DURATION_MS) # ms (single source: world_model_robot.json)
        
        self.connect()

    def connect(self):
        try:
            print(f"[ROBOT] Connecting to Arduino on {self.SERIAL_PORT}...")
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            time.sleep(2) 
            self.ser.reset_input_buffer()
            print("[ROBOT] Connected.")
            print(f"[ROBOT] command_remap={self._command_remap()}")
        except Exception as e:
            print(f"[ROBOT] ERROR: {e}")
            sys.exit(1)

    def _command_remap(self):
        model_path = getattr(telemetry_robot_module, "ROBOT_MODEL_FILE", None)
        if isinstance(model_path, Path) and model_path.exists():
            try:
                mtime = float(model_path.stat().st_mtime)
            except OSError:
                mtime = None
            if (
                mtime is not None
                and self._command_remap_cache is not None
                and self._command_remap_mtime is not None
                and abs(mtime - float(self._command_remap_mtime)) < 1e-6
            ):
                return dict(self._command_remap_cache)
            mapping = {}
            try:
                raw = json.loads(model_path.read_text())
            except (OSError, json.JSONDecodeError):
                raw = {}
            if isinstance(raw, dict):
                candidate = raw.get("command_remap")
                if isinstance(candidate, dict):
                    for key, value in candidate.items():
                        if key is None or value is None:
                            continue
                        mapping[str(key)] = str(value)
            self._command_remap_cache = dict(mapping)
            self._command_remap_mtime = mtime
            try:
                telemetry_robot_module.COMMAND_REMAP = dict(mapping)
            except Exception:
                pass
            return dict(mapping)

        mapping = getattr(telemetry_robot_module, "COMMAND_REMAP", None)
        if isinstance(mapping, dict):
            return dict(mapping)
        return {}

    def _send(self, command_str):
        """Internal helper to write the string to Serial"""
        self.last_command = str(command_str).strip()
        if self.ser:
            try:
                # The Arduino expects bytes
                self.ser.write(command_str.encode('utf-8'))
            except Exception as e:
                print(f"[ROBOT] Write Error: {e}")

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
        Sends a high-level command: {char} {pwm} {duration}
        cmd_char: f, b, l, r, u, d
        speed: 0.0 to 1.0
        """
        real_hw_cmd = self._command_remap().get(cmd_char, cmd_char)
        speed, pwm = self.normalize_speed(cmd_char, speed)
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        if speed <= 0.0:
            # For safety, sending 0 speed usually stops that action
            self._send(f"{real_hw_cmd} 0 {duration}\n")
            return {"cmd_sent": real_hw_cmd, "pwm": 0, "duration_ms": int(duration)}

        # 4. Send
        self._send(f"{real_hw_cmd} {pwm} {duration}\n")
        return {"cmd_sent": real_hw_cmd, "pwm": int(pwm), "duration_ms": int(duration)}

    def send_command_pwm(self, cmd_char, pwm, duration_ms=None):
        """Send a command using a precomputed PWM value from world_model_robot."""
        real_hw_cmd = self._command_remap().get(cmd_char, cmd_char)
        try:
            pwm_val = int(round(pwm))
        except (TypeError, ValueError):
            pwm_val = 0
        pwm_val = max(0, min(int(self.MAX_PWM), pwm_val))
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        if pwm_val <= 0:
            self._send(f"{real_hw_cmd} 0 {duration}\n")
            return {"cmd_sent": real_hw_cmd, "pwm": 0, "duration_ms": int(duration)}
        self._send(f"{real_hw_cmd} {pwm_val} {duration}\n")
        return {"cmd_sent": real_hw_cmd, "pwm": int(pwm_val), "duration_ms": int(duration)}

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
        # Stop everything. 'f 0' usually stops the base?
        # Let's send a stop for drive and lift to be sure.
        self._last_turn_cmd = None
        self._send(f"f 0 {self.CMD_DURATION}\n")
        self._send(f"u 0 {self.CMD_DURATION}\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()
