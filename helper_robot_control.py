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
from telemetry_robot import MIN_PWM, MAX_PWM, MIN_TURN_POWER, COMMAND_REMAP, ACT_DURATION_MS


class Robot:
    def __init__(self):
        self.SERIAL_PORT = '/dev/ttyCH341USB0' 
        self.BAUD_RATE = 115200
        self.ser = None
        self._last_turn_cmd = None
        
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
        except Exception as e:
            print(f"[ROBOT] ERROR: {e}")
            sys.exit(1)

    def _send(self, command_str):
        """Internal helper to write the string to Serial"""
        if self.ser:
            try:
                # The Arduino expects bytes
                self.ser.write(command_str.encode('utf-8'))
            except Exception as e:
                print(f"[ROBOT] Write Error: {e}")

    def normalize_speed(self, cmd_char, speed):
        speed = abs(speed)
        if cmd_char in ('l', 'r') and 0.0 < speed < self.MIN_TURN_POWER:
            speed = self.MIN_TURN_POWER
        if speed < 0.05:
            return 0.0, 0
        pwm = int(self.MIN_PWM + (self.MAX_PWM - self.MIN_PWM) * speed)
        pwm = min(pwm, 255)
        return speed, pwm

    def send_command(self, cmd_char, speed, duration_ms=None):
        """
        Sends a high-level command: {char} {pwm} {duration}
        cmd_char: f, b, l, r, u, d
        speed: 0.0 to 1.0
        """
        real_hw_cmd = COMMAND_REMAP.get(cmd_char, cmd_char)
        speed, pwm = self.normalize_speed(cmd_char, speed)
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        if speed <= 0.0:
            # For safety, sending 0 speed usually stops that action
            self._send(f"{real_hw_cmd} 0 {duration}\n")
            return

        # 4. Send
        self._send(f"{real_hw_cmd} {pwm} {duration}\n")

    def send_command_pwm(self, cmd_char, pwm, duration_ms=None):
        """Send a command using a precomputed PWM value from world_model_robot."""
        real_hw_cmd = COMMAND_REMAP.get(cmd_char, cmd_char)
        try:
            pwm_val = int(round(pwm))
        except (TypeError, ValueError):
            pwm_val = 0
        pwm_val = max(0, min(255, pwm_val))
        duration = self.CMD_DURATION if duration_ms is None else int(duration_ms)
        if pwm_val <= 0:
            self._send(f"{real_hw_cmd} 0 {duration}\n")
            return
        self._send(f"{real_hw_cmd} {pwm_val} {duration}\n")

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
