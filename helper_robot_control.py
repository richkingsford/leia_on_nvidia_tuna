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
from telemetry_robot import MIN_PWM, MAX_PWM, MIN_TURN_POWER, COMMAND_REMAP


class Robot:
    def __init__(self):
        self.SERIAL_PORT = '/dev/ttyCH341USB0' 
        self.BAUD_RATE = 115200
        self.ser = None
        
        # --- PHYSICAL CONSTANTS (Single source: telemetry_robot) ---
        self.MIN_PWM = MIN_PWM
        self.MAX_PWM = MAX_PWM
        self.MIN_TURN_POWER = MIN_TURN_POWER
        self.CMD_DURATION = 100 # ms (Keep it running slightly longer for smooth auto-drive)
        
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

    def send_command(self, cmd_char, speed):
        """
        Sends a high-level command: {char} {pwm} {duration}
        cmd_char: f, b, l, r, u, d
        speed: 0.0 to 1.0
        """
        real_hw_cmd = COMMAND_REMAP.get(cmd_char, cmd_char)
        speed, pwm = self.normalize_speed(cmd_char, speed)
        if speed <= 0.0:
            # For safety, sending 0 speed usually stops that action
            self._send(f"{real_hw_cmd} 0 {self.CMD_DURATION}\n")
            return

        # 4. Send
        self._send(f"{real_hw_cmd} {pwm} {self.CMD_DURATION}\n")

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
        self._send(f"f 0 {self.CMD_DURATION}\n")
        self._send(f"u 0 {self.CMD_DURATION}\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()
