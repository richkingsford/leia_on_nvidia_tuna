import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_robot_control


class _DummySerial:
    def __init__(self):
        self.commands = []

    def write(self, data):
        self.commands.append(data.decode("utf-8"))

    def close(self):
        return None


class TestHelperRobotControlWireMap(unittest.TestCase):
    def test_send_command_pwm_uses_wire_command_map(self):
        wire_map = {
            "f": "b",
            "b": "f",
            "l": "r",
            "r": "l",
            "u": "d",
            "d": "u",
        }
        dummy_serial = _DummySerial()

        with patch.object(helper_robot_control.Robot, "connect", lambda self: None):
            robot = helper_robot_control.Robot()
        robot.ser = dummy_serial

        orig_map = helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP
        try:
            helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP = dict(wire_map)
            for logical_cmd, wire_cmd in wire_map.items():
                result = robot.send_command_pwm(logical_cmd, 40, duration_ms=120)
                self.assertEqual(result["cmd_sent"], wire_cmd)
                self.assertEqual(robot.last_command, f"{wire_cmd} 40 120")
        finally:
            helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP = orig_map

    def test_stop_uses_mapped_zero_commands(self):
        dummy_serial = _DummySerial()
        with patch.object(helper_robot_control.Robot, "connect", lambda self: None):
            robot = helper_robot_control.Robot()
        robot.ser = dummy_serial

        orig_map = helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP
        try:
            helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP = {
                "f": "b",
                "u": "d",
            }
            robot.stop()
        finally:
            helper_robot_control.telemetry_robot_module.ROBOT_WIRE_COMMAND_MAP = orig_map

        self.assertEqual(
            dummy_serial.commands,
            [
                f"b 0 {robot.CMD_DURATION}\n",
                f"d 0 {robot.CMD_DURATION}\n",
            ],
        )


if __name__ == "__main__":
    unittest.main()
