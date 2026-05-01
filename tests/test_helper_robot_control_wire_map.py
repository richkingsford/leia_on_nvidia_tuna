import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_robot_control


class _DummySerial:
    def __init__(self):
        self.commands = []
        self.reset_calls = 0

    def write(self, data):
        self.commands.append(data.decode("utf-8"))

    def reset_input_buffer(self):
        self.reset_calls += 1

    def close(self):
        return None


class TestHelperRobotControlWireMap(unittest.TestCase):
    def _make_robot(self):
        dummy_serial = _DummySerial()
        with patch.object(helper_robot_control.Robot, "connect", lambda self: None):
            robot = helper_robot_control.Robot()
        robot.ser = dummy_serial
        return robot, dummy_serial

    def test_send_command_pwm_serializes_existing_moves_for_uno_protocol(self):
        robot, _dummy_serial = self._make_robot()
        for logical_cmd in ("f", "b", "l", "r", "u", "d"):
            _, min_duration = robot._min_floor_for_cmd(logical_cmd)
            min_duration = int(min_duration or 0)
            result = robot.send_command_pwm(logical_cmd, 40, duration_ms=120)
            self.assertEqual(result["cmd_sent"], logical_cmd)
            self.assertGreaterEqual(int(result["duration_ms"]), min_duration)
            self.assertGreaterEqual(int(result["pwm"]), int(robot._min_floor_for_cmd(logical_cmd)[0] or 0))
            self.assertEqual(robot.last_command, result["wire_text"])

    def test_send_command_pwm_quantizes_requested_pwm_to_uno_percent_steps(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_command_pwm("f", 103, duration_ms=255)
        min_pwm, min_duration = robot._min_floor_for_cmd("f")

        self.assertEqual(result["cmd_sent"], "f")
        self.assertGreaterEqual(int(result["pwm"]), int(min_pwm))
        self.assertGreaterEqual(int(result["duration_ms"]), int(min_duration))
        self.assertEqual(robot.last_command, result["wire_text"])

    def test_zero_pwm_stops_only_the_targeted_actuators(self):
        robot, _dummy_serial = self._make_robot()

        drive_stop = robot.send_command_pwm("f", 0, duration_ms=120)
        mast_stop = robot.send_command_pwm("u", 0, duration_ms=120)

        self.assertEqual(drive_stop["wire_text"], "l.s,r.s")
        self.assertEqual(mast_stop["wire_text"], "m.s")
        self.assertEqual(robot.last_command, "m.s")

    def test_send_custom_actions_pwm_serializes_mixed_tread_commands(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_custom_actions_pwm(
            "r",
            [
                {"target": "l", "action": "b", "pwm": 255},
                {"target": "r", "action": "f", "pwm": 128},
            ],
            duration_ms=120,
        )

        self.assertEqual(result["cmd_sent"], "r")
        self.assertIn("l.b.100.255", result["wire_text"])
        self.assertIn("r.f.50.255", result["wire_text"])
        self.assertEqual(result["pwm"], 255)
        self.assertEqual(robot.last_command, result["wire_text"])

    def test_send_custom_actions_pwm_keeps_zero_inner_tread_in_timed_token_shape(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_custom_actions_pwm(
            "l",
            [
                {"target": "l", "action": "b", "pwm": 0},
                {"target": "r", "action": "f", "pwm": 255},
            ],
            duration_ms=200,
        )

        self.assertEqual(result["wire_text"], "l.s,r.f.100.255")
        self.assertEqual(robot.last_command, "l.s,r.f.100.255")

    def test_send_custom_actions_pwm_converts_zero_percent_directional_steps_to_stop_tokens(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_custom_actions_pwm(
            "r",
            [
                {"target": "l", "action": "b", "pwm": 255},
                {"target": "r", "action": "f", "pwm": 0},
            ],
            duration_ms=255,
        )

        self.assertEqual(result["wire_text"], "l.b.100.255,r.s")
        self.assertEqual(robot.last_command, "l.b.100.255,r.s")

    def test_send_custom_actions_pwm_preserves_explicit_stop_actions(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_custom_actions_pwm(
            "l",
            [
                {"target": "l", "action": "s", "pwm": 0},
                {"target": "r", "action": "f", "pwm": 255},
            ],
            duration_ms=200,
        )

        self.assertEqual(result["wire_text"], "l.s,r.f.100.255")
        self.assertEqual(robot.last_command, "l.s,r.f.100.255")

    def test_send_command_pwm_clamps_subfloor_duration_instead_of_sending_invalid(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_command_pwm("b", 132, duration_ms=250)
        self.assertGreaterEqual(int(result["duration_ms"]), 255)
        self.assertGreaterEqual(int(result["pwm"]), int(robot._min_floor_for_cmd("b")[0] or 0))

    def test_send_custom_actions_pwm_clamps_subfloor_duration_instead_of_sending_invalid(self):
        robot, _dummy_serial = self._make_robot()

        result = robot.send_custom_actions_pwm(
            "b",
            [
                {"target": "l", "action": "f", "pwm": 132},
                {"target": "r", "action": "b", "pwm": 132},
            ],
            duration_ms=250,
        )
        self.assertIn(".255", result["wire_text"])
        for action in result.get("actions") or []:
            if not isinstance(action, dict):
                continue
            act = str(action.get("action") or "").strip().lower()
            if act in helper_robot_control.VALID_MOTION_COMMANDS and act != "s":
                min_pwm, _ = robot._min_floor_for_cmd(act)
                self.assertGreaterEqual(int(action.get("pwm") or 0), int(min_pwm or 0))

    def test_floor_error_message_includes_sender_curve_actual_and_curve_values(self):
        robot, _dummy_serial = self._make_robot()

        with self.assertRaises(RuntimeError) as ctx:
            robot._validate_minimum_act("b", 132, 250, source_fn="unit_test_sender")

        text = str(ctx.exception)
        self.assertIn("sender=unit_test_sender", text)
        self.assertIn("curve=score_power_pwm_drive", text)
        self.assertIn("actual[pwm=132,pwr=", text)
        self.assertIn("curve_1pct[pwm=", text)

    def test_stop_uses_global_stop_token(self):
        robot, dummy_serial = self._make_robot()

        robot.stop()

        self.assertEqual(dummy_serial.commands, ["s\n"])
        self.assertEqual(robot.last_command, "s")

    def test_connect_prefers_env_override_when_present(self):
        serial_device = _DummySerial()
        with patch.dict("os.environ", {"LEIA_SERIAL_PORT": "/dev/test-robot-port"}, clear=False):
            with patch.object(helper_robot_control.serial.tools.list_ports, "comports", return_value=[]):
                with patch.object(helper_robot_control.glob, "glob", return_value=[]):
                    with patch.object(helper_robot_control.serial, "Serial", return_value=serial_device) as serial_ctor:
                        with patch.object(helper_robot_control.time, "sleep", return_value=None):
                            robot = helper_robot_control.Robot()

        serial_ctor.assert_called_once_with("/dev/test-robot-port", robot.BAUD_RATE, timeout=1)
        self.assertEqual(robot.SERIAL_PORT, "/dev/test-robot-port")
        self.assertIs(robot.ser, serial_device)
        self.assertEqual(serial_device.reset_calls, 1)

    def test_connect_prefers_constructor_serial_override_when_present(self):
        serial_device = _DummySerial()
        with patch.dict("os.environ", {"LEIA_SERIAL_PORT": "/dev/env-port"}, clear=False):
            with patch.object(helper_robot_control.serial.tools.list_ports, "comports", return_value=[]):
                with patch.object(helper_robot_control.glob, "glob", return_value=[]):
                    with patch.object(helper_robot_control.serial, "Serial", return_value=serial_device) as serial_ctor:
                        with patch.object(helper_robot_control.time, "sleep", return_value=None):
                            robot = helper_robot_control.Robot(serial_port="/dev/explicit-port")

        serial_ctor.assert_called_once_with("/dev/explicit-port", robot.BAUD_RATE, timeout=1)
        self.assertEqual(robot.SERIAL_PORT, "/dev/explicit-port")
        self.assertIs(robot.ser, serial_device)
        self.assertEqual(serial_device.reset_calls, 1)

    def test_connect_retries_after_permission_repair(self):
        serial_device = _DummySerial()

        with patch.dict("os.environ", {}, clear=False):
            with patch.object(helper_robot_control.serial.tools.list_ports, "comports", return_value=[]):
                with patch.object(helper_robot_control.glob, "glob", return_value=[]):
                    with patch.object(
                        helper_robot_control.serial,
                        "Serial",
                        side_effect=[PermissionError("permission denied"), serial_device],
                    ) as serial_ctor:
                        with patch.object(helper_robot_control.Robot, "_try_relax_serial_permissions", return_value=True):
                            with patch.object(helper_robot_control.time, "sleep", return_value=None):
                                robot = helper_robot_control.Robot(
                                    exit_on_failure=False,
                                    serial_port="/dev/ttyCH341USB0",
                                )

        self.assertEqual(serial_ctor.call_count, 2)
        self.assertEqual(robot.SERIAL_PORT, "/dev/ttyCH341USB0")
        self.assertIs(robot.ser, serial_device)
        self.assertEqual(serial_device.reset_calls, 1)

    def test_connect_falls_back_to_detected_port_when_default_path_is_missing(self):
        serial_device = _DummySerial()
        detected_port = SimpleNamespace(
            device="/dev/ttyUSB0",
            description="USB Serial",
            manufacturer="wch.cn",
            hwid="USB VID:PID=1A86:7523",
        )

        def serial_side_effect(port, baud_rate, timeout=1):
            if port == helper_robot_control.DEFAULT_SERIAL_PORT:
                raise FileNotFoundError(port)
            if port == "/dev/ttyUSB0":
                return serial_device
            raise AssertionError(f"unexpected port attempt: {port}")

        with patch.dict("os.environ", {}, clear=False):
            with patch.object(helper_robot_control.serial.tools.list_ports, "comports", return_value=[detected_port]):
                with patch.object(helper_robot_control.glob, "glob", return_value=[]):
                    with patch.object(helper_robot_control.os.path, "exists", return_value=False):
                        with patch.object(helper_robot_control.serial, "Serial", side_effect=serial_side_effect) as serial_ctor:
                            with patch.object(helper_robot_control.time, "sleep", return_value=None):
                                robot = helper_robot_control.Robot()

        self.assertEqual(serial_ctor.call_args_list[0].args[0], helper_robot_control.DEFAULT_SERIAL_PORT)
        self.assertEqual(serial_ctor.call_args_list[1].args[0], "/dev/ttyUSB0")
        self.assertEqual(robot.SERIAL_PORT, "/dev/ttyUSB0")
        self.assertIs(robot.ser, serial_device)
        self.assertEqual(serial_device.reset_calls, 1)

    def test_connect_error_reports_detected_ports(self):
        detected_port = SimpleNamespace(
            device="/dev/ttyUSB0",
            description="USB Serial",
            manufacturer="wch.cn",
            hwid="USB VID:PID=1A86:7523",
        )
        output = io.StringIO()

        with patch.dict("os.environ", {}, clear=False):
            with patch.object(helper_robot_control.serial.tools.list_ports, "comports", return_value=[detected_port]):
                with patch.object(helper_robot_control.glob, "glob", return_value=[]):
                    with patch.object(
                        helper_robot_control.serial,
                        "Serial",
                        side_effect=FileNotFoundError("missing"),
                    ):
                        with patch.object(helper_robot_control.time, "sleep", return_value=None):
                            with self.assertRaises(SystemExit):
                                with redirect_stdout(output):
                                    helper_robot_control.Robot()

        text = output.getvalue()
        self.assertIn("Available serial ports detected", text)
        self.assertIn("/dev/ttyUSB0", text)
        self.assertIn("Set LEIA_SERIAL_PORT", text)


if __name__ == "__main__":
    unittest.main()
