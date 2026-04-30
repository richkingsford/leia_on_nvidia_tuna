import sys
import time
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot
from helper_manual_hotkey_controls import (
    ManualHotkeyController,
    format_hotkey_help,
    hotkey_action_from_key,
)


class _FakeRobot:
    def __init__(self):
        self.sent = []
        self.custom_sent = []
        self.stopped = False
        self.last_command = ""
        self.supports_timed_command_queue = False
        self._last_turn_cmd = None

    def normalize_speed(self, cmd, speed):
        try:
            speed_val = max(0.0, min(1.0, float(speed or 0.0)))
        except (TypeError, ValueError):
            speed_val = 0.0
        pwm = telemetry_robot.power_to_pwm(speed_val) or 0
        return speed_val, int(pwm)

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        power = telemetry_robot.pwm_to_power(pwm) or 0.0
        payload = {
            "cmd_sent": str(cmd),
            "pwm": int(pwm),
            "power": float(power),
            "duration_ms": int(duration_ms),
            "wire_text": f"{cmd}.{int(pwm)}.{int(duration_ms)}",
        }
        self.sent.append(payload)
        self.last_command = str(payload["wire_text"])
        return payload

    def send_custom_actions_pwm(self, cmd, action_specs, duration_ms=0):
        actions = [dict(action) for action in (action_specs or []) if isinstance(action, dict)]
        pwm = max([int(action.get("pwm") or 0) for action in actions] or [0])
        power = telemetry_robot.pwm_to_power(pwm) or 0.0
        payload = {
            "cmd_sent": str(cmd),
            "pwm": int(pwm),
            "power": float(power),
            "duration_ms": int(duration_ms),
            "wire_text": f"{cmd}.custom.{int(pwm)}.{int(duration_ms)}",
            "actions": actions,
        }
        self.custom_sent.append(payload)
        self.last_command = str(payload["wire_text"])
        return payload

    def stop(self):
        self.stopped = True
        self.last_command = "s"


class _FakeWorld:
    def __init__(self):
        self.step_state = "MANUAL"


class TestManualHotkeyControls(unittest.TestCase):
    def test_hotkey_action_comes_from_world_model_mapping(self):
        left = hotkey_action_from_key("q")
        self.assertEqual(left["cmd"], "l")
        self.assertEqual(left["score"], 1)

        forward = hotkey_action_from_key("W")
        self.assertEqual(forward["cmd"], "f")
        self.assertEqual(forward["score"], 50)

    def test_help_line_reports_loaded_lift_hotkeys(self):
        line = format_hotkey_help()
        self.assertIn("Lift up: O 1%, U 50%, P 100%", line)
        self.assertIn("Lift down: K 1%, L 100%", line)
        self.assertIn("uppercase Q quits", line)

    def test_controller_sends_one_logical_hotkey_pulse(self):
        robot = _FakeRobot()
        messages = []
        controller = ManualHotkeyController(
            robot=robot,
            world=_FakeWorld(),
            logger=messages.append,
            enable_terminal=False,
        )

        action = controller.queue_key("w")
        result = controller.process_once(now=time.time())

        self.assertEqual(action["cmd"], "f")
        self.assertIsInstance(result, dict)
        self.assertEqual(robot.sent[-1]["cmd_sent"], "f")
        self.assertIsNone(controller.active_command)
        self.assertTrue(messages[0].startswith("[CTRL] Key W received"))
        self.assertTrue(messages[-1].startswith("[HOTKEY] W -> move forward"))
        self.assertNotIn("wire", messages[-1].lower())

    def test_lowercase_q_turns_left_and_uppercase_q_quits(self):
        robot = _FakeRobot()
        stopped = []
        controller = ManualHotkeyController(
            robot=robot,
            world=_FakeWorld(),
            logger=lambda _line: None,
            stop_callback=lambda: stopped.append(True),
            enable_terminal=False,
        )

        action = controller.queue_key("q")
        result = controller.process_once(now=time.time())

        self.assertEqual(action["cmd"], "l")
        self.assertTrue(controller.is_running())
        self.assertIsInstance(result, dict)
        self.assertEqual(robot.custom_sent[-1]["cmd_sent"], "l")

        quit_result = controller.queue_key("Q")

        self.assertEqual(quit_result, "quit")
        self.assertFalse(controller.is_running())
        self.assertEqual(stopped, [True])


if __name__ == "__main__":
    unittest.main()
