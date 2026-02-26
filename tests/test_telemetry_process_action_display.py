import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class _DummyRobot:
    def __init__(self, *, supports_timed_command_queue=False):
        self.sent = []
        self._last_turn_cmd = None
        self.supports_timed_command_queue = bool(supports_timed_command_queue)

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))


class _DummyWorld:
    def __init__(self):
        self.brick = {
            "visible": True,
            "dist": 200.0,
            "angle": 0.0,
            "x_axis": 0.0,
            "offset_x": 0.0,
            "confidence": 90.0,
        }
        self.process_rules = {}


class TestTelemetryProcessActionDisplay(unittest.TestCase):
    def test_send_robot_command_records_display_from_logical_command(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="l",
            speed=0.0,
            speed_score=telemetry_robot.SPEED_SCORE_MIN,
            auto_mode=True,
        )
        self.assertTrue(robot.sent)
        self.assertEqual(getattr(world, "_last_action_cmd", None), "l")
        cmd_display = getattr(world, "_last_action_cmd", None) or "l"
        expected_score = telemetry_robot.quantize_speed("l", speed=getattr(world, "_last_action_speed", 0.0))[1]
        expected = telemetry_process.action_display_text(cmd_display, expected_score)
        self.assertEqual(getattr(world, "_last_action_display", None), expected)

    def test_format_control_action_line_uses_logical_direction_even_with_remap(self):
        orig_remap = getattr(telemetry_process.telemetry_robot_module, "COMMAND_REMAP", None)
        try:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = {"f": "b"}
            line = telemetry_process.format_control_action_line("f", 0.3, "align")
        finally:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = orig_remap
        self.assertIn("move forward", line)

    def test_send_robot_command_auto_mode_caps_score_to_25(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=80,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertLessEqual(int(meta.get("score_model") or 0), 25)
        self.assertLessEqual(int(meta.get("score_effective") or 0), 25)

    def test_send_robot_command_auto_mode_caps_find_steps_to_25(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="FIND_WALL",
            cmd="r",
            speed=0.0,
            speed_score=80,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertLessEqual(int(meta.get("score_model") or 0), 25)
        self.assertLessEqual(int(meta.get("score_effective") or 0), 25)

    def test_auto_drive_commands_do_not_use_anti_alias_ramp_segments_without_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("AA(", str(getattr(world, "_last_action_sent_display", "")))
        detail = telemetry_process.auto_action_detail_text("f", 50, action_meta=meta)
        self.assertNotIn("AA(", detail)

    def test_auto_drive_commands_use_anti_alias_ramp_segments_with_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot(supports_timed_command_queue=True)
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertIsInstance(meta.get("segments"), list)
        self.assertGreater(len(meta.get("segments") or []), 1)
        self.assertGreater(len(robot.sent), 1)
        self.assertIn("AA(", str(getattr(world, "_last_action_sent_display", "")))
        detail = telemetry_process.auto_action_detail_text("f", 50, action_meta=meta)
        self.assertIn("AA(", detail)

    def test_manual_drive_commands_do_not_use_anti_alias_ramp_segments(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="MANUAL",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=False,
        )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("AA(", str(getattr(world, "_last_action_sent_display", "")))


if __name__ == "__main__":
    unittest.main()
