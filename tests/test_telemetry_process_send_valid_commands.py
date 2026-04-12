import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessSendValidCommands(unittest.TestCase):
    def setUp(self):
        self.robot = type("_MockRobot", (), {"send_command": lambda *_a, **_k: None})()
        self.world = type("_MockWorld", (), {})()
        self.step = "TEST"
        self.sent = []
        self.orig_send_pwm = telemetry_process.send_robot_command_pwm

        def _fake_send_robot_command_pwm(*args, **kwargs):
            entry = dict(kwargs)
            entry["cmd_pos"] = args[3] if len(args) > 3 else None
            entry["duration_ms_pos"] = args[6] if len(args) > 6 else None
            self.sent.append(entry)
            return {"cmd_sent": entry.get("cmd_pos"), "duration_ms": entry.get("duration_ms_pos")}

        telemetry_process.send_robot_command_pwm = _fake_send_robot_command_pwm

    def tearDown(self):
        telemetry_process.send_robot_command_pwm = self.orig_send_pwm

    def test_send_robot_command_clamps_low_speed_score_to_minimum(self):
        result = telemetry_process.send_robot_command(
            self.robot,
            self.world,
            self.step,
            "f",
            None,
            speed_score=0,
            auto_mode=False,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(int(self.sent[0].get("speed_score")), int(telemetry_process.telemetry_robot_module.SPEED_SCORE_MIN))

    def test_send_robot_command_clamps_high_speed_score_to_maximum(self):
        result = telemetry_process.send_robot_command(
            self.robot,
            self.world,
            self.step,
            "f",
            None,
            speed_score=1000,
            auto_mode=False,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(int(self.sent[0].get("speed_score")), int(telemetry_process.telemetry_robot_module.SPEED_SCORE_MAX))

    def test_send_robot_command_clamps_low_duration_override_to_minimum(self):
        _, _, _, min_duration_ms = telemetry_process.telemetry_robot_module.speed_power_pwm_for_cmd(
            "b",
            telemetry_process.telemetry_robot_module.SPEED_SCORE_MIN,
        )
        result = telemetry_process.send_robot_command(
            self.robot,
            self.world,
            self.step,
            "b",
            None,
            speed_score=15,
            auto_mode=False,
            duration_override_ms=int(min_duration_ms) - 5,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(int(self.sent[0].get("duration_ms_pos")), int(min_duration_ms))

    def test_send_robot_command_rejects_invalid_cmd(self):
        result = telemetry_process.send_robot_command(
            self.robot,
            self.world,
            self.step,
            "invalid",
            None,
            speed_score=10,
        )
        self.assertIsNone(result)
        self.assertEqual(len(self.sent), 0)

    def test_send_robot_command_normalizes_cmd_case(self):
        result = telemetry_process.send_robot_command(
            self.robot,
            self.world,
            self.step,
            "B",
            None,
            speed_score=10,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(str(self.sent[0].get("cmd_pos")).lower(), "b")

    def test_send_robot_command_pwm_rejects_invalid_cmd(self):
        result = self.orig_send_pwm(
            self.robot,
            self.world,
            self.step,
            "invalid",
            0.2,
            100,
            250,
            speed_score=10,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()