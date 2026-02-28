import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestTelemetryRobotEaseInOut(unittest.TestCase):
    def test_ramp_ms_scales_with_score_for_drive_threshold(self):
        low = telemetry_robot.ease_in_out_ramp_ms_for_score(10, cmd="f")
        mid = telemetry_robot.ease_in_out_ramp_ms_for_score(50, cmd="f")
        high = telemetry_robot.ease_in_out_ramp_ms_for_score(100, cmd="f")
        self.assertEqual(low, 300)
        self.assertTrue(low < mid < high)
        self.assertEqual(high, 800)

    def test_ramp_ms_scales_with_score_for_turn_threshold(self):
        low = telemetry_robot.ease_in_out_ramp_ms_for_score(20, cmd="l")
        mid = telemetry_robot.ease_in_out_ramp_ms_for_score(60, cmd="l")
        high = telemetry_robot.ease_in_out_ramp_ms_for_score(100, cmd="l")
        self.assertEqual(low, 300)
        self.assertTrue(low < mid < high)
        self.assertEqual(high, 800)

    def test_profile_applies_at_drive_threshold_score(self):
        profile = telemetry_robot.drive_ease_in_out_segments(
            "f",
            speed_score=10,
            duration_ms=600,
        )
        self.assertIsInstance(profile, dict)
        self.assertEqual(int(profile.get("threshold_score") or 0), 10)
        self.assertEqual(int(profile.get("ramp_ms") or 0), 300)
        self.assertIsInstance(profile.get("segments"), list)
        self.assertGreater(len(profile.get("segments") or []), 1)

    def test_profile_applies_at_turn_threshold_score(self):
        profile = telemetry_robot.drive_ease_in_out_segments(
            "l",
            speed_score=20,
            duration_ms=600,
        )
        self.assertIsInstance(profile, dict)
        self.assertEqual(int(profile.get("threshold_score") or 0), 20)
        self.assertEqual(int(profile.get("ramp_ms") or 0), 300)

    def test_profile_does_not_apply_below_drive_threshold(self):
        profile = telemetry_robot.drive_ease_in_out_segments(
            "f",
            speed_score=9,
            duration_ms=600,
        )
        self.assertIsNone(profile)

    def test_profile_does_not_apply_below_turn_threshold(self):
        profile = telemetry_robot.drive_ease_in_out_segments(
            "r",
            speed_score=19,
            duration_ms=600,
        )
        self.assertIsNone(profile)

    def test_turn_commands_supported(self):
        profile = telemetry_robot.drive_ease_in_out_segments(
            "l",
            speed_score=50,
            duration_ms=600,
        )
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile.get("cmd"), "l")
        self.assertIsInstance(profile.get("segments"), list)
        self.assertGreater(len(profile.get("segments") or []), 1)


if __name__ == "__main__":
    unittest.main()
