import unittest
from unittest.mock import patch

from calibration import helper_calibrate_x as calibrate_x


class CalibrateXTests(unittest.TestCase):
    def test_plot_title_text_uses_single_line_distance_title(self):
        self.assertEqual(calibrate_x._plot_title_text([]), "X Calibration at 166mm")
        self.assertEqual(calibrate_x._plot_title_text([124.8, 125.2, 125.0]), "X Calibration at 125mm")

    def test_movement_metrics_reports_total_distance_even_when_wrong_way(self):
        metrics = calibrate_x._movement_metrics("r", 2.0, 1.25)
        self.assertAlmostEqual(metrics["raw_delta_mm"], -0.75)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], -0.75)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 0.75)
        self.assertTrue(metrics["wrong_way"])

    def test_x_axis_positive_motion_defaults_to_r(self):
        self.assertEqual(calibrate_x._x_cmd_for_positive_motion(), "r")
        self.assertEqual(calibrate_x._x_cmd_for_negative_motion(), "l")

    def test_wrong_way_reason_mentions_center_and_tiny_motion_when_applicable(self):
        reason = calibrate_x._wrong_way_reason_text(
            pre_x_mm=0.20,
            post_x_mm=0.05,
            target_x_mm=0.0,
        )
        self.assertIn("near 0", reason)
        self.assertIn("tiny", reason)

    def test_build_duration_schedule_uses_ten_ms_steps(self):
        durations = calibrate_x._build_duration_schedule(
            trials=None,
            min_duration_ms=200,
            max_duration_ms=230,
        )
        self.assertEqual(durations, [200, 210, 220, 230])

    def test_build_trial_plan_runs_left_and_right_twice_per_duration(self):
        plan = calibrate_x._build_trial_plan(
            durations_ms=[200, 210],
            trials=None,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 200, "cmd": "r"},
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 200, "cmd": "r"},
                {"duration_ms": 210, "cmd": "l"},
                {"duration_ms": 210, "cmd": "r"},
                {"duration_ms": 210, "cmd": "l"},
                {"duration_ms": 210, "cmd": "r"},
            ],
        )

    def test_build_trial_plan_honors_total_trial_cap(self):
        plan = calibrate_x._build_trial_plan(
            durations_ms=[200, 210],
            trials=5,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 200, "cmd": "r"},
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 200, "cmd": "r"},
                {"duration_ms": 210, "cmd": "l"},
            ],
        )

    def test_planned_durations_ms_preserves_first_use_order(self):
        durations = calibrate_x._planned_durations_ms(
            [
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 200, "cmd": "r"},
                {"duration_ms": 210, "cmd": "l"},
                {"duration_ms": 200, "cmd": "l"},
                {"duration_ms": 220, "cmd": "r"},
            ]
        )
        self.assertEqual(durations, [200, 210, 220])

    def test_send_fixed_score_command_disables_first_turn_halving_for_pwm_path(self):
        with patch.object(
            calibrate_x,
            "_turn_hotkey_profile",
            return_value={"power": 0.2, "pwm": 40, "hotkey": "Q"},
        ), patch.object(
            calibrate_x,
            "send_robot_command_pwm",
            return_value={"cmd_sent": "l", "duration_ms": 240},
        ) as mock_send:
            calibrate_x._send_fixed_score_command(
                robot=object(),
                world=object(),
                step="CALIBRATE_X",
                cmd="l",
                score=1,
                duration_override_ms=240,
            )
        self.assertFalse(mock_send.call_args.kwargs["half_first_turn_pulse"])

    def test_send_fixed_score_command_disables_first_turn_halving_for_score_path(self):
        with patch.object(calibrate_x, "_turn_hotkey_profile", return_value=None), patch.object(
            calibrate_x,
            "send_robot_command",
            return_value={"cmd_sent": "r", "duration_ms": 240},
        ) as mock_send:
            calibrate_x._send_fixed_score_command(
                robot=object(),
                world=object(),
                step="CALIBRATE_X",
                cmd="r",
                score=1,
                duration_override_ms=240,
            )
        self.assertFalse(mock_send.call_args.kwargs["half_first_turn_pulse"])


if __name__ == "__main__":
    unittest.main()
