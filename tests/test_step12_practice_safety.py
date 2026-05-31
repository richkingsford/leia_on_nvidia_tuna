import sys
import types
import unittest
from unittest import mock
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _install_robot_module_stubs() -> None:
    native_oak = types.ModuleType("helper_brick_detector_native_oak")
    native_oak.BrickDetector = type("BrickDetector", (), {})
    sys.modules.setdefault("helper_brick_detector_native_oak", native_oak)

    yolo = types.ModuleType("helper_brick_detector_yolo")
    yolo.CYAN_HSV_BALANCED_LOWER = (0, 0, 0)
    yolo.CYAN_HSV_BALANCED_UPPER = (0, 0, 0)
    yolo.CYAN_HSV_WIDE_LOWER = (0, 0, 0)
    yolo.CYAN_HSV_WIDE_UPPER = (0, 0, 0)
    sys.modules.setdefault("helper_brick_detector_yolo", yolo)

    visibility = types.ModuleType("helper_brick_visibility_safety")
    visibility.brick_motion_measurement_from_result = lambda *args, **kwargs: None
    visibility.guarded_send_command_pwm = lambda *args, **kwargs: None
    visibility.guarded_send_custom_actions_pwm = lambda *args, **kwargs: None
    visibility.load_brick_visibility_motion_safety_config = lambda *args, **kwargs: {}
    sys.modules.setdefault("helper_brick_visibility_safety", visibility)

    holding = types.ModuleType("helper_holding_brick")
    holding.contour_target_result_tuple = lambda *args, **kwargs: None
    holding.detect_holding_brick = lambda *args, **kwargs: None
    holding.detect_masked_target_brick_contour = lambda *args, **kwargs: None
    holding.mask_held_brick_for_target_frame = lambda *args, **kwargs: None
    sys.modules.setdefault("helper_holding_brick", holding)

    mast_guard = types.ModuleType("helper_mast_direction_guard")
    mast_guard.classify_mast_y_effect = lambda *args, **kwargs: None
    mast_guard.mast_effect_is_reversal = lambda *args, **kwargs: False
    sys.modules.setdefault("helper_mast_direction_guard", mast_guard)

    robot_control = types.ModuleType("helper_robot_control")
    robot_control.Robot = type("Robot", (), {})
    sys.modules.setdefault("helper_robot_control", robot_control)

    telemetry = types.ModuleType("telemetry_robot")
    sys.modules.setdefault("telemetry_robot", telemetry)


_install_robot_module_stubs()

import practiceStep12EmptyHalfReset as step12


class TestStep12PracticeSafety(unittest.TestCase):
    def test_85_percent_requires_5_of_5_trials(self):
        self.assertEqual(step12._required_wins_for_rate(5, 0.85), 5)

    def test_block_summary_marks_partial_clean_run_incomplete(self):
        summary = step12._block_summary_counts(
            [
                {
                    "clean": True,
                    "s1_won": True,
                    "s2_won": True,
                    "wrong_way_count": 0,
                    "severe_overshoot_count": 0,
                }
            ],
            requested_trials=5,
            min_win_rate=0.85,
        )

        self.assertEqual(summary["completed_trials"], 1)
        self.assertEqual(summary["required_wins"], 5)
        self.assertFalse(summary["meets_trial_count"])
        self.assertFalse(summary["meets_min_win_rate"])
        self.assertAlmostEqual(summary["completed_s1_win_rate"], 1.0)
        self.assertAlmostEqual(summary["s1_win_rate"], 0.2)

    def test_block_summary_flags_100x_clean_proof(self):
        rows = [
            {
                "clean": True,
                "s1_won": True,
                "s2_won": True,
                "wrong_way_count": 0,
                "severe_overshoot_count": 0,
            }
            for _ in range(100)
        ]

        summary = step12._block_summary_counts(rows, requested_trials=100, min_win_rate=0.85)

        self.assertEqual(summary["longest_clean_streak"], 100)
        self.assertTrue(summary["meets_trial_count"])
        self.assertTrue(summary["meets_min_win_rate"])
        self.assertTrue(summary["meets_safety"])
        self.assertTrue(summary["meets_100x_proof"])

    def test_far_low_confident_reading_is_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertTrue(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_shifted_live_far_low_reading_is_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 299.5, "y_mm": -56.7}

        self.assertTrue(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_normal_far_start_is_not_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 323.8, "y_mm": -32.2}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_not_confident_reading_uses_existing_visibility_stop_path(self):
        reading = {"confident": False, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_reset_mast_cheat_defaults_to_2p5s_up_and_no_down(self):
        args = types.SimpleNamespace(reset_mast_cheat=True)

        cfg = step12._reset_mast_cheat_config(args)

        self.assertEqual(cfg, {"enabled": True, "up_ms": 2500, "down_ms": 0})

    def test_reset_mast_cheat_splits_long_moves_into_one_second_pulses(self):
        self.assertEqual(step12._split_mast_pulses(2500), [1000, 1000, 500])
        self.assertEqual(step12._split_mast_pulses(0), [])
        self.assertEqual(step12._split_mast_pulses(-100), [])

    def test_send_mast_pulse_train_stops_after_each_pulse(self):
        robot = mock.Mock()

        with mock.patch.object(step12.time, "sleep"), mock.patch.object(step12.follow, "_stop_robot") as stop_robot:
            record = step12._send_mast_pulse_train(robot, "u", 2500, label="test")

        self.assertEqual(record["sent_ms"], 2500)
        self.assertEqual(record["pulses"], [1000, 1000, 500])
        self.assertEqual(robot.send_command_pwm.call_count, 3)
        self.assertEqual(stop_robot.call_count, 3)

    def test_send_mast_pulse_train_rejects_unknown_direction_without_motion(self):
        robot = mock.Mock()

        record = step12._send_mast_pulse_train(robot, "x", 2500, label="test")

        self.assertTrue(record["skipped"])
        self.assertEqual(record["sent_ms"], 0)
        robot.send_command_pwm.assert_not_called()

    def test_measurement_delta_reports_axis_differences(self):
        before = {"dist_mm": 220.0, "x_mm": 4.0, "y_mm": -41.5}
        after = {"dist_mm": 225.5, "x_mm": 2.5, "y_mm": -39.0}

        delta = step12._measurement_delta(before, after)

        self.assertEqual(delta, {"dist_mm": 5.5, "x_mm": -1.5, "y_mm": 2.5})


if __name__ == "__main__":
    unittest.main()
