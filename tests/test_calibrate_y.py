import json
import math
import random
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from calibration import helper_calibrate_y as calibrate_y


class CalibrateYTests(unittest.TestCase):
    def test_exit_as_script_returns_under_debugger(self):
        with patch.object(calibrate_y.sys, "gettrace", return_value=object()):
            self.assertIsNone(calibrate_y._exit_as_script(1))

    def test_exit_as_script_raises_without_debugger(self):
        with patch.object(calibrate_y.sys, "gettrace", return_value=None):
            with self.assertRaises(SystemExit) as ctx:
                calibrate_y._exit_as_script(3)
        self.assertEqual(ctx.exception.code, 3)

    def test_command_delta_is_positive_in_command_direction(self):
        self.assertAlmostEqual(calibrate_y._command_delta_mm("u", 5.5, 8.0), 2.5)
        self.assertAlmostEqual(calibrate_y._command_delta_mm("d", -2.5, -6.0), 3.5)

    def test_movement_metrics_reports_absolute_distance_and_wrong_way(self):
        metrics = calibrate_y._movement_metrics("u", 5.5, 8.0)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 2.5)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], 2.5)
        self.assertFalse(metrics["wrong_way"])

        wrong_way = calibrate_y._movement_metrics("u", 8.0, 5.5)
        self.assertAlmostEqual(wrong_way["cmd_delta_mm"], 2.5)
        self.assertAlmostEqual(wrong_way["signed_cmd_delta_mm"], -2.5)
        self.assertTrue(wrong_way["wrong_way"])

    def test_normalize_cmd_accepts_center_alias(self):
        self.assertEqual(calibrate_y._normalize_cmd("center", allow_auto=True), "auto")

    def test_auto_cmd_for_y_moves_toward_center(self):
        self.assertEqual(
            calibrate_y._auto_cmd_for_y(-3.0, center_y_mm=0.0, deadband_mm=0.5, fallback_cmd="u"),
            "u",
        )
        self.assertEqual(
            calibrate_y._auto_cmd_for_y(3.0, center_y_mm=0.0, deadband_mm=0.5, fallback_cmd="d"),
            "d",
        )

    def test_auto_cmd_for_y_uses_fallback_inside_deadband(self):
        self.assertEqual(
            calibrate_y._auto_cmd_for_y(0.2, center_y_mm=0.0, deadband_mm=0.5, fallback_cmd="u"),
            "u",
        )

    def test_auto_cmd_for_y_uses_script_sweet_spot_default(self):
        self.assertEqual(
            calibrate_y._auto_cmd_for_y(
                6.0,
                center_y_mm=calibrate_y.Y_AXIS_SWEET_SPOT_MM_DEFAULT,
                deadband_mm=0.5,
                fallback_cmd="d",
            ),
            "u",
        )
        self.assertEqual(
            calibrate_y._auto_cmd_for_y(
                14.0,
                center_y_mm=calibrate_y.Y_AXIS_SWEET_SPOT_MM_DEFAULT,
                deadband_mm=0.5,
                fallback_cmd="d",
            ),
            "d",
        )

    def test_trial_band_for_cmd_uses_hardcoded_up_and_down_ranges(self):
        up_band = calibrate_y._trial_band_for_cmd("u")
        down_band = calibrate_y._trial_band_for_cmd("d")
        self.assertEqual(
            (up_band["min_mm"], up_band["max_mm"]),
            (calibrate_y.UP_TRIAL_BAND_MIN_MM, calibrate_y.UP_TRIAL_BAND_MAX_MM),
        )
        self.assertEqual(
            (down_band["min_mm"], down_band["max_mm"]),
            (calibrate_y.DOWN_TRIAL_BAND_MIN_MM, calibrate_y.DOWN_TRIAL_BAND_MAX_MM),
        )

    def test_trial_band_correction_cmd_uses_observed_y_motion_sign(self):
        self.assertEqual(calibrate_y._trial_band_correction_cmd("u", 22.98), "d")
        self.assertEqual(calibrate_y._trial_band_correction_cmd("u", -10.0), "u")
        self.assertEqual(calibrate_y._trial_band_correction_cmd("d", 10.0), "d")
        self.assertEqual(calibrate_y._trial_band_correction_cmd("d", 0.0), "u")

    def test_plot_color_for_cmd_separates_up_and_down(self):
        self.assertEqual(calibrate_y._plot_color_for_cmd("u"), "#1f77b4")
        self.assertEqual(calibrate_y._plot_color_for_cmd("d"), "#ff7f0e")
        self.assertEqual(calibrate_y._plot_color_for_cmd("u", "repeat"), "#1f77b4")
        self.assertEqual(calibrate_y._plot_color_for_cmd("d", "repeat"), "#ff7f0e")

    def test_plot_series_collapses_setup_and_measured_points_by_command_family(self):
        self.assertEqual(calibrate_y._plot_series_key("u", "trial"), "u")
        self.assertEqual(calibrate_y._plot_series_key("u", "reset"), "u")
        self.assertEqual(calibrate_y._plot_series_key("u", "repeat"), "u")
        self.assertEqual(calibrate_y._plot_series_label("u", "trial"), "mast_up")
        self.assertEqual(calibrate_y._plot_series_label("u", "reset"), "mast_up")
        self.assertEqual(calibrate_y._plot_series_label("d", "reset"), "mast_down")
        self.assertEqual(calibrate_y._plot_series_label("u", "repeat"), "mast_up")
        self.assertEqual(calibrate_y._plot_series_label("d", "repeat"), "mast_down")

    def test_plot_title_text_defines_brick_distance_context(self):
        empty_title = calibrate_y._plot_title_text([])
        self.assertIn("Brick distance = vision.dist", empty_title)
        titled = calibrate_y._plot_title_text([100.0, 104.0, 98.0])
        self.assertIn("latest=98.0mm", titled)
        self.assertIn("median=100.0mm", titled)
        self.assertIn("range=98.0..104.0mm", titled)

    def test_plot_offsets_handles_empty_series(self):
        offsets = calibrate_y._plot_offsets([], [])
        self.assertEqual(len(offsets), 1)
        self.assertTrue(math.isnan(offsets[0][0]))
        self.assertTrue(math.isnan(offsets[0][1]))
        self.assertEqual(calibrate_y._plot_offsets([250.0], [3.5]), [(250.0, 3.5)])

    def test_planned_action_meta_uses_central_speed_curve(self):
        power, pwm, score_used, duration_ms = calibrate_y.speed_power_pwm_for_cmd("u", 1)
        meta = calibrate_y._planned_action_meta("u", 1, duration_ms)

        self.assertEqual(meta["pwm"], pwm)
        self.assertAlmostEqual(meta["power"], power)
        self.assertEqual(meta["score_model"], score_used)
        self.assertEqual(meta["duration_ms"], duration_ms)

    def test_trial_band_status_line_reports_within_and_not_within(self):
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=False):
            within, within_line = calibrate_y._trial_band_status_line("d", 6.0)
            outside, outside_line = calibrate_y._trial_band_status_line("u", 18.0)
        self.assertTrue(within)
        self.assertIn("Within trial band for mast_down", within_line)
        self.assertFalse(outside)
        self.assertIn("Not within trial band for mast_up", outside_line)

    def test_build_duration_schedule_walks_section_bands_in_order(self):
        durations = calibrate_y._build_duration_schedule(
            trials=None,
            min_duration_ms=250,
            max_duration_ms=500,
            rng=random.Random(7),
        )
        self.assertEqual(len(durations), 15)
        self.assertTrue(all(250 <= value <= 300 for value in durations[:5]))
        self.assertTrue(all(350 <= value <= 400 for value in durations[5:10]))
        self.assertTrue(all(450 <= value <= 500 for value in durations[10:15]))

    def test_build_duration_schedule_honors_trial_cap_against_section_pattern(self):
        durations = calibrate_y._build_duration_schedule(
            trials=7,
            min_duration_ms=250,
            max_duration_ms=500,
            rng=random.Random(7),
        )
        self.assertEqual(len(durations), 7)
        self.assertTrue(all(250 <= value <= 300 for value in durations[:5]))
        self.assertTrue(all(350 <= value <= 400 for value in durations[5:7]))

    def test_build_reset_duration_schedule_uses_smaller_variable_band(self):
        durations = calibrate_y._build_reset_duration_schedule(rng=random.Random(7))
        self.assertEqual(len(durations), 15)
        self.assertTrue(all(250 <= value <= 300 for value in durations[:5]))
        self.assertTrue(all(450 <= value <= 500 for value in durations[-5:]))

    def test_next_cycled_duration_ms_rotates_reset_schedule(self):
        schedule = deque([250, 350, 450])
        self.assertEqual(calibrate_y._next_cycled_duration_ms(schedule), 250)
        self.assertEqual(list(schedule), [350, 450, 250])
        self.assertEqual(calibrate_y._next_cycled_duration_ms(schedule), 350)

    def test_effective_preflight_speed_score_uses_detected_minimum(self):
        self.assertEqual(
            calibrate_y._effective_preflight_speed_score(1, {"score_used": 3}),
            3,
        )

    def test_effective_preflight_speed_score_falls_back_to_requested_score(self):
        self.assertEqual(
            calibrate_y._effective_preflight_speed_score(5, None),
            5,
        )
        self.assertEqual(
            calibrate_y._effective_preflight_speed_score(5, {"score_used": None}),
            5,
        )

    def test_calculate_prediction_comparison_reports_prediction_closeness(self):
        comparison = calibrate_y._calculate_prediction_comparison(
            actual_distance_mm=1.8,
            predicted_distance_mm=2.0,
            curve_source="curve_prediction",
        )
        self.assertAlmostEqual(comparison["predicted_distance_mm"], 2.0)
        self.assertAlmostEqual(comparison["absolute_difference_mm"], 0.2)
        self.assertAlmostEqual(comparison["prediction_closeness_percentage"], 90.0)

    def test_calculate_prediction_comparison_clamps_prediction_closeness_at_zero(self):
        comparison = calibrate_y._calculate_prediction_comparison(
            actual_distance_mm=5.5,
            predicted_distance_mm=2.0,
            curve_source="curve_prediction",
        )
        self.assertAlmostEqual(comparison["absolute_difference_mm"], 3.5)
        self.assertAlmostEqual(comparison["prediction_closeness_percentage"], 0.0)

    def test_load_y_duration_calibration_collects_named_curves(self):
        payload = {
            "schema_version": 1,
            "near_curve": {
                "reference_distance_mm": 50.0,
                "speed_score_pct": 1,
                "by_cmd": {
                    "u": {"slope_mm_per_ms": 0.01, "intercept_mm": 0.1},
                    "d": {"slope_mm_per_ms": 0.02, "intercept_mm": 0.2},
                },
            },
            "far_curve": {
                "reference_distance_mm": 120.0,
                "speed_score_pct": 3,
                "by_cmd": {
                    "u": {"slope_mm_per_ms": 0.03, "intercept_mm": 0.3},
                    "d": {"slope_mm_per_ms": 0.04, "intercept_mm": 0.4},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            curve_path = Path(tmpdir) / "world_model_up_down_curve.json"
            curve_path.write_text(json.dumps(payload))
            loaded = calibrate_y._load_y_duration_calibration(curve_path)
        self.assertEqual(
            [curve["curve_name"] for curve in loaded["curves"]],
            [
                "near_curve at 50mm distance at 1% speed",
                "far_curve at 120mm distance at 3% speed",
            ],
        )

    def test_predict_movement_from_curve_uses_closest_curve_name_for_observed_distance(self):
        y_calibration = {
            "curves": [
                {
                    "curve_name": "near_curve at 50mm distance at 1% speed",
                    "reference_distance_mm": 50.0,
                    "by_cmd": {
                        "d": {"slope_mm_per_ms": 0.01, "intercept_mm": 0.5},
                    },
                },
                {
                    "curve_name": "far_curve at 120mm distance at 3% speed",
                    "reference_distance_mm": 120.0,
                    "by_cmd": {
                        "d": {"slope_mm_per_ms": 0.03, "intercept_mm": 0.25},
                    },
                },
            ],
        }
        predicted_mm, curve_source = calibrate_y._predict_movement_from_curve(
            cmd="d",
            duration_ms=100,
            y_calibration=y_calibration,
            observed_distance_mm=110.0,
        )
        self.assertAlmostEqual(predicted_mm, 3.25)
        self.assertEqual(curve_source, "far_curve at 120mm distance at 3% speed")

    def test_predict_duration_for_target_delta_mm_uses_closest_curve_for_observed_distance(self):
        y_calibration = {
            "curves": [
                {
                    "curve_name": "near_curve at 50mm distance at 1% speed",
                    "reference_distance_mm": 50.0,
                    "by_cmd": {
                        "u": {"slope_mm_per_ms": 0.02, "intercept_mm": 0.5},
                    },
                },
                {
                    "curve_name": "far_curve at 120mm distance at 3% speed",
                    "reference_distance_mm": 120.0,
                    "by_cmd": {
                        "u": {"slope_mm_per_ms": 0.01, "intercept_mm": 0.5},
                    },
                },
            ],
        }
        duration_ms = calibrate_y._predict_duration_for_target_delta_mm(
            cmd="u",
            abs_delta_mm=4.5,
            duration_min_ms=100,
            duration_max_ms=500,
            y_calibration=y_calibration,
            observed_distance_mm=52.0,
        )
        self.assertEqual(duration_ms, 200)

    def test_format_prediction_comparison_fields_omits_mm_suffix(self):
        comparison = {
            "absolute_difference_mm": 1.62,
            "prediction_closeness_percentage": 0.0,
        }
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=False):
            text = calibrate_y._format_prediction_comparison_fields(comparison)
        self.assertEqual(
            text,
            "absolute_difference=1.62 prediction_closeness=0.0%",
        )

    def test_format_prediction_comparison_fields_colors_numbers_only_from_closeness_rule(self):
        comparison = {
            "absolute_difference_mm": 1.62,
            "prediction_closeness_percentage": 0.0,
        }
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=True):
            text = calibrate_y._format_prediction_comparison_fields(comparison)
        self.assertIn("absolute_difference=\033[92m1.62\033[0m", text)
        self.assertIn("prediction_closeness=\033[92m0.0\033[0m%", text)

    def test_format_prediction_comparison_fields_uses_red_when_closeness_is_not_below_25(self):
        comparison = {
            "absolute_difference_mm": 1.62,
            "prediction_closeness_percentage": 25.0,
        }
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=True):
            text = calibrate_y._format_prediction_comparison_fields(comparison)
        self.assertIn("absolute_difference=\033[91m1.62\033[0m", text)
        self.assertIn("prediction_closeness=\033[91m25.0\033[0m%", text)

    def test_trials_setup_log_line_reports_observed_distance_and_curve(self):
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=False):
            text = calibrate_y._trials_setup_log_line(
                observed_distance_mm=49.0,
                closest_curve_name="aruco_marker_calibration at 49mm distance at 1% speed",
            )
        self.assertEqual(
            text,
            "[TRIALS SETUP] observed_distance=49.00mm "
            "closest_speed_curve=aruco_marker_calibration at 49mm distance at 1% speed",
        )

    def test_trials_setup_log_line_is_green(self):
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=True):
            text = calibrate_y._trials_setup_log_line(
                observed_distance_mm=49.0,
                closest_curve_name="aruco_marker_calibration at 49mm distance at 1% speed",
            )
        self.assertTrue(
            text.startswith(
                "\033[92m[TRIALS SETUP] observed_distance=49.00mm "
                "closest_speed_curve=aruco_marker_calibration at 49mm distance at 1% speed"
            )
        )

    def test_should_log_command_inversion_detail_only_on_expectation_failure(self):
        self.assertTrue(
            calibrate_y._should_log_command_inversion_detail(
                logical_cmd="d",
                wire_cmd="u",
                raw_delta_mm=-0.63,
                threshold_mm=0.60,
            )
        )

    def test_trial_result_status_text_uses_useful_and_fail_labels(self):
        with patch.object(calibrate_y, "_supports_ansi_color", return_value=False):
            self.assertEqual(calibrate_y._trial_result_status_text(useful=True), "USEFUL")
            self.assertEqual(calibrate_y._trial_result_status_text(useful=False), "FAIL")

    def test_trial_result_label_uses_trial_wording(self):
        self.assertEqual(
            calibrate_y._trial_result_label(trial_idx=1, trials_planned=2),
            "Trial 1/2",
        )
        self.assertEqual(
            calibrate_y._trial_result_label(trial_idx=2, trials_planned=2),
            "Trial 2/2",
        )
        self.assertTrue(
            calibrate_y._should_log_command_inversion_detail(
                logical_cmd="d",
                wire_cmd="u",
                raw_delta_mm=-0.63,
                threshold_mm=0.60,
            )
        )
        self.assertFalse(
            calibrate_y._should_log_command_inversion_detail(
                logical_cmd="d",
                wire_cmd="u",
                raw_delta_mm=0.63,
                threshold_mm=0.60,
            )
        )
        self.assertTrue(
            calibrate_y._should_log_command_inversion_detail(
                logical_cmd="d",
                wire_cmd="u",
                raw_delta_mm=None,
                threshold_mm=0.60,
            )
        )

    def test_log_command_inversion_detail_starts_with_trial_label(self):
        with patch.object(calibrate_y, "log_line") as mock_log_line:
            calibrate_y._log_command_inversion_detail(
                prefix="[CALIBRATE_Y] ⚠️  Command inversion detail:",
                trial_label="Trial 59/60",
                logical_cmd="d",
                wire_cmd="u",
                raw_delta_mm=-0.81,
                threshold_mm=0.60,
            )
        mock_log_line.assert_called_once_with(
            "[CALIBRATE_Y] Trial 59/60: ⚠️  Command inversion detail: "
            "logical mast_down expects the brick to go upwards on camera, and observed it move upwards "
            "(0.81mm; raw_delta=-0.81mm). Wire mast_up implies downwards camera motion."
        )

    def test_run_trial_action_plots_wrong_way_trials(self):
        pre_pose = {
            "offset_y": 8.0,
            "offset_x": 0.0,
            "dist": 105.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        post_pose = dict(pre_pose)
        post_pose["offset_y"] = 7.19
        post_pose["confidence"] = 91.0
        plotter = Mock()

        with patch.object(
            calibrate_y,
            "_send_fixed_score_command",
            return_value={"duration_ms": 300, "cmd_sent": "u", "pwm": 40, "power": 0.2},
        ), patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(post_pose, {"mode": "trial_full", "reobserved": False}),
        ), patch.object(
            calibrate_y,
            "_predict_movement_from_curve",
            return_value=(0.75, "curve_a"),
        ), patch.object(
            calibrate_y,
            "_calculate_prediction_comparison",
            return_value={
                "predicted_distance_mm": 0.75,
                "curve_source": "curve_a",
                "absolute_difference_mm": 0.06,
                "prediction_closeness_percentage": 92.0,
            },
        ), patch.object(
            calibrate_y,
            "_format_prediction_comparison_fields",
            return_value="absolute_difference=0.06 prediction_closeness=92.0%",
        ), patch.object(calibrate_y, "log_line"):
            row, abort_reason = calibrate_y._run_trial_action(
                trial_idx=4,
                trials_planned=60,
                trial_label="Trial 4/60",
                cmd="u",
                duration_ms=300,
                phase="trial",
                source_trial=None,
                action_step="CALIBRATE_Y",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque(maxlen=32),
                setup_score=1,
                center_target_y_mm=0.0,
                observe_samples=1,
                observe_timeout_s=1.0,
                post_act_settle_s=0.0,
                camera_direction_check={},
                plotter=plotter,
                initial_pre_pose=pre_pose,
                initial_pre_obs_meta={"mode": "trial_full", "reobserved": False},
                y_duration_cal={"curves": []},
                stream_refresh_fn=None,
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        self.assertTrue(row.wrong_way)
        plotter.add_point.assert_called_once()
        plotted_kwargs = plotter.add_point.call_args.kwargs
        self.assertEqual(plotted_kwargs["duration_ms"], 300)
        self.assertAlmostEqual(plotted_kwargs["distance_mm"], 0.81)
        self.assertEqual(plotted_kwargs["trial"], 4)
        self.assertEqual(plotted_kwargs["cmd"], "u")
        self.assertEqual(plotted_kwargs["kind"], "trial")
        self.assertAlmostEqual(plotted_kwargs["pre_brick_distance_mm"], 105.0)
        self.assertAlmostEqual(plotted_kwargs["post_brick_distance_mm"], 105.0)
        self.assertIsNone(plotted_kwargs["annotation_label"])

    def test_run_trial_action_uses_distance_curve_speed_score(self):
        pre_pose = {
            "offset_y": 8.0,
            "offset_x": 0.0,
            "dist": 250.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        post_pose = dict(pre_pose)
        post_pose["offset_y"] = 7.2
        post_pose["confidence"] = 91.0
        trial_speed_profile = {
            "metric": "brick_distance_mm",
            "curve_points": [
                {"distance_mm": 120.0, "speed_score": 1},
                {"distance_mm": 250.0, "speed_score": 100},
            ],
        }

        with patch.object(
            calibrate_y,
            "_send_fixed_score_command",
            return_value={"duration_ms": 300, "cmd_sent": "d", "pwm": 40, "power": 0.2},
        ) as mock_send, patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(post_pose, {"mode": "trial_full", "reobserved": False}),
        ), patch.object(
            calibrate_y,
            "_predict_movement_from_curve",
            return_value=(0.75, "curve_a"),
        ), patch.object(
            calibrate_y,
            "_calculate_prediction_comparison",
            return_value={
                "predicted_distance_mm": 0.75,
                "curve_source": "curve_a",
                "absolute_difference_mm": 0.05,
                "prediction_closeness_percentage": 93.0,
            },
        ), patch.object(
            calibrate_y,
            "_format_prediction_comparison_fields",
            return_value="absolute_difference=0.05 prediction_closeness=93.0%",
        ), patch.object(calibrate_y, "log_line"):
            row, abort_reason = calibrate_y._run_trial_action(
                trial_idx=6,
                trials_planned=60,
                trial_label="Trial 6/60",
                cmd="d",
                duration_ms=300,
                phase="trial",
                source_trial=None,
                action_step="CALIBRATE_Y",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque(maxlen=32),
                setup_score=1,
                center_target_y_mm=0.0,
                observe_samples=1,
                observe_timeout_s=1.0,
                post_act_settle_s=0.0,
                camera_direction_check={},
                plotter=None,
                initial_pre_pose=pre_pose,
                initial_pre_obs_meta={"mode": "trial_full", "reobserved": False},
                y_duration_cal={"curves": []},
                trial_speed_profile=trial_speed_profile,
                stream_refresh_fn=None,
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        self.assertEqual(mock_send.call_args.kwargs["score"], 100)
        self.assertEqual(row.score_requested, 100)

    def test_run_trial_action_logs_trial_summary_before_inversion_detail(self):
        pre_pose = {
            "offset_y": 8.0,
            "offset_x": 0.0,
            "dist": 105.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        post_pose = dict(pre_pose)
        post_pose["offset_y"] = 7.19
        post_pose["confidence"] = 91.0

        with patch.object(
            calibrate_y,
            "_send_fixed_score_command",
            return_value={"duration_ms": 300, "cmd_sent": "u", "pwm": 40, "power": 0.2},
        ), patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(post_pose, {"mode": "trial_full", "reobserved": False}),
        ), patch.object(
            calibrate_y,
            "_predict_movement_from_curve",
            return_value=(0.75, "curve_a"),
        ), patch.object(
            calibrate_y,
            "_calculate_prediction_comparison",
            return_value={
                "predicted_distance_mm": 0.75,
                "curve_source": "curve_a",
                "absolute_difference_mm": 0.06,
                "prediction_closeness_percentage": 92.0,
            },
        ), patch.object(
            calibrate_y,
            "_format_prediction_comparison_fields",
            return_value="absolute_difference=0.06 prediction_closeness=92.0%",
        ), patch.object(calibrate_y, "log_line") as mock_log_line:
            row, abort_reason = calibrate_y._run_trial_action(
                trial_idx=5,
                trials_planned=60,
                trial_label="Trial 5/60",
                cmd="d",
                duration_ms=300,
                phase="trial",
                source_trial=None,
                action_step="CALIBRATE_Y",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque(maxlen=32),
                setup_score=1,
                center_target_y_mm=0.0,
                observe_samples=1,
                observe_timeout_s=1.0,
                post_act_settle_s=0.0,
                camera_direction_check={},
                plotter=None,
                initial_pre_pose=pre_pose,
                initial_pre_obs_meta={"mode": "trial_full", "reobserved": False},
                y_duration_cal={"curves": []},
                stream_refresh_fn=None,
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        logged_lines = [call.args[0] for call in mock_log_line.call_args_list]
        summary_idx = next(
            idx for idx, line in enumerate(logged_lines) if "[CALIBRATE_Y] Trial 5/60: cmd=D score=1%" in line
        )
        self.assertIn("shared 1% floor", logged_lines[summary_idx])
        detail_idx = next(
            idx
            for idx, line in enumerate(logged_lines)
            if "[CALIBRATE_Y] Trial 5/60: ⚠️  Command inversion detail:" in line
        )
        self.assertLess(summary_idx, detail_idx)

    def test_build_payload_summarizes_observed_brick_distance(self):
        payload = calibrate_y._build_payload(
            config={"brick_distance_source": calibrate_y.BRICK_DISTANCE_SOURCE},
            durations_ms=[250],
            trials=[
                calibrate_y.TrialResult(
                    trial=1,
                    duration_ms=250,
                    cmd="u",
                    score_requested=1,
                    cmd_sent="u",
                    pwm=40,
                    power=0.2,
                    pre_y_mm=1.0,
                    post_y_mm=3.0,
                    raw_delta_mm=2.0,
                    signed_cmd_delta_mm=2.0,
                    cmd_delta_mm=2.0,
                    wrong_way=False,
                    pre_dist_mm=100.0,
                    post_dist_mm=104.0,
                    pre_brick_dist_mm=100.0,
                    post_brick_dist_mm=104.0,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_samples_used=1,
                    post_samples_used=1,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    pre_observation_mode="primary_full",
                    post_observation_mode="primary_full",
                    post_reobserved=False,
                ),
                calibrate_y.TrialResult(
                    trial=1,
                    duration_ms=250,
                    cmd="u",
                    score_requested=1,
                    cmd_sent="u",
                    pwm=40,
                    power=0.2,
                    pre_y_mm=1.0,
                    post_y_mm=2.0,
                    raw_delta_mm=1.0,
                    signed_cmd_delta_mm=1.0,
                    cmd_delta_mm=1.0,
                    wrong_way=False,
                    pre_dist_mm=101.0,
                    post_dist_mm=103.0,
                    pre_brick_dist_mm=101.0,
                    post_brick_dist_mm=103.0,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_samples_used=1,
                    post_samples_used=1,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    pre_observation_mode="primary_full",
                    post_observation_mode="primary_full",
                    post_reobserved=False,
                ),
            ],
            reset_efforts=[
                calibrate_y.ResetEffort(
                    trial=1,
                    reset_act=1,
                    cmd="d",
                    score_requested=1,
                    cmd_sent="d",
                    pwm=40,
                    power=0.2,
                    duration_ms=250,
                    pre_y_mm=10.0,
                    post_y_mm=5.0,
                    raw_delta_mm=-5.0,
                    signed_cmd_delta_mm=5.0,
                    cmd_delta_mm=5.0,
                    wrong_way=False,
                    pre_brick_dist_mm=102.0,
                    post_brick_dist_mm=98.0,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    post_observation_mode="primary_full",
                )
            ],
            status="completed",
            abort_reason=None,
        )
        self.assertEqual(payload["summary"]["trial_count"], 2)
        self.assertEqual(payload["summary"]["repeat_trial_count"], 0)
        self.assertIsNone(payload["summary"]["repeat_median_distance_mm"])
        self.assertEqual(payload["summary"]["brick_distance_min_mm"], 98.0)
        self.assertEqual(payload["summary"]["brick_distance_max_mm"], 104.0)
        self.assertEqual(payload["summary"]["brick_distance_median_mm"], 101.5)

    def test_aggregate_pose_samples_returns_medians(self):
        pose = calibrate_y._aggregate_pose_samples(
            [
                {"offset_y": 5.0, "offset_x": 1.0, "dist": 100.0, "angle": 0.0, "confidence": 90.0, "obs_ts": 1.0, "pose_source": "raw_visible", "lite_required_frames": None},
                {"offset_y": 7.0, "offset_x": 3.0, "dist": 102.0, "angle": 2.0, "confidence": 92.0, "obs_ts": 2.0, "pose_source": "lite_smoothed", "lite_required_frames": 3},
                {"offset_y": 6.0, "offset_x": 2.0, "dist": 101.0, "angle": 1.0, "confidence": 91.0, "obs_ts": 3.0, "pose_source": "lite_smoothed", "lite_required_frames": 3},
            ]
        )
        self.assertAlmostEqual(pose["offset_y"], 6.0)
        self.assertAlmostEqual(pose["offset_x"], 2.0)
        self.assertAlmostEqual(pose["dist"], 101.0)
        self.assertEqual(pose["samples_used"], 3)

    def test_pose_meets_multiframe_requirement_requires_lite_smoothed_pose(self):
        self.assertTrue(
            calibrate_y._pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 3,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            calibrate_y._pose_meets_multiframe_requirement(
                {
                    "pose_source": "raw_visible",
                    "samples_used": 3,
                    "lite_required_frames": None,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            calibrate_y._pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 1,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )

    def test_read_pose_requires_lite_smoothed_frames_and_never_falls_back_to_raw_frame(self):
        world = Mock()
        world.step_state = None
        world.process_rules = None
        vision = Mock()
        vision.read.side_effect = AssertionError("vision.read should not be called")

        with patch.object(calibrate_y, "update_world_from_vision"), patch.object(
            calibrate_y,
            "_lite_pose_from_world",
            return_value=None,
        ), patch.object(
            calibrate_y,
            "_brick_pose_from_world",
        ) as mock_brick_pose:
            pose = calibrate_y.read_pose(
                vision,
                world,
                samples=3,
                timeout_s=0.01,
            )

        self.assertIsNone(pose)
        mock_brick_pose.assert_not_called()

    def test_pose_from_measurement_normalizes_world_and_camera_y_into_one_axis(self):
        world_pose = calibrate_y._pose_from_measurement(
            {
                "visible": True,
                "y_axis": -8.0,
                "offset_x": 0.0,
                "dist": 100.0,
                "angle": 0.0,
                "confidence": 90.0,
            },
            obs_ts=1.0,
            pose_source="brick_state",
            measurement_space="world",
        )
        camera_pose = calibrate_y._pose_from_measurement(
            {
                "visible": True,
                "cam_h": 8.0,
                "offset_x": 0.0,
                "dist": 100.0,
                "angle": 0.0,
                "confidence": 90.0,
            },
            obs_ts=1.0,
            pose_source="raw_visible",
            measurement_space="camera",
        )
        self.assertEqual(world_pose["offset_y"], 8.0)
        self.assertEqual(camera_pose["offset_y"], 8.0)

    def test_camera_direction_from_raw_delta_uses_helper_camera_axis(self):
        self.assertEqual(
            calibrate_y._camera_direction_from_raw_delta(0.81, threshold_mm=0.60),
            "down",
        )
        self.assertEqual(
            calibrate_y._camera_direction_from_raw_delta(-0.81, threshold_mm=0.60),
            "up",
        )

    def test_observe_pose_with_reobserve_rejects_partial_pose_after_primary_failure(self):
        partial_pose = {
            "offset_y": 4.2,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 88.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        with patch.object(calibrate_y, "read_pose", side_effect=[None, partial_pose, partial_pose]), patch.object(
            calibrate_y.time, "sleep"
        ), patch.object(calibrate_y, "log_line"):
            pose, meta = calibrate_y._observe_pose_with_reobserve(
                vision=object(),
                world=object(),
                samples=3,
                timeout_s=1.8,
                min_sample_time=None,
            )
        self.assertIsNone(pose)
        self.assertEqual(meta["mode"], "unavailable")
        self.assertTrue(meta["reobserved"])

    def test_attempt_recovery_reacquires_by_holding_still_before_inverse(self):
        pose = {
            "offset_y": 3.0,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        with patch.object(calibrate_y, "read_pose", return_value=pose) as mock_read_pose, patch.object(
            calibrate_y, "_recover_visibility"
        ) as mock_recover_visibility:
            recovered, meta = calibrate_y._attempt_recovery(
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque([{"cmd": "u", "duration_ms": 300, "score_requested": 1}], maxlen=32),
            )
        self.assertIs(recovered, pose)
        self.assertEqual(meta["mode"], "hold_reobserve")
        mock_read_pose.assert_called_once()
        mock_recover_visibility.assert_not_called()

    def test_recover_pose_for_trial_wraps_recovery_meta(self):
        pose = {
            "offset_y": 1.5,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        with patch.object(
            calibrate_y,
            "_attempt_recovery",
            return_value=(pose, {"mode": "inverse_hold_reobserve_full", "inverse_acts": 2}),
        ):
            recovered, meta = calibrate_y._recover_pose_for_trial(
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque(maxlen=32),
                trial_idx=3,
                trials_requested=4,
                stage_label="before act",
            )
        self.assertIs(recovered, pose)
        self.assertEqual(meta["mode"], "inverse_hold_reobserve_full")
        self.assertEqual(meta["inverse_acts"], 2)
        self.assertTrue(meta["reobserved"])

    def test_ensure_pose_within_trial_band_returns_immediately_when_pose_is_inside_band(self):
        pose = {
            "offset_y": 6.0,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        recent_acts = deque(maxlen=32)
        with patch.object(calibrate_y, "_send_fixed_score_command") as mock_send, patch.object(
            calibrate_y, "_supports_ansi_color", return_value=False
        ):
            positioned, meta = calibrate_y._ensure_pose_within_trial_band(
                initial_pose=pose,
                trial_cmd="d",
                trial_idx=2,
                trials_planned=50,
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=recent_acts,
                setup_score=1,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
            )
        self.assertEqual(positioned, pose)
        self.assertEqual(meta["mode"], "trial_band_within")
        self.assertEqual(meta["setup_acts"], 0)
        mock_send.assert_not_called()

    def test_ensure_pose_within_trial_band_records_reset_effort_for_plot(self):
        start_pose = {
            "offset_y": 18.0,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        end_pose = dict(start_pose)
        end_pose["offset_y"] = -2.0
        end_pose["confidence"] = 92.0
        plotter = Mock()
        reset_efforts = []
        recent_acts = deque(maxlen=32)

        with patch.object(
            calibrate_y,
            "_send_fixed_score_command",
            return_value={"duration_ms": 250, "cmd_sent": "d", "pwm": 40, "power": 0.2},
        ) as mock_send, patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(end_pose, {"mode": "trial_full", "reobserved": False}),
        ), patch.object(calibrate_y, "_supports_ansi_color", return_value=False):
            positioned, meta = calibrate_y._ensure_pose_within_trial_band(
                initial_pose=start_pose,
                trial_cmd="u",
                trial_idx=1,
                trials_planned=50,
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=recent_acts,
                setup_score=1,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
                reset_duration_schedule=deque([250]),
                plotter=plotter,
                reset_efforts=reset_efforts,
            )

        self.assertEqual(positioned["offset_y"], -2.0)
        self.assertEqual(meta["mode"], "trial_band_positioned")
        self.assertEqual(meta["setup_acts"], 1)
        self.assertEqual(mock_send.call_args.kwargs["cmd"], "d")
        plotter.add_point.assert_called_once_with(
            duration_ms=250,
            distance_mm=20.0,
            trial=1,
            cmd="d",
            kind="reset",
            pre_brick_distance_mm=100.0,
            post_brick_distance_mm=100.0,
        )
        self.assertEqual(len(reset_efforts), 1)
        self.assertEqual(reset_efforts[0].cmd, "d")
        self.assertAlmostEqual(reset_efforts[0].cmd_delta_mm, 20.0)
        self.assertAlmostEqual(reset_efforts[0].signed_cmd_delta_mm, 20.0)
        self.assertFalse(reset_efforts[0].wrong_way)
        self.assertAlmostEqual(reset_efforts[0].pre_brick_dist_mm, 100.0)
        self.assertAlmostEqual(reset_efforts[0].post_brick_dist_mm, 100.0)

    def test_recover_visibility_inverts_recent_act(self):
        pose = {
            "offset_y": 2.0,
            "offset_x": 0.0,
            "dist": 100.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        with patch.object(
            calibrate_y,
            "_send_fixed_score_command",
            return_value={"duration_ms": 320},
        ) as mock_send, patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(pose, {"mode": "hold_reobserve_full", "reobserved": True}),
        ):
            recovered, meta = calibrate_y._recover_visibility(
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=deque([{"cmd": "u", "duration_ms": 320, "score_requested": 1}], maxlen=32),
            )
        self.assertIs(recovered, pose)
        self.assertEqual(meta["mode"], "inverse_hold_reobserve_full")
        self.assertEqual(meta["inverse_acts"], 1)
        self.assertEqual(mock_send.call_args.kwargs["cmd"], "d")
        self.assertEqual(mock_send.call_args.kwargs["duration_override_ms"], 320)

    def test_main_uses_prompted_speed_and_duration_after_observing_distance(self):
        payloads = []
        setup_pose = {
            "offset_y": 8.0,
            "offset_x": 0.0,
            "dist": 111.0,
            "angle": 0.0,
            "confidence": 90.0,
            "obs_ts": 1.0,
            "pose_source": "raw_visible",
            "lite_required_frames": None,
            "samples_used": 1,
        }
        trial_row = SimpleNamespace(
            wrong_way=False,
            cmd="u",
            duration_ms=260,
            cmd_delta_mm=0.4,
        )

        with patch.object(sys, "argv", ["helper_calibrate_y.py", "--trials", "1", "--no-livestream"]), patch.object(
            calibrate_y,
            "_ensure_run_dir",
        ), patch.object(
            calibrate_y,
            "shared_prompt_calibration_run_settings",
            return_value={
                "speed_score": 6,
                "min_duration_ms": 260,
                "max_duration_ms": 260,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
            },
        ) as mock_prompt, patch.object(
            calibrate_y,
            "WorldModel",
            return_value=SimpleNamespace(step_state=None, _post_action_observe_delay_s=0.0),
        ), patch.object(
            calibrate_y,
            "Robot",
            return_value=SimpleNamespace(close=lambda: None),
        ), patch.object(
            calibrate_y,
            "YoloBrickDetector",
            return_value=SimpleNamespace(close=lambda: None),
        ), patch.object(
            calibrate_y,
            "LivePlot",
            return_value=SimpleNamespace(finish=lambda: None),
        ), patch.object(
            calibrate_y,
            "_load_y_duration_calibration",
            return_value={"curves": []},
        ), patch.object(
            calibrate_y,
            "_observe_pose_with_reobserve",
            return_value=(setup_pose, {"mode": "primary_full", "reobserved": False}),
        ), patch.object(
            calibrate_y,
            "_run_trial_action",
            return_value=(trial_row, None),
        ) as mock_run_trial_action, patch.object(
            calibrate_y,
            "_build_payload",
            side_effect=lambda **kwargs: {
                "config": kwargs["config"],
                "durations_ms": list(kwargs["durations_ms"]),
            },
        ), patch.object(
            calibrate_y,
            "_write_results",
            side_effect=lambda _path, payload: payloads.append(payload),
        ), patch.object(
            calibrate_y,
            "log_line",
        ):
            exit_code = calibrate_y.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_prompt.call_args.kwargs["observed_distance_mm"], 111.0)
        self.assertEqual(payloads[-1]["config"]["speed_score"], 6)
        self.assertEqual(payloads[-1]["config"]["requested_speed_score"], 6)
        self.assertEqual(payloads[-1]["config"]["speed_score_source"], "prompt")
        self.assertTrue(payloads[-1]["config"]["prompted_speed_score"])
        self.assertEqual(payloads[-1]["config"]["min_duration_ms"], 260)
        self.assertEqual(payloads[-1]["config"]["max_duration_ms"], 260)
        self.assertTrue(payloads[-1]["config"]["prompted_duration_bounds"])
        self.assertEqual(payloads[-1]["durations_ms"], [260])
        self.assertEqual(mock_run_trial_action.call_args.kwargs["setup_score"], 6)
        self.assertEqual(mock_run_trial_action.call_args.kwargs["duration_ms"], 260)


if __name__ == "__main__":
    unittest.main()
