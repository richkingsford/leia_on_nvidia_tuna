import math
import random
import unittest
from collections import deque
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
        self.assertEqual(calibrate_y._plot_color_for_cmd("u", "repeat"), calibrate_y.PLOT_REPEAT_COLOR_BY_CMD["u"])
        self.assertEqual(calibrate_y._plot_color_for_cmd("d", "repeat"), calibrate_y.PLOT_REPEAT_COLOR_BY_CMD["d"])

    def test_plot_series_collapses_setup_and_measured_points_by_command_family(self):
        self.assertEqual(calibrate_y._plot_series_key("u", "trial"), "u")
        self.assertEqual(calibrate_y._plot_series_key("u", "reset"), "u")
        self.assertEqual(calibrate_y._plot_series_key("u", "repeat"), "u:repeat")
        self.assertEqual(calibrate_y._plot_series_label("u", "trial"), "mast_up")
        self.assertEqual(calibrate_y._plot_series_label("u", "reset"), "mast_up")
        self.assertEqual(calibrate_y._plot_series_label("d", "reset"), "mast_down")
        self.assertEqual(calibrate_y._plot_series_label("u", "repeat"), "Repeat up")
        self.assertEqual(calibrate_y._plot_series_label("d", "repeat"), "Repeat down")

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
                    phase="repeat",
                    source_trial=1,
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
        self.assertEqual(payload["summary"]["trial_count"], 1)
        self.assertEqual(payload["summary"]["repeat_trial_count"], 1)
        self.assertEqual(payload["summary"]["repeat_median_distance_mm"], 1.0)
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

    def test_observe_pose_with_reobserve_accepts_partial_pose_after_primary_failure(self):
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
        ):
            pose, meta = calibrate_y._observe_pose_with_reobserve(
                vision=object(),
                world=object(),
                samples=3,
                timeout_s=1.8,
                min_sample_time=None,
            )
        self.assertIs(pose, partial_pose)
        self.assertEqual(meta["mode"], "hold_reobserve_confirmed_partial")
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
            return_value=(end_pose, {"mode": "primary_full", "reobserved": False}),
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


if __name__ == "__main__":
    unittest.main()
