import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from calibration import helper_calibrate_x as calibrate_x
from pathlib import Path


class _MockVision:
    def close(self):
        return None


class _MockRobot:
    def close(self):
        return None


class _MockPlot:
    def finish(self):
        return None


class CalibrateXTests(unittest.TestCase):
    def test_default_vision_mode_matches_cyan_runtime(self):
        self.assertEqual(calibrate_x.DEFAULT_VISION_MODE, "yolo")

    def test_run_dir_for_vision_tracks_backend_family(self):
        self.assertEqual(calibrate_x._run_dir_for_vision("yolo"), Path("Runs - cyan"))
        self.assertEqual(calibrate_x._run_dir_for_vision("leia"), Path("Runs - cyan"))
        self.assertEqual(calibrate_x._run_dir_for_vision("aruco"), Path("Runs - aruco"))

    def test_x_axis_trial_speed_profile_is_available_in_robot_model(self):
        profile = calibrate_x.shared_load_calibration_trial_speed_profile("x_axis")
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile["axis"], "x_axis")
        self.assertEqual(
            profile["curve_points"],
            [
                {"distance_mm": 105.0, "speed_score": 1},
                {"distance_mm": 180.0, "speed_score": 20},
            ],
        )

    def test_trial_speed_profile_for_fixed_mode_disables_distance_curve(self):
        self.assertIsNone(calibrate_x._trial_speed_profile_for_mode("fixed"))
        self.assertIsInstance(calibrate_x._trial_speed_profile_for_mode("distance_curve"), dict)

    def test_plot_title_text_uses_single_line_distance_title(self):
        self.assertEqual(calibrate_x._plot_title_text([]), "X Calibration at 166mm")
        self.assertEqual(calibrate_x._plot_title_text([124.8, 125.2, 125.0]), "X Calibration at 125mm")

    def test_movement_metrics_reports_total_distance_even_when_wrong_way(self):
        metrics = calibrate_x._movement_metrics("l", 2.0, 1.25)
        self.assertAlmostEqual(metrics["raw_delta_mm"], -0.75)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], -0.75)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 0.75)
        self.assertTrue(metrics["wrong_way"])

    def test_x_axis_positive_motion_defaults_to_l(self):
        self.assertEqual(calibrate_x._x_cmd_for_positive_motion(), "l")
        self.assertEqual(calibrate_x._x_cmd_for_negative_motion(), "r")

    def test_wrong_way_reason_mentions_center_and_tiny_motion_when_applicable(self):
        reason = calibrate_x._wrong_way_reason_text(
            pre_x_mm=0.20,
            post_x_mm=0.05,
            target_x_mm=0.0,
        )
        self.assertIn("near 0", reason)
        self.assertIn("tiny", reason)

    def test_auto_cmd_for_x_preserves_visibility_rule(self):
        self.assertEqual(calibrate_x._auto_cmd_for_x(-0.1, center_x_mm=0.0), "r")
        self.assertEqual(calibrate_x._auto_cmd_for_x(0.0, center_x_mm=0.0), "l")
        self.assertEqual(calibrate_x._auto_cmd_for_x(0.1, center_x_mm=0.0), "l")

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
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
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
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 200, "cmd": "auto"},
            ],
        )

    def test_build_trial_plan_spreads_initial_trials_across_durations(self):
        plan = calibrate_x._build_trial_plan(
            durations_ms=[200, 210, 220, 230],
            trials=4,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
                {"duration_ms": 230, "cmd": "auto"},
            ],
        )

    def test_build_trial_plan_balances_directions_per_duration_across_rounds(self):
        plan = calibrate_x._build_trial_plan(
            durations_ms=[200, 210, 220],
            trials=9,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 210, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
            ],
        )

    def test_build_trial_plan_spreads_sparse_durations_across_both_directions(self):
        plan = calibrate_x._build_trial_plan(
            durations_ms=[200, 220, 240, 260, 280, 300],
            trials=6,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
                {"duration_ms": 240, "cmd": "auto"},
                {"duration_ms": 260, "cmd": "auto"},
                {"duration_ms": 280, "cmd": "auto"},
                {"duration_ms": 300, "cmd": "auto"},
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

    def test_pose_meets_multiframe_requirement_requires_lite_smoothed_pose(self):
        self.assertTrue(
            calibrate_x._pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 3,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            calibrate_x._pose_meets_multiframe_requirement(
                {
                    "pose_source": "raw_visible",
                    "samples_used": 3,
                    "lite_required_frames": None,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            calibrate_x._pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 1,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )

    def test_read_pose_requires_lite_smoothed_frames_and_never_falls_back_to_raw_frame(self):
        world = SimpleNamespace(step_state=None)
        vision = SimpleNamespace(read=lambda: (_ for _ in ()).throw(AssertionError("vision.read should not be called")))

        with patch.object(calibrate_x, "update_world_from_vision"), patch.object(
            calibrate_x,
            "_lite_pose_from_world",
            return_value=None,
        ), patch.object(
            calibrate_x,
            "_brick_pose_from_world",
        ) as mock_brick_pose:
            pose = calibrate_x.read_pose(
                vision,
                world,
                samples=3,
                timeout_s=0.01,
            )

        self.assertIsNone(pose)
        mock_brick_pose.assert_not_called()

    def test_observe_pose_with_reobserve_rejects_single_frame_rescue_pose(self):
        partial_pose = {
            "pose_source": "raw_visible",
            "samples_used": 1,
            "lite_required_frames": None,
        }
        with patch.object(
            calibrate_x,
            "read_pose",
            side_effect=[None, partial_pose, partial_pose],
        ), patch.object(
            calibrate_x.time,
            "sleep",
        ), patch.object(
            calibrate_x,
            "log_line",
        ):
            pose, meta = calibrate_x._observe_pose_with_reobserve(
                vision=object(),
                world=SimpleNamespace(step_state=None),
                samples=3,
                timeout_s=0.1,
            )

        self.assertIsNone(pose)
        self.assertEqual(meta["mode"], "unavailable")

    def test_send_fixed_score_command_disables_first_turn_halving(self):
        with patch.object(
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

    def test_planned_action_meta_uses_same_central_1pct_turn_curve_as_runtime(self):
        right_meta = calibrate_x._planned_action_meta("r", 1, 65)
        left_meta = calibrate_x._planned_action_meta("l", 1, 135)

        self.assertEqual(
            (right_meta["pwm"], round(right_meta["power"], 3), right_meta["score_model"], right_meta["duration_ms"]),
            (102, 0.301, 1, 65),
        )
        self.assertEqual(
            (left_meta["pwm"], round(left_meta["power"], 3), left_meta["score_model"], left_meta["duration_ms"]),
            (102, 0.301, 1, 135),
        )

    def test_predict_movement_from_curve_uses_linear_fit(self):
        predicted_mm, curve_source = calibrate_x._predict_movement_from_curve(
            cmd="l",
            duration_ms=250,
            x_calibration={
                "source": "curve_a",
                "reference_distance_mm": 122.0,
                "speed_score_pct": 1,
                "by_cmd": {
                    "l": {
                        "slope_mm_per_ms": 0.02,
                        "intercept_mm": 1.5,
                    }
                },
            },
        )
        self.assertAlmostEqual(predicted_mm, 6.5)
        self.assertEqual(curve_source, "curve_a at 122mm distance at 1% speed")

    def test_calculate_prediction_comparison_reports_difference_and_closeness(self):
        comparison = calibrate_x._calculate_prediction_comparison(
            actual_distance_mm=8.0,
            predicted_distance_mm=10.0,
            curve_source="curve_a",
        )
        self.assertEqual(comparison["curve_source"], "curve_a")
        self.assertAlmostEqual(comparison["absolute_difference_mm"], 2.0)
        self.assertAlmostEqual(comparison["prediction_closeness_percentage"], 80.0)

    def test_x_curve_display_name_reports_no_curve_when_missing(self):
        self.assertEqual(calibrate_x._x_curve_display_name(None), "no_curve")

    def test_run_trial_action_uses_effective_score_from_shared_profile_resolution(self):
        pre_pose = {
            "offset_x": 25.0,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        post_pose = {
            "offset_x": 45.0,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        with patch.object(
            calibrate_x,
            "_observe_pose_with_reobserve",
            side_effect=[
                (pre_pose, {"mode": "primary_full", "reobserved": False}),
                (post_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(
            calibrate_x,
            "shared_resolve_calibration_trial_speed_score",
            return_value=(7, {"source": "distance_curve"}),
        ), patch.object(
            calibrate_x,
            "_send_fixed_score_command",
            return_value={"cmd_sent": "l", "duration_ms": 250},
        ) as mock_send, patch.object(
            calibrate_x,
            "log_line",
        ):
            row, abort_reason = calibrate_x._run_trial_action(
                trial_idx=1,
                trials_planned=1,
                trial_label="Trial 1/1",
                cmd="l",
                duration_ms=250,
                phase="primary",
                source_trial=1,
                action_step="CALIBRATE_X",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=[],
                setup_score=1,
                center_target_x_mm=0.0,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
            )
        self.assertIsNone(abort_reason)
        self.assertEqual(mock_send.call_args.kwargs["score"], 7)
        self.assertEqual(row.score_requested, 7)

    def test_run_trial_action_log_includes_pwm_power_and_duration_next_to_score(self):
        pre_pose = {
            "offset_x": 3.0,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        post_pose = {
            "offset_x": 5.0,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        with patch.object(
            calibrate_x,
            "_observe_pose_with_reobserve",
            side_effect=[
                (pre_pose, {"mode": "primary_full", "reobserved": False}),
                (post_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(
            calibrate_x,
            "shared_resolve_calibration_trial_speed_score",
            return_value=(1, {"source": "distance_curve"}),
        ), patch.object(
            calibrate_x,
            "_send_fixed_score_command",
            return_value={
                "cmd_sent": "l",
                "duration_ms": 250,
                "pwm": 102,
                "power": 0.3013698630136986,
            },
        ), patch.object(
            calibrate_x,
            "log_line",
        ) as mock_log_line:
            row, abort_reason = calibrate_x._run_trial_action(
                trial_idx=1,
                trials_planned=1,
                trial_label="Trial 1/1",
                cmd="l",
                duration_ms=250,
                phase="primary",
                source_trial=1,
                action_step="CALIBRATE_X",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=[],
                setup_score=1,
                center_target_x_mm=0.0,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        logged_lines = [call.args[0] for call in mock_log_line.call_args_list]
        self.assertTrue(
            any(
                "[CALIBRATE_X] Trial 1/1: cmd=L score=1% (pwm=102, pwr=0.301, t=250ms; shared 1% floor" in line
                for line in logged_lines
            )
        )

    def test_main_disables_repeat_pass_by_default(self):
        payloads = []
        setup_pose = {
            "offset_x": 0.5,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        trial_row = calibrate_x.TrialResult(
            trial=1,
            duration_ms=250,
            cmd="r",
            score_requested=1,
            cmd_sent="l",
            pwm=40,
            power=0.2,
            pre_x_mm=0.5,
            post_x_mm=0.1,
            raw_delta_mm=-0.4,
            signed_cmd_delta_mm=0.4,
            cmd_delta_mm=0.4,
            wrong_way=False,
            pre_dist_mm=105.0,
            post_dist_mm=105.0,
            pre_brick_dist_mm=105.0,
            post_brick_dist_mm=105.0,
            pre_confidence=0.9,
            post_confidence=0.9,
            pre_samples_used=3,
            post_samples_used=3,
            pre_pose_source="lite_smoothed",
            post_pose_source="lite_smoothed",
            pre_observation_mode="primary_full",
            post_observation_mode="primary_full",
            post_reobserved=False,
            phase="primary",
            source_trial=1,
        )
        with patch.object(sys, "argv", ["helper_calibrate_x.py", "--trials", "1", "--no-livestream"]), patch.object(
            calibrate_x,
            "_ensure_run_dir",
        ), patch.object(
            calibrate_x,
            "shared_prompt_calibration_run_settings",
            return_value={
                "speed_score": 7,
                "min_duration_ms": 150,
                "max_duration_ms": 170,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
            },
        ) as mock_prompt, patch.object(
            calibrate_x,
            "WorldModel",
            return_value=SimpleNamespace(step_state=None, _post_action_observe_delay_s=0.0),
        ), patch.object(
            calibrate_x,
            "Robot",
            return_value=_MockRobot(),
        ), patch.object(
            calibrate_x,
            "YoloBrickDetector",
            return_value=_MockVision(),
        ), patch.object(
            calibrate_x,
            "LivePlot",
            return_value=_MockPlot(),
        ), patch.object(
            calibrate_x,
            "_observe_pose_with_reobserve",
            return_value=(setup_pose, {"mode": "primary_full", "reobserved": False}),
        ), patch.object(
            calibrate_x,
            "_run_trial_action",
            return_value=(trial_row, None),
        ) as mock_run_trial_action, patch.object(
            calibrate_x,
            "_write_results",
            side_effect=lambda _path, payload: payloads.append(payload),
        ), patch.object(
            calibrate_x,
            "log_line",
        ):
            exit_code = calibrate_x.main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run_trial_action.call_count, 1)
        self.assertEqual(payloads[-1]["config"]["repeat_pass_enabled"], False)
        self.assertEqual(payloads[-1]["config"]["speed_score"], 7)
        self.assertEqual(payloads[-1]["config"]["requested_speed_score"], 7)
        self.assertEqual(payloads[-1]["config"]["speed_score_source"], "prompt")
        self.assertTrue(payloads[-1]["config"]["prompted_speed_score"])
        self.assertEqual(payloads[-1]["config"]["min_duration_ms"], 150)
        self.assertEqual(payloads[-1]["config"]["max_duration_ms"], 170)
        self.assertTrue(payloads[-1]["config"]["prompted_duration_bounds"])
        self.assertEqual(payloads[-1]["config"]["trial_cmd_mode"], "auto")
        self.assertEqual(mock_run_trial_action.call_args.kwargs["cmd"], "l")
        self.assertEqual(mock_run_trial_action.call_args.kwargs["setup_score"], 7)
        self.assertIsNotNone(mock_run_trial_action.call_args.kwargs["initial_pre_pose"])
        self.assertEqual(mock_prompt.call_args.kwargs["observed_distance_mm"], 105.0)

    def test_main_only_runs_repeat_pass_when_requested(self):
        payloads = []
        setup_pose = {
            "offset_x": 0.5,
            "dist": 105.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        primary_row = calibrate_x.TrialResult(
            trial=1,
            duration_ms=250,
            cmd="r",
            score_requested=1,
            cmd_sent="l",
            pwm=40,
            power=0.2,
            pre_x_mm=0.5,
            post_x_mm=0.1,
            raw_delta_mm=-0.4,
            signed_cmd_delta_mm=0.4,
            cmd_delta_mm=0.4,
            wrong_way=False,
            pre_dist_mm=105.0,
            post_dist_mm=105.0,
            pre_brick_dist_mm=105.0,
            post_brick_dist_mm=105.0,
            pre_confidence=0.9,
            post_confidence=0.9,
            pre_samples_used=3,
            post_samples_used=3,
            pre_pose_source="lite_smoothed",
            post_pose_source="lite_smoothed",
            pre_observation_mode="primary_full",
            post_observation_mode="primary_full",
            post_reobserved=False,
            phase="primary",
            source_trial=1,
        )
        repeat_row = calibrate_x.TrialResult(
            trial=1,
            duration_ms=250,
            cmd="r",
            score_requested=1,
            cmd_sent="l",
            pwm=40,
            power=0.2,
            pre_x_mm=0.2,
            post_x_mm=0.0,
            raw_delta_mm=-0.2,
            signed_cmd_delta_mm=0.2,
            cmd_delta_mm=0.2,
            wrong_way=False,
            pre_dist_mm=105.0,
            post_dist_mm=105.0,
            pre_brick_dist_mm=105.0,
            post_brick_dist_mm=105.0,
            pre_confidence=0.9,
            post_confidence=0.9,
            pre_samples_used=3,
            post_samples_used=3,
            pre_pose_source="lite_smoothed",
            post_pose_source="lite_smoothed",
            pre_observation_mode="primary_full",
            post_observation_mode="primary_full",
            post_reobserved=False,
            phase="repeat",
            source_trial=1,
        )
        with patch.object(sys, "argv", ["helper_calibrate_x.py", "--trials", "1", "--repeat-trials", "1", "--no-livestream"]), patch.object(
            calibrate_x,
            "_ensure_run_dir",
        ), patch.object(
            calibrate_x,
            "shared_prompt_calibration_run_settings",
            return_value={
                "speed_score": 3,
                "min_duration_ms": 200,
                "max_duration_ms": 1400,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
            },
        ), patch.object(
            calibrate_x,
            "WorldModel",
            return_value=SimpleNamespace(step_state=None, _post_action_observe_delay_s=0.0),
        ), patch.object(
            calibrate_x,
            "Robot",
            return_value=_MockRobot(),
        ), patch.object(
            calibrate_x,
            "YoloBrickDetector",
            return_value=_MockVision(),
        ), patch.object(
            calibrate_x,
            "LivePlot",
            return_value=_MockPlot(),
        ), patch.object(
            calibrate_x,
            "_observe_pose_with_reobserve",
            return_value=(setup_pose, {"mode": "primary_full", "reobserved": False}),
        ), patch.object(
            calibrate_x,
            "_run_trial_action",
            side_effect=[(primary_row, None), (repeat_row, None)],
        ) as mock_run_trial_action, patch.object(
            calibrate_x,
            "_write_results",
            side_effect=lambda _path, payload: payloads.append(payload),
        ), patch.object(
            calibrate_x,
            "log_line",
        ):
            exit_code = calibrate_x.main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run_trial_action.call_count, 2)
        self.assertEqual(payloads[-1]["config"]["repeat_pass_enabled"], True)

    def test_main_fixed_trial_speed_mode_uses_prompted_score(self):
        payloads = []
        setup_pose = {
            "offset_x": 0.5,
            "dist": 150.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        trial_row = calibrate_x.TrialResult(
            trial=1,
            duration_ms=250,
            cmd="r",
            score_requested=20,
            cmd_sent="l",
            pwm=40,
            power=0.2,
            pre_x_mm=0.5,
            post_x_mm=0.1,
            raw_delta_mm=-0.4,
            signed_cmd_delta_mm=0.4,
            cmd_delta_mm=0.4,
            wrong_way=False,
            pre_dist_mm=150.0,
            post_dist_mm=150.0,
            pre_brick_dist_mm=150.0,
            post_brick_dist_mm=150.0,
            pre_confidence=0.9,
            post_confidence=0.9,
            pre_samples_used=3,
            post_samples_used=3,
            pre_pose_source="lite_smoothed",
            post_pose_source="lite_smoothed",
            pre_observation_mode="primary_full",
            post_observation_mode="primary_full",
            post_reobserved=False,
            phase="primary",
            source_trial=1,
        )
        with patch.object(
            sys,
            "argv",
            [
                "helper_calibrate_x.py",
                "--trials",
                "1",
                "--trial-speed-mode",
                "fixed",
                "--speed-score",
                "20",
                "--no-livestream",
            ],
        ), patch.object(
            calibrate_x,
            "_ensure_run_dir",
        ), patch.object(
            calibrate_x,
            "shared_prompt_calibration_run_settings",
            return_value={
                "speed_score": 9,
                "min_duration_ms": 200,
                "max_duration_ms": 1400,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
            },
        ), patch.object(
            calibrate_x,
            "WorldModel",
            return_value=SimpleNamespace(step_state=None, _post_action_observe_delay_s=0.0),
        ), patch.object(
            calibrate_x,
            "Robot",
            return_value=_MockRobot(),
        ), patch.object(
            calibrate_x,
            "YoloBrickDetector",
            return_value=_MockVision(),
        ), patch.object(
            calibrate_x,
            "LivePlot",
            return_value=_MockPlot(),
        ), patch.object(
            calibrate_x,
            "_observe_pose_with_reobserve",
            return_value=(setup_pose, {"mode": "primary_full", "reobserved": False}),
        ), patch.object(
            calibrate_x,
            "_run_trial_action",
            return_value=(trial_row, None),
        ) as mock_run_trial_action, patch.object(
            calibrate_x,
            "_write_results",
            side_effect=lambda _path, payload: payloads.append(payload),
        ), patch.object(
            calibrate_x,
            "log_line",
        ):
            exit_code = calibrate_x.main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(payloads[-1]["config"]["trial_speed_mode"], "fixed")
        self.assertEqual(payloads[-1]["config"]["trial_speed_score_source"], "arg")
        self.assertEqual(payloads[-1]["config"]["speed_score"], 9)
        self.assertEqual(payloads[-1]["config"]["speed_score_source"], "prompt")
        self.assertNotIn("trial_speed_score_profile", payloads[-1]["config"])
        self.assertIsNone(mock_run_trial_action.call_args.kwargs["trial_speed_profile"])
        self.assertEqual(mock_run_trial_action.call_args.kwargs["setup_score"], 9)


if __name__ == "__main__":
    unittest.main()
