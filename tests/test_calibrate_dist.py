import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calibration import helper_calibrate_dist as calibrate_dist


class CalibrateDistTests(unittest.TestCase):
    def test_plot_title_text_uses_single_line_distance_title(self):
        self.assertEqual(calibrate_dist._plot_title_text([]), "Distance Calibration at 166mm")
        self.assertEqual(
            calibrate_dist._plot_title_text([164.8, 165.2, 165.0]),
            "Distance Calibration at 165mm",
        )

    def test_distance_positive_motion_defaults_to_backward(self):
        self.assertEqual(calibrate_dist._dist_cmd_for_positive_motion(), "b")
        self.assertEqual(calibrate_dist._dist_cmd_for_negative_motion(), "f")

    def test_auto_cmd_for_dist_uses_target_distance(self):
        self.assertEqual(calibrate_dist._auto_cmd_for_dist(166.0, target_dist_mm=160.0), "f")
        self.assertEqual(calibrate_dist._auto_cmd_for_dist(154.0, target_dist_mm=160.0), "b")
        self.assertEqual(calibrate_dist._auto_cmd_for_dist(160.0, target_dist_mm=160.0), "b")

    def test_command_delta_is_positive_in_command_direction(self):
        self.assertAlmostEqual(calibrate_dist._command_delta_mm("f", 166.0, 160.0), 6.0)
        self.assertAlmostEqual(calibrate_dist._command_delta_mm("b", 160.0, 166.0), 6.0)

    def test_movement_metrics_reports_total_distance_even_when_wrong_way(self):
        metrics = calibrate_dist._movement_metrics("f", 160.0, 166.0)
        self.assertAlmostEqual(metrics["raw_delta_mm"], 6.0)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], -6.0)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 6.0)
        self.assertTrue(metrics["wrong_way"])
        self.assertFalse(metrics["no_meaningful_movement"])

    def test_movement_metrics_below_threshold_becomes_no_change_sample(self):
        metrics = calibrate_dist._movement_metrics("b", 103.67, 103.58, threshold_mm=0.10)
        self.assertAlmostEqual(metrics["raw_delta_mm"], -0.09, places=2)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], 0.0)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 0.0)
        self.assertFalse(metrics["wrong_way"])
        self.assertTrue(metrics["no_meaningful_movement"])

    def test_build_duration_schedule_uses_helper_step(self):
        durations = calibrate_dist._build_duration_schedule(
            trials=None,
            min_duration_ms=200,
            max_duration_ms=260,
            duration_step_ms=20,
        )
        self.assertEqual(durations, [200, 220, 240, 260])

    def test_planned_action_meta_uses_central_speed_curve(self):
        power, pwm, score_used, duration_ms = calibrate_dist.speed_power_pwm_for_cmd("f", 1)
        meta = calibrate_dist._planned_action_meta("f", 1, duration_ms)

        self.assertEqual(meta["pwm"], pwm)
        self.assertAlmostEqual(meta["power"], power)
        self.assertEqual(meta["score_model"], score_used)
        self.assertEqual(meta["duration_ms"], duration_ms)

    def test_build_trial_plan_uses_auto_target_mode(self):
        plan = calibrate_dist._build_trial_plan(
            durations_ms=[200, 220],
            trials=None,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "auto"},
                {"duration_ms": 220, "cmd": "auto"},
            ],
        )

    def test_predict_movement_from_curve_uses_linear_fit(self):
        predicted_mm, curve_source = calibrate_dist._predict_movement_from_curve(
            cmd="f",
            duration_ms=250,
            dist_calibration={
                "source": "curve_a",
                "reference_distance_mm": 122.0,
                "speed_score_pct": 1,
                "by_cmd": {
                    "f": {
                        "slope_mm_per_ms": 0.02,
                        "intercept_mm": 1.5,
                    }
                },
            },
        )
        self.assertAlmostEqual(predicted_mm, 6.5)
        self.assertEqual(curve_source, "curve_a at 122mm distance at 1% speed")

    def test_dist_curve_display_name_reports_no_curve_when_missing(self):
        self.assertEqual(calibrate_dist._dist_curve_display_name(None), "no_curve")

    def test_run_trial_action_uses_effective_score_from_shared_profile_resolution(self):
        pre_pose = {
            "dist": 125.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        post_pose = {
            "dist": 130.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        with patch.object(
            calibrate_dist,
            "_observe_pose_with_reobserve",
            side_effect=[
                (pre_pose, {"mode": "primary_full", "reobserved": False}),
                (post_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(
            calibrate_dist,
            "shared_resolve_calibration_trial_speed_score",
            return_value=(7, {"source": "distance_curve"}),
        ), patch.object(
            calibrate_dist,
            "_send_fixed_score_command",
            return_value={"cmd_sent": "b", "duration_ms": 250},
        ) as mock_send, patch.object(
            calibrate_dist,
            "log_line",
        ):
            row, abort_reason = calibrate_dist._run_trial_action(
                trial_idx=1,
                trials_planned=1,
                trial_label="Trial 1/1",
                cmd="b",
                duration_ms=250,
                phase="primary",
                source_trial=1,
                action_step="CALIBRATE_DIST",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=[],
                setup_score=1,
                target_dist_mm=120.0,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
            )
        self.assertIsNone(abort_reason)
        self.assertEqual(mock_send.call_args.kwargs["score"], 7)
        self.assertEqual(row.score_requested, 7)

    def test_run_trial_action_log_includes_pwm_power_duration_and_prediction_fields(self):
        pre_pose = {
            "dist": 125.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        post_pose = {
            "dist": 130.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        with patch.object(
            calibrate_dist,
            "_observe_pose_with_reobserve",
            side_effect=[
                (pre_pose, {"mode": "primary_full", "reobserved": False}),
                (post_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(
            calibrate_dist,
            "shared_resolve_calibration_trial_speed_score",
            return_value=(1, {"source": "distance_curve"}),
        ), patch.object(
            calibrate_dist,
            "_send_fixed_score_command",
            return_value={
                "cmd_sent": "b",
                "duration_ms": 250,
                "pwm": 102,
                "power": 0.3013698630136986,
            },
        ), patch.object(
            calibrate_dist,
            "log_line",
        ) as mock_log_line:
            row, abort_reason = calibrate_dist._run_trial_action(
                trial_idx=1,
                trials_planned=1,
                trial_label="Trial 1/1",
                cmd="b",
                duration_ms=250,
                phase="primary",
                source_trial=1,
                action_step="CALIBRATE_DIST",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=[],
                setup_score=1,
                target_dist_mm=120.0,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
                dist_duration_cal={
                    "source": "curve_a",
                    "reference_distance_mm": 122.0,
                    "speed_score_pct": 1,
                    "by_cmd": {
                        "b": {
                            "slope_mm_per_ms": 0.02,
                            "intercept_mm": 1.5,
                        }
                    },
                },
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        logged_lines = [call.args[0] for call in mock_log_line.call_args_list]
        self.assertTrue(
            any(
                "[CALIBRATE_DIST] Trial 1/1: cmd=B score=1% (pwm=102, pwr=0.301, t=250ms; shared 1% floor" in line
                and "predicted=6.50mm curve_source=curve_a at 122mm distance at 1% speed" in line
                for line in logged_lines
            )
        )

    def test_build_payload_summarizes_trials(self):
        payload = calibrate_dist._build_payload(
            config={"brick_distance_source": calibrate_dist.BRICK_DISTANCE_SOURCE},
            durations_ms=[200],
            trials=[
                calibrate_dist.TrialResult(
                    trial=1,
                    duration_ms=200,
                    cmd="f",
                    score_requested=5,
                    cmd_sent="f",
                    pwm=40,
                    power=0.2,
                    pre_dist_mm=166.0,
                    post_dist_mm=160.0,
                    raw_delta_mm=-6.0,
                    signed_cmd_delta_mm=6.0,
                    cmd_delta_mm=6.0,
                    wrong_way=False,
                    pre_brick_dist_mm=166.0,
                    post_brick_dist_mm=160.0,
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
                calibrate_dist.TrialResult(
                    trial=1,
                    duration_ms=200,
                    cmd="f",
                    score_requested=5,
                    cmd_sent="f",
                    pwm=40,
                    power=0.2,
                    pre_dist_mm=166.0,
                    post_dist_mm=161.0,
                    raw_delta_mm=-5.0,
                    signed_cmd_delta_mm=5.0,
                    cmd_delta_mm=5.0,
                    wrong_way=False,
                    pre_brick_dist_mm=166.0,
                    post_brick_dist_mm=161.0,
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
            status="completed",
            abort_reason=None,
        )
        self.assertEqual(payload["summary"]["trial_count"], 1)
        self.assertEqual(payload["summary"]["repeat_trial_count"], 1)
        self.assertEqual(payload["summary"]["median_distance_mm"], 6.0)
        self.assertEqual(payload["summary"]["repeat_median_distance_mm"], 5.0)
        self.assertEqual(payload["summary"]["brick_distance_min_mm"], 160.0)
        self.assertEqual(payload["summary"]["brick_distance_max_mm"], 166.0)
        self.assertEqual(payload["distance_direction_check"], {})
        self.assertEqual(payload["fast_alignment_analysis"], {})

    def test_distance_direction_check_flags_bidirectional_inversion(self):
        check = calibrate_dist._new_distance_direction_check_state()
        calibrate_dist._record_distance_direction_check(
            check,
            trial_label="Trial 1/2",
            cmd="f",
            cmd_sent="b",
            raw_delta_mm=1.2,
        )
        calibrate_dist._record_distance_direction_check(
            check,
            trial_label="Trial 2/2",
            cmd="b",
            cmd_sent="f",
            raw_delta_mm=-1.4,
        )

        self.assertEqual(check["by_cmd"]["f"]["status"], "mismatch")
        self.assertEqual(check["by_cmd"]["b"]["status"], "mismatch")
        self.assertEqual(calibrate_dist._inferred_distance_positive_cmd(check), "f")
        self.assertTrue(calibrate_dist._should_abort_for_distance_inversion(check))
        summary = calibrate_dist._distance_direction_check_summary_line(check)
        self.assertIn("suspected_drive_inversion", summary)

    def test_distance_direction_check_records_no_change_sample_for_small_delta(self):
        check = calibrate_dist._new_distance_direction_check_state()
        with patch.object(calibrate_dist, "log_line") as mock_log_line:
            calibrate_dist._record_distance_direction_check(
                check,
                trial_label="Trial 6/6",
                cmd="b",
                cmd_sent="f",
                raw_delta_mm=-0.09,
            )

        entry = check["by_cmd"]["b"]
        self.assertEqual(entry["status"], "no_movement")
        self.assertEqual(entry["evidence_count"], 0)
        self.assertEqual(entry["match_count"], 0)
        self.assertEqual(entry["mismatch_count"], 0)
        self.assertEqual(entry["no_movement_count"], 1)
        self.assertEqual(entry["inconclusive_count"], 1)
        summary = calibrate_dist._distance_direction_check_summary_line(check)
        self.assertIn("observed=no_change", summary)
        self.assertIn("no_change=1", summary)
        self.assertFalse(calibrate_dist._should_abort_for_distance_inversion(check))
        logged_lines = [call.args[0] for call in mock_log_line.call_args_list]
        self.assertTrue(any("recording a no-change sample" in line for line in logged_lines))

    def test_log_distance_command_inversion_detail_uses_no_change_wording_below_threshold(self):
        with patch.object(calibrate_dist, "log_line") as mock_log_line:
            calibrate_dist._log_distance_command_inversion_detail(
                prefix="[CALIBRATE_DIST] Note:",
                trial_label="Trial 6/6",
                logical_cmd="b",
                wire_cmd="f",
                raw_delta_mm=-0.09,
                threshold_mm=0.10,
            )

        message = mock_log_line.call_args.args[0]
        self.assertIn("observed no meaningful distance change", message)
        self.assertIn("Recording this as a no-change sample", message)
        self.assertIn("wire forward still implies decrease", message)

    def test_fast_alignment_analysis_prefers_best_valid_duration_within_budget(self):
        analysis = calibrate_dist._fast_alignment_analysis(
            [
                calibrate_dist.TrialResult(
                    trial=1,
                    duration_ms=200,
                    cmd="f",
                    score_requested=5,
                    cmd_sent="f",
                    pwm=40,
                    power=0.2,
                    pre_dist_mm=166.0,
                    post_dist_mm=164.0,
                    raw_delta_mm=-2.0,
                    signed_cmd_delta_mm=2.0,
                    cmd_delta_mm=2.0,
                    wrong_way=False,
                    pre_brick_dist_mm=166.0,
                    post_brick_dist_mm=164.0,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_samples_used=1,
                    post_samples_used=1,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    pre_observation_mode="primary_full",
                    post_observation_mode="primary_full",
                    post_reobserved=False,
                    total_trial_elapsed_s=1.0,
                    signed_effective_mm_per_s=2.0,
                ),
                calibrate_dist.TrialResult(
                    trial=2,
                    duration_ms=300,
                    cmd="f",
                    score_requested=5,
                    cmd_sent="f",
                    pwm=40,
                    power=0.2,
                    pre_dist_mm=166.0,
                    post_dist_mm=162.5,
                    raw_delta_mm=-3.5,
                    signed_cmd_delta_mm=3.5,
                    cmd_delta_mm=3.5,
                    wrong_way=False,
                    pre_brick_dist_mm=166.0,
                    post_brick_dist_mm=162.5,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_samples_used=1,
                    post_samples_used=1,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    pre_observation_mode="primary_full",
                    post_observation_mode="primary_full",
                    post_reobserved=False,
                    total_trial_elapsed_s=1.2,
                    signed_effective_mm_per_s=(3.5 / 1.2),
                ),
                calibrate_dist.TrialResult(
                    trial=3,
                    duration_ms=400,
                    cmd="f",
                    score_requested=5,
                    cmd_sent="f",
                    pwm=40,
                    power=0.2,
                    pre_dist_mm=166.0,
                    post_dist_mm=161.0,
                    raw_delta_mm=-5.0,
                    signed_cmd_delta_mm=5.0,
                    cmd_delta_mm=5.0,
                    wrong_way=False,
                    pre_brick_dist_mm=166.0,
                    post_brick_dist_mm=161.0,
                    pre_confidence=90.0,
                    post_confidence=91.0,
                    pre_samples_used=1,
                    post_samples_used=1,
                    pre_pose_source="raw_visible",
                    post_pose_source="raw_visible",
                    pre_observation_mode="primary_full",
                    post_observation_mode="primary_full",
                    post_reobserved=False,
                    total_trial_elapsed_s=1.8,
                    signed_effective_mm_per_s=(5.0 / 1.8),
                ),
            ],
            target_budget_s=1.5,
        )

        self.assertEqual(analysis["recommended_duration_ms"], 300)
        self.assertEqual(analysis["recommended_reason"], "best_valid_mm_per_s_within_budget")

    def test_run_trial_action_below_threshold_preserves_zero_effect_sample(self):
        pre_pose = {
            "dist": 103.67,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        post_pose = {
            "dist": 103.58,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        with patch.object(
            calibrate_dist,
            "_observe_pose_with_reobserve",
            side_effect=[
                (pre_pose, {"mode": "primary_full", "reobserved": False}),
                (post_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(
            calibrate_dist,
            "shared_resolve_calibration_trial_speed_score",
            return_value=(1, {"source": "distance_curve"}),
        ), patch.object(
            calibrate_dist,
            "_send_fixed_score_command",
            return_value={
                "cmd_sent": "f",
                "duration_ms": 255,
                "pwm": 106,
                "power": 0.320,
            },
        ), patch.object(
            calibrate_dist,
            "log_line",
        ) as mock_log_line:
            row, abort_reason = calibrate_dist._run_trial_action(
                trial_idx=6,
                trials_planned=6,
                trial_label="Trial 6/6",
                cmd="b",
                duration_ms=255,
                phase="primary",
                source_trial=6,
                action_step="CALIBRATE_DIST",
                plot_kind="trial",
                vision=object(),
                world=object(),
                robot=object(),
                recent_acts=[],
                setup_score=1,
                target_dist_mm=120.0,
                observe_samples=3,
                observe_timeout_s=1.8,
                post_act_settle_s=0.1,
                distance_direction_check=calibrate_dist._new_distance_direction_check_state(),
            )

        self.assertIsNone(abort_reason)
        self.assertIsNotNone(row)
        self.assertTrue(row.no_meaningful_movement)
        self.assertAlmostEqual(row.raw_delta_mm, -0.09, places=2)
        self.assertAlmostEqual(row.signed_cmd_delta_mm, 0.0)
        self.assertAlmostEqual(row.cmd_delta_mm, 0.0)
        self.assertFalse(row.wrong_way)
        logged_lines = [call.args[0] for call in mock_log_line.call_args_list]
        self.assertTrue(any("no_change=True raw_delta=-0.09mm" in line for line in logged_lines))
        self.assertTrue(any("Recording this as a no-change sample" in line for line in logged_lines))

    def test_main_uses_prompted_speed_and_duration_after_observing_distance(self):
        payloads = []
        setup_pose = {
            "dist": 112.0,
            "confidence": 0.9,
            "samples_used": 3,
            "pose_source": "lite_smoothed",
            "lite_required_frames": 3,
        }
        trial_row = SimpleNamespace(
            wrong_way=False,
            cmd="b",
            duration_ms=260,
            cmd_delta_mm=1.25,
        )

        with patch.object(sys, "argv", ["helper_calibrate_dist.py", "--trials", "1", "--no-livestream"]), patch.object(
            calibrate_dist,
            "_ensure_run_dir",
        ), patch.object(
            calibrate_dist,
            "shared_prompt_calibration_run_settings",
            return_value={
                "speed_score": 8,
                "min_duration_ms": 260,
                "max_duration_ms": 260,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
            },
        ) as mock_prompt, patch.object(
            calibrate_dist,
            "WorldModel",
            return_value=SimpleNamespace(step_state=None, _post_action_observe_delay_s=0.0),
        ), patch.object(
            calibrate_dist,
            "Robot",
            return_value=SimpleNamespace(close=lambda: None),
        ), patch.object(
            calibrate_dist,
            "YoloBrickDetector",
            return_value=SimpleNamespace(close=lambda: None),
        ), patch.object(
            calibrate_dist,
            "LivePlot",
            return_value=SimpleNamespace(finish=lambda: None),
        ), patch.object(
            calibrate_dist,
            "_observe_pose_with_reobserve",
            return_value=(setup_pose, {"mode": "primary_full", "reobserved": False}),
        ), patch.object(
            calibrate_dist,
            "_run_trial_action",
            return_value=(trial_row, None),
        ) as mock_run_trial_action, patch.object(
            calibrate_dist,
            "_build_payload",
            side_effect=lambda **kwargs: {
                "config": kwargs["config"],
                "durations_ms": list(kwargs["durations_ms"]),
            },
        ), patch.object(
            calibrate_dist,
            "_write_results",
            side_effect=lambda _path, payload: payloads.append(payload),
        ), patch.object(
            calibrate_dist,
            "log_line",
        ):
            exit_code = calibrate_dist.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_prompt.call_args.kwargs["observed_distance_mm"], 112.0)
        self.assertEqual(payloads[-1]["config"]["speed_score"], 8)
        self.assertEqual(payloads[-1]["config"]["requested_speed_score"], 8)
        self.assertEqual(payloads[-1]["config"]["speed_score_source"], "prompt")
        self.assertTrue(payloads[-1]["config"]["prompted_speed_score"])
        self.assertEqual(payloads[-1]["config"]["min_duration_ms"], 260)
        self.assertEqual(payloads[-1]["config"]["max_duration_ms"], 260)
        self.assertTrue(payloads[-1]["config"]["prompted_duration_bounds"])
        self.assertEqual(payloads[-1]["config"]["target_dist_mm"], 112.0)
        self.assertEqual(payloads[-1]["durations_ms"], [260])
        self.assertEqual(mock_run_trial_action.call_args.kwargs["setup_score"], 8)
        self.assertEqual(mock_run_trial_action.call_args.kwargs["duration_ms"], 260)
        self.assertEqual(mock_run_trial_action.call_args.kwargs["target_dist_mm"], 112.0)


if __name__ == "__main__":
    unittest.main()
