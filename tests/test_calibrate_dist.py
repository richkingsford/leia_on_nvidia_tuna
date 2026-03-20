import unittest

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

    def test_command_delta_is_positive_in_command_direction(self):
        self.assertAlmostEqual(calibrate_dist._command_delta_mm("f", 166.0, 160.0), 6.0)
        self.assertAlmostEqual(calibrate_dist._command_delta_mm("b", 160.0, 166.0), 6.0)

    def test_movement_metrics_reports_total_distance_even_when_wrong_way(self):
        metrics = calibrate_dist._movement_metrics("f", 160.0, 166.0)
        self.assertAlmostEqual(metrics["raw_delta_mm"], 6.0)
        self.assertAlmostEqual(metrics["signed_cmd_delta_mm"], -6.0)
        self.assertAlmostEqual(metrics["cmd_delta_mm"], 6.0)
        self.assertTrue(metrics["wrong_way"])

    def test_build_duration_schedule_uses_helper_step(self):
        durations = calibrate_dist._build_duration_schedule(
            trials=None,
            min_duration_ms=200,
            max_duration_ms=260,
            duration_step_ms=20,
        )
        self.assertEqual(durations, [200, 220, 240, 260])

    def test_build_trial_plan_runs_forward_and_backward_twice_per_duration(self):
        plan = calibrate_dist._build_trial_plan(
            durations_ms=[200, 220],
            trials=None,
        )
        self.assertEqual(
            plan,
            [
                {"duration_ms": 200, "cmd": "f"},
                {"duration_ms": 200, "cmd": "b"},
                {"duration_ms": 200, "cmd": "f"},
                {"duration_ms": 200, "cmd": "b"},
                {"duration_ms": 220, "cmd": "f"},
                {"duration_ms": 220, "cmd": "b"},
                {"duration_ms": 220, "cmd": "f"},
                {"duration_ms": 220, "cmd": "b"},
            ],
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


if __name__ == "__main__":
    unittest.main()
