import unittest

import zdebug_Yaxis_consistency as yconsistency


class YAxisConsistencyTests(unittest.TestCase):
    def test_parse_int_list_accepts_comma_separated_positive_values(self):
        self.assertEqual(yconsistency._parse_int_list("300, 500,900"), [300, 500, 900])

    def test_normalize_cmd_accepts_auto_when_enabled(self):
        self.assertEqual(yconsistency._normalize_cmd("auto", allow_auto=True), "auto")

    def test_command_delta_is_positive_in_command_direction(self):
        self.assertAlmostEqual(yconsistency._command_delta_mm("u", 21.5, 10.0), 11.5)
        self.assertAlmostEqual(yconsistency._command_delta_mm("d", 10.0, 21.5), 11.5)

    def test_auto_first_cmd_for_y_moves_toward_centerline(self):
        self.assertEqual(yconsistency._auto_first_cmd_for_y(-3.0, 0.0, 0.0), "d")
        self.assertEqual(yconsistency._auto_first_cmd_for_y(3.0, 0.0, 0.0), "u")

    def test_trial_target_y_for_cmd_uses_expected_side_of_center(self):
        self.assertAlmostEqual(yconsistency._trial_target_y_for_cmd("u", center_y_mm=0.0, band_mm=4.0), 4.0)
        self.assertAlmostEqual(yconsistency._trial_target_y_for_cmd("d", center_y_mm=0.0, band_mm=4.0), -4.0)

    def test_trial_story_line_matches_operator_phrase(self):
        line = yconsistency._trial_story_line(
            cmd="u",
            score=100,
            duration_ms=500,
            pre_y_mm=4.25,
            post_y_mm=2.10,
            raw_delta_mm=-2.15,
            center_y_mm=0.0,
        )
        self.assertEqual(
            line,
            'I wanted to go up 4.25mm, so I did the "U 100% 500ms" act, resulting in the brick raising 2.15mm (2.10mm off).',
        )

    def test_consistency_label_buckets_spread(self):
        self.assertEqual(yconsistency._consistency_label(3.0, 0.2), "tight")
        self.assertEqual(yconsistency._consistency_label(3.0, 0.8), "semi-tight")
        self.assertEqual(yconsistency._consistency_label(3.0, 2.0), "all over the place")

    def test_consistency_summary_line_matches_operator_format(self):
        line = yconsistency._consistency_summary_line(
            trial_count=6,
            cmd="u",
            score=100,
            duration_ms=250,
            summary={
                "cmd_delta_mm": {
                    "median": 3.17,
                    "stdev": 0.42,
                }
            },
        )
        self.assertEqual(
            line,
            "6 trials; speed sent: U 100% 250ms; median distance covered: 3.17mm; "
            "standard deviation: 0.42mm (tight)",
        )

    def test_setup_cmd_for_target_y_chooses_direction_into_band(self):
        self.assertEqual(yconsistency._setup_cmd_for_target_y(10.0, 14.0, 1.0), "d")
        self.assertEqual(yconsistency._setup_cmd_for_target_y(18.0, 14.0, 1.0), "u")
        self.assertIsNone(yconsistency._setup_cmd_for_target_y(14.4, 14.0, 0.5))

    def test_setup_next_cmd_can_enforce_downward_final_approach(self):
        done, ready, cmd = yconsistency._setup_next_cmd_for_target_y(
            14.2,
            14.0,
            1.0,
            approach_cmd="d",
            approach_margin_mm=0.5,
            approach_ready=False,
        )
        self.assertFalse(done)
        self.assertFalse(ready)
        self.assertEqual(cmd, "u")

        done, ready, cmd = yconsistency._setup_next_cmd_for_target_y(
            12.2,
            14.0,
            1.0,
            approach_cmd="d",
            approach_margin_mm=0.5,
            approach_ready=False,
        )
        self.assertFalse(done)
        self.assertTrue(ready)
        self.assertEqual(cmd, "d")

        done, ready, cmd = yconsistency._setup_next_cmd_for_target_y(
            14.6,
            14.0,
            1.0,
            approach_cmd="d",
            approach_margin_mm=0.5,
            approach_ready=True,
        )
        self.assertTrue(done)
        self.assertTrue(ready)
        self.assertIsNone(cmd)

    def test_default_setup_fine_duration_halves_coarse_override(self):
        self.assertEqual(yconsistency._default_setup_fine_duration_ms(300, None), 150)
        self.assertEqual(yconsistency._default_setup_fine_duration_ms(None, None), 150)
        self.assertEqual(yconsistency._default_setup_fine_duration_ms(300, 120), 120)

    def test_select_setup_motion_uses_fine_near_plain_target_band(self):
        motion = yconsistency._select_setup_motion(
            12.6,
            14.0,
            1.0,
            setup_score=1,
            setup_duration_ms=300,
            setup_fine_score=1,
            setup_fine_duration_ms=150,
            setup_fine_window_mm=0.5,
        )
        self.assertEqual(motion["mode"], "fine")
        self.assertEqual(motion["score"], 1)
        self.assertEqual(motion["duration_ms"], 150)

    def test_select_setup_motion_uses_fine_near_forced_downward_preload_boundary(self):
        motion = yconsistency._select_setup_motion(
            14.2,
            14.0,
            1.0,
            setup_score=1,
            setup_duration_ms=300,
            setup_fine_score=1,
            setup_fine_duration_ms=150,
            setup_fine_window_mm=1.5,
            approach_cmd="d",
            approach_margin_mm=0.5,
            approach_ready=False,
        )
        self.assertEqual(motion["mode"], "fine")
        self.assertEqual(motion["duration_ms"], 150)

    def test_select_setup_motion_stays_coarse_when_far_from_next_boundary(self):
        motion = yconsistency._select_setup_motion(
            10.0,
            14.0,
            1.0,
            setup_score=1,
            setup_duration_ms=300,
            setup_fine_score=1,
            setup_fine_duration_ms=150,
            setup_fine_window_mm=1.0,
            approach_cmd="d",
            approach_margin_mm=0.5,
            approach_ready=False,
        )
        self.assertEqual(motion["mode"], "coarse")
        self.assertEqual(motion["duration_ms"], 300)

    def test_approach_shelf_target_y_uses_upper_shelf_for_downward_branch(self):
        self.assertAlmostEqual(
            yconsistency._approach_shelf_target_y(14.0, 1.0, approach_cmd="d", approach_margin_mm=0.5),
            12.5,
        )
        self.assertAlmostEqual(
            yconsistency._approach_shelf_target_y(14.0, 1.0, approach_cmd="u", approach_margin_mm=0.5),
            15.5,
        )
        self.assertIsNone(
            yconsistency._approach_shelf_target_y(14.0, 1.0, approach_cmd=None, approach_margin_mm=0.5)
        )

    def test_pose_effect_reports_y_and_dist_deltas(self):
        effect = yconsistency._pose_effect(
            "d",
            {"offset_y": 10.0, "dist": 100.0},
            {"offset_y": 21.5, "dist": 96.5},
        )
        self.assertAlmostEqual(effect["cmd_delta_mm"], 11.5)
        self.assertAlmostEqual(effect["raw_delta_mm"], 11.5)
        self.assertAlmostEqual(effect["dist_delta_mm"], -3.5)

    def test_stats_reports_max_abs_deviation_from_median(self):
        stats = yconsistency._stats([10.0, 10.5, 11.5])
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["median"], 10.5)
        self.assertAlmostEqual(stats["max_abs_deviation_from_median"], 1.0)

    def test_build_trial_summary_includes_reset_stats(self):
        trial = yconsistency.TrialResult(
            trial=1,
            duration_group_ms=100,
            repeat_in_group=1,
            cmd="d",
            score_requested=1,
            duration_override_ms=100,
            cmd_sent="d",
            pwm=40,
            power=0.018,
            duration_ms=100,
            pre_y_mm=14.0,
            post_y_mm=12.0,
            pre_dist_mm=100.0,
            post_dist_mm=99.9,
            raw_delta_mm=-2.0,
            cmd_delta_mm=2.0,
            dist_delta_mm=-0.1,
            pre_pose_source="lite_smoothed",
            post_pose_source="lite_smoothed",
            pre_lite_required_frames=3,
            post_lite_required_frames=3,
            reset_acts=2,
            reset_final_y_error_mm=0.1,
            setup_acts=3,
            target_gap_mm=4.0,
            off_center_mm=1.2,
            success_margin_mm=2.0,
            hit_target=True,
        )
        summary = yconsistency._build_trial_summary([trial])
        self.assertEqual(summary["reset_acts"]["count"], 1)
        self.assertAlmostEqual(summary["reset_acts"]["median"], 2.0)
        self.assertAlmostEqual(summary["reset_final_y_error_mm"]["median"], 0.1)
        self.assertEqual(summary["hit_rate_display"], "100.0% (1/1)")
        self.assertAlmostEqual(summary["off_center_mm"]["median"], 1.2)


if __name__ == "__main__":
    unittest.main()
