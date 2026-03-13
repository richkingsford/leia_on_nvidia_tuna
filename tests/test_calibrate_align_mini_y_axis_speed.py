import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import calibrate_align_mini as cam


class _FakeVision:
    def __init__(self, readings):
        self._readings = list(readings)
        self._idx = 0

    def read(self):
        if self._idx >= len(self._readings):
            return self._readings[-1]
        reading = self._readings[self._idx]
        self._idx += 1
        return reading


class _FakeWorld:
    def update_vision(self, *args):
        self.last_update = args


class _FakeRuntimeWorld:
    def __init__(self):
        self.brick = {
            "visible": True,
            "offset_x": 9.0,
            "x_axis": 9.0,
            "offset_y": 9.0,
            "y_axis": 9.0,
            "dist": 120.0,
            "angle": 5.0,
            "confidence": 50.0,
        }
        self.process_rules = {}
        self.step_state = "ALIGN_BRICK"


class TestCalibrateAlignMiniYAxisSpeed(unittest.TestCase):
    def test_build_trial_gap_families_creates_large_medium_small_targets(self):
        families = cam._build_trial_gap_families(
            min_err_mm=2.5,
            max_err_mm=20.0,
            target_tol_mm=2.0,
        )

        self.assertEqual([row["family"] for row in families], ["large", "medium", "small"])
        self.assertAlmostEqual(float(families[0]["range_min_mm"]), 7.5, places=3)
        self.assertAlmostEqual(float(families[0]["range_max_mm"]), 20.0, places=3)
        self.assertAlmostEqual(float(families[0]["target_abs_mm"]), 13.75, places=3)
        self.assertAlmostEqual(float(families[1]["range_min_mm"]), 5.0, places=3)
        self.assertAlmostEqual(float(families[1]["range_max_mm"]), 7.5, places=3)
        self.assertAlmostEqual(float(families[2]["range_min_mm"]), 2.5, places=3)
        self.assertAlmostEqual(float(families[2]["range_max_mm"]), 5.0, places=3)

    def test_classify_gap_family_uses_current_gap(self):
        families = cam._build_trial_gap_families(
            min_err_mm=2.5,
            max_err_mm=20.0,
            target_tol_mm=2.0,
        )

        large = cam._classify_gap_family(16.0, families)
        medium = cam._classify_gap_family(6.0, families)
        small = cam._classify_gap_family(3.2, families)

        self.assertEqual(large["family"], "large")
        self.assertEqual(medium["family"], "medium")
        self.assertEqual(small["family"], "small")

    def test_trial_hit_tolerance_mm_relaxes_for_larger_y_families(self):
        self.assertEqual(cam._trial_hit_tolerance_mm(axis="y", axis_tol_mm=1.5, family="small"), 2.0)
        self.assertEqual(cam._trial_hit_tolerance_mm(axis="y", axis_tol_mm=1.5, family="medium"), 4.0)
        self.assertEqual(cam._trial_hit_tolerance_mm(axis="y", axis_tol_mm=1.5, family="large"), 4.0)

    def test_large_family_observation_profile_uses_full_speed_fixed_pulse(self):
        profile = cam._large_family_observation_profile(axis="y", family="large")

        self.assertIsNotNone(profile)
        self.assertEqual(profile["mode"], cam.Y_AXIS_LARGE_OBSERVE_MODE)
        self.assertEqual(float(profile["motion_intensity_pct"]), 100.0)
        self.assertEqual(int(profile["duration_override_ms"]), 250)
        self.assertEqual(int(profile["duration_model_ms"]), 250)

    def test_large_family_observation_profile_does_not_apply_to_medium(self):
        self.assertIsNone(cam._large_family_observation_profile(axis="y", family="medium"))
        self.assertIsNone(cam._large_family_observation_profile(axis="x", family="large"))

    def test_results_trial_summary_line_includes_counted_percentages(self):
        line = cam._results_trial_summary_line(
            "small",
            {
                "trials": 7,
                "success_count": 2,
                "success_total": 7,
                "trial_hit_count": 2,
                "trial_hit_total": 7,
                "median_trial_target_miss_mm": 1.75,
                "median_trial_cmd_delta_mm": 1.12,
            },
        )

        self.assertEqual(
            line,
            "small: success=28.6% (2/7) one_shot_hit=28.6% (2/7) trials=7 "
            "median_miss=1.75mm median_cmd_delta=1.12mm",
        )

    def test_parse_progress_families_allows_disable_and_defaults(self):
        self.assertEqual(cam._parse_progress_families("medium,small"), ("medium", "small"))
        self.assertEqual(cam._parse_progress_families("off"), ())
        self.assertEqual(cam._parse_progress_families("bogus"), cam.Y_AXIS_PROGRESS_FAMILIES_DEFAULT)

    def test_y_cmd_delta_mm_reports_positive_progress_in_command_direction(self):
        self.assertAlmostEqual(cam._y_cmd_delta_mm("u", 3.5, 1.0), 2.5, places=3)
        self.assertAlmostEqual(cam._y_cmd_delta_mm("d", -7.0, -4.0), 3.0, places=3)

    def test_position_y_into_target_families_reaches_medium_with_visible_remeasure(self):
        families = cam._build_trial_gap_families(
            min_err_mm=2.5,
            max_err_mm=20.0,
            target_tol_mm=2.0,
        )
        current_pose = {"offset_y": 15.0}
        sent = []

        def _fake_send(**kwargs):
            sent.append((kwargs.get("cmd"), kwargs.get("duration_override_ms")))
            return {"duration_ms": int(kwargs.get("duration_override_ms") or 0)}

        with patch.object(cam, "_send_fixed_score_axis_command", side_effect=_fake_send), patch.object(
            cam,
            "_read_pose",
            side_effect=[
                {"offset_y": 12.6},
                {"offset_y": 9.8},
                {"offset_y": 7.2},
            ],
        ), patch.object(cam, "_observe_after_action", return_value=0.0):
            pose, meta = cam._position_y_into_target_families(
                vision=object(),
                world=object(),
                robot=object(),
                step_state=object(),
                current_pose=current_pose,
                axis_target_mm=0.0,
                axis_sign=1.0,
                families=families,
                target_families=("medium", "small"),
                max_acts=4,
                allowed_y_cmds=None,
            )

        self.assertIsNotNone(pose)
        self.assertEqual(meta["status"], "positioned")
        self.assertEqual(meta["acts"], 3)
        self.assertEqual(meta["start_family"], "large")
        self.assertEqual(meta["end_family"], "medium")
        self.assertEqual([item[0] for item in sent], ["u", "u", "u"])
        self.assertEqual([step["post_family"] for step in meta["steps"]], ["large", "large", "medium"])

    def test_y_confirmed_window_metrics_uses_median_of_three(self):
        confirmed = cam._y_confirmed_window_metrics(
            [
                {
                    "pre_abs_err_mm": 12.4,
                    "trial_cmd_delta_mm": 0.05,
                    "curve_intensity_pct": 11.0,
                    "motion_intensity_pct": 11.0,
                    "trial_target_miss_mm": 12.35,
                    "trial_hit_success": False,
                },
                {
                    "pre_abs_err_mm": 12.3,
                    "trial_cmd_delta_mm": 5.08,
                    "curve_intensity_pct": 11.0,
                    "motion_intensity_pct": 11.0,
                    "trial_target_miss_mm": 7.2,
                    "trial_hit_success": False,
                },
                {
                    "pre_abs_err_mm": 11.2,
                    "trial_cmd_delta_mm": 0.53,
                    "curve_intensity_pct": 11.0,
                    "motion_intensity_pct": 11.0,
                    "trial_target_miss_mm": 10.7,
                    "trial_hit_success": False,
                },
            ],
            hit_tol_mm=2.0,
        )

        self.assertIsNotNone(confirmed)
        self.assertAlmostEqual(float(confirmed["median_pre_abs_err_mm"]), 12.3, places=3)
        self.assertAlmostEqual(float(confirmed["median_trial_cmd_delta_mm"]), 0.53, places=3)
        self.assertAlmostEqual(float(confirmed["coverage_ratio"]), 0.53 / 12.3, places=3)
        self.assertAlmostEqual(float(confirmed["hit_rate"]), 0.0, places=3)

    def test_tune_y_family_scale_escalates_large_family_more_aggressively(self):
        new_scale, reason = cam._tune_y_family_scale(
            1.0,
            "large",
            {
                "coverage_ratio": 0.10,
                "hit_rate": 0.0,
            },
        )

        self.assertAlmostEqual(float(new_scale), 1.35, places=3)
        self.assertEqual(reason, "confirmed_underpowered_hard")

    def test_resolve_axis_motion_profile_can_fix_y_motion_intensity_and_vary_duration(self):
        with patch.object(
            cam,
            "speed_power_pwm_for_motion_intensity",
            side_effect=[
                (0.1, 40, 1, 725, 12.16),
                (0.05, 39, 1, 315, 7.0),
            ],
        ):
            profile = cam._resolve_axis_motion_profile(
                axis="y",
                cmd="d",
                curve_intensity_pct=12.16,
                y_motion_profile=cam.Y_AXIS_MOTION_PROFILE_FIXED_PWM_DURATION,
                y_fixed_motion_intensity_pct=7.0,
            )

        self.assertEqual(profile["mode"], cam.Y_AXIS_MOTION_PROFILE_FIXED_PWM_DURATION)
        self.assertEqual(profile["curve_intensity_pct"], 12.16)
        self.assertEqual(profile["motion_intensity_pct"], 7.0)
        self.assertEqual(profile["duration_override_ms"], 725)
        self.assertEqual(profile["duration_model_ms"], 725)

    def test_read_pose_averages_repeated_y_axis_samples_without_waiting_for_x_change(self):
        vision = _FakeVision(
            [
                (True, 0.0, 100.0, 4.0, 99.0, 1.0, False, False),
                (True, 0.0, 100.0, 4.0, 99.0, 2.0, False, False),
                (True, 0.0, 100.0, 4.0, 99.0, 3.0, False, False),
            ]
        )
        world = _FakeWorld()

        with patch.object(cam, "OBSERVE_SLEEP_S", 0.0):
            pose = cam._read_pose(vision, world, samples=3, timeout_s=0.1)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(float(pose["offset_x"]), 4.0, places=3)
        self.assertAlmostEqual(float(pose["offset_y"]), 2.0, places=3)

    def test_read_pose_prefers_lite_smoothed_measurement_when_available(self):
        world = _FakeRuntimeWorld()

        with patch.object(cam, "update_world_from_vision") as mock_update, patch.object(
            cam, "lite_gate_unique_frames", return_value=1
        ), patch.object(
            cam,
            "telemetry_latest_unique_smoothed_frames",
            return_value=[{"frame_id": 1}, {"frame_id": 2}, {"frame_id": 3}],
        ), patch.object(
            cam,
            "telemetry_average_smoothed_frames",
            return_value={
                "visible": True,
                "offset_x": 1.5,
                "x_axis": 1.5,
                "offset_y": 2.5,
                "y_axis": 2.5,
                "dist": 105.0,
                "angle": 0.5,
                "confidence": 88.0,
            },
        ), patch.object(cam, "OBSERVE_SLEEP_S", 0.0):
            pose = cam._read_pose(object(), world, samples=1, timeout_s=0.1)

        self.assertIsNotNone(pose)
        self.assertEqual(str(pose.get("pose_source")), "lite_smoothed")
        self.assertAlmostEqual(float(pose["offset_x"]), 1.5, places=3)
        self.assertAlmostEqual(float(pose["offset_y"]), 2.5, places=3)
        mock_update.assert_called_once()

    def test_observe_after_action_waits_until_pulse_finishes(self):
        with patch.object(cam.time, "time", return_value=100.0):
            min_sample_time = cam._observe_after_action(0.30)

        self.assertGreaterEqual(float(min_sample_time), 100.0 + float(cam.POST_ACT_SETTLE_S))
        self.assertLessEqual(float(min_sample_time), 100.0 + float(cam.POST_ACT_MAX_WAIT_S))

    def test_observe_after_action_long_pulse_waits_past_full_duration(self):
        with patch.object(cam.time, "time", return_value=100.0):
            min_sample_time = cam._observe_after_action(0.80)

        self.assertGreaterEqual(float(min_sample_time), 100.8)

    def test_maybe_confirm_y_post_pose_replaces_strongly_worse_initial_read(self):
        initial_pose = {"offset_y": 7.2}
        confirmed_pose = {"offset_y": 5.4}

        with patch.object(cam.time, "sleep") as mock_sleep, patch.object(
            cam, "_read_pose", return_value=confirmed_pose
        ):
            pose, reobserved = cam._maybe_confirm_y_post_pose(
                vision=object(),
                world=object(),
                axis="y",
                axis_target_mm=2.5,
                axis_sign=1.0,
                pre_abs_err=3.4,
                pose_after=initial_pose,
            )

        self.assertIs(pose, confirmed_pose)
        self.assertTrue(reobserved)
        mock_sleep.assert_called_once()

    def test_probe_axis_sign_uses_abs_error_reduction_when_nonzero_target_is_ambiguous(self):
        vision = _FakeVision(
            [
                (True, 0.0, 100.0, 0.0, 99.0, 1.5, False, False),
                (True, 0.0, 100.0, 0.0, 99.0, 0.5, False, False),
            ]
        )
        world = _FakeWorld()

        with patch.object(cam, "OBSERVE_SAMPLES", 1), patch.object(cam, "OBSERVE_SLEEP_S", 0.0), patch.object(
            cam, "_observe_after_action", return_value=0.0
        ), patch.object(cam, "_send_axis_command", return_value={"duration_ms": 0}):
            sign, source = cam._probe_axis_sign(
                vision=vision,
                world=world,
                robot=object(),
                step_state=object(),
                axis="y",
                axis_target_mm=2.5,
                axis_sign=1.0,
                intensity=20.0,
            )

        self.assertEqual(float(sign), -1.0)
        self.assertEqual(str(source), "probe_match")

    def test_symmetrize_curve_skips_y_axis_to_preserve_directional_learning(self):
        candidate = {
            "by_cmd": {
                "l": [20.0, 30.0, 40.0],
                "r": [60.0, 70.0, 80.0],
            }
        }
        samples = [{"cmd": "r"} for _ in range(8)] + [{"cmd": "l"}]

        out = cam._symmetrize_curve_if_needed(candidate, samples, axis="y")

        self.assertEqual(out["by_cmd"]["l"], [20.0, 30.0, 40.0])
        self.assertEqual(out["by_cmd"]["r"], [60.0, 70.0, 80.0])

    def test_axis_sign_evidence_penalizes_wrong_auto_flip_direction(self):
        current = cam._axis_sign_evidence(
            axis="y",
            cmd="d",
            axis_before_mm=7.18,
            axis_after_mm=4.41,
            axis_target_mm=2.5,
            candidate_sign=1.0,
        )
        alternate = cam._axis_sign_evidence(
            axis="y",
            cmd="d",
            axis_before_mm=7.18,
            axis_after_mm=4.41,
            axis_target_mm=2.5,
            candidate_sign=-1.0,
        )

        self.assertEqual(current, -1)
        self.assertEqual(alternate, 1)

    def test_attempt_recovery_reobserves_before_moving(self):
        pose = {"offset_y": 3.0}

        with patch.object(cam, "_read_pose", return_value=pose) as mock_read_pose, patch.object(
            cam, "_recover_visibility"
        ) as mock_recover_visibility, patch.object(cam, "_scan_recover_visibility") as mock_scan_recover:
            recovered = cam._attempt_recovery(
                vision=object(),
                world=object(),
                robot=object(),
                step_state=object(),
                recent_acts=deque([{"cmd": "d", "turn_intensity_pct": 30.0}], maxlen=32),
            )

        self.assertIs(recovered, pose)
        mock_read_pose.assert_called_once()
        mock_recover_visibility.assert_not_called()
        mock_scan_recover.assert_not_called()

    def test_attempt_recovery_uses_scan_after_inverse_recovery_fails(self):
        pose = {"offset_y": 1.0}

        with patch.object(cam, "_read_pose", return_value=None), patch.object(
            cam, "_recover_visibility", return_value=None
        ) as mock_recover_visibility, patch.object(
            cam, "_scan_recover_visibility", return_value=pose
        ) as mock_scan_recover:
            recovered = cam._attempt_recovery(
                vision=object(),
                world=object(),
                robot=object(),
                step_state=object(),
                recent_acts=deque([{"cmd": "d", "turn_intensity_pct": 30.0}], maxlen=32),
            )

        self.assertIs(recovered, pose)
        mock_recover_visibility.assert_called_once()
        mock_scan_recover.assert_called_once()

    def test_scan_recover_visibility_continues_inverse_of_last_axis_act(self):
        sent_cmds = []
        pose = {"offset_y": -4.0}

        def _fake_send_axis_command(**kwargs):
            sent_cmds.append((kwargs.get("cmd"), kwargs.get("intensity")))
            return {"duration_ms": 0}

        with patch.object(cam, "_send_axis_command", side_effect=_fake_send_axis_command), patch.object(
            cam, "_observe_after_action", return_value=0.0
        ), patch.object(cam, "_read_pose", side_effect=[None, pose]):
            recovered = cam._scan_recover_visibility(
                vision=object(),
                world=object(),
                robot=object(),
                step_state=object(),
                recent_acts=deque([{"cmd": "d", "turn_intensity_pct": 20.0}], maxlen=32),
                max_acts=2,
            )

        self.assertIs(recovered, pose)
        self.assertEqual(sent_cmds[0][0], "u")
        self.assertAlmostEqual(float(sent_cmds[0][1]), 17.0, places=3)

if __name__ == "__main__":
    unittest.main()
