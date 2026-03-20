import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_close_gaps as helper_next2


class TestHelperNext2YAxisAlias(unittest.TestCase):
    def test_normalize_curve_accepts_up_down_aliases(self):
        curve = helper_next2.normalize_curve(
            {
                "curve": {
                    "bins_mm": [1.0, 2.0],
                    "by_cmd": {
                        "u": [11.0, 22.0, 33.0],
                        "d": [44.0, 55.0, 66.0],
                    },
                }
            }
        )

        self.assertEqual(curve["by_cmd"]["l"], [11.0, 22.0, 33.0])
        self.assertEqual(curve["by_cmd"]["r"], [44.0, 55.0, 66.0])
        self.assertEqual(helper_next2.curve_intensity_for_error(curve, "u", 0.5), 11.0)
        self.assertEqual(helper_next2.curve_intensity_for_error(curve, "d", 3.0), 66.0)

    def test_fit_monotonic_curve_coalesces_up_down_samples_into_left_right_bins(self):
        curve = helper_next2.fit_monotonic_curve(
            [
                {"cmd": "u", "abs_err_mm": 0.5, "required_intensity": 12.0},
                {"cmd": "u", "abs_err_mm": 2.5, "required_intensity": 28.0},
                {"cmd": "d", "abs_err_mm": 0.5, "required_intensity": 18.0},
                {"cmd": "d", "abs_err_mm": 2.5, "required_intensity": 36.0},
            ],
            bins_mm=[1.0, 2.0],
            min_intensity_pct=1.0,
            max_intensity_pct=100.0,
        )

        self.assertIsNotNone(curve)
        self.assertEqual(curve["by_cmd"]["l"], [12.0, 12.0, 28.0])
        self.assertEqual(curve["by_cmd"]["r"], [18.0, 18.0, 36.0])

    def test_curve_metadata_compatible_rejects_mismatched_gate_target(self):
        compatible = helper_next2.curve_metadata_compatible(
            {
                "step": "ALIGN_BRICK",
                "axis": "y",
                "axis_gate_target_mm": -0.57,
                "axis_gate_tol_mm": 1.5,
            },
            step="ALIGN_BRICK",
            axis="y",
            axis_target_mm=2.5,
            axis_tol_mm=1.5,
        )

        self.assertFalse(compatible)

    def test_curve_metadata_compatible_rejects_mismatched_y_motion_profile(self):
        compatible = helper_next2.curve_metadata_compatible(
            {
                "step": "ALIGN_BRICK",
                "axis": "y",
                "axis_gate_target_mm": 2.5,
                "axis_gate_tol_mm": 1.5,
                "y_motion_profile": "intensity_duration",
            },
            step="ALIGN_BRICK",
            axis="y",
            axis_target_mm=2.5,
            axis_tol_mm=1.5,
            motion_profile_mode="fixed_pwm_duration",
            fixed_motion_intensity_pct=7.0,
        )

        self.assertFalse(compatible)

    def test_curve_is_usable_accepts_small_but_real_progress_curve(self):
        self.assertTrue(
            helper_next2.curve_is_usable(
                {
                    "success_rate": 0.0833,
                    "median_improvement_mm": 0.109,
                    "median_abs_err_after_mm": 7.42,
                    "negative_improve_rate": 0.458,
                }
            )
        )
        self.assertFalse(
            helper_next2.curve_is_usable(
                {
                    "success_rate": 0.0,
                    "median_improvement_mm": -0.02,
                    "median_abs_err_after_mm": 32.19,
                    "negative_improve_rate": 0.522,
                }
            )
        )

    def test_select_axis_act_uses_down_for_positive_y_error_and_up_for_negative(self):
        curve = helper_next2.normalize_curve(
            {
                "curve": {
                    "bins_mm": [2.0, 4.0],
                    "by_cmd": {
                        "u": [12.0, 18.0, 24.0],
                        "d": [14.0, 20.0, 26.0],
                    },
                }
            }
        )

        process_rules = {
            "steps": {
                "ALIGN_BRICK": {
                    "success_gates": {
                        "y_axis": {"target": 0.0, "tol": 1.5},
                    }
                }
            }
        }

        pos = helper_next2.select_axis_act(
            process_rules=process_rules,
            step="ALIGN_BRICK",
            axis="y",
            axis_mm=6.0,
            visible=True,
            curve=curve,
        )
        neg = helper_next2.select_axis_act(
            process_rules=process_rules,
            step="ALIGN_BRICK",
            axis="y",
            axis_mm=-6.0,
            visible=True,
            curve=curve,
        )

        self.assertEqual(pos["cmd"], "d")
        self.assertEqual(neg["cmd"], "u")

    def test_select_axis_act_uses_left_for_positive_x_error_and_right_for_negative(self):
        curve = helper_next2.normalize_curve(
            {
                "curve": {
                    "bins_mm": [2.0, 4.0],
                    "by_cmd": {
                        "l": [12.0, 18.0, 24.0],
                        "r": [14.0, 20.0, 26.0],
                    },
                }
            }
        )

        process_rules = {
            "steps": {
                "ALIGN_BRICK": {
                    "success_gates": {
                        "x_axis": {"target": 0.0, "tol": 1.5},
                    }
                }
            }
        }

        pos = helper_next2.select_axis_act(
            process_rules=process_rules,
            step="ALIGN_BRICK",
            axis="x",
            axis_mm=6.0,
            visible=True,
            curve=curve,
        )
        neg = helper_next2.select_axis_act(
            process_rules=process_rules,
            step="ALIGN_BRICK",
            axis="x",
            axis_mm=-6.0,
            visible=True,
            curve=curve,
        )

        self.assertEqual(pos["cmd"], "l")
        self.assertEqual(neg["cmd"], "r")

    def test_calibrated_axis_motion_profile_inverts_linear_curve_to_duration(self):
        payload = {
            "aruco_marker_calibration": {
                "speed_score_pct": 20,
                "duration_range_ms": [250, 500],
                "reference_distance_mm": 122.0,
                "by_cmd": {
                    "l": {
                        "intercept_mm": -3.0,
                        "slope_mm_per_ms": 0.02,
                        "equation": "distance_mm = -3.0 + 0.020000 * duration_ms",
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curve.json"
            path.write_text(json.dumps(payload) + "\n")
            profile = helper_next2.calibrated_axis_motion_profile(
                axis="x",
                cmd="l",
                gap_mm=7.0,
                path=path,
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["cmd"], "l")
        self.assertEqual(profile["score"], 20)
        self.assertEqual(profile["duration_override_ms"], 500)
        self.assertEqual(profile["duration_clamped_to"], None)
        self.assertAlmostEqual(profile["predicted_distance_mm"], 7.0, places=3)

    def test_calibrated_axis_motion_profile_supports_distance_axis(self):
        payload = {
            "aruco_marker_calibration": {
                "speed_score_pct": 5,
                "duration_range_ms": [200, 400],
                "reference_distance_mm": 125.6,
                "by_cmd": {
                    "f": {
                        "intercept_mm": -2.0,
                        "slope_mm_per_ms": 0.01,
                        "equation": "distance_mm = -2.0 + 0.010000 * duration_ms",
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dist_curve.json"
            path.write_text(json.dumps(payload) + "\n")
            profile = helper_next2.calibrated_axis_motion_profile(
                axis="dist",
                cmd="f",
                gap_mm=2.0,
                path=path,
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["cmd"], "f")
        self.assertEqual(profile["score"], 5)
        self.assertEqual(profile["duration_override_ms"], 400)
        self.assertEqual(profile["duration_clamped_to"], None)
        self.assertAlmostEqual(profile["predicted_distance_mm"], 2.0, places=3)

    def test_calibrated_axis_motion_profile_clamps_to_max_duration(self):
        payload = {
            "aruco_marker_calibration": {
                "speed_score_pct": 5,
                "duration_range_ms": [200, 400],
                "reference_distance_mm": 125.6,
                "by_cmd": {
                    "f": {
                        "intercept_mm": -2.0,
                        "slope_mm_per_ms": 0.01,
                        "equation": "distance_mm = -2.0 + 0.010000 * duration_ms",
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dist_curve.json"
            path.write_text(json.dumps(payload) + "\n")
            profile = helper_next2.calibrated_axis_motion_profile(
                axis="dist",
                cmd="f",
                gap_mm=8.0,
                path=path,
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["cmd"], "f")
        self.assertEqual(profile["score"], 5)
        self.assertEqual(profile["duration_override_ms"], 400)
        self.assertEqual(profile["duration_clamped_to"], "max")
        self.assertAlmostEqual(profile["predicted_distance_mm"], 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
