import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_close_gaps


class TestHelperCloseGapsTurnDriveTrials(unittest.TestCase):
    def test_production_turn_drive_curve_plan_skips_excluded_outlier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trials_dir = Path(tmp_dir)
            payload = {
                "production": True,
                "file_type": "turn_drive_trials",
                "name": "forwardLeft_backRight",
                "curve_stats": {"production_worthy": True},
                "curve": {
                    "measured_phase": {
                        "cmd": "r",
                        "drive_mode": "backward",
                        "score_pct": 1,
                        "pwm_override": 191,
                        "profile_name": "backward_pivot_min_inner",
                        "action_note": "TURN+BWD",
                        "motor_pair": {
                            "left_motor_pwm": 191,
                            "left_motor_action": "f",
                            "right_motor_pwm": 100,
                            "right_motor_action": "b",
                        },
                    }
                },
                "trials_backwards": [
                    {
                        "trial": 1,
                        "measuredDurationMs": 150,
                        "startDist": 257.118,
                        "xGapClosed": 8.491,
                        "usable": False,
                    },
                    {
                        "trial": 2,
                        "measuredDurationMs": 178,
                        "startDist": 259.128,
                        "xGapClosed": 1.256,
                        "usable": True,
                    },
                    {
                        "trial": 3,
                        "measuredDurationMs": 206,
                        "startDist": 263.491,
                        "xGapClosed": 5.056,
                        "usable": True,
                    },
                ],
            }
            (trials_dir / "forwardLeft_backRight.json").write_text(json.dumps(payload))

            plan = helper_close_gaps.production_turn_drive_curve_plan(
                cmd="r",
                drive_mode="backward",
                current_dist_mm=258.0,
                x_err_mm=-2.0,
                trials_dir=trials_dir,
            )

            self.assertIsInstance(plan, dict)
            self.assertEqual(plan["trial"], 3)
            self.assertEqual(plan["duration_override_ms"], 206)
            self.assertEqual(plan["pwm_override"], 191)
            self.assertEqual(plan["score"], 1)
            self.assertEqual(plan["profile_override"]["drive_mode"], "backward")
            self.assertAlmostEqual(plan["profile_override"]["inner_ratio"], 100.0 / 191.0, places=6)

    def test_forward_setup_plan_has_no_measured_duration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trials_dir = Path(tmp_dir)
            payload = {
                "production": True,
                "file_type": "turn_drive_trials",
                "name": "forwardRight_backLeft",
                "curve_stats": {"production_worthy": True},
                "curve": {
                    "setup_phase": {
                        "cmd": "l",
                        "drive_mode": "forward",
                        "score_pct": 1,
                        "profile_name": "forward_pivot_min_inner",
                        "action_note": "TURN+FWD",
                        "motor_pair": {
                            "left_motor_pwm": 1,
                            "left_motor_action": "b",
                            "right_motor_pwm": 199,
                            "right_motor_action": "f",
                        },
                    },
                    "measured_phase": {
                        "cmd": "l",
                        "drive_mode": "backward",
                        "score_pct": 1,
                        "pwm_override": 199,
                        "profile_name": "backward_pivot_min_inner",
                        "action_note": "TURN+BWD",
                        "motor_pair": {
                            "left_motor_pwm": 61,
                            "left_motor_action": "f",
                            "right_motor_pwm": 199,
                            "right_motor_action": "b",
                        },
                    },
                },
                "trials_backwards": [
                    {
                        "trial": 10,
                        "measuredDurationMs": 400,
                        "startDist": 162.479,
                        "xGapClosed": 4.164,
                        "usable": True,
                    }
                ],
            }
            (trials_dir / "forwardRight_backLeft.json").write_text(json.dumps(payload))

            setup_plan = helper_close_gaps.production_turn_drive_curve_plan(
                cmd="l",
                drive_mode="forward",
                current_dist_mm=173.0,
                x_err_mm=4.0,
                trials_dir=trials_dir,
            )
            measured_plan = helper_close_gaps.production_turn_drive_curve_plan(
                cmd="l",
                drive_mode="backward",
                current_dist_mm=173.0,
                x_err_mm=4.0,
                trials_dir=trials_dir,
            )

            self.assertEqual(setup_plan["source"], "production_turn_drive_trials_forward")
            self.assertIsNone(setup_plan["duration_override_ms"])
            self.assertEqual(measured_plan["source"], "production_turn_drive_trials")
            self.assertEqual(measured_plan["duration_override_ms"], 400)
            self.assertEqual(measured_plan["pwm_override"], 199)


if __name__ == "__main__":
    unittest.main()
