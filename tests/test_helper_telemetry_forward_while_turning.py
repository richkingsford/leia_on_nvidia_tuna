import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_close_gaps
import helper_telemetry_forward_while_turning


class TestHelperTelemetryForwardWhileTurning(unittest.TestCase):
    def test_x_axis_correction_cmd_uses_calibrated_runtime_polarity(self):
        self.assertEqual(helper_close_gaps.x_axis_correction_cmd(6.0), "l")
        self.assertEqual(helper_close_gaps.x_axis_correction_cmd(-6.0), "r")

    def test_build_plan_is_disabled_without_explicit_step_config(self):
        plan = helper_telemetry_forward_while_turning.build_forward_while_turning_plan(
            step_rules={},
            cmd="f",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 6.0,
                "x_tol": 1.4,
                "dist": 130.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsNone(plan)

    def test_choose_forward_bias_cmd_converts_turn_to_forward_when_forward_progress_is_valid(self):
        cmd = helper_telemetry_forward_while_turning.choose_forward_bias_cmd(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "prefer_forward_for_x_axis": True,
                        "prefer_forward_min_x_err_mm": 0.3,
                    }
                }
            },
            cmd="l",
            local_gate_status={
                "visible": True,
                "x_err": 6.0,
                "x_tol": 1.4,
                "dist": 130.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
        )

        self.assertEqual(cmd, "f")

    def test_choose_forward_bias_cmd_keeps_turn_when_forward_would_worsen_dist(self):
        cmd = helper_telemetry_forward_while_turning.choose_forward_bias_cmd(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "prefer_forward_for_x_axis": True,
                    }
                }
            },
            cmd="l",
            local_gate_status={
                "visible": True,
                "x_err": 6.0,
                "x_tol": 1.4,
                "dist": 105.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
        )

        self.assertEqual(cmd, "l")

    def test_build_plan_uses_gentle_asymmetric_forward_actions(self):
        plan = helper_telemetry_forward_while_turning.build_forward_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "min_reduction_ratio": 0.01,
                        "max_reduction_ratio": 0.04,
                        "max_x_err_mm_for_full_reduction": 30.0,
                        "min_inner_ratio": 0.95,
                        "min_pwm_delta": 1,
                    }
                }
            },
            cmd="f",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 6.0,
                "x_tol": 1.4,
                "dist": 130.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("cmd"), "f")
        self.assertEqual(plan.get("turn_cmd"), "l")
        self.assertEqual(plan.get("action_note"), "FWD+TURN-BIAS")
        self.assertGreater(int(plan.get("outer_pwm") or 0), int(plan.get("inner_pwm") or 0))
        self.assertEqual(
            plan.get("actions"),
            [
                {"target": "l", "action": "b", "pwm": int(plan.get("inner_pwm"))},
                {"target": "r", "action": "f", "pwm": int(plan.get("outer_pwm"))},
            ],
        )

    def test_build_plan_can_stop_inner_tread_when_x_is_far_off(self):
        plan = helper_telemetry_forward_while_turning.build_forward_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "min_reduction_ratio": 0.02,
                        "max_reduction_ratio": 0.16,
                        "max_x_err_mm_for_full_reduction": 20.0,
                        "min_inner_ratio": 0.8,
                        "min_pwm_delta": 4,
                        "inner_stop_x_err_mm": 10.0,
                    }
                }
            },
            cmd="f",
            speed_score=7,
            local_gate_status={
                "visible": True,
                "x_err": -15.0,
                "x_tol": 1.4,
                "dist": 180.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=240,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("turn_cmd"), "r")
        self.assertEqual(
            plan.get("actions"),
            [
                {"target": "l", "action": "b", "pwm": int(plan.get("outer_pwm"))},
                {"target": "r", "action": "s", "pwm": 0},
            ],
        )

    def test_build_drive_plan_supports_backward_motion(self):
        plan = helper_telemetry_forward_while_turning.build_drive_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "min_reduction_ratio": 0.01,
                        "max_reduction_ratio": 0.04,
                        "max_x_err_mm_for_full_reduction": 30.0,
                        "min_inner_ratio": 0.95,
                        "min_pwm_delta": 1,
                    }
                }
            },
            cmd="b",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": -6.0,
                "x_tol": 1.4,
                "dist": 100.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("cmd"), "b")
        self.assertEqual(plan.get("turn_cmd"), "r")
        self.assertEqual(plan.get("action_note"), "BWD+TURN-BIAS")
        self.assertEqual(
            plan.get("actions"),
            [
                {"target": "l", "action": "f", "pwm": int(plan.get("outer_pwm"))},
                {"target": "r", "action": "b", "pwm": int(plan.get("inner_pwm"))},
            ],
        )

    def test_build_plan_still_biases_turn_inside_x_tolerance_when_not_perfectly_aligned(self):
        plan = helper_telemetry_forward_while_turning.build_forward_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                    }
                }
            },
            cmd="f",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 0.2,
                "x_tol": 1.4,
                "dist": 130.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("turn_cmd"), "l")

    def test_build_plan_goes_straight_only_when_x_is_exactly_aligned(self):
        plan = helper_telemetry_forward_while_turning.build_forward_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                    }
                }
            },
            cmd="f",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 0.0,
                "x_tol": 1.4,
                "dist": 130.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsNone(plan)

    def test_build_plan_can_force_turn_direction_and_quantized_percent_gap(self):
        plan = helper_telemetry_forward_while_turning.build_drive_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "min_reduction_ratio": 0.0,
                        "max_reduction_ratio": 0.0,
                        "min_inner_ratio": 1.0,
                        "min_pwm_delta": 0,
                    }
                }
            },
            cmd="f",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 0.2,
                "x_tol": 1.4,
                "dist": 180.0,
                "dist_target": 111.45,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
            turn_cmd_override="r",
            override_abs_x_err_mm=3.0,
            min_percent_delta=3,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("turn_cmd"), "r")
        outer_pwm = int(plan.get("outer_pwm") or 0)
        inner_pwm = int(plan.get("inner_pwm") or 0)
        outer_pct = helper_telemetry_forward_while_turning._pwm_to_percent(outer_pwm)
        inner_pct = helper_telemetry_forward_while_turning._pwm_to_percent(inner_pwm)
        self.assertGreaterEqual(outer_pct - inner_pct, 3)
        self.assertEqual(
            plan.get("actions"),
            [
                {"target": "l", "action": "b", "pwm": outer_pwm},
                {"target": "r", "action": "f", "pwm": inner_pwm},
            ],
        )

    def test_build_plan_can_boost_score_from_large_x_and_dist_gap(self):
        plan = helper_telemetry_forward_while_turning.build_drive_while_turning_plan(
            step_rules={
                "align_policy": {
                    "forward_while_turning_assist": {
                        "enabled": True,
                        "max_score_boost_pct": 8,
                        "max_x_err_mm_for_full_score_boost": 20.0,
                        "max_dist_err_mm_for_full_score_boost": 100.0,
                    }
                }
            },
            cmd="b",
            speed_score=1,
            local_gate_status={
                "visible": True,
                "x_err": 20.0,
                "x_tol": 1.4,
                "dist": 220.0,
                "dist_target": 100.0,
                "dist_required": True,
                "dist_direction": "low",
            },
            hold_duration_ms=180,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan.get("score_requested"), 1)
        self.assertEqual(plan.get("score"), 9)
        self.assertEqual(plan.get("score_boost_pct"), 8)
        self.assertAlmostEqual(float(plan.get("dist_gap_mm") or 0.0), 120.0)


if __name__ == "__main__":
    unittest.main()
