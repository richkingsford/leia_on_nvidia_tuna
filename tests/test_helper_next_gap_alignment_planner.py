import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next


class TestHelperNextGapAlignmentPlanner(unittest.TestCase):
    def test_brick_lock_wall_uses_gap_planner_like_align_brick(self):
        process_rules = {
            "BRICK_LOCK_WALL": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.4},
                    "dist": {"target": 95.0, "tol": 4.0},
                }
            },
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.4},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 90.0, "tol": 1.5},
                }
            },
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                }
            },
        }

        self.assertTrue(helper_next.step_uses_gap_alignment_planner(process_rules, "ALIGN_BRICK"))
        self.assertTrue(helper_next.step_uses_gap_alignment_planner(process_rules, "BRICK_LOCK_WALL"))
        self.assertFalse(helper_next.step_uses_gap_alignment_planner(process_rules, "FIND_WALL"))

        brick_lock_plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="BRICK_LOCK_WALL",
            x_axis_mm=20.0,
            y_axis_mm=0.0,
            dist_mm=150.0,
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
        )
        align_plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="ALIGN_BRICK",
            x_axis_mm=20.0,
            y_axis_mm=10.0,
            dist_mm=150.0,
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
        )
        self.assertEqual(brick_lock_plan.get("planner"), "gap")
        self.assertEqual(align_plan.get("planner"), "gap")

    def test_gap_planner_deprioritizes_y_axis_until_gap_is_much_larger(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "u",
                "worst_metric": "yAxis_offset_abs",
                "speed_score": 10,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=2.2,   # x_ratio ~= 1.2
                y_axis_mm=3.0,   # y_ratio ~= 2.0 (larger, but not 3x larger)
                dist_mm=100.0,   # d_ratio = 0
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("correction_type"), "x_axis", plan)
        self.assertIn(plan.get("cmd"), ("l", "r"), plan)

    def test_gap_planner_blocks_y_axis_until_other_gaps_are_near_perfect(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "u",
                "worst_metric": "yAxis_offset_abs",
                "speed_score": 10,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.2,   # x_ratio ~= 0.2 (> 5% threshold)
                y_axis_mm=8.0,   # y_ratio huge, but should still be blocked
                dist_mm=102.2,   # d_ratio ~= 0.1 (> 5% threshold)
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertNotEqual(plan.get("correction_type"), "y_axis", plan)

    def test_gap_planner_forces_y_axis_when_marker_is_near_frame_edge(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 5,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.2,    # x_ratio ~= 0.2 (> near-ready threshold)
                y_axis_mm=12.0,   # very high y offset: should force y-axis correction
                dist_mm=102.2,    # d_ratio ~= 0.1 (> near-ready threshold)
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("correction_type"), "y_axis", plan)
        self.assertEqual(plan.get("reason"), "y_axis_edge_force", plan)
        self.assertTrue(bool(plan.get("y_axis_edge_force_triggered")), plan)
        self.assertIn(plan.get("cmd"), ("u", "d"), plan)

    def test_gap_planner_edge_force_can_be_disabled_per_step(self):
        process_rules = {
            "ALIGN_BRICK": {
                "align_policy": {
                    "y_axis_edge_force_enabled": False,
                },
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 5,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.2,
                y_axis_mm=12.0,
                dist_mm=102.2,
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertNotEqual(plan.get("correction_type"), "y_axis", plan)
        self.assertFalse(bool(plan.get("y_axis_edge_force_triggered")), plan)

    def test_gap_planner_biases_y_axis_when_close_and_marker_is_low(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 5,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.2,   # x ratio > near-ready threshold
                y_axis_mm=2.2,   # marker low enough in frame (+y)
                dist_mm=95.0,    # close to brick (<100mm): bias toward y-axis
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("correction_type"), "y_axis", plan)
        self.assertEqual(plan.get("reason"), "y_axis_close_bottom_bias", plan)
        self.assertTrue(bool(plan.get("y_axis_close_bottom_bias_triggered")), plan)
        self.assertIn(plan.get("cmd"), ("u", "d"), plan)

    def test_gap_planner_close_bottom_bias_requires_near_distance(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 5,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.2,
                y_axis_mm=2.2,
                dist_mm=110.0,   # not close enough for bottom-bias
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertNotEqual(plan.get("reason"), "y_axis_close_bottom_bias", plan)
        self.assertFalse(bool(plan.get("y_axis_close_bottom_bias_triggered")), plan)

    def test_gap_planner_allows_y_axis_when_other_gaps_are_within_5pct(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "u",
                "worst_metric": "yAxis_offset_abs",
                "speed_score": 10,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=1.04,   # x_ratio ~= 0.04 (within 5%)
                y_axis_mm=8.0,    # y_ratio huge and now eligible
                dist_mm=102.08,   # d_ratio ~= 0.04 (within 5%)
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("correction_type"), "y_axis", plan)
        self.assertIn(plan.get("cmd"), ("u", "d"), plan)

    def test_gap_planner_rotation_prefers_distance_over_y_when_y_not_severe_enough(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 10,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules=None,
                step="ALIGN_BRICK",
                x_axis_mm=3.2,    # x_ratio ~= 2.2 (initially chosen x)
                y_axis_mm=3.5,    # y_ratio ~= 2.5 (bigger raw, but penalized)
                dist_mm=106.2,    # d_ratio ~= (6.2-2)/2 = 2.1
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
                previous_correction_type="x_axis",
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertTrue(bool(plan.get("rotation_override")), plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertIn(plan.get("cmd"), ("f", "b"), plan)

    def test_gap_planner_does_not_let_analytics_cmd_force_distance_over_bigger_x_gap(self):
        process_rules = {
            "BRICK_LOCK_WALL": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.4},
                    "dist": {"target": 95.0, "tol": 4.34},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "f",  # generic analytics says drive
                "worst_metric": "dist",
                "speed_score": 5,
            }
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="BRICK_LOCK_WALL",
                x_axis_mm=12.0,   # x gap large
                y_axis_mm=0.0,
                dist_mm=110.0,    # dist gap present, but smaller normalized gap
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("correction_type"), "x_axis", plan)
        self.assertIn(plan.get("cmd"), ("l", "r"), plan)

    def test_gap_planner_keeps_its_own_score_instead_of_analytics_override(self):
        process_rules = {
            "BRICK_LOCK_WALL": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.4},
                    "dist": {"target": 95.0, "tol": 4.34},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "r",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 1,  # should no longer override the gap micro score
            }
            x_err_mm = 10.0
            plan = helper_next.select_align_brick_next_act(
                process_rules=process_rules,
                learned_rules={},  # auto path (used to trigger analytics-score override)
                step="BRICK_LOCK_WALL",
                x_axis_mm=x_err_mm,
                y_axis_mm=0.0,
                dist_mm=95.0,
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        expected_score = int(helper_next.align_brick_x_axis_one_shot_score(x_err_mm))
        self.assertEqual(plan.get("correction_type"), "x_axis", plan)
        self.assertEqual(int(plan.get("score") or 0), expected_score, plan)
        self.assertNotEqual(int(plan.get("score") or 0), 1, plan)

    def test_gap_planner_dist_only_step_does_not_invent_x_gap(self):
        process_rules = {
            "SEAT_BRICK2": {
                "success_gates": {
                    "visible": {"min": True},
                    "dist": {"target": 41.26, "tol": 1.5},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "l",  # generic analytics may suggest turn; gap planner should ignore for dist-only step
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 5,
            }
            plan = helper_next.select_alignment_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="SEAT_BRICK2",
                x_axis_mm=25.0,   # large x offset should not matter without an x gate
                y_axis_mm=0.0,
                dist_mm=70.0,     # distance clearly outside gate
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertIn(plan.get("cmd"), ("f", "b"), plan)

    def test_gap_planner_seat_brick2_uses_y_axis_when_y_gate_is_present(self):
        process_rules = {
            "SEAT_BRICK2": {
                "success_gates": {
                    "visible": {"min": True},
                    "yAxis_offset_abs": {"target": 3.65, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 1.5},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "u",
                "worst_metric": "yAxis_offset_abs",
                "speed_score": 3,
            }
            plan = helper_next.select_alignment_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="SEAT_BRICK2",
                x_axis_mm=25.0,   # ignored because no x gate
                y_axis_mm=8.0,    # outside y gate by > tol
                dist_mm=48.2,     # inside dist gate
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "y_axis", plan)
        self.assertIn(plan.get("cmd"), ("u", "d"), plan)

    def test_gap_planner_recovery_disqualify_switches_from_y_axis_to_distance(self):
        process_rules = {
            "SEAT_BRICK2": {
                "success_gates": {
                    "visible": {"min": True},
                    "yAxis_offset_abs": {"target": 3.65, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "u",
                "worst_metric": "yAxis_offset_abs",
                "speed_score": 3,
            }
            plan = helper_next.select_alignment_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="SEAT_BRICK2",
                x_axis_mm=0.0,
                y_axis_mm=8.0,          # y is clearly outside gate
                dist_mm=49.0,           # in gate, but above target so fallback can move forward
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
                avoid_correction_type="y_axis",  # simulate post-recovery disqualification
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertEqual(plan.get("cmd"), "f", plan)
        self.assertTrue(bool(plan.get("rotation_override")), plan)

    def test_gap_planner_holds_when_all_gaps_are_within_gates(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            # Even if analytics proposes a command, the gap planner should hold
            # because no gated metric is outside tolerance.
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "f",
                "worst_metric": "dist",
                "speed_score": 10,
            }
            plan = helper_next.select_alignment_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="ALIGN_BRICK",
                x_axis_mm=0.5,   # within ±1.0 gate
                y_axis_mm=-0.4,  # within ±1.0 gate
                dist_mm=101.2,   # within 100±2 gate
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertIsNone(plan.get("cmd"), plan)
        self.assertIsNone(plan.get("correction_type"), plan)
        self.assertEqual(plan.get("reason"), "all_gaps_within_gate", plan)

    def test_gap_planner_never_selects_in_gate_metric_over_outside_gap(self):
        process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 2.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        orig = helper_next.compute_alignment_decision
        try:
            # Analytics may report x as "worst", but x is in-gate here.
            helper_next.compute_alignment_decision = lambda **kwargs: {
                "cmd": "r",
                "worst_metric": "xAxis_offset_abs",
                "speed_score": 6,
            }
            plan = helper_next.select_alignment_next_act(
                process_rules=process_rules,
                learned_rules={},
                step="ALIGN_BRICK",
                x_axis_mm=1.0,   # within ±2.0 gate (good)
                y_axis_mm=0.1,   # within ±1.0 gate (good)
                dist_mm=104.8,   # outside 100±2 gate (bad)
                visible=True,
                angle_deg=0.0,
                duration_s=0.05,
            )
        finally:
            helper_next.compute_alignment_decision = orig

        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertIn(plan.get("cmd"), ("f", "b"), plan)

    def test_gap_planner_holds_when_dist_is_within_directional_low_gate(self):
        process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="BRICK_LOCK",
            x_axis_mm=-6.0,   # within x gate
            y_axis_mm=0.0,
            dist_mm=148.6,    # below target-tol, still in-gate for directional "low"
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
        )
        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertIsNone(plan.get("cmd"), plan)
        self.assertIsNone(plan.get("correction_type"), plan)
        self.assertEqual(plan.get("reason"), "all_gaps_within_gate", plan)

    def test_gap_planner_still_corrects_when_dist_exceeds_directional_low_gate(self):
        process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="BRICK_LOCK",
            x_axis_mm=-6.0,   # within x gate
            y_axis_mm=0.0,
            dist_mm=166.0,    # above target+tol, outside directional "low"
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
        )
        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertEqual(plan.get("cmd"), "f", plan)

    def test_gap_planner_dist_priority_cheat_forces_distance_choice(self):
        process_rules = {
            "BRICK_LOCK": {
                "align_policy": {
                    "dist_priority_cheat_enabled": True,
                },
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                },
            }
        }
        plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="BRICK_LOCK",
            x_axis_mm=-20.0,  # x gap is larger than dist gap
            y_axis_mm=0.0,
            dist_mm=166.0,    # still outside dist gate
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
        )
        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertEqual(plan.get("correction_type"), "distance", plan)
        self.assertEqual(plan.get("cmd"), "f", plan)
        self.assertTrue(bool(plan.get("cheat_dist_priority")), plan)

    def test_gap_planner_recovery_disqualify_overrides_dist_priority_cheat(self):
        process_rules = {
            "BRICK_LOCK": {
                "align_policy": {
                    "dist_priority_cheat_enabled": True,
                },
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                },
            }
        }
        plan = helper_next.select_alignment_next_act(
            process_rules=process_rules,
            learned_rules={},
            step="BRICK_LOCK",
            x_axis_mm=-20.0,
            y_axis_mm=0.0,
            dist_mm=166.0,
            visible=True,
            angle_deg=0.0,
            duration_s=0.05,
            avoid_correction_type="distance",
        )
        self.assertEqual(plan.get("planner"), "gap", plan)
        self.assertNotEqual(plan.get("correction_type"), "distance", plan)
        self.assertIn(plan.get("cmd"), ("l", "r"), plan)
        self.assertFalse(bool(plan.get("cheat_dist_priority")), plan)


if __name__ == "__main__":
    unittest.main()
