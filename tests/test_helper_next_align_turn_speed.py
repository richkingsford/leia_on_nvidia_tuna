import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next
import telemetry_robot


class _DummyWorld:
    def __init__(self, x_axis, dist=80.0):
        self.brick = {
            "visible": True,
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": float(dist),
        }


class TestHelperNextAlignTurnSpeed(unittest.TestCase):
    def _rules(self):
        return {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 1.0},
                }
            }
        }

    def test_align_gap_correction_speed_score_clamps_distance_gap_below_8mm_to_1pct(self):
        score = helper_next.align_gap_correction_speed_score("distance", 3.75, cmd="b")
        self.assertEqual(score, telemetry_robot.normalize_speed_score(1))

    def test_align_gap_correction_speed_score_clamps_x_axis_gap_below_8mm_to_1pct(self):
        score = helper_next.align_gap_correction_speed_score("x_axis", 4.8, cmd="l")
        self.assertEqual(score, telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_x_axis_gap_below_1p5mm(self):
        world = _DummyWorld(-0.6)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_x_axis_gap_between_0p5_and_2mm(self):
        world = _DummyWorld(-0.2)  # target=-2.0 => gap=1.8mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_x_axis_gap_below_4mm_near_dist_gate(self):
        world = _DummyWorld(1.0, dist=80.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_x_axis_gap_is_exactly_2mm(self):
        world = _DummyWorld(0.0)  # target=-2.0 => gap=2.0mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_x_axis_gap_is_exactly_4mm_near_dist_gate(self):
        world = _DummyWorld(2.0, dist=80.0)  # target=-2.0 => gap=4.0mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_3pct_when_x_axis_gap_is_above_12mm_and_below_20mm(self):
        world = _DummyWorld(11.0, dist=90.0)  # target=-2.0 => gap=13mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(3))

    def test_align_turn_uses_5pct_when_x_axis_gap_below_50mm(self):
        world = _DummyWorld(25.0, dist=100.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(5))

    def test_align_turn_uses_fallback_when_x_axis_gap_at_or_above_50mm(self):
        world = _DummyWorld(60.0, dist=100.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(6))

    def test_align_loosened_x_axis_tolerance_when_far_keeps_dist_priority_for_minor_x_error(self):
        world = _DummyWorld(1.5, dist=420.0)  # slightly outside tol=1.0
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 400.0, "tol": 5.0},
                }
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertFalse(getattr(world, "_align_focus_dist", False))

    def test_align_prefers_dist_cmd_when_dist_is_more_egregious_than_x_axis(self):
        world = _DummyWorld(20.0, dist=120.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")

    def test_align_distance_uses_saved_duration_override_when_curve_available(self):
        with patch.object(
            helper_next,
            "_axis_curve_motion_plan",
            return_value={
                "cmd": "f",
                "score": 5,
                "duration_override_ms": 380,
                "predicted_distance_mm": 3.5,
                "equation": "distance_mm = -1.94 + 0.013024 * duration_ms",
            },
        ) as mock_curve:
            act = helper_next.select_align_brick_next_act(
                process_rules=self._rules(),
                learned_rules={},
                step="ALIGN_BRICK",
                x_axis_mm=-2.0,
                y_axis_mm=0.0,
                dist_mm=120.0,
                visible=True,
                duration_s=0.05,
            )

        self.assertEqual(act.get("worst_metric"), "dist")
        self.assertEqual(act.get("cmd"), "f")
        self.assertEqual(act.get("duration_override_ms"), 380)
        self.assertEqual(act.get("score"), 5)
        mock_curve.assert_any_call("dist", 40.0, fallback_score=5)

    def test_dist_axis_curve_motion_plan_prefers_calibrated_distance_profile_for_large_gap(self):
        calibrated = {
            "axis": "dist",
            "cmd": "f",
            "gap_mm": 20.0,
            "score": 1,
            "speed_score_pct": 1.0,
            "duration_override_ms": 400,
            "predicted_distance_mm": 3.2,
            "source": "aruco_marker_calibration",
        }

        with patch.object(
            helper_next.helper_next2,
            "calibrated_axis_motion_for_error",
            return_value=calibrated,
        ) as mock_calibrated:
            plan = helper_next._axis_curve_motion_plan("dist", 20.0, fallback_score=5)

        self.assertEqual(plan, calibrated)
        mock_calibrated.assert_called_once_with(axis="dist", err_mm=20.0)

    def test_align_prioritizes_dist_when_gap_exceeds_150mm_even_if_x_axis_off(self):
        world = _DummyWorld(20.0, dist=300.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

    def test_align_dist_focus_persists_until_gap_below_100mm(self):
        world = _DummyWorld(20.0, dist=300.0)

        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

        world.brick["dist"] = 190.0  # gap ~110mm: keep focusing dist
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

        world.brick["dist"] = 170.0  # gap ~90mm: dist is still more egregious than x-axis
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertFalse(getattr(world, "_align_focus_dist", False))

    def test_align_speed_score_never_exceeds_hard_cap(self):
        world = _DummyWorld(0.0, dist=500.0)
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 40.0, "tol": 1.0},
                }
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertLessEqual(int(analytics.get("speed_score") or 0), 20)

    def test_position_brick_respects_step_max_speed_score(self):
        world = _DummyWorld(0.0, dist=220.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertLessEqual(int(analytics.get("speed_score") or 0), 10)

    def test_position_brick_uses_5pct_when_dist_below_80mm(self):
        world = _DummyWorld(0.0, dist=79.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertEqual(int(analytics.get("speed_score") or 0), 5)

    def test_position_brick_uses_1pct_when_dist_below_60mm(self):
        world = _DummyWorld(0.0, dist=59.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_1pct_when_x_axis_gap_below_1p5mm(self):
        world = _DummyWorld(1.0, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_1pct_when_x_axis_gap_between_0p5_and_2mm(self):
        world = _DummyWorld(1.8, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_1pct_when_x_axis_gap_below_4mm_near_dist_gate(self):
        world = _DummyWorld(3.0, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_1pct_when_x_axis_gap_is_exactly_2mm(self):
        world = _DummyWorld(2.0, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_1pct_when_x_axis_gap_is_exactly_4mm_near_dist_gate(self):
        world = _DummyWorld(4.0, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(
            int(analytics.get("speed_score") or 0),
            telemetry_robot.normalize_speed_score(1),
        )

    def test_position_brick_turn_uses_3pct_when_x_axis_gap_below_20mm(self):
        world = _DummyWorld(19.0, dist=70.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), 3)

    def test_position_brick_turn_uses_5pct_when_x_axis_gap_below_50mm(self):
        world = _DummyWorld(49.0, dist=70.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), 5)

    def test_align_and_position_share_turn_gap_speed_bands(self):
        world = _DummyWorld(19.0, dist=70.0)
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                }
            },
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            },
        }
        align = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        position = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(align.get("cmd"), ("l", "r"))
        self.assertIn(position.get("cmd"), ("l", "r"))
        self.assertEqual(int(align.get("speed_score") or 0), 3)
        self.assertEqual(int(position.get("speed_score") or 0), 3)

    def test_align_turn_caps_to_2pct_when_dist_gate_is_near(self):
        world = _DummyWorld(19.0, dist=80.0)  # dist target in _rules is 80.0
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), 2)

    def test_position_turn_caps_to_2pct_when_dist_gate_is_near(self):
        world = _DummyWorld(19.0, dist=48.0)
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), 2)

    def test_align_and_position_share_dist_gap_speed_bands(self):
        world = _DummyWorld(0.0, dist=79.0)
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                }
            },
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            },
        }
        align = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        position = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(align.get("cmd"), "f")
        self.assertEqual(position.get("cmd"), "f")
        self.assertEqual(int(align.get("speed_score") or 0), 5)
        self.assertEqual(int(position.get("speed_score") or 0), 5)


if __name__ == "__main__":
    unittest.main()
