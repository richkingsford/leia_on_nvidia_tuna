import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestDefaultTolerances(unittest.TestCase):
    def test_derive_success_gates_uses_2_3_mm_min_floor(self):
        success_segments = {
            "ALIGN_BRICK": [
                {
                    "states": [
                        {
                            "timestamp": 1.0,
                            "brick": {
                                "visible": True,
                                "x_axis": -3.0,
                                "y_axis": 2.0,
                                "dist": 100.0,
                            },
                        },
                        {
                            "timestamp": 1.1,
                            "brick": {
                                "visible": True,
                                "x_axis": -2.5,
                                "y_axis": 2.4,
                                "dist": 100.8,
                            },
                        },
                    ]
                }
            ]
        }

        gates = telemetry_process.derive_success_gates(
            success_segments,
            scale_by_step={},
            step_rules={},
        )

        align_gates = gates.get("ALIGN_BRICK") or {}
        self.assertEqual((align_gates.get("yAxis_offset_abs") or {}).get("tol"), 2.3)
        self.assertEqual((align_gates.get("dist") or {}).get("tol"), 2.3)

    def test_process_model_align_and_position_x_dist_tolerances_are_5_mm(self):
        model = json.loads(Path("world_model_process.json").read_text())
        steps = model.get("steps") or {}

        self.assertEqual(
            (((steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}).get("xAxis_offset_abs") or {}).get("tol"),
            5.0,
        )
        self.assertEqual(
            (((steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}).get("yAxis_offset_abs") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            5.0,
        )
        self.assertEqual(
            (((steps.get("POSITION_BRICK") or {}).get("success_gates") or {}).get("xAxis_offset_abs") or {}).get("tol"),
            5.0,
        )
        self.assertEqual(
            (((steps.get("SEAT_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("POSITION_BRICK") or {}).get("success_gates") or {}).get("yAxis_offset_abs") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("POSITION_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            5.0,
        )
        self.assertIs(((steps.get("ALIGN_BRICK") or {}).get("lock_success_gates")), True)
        self.assertIs(((steps.get("POSITION_BRICK") or {}).get("lock_success_gates")), True)

    def test_process_model_brick_lock_is_visible_only_without_pre_align_descend(self):
        model = json.loads(Path("world_model_process.json").read_text())
        steps = model.get("steps") or {}
        brick_lock = steps.get("BRICK_LOCK") or {}
        success_gates = brick_lock.get("success_gates") or {}
        pre_align_descend = brick_lock.get("pre_align_descend") or {}

        self.assertEqual(success_gates, {"visible": {"min": True}})
        self.assertIs(pre_align_descend.get("enabled"), False)

    def test_process_model_brick_lock_wall_uses_visible_seek_then_three_down_pulses(self):
        model = json.loads(Path("world_model_process.json").read_text())
        steps = model.get("steps") or {}
        brick_lock = steps.get("BRICK_LOCK") or {}
        brick_lock_wall = steps.get("BRICK_LOCK_WALL") or {}
        wall_pre_align_descend = brick_lock_wall.get("pre_align_descend") or {}
        wall_post_descend = brick_lock_wall.get("post_success_descend") or {}
        wall_post_follow = brick_lock_wall.get("post_success_follow_through") or {}
        brick_lock_post_descend = brick_lock.get("post_success_descend") or {}

        self.assertEqual(brick_lock_wall.get("success_gates"), brick_lock.get("success_gates"))
        self.assertIs(wall_pre_align_descend.get("enabled"), True)
        self.assertIs(wall_pre_align_descend.get("consistent_observation_require_non_wait"), False)
        self.assertEqual(wall_pre_align_descend.get("completion_mode"), "visible_true_streak")
        self.assertEqual(wall_pre_align_descend.get("visible_true_required"), 1)
        self.assertEqual(wall_pre_align_descend.get("exclude_when_active_steps"), [])
        self.assertEqual(wall_pre_align_descend.get("exclude_when_active_step_numbers"), [])
        self.assertIs(wall_post_descend.get("enabled"), False)
        self.assertIs(brick_lock_post_descend.get("enabled"), True)
        self.assertIs(wall_post_follow.get("enabled"), True)
        self.assertEqual(wall_post_follow.get("command"), "d")
        self.assertEqual(wall_post_follow.get("acts"), 3)
