import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next


class _DummyWorld:
    def __init__(self, *, x_axis=19.0, dist=48.0, visible=True):
        self.brick = {
            "visible": bool(visible),
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": float(dist),
        }


class TestHelperNextAlignmentPolicyGeneralization(unittest.TestCase):
    def test_turn_score_mode_is_driven_by_world_model_align_policy(self):
        world = _DummyWorld(x_axis=19.0, dist=48.0, visible=True)
        rules = {
            "CUSTOM_ONE_SHOT": {
                "align_policy": {
                    "turn_score_mode": "x_axis_one_shot_curve",
                    "slow_fast_mm_scale": 0.25,
                },
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            },
            "CUSTOM_SHARED": {
                "align_policy": {
                    "turn_score_mode": "shared_turn_bands",
                    "turn_fallback_score": 6,
                    "slow_fast_mm_scale": 0.25,
                },
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            },
        }

        one_shot = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="CUSTOM_ONE_SHOT",
            duration_s=0.05,
        )
        shared = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="CUSTOM_SHARED",
            duration_s=0.05,
        )

        self.assertIn(one_shot.get("cmd"), ("l", "r"))
        self.assertIn(shared.get("cmd"), ("l", "r"))
        self.assertGreater(int(one_shot.get("speed_score") or 0), int(shared.get("speed_score") or 0))

    def test_helper_next_has_no_hardcoded_step_name_branches_for_align_logic(self):
        src = (Path(__file__).resolve().parents[1] / "helper_next.py").read_text()
        forbidden_snippets = (
            '== "ALIGN_BRICK"',
            '== "POSITION_BRICK"',
            '== "FIND_BRICK"',
            'in ("ALIGN_BRICK",',
            'in ("POSITION_BRICK",',
            'in ("FIND_BRICK",',
            'in {"ALIGN_BRICK"',
            'in {"POSITION_BRICK"',
            'in {"FIND_BRICK"',
        )
        for snippet in forbidden_snippets:
            self.assertNotIn(snippet, src, msg=f"Found forbidden step-specific branch snippet: {snippet}")


if __name__ == "__main__":
    unittest.main()
