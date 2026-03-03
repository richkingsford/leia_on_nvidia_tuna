import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class _DummyWorld:
    def __init__(self, process_rules=None, *, visible=False):
        self.process_rules = process_rules or {}
        self._visible_speed_cycle = 0
        self.brick = {"visible": bool(visible)}
        self.wall_height_mm = None
        self.wall_height_bricks = None
        self.brick_supply_height_mm = None
        self.brick_supply_height_bricks = None
        self.lift_height = 0.0


class TestTelemetryProcessFindBrickTurnSpeedPolicy(unittest.TestCase):
    def test_find_brick_turn_check_phase_caps_to_min_score(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "l",
            25,
            phase="check",
        )
        expected = telemetry_robot.normalize_speed_score(telemetry_robot.SPEED_SCORE_MIN)
        self.assertEqual(score, expected)

    def test_find_brick_turn_move_phase_caps_to_two_percent(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "r",
            25,
            phase="move",
        )
        self.assertEqual(score, telemetry_robot.normalize_speed_score(2))

    def test_find_brick_turn_demo_phase_preserves_demo_score(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "r",
            9,
            phase="demo",
        )
        self.assertEqual(score, telemetry_robot.normalize_speed_score(9))

    def test_find_brick_turn_search_phase_preserves_search_score(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "r",
            20,
            phase="search",
        )
        self.assertEqual(score, telemetry_robot.normalize_speed_score(20))

    def test_non_turn_command_is_unchanged(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "f",
            9,
            phase="move",
        )
        self.assertEqual(score, 9)

    def test_other_steps_are_unchanged(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "ALIGN_BRICK",
            "l",
            9,
            phase="move",
        )
        self.assertEqual(score, 9)

    def test_visible_only_replay_speed_tier_disabled_without_config(self):
        world = _DummyWorld(
            process_rules={
                "EXIT_WALL": {
                    "success_gates": {"visible": {"min": False}},
                }
            }
        )
        tier = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")
        self.assertIsNone(tier)

    def test_visible_only_replay_speed_tier_progression_uses_normal_standard_fast(self):
        world = _DummyWorld(
            process_rules={
                "EXIT_WALL": {
                    "success_gates": {"visible": {"min": False}},
                    "visible_only_speed_tiers": {
                        "enabled": True,
                        "normal": 2,
                        "standard": 6,
                        "fast": 25,
                    },
                }
            }
        )
        world._visible_speed_cycle = 0
        tier0 = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")
        world._visible_speed_cycle = 1
        tier1 = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")
        world._visible_speed_cycle = 2
        tier2 = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")
        world._visible_speed_cycle = 9
        tier9 = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")

        self.assertEqual(tier0.get("label"), "normal")
        self.assertEqual(int(tier0.get("score") or 0), telemetry_robot.normalize_speed_score(2))
        self.assertEqual(tier1.get("label"), "standard")
        self.assertEqual(int(tier1.get("score") or 0), telemetry_robot.normalize_speed_score(6))
        self.assertEqual(tier2.get("label"), "fast")
        self.assertEqual(int(tier2.get("score") or 0), telemetry_robot.normalize_speed_score(25))
        self.assertEqual(tier9.get("label"), "fast")
        self.assertEqual(int(tier9.get("score") or 0), telemetry_robot.normalize_speed_score(25))

    def test_auto_action_detail_text_appends_context_note(self):
        detail = telemetry_process.auto_action_detail_text(
            "r",
            6,
            action_meta={
                "score_model": 6,
                "pwm": 110,
                "power": 0.33,
                "duration_ms": 120,
            },
            context_note="standard tier",
        )
        self.assertIn("standard tier", detail)

    def test_height_intel_override_find_brick_low_supply_uses_mast_down(self):
        world = _DummyWorld(
            process_rules={
                "FIND_BRICK": {
                    "height_intelligence": {
                        "enabled": True,
                        "source": "brick_supply_height",
                        "low_bricks_max": 1,
                        "high_bricks_min": 3,
                        "low_cmd": "d",
                        "high_cmd": "u",
                        "score": 1,
                    }
                }
            },
            visible=False,
        )
        world.brick_supply_height_bricks = 1
        world.brick_supply_height_mm = 44.0
        world.wall_height_bricks = 2
        world.wall_height_mm = 88.0

        cmd, score, note, used = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_BRICK",
            "r",
            6,
            phase="replay",
            log=False,
        )

        self.assertTrue(used)
        self.assertEqual(cmd, "d")
        self.assertEqual(score, telemetry_robot.normalize_speed_score(1))
        self.assertIn("height intel", str(note))

    def test_height_intel_override_find_wall_high_stack_uses_mast_up(self):
        world = _DummyWorld(
            process_rules={
                "FIND_WALL": {
                    "height_intelligence": {
                        "enabled": True,
                        "source": "wall_height",
                        "low_bricks_max": 2,
                        "high_bricks_min": 4,
                        "low_cmd": "d",
                        "high_cmd": "u",
                        "score": 1,
                    }
                }
            },
            visible=False,
        )
        world.wall_height_bricks = 5
        world.wall_height_mm = 220.0

        cmd, score, _note, used = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL",
            "l",
            6,
            phase="replay",
            log=False,
        )

        self.assertTrue(used)
        self.assertEqual(cmd, "u")
        self.assertEqual(score, telemetry_robot.normalize_speed_score(1))

    def test_height_intel_override_requires_known_height(self):
        world = _DummyWorld(
            process_rules={
                "FIND_WALL2": {
                    "height_intelligence": {
                        "enabled": True,
                        "source": "wall_height",
                        "low_bricks_max": 2,
                        "high_bricks_min": 4,
                        "low_cmd": "d",
                        "high_cmd": "u",
                        "score": 1,
                    }
                }
            },
            visible=False,
        )

        cmd, score, note, used = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL2",
            "r",
            6,
            phase="replay",
            log=False,
        )

        self.assertFalse(used)
        self.assertEqual(cmd, "r")
        self.assertEqual(score, 6)
        self.assertIsNone(note)

    def test_apply_height_snapshot_from_step_updates_supply_height(self):
        world = _DummyWorld(
            process_rules={
                "FIND_TOPMOST_BRICK": {
                    "height_snapshot": {
                        "enabled": True,
                        "target": "brick_supply_height",
                        "source": "lift_height",
                        "min_bricks": 1,
                        "max_bricks": 20,
                    }
                }
            }
        )
        world.lift_height = 91.0

        result = telemetry_process.apply_height_snapshot_from_step(
            world,
            "FIND_TOPMOST_BRICK",
            log=False,
        )

        self.assertTrue(bool(result.get("applied")))
        self.assertEqual(int(result.get("bricks") or 0), 2)
        self.assertEqual(int(world.brick_supply_height_bricks or 0), 2)
        self.assertAlmostEqual(float(world.brick_supply_height_mm or 0.0), 88.0, places=3)


if __name__ == "__main__":
    unittest.main()
