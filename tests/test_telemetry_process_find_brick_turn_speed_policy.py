import json
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
    def test_world_model_find_brick_visible_false_search_turns_then_backs_off(self):
        root = Path(__file__).resolve().parents[1]
        process_model = json.loads((root / "world_model_process.json").read_text())
        steps = (process_model.get("steps") or {}) if isinstance(process_model, dict) else {}
        find_brick_cfg = (steps.get("FIND_BRICK") or {}) if isinstance(steps, dict) else {}
        search_cfg = (
            find_brick_cfg.get("search_visible_false_speed_cycle") or {}
            if isinstance(find_brick_cfg, dict)
            else {}
        )

        self.assertEqual(search_cfg.get("commands"), ["r", "b"])
        self.assertEqual(int((search_cfg.get("command_scores") or {}).get("b") or 0), 10)

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

    def test_visible_only_replay_speed_tier_includes_duration_override(self):
        world = _DummyWorld(
            process_rules={
                "EXIT_WALL": {
                    "success_gates": {"visible": {"min": False}},
                    "visible_only_speed_tiers": {
                        "enabled": True,
                        "normal": 2,
                        "standard": 2,
                        "fast": 2,
                        "duration_override_ms": 1500,
                    },
                }
            }
        )

        tier = telemetry_process._visible_only_replay_speed_tier(world, "EXIT_WALL", cmd="r")

        self.assertIsInstance(tier, dict)
        self.assertEqual(int(tier.get("score") or 0), telemetry_robot.normalize_speed_score(2))
        self.assertEqual(int(tier.get("duration_override_ms") or 0), 1500)

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

    def test_find_wall2_height_intel_replay_phase_stays_disabled(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        world = _DummyWorld(process_rules=steps or {}, visible=False)
        world.wall_height_bricks = 3
        world.wall_height_mm = 132.0
        world.brick_supply_height_bricks = 7
        world.brick_supply_height_mm = 308.0

        cmd, score, note, used = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL2",
            "l",
            6,
            phase="replay",
            log=False,
        )

        self.assertFalse(used)
        self.assertEqual(cmd, "l")
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

    def test_apply_height_inventory_adjustment_from_step_decrements_supply_height(self):
        world = _DummyWorld(
            process_rules={
                "ELEVATE_BRICK": {
                    "height_inventory_adjustment": {
                        "enabled": True,
                        "target": "brick_supply_height",
                        "bricks_delta": -1,
                        "min_bricks": 0,
                        "max_bricks": 20,
                    }
                }
            }
        )
        world.brick_supply_height_bricks = 7
        world.brick_supply_height_mm = 308.0

        result = telemetry_process.apply_height_inventory_adjustment_from_step(
            world,
            "ELEVATE_BRICK",
            log=False,
        )

        self.assertTrue(bool(result.get("applied")))
        self.assertEqual(int(result.get("previous_bricks") or 0), 7)
        self.assertEqual(int(result.get("bricks") or 0), 6)
        self.assertEqual(int(world.brick_supply_height_bricks or 0), 6)
        self.assertAlmostEqual(float(world.brick_supply_height_mm or 0.0), 264.0, places=3)

    def test_apply_height_inventory_adjustment_from_step_increments_wall_height(self):
        world = _DummyWorld(
            process_rules={
                "RETREAT": {
                    "height_inventory_adjustment": {
                        "enabled": True,
                        "target": "wall_height",
                        "bricks_delta": 1,
                        "min_bricks": 0,
                        "max_bricks": 20,
                    }
                }
            }
        )
        world.wall_height_bricks = 4
        world.wall_height_mm = 176.0

        result = telemetry_process.apply_height_inventory_adjustment_from_step(
            world,
            "RETREAT",
            log=False,
        )

        self.assertTrue(bool(result.get("applied")))
        self.assertEqual(int(result.get("previous_bricks") or 0), 4)
        self.assertEqual(int(result.get("bricks") or 0), 5)
        self.assertEqual(int(world.wall_height_bricks or 0), 5)
        self.assertAlmostEqual(float(world.wall_height_mm or 0.0), 220.0, places=3)


if __name__ == "__main__":
    unittest.main()
