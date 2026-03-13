import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_xyz_coords
from telemetry_robot import MotionEvent, WorldModel


class _DummyWorld:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.lift_height = 0.0
        self.height_mm = 142.0
        self.brick_supply_height_bricks = None
        self.brick_supply_height_mm = None
        self.brick = {
            "visible": False,
            "dist": None,
            "x_axis": None,
            "y_axis": None,
            "confidence": None,
            "held": False,
        }
        self.wall = {
            "origin": None,
            "valid": False,
            "last_seen": None,
            "source": None,
        }


class HelperXyzCoordsTests(unittest.TestCase):
    def test_sync_from_world_tracks_robot_and_leia_pose(self):
        world = _DummyWorld()
        world.x = 45.0
        world.y = -18.0
        world.theta = 30.0
        world.lift_height = 12.0
        world.height_mm = 160.0

        state = helper_xyz_coords.sync_from_world(world, render=False)

        self.assertAlmostEqual(float(state["robot"]["x_mm"]), 45.0, places=3)
        self.assertAlmostEqual(float(state["robot"]["y_mm"]), -18.0, places=3)
        self.assertAlmostEqual(float(state["robot"]["theta_deg"]), 30.0, places=3)
        self.assertAlmostEqual(float(state["leia"]["z_mm"]), 160.0, places=3)

    def test_supply_count_renders_label_and_count(self):
        world = _DummyWorld()

        helper_xyz_coords.set_brick_supply_count(world, 5, render=False)
        state = helper_xyz_coords.workspace_snapshot(world)
        svg = helper_xyz_coords.render_workspace_svg(state)

        self.assertEqual(int(state["objects"]["brick_supply"]["count"]), 5)
        self.assertIn("Brick Supply", svg)
        self.assertIn(">5<", svg)

    def test_holding_brick_state_shows_front_brick(self):
        world = _DummyWorld()

        helper_xyz_coords.set_holding_brick(world, True, render=False)
        state = helper_xyz_coords.workspace_snapshot(world)
        svg = helper_xyz_coords.render_workspace_svg(state)

        self.assertTrue(bool(state["held_brick"]["held"]))
        self.assertIn("Held Brick", svg)

    def test_reconcile_object_distance_moves_supply_to_observed_range(self):
        world = _DummyWorld()
        helper_xyz_coords.sync_from_world(world, render=False)

        state = helper_xyz_coords.reconcile_object_distance(world, "brick_supply", 30.0, render=False)
        pose = helper_xyz_coords.relative_object_pose(world, "brick_supply")

        self.assertNotAlmostEqual(float(state["objects"]["brick_supply"]["x_mm"]), 140.0, places=3)
        self.assertAlmostEqual(float(pose["range_mm"]), 30.0, places=3)

    def test_plan_reverse_then_turn_uses_known_wall_distance(self):
        world = _DummyWorld()
        helper_xyz_coords.observe_wall(world, distance_mm=130.0, bearing_deg=180.0, render=False)

        plan = helper_xyz_coords.plan_reverse_then_turn_for_wall(
            world,
            turn_cmd="l",
            reverse_step_mm=30.0,
            turn_when_wall_within_mm=40.0,
        )

        self.assertTrue(bool(plan["ok"]))
        self.assertEqual(int(plan["reverse_acts"]), 3)
        self.assertEqual([action["cmd"] for action in plan["actions"][:-1]], ["b", "b", "b"])
        self.assertEqual(plan["actions"][-1]["cmd"], "l")

    def test_world_model_motion_updates_xyz_workspace(self):
        world = WorldModel()
        helper_xyz_coords.ensure_workspace(world, render_enabled=False)

        event = MotionEvent("forward", speed_score=50, duration_ms=400)
        world.update_from_motion(event)
        state = helper_xyz_coords.workspace_snapshot(world)

        self.assertIsNotNone(state)
        self.assertAlmostEqual(float(state["robot"]["x_mm"]), float(world.x), places=3)
        self.assertAlmostEqual(float(state["robot"]["y_mm"]), float(world.y), places=3)
        self.assertAlmostEqual(float(state["leia"]["x_mm"]), float(world.x), places=3)


if __name__ == "__main__":
    unittest.main()
