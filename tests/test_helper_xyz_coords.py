import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.step_state = "FIND_BRICK"
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
        self.assertIn("Supply", svg)
        self.assertIn(">5<", svg)

    def test_workspace_renders_fixed_supply_stack_without_observation(self):
        world = _DummyWorld()

        state = helper_xyz_coords.sync_from_world(world, render=False)
        svg = helper_xyz_coords.render_workspace_svg(state)
        wall_pose = helper_xyz_coords.relative_object_pose(world, "wall")

        self.assertIn("Wall", svg)
        self.assertIn("Supply", svg)
        self.assertEqual(state["active_target"]["object_name"], "brick_supply")
        self.assertAlmostEqual(float(state["robot"]["theta_deg"]), -90.0, places=3)
        self.assertAlmostEqual(float(wall_pose["bearing_deg"]), 90.0, places=3)

    def test_wall_render_points_use_square_stack_footprint(self):
        wall = {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "theta_deg": 0.0,
            "render_footprint_mm": 54.0,
        }

        points = helper_xyz_coords._wall_render_points(wall)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        self.assertAlmostEqual(max(xs) - min(xs), 54.0, places=3)
        self.assertAlmostEqual(max(ys) - min(ys), 54.0, places=3)

    def test_holding_brick_state_shows_front_brick(self):
        world = _DummyWorld()

        helper_xyz_coords.set_holding_brick(world, True, render=False)
        state = helper_xyz_coords.workspace_snapshot(world)
        svg = helper_xyz_coords.render_workspace_svg(state)

        self.assertTrue(bool(state["held_brick"]["held"]))
        self.assertIn("Held Brick", svg)

    def test_reconcile_object_distance_moves_robot_not_supply(self):
        world = _DummyWorld()
        helper_xyz_coords.sync_from_world(world, render=False)
        supply_before = helper_xyz_coords.workspace_snapshot(world)["objects"]["brick_supply"].copy()

        state = helper_xyz_coords.reconcile_object_distance(world, "brick_supply", 30.0, render=False)
        pose = helper_xyz_coords.relative_object_pose(world, "brick_supply")

        self.assertAlmostEqual(float(state["objects"]["brick_supply"]["x_mm"]), float(supply_before["x_mm"]), places=3)
        self.assertAlmostEqual(float(state["objects"]["brick_supply"]["y_mm"]), float(supply_before["y_mm"]), places=3)
        self.assertNotAlmostEqual(float(state["robot"]["y_mm"]), 0.0, places=3)
        self.assertAlmostEqual(float(pose["range_mm"]), 33.0, places=3)

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

    def test_render_workspace_svg_shows_current_step_history_and_camera_glyph(self):
        world = _DummyWorld()
        helper_xyz_coords.sync_from_world(world, render=False)

        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="forward", duration_ms=400, speed_score=50),
            delta=SimpleNamespace(dist_mm=40.0, rot_deg=0.0, lift_mm=0.0),
            render=False,
        )
        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="forward", duration_ms=250, speed_score=25),
            delta=SimpleNamespace(dist_mm=20.0, rot_deg=0.0, lift_mm=0.0),
            render=False,
        )
        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="backward", duration_ms=250, speed_score=15),
            delta=SimpleNamespace(dist_mm=10.0, rot_deg=0.0, lift_mm=0.0),
            render=False,
        )
        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="forward", duration_ms=250, speed_score=15),
            delta=SimpleNamespace(dist_mm=5.0, rot_deg=0.0, lift_mm=0.0),
            render=False,
        )
        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="mast_up", duration_ms=250, speed_score=5),
            delta=SimpleNamespace(dist_mm=0.0, rot_deg=0.0, lift_mm=8.0),
            render=False,
        )
        state = helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="backward", duration_ms=250, speed_score=15),
            delta=SimpleNamespace(dist_mm=5.0, rot_deg=0.0, lift_mm=0.0),
            render=False,
        )

        svg = helper_xyz_coords.render_workspace_svg(state)
        radii = re.findall(r'class="step-history-dot"[^>]* r="([0-9.]+)"', svg)

        self.assertIn('class="camera-dot"', svg)
        self.assertEqual(svg.count('class="step-history-dot"'), 5)
        self.assertEqual(radii[:2], ["2.0", "2.0"])
        self.assertEqual(radii[2:], ["6.0", "6.0", "6.0"])
        self.assertNotIn('id="grid"', svg)
        self.assertIn('data-trend="closer"', svg)
        self.assertIn('data-trend="further"', svg)

    def test_render_workspace_svg_hides_camera_dot_during_micro_adjust_phase(self):
        world = _DummyWorld()
        state = helper_xyz_coords.sync_from_world(world, render=False)
        state["micro_adjust_phase"] = True

        svg = helper_xyz_coords.render_workspace_svg(state)

        self.assertNotIn('class="camera-dot"', svg)

    def test_render_mast_svg_shows_y_axis_history_and_current_camera(self):
        world = _DummyWorld()
        world.wall_height_bricks = 1
        world.wall_height_mm = 44.0
        world.brick_supply_height_bricks = 5
        world.brick_supply_height_mm = 220.0
        world.brick["visible"] = True
        world.brick["dist"] = 120.0
        world.brick["y_axis"] = 18.0
        helper_xyz_coords.sync_from_world(world, reason="vision", render=False)

        world.lift_height = 20.0
        world.height_mm = 165.0
        world.brick["y_axis"] = 10.0
        helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="mast_up", duration_ms=250, speed_score=10),
            delta=SimpleNamespace(dist_mm=0.0, rot_deg=0.0, lift_mm=20.0),
            render=False,
        )

        world.brick["y_axis"] = 16.0
        helper_xyz_coords.sync_from_world(world, reason="vision", render=False)

        world.lift_height = 8.0
        world.height_mm = 150.0
        world.brick["y_axis"] = 6.0
        state = helper_xyz_coords.update_from_motion(
            world,
            event=SimpleNamespace(action_type="mast_down", duration_ms=250, speed_score=10),
            delta=SimpleNamespace(dist_mm=0.0, rot_deg=0.0, lift_mm=12.0),
            render=False,
        )

        mast_svg = helper_xyz_coords.render_mast_svg(state)

        self.assertIn("Mast View", mast_svg)
        self.assertIn('class="mast-history-dot"', mast_svg)
        self.assertIn('class="mast-camera-dot"', mast_svg)
        self.assertIn("Supply", mast_svg)
        self.assertIn("Wall", mast_svg)
        self.assertIn('data-trend="closer"', mast_svg)
        self.assertIn('data-trend="further"', mast_svg)

    def test_build_viewbox_uses_tighter_default_zoom(self):
        world = _DummyWorld()
        state = helper_xyz_coords.sync_from_world(world, render=False)

        min_x, max_x, min_y, max_y = helper_xyz_coords._build_viewbox(state)

        self.assertLess(float(max_x - min_x), 320.0)
        self.assertLess(float(max_y - min_y), 320.0)

    def test_vision_reconciles_robot_pose_against_fixed_supply_target(self):
        world = _DummyWorld()
        helper_xyz_coords.sync_from_world(world, render=False)
        world.brick["visible"] = True
        world.brick["dist"] = 100.0
        world.brick["x_axis"] = 0.0

        state = helper_xyz_coords.sync_from_world(world, reason="vision", render=False)
        pose = helper_xyz_coords.relative_object_pose(world, "brick_supply")

        self.assertAlmostEqual(float(state["objects"]["brick_supply"]["x_mm"]), 0.0, places=3)
        self.assertAlmostEqual(float(state["objects"]["brick_supply"]["y_mm"]), -180.0, places=3)
        self.assertAlmostEqual(float(pose["range_mm"]), 100.0, places=3)
        self.assertEqual(state["active_target"]["object_name"], "brick_supply")

    def test_step_mapping_switches_facing_target_between_supply_and_wall(self):
        world = _DummyWorld()

        supply_state = helper_xyz_coords.sync_from_world(world, render=False)
        self.assertEqual(supply_state["active_target"]["object_name"], "brick_supply")
        self.assertEqual(int(supply_state["active_target"]["step_number"]), 3)
        self.assertAlmostEqual(float(supply_state["robot"]["theta_deg"]), -90.0, places=3)

        world.step_state = "POSITION_BRICK"
        wall_state = helper_xyz_coords.sync_from_world(world, render=False)
        supply_pose = helper_xyz_coords.relative_object_pose(world, "brick_supply")
        self.assertEqual(wall_state["active_target"]["object_name"], "wall")
        self.assertEqual(int(wall_state["active_target"]["step_number"]), 14)
        self.assertAlmostEqual(float(wall_state["robot"]["theta_deg"]), 0.0, places=3)
        self.assertAlmostEqual(float(supply_pose["bearing_deg"]), -90.0, places=3)

    def test_robot_pose_is_clamped_outside_fixed_objects(self):
        world = _DummyWorld()

        state = helper_xyz_coords.observe_brick_supply(world, distance_mm=0.0, bearing_deg=0.0, render=False)
        supply = state["objects"]["brick_supply"]

        self.assertFalse(
            helper_xyz_coords._point_inside_object(
                float(state["robot"]["x_mm"]),
                float(state["robot"]["y_mm"]),
                supply,
                margin_mm=helper_xyz_coords.ROBOT_OBJECT_MARGIN_MM - 0.1,
            )
        )

    def test_world_model_motion_updates_xyz_workspace(self):
        world = WorldModel()
        helper_xyz_coords.ensure_workspace(world, render_enabled=False)

        event = MotionEvent("forward", speed_score=50, duration_ms=400)
        world.update_from_motion(event)
        state = helper_xyz_coords.workspace_snapshot(world)

        self.assertIsNotNone(state)
        self.assertAlmostEqual(float(state["raw_robot"]["x_mm"]), float(world.x), places=3)
        self.assertAlmostEqual(float(state["raw_robot"]["y_mm"]), float(world.y), places=3)
        self.assertNotAlmostEqual(float(state["robot"]["y_mm"]), 0.0, places=3)
        self.assertAlmostEqual(float(state["leia"]["x_mm"]), float(state["robot"]["x_mm"]), places=3)


if __name__ == "__main__":
    unittest.main()
