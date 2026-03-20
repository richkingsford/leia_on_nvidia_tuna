import builtins
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.learned_rules = {}
        self._frame_id = 0
        self.wall_height_bricks = None
        self.wall_height_mm = None
        self.brick_supply_height_bricks = None
        self.brick_supply_height_mm = None
        self.process_rules = {
            "FIND_TOPMOST_BRICK_WALL": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": True,
                    "ground_reset_command": "d",
                    "ground_reset_score": 100,
                    "ground_reset_min_acts": 0,
                    "ground_reset_max_acts": 12,
                    "ground_level_y_axis_max": 50.0,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 16,
                    "require_true_before_false": True,
                    "min_true_observations": 1,
                    "false_up_acts_required": 3,
                    "false_confirm_frames": 1,
                    "post_success_descend": {
                        "enabled": False,
                    },
                }
            }
        }
        self.brick = {
            "inCrosshairs": None,
            "brickAbove": False,
            "confidence": 99.0,
            "y_axis": 40.0,
            "offset_y": 40.0,
        }


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class TestGroundUpLevel2Transition(unittest.TestCase):
    def test_non_topmost_steps_do_not_run_ground_up_level2_exception(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "mast_up_command": "u",
                    "mast_up_max_acts": 3,
                }
            }
        }

        result = telemetry_process._run_ground_up_level2_exception(
            world,
            vision=object(),
            step="FIND_WALL2",
            robot=_DummyRobot(),
            observer=None,
            confirm_callback=None,
            align_silent=True,
        )

        self.assertFalse(bool(result.get("enabled")))
        self.assertFalse(bool(result.get("handled")))

    def test_wall_path_still_succeeds_with_level2_and_no_descend(self):
        world = _DummyWorld()
        robot = _DummyRobot()

        sequence = [
            (None, 40.0),  # ground check already satisfied
            (True, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 120.0)
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = y_val
            world_obj.brick["offset_y"] = y_val

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=True), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK_WALL",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("d"), 0)
        self.assertEqual(sent_cmds.count("u"), 4)
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_find_topmost_brick_runs_bottom_discovery_before_mast_up(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "bottom_brick_discovery": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "max_acts": 4,
                        "consecutive_no_required": 3,
                        "require_visible_for_confirm": True,
                        "reset_on_skipped_observation": True,
                        "confidence_gates_observation": False,
                    },
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "crosshair_transition_brick_count": {
                        "enabled": True,
                        "target": "brick_supply_height",
                        "min_bricks": 1,
                        "max_bricks": 20,
                    },
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        sequence = [
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": False},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {"brick_below": False, "in_crosshairs": False}
            world_obj.brick["visible"] = True
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["brickAbove"] = False
            world_obj.brick["brickBelow"] = sample["brick_below"]
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds[:3], ["d", "d", "d"])
        self.assertEqual(sent_cmds[3:], ["u", "u"])

    def test_find_topmost_brick_bottom_discovery_can_complete_on_first_no(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "bottom_brick_discovery": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "max_acts": 4,
                        "consecutive_no_required": 1,
                        "require_visible_for_confirm": True,
                        "reset_on_skipped_observation": True,
                        "confidence_gates_observation": False,
                    },
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 4,
                    "crosshair_transition_brick_count": {
                        "enabled": True,
                        "target": "brick_supply_height",
                        "min_bricks": 1,
                        "max_bricks": 20,
                    },
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        sequence = [
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": False},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {"brick_below": False, "in_crosshairs": False}
            world_obj.brick["visible"] = True
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["brickAbove"] = False
            world_obj.brick["brickBelow"] = sample["brick_below"]
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds[0], "d")
        self.assertEqual(sent_cmds[1:], ["u", "u"])

    def test_find_topmost_brick_band_counter_counts_bricks_from_gap_runs(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "bottom_brick_discovery": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "max_acts": 2,
                        "consecutive_no_required": 1,
                        "require_visible_for_confirm": True,
                        "reset_on_skipped_observation": True,
                        "confidence_gates_observation": False,
                    },
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 8,
                    "crosshair_stack_band_count": {
                        "enabled": True,
                        "method": "brick_gap_bands",
                        "target": "brick_supply_height",
                        "min_bricks": 1,
                        "max_bricks": 20,
                        "initial_bricks": 2,
                    },
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        sequence = [
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": True},
            {"brick_below": False, "in_crosshairs": False},
            {"brick_below": False, "in_crosshairs": False},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {"brick_below": False, "in_crosshairs": False}
            world_obj.brick["visible"] = True
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["brickAbove"] = False
            world_obj.brick["brickBelow"] = sample["brick_below"]
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(int(world.brick_supply_height_bricks or 0), 5)
        self.assertAlmostEqual(float(world.brick_supply_height_mm or 0.0), 220.0, places=3)
        self.assertEqual(sent_cmds[0], "d")
        self.assertEqual(sent_cmds[1:], ["u", "u", "u", "u", "u", "u", "u"])

    def test_find_topmost_brick_bottom_discovery_uses_raw_brick_below_when_confirmed_lags(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "bottom_brick_discovery": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "max_acts": 4,
                        "consecutive_no_required": 3,
                        "require_visible_for_confirm": True,
                        "reset_on_skipped_observation": True,
                        "confidence_gates_observation": False,
                    },
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "crosshair_transition_brick_count": {
                        "enabled": True,
                        "target": "brick_supply_height",
                        "min_bricks": 1,
                        "max_bricks": 20,
                    },
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        print_lines = []
        sequence = [
            {"brick_below_raw": False, "brick_below_confirmed": True, "in_crosshairs": True},
            {"brick_below_raw": False, "brick_below_confirmed": True, "in_crosshairs": True},
            {"brick_below_raw": False, "brick_below_confirmed": True, "in_crosshairs": True},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": False},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": True},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": False},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": True},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": False},
            {"brick_below_raw": False, "brick_below_confirmed": False, "in_crosshairs": False},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {
                "brick_below_raw": False,
                "brick_below_confirmed": False,
                "in_crosshairs": False,
            }
            world_obj.brick["visible"] = True
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["brickAbove"] = False
            world_obj.brick["brickBelow"] = sample["brick_below_confirmed"]
            world_obj.brick["brick_below_raw"] = sample["brick_below_raw"]
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds[:3], ["d", "d", "d"])
        self.assertEqual(sent_cmds[3:], ["u", "u", "u", "u", "u", "u"])
        self.assertEqual(int(world.brick_supply_height_bricks or 0), 3)
        self.assertAlmostEqual(float(world.brick_supply_height_mm or 0.0), 132.0, places=3)
        self.assertTrue(
            any(
                "Bottom cycle" in line
                and "brickBelow=NO (confirmed=YES)" in line
                and "cycle_condition=" in line
                and "PASS" in line
                for line in print_lines
            )
        )

    def test_find_topmost_brick_bottom_discovery_fails_without_false_streak(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "bottom_brick_discovery": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "max_acts": 3,
                        "consecutive_no_required": 3,
                        "require_visible_for_confirm": True,
                        "reset_on_skipped_observation": True,
                        "confidence_gates_observation": False,
                    },
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 3,
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            world_obj.brick["visible"] = True
            world_obj.brick["inCrosshairs"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["brickBelow"] = True
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertFalse(bool(result.get("success")))
        self.assertIn("bottom discovery no-streak condition not satisfied", str(result.get("reason")))
        self.assertEqual(sent_cmds, ["d", "d", "d"])

    def test_find_topmost_brick_requires_mast_confidence_and_success_gate(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "brick_above": {"target": False, "tol": 0.0},
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 8,
                    "require_true_before_false": True,
                    "min_true_observations": 1,
                    "false_up_acts_required": 4,
                    "false_confirm_frames": 4,
                }
            }
        }
        robot = _DummyRobot()

        # One YES then NO x4 establishes confidence. Success gate only turns true on pulse 5.
        sequence = [
            (True, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 120.0)
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = y_val
            world_obj.brick["offset_y"] = y_val

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", side_effect=[False, False, False, False, True]), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 5)
        self.assertEqual(sent_cmds.count("d"), 0)
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_find_topmost_brick_fails_if_no_consecutive_confident_no_observations(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 5,
                    "require_true_before_false": False,
                    "false_up_acts_required": 2,
                    "false_confirm_frames": 2,
                }
            }
        }
        robot = _DummyRobot()

        sequence = [
            (True, 120.0),
            (True, 120.0),
            (True, 120.0),
            (True, 120.0),
            (True, 120.0),
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (True, 120.0)
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = y_val
            world_obj.brick["offset_y"] = y_val

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", return_value={}), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertFalse(bool(result.get("success")))
        self.assertIn("no-streak condition not satisfied", str(result.get("reason")))
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_find_topmost_brick_no_streak_waits_for_brick_above_false(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "brick_above": {"target": False, "tol": 0.0},
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 3,
                    "consecutive_no_required": 3,
                    "level2_use_full_gatecheck": False,
                }
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        sequence = [
            {"in_crosshairs": False, "brick_above": True, "y_axis": 120.0},
            {"in_crosshairs": False, "brick_above": True, "y_axis": 120.0},
            {"in_crosshairs": False, "brick_above": False, "y_axis": 120.0},
            {"in_crosshairs": False, "brick_above": False, "y_axis": 120.0},
            {"in_crosshairs": False, "brick_above": False, "y_axis": 120.0},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {"in_crosshairs": False, "brick_above": False, "y_axis": 120.0}
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["brickAbove"] = sample["brick_above"]
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = sample["y_axis"]
            world_obj.brick["offset_y"] = sample["y_axis"]

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False) as mock_gatecheck, \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 5)

    def test_find_topmost_brick_no_streak_ignores_brick_above_when_not_gated(self):
        world = _DummyWorld()
        world.lift_height = 0.0
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 3,
                    "consecutive_no_required": 3,
                    "level2_use_full_gatecheck": False,
                }
            }
        }
        robot = _DummyRobot()
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            world_obj.brick["inCrosshairs"] = False
            world_obj.brick["brickAbove"] = True
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0
            world_obj.lift_height = float(getattr(world_obj, "lift_height", 0.0) or 0.0) + 1.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False) as mock_gatecheck, \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 3)
        self.assertEqual(mock_gatecheck.call_count, 0)

    def test_find_topmost_brick_level2_requires_strict_consecutive_pass_cycles(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 5,
                    "consecutive_no_required": 5,
                    "level2_use_full_gatecheck": False,
                }
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        # Four PASS cycles, one FAIL, then one PASS. Strict consecutive rule
        # should reset after the FAIL, so this must not succeed.
        sequence = [False, False, False, False, True, False]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else False
            world_obj.brick["inCrosshairs"] = sample
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertFalse(bool(result.get("success")))
        self.assertIn("no-streak condition not satisfied", str(result.get("reason")))
        self.assertEqual(sent_cmds.count("u"), 6)

    def test_find_topmost_brick_level2_log_counter_resets_and_first_pass_is_one(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 8,
                    "false_up_acts_required": 5,
                    "consecutive_no_required": 5,
                    "level2_use_full_gatecheck": False,
                },
            }
        }
        robot = _DummyRobot()
        print_lines = []

        # False=PASS for per-cycle condition when inCrosshairs target=false, True=FAIL.
        # Gate-progress status should remain WAIT until threshold is met.
        # Expect streak logs: 1/5 WAIT, 2/5 WAIT, 0/5 FAIL, then 1..4 WAIT, 5/5 PASS.
        sequence = [False, False, True, False, False, False, False, False]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else False
            world_obj.brick["inCrosshairs"] = sample
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", return_value={}), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")

        cycle_lines = [line for line in print_lines if "level2 success gate check=" in line]
        self.assertEqual(len(cycle_lines), 8)
        self.assertIn("level2 success gate check=1/5", cycle_lines[0])
        self.assertIn(f"({telemetry_process.COLOR_ORANGE_BRIGHT}WAIT{telemetry_process.COLOR_RESET})", cycle_lines[0])
        self.assertIn("level2 success gate check=2/5", cycle_lines[1])
        self.assertIn("level2 success gate check=0/5", cycle_lines[2])
        self.assertIn(f"({telemetry_process.COLOR_RED}FAIL{telemetry_process.COLOR_RESET})", cycle_lines[2])
        self.assertIn("level2 success gate check=1/5", cycle_lines[3])
        self.assertIn("level2 success gate check=2/5", cycle_lines[4])
        self.assertIn("level2 success gate check=3/5", cycle_lines[5])
        self.assertIn("level2 success gate check=4/5", cycle_lines[6])
        self.assertIn("level2 success gate check=5/5", cycle_lines[7])
        self.assertIn(f"({telemetry_process.COLOR_GREEN}PASS{telemetry_process.COLOR_RESET})", cycle_lines[7])

    def test_find_topmost_brick_does_not_count_stale_observations(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 4,
                    "false_up_acts_required": 3,
                    "consecutive_no_required": 3,
                }
            }
        }
        robot = _DummyRobot()

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (world_obj, vision_obj, log)
            # Intentionally do not increment frame_id: all observations are stale.
            world_obj.brick["inCrosshairs"] = False
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", return_value={}), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertFalse(bool(result.get("success")))
        self.assertIn("last_obs_fresh=NO", str(result.get("reason")))

    def test_find_topmost_brick_level2_fail_fast_on_skipped_observation(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 3,
                    "consecutive_no_required": 3,
                    "false_confirm_frames": 3,
                    "level2_use_full_gatecheck": False,
                    "observation_confidence_min": 50.0,
                    "level2_confidence_gates_observation": True,
                    "level2_fail_on_skipped_observation": True,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        sequence = [
            {"in_crosshairs": False, "confidence": 95.0},
            {"in_crosshairs": False, "confidence": 95.0},
            {"in_crosshairs": False, "confidence": 95.0},
            {"in_crosshairs": False, "confidence": 0.0},
            {"in_crosshairs": False, "confidence": 0.0},
            {"in_crosshairs": False, "confidence": 0.0},
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            sample = sequence.pop(0) if sequence else {"in_crosshairs": False, "confidence": 0.0}
            world_obj.brick["inCrosshairs"] = sample["in_crosshairs"]
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = sample["confidence"]
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertFalse(bool(result.get("success")))
        self.assertIn("level2 observation skipped", str(result.get("reason")))
        self.assertIn("conf=0.0% < 50.0%", str(result.get("reason")))
        self.assertGreaterEqual(sent_cmds.count("u"), 1)
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_find_topmost_brick_level2_holds_until_obs3_stabilizes(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 1,
                    "false_up_acts_required": 1,
                    "consecutive_no_required": 1,
                    "false_confirm_frames": 3,
                    "level2_use_full_gatecheck": False,
                    "level2_require_visible_for_confirm": True,
                    "level2_reset_on_skipped_observation": True,
                    "level2_fail_on_skipped_observation": False,
                },
            }
        }
        world.brick["inCrosshairs"] = True
        world.brick["visible"] = True
        robot = _DummyRobot()
        sent_cmds = []
        print_lines = []
        sequence = [True, False, False, False]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            crosshair = sequence.pop(0) if sequence else False
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 95.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds, ["u"])
        self.assertEqual(len(sequence), 0)
        cycle_lines = [line for line in print_lines if "level2 success gate check=" in line]
        self.assertEqual(len(cycle_lines), 1)
        self.assertIn("obs3=[NO,NO,NO]", cycle_lines[0])
        self.assertIn("hold+1f", cycle_lines[0])
        self.assertNotIn("UNSTABLE", cycle_lines[0])

    def test_find_topmost_brick_level2_confidence_gate_waits_for_full_obs3_window(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 1,
                    "false_up_acts_required": 1,
                    "consecutive_no_required": 1,
                    "false_confirm_frames": 3,
                    "level2_use_full_gatecheck": False,
                    "observation_confidence_min": 50.0,
                    "level2_confidence_gates_observation": True,
                },
            }
        }
        world.brick["inCrosshairs"] = True
        world.brick["visible"] = True
        robot = _DummyRobot()
        sent_cmds = []
        confidence_sequence = [95.0, 0.0, 95.0, 95.0, 95.0]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            conf = confidence_sequence.pop(0) if confidence_sequence else 95.0
            world_obj.brick["inCrosshairs"] = False
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = conf
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds, ["u"])
        self.assertEqual(len(confidence_sequence), 0)

    def test_find_topmost_brick_wall_level2_counts_consistent_obs3_with_one_low_conf_tail(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK_WALL": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 2,
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "false_confirm_frames": 3,
                    "level2_use_full_gatecheck": False,
                    "observation_confidence_min": 50.0,
                    "level2_require_visible_for_confirm": False,
                    "level2_reset_on_skipped_observation": False,
                    "level2_fail_on_skipped_observation": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        confidence_sequence = [95.0, 96.0, 97.0, 98.0, 99.0, 0.0]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            conf = confidence_sequence.pop(0) if confidence_sequence else 95.0
            world_obj.brick["inCrosshairs"] = False
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = conf
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK_WALL",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 2)

    def test_find_topmost_brick_wall_level2_counts_low_conf_obs_when_conf_gate_disabled(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK_WALL": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 2,
                    "false_up_acts_required": 2,
                    "consecutive_no_required": 2,
                    "false_confirm_frames": 3,
                    "level2_use_full_gatecheck": False,
                    "observation_confidence_min": 50.0,
                    "level2_require_visible_for_confirm": False,
                    "level2_reset_on_skipped_observation": False,
                    "level2_fail_on_skipped_observation": False,
                    "level2_confidence_gates_observation": False,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        # Cycle 1 pre+obs3 and cycle 2 pre+obs3: all low confidence, stable NO.
        confidence_sequence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            conf = confidence_sequence.pop(0) if confidence_sequence else 0.0
            world_obj.brick["inCrosshairs"] = False
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = conf
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK_WALL",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 2)

    def test_handoff_defaults_disabled_without_explicit_enabled_true(self):
        world = _DummyWorld()
        world.learned_rules = {}
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": True,
                    "ground_reset_command": "d",
                    "ground_reset_score": 100,
                    "ground_reset_min_acts": 0,
                    "ground_reset_max_acts": 1,
                    "ground_level_y_axis_max": 50.0,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 2,
                    "require_true_before_false": False,
                    "false_up_acts_required": 1,
                    "false_confirm_frames": 1,
                    "post_ground_reset_handoff": {
                        "step": "BRICK_LOCK",
                        "acts": 1,
                    },
                }
            },
            "BRICK_LOCK": {
                "success_gates": {
                    "visible": {"min": True}
                }
            },
        }
        robot = _DummyRobot()
        sequence = [
            (None, 40.0),
            (False, 120.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 120.0)
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = y_val
            world_obj.brick["offset_y"] = y_val

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=True), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act") as mock_select, \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(sent_cmds.count("l"), 0)
        self.assertEqual(sent_cmds.count("r"), 0)
        self.assertEqual(mock_select.call_count, 0)

    def test_find_topmost_brick_completes_on_crosshair_drop_when_enabled(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 1,
                    "consecutive_no_required": 1,
                    "false_confirm_frames": 1,
                    "level2_use_full_gatecheck": False,
                    "complete_on_crosshair_drop": True,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        world.brick["inCrosshairs"] = True
        world.brick["visible"] = True

        sequence = [False, False, False]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            crosshair = sequence.pop(0) if sequence else False
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 1)

    def test_find_topmost_brick_crosshair_drop_counts_one_cycle_when_multiple_no_required(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "success_gates": {
                    "inCrosshairs": {"target": False, "tol": 0.0},
                },
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_enabled": False,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 6,
                    "false_up_acts_required": 4,
                    "consecutive_no_required": 4,
                    "false_confirm_frames": 1,
                    "level2_use_full_gatecheck": False,
                    "complete_on_crosshair_drop": True,
                },
            }
        }
        robot = _DummyRobot()
        sent_cmds = []
        print_lines = []
        world.brick["inCrosshairs"] = True
        world.brick["visible"] = True

        sequence = [False, False, False, False]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            crosshair = sequence.pop(0) if sequence else False
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["visible"] = True
            world_obj.brick["brickAbove"] = False
            world_obj.brick["confidence"] = 99.0
            world_obj.brick["y_axis"] = 120.0
            world_obj.brick["offset_y"] = 120.0

        def _fake_send_robot_command(*args, **kwargs):
            _ = kwargs
            sent_cmds.append(str(args[3]))
            return {}

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            result = telemetry_process._run_ground_up_level2_exception(
                world,
                vision=object(),
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 4)
        cycle_lines = [line for line in print_lines if "level2 success gate check=" in line]
        self.assertEqual(len(cycle_lines), 4)
        self.assertIn("level2 success gate check=1/4", cycle_lines[0])
        self.assertIn("crosshair_drop=YES", cycle_lines[0])
        self.assertIn("level2 success gate check=2/4", cycle_lines[1])
        self.assertIn("level2 success gate check=3/4", cycle_lines[2])
        self.assertIn("level2 success gate check=4/4", cycle_lines[3])


if __name__ == "__main__":
    unittest.main()
