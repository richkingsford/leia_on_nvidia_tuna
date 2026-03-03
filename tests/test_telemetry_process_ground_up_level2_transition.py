import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "FIND_TOPMOST_BRICK_WALL": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_command": "d",
                    "ground_reset_score": 100,
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
                        "command": "d",
                        "score": 100,
                        "true_down_acts_required": 3,
                        "max_acts": 16,
                    },
                }
            }
        }
        self.brick = {
            "inCrosshairs": None,
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

    def test_success_yes_and_no_streak_transitions_to_descend(self):
        world = _DummyWorld()
        robot = _DummyRobot()

        # First update is ground check (already grounded), then mast-up reads YES once
        # followed by NO x3, then post-success descend reads YES x3.
        sequence = [
            (None, 40.0),
            (True, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
            (True, 110.0),
            (True, 110.0),
            (True, 110.0),
        ]

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (True, 110.0)
            world_obj.brick["inCrosshairs"] = crosshair
            world_obj.brick["y_axis"] = y_val
            world_obj.brick["offset_y"] = y_val

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "send_robot_command", return_value={}), \
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

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("handled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_success_yes_and_no_streak_completes_immediately_when_descend_disabled(self):
        world = _DummyWorld()
        world.process_rules["FIND_TOPMOST_BRICK_WALL"]["topmost_crosshair_exception"]["post_success_descend"]["enabled"] = False
        robot = _DummyRobot()

        sequence = [
            (None, 40.0),
            (True, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 120.0)
            world_obj.brick["inCrosshairs"] = crosshair
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

    def test_find_topmost_brick_descend_requires_true_then_two_false_down_acts(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_command": "d",
                    "ground_reset_score": 100,
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
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "completion_mode": "true_then_false_streak",
                        "false_after_true_down_acts_required": 2,
                        "true_down_acts_required": 3,
                        "max_acts": 16,
                    },
                }
            }
        }
        robot = _DummyRobot()

        # Ground check (already grounded), mast-up YES once then NO x3 to hit level2,
        # then descend sees YES then NO x2 and succeeds.
        sequence = [
            (None, 40.0),
            (True, 120.0),
            (False, 120.0),
            (False, 120.0),
            (False, 120.0),
            (True, 110.0),
            (False, 110.0),
            (False, 110.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 110.0)
            world_obj.brick["inCrosshairs"] = crosshair
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
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("handled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("u"), 4)
        self.assertEqual(sent_cmds.count("d"), 3)

    def test_post_ground_reset_handoff_is_phase_bridge_not_terminal_success(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "topmost_crosshair_exception": {
                    "enabled": True,
                    "ground_up_level2_enabled": True,
                    "ground_reset_command": "d",
                    "ground_reset_score": 100,
                    "ground_reset_min_acts": 2,
                    "ground_reset_max_acts": 12,
                    "ground_level_y_axis_max": 50.0,
                    "mast_up_command": "u",
                    "mast_up_score": 100,
                    "mast_up_max_acts": 2,
                    "require_true_before_false": False,
                    "false_up_acts_required": 1,
                    "false_confirm_frames": 1,
                    "post_ground_reset_handoff": {
                        "enabled": True,
                        "step": "BRICK_LOCK",
                    },
                    "post_success_descend": {
                        "enabled": False,
                    },
                }
            }
        }
        robot = _DummyRobot()

        # Ground check sees we're already grounded, but min acts still applies.
        # Handoff is an in-step phase bridge only; mast-up still runs and drives success.
        sequence = [
            (None, 40.0),
            (False, 120.0),
        ]
        sent_cmds = []

        def _fake_update_world_from_vision(world_obj, vision_obj, log=True):
            _ = (vision_obj, log)
            if sequence:
                crosshair, y_val = sequence.pop(0)
            else:
                crosshair, y_val = (False, 40.0)
            world_obj.brick["inCrosshairs"] = crosshair
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
                step="FIND_TOPMOST_BRICK",
                robot=robot,
                observer=None,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("handled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason")), "success gate + level2")
        self.assertEqual(sent_cmds.count("d"), 2)
        self.assertEqual(sent_cmds.count("u"), 1)


if __name__ == "__main__":
    unittest.main()
