import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _DummyWorld:
    def __init__(self, *, max_acts=4):
        self.process_rules = {
            "RETREAT": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                },
                "post_success_bottom_discovery": {
                    "enabled": True,
                    "command": "d",
                    "score": 100,
                    "max_acts": int(max_acts),
                    "consecutive_no_required": 1,
                    "require_visible_for_confirm": True,
                    "reset_on_skipped_observation": True,
                    "fail_on_skipped_observation": False,
                    "fail_on_unconfirmed": True,
                    "confidence_gates_observation": False,
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 124.69,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 95.0,
            "brickBelow": True,
            "brick_below_raw": True,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestTelemetryProcessPostSuccessBottomDiscovery(unittest.TestCase):
    def test_process_model_retreat_enables_post_success_bottom_discovery(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("RETREAT") if isinstance(steps, dict) else {}
        post_cfg = (step_cfg or {}).get("post_success_bottom_discovery") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(post_cfg, dict)
        self.assertTrue(bool(post_cfg.get("enabled")))
        self.assertEqual(int(post_cfg.get("score") or 0), 100)

    def test_retreat_runs_post_success_bottom_discovery_until_brick_below_false(self):
        world = _DummyWorld(max_acts=4)
        robot = _DummyRobot()
        sent_cmds = []

        def _fake_update_world(world_obj, _vision_obj, log=True):
            _ = log
            world_obj.brick["visible"] = True
            world_obj.brick["confidence"] = 95.0
            if len(sent_cmds) >= 1:
                world_obj.brick["brickBelow"] = False
                world_obj.brick["brick_below_raw"] = False
            else:
                world_obj.brick["brickBelow"] = True
                world_obj.brick["brick_below_raw"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            sent_cmds.append((str(cmd), int(kwargs.get("speed_score") or 0)))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="RETREAT",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(sent_cmds, [("d", 100)])
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_retreat_post_success_bottom_discovery_failure_fails_step(self):
        world = _DummyWorld(max_acts=2)
        robot = _DummyRobot()
        sent_cmds = []

        def _fake_update_world(world_obj, _vision_obj, log=True):
            _ = log
            world_obj.brick["visible"] = True
            world_obj.brick["confidence"] = 95.0
            world_obj.brick["brickBelow"] = True
            world_obj.brick["brick_below_raw"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            sent_cmds.append((str(cmd), int(kwargs.get("speed_score") or 0)))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="RETREAT",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertFalse(ok)
        self.assertIn("post-success bottom discovery could not confirm brickBelow=NO", reason)
        self.assertEqual(sent_cmds, [("d", 100), ("d", 100)])
        self.assertGreaterEqual(robot.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
