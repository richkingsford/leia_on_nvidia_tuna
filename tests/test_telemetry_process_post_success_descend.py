import builtins
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
    def __init__(self, *, max_acts=4, post_enabled=True, pre_enabled=False):
        self.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "visible": {"min": True},
                },
                "pre_align_descend": {
                    "enabled": bool(pre_enabled),
                    "command": "d",
                    "score": 100,
                    "confirm_frames": 1,
                    "completion_mode": "true_then_false_streak",
                    "false_after_true_down_acts_required": 1,
                    "max_acts": int(max_acts),
                },
                "post_success_descend": {
                    "enabled": bool(post_enabled),
                    "command": "d",
                    "score": 100,
                    "completion_mode": "true_then_false_streak",
                    "false_after_true_down_acts_required": 2,
                    "max_acts": int(max_acts),
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "inCrosshairs": None,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestTelemetryProcessPostSuccessDescend(unittest.TestCase):
    def test_brick_lock_pre_align_descend_uses_five_percent_after_seen_true(self):
        world = _DummyWorld(max_acts=4, pre_enabled=True, post_enabled=False)
        world.process_rules["BRICK_LOCK"]["pre_align_descend"]["score_after_seen_true"] = 5
        robot = _DummyRobot()
        sent_scores = []
        print_lines = []

        crosshair_sequence = [True, False]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if crosshair_sequence:
                world_obj.brick["inCrosshairs"] = crosshair_sequence.pop(0)
            world_obj.brick["visible"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            _ = cmd
            sent_scores.append(int(kwargs.get("speed_score") or 0))
            return {
                "cmd_sent": "d",
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(sent_scores, [5])
        self.assertTrue(
            any(
                "Pre-align descend pulse" in line
                and "speed=5%" in line
                for line in print_lines
            )
        )

    def test_brick_lock_runs_pre_align_descend_before_alignment(self):
        world = _DummyWorld(max_acts=4, pre_enabled=True, post_enabled=False)
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        # Pre-align descend phase should see YES, then a single NO, and handoff to align.
        crosshair_sequence = [True, False]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if crosshair_sequence:
                world_obj.brick["inCrosshairs"] = crosshair_sequence.pop(0)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, ["d"])
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertTrue(
            any("Pre-align descend pulse" in line for line in print_lines)
        )

    def test_brick_lock_pre_align_descend_accepts_final_pulse_boundary_success_and_logs_visible(self):
        world = _DummyWorld(max_acts=2, pre_enabled=True, post_enabled=False)
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []
        crosshair_sequence = [True, False]

        def _fake_refresh_world(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if crosshair_sequence:
                world_obj.brick["inCrosshairs"] = crosshair_sequence.pop(0)
            world_obj.brick["visible"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1
            return None

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_refresh_world), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_refresh_world), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, ["d"])
        self.assertTrue(
            any(
                "Pre-align descend pulse" in line
                and "visible=YES" in line
                and "false-after-true=" in line
                and "/1" in line
                and "gate=PASS" in line
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                "Pre-align descend complete: gate=PASS." in line
                for line in print_lines
            )
        )

    def test_brick_lock_pre_align_descend_excludes_active_step_12(self):
        world = _DummyWorld(max_acts=4, pre_enabled=True, post_enabled=False)
        world.process_rules["BRICK_LOCK"]["pre_align_descend"]["exclude_when_active_steps"] = [
            "FIND_TOPMOST_BRICK_WALL",
        ]
        world.step_state = telemetry_process.telemetry_robot_module.StepState.FIND_TOPMOST_BRICK_WALL
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True):
            _ = log
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, [])
        self.assertTrue(
            any(
                "Pre-align descend skipped: excluded while active step is #12 FIND_TOPMOST_BRICK_WALL."
                in line
                for line in print_lines
            )
        )

    def test_brick_lock_wall_runs_pre_align_descend_from_invisible_topmost_wall_handoff(self):
        world = _DummyWorld(max_acts=4, pre_enabled=False, post_enabled=False)
        world.process_rules["BRICK_LOCK_WALL"] = {
            "success_gates": {
                "visible": {"min": True},
            },
            "pre_align_descend": {
                "enabled": True,
                "command": "d",
                "score": 100,
                "score_after_seen_true": 2,
                "confirm_frames": 1,
                "require_consistent_observation": False,
                "consistent_observation_require_non_wait": False,
                "consistent_observation_max_checks": 4,
                "completion_mode": "visible_true_streak",
                "visible_true_required": 1,
                "false_after_true_down_acts_required": 1,
                "max_acts": 4,
                "exclude_when_active_steps": [],
                "exclude_when_active_step_numbers": [],
            },
            "post_success_descend": {
                "enabled": False,
            },
        }
        world.step_state = telemetry_process.telemetry_robot_module.StepState.FIND_TOPMOST_BRICK_WALL
        world.brick["visible"] = False
        world.brick["inCrosshairs"] = None
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        descend_states = [
            {"visible": False, "inCrosshairs": None},
            {"visible": True, "inCrosshairs": None},
        ]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if descend_states:
                next_state = descend_states.pop(0)
                world_obj.brick["visible"] = bool(next_state.get("visible"))
                world_obj.brick["inCrosshairs"] = next_state.get("inCrosshairs")
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(_kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_update_world_from_vision), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK_WALL",
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
        self.assertGreaterEqual(len(send_cmds), 1)
        self.assertTrue(all(cmd == "d" for cmd in send_cmds))
        self.assertTrue(
            any(
                "[BRICK_LOCK_WALL] Pre-align descend pulse" in line
                and "visible=NO" in line
                and "inCrosshairs=WAIT" in line
                for line in print_lines
            )
        )
        self.assertFalse(
            any(
                "Pre-align descend skipped: excluded while active step is #12 FIND_TOPMOST_BRICK_WALL."
                in line
                for line in print_lines
            )
        )

    def test_brick_lock_runs_post_success_descend_and_logs_clean_gate_words(self):
        world = _DummyWorld(max_acts=4)
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        # First update is align loop sampling; descend phase then sees YES, NO, NO.
        crosshair_sequence = [None, True, False, False]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True):
            _ = log
            if crosshair_sequence:
                world_obj.brick["inCrosshairs"] = crosshair_sequence.pop(0)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, ["d", "d", "d"])
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertTrue(
            any(
                ("[EXCEPTION] [BRICK_LOCK] Post-success descend pulse" in line)
                and ("gate=" in line)
                and (telemetry_process.COLOR_GREEN in line or telemetry_process.COLOR_RED in line)
                for line in print_lines
            )
        )

    def test_brick_lock_post_success_descend_failure_fails_step(self):
        world = _DummyWorld(max_acts=2)
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        # YES then only one NO is insufficient for false-after-true requirement of 2.
        crosshair_sequence = [None, True, False]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True):
            _ = log
            if crosshair_sequence:
                world_obj.brick["inCrosshairs"] = crosshair_sequence.pop(0)
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertIn("post-success descend could not confirm", reason)
        self.assertEqual(send_cmds, ["d", "d"])
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertTrue(
            any(
                "[EXCEPTION] [BRICK_LOCK] Post-success descend complete: gate=" in line
                and telemetry_process.COLOR_RED in line
                for line in print_lines
            )
        )

    def test_brick_lock_post_success_descend_reaches_y_axis_target_without_pause(self):
        world = _DummyWorld(max_acts=4)
        world.process_rules["BRICK_LOCK"]["post_success_descend"] = {
            "enabled": True,
            "command": "d",
            "score": 1,
            "completion_mode": "y_axis_target",
            "target_y_axis": 2,
            "tol": 0,
            "max_acts": 4,
        }
        world.brick["y_axis"] = 4.0
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []
        y_axis_sequence = [4.0, 3.0, 2.0]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if y_axis_sequence:
                world_obj.brick["y_axis"] = y_axis_sequence.pop(0)
            world_obj.brick["visible"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            send_cmds.append((str(cmd), int(kwargs.get("speed_score") or 0)))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "pause_after_exception") as mock_pause, \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None) as mock_sleep, \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, [("d", 1), ("d", 1)])
        mock_pause.assert_not_called()
        mock_sleep.assert_not_called()
        self.assertTrue(
            any(
                "Post-success descend start" in line
                and "D 1%" in line
                and "goal=y_axis<=+2.00mm" in line
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                "Post-success descend pulse" in line
                and "visible=YES" in line
                and "y_axis=+3.00mm" in line
                and "next=D 1%" in line
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                "Post-success descend pulse" in line
                and "visible=YES" in line
                and "y_axis=+2.00mm" in line
                and "gate=" in line
                and "next=STOP" in line
                for line in print_lines
            )
        )

    def test_brick_lock_post_success_descend_y_axis_target_failure_fails_step(self):
        world = _DummyWorld(max_acts=2)
        world.process_rules["BRICK_LOCK"]["post_success_descend"] = {
            "enabled": True,
            "command": "d",
            "score": 1,
            "completion_mode": "y_axis_target",
            "target_y_axis": 2,
            "tol": 0,
            "max_acts": 2,
        }
        world.brick["y_axis"] = 4.0
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []
        y_axis_sequence = [4.0, 3.5, 3.0]

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            if y_axis_sequence:
                world_obj.brick["y_axis"] = y_axis_sequence.pop(0)
            world_obj.brick["visible"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            send_cmds.append((str(cmd), int(kwargs.get("speed_score") or 0)))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process, "pause_after_exception") as mock_pause, \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None) as mock_sleep, \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertIn("post-success descend could not reach y_axis +2.00mm", reason)
        self.assertEqual(send_cmds, [("d", 1), ("d", 1)])
        mock_pause.assert_not_called()
        mock_sleep.assert_not_called()
        self.assertTrue(
            any(
                "Post-success descend pulse" in line
                and "visible=YES" in line
                and "y_axis=+3.50mm" in line
                and "next=D 1%" in line
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                "[EXCEPTION] [BRICK_LOCK] Post-success descend complete: gate=" in line
                and telemetry_process.COLOR_RED in line
                for line in print_lines
            )
        )

    def test_brick_lock_post_success_follow_through_sends_three_pulses(self):
        world = _DummyWorld(max_acts=2, post_enabled=False)
        world.process_rules["BRICK_LOCK"]["post_success_follow_through"] = {
            "enabled": True,
            "command": "d",
            "score": 100,
            "acts": 3,
        }
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=True, **_kwargs):
            _ = log
            world_obj.brick["visible"] = True
            world_obj._frame_id = int(getattr(world_obj, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            send_cmds.append((str(cmd), int(kwargs.get("speed_score") or 0)))
            return {
                "cmd_sent": str(cmd),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 284,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world_from_vision), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None) as mock_sleep, \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
        self.assertEqual(send_cmds, [("d", 100), ("d", 100), ("d", 100)])
        self.assertGreaterEqual(mock_sleep.call_count, 3)
        self.assertTrue(
            any(
                "Post-success follow-through pulse 1/3: D 100%." in line
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                "Post-success follow-through pulse 3/3: D 100%." in line
                for line in print_lines
            )
        )


if __name__ == "__main__":
    unittest.main()
