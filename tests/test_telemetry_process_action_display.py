import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class _DummyRobot:
    def __init__(self, *, supports_timed_command_queue=False):
        self.sent = []
        self._last_turn_cmd = None
        self.supports_timed_command_queue = bool(supports_timed_command_queue)
        self.stop_calls = 0

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))

    def stop(self):
        self.stop_calls += 1


class _WireAwareDummyRobot(_DummyRobot):
    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))
        return {
            "cmd_sent": str(cmd),
            "pwm": int(pwm),
            "duration_ms": int(duration_ms),
            "wire_text": f"raw:{str(cmd)}:{int(pwm)}:{int(duration_ms)}",
        }


class _CustomActionDummyRobot(_DummyRobot):
    def send_custom_actions_pwm(self, cmd, actions, duration_ms=0):
        actions_out = [dict(action) for action in actions]
        self.sent.append((cmd, actions_out, duration_ms))
        return {
            "cmd_sent": str(cmd),
            "pwm": max(int(action.get("pwm") or 0) for action in actions_out),
            "power": 1.0,
            "duration_ms": int(duration_ms),
            "wire_text": f"custom:{str(cmd)}:{int(duration_ms)}",
            "actions": actions_out,
        }


class _DummyWorld:
    def __init__(self):
        self.brick = {
            "visible": True,
            "dist": 200.0,
            "angle": 0.0,
            "x_axis": 0.0,
            "offset_x": 0.0,
            "confidence": 90.0,
        }
        self.process_rules = {}

    def update_from_motion(self, _evt):
        return None


class _AlignLogWorld:
    def __init__(self):
        self.process_rules = {
            "ALIGN_BRICK": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "x_axis": {"target": -4.74, "tol": 1.40},
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 48.0,
            "angle": 0.0,
            "x_axis": 2.89,
            "offset_x": 2.89,
            "y_axis": 0.0,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class _AlignHoldLogWorld:
    def __init__(self):
        self.process_rules = {
            "ALIGN_BRICK": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.74, "tol": 1.40},
                    "yAxis_offset_abs": {"target": 2.50, "tol": 1.50},
                    "dist": {"target": 105.63, "tol": 1.50},
                },
                "y_axis_hold_offset_exception": {
                    "enabled": True,
                    "operator_log_reminder": "Exception active: keep y-axis hold at +4mm above final target until x/dist are settled, then finish the descent.",
                    "hold_offset_mm": 4.0,
                    "release_when_metrics_within_tol": ["x_axis", "dist"],
                    "release_confirm_frames": 2,
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 117.5,
            "angle": 0.0,
            "x_axis": -5.2,
            "offset_x": -5.2,
            "y_axis": 2.72,
            "offset_y": 2.72,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class _AlignDistLogWorld:
    def __init__(self):
        self.process_rules = {
            "ALIGN_BRICK": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.74, "tol": 1.40},
                    "yAxis_offset_abs": {"target": 2.50, "tol": 1.50},
                    "dist": {"target": 105.63, "tol": 1.50},
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 148.60,
            "angle": 0.0,
            "x_axis": -4.8,
            "offset_x": -4.8,
            "y_axis": 2.5,
            "offset_y": 2.5,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestTelemetryProcessActionDisplay(unittest.TestCase):
    def test_send_robot_command_records_display_from_logical_command(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="l",
            speed=0.0,
            speed_score=telemetry_robot.SPEED_SCORE_MIN,
            auto_mode=True,
        )
        self.assertTrue(robot.sent)
        self.assertEqual(getattr(world, "_last_action_cmd", None), "l")
        cmd_display = getattr(world, "_last_action_cmd", None) or "l"
        expected_score = telemetry_robot.quantize_speed("l", speed=getattr(world, "_last_action_speed", 0.0))[1]
        expected = telemetry_process.action_display_text(cmd_display, expected_score)
        self.assertEqual(getattr(world, "_last_action_display", None), expected)

    def test_format_control_action_line_uses_logical_direction_even_with_remap(self):
        orig_remap = getattr(telemetry_process.telemetry_robot_module, "COMMAND_REMAP", None)
        try:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = {"f": "b"}
            line = telemetry_process.format_control_action_line("f", 0.3, "align")
        finally:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = orig_remap
        self.assertIn("move forward", line)

    def test_send_robot_command_ignores_stale_runtime_remap(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        orig_remap = getattr(telemetry_process.telemetry_robot_module, "COMMAND_REMAP", None)
        try:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = {"f": "b"}
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="ALIGN_BRICK",
                cmd="f",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=False,
            )
        finally:
            telemetry_process.telemetry_robot_module.COMMAND_REMAP = orig_remap

        self.assertTrue(robot.sent)
        self.assertEqual(robot.sent[0][0], "f")
        self.assertEqual(meta.get("cmd_sent"), "f")

    def test_auto_chassis_speed_cap_applies_even_when_step_exempts_cap(self):
        world = _DummyWorld()
        robot = _WireAwareDummyRobot()

        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=100,
            auto_mode=True,
            score_cap_exempt=True,
        )

        _power, expected_pwm, expected_score, _duration_ms = telemetry_robot.speed_power_pwm_for_cmd(
            "f",
            telemetry_process.AUTO_SPEED_SCORE_HARD_MAX,
        )
        self.assertEqual(meta.get("score_model"), expected_score)
        self.assertEqual(meta.get("pwm"), expected_pwm)
        self.assertLess(int(meta.get("pwm")), 255)

    def test_send_robot_command_uses_returned_wire_text_for_single_send(self):
        world = _DummyWorld()
        robot = _WireAwareDummyRobot()

        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="MANUAL",
            cmd="f",
            speed=0.0,
            speed_score=telemetry_robot.SPEED_SCORE_MIN,
            auto_mode=False,
        )

        self.assertIsInstance(meta, dict)
        self.assertEqual(getattr(world, "_last_action_wire", None), meta.get("wire_text"))
        self.assertTrue(str(meta.get("wire_text") or "").startswith("raw:f:"))

    def test_send_robot_command_supports_custom_turn_drive_actions(self):
        world = _DummyWorld()
        robot = _CustomActionDummyRobot()

        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="POSITION_BRICK",
            cmd="r",
            speed=0.0,
            speed_score=1,
            auto_mode=True,
            duration_override_ms=180,
            custom_action_specs=[
                {"target": "l", "action": "b", "pwm": 133},
                {"target": "r", "action": "f", "pwm": 0},
            ],
            action_note="TURN+FWD",
        )

        self.assertEqual(len(robot.sent), 1)
        self.assertEqual(robot.sent[0][0], "r")
        self.assertEqual(robot.sent[0][2], 180)
        self.assertEqual(meta.get("cmd_sent"), "r")
        self.assertEqual(meta.get("wire_text"), "custom:r:180")
        self.assertEqual(meta.get("action_note"), "TURN+FWD")
        self.assertIn("TURN+FWD", getattr(world, "_last_action_sent_display", ""))

    def test_timed_action_with_confirm_callback_does_not_force_immediate_stop(self):
        world = _AlignDistLogWorld()
        robot = _DummyRobot()

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = 0.77
            _world.brick["offset_x"] = 0.77
            _world.brick["dist"] = 250.15
            _world.brick["y_axis"] = 2.99
            _world.brick["offset_y"] = 2.99

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "r",
                "speed": 0.438,
                "score": 1,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            return {
                "cmd_sent": str(cmd),
                "score_effective": 1,
                "pwm": 132,
                "power": 0.438,
                "duration_ms": 255,
                "action_note": "TURN+FWD",
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", side_effect=RuntimeError("sentinel_after_send")), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "sentinel_after_send"):
                telemetry_process.run_alignment_segment(
                    segment={"events": []},
                    step="ALIGN_BRICK",
                    robot=robot,
                    vision=object(),
                    world=world,
                    steps=[],
                    raw_steps=[],
                    observer=None,
                    analysis_pause_s=0.0,
                    confirm_callback=lambda *_args, **_kwargs: True,
                    align_silent=True,
                )

        self.assertEqual(robot.stop_calls, 0)

    def test_send_robot_command_auto_mode_caps_score_to_25(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=80,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertLessEqual(int(meta.get("score_model") or 0), 25)
        self.assertLessEqual(int(meta.get("score_effective") or 0), 25)

    def test_send_robot_command_auto_mode_caps_find_steps_to_25(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="FIND_WALL",
            cmd="r",
            speed=0.0,
            speed_score=80,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertLessEqual(int(meta.get("score_model") or 0), 25)
        self.assertLessEqual(int(meta.get("score_effective") or 0), 25)

    def test_auto_drive_commands_do_not_use_ease_segments_without_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))
        detail = telemetry_process.auto_action_detail_text("f", 50, action_meta=meta)
        self.assertNotIn("EASE(", detail)

    def test_auto_drive_commands_use_ease_segments_with_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot(supports_timed_command_queue=True)
        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertIsInstance(meta.get("segments"), list)
        self.assertGreater(len(meta.get("segments") or []), 1)
        self.assertGreater(len(robot.sent), 1)
        self.assertIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))
        detail = telemetry_process.auto_action_detail_text("f", 50, action_meta=meta)
        self.assertIn("EASE(", detail)

    def test_ease_segments_preserve_returned_wire_text(self):
        world = _DummyWorld()
        robot = _WireAwareDummyRobot(supports_timed_command_queue=True)

        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=50,
            auto_mode=True,
        )

        self.assertIsInstance(meta, dict)
        self.assertIsInstance(meta.get("segments"), list)
        self.assertTrue(meta.get("wire_text"))
        self.assertTrue(all(str(seg.get("wire_text") or "").startswith("raw:f:") for seg in meta.get("segments") or []))
        self.assertIn("raw:f:", str(getattr(world, "_last_action_wire", "")))

    def test_manual_drive_commands_use_ease_segments_without_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        with patch("telemetry_process.time.sleep", return_value=None) as sleep_mock:
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="f",
                speed=0.0,
                speed_score=50,
                auto_mode=False,
            )
        self.assertIsInstance(meta, dict)
        self.assertIsInstance(meta.get("segments"), list)
        self.assertGreater(len(meta.get("segments") or []), 1)
        self.assertGreater(len(robot.sent), 1)
        self.assertGreaterEqual(sleep_mock.call_count, 1)
        self.assertIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))

    def test_manual_turn_commands_use_ease_segments_without_queue_support(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        with patch("telemetry_process.time.sleep", return_value=None) as sleep_mock:
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="l",
                speed=0.0,
                speed_score=50,
                auto_mode=False,
            )
        self.assertIsInstance(meta, dict)
        self.assertIsInstance(meta.get("segments"), list)
        self.assertGreater(len(meta.get("segments") or []), 1)
        self.assertGreater(len(robot.sent), 1)
        self.assertGreaterEqual(sleep_mock.call_count, 1)
        self.assertIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))

    def test_manual_commands_below_ease_threshold_do_not_use_segments(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        with patch("telemetry_process.time.sleep", return_value=None):
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="f",
                speed=0.0,
                speed_score=9,
                auto_mode=False,
            )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))

    def test_manual_turn_commands_below_turn_ease_threshold_do_not_use_segments(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        with patch("telemetry_process.time.sleep", return_value=None):
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="r",
                speed=0.0,
                speed_score=19,
                auto_mode=False,
            )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))

    def test_manual_commands_can_explicitly_disable_ease_segments(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        with patch("telemetry_process.time.sleep", return_value=None):
            meta = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="f",
                speed=0.0,
                speed_score=50,
                auto_mode=False,
                ease_in_out_enabled=False,
            )
        self.assertIsInstance(meta, dict)
        self.assertFalse(isinstance(meta.get("segments"), list))
        self.assertEqual(len(robot.sent), 1)
        self.assertNotIn("EASE(", str(getattr(world, "_last_action_sent_display", "")))

    def test_align_observe_act_log_uses_sentence_style_and_curve_note(self):
        world = _AlignLogWorld()
        robot = _DummyRobot()
        print_lines = []
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0)) + 1

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if gate_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "speed": 0.434,
                "score": 20,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
                "duration_override_ms": 471,
                "curve_name": "x-axis curve",
                "curve_value_mm": 4.63,
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            _ = kwargs
            return {
                "cmd_sent": str(cmd),
                "score_effective": 20,
                "pwm": 131,
                "power": 0.434,
                "duration_ms": 471,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch(
                 "builtins.print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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
        observe_line = next(
            line for line in print_lines if "[T1.1 ALIGN]" in line and "I'm xAxis_offset=" in line
        )
        step_number = telemetry_process._step_number_for_label("ALIGN_BRICK")
        self.assertIn(f"[step#{int(step_number)}]", observe_line)
        self.assertIn(
            f"I'm xAxis_offset=2.89 Δ{telemetry_process.COLOR_YELLOW}+7.63{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("to the right of our target of -4.74+/-1.40.", observe_line)
        self.assertIn(
            f"{telemetry_process.COLOR_ORANGE_BRIGHT}> L 20%{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("pwm=131, pwr=0.434, t=471ms; used our x-axis curve at ", observe_line)
        self.assertIn(
            f"{telemetry_process.COLOR_ORANGE_DARK}4.63{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertNotIn("mm", observe_line)

    def test_align_observe_act_log_shows_forward_turn_drive_assist_and_passes_custom_actions(self):
        world = _AlignDistLogWorld()
        world.process_rules["ALIGN_BRICK"]["align_policy"] = {
            "x_axis_turn_drive_assist": {
                "enabled": True,
                "require_dist_outside_gate": True,
                "min_dist_outside_mm": 0.0,
                "forward_profile": "forward_pivot",
            }
        }
        world.process_rules["ALIGN_BRICK"]["success_gates"]["xAxis_offset_abs"] = {"target": 5.33, "tol": 1.40}
        world.process_rules["ALIGN_BRICK"]["success_gates"]["yAxis_offset_abs"] = {"target": 3.03, "tol": 2.30}
        world.process_rules["ALIGN_BRICK"]["success_gates"]["dist"] = {"target": 157.93, "tol": 2.30}
        world.brick["x_axis"] = 37.46
        world.brick["offset_x"] = 37.46
        world.brick["dist"] = 230.71
        world.brick["y_axis"] = 2.94
        world.brick["offset_y"] = 2.94
        robot = _DummyRobot()
        print_lines = []
        gate_calls = {"n": 0}
        sent_calls = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = 37.46
            _world.brick["offset_x"] = 37.46
            _world.brick["dist"] = 230.71
            _world.brick["y_axis"] = 2.94
            _world.brick["offset_y"] = 2.94

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if gate_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "speed": 0.35,
                "score": 7,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
                "curve_name": "x_axis monotonic curve",
                "curve_value_mm": 32.13,
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            sent_calls.append({"cmd": cmd, "kwargs": dict(kwargs)})
            return {
                "cmd_sent": str(cmd),
                "score_effective": 7,
                "pwm": 140,
                "power": 0.35,
                "duration_ms": 275,
                "action_note": kwargs.get("action_note"),
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch(
                 "builtins.print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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
        self.assertEqual(len(sent_calls), 1)
        sent_kwargs = sent_calls[0]["kwargs"]
        self.assertIn("TURN+FWD", sent_kwargs.get("action_note"))
        self.assertIn("Matching curve:", sent_kwargs.get("action_note"))
        self.assertIn("Turning L_motor", sent_kwargs.get("action_note"))
        custom_action_specs = list(sent_kwargs.get("custom_action_specs") or [])
        self.assertEqual(len(custom_action_specs), 2)
        self.assertEqual(custom_action_specs[0].get("target"), "l")
        self.assertEqual(custom_action_specs[0].get("action"), "b")
        self.assertEqual(custom_action_specs[1].get("target"), "r")
        self.assertEqual(custom_action_specs[1].get("action"), "f")
        self.assertGreaterEqual(int(custom_action_specs[0].get("pwm") or 0), 0)
        self.assertGreater(int(custom_action_specs[1].get("pwm") or 0), 0)
        self.assertLessEqual(
            int(custom_action_specs[0].get("pwm") or 0),
            int(custom_action_specs[1].get("pwm") or 0),
        )
        observe_line = next(
            line for line in print_lines if "[T1.1 ALIGN]" in line and "I'm xAxis_offset=37.46" in line
        )
        self.assertIn(
            f"{telemetry_process.COLOR_ORANGE_BRIGHT}> L 7%{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("TURN+FWD", observe_line)

    def test_align_distance_action_uses_forward_right_curve_when_x_gap_is_left(self):
        world = _AlignDistLogWorld()
        world.process_rules["ALIGN_BRICK"]["align_policy"] = {
            "forward_while_turning_assist": {
                "enabled": True,
                "prefer_curve_for_x_axis": True,
                "curve_score_mode": "requested",
            }
        }
        world.process_rules["ALIGN_BRICK"]["success_gates"]["xAxis_offset_abs"] = {"target": 5.33, "tol": 1.40}
        world.process_rules["ALIGN_BRICK"]["success_gates"]["yAxis_offset_abs"] = {"target": 3.03, "tol": 2.30}
        world.process_rules["ALIGN_BRICK"]["success_gates"]["dist"] = {"target": 157.93, "tol": 2.30}
        world.brick["x_axis"] = -49.69
        world.brick["offset_x"] = -49.69
        world.brick["dist"] = 308.80
        world.brick["y_axis"] = 2.94
        world.brick["offset_y"] = 2.94
        robot = _DummyRobot()
        print_lines = []
        gate_calls = {"n": 0}
        sent_calls = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = -49.69
            _world.brick["offset_x"] = -49.69
            _world.brick["dist"] = 308.80
            _world.brick["y_axis"] = 2.94
            _world.brick["offset_y"] = 2.94

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if gate_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "f",
                "speed": 1.0,
                "score": 100,
                "reason": "distance_alignment",
                "correction_type": "distance",
                "curve_name": "dist error bands",
                "curve_value_mm": 150.0,
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            sent_calls.append({"cmd": cmd, "kwargs": dict(kwargs)})
            actions = list(kwargs.get("custom_action_specs") or [])
            return {
                "cmd_sent": str(cmd),
                "score_effective": 100,
                "pwm": max(int(action.get("pwm") or 0) for action in actions),
                "power": 1.0,
                "duration_ms": kwargs.get("duration_override_ms"),
                "action_note": kwargs.get("action_note"),
                "actions": actions,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch(
                 "builtins.print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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
        self.assertEqual(len(sent_calls), 1)
        self.assertEqual(sent_calls[0]["cmd"], "f")
        sent_kwargs = sent_calls[0]["kwargs"]
        self.assertIn("FWD+RIGHT-CURVE", sent_kwargs.get("action_note"))
        self.assertIn("Matching curve: forwardRight_backLeft", sent_kwargs.get("action_note"))
        custom_action_specs = list(sent_kwargs.get("custom_action_specs") or [])
        self.assertEqual(
            custom_action_specs,
            [
                {"target": "l", "action": "b", "pwm": 199},
                {"target": "r", "action": "f", "pwm": 0},
            ],
        )
        observe_line = next(
            line for line in print_lines if "[T1.1 ALIGN]" in line and "> F 100%" in line
        )
        self.assertIn("FWD+RIGHT-CURVE", observe_line)

    def test_align_y_hold_log_uses_hold_target_for_y_axis_action(self):
        world = _AlignHoldLogWorld()
        robot = _DummyRobot()
        print_lines = []
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = -5.2
            _world.brick["offset_x"] = -5.2
            _world.brick["dist"] = 117.5
            _world.brick["y_axis"] = 2.72
            _world.brick["offset_y"] = 2.72

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if gate_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "u",
                "speed": 0.018,
                "score": 1,
                "reason": "y_axis_alignment",
                "correction_type": "y_axis",
                "curve_name": "y_axis monotonic curve (error=3.78mm, score=1%)",
                "curve_value_mm": 3.78,
                "y_err_mm": -3.78,
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            return {
                "cmd_sent": str(cmd),
                "score_effective": 1,
                "pwm": 40,
                "power": 0.018,
                "duration_ms": 300,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch(
                 "builtins.print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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
        observe_line = next(
            line for line in print_lines if "[T1.1 ALIGN]" in line and "I see y_err=" in line
        )
        self.assertIn("below our hold target=+6.50", observe_line)
        self.assertIn("(success target=+2.50", observe_line)
        self.assertIn(
            f"{telemetry_process.COLOR_ORANGE_BRIGHT}> U 1%{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("shared 1% floor", observe_line)
        self.assertNotIn("above our target=+2.50", observe_line)

    def test_align_distance_log_uses_absolute_and_delta_values(self):
        world = _AlignDistLogWorld()
        robot = _DummyRobot()
        print_lines = []
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = -4.8
            _world.brick["offset_x"] = -4.8
            _world.brick["dist"] = 148.60
            _world.brick["y_axis"] = 2.5
            _world.brick["offset_y"] = 2.5

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if gate_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "b",
                "speed": 0.03,
                "score": 1,
                "reason": "distance_alignment",
                "correction_type": "distance",
                "curve_name": "distance monotonic curve (error=42.97mm, score=1%)",
                "curve_value_mm": 42.97,
            }

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            return {
                "cmd_sent": str(cmd),
                "score_effective": 1,
                "pwm": 40,
                "power": 0.03,
                "duration_ms": 300,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch(
                 "builtins.print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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
        observe_line = next(
            line for line in print_lines if "[T1.1 ALIGN]" in line and "I see dist=" in line
        )
        self.assertIn("I see dist=148.60", observe_line)
        self.assertIn(
            f"Δ{telemetry_process.COLOR_YELLOW}+42.97{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("away from our distance target target=105.63 ±1.50.", observe_line)
        self.assertIn(
            f"{telemetry_process.COLOR_ORANGE_BRIGHT}> B 1%{telemetry_process.COLOR_RESET}",
            observe_line,
        )
        self.assertIn("shared 1% floor", observe_line)

    def test_repeated_identical_act_guard_blocks_51st_send(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        for _ in range(int(telemetry_process.MAX_CONSECUTIVE_IDENTICAL_ACTS)):
            telemetry_process.send_robot_command(
                robot,
                world,
                step="ALIGN_BRICK",
                cmd="f",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=True,
            )
        self.assertEqual(len(robot.sent), int(telemetry_process.MAX_CONSECUTIVE_IDENTICAL_ACTS))
        with self.assertRaisesRegex(RuntimeError, r"\[SAFETY-FAIL\].*repeat 51 times"):
            telemetry_process.send_robot_command(
                robot,
                world,
                step="ALIGN_BRICK",
                cmd="f",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=True,
            )
        self.assertEqual(len(robot.sent), int(telemetry_process.MAX_CONSECUTIVE_IDENTICAL_ACTS))
        self.assertGreaterEqual(robot.stop_calls, 1)

        with self.assertRaisesRegex(RuntimeError, r"\[SAFETY-FAIL\]"):
            telemetry_process.send_robot_command(
                robot,
                world,
                step="ALIGN_BRICK",
                cmd="l",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=True,
            )
        self.assertEqual(len(robot.sent), int(telemetry_process.MAX_CONSECUTIVE_IDENTICAL_ACTS))
        self.assertGreaterEqual(robot.stop_calls, 2)


if __name__ == "__main__":
    unittest.main()
