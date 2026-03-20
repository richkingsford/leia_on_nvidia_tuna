import builtins
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "SEAT_BRICK": {
                "nominalDemosOnly": True,
            }
        }
        self.suppress_brick_state_log = False
        self._gatecheck_status = {"stale": True}
        self._last_gate_summary = {"stale": True}

    def update_from_motion(self, evt):
        _ = evt


class TestTelemetryProcessNominalReplay(unittest.TestCase):
    def test_nominal_replay_skips_gatecheck_and_keeps_inter_action_pause(self):
        segment = {
            "events": [
                {"type": "action", "command": "forward", "speedScore": 1},
                {"type": "action", "command": "backward", "speedScore": 1},
            ]
        }
        world = _DummyWorld()
        robot = _DummyRobot()
        pauses = []
        calls = []

        orig_send = telemetry_process.send_robot_command
        orig_post = telemetry_process.post_act_analysis
        orig_wait = telemetry_process.wait_for_frame_settle
        orig_sleep = telemetry_process.time.sleep
        orig_gatecheck_after = telemetry_process.run_full_gatecheck_after_act
        try:
            telemetry_process.send_robot_command = lambda *args, **kwargs: calls.append("send")
            telemetry_process.post_act_analysis = lambda *args, **kwargs: calls.append("post")
            telemetry_process.wait_for_frame_settle = (
                lambda _world, _vision, frames, log=False: pauses.append(int(frames))
            )
            telemetry_process.time.sleep = lambda *_args, **_kwargs: None

            def _unexpected_gatecheck(*args, **kwargs):
                raise AssertionError("nominal replay should not run gatecheck")

            telemetry_process.run_full_gatecheck_after_act = _unexpected_gatecheck

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="SEAT_BRICK",
                robot=robot,
                vision=None,
                world=world,
            )
        finally:
            telemetry_process.send_robot_command = orig_send
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.wait_for_frame_settle = orig_wait
            telemetry_process.time.sleep = orig_sleep
            telemetry_process.run_full_gatecheck_after_act = orig_gatecheck_after

        self.assertTrue(ok)
        self.assertEqual(reason, "nominal demo replay")
        self.assertIsNone(world._gatecheck_status)
        self.assertIsNone(world._last_gate_summary)
        self.assertEqual(calls.count("send"), 2)
        self.assertEqual(calls.count("post"), 2)
        self.assertEqual(pauses, [telemetry_process.DEMO_ACTION_PAUSE_FRAMES])

    def test_replay_startup_action_runs_before_start_ground_reset_when_deferred(self):
        segment = {
            "events": [
                {"type": "action", "command": "left_turn", "speedScore": 10},
            ]
        }
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "controller": "replay",
                "startup_action_exception": {
                    "enabled": True,
                    "command": "l",
                    "score": 100,
                    "acts": 3,
                },
                "start_ground_reset_exception": {
                    "enabled": True,
                    "run_after_startup_action": True,
                },
            }
        }
        world.brick = {"visible": True}
        robot = _DummyRobot()
        order = []

        def _fake_wait_for_start_gates(*_args, **_kwargs):
            order.append("wait_start_gates")
            return "success"

        def _fake_send_robot_command(*args, **_kwargs):
            order.append(f"startup:{str(args[3])}")
            return {}

        def _fake_ground_reset(*_args, **_kwargs):
            order.append("ground_reset")
            return {"enabled": True, "success": True, "reason": "ground reset pass"}

        def _fake_ground_up(*_args, **_kwargs):
            order.append("ground_up")
            return {"enabled": True, "handled": True, "success": True, "reason": "ok"}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_send = telemetry_process.send_robot_command
        orig_ground_reset = telemetry_process._run_start_ground_reset_exception
        orig_ground_up = telemetry_process._run_ground_up_level2_exception
        orig_post = telemetry_process.post_act_analysis
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = _fake_wait_for_start_gates
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._run_start_ground_reset_exception = _fake_ground_reset
            telemetry_process._run_ground_up_level2_exception = _fake_ground_up
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="FIND_WALL2",
                robot=robot,
                vision=object(),
                world=world,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.send_robot_command = orig_send
            telemetry_process._run_start_ground_reset_exception = orig_ground_reset
            telemetry_process._run_ground_up_level2_exception = orig_ground_up
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        startup_calls = [entry for entry in order if entry.startswith("startup:")]
        self.assertEqual(len(startup_calls), 3)
        self.assertIn("ground_reset", order)
        self.assertLess(order.index(startup_calls[-1]), order.index("ground_reset"))

    def test_replay_startup_pre_action_runs_alone_with_duration_override(self):
        segment = {
            "events": [
                {"type": "action", "command": "left_turn", "speedScore": 10},
            ]
        }
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "controller": "replay",
                "startup_pre_action_exception": {
                    "enabled": True,
                    "operator_log_reminder": "Exception active: before startup hard-left scan, slow left turn L 1% for 1500ms.",
                    "command": "l",
                    "score": 1,
                    "acts": 1,
                    "duration_override_ms": 1500,
                },
                "startup_action_exception": {
                    "enabled": False,
                    "command": "l",
                    "score": 100,
                    "acts": 7,
                },
            }
        }
        world.brick = {"visible": False}
        robot = _DummyRobot()
        send_calls = []
        print_lines = []

        def _fake_wait_for_start_gates(*_args, **_kwargs):
            return "start"

        def _fake_send_robot_command(*args, **kwargs):
            send_calls.append(
                {
                    "cmd": str(args[3]),
                    "score": int(kwargs.get("speed_score") or 0),
                    "duration_override_ms": kwargs.get("duration_override_ms"),
                }
            )
            return {
                "power": 0.03,
                "score_effective": kwargs.get("speed_score"),
                "duration_ms": kwargs.get("duration_override_ms"),
            }

        def _fake_ground_up(*_args, **_kwargs):
            return {"enabled": True, "handled": True, "success": True, "reason": "ok"}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_send = telemetry_process.send_robot_command
        orig_ground_up = telemetry_process._run_ground_up_level2_exception
        orig_sleep = telemetry_process.time.sleep
        orig_print = builtins.print
        try:
            telemetry_process.wait_for_start_gates = _fake_wait_for_start_gates
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._run_ground_up_level2_exception = _fake_ground_up
            telemetry_process.time.sleep = lambda *_a, **_k: None
            builtins.print = lambda *args, **kwargs: print_lines.append(
                " ".join(str(arg) for arg in args)
            )

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="FIND_WALL2",
                robot=robot,
                vision=object(),
                world=world,
                align_silent=False,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.send_robot_command = orig_send
            telemetry_process._run_ground_up_level2_exception = orig_ground_up
            telemetry_process.time.sleep = orig_sleep
            builtins.print = orig_print

        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(len(send_calls), 1)
        self.assertEqual(send_calls[0]["cmd"], "l")
        self.assertEqual(send_calls[0]["score"], 1)
        self.assertEqual(send_calls[0]["duration_override_ms"], 1500)
        self.assertTrue(any("L 1% for 1500ms." in line for line in print_lines))
        self.assertFalse(any("Executing startup pre-action L 1% x1." in line for line in print_lines))

    def test_replay_startup_pre_action_waits_before_ground_reset(self):
        segment = {
            "events": [
                {"type": "action", "command": "left_turn", "speedScore": 10},
            ]
        }
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "controller": "replay",
                "startup_pre_action_exception": {
                    "enabled": True,
                    "command": "l",
                    "score": 1,
                    "acts": 1,
                    "duration_override_ms": 1500,
                },
                "start_ground_reset_exception": {
                    "enabled": True,
                    "observe_between_acts": False,
                    "run_after_startup_action": True,
                },
            }
        }
        world.brick = {"visible": False}
        robot = _DummyRobot()
        order = []

        def _fake_wait_for_start_gates(*_args, **_kwargs):
            order.append("wait_start_gates")
            return "start"

        def _fake_send_robot_command(*args, **kwargs):
            order.append(f"send:{str(args[3])}")
            return {
                "power": 0.03,
                "score_effective": kwargs.get("speed_score"),
                "duration_ms": kwargs.get("duration_override_ms") or 300,
            }

        def _fake_ground_reset(*_args, **_kwargs):
            order.append("ground_reset")
            return {"enabled": True, "success": True, "reason": "ground reset pass"}

        def _fake_ground_up(*_args, **_kwargs):
            order.append("ground_up")
            return {"enabled": True, "handled": True, "success": True, "reason": "ok"}

        def _fake_sleep(seconds):
            order.append(f"sleep:{float(seconds):.3f}")

        orig_wait = telemetry_process.wait_for_start_gates
        orig_send = telemetry_process.send_robot_command
        orig_ground_reset = telemetry_process._run_start_ground_reset_exception
        orig_ground_up = telemetry_process._run_ground_up_level2_exception
        orig_pause_after_exception = telemetry_process.pause_after_exception
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = _fake_wait_for_start_gates
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._run_start_ground_reset_exception = _fake_ground_reset
            telemetry_process._run_ground_up_level2_exception = _fake_ground_up
            telemetry_process.pause_after_exception = lambda *_a, **_k: None
            telemetry_process.time.sleep = _fake_sleep

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="FIND_WALL2",
                robot=robot,
                vision=object(),
                world=world,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.send_robot_command = orig_send
            telemetry_process._run_start_ground_reset_exception = orig_ground_reset
            telemetry_process._run_ground_up_level2_exception = orig_ground_up
            telemetry_process.pause_after_exception = orig_pause_after_exception
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertIn("send:l", order)
        self.assertIn("sleep:1.500", order)
        self.assertIn("ground_reset", order)
        self.assertLess(order.index("send:l"), order.index("sleep:1.500"))
        self.assertLess(order.index("sleep:1.500"), order.index("ground_reset"))

    def test_replay_visible_false_search_uses_configured_cycle(self):
        segment = {
            "events": [
                {"type": "action", "command": "left_turn", "speedScore": 10},
                {"type": "action", "command": "left_turn", "speedScore": 10},
            ]
        }
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "controller": "replay",
                "prefer_demo_speed": True,
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 20,
                    "low_score": 10,
                    "command_scores": {"l": 20, "b": 10},
                    "start_with_high": True,
                    "commands": ["l", "b"],
                },
                "startup_pre_action_exception": {
                    "enabled": True,
                    "command": "l",
                    "score": 1,
                    "acts": 1,
                    "duration_override_ms": 1500,
                },
                "startup_action_exception": {
                    "enabled": False,
                    "command": "l",
                    "score": 100,
                    "acts": 7,
                },
                "success_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {"visible": False}
        robot = _DummyRobot()
        send_cmds = []
        send_scores = []

        def _fake_wait_for_start_gates(*_args, **_kwargs):
            return "start"

        def _fake_send_robot_command(*args, **kwargs):
            cmd = str(args[3])
            send_cmds.append(cmd)
            send_scores.append(int(kwargs.get("speed_score") or 0))
            if len(send_cmds) >= 3:
                world.brick["visible"] = True
            return {
                "power": 0.03,
                "score_effective": kwargs.get("speed_score"),
                "duration_ms": kwargs.get("duration_override_ms") or 300,
            }

        orig_wait = telemetry_process.wait_for_start_gates
        orig_send = telemetry_process.send_robot_command
        orig_sleep = telemetry_process.time.sleep
        orig_post = telemetry_process.post_act_analysis
        orig_settle = telemetry_process.post_act_settle_pause
        orig_gatecheck = telemetry_process.run_full_gatecheck_after_act
        orig_pre_action_obs = telemetry_process.pre_action_success_observation
        orig_ground_up = telemetry_process._run_ground_up_level2_exception
        try:
            telemetry_process.wait_for_start_gates = _fake_wait_for_start_gates
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.time.sleep = lambda *_a, **_k: None
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
            telemetry_process.post_act_settle_pause = lambda *_a, **_k: None
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: bool(world.brick.get("visible"))
            telemetry_process.pre_action_success_observation = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process._run_ground_up_level2_exception = (
                lambda *_a, **_k: {"enabled": False, "handled": False}
            )

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="FIND_WALL2",
                robot=robot,
                vision=object(),
                world=world,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.send_robot_command = orig_send
            telemetry_process.time.sleep = orig_sleep
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.post_act_settle_pause = orig_settle
            telemetry_process.run_full_gatecheck_after_act = orig_gatecheck
            telemetry_process.pre_action_success_observation = orig_pre_action_obs
            telemetry_process._run_ground_up_level2_exception = orig_ground_up

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(send_cmds), 3)
        self.assertEqual(send_cmds[:3], ["l", "l", "b"])
        self.assertEqual(send_scores[:3], [1, 20, 10])

    def test_replay_visible_false_search_keeps_demo_mast_down_phase(self):
        segment = {
            "events": [
                {"type": "action", "command": "left_turn", "speedScore": 10},
                {"type": "action", "command": "mast_down", "speedScore": 100},
            ]
        }
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "controller": "replay",
                "prefer_demo_speed": True,
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "commands": ["r", "b"],
                },
                "height_intelligence": {
                    "enabled": False,
                },
                "success_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {"visible": False}
        robot = _DummyRobot()
        send_cmds = []
        send_scores = []

        def _fake_wait_for_start_gates(*_args, **_kwargs):
            return "start"

        def _fake_send_robot_command(*args, **kwargs):
            cmd = str(args[3])
            score = int(kwargs.get("speed_score") or 0)
            send_cmds.append(cmd)
            send_scores.append(score)
            if len(send_cmds) >= 2:
                world.brick["visible"] = True
            return {
                "power": 0.03,
                "score_effective": score,
                "duration_ms": 300,
            }

        orig_wait = telemetry_process.wait_for_start_gates
        orig_send = telemetry_process.send_robot_command
        orig_sleep = telemetry_process.time.sleep
        orig_post = telemetry_process.post_act_analysis
        orig_settle = telemetry_process.post_act_settle_pause
        orig_gatecheck = telemetry_process.run_full_gatecheck_after_act
        orig_pre_action_obs = telemetry_process.pre_action_success_observation
        orig_ground_up = telemetry_process._run_ground_up_level2_exception
        try:
            telemetry_process.wait_for_start_gates = _fake_wait_for_start_gates
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.time.sleep = lambda *_a, **_k: None
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
            telemetry_process.post_act_settle_pause = lambda *_a, **_k: None
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: bool(world.brick.get("visible"))
            telemetry_process.pre_action_success_observation = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process._run_ground_up_level2_exception = (
                lambda *_a, **_k: {"enabled": False, "handled": False}
            )

            ok, reason = telemetry_process.replay_segment(
                segment=segment,
                step="FIND_WALL2",
                robot=robot,
                vision=object(),
                world=world,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.send_robot_command = orig_send
            telemetry_process.time.sleep = orig_sleep
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.post_act_settle_pause = orig_settle
            telemetry_process.run_full_gatecheck_after_act = orig_gatecheck
            telemetry_process.pre_action_success_observation = orig_pre_action_obs
            telemetry_process._run_ground_up_level2_exception = orig_ground_up

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(send_cmds), 2)
        self.assertEqual(send_cmds[:2], ["r", "d"])
        self.assertEqual(send_scores[:2], [25, 100])


if __name__ == "__main__":
    unittest.main()
