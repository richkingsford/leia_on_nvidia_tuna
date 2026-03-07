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
                {"type": "action", "command": "left", "speedScore": 10},
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


if __name__ == "__main__":
    unittest.main()
