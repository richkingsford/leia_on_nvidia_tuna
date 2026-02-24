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


if __name__ == "__main__":
    unittest.main()
