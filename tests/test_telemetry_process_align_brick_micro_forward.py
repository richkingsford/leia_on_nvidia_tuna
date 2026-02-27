import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyRobot:
    def __init__(self, *, supports_timed_command_queue=False):
        self.sent = []
        self._last_turn_cmd = None
        self.supports_timed_command_queue = bool(supports_timed_command_queue)

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))


class _DummyWorld:
    pass


class TestTelemetryProcessAlignBrickMicroForward(unittest.TestCase):
    def test_auto_align_brick_forward_reaches_target_score_segment(self):
        world = _DummyWorld()
        robot = _DummyRobot(supports_timed_command_queue=True)

        process = json.loads((Path(__file__).resolve().parents[1] / "world_model_process.json").read_text())
        world.process_rules = process.get("steps", {})
        world.learned_rules = {}
        world.brick = {"dist": 130.0}

        meta = telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="f",
            speed=0.0,
            speed_score=12,
            auto_mode=True,
        )

        self.assertTrue(robot.sent)
        self.assertIsInstance(meta, dict)
        segments = meta.get("segments")
        self.assertIsInstance(segments, list)
        self.assertTrue(segments)
        scores = [seg.get("score_model") for seg in segments if isinstance(seg, dict)]
        self.assertIn(12, scores)
        self.assertLess(min(scores), 12)


if __name__ == "__main__":
    unittest.main()
