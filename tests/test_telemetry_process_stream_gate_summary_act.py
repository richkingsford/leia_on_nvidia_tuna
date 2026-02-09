import sys
import time
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {}
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 90.0,
        }
        self._smoothed_frame_history = []
        self._frame_id = 0
        self.last_visible_time = time.time()
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False


class TestTelemetryProcessStreamGateSummaryAct(unittest.TestCase):
    def test_action_sent_display_text_does_not_show_cmd_remap_arrow(self):
        text = telemetry_process.action_sent_display_text(
            "f",
            2,
            cmd_sent="b",
            pwm=123,
            duration_ms=250,
        )
        self.assertNotIn("->", text)
        self.assertTrue(text.startswith("F 2%"), text)
        self.assertIn("(pwm", text)
        self.assertIn("ms", text)

    def test_stream_summary_act_uses_alignment_cmd_instead_of_hold(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick["dist"] = 200.0  # suggests forward move
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        act_lines = [line for line in summary if isinstance(line, str) and line.startswith("ACT:")]
        self.assertEqual(len(act_lines), 1)
        self.assertTrue(act_lines[0].startswith("ACT: F"), act_lines[0])
        self.assertIn("(pwm", act_lines[0])
        self.assertIn("ms", act_lines[0])

    def test_stream_summary_act_holds_when_brick_not_visible_in_align_brick(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick["visible"] = False
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        act_lines = [line for line in summary if isinstance(line, str) and line.startswith("ACT:")]
        self.assertEqual(len(act_lines), 1)
        self.assertTrue(act_lines[0].startswith("ACT: HOLD"), act_lines[0])
        self.assertIn("brick not visible", act_lines[0].lower())

    def test_stream_summary_prefers_recent_sent_action_for_current_step(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick["dist"] = 80.0
        world.brick["x_axis"] = 0.0
        world._last_action_sent_display = "L 1%"
        world._last_action_obj = "ALIGN_BRICK"
        world._last_action_time = time.time()
        world._last_action_duration_ms = 200

        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        act_lines = [line for line in summary if isinstance(line, str) and line.startswith("ACT:")]
        self.assertEqual(len(act_lines), 1)
        self.assertEqual(act_lines[0], "ACT: L 1%")


if __name__ == "__main__":
    unittest.main()
