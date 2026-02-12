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
        self.assertTrue(text.startswith("B 2%"), text)
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
        next_lines = [line for line in summary if isinstance(line, str) and line.startswith("NEXT:")]
        self.assertEqual(len(next_lines), 1)
        expected_cmd = (
            telemetry_process.telemetry_robot_module.COMMAND_REMAP.get("f", "f").upper()
            if isinstance(getattr(telemetry_process.telemetry_robot_module, "COMMAND_REMAP", None), dict)
            else "F"
        )
        self.assertTrue(next_lines[0].startswith(f"NEXT: {expected_cmd}"), next_lines[0])
        self.assertIn("(pwm", next_lines[0])
        self.assertIn("ms", next_lines[0])

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
        next_lines = [line for line in summary if isinstance(line, str) and line.startswith("NEXT:")]
        self.assertEqual(len(next_lines), 1)
        self.assertTrue(next_lines[0].startswith("NEXT: HOLD"), next_lines[0])
        self.assertIn("brick not visible", next_lines[0].lower())

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
        sent_lines = [line for line in summary if isinstance(line, str) and line.startswith("SENT:")]
        self.assertEqual(len(sent_lines), 1)
        self.assertEqual(sent_lines[0], "SENT: L 1%")

    def test_stream_summary_uses_fresh_auto_snapshot_for_sent_line(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = False
        world._last_action_sent_display = "F 20% (pwm131, 260ms)"
        world._last_action_obj = "ALIGN_BRICK"
        world._last_action_wire = "f 131 260"
        world._last_action_wire_step = "ALIGN_BRICK"
        world._last_auto_step_diag_line = (
            "[AUTO] Did NOT see success gates (visible=false (>=true)), so I R 20%; "
            "resulting in NOT meeting the success gates (visible=false)."
        )
        world._last_auto_step_diag_step = "ALIGN_BRICK"
        world._last_auto_step_diag_time = time.time()
        world._last_auto_step_diag_sent = {
            "sent_display": "R 20% (pwm131, 260ms)",
            "sent_step": "ALIGN_BRICK",
            "wire_text": "l 131 260",
            "wire_step": "ALIGN_BRICK",
        }

        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        sent_lines = [line for line in summary if isinstance(line, str) and line.startswith("SENT:")]
        auto_lines = [line for line in summary if isinstance(line, str) and line.startswith("AUTO:")]
        self.assertEqual(len(sent_lines), 1)
        self.assertEqual(
            sent_lines[0],
            "SENT: R 20% (pwm131, 260ms) [wire: l 131 260]",
        )
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("so I R 20%", auto_lines[0])

    def test_stream_summary_prefers_newer_live_sent_over_old_diag_snapshot(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = True
        now = time.time()
        world._last_action_sent_display = "L 20% (pwm131, 260ms)"
        world._last_action_obj = "ALIGN_BRICK"
        world._last_action_time = now
        world._last_action_wire = "l 131 260"
        world._last_action_wire_step = "ALIGN_BRICK"
        world._last_action_wire_time = now
        world._last_auto_step_diag_line = (
            "[AUTO] Did NOT see success gates (visible=false (>=true)), so I R 20%; "
            "resulting in NOT meeting the success gates (visible=false)."
        )
        world._last_auto_step_diag_step = "ALIGN_BRICK"
        world._last_auto_step_diag_time = now
        world._last_auto_step_diag_sent = {
            "sent_display": "R 20% (pwm131, 260ms)",
            "sent_step": "ALIGN_BRICK",
            "wire_text": "r 131 260",
            "wire_step": "ALIGN_BRICK",
            "sent_time": now - 1.0,
            "wire_time": now - 1.0,
        }

        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        sent_lines = [line for line in summary if isinstance(line, str) and line.startswith("SENT:")]
        self.assertEqual(len(sent_lines), 1)
        self.assertEqual(
            sent_lines[0],
            "SENT: L 20% (pwm131, 260ms) [wire: l 131 260]",
        )


if __name__ == "__main__":
    unittest.main()
