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


def _line_text(line):
    if isinstance(line, str):
        return line
    if isinstance(line, dict):
        segments = line.get("segments")
        if isinstance(segments, list):
            out = []
            for seg in segments:
                if isinstance(seg, dict):
                    out.append(str(seg.get("text") or ""))
                elif isinstance(seg, (tuple, list)) and seg:
                    out.append(str(seg[0]))
            return "".join(out)
    return str(line)


def _auto_lines(summary):
    lines = []
    for line in summary:
        text = _line_text(line)
        if text.startswith("AUTO:"):
            lines.append(text)
    return lines


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
        self.assertIn("sent=B", text)
        self.assertIn("pwm=", text)
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
        self.assertFalse(any(str(line).startswith("SENT:") for line in summary))
        self.assertFalse(any(str(line).startswith("NEXT:") for line in summary))
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("so I", auto_lines[0])
        self.assertNotIn("HOLD", auto_lines[0])

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
        self.assertFalse(any(str(line).startswith("SENT:") for line in summary))
        self.assertFalse(any(str(line).startswith("NEXT:") for line in summary))
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("HOLD", auto_lines[0])
        self.assertIn("brick not visible", auto_lines[0].lower())

    def test_stream_summary_visible_only_step_uses_resulting_wording(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.process_rules["ALIGN_BRICK"]["scan_direction"] = "b"
        world.brick["visible"] = False
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("FAILED success gates (visible=false), so I", auto_lines[0])
        self.assertIn("resulting in NOT meeting the success gates", auto_lines[0])

    def test_stream_summary_numeric_gates_keep_getting_us_wording(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick["dist"] = 120.0
        world.brick["x_axis"] = 10.0
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("getting us", auto_lines[0])

    def test_stream_summary_uses_fresh_auto_snapshot_for_auto_line(self):
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
        self.assertFalse(any(str(line).startswith("SENT:") for line in summary))
        self.assertFalse(any(str(line).startswith("NEXT:") for line in summary))
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("so I R 20%", auto_lines[0])

    def test_stream_summary_nominal_step_keeps_act_only_wording(self):
        world = _DummyWorld()
        world.process_rules = {
            "PLACE": {
                "nominalDemosOnly": True,
            }
        }
        world._last_action_sent_display = "B 20% (pwm131, 260ms)"
        world._last_action_obj = "PLACE"
        world._last_action_time = time.time()
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "PLACE", active=True)
        self.assertFalse(any(str(line).startswith("SENT:") for line in summary))
        self.assertFalse(any(str(line).startswith("NEXT:") for line in summary))
        auto_lines = _auto_lines(summary)
        self.assertEqual(len(auto_lines), 1)
        self.assertIn("ACT ONLY (nominal): B 20%", auto_lines[0])

    def test_stream_summary_auto_line_contains_colored_segments(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick["dist"] = 120.0
        world.brick["x_axis"] = 10.0
        summary, _ = telemetry_process.compute_stream_gate_summary(world, "ALIGN_BRICK", active=True)
        auto_segment_lines = [
            line
            for line in summary
            if isinstance(line, dict)
            and isinstance(line.get("segments"), list)
            and _line_text(line).startswith("AUTO:")
        ]
        self.assertEqual(len(auto_segment_lines), 1)
        segment_colors = []
        for seg in auto_segment_lines[0].get("segments") or []:
            if isinstance(seg, (tuple, list)) and len(seg) > 1:
                segment_colors.append(tuple(seg[1]))
        self.assertIn(tuple(telemetry_process.STREAM_ORANGE), segment_colors)


if __name__ == "__main__":
    unittest.main()
