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
    def test_stream_success_gate_line_shows_dist_gap_and_keeps_mismatch_color(self):
        line = telemetry_process._stream_success_gate_line(
            "BRICK_LOCK_WALL",
            {
                "metric": "dist",
                "stats": {"target": 94.6, "tol": 4.3},
                "value": 112.9,
            },
        )
        self.assertIsInstance(line, dict)
        segments = line.get("segments") or []
        text = _line_text(line)
        self.assertIn("dist=94.6+/-4.3", text)
        self.assertIn("(+18.3)", text)
        self.assertNotIn("(112.9)", text)
        self.assertGreaterEqual(len(segments), 2)
        color = tuple(segments[1][1]) if isinstance(segments[1], (tuple, list)) and len(segments[1]) > 1 else None
        self.assertEqual(color, tuple(telemetry_process.STREAM_RED))

    def test_action_sent_display_text_uses_logical_command_even_with_wire_remap(self):
        text = telemetry_process.action_sent_display_text(
            "f",
            2,
            cmd_sent="b",
            pwm=123,
            duration_ms=250,
        )
        self.assertNotIn("->", text)
        self.assertTrue(text.startswith("F 2%"), text)
        self.assertNotIn("sent=", text)
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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(_auto_lines(summary), [])

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
        self.assertEqual(auto_segment_lines, [])

    def test_stream_summary_includes_auto_lite_gatecheck_continuation_line(self):
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.22, "tol": 4.0},
                    "dist": {"target": 154.61, "tol": 4.0},
                }
            }
        }
        world.brick["visible"] = True
        world.brick["x_axis"] = -6.0
        world.brick["dist"] = 170.0
        world._gatecheck_status = {
            "mode": "lite",
            "checks": 2,
            "truth_ok": False,
            "lite_collected": 2,
            "lite_required": 3,
        }
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 2
        world._gatecheck_lite_passed = False

        summary, _ = telemetry_process.compute_stream_gate_summary(world, "BRICK_LOCK", active=True)
        all_text = [_line_text(line) for line in summary]
        self.assertFalse(any(text.startswith("AUTO: ") for text in all_text))
        self.assertFalse(any("lite gatecheck:" in text for text in all_text))


if __name__ == "__main__":
    unittest.main()
