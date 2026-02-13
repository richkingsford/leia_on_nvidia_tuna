import sys
import io
import unittest
from contextlib import redirect_stdout
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
            "dist": 100.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 90.0,
        }
        self.last_visible_time = None
        self._frame_id = 0


class TestTelemetryProcessAutoDiag(unittest.TestCase):
    def test_success_gate_metrics_excludes_angle_for_align_and_position(self):
        metrics = ["angle_abs", "xAxis_offset_abs", "dist", "visible"]
        align = telemetry_process.success_gate_metrics_for_step(metrics, "ALIGN_BRICK", step_rules={})
        position = telemetry_process.success_gate_metrics_for_step(metrics, "POSITION_BRICK", step_rules={})
        find_wall = telemetry_process.success_gate_metrics_for_step(metrics, "FIND_WALL", step_rules={})
        self.assertNotIn("angle_abs", align)
        self.assertNotIn("angle_abs", position)
        self.assertIn("angle_abs", find_wall)

    def test_auto_diag_reports_mm_closer_and_colors_act_and_delta(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        world.brick["x_axis"] = 10.0
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 7.0
        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "L 20%",
            "xAxis_offset_abs=10.0mm",
            pre_focus=pre_focus,
        )
        self.assertIn("FAILED success gates", diag["plain"])
        self.assertIn("so I L 20%,", diag["plain"])
        self.assertIn("getting us 3.0mm closer to the success gates", diag["plain"])
        self.assertIn("(xAxis_offset_abs=7.0mm)", diag["plain"])
        self.assertIn(telemetry_process.COLOR_ORANGE_BRIGHT, diag["colored"])
        self.assertIn(telemetry_process.COLOR_GREEN, diag["colored"])

    def test_auto_diag_reports_mm_further_in_red(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        world.brick["x_axis"] = 7.0
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 10.0
        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "R 20%",
            "xAxis_offset_abs=7.0mm",
            pre_focus=pre_focus,
        )
        self.assertIn("getting us 3.0mm further from the success gates", diag["plain"])
        self.assertIn(telemetry_process.COLOR_RED, diag["colored"])

    def test_auto_diag_align_step_pre_snapshot_shows_only_focus_offset_metric(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -1.4, "tol": 0.7},
                    "dist": {"target": 98.0, "tol": 1.5},
                }
            }
        }
        world.brick["visible"] = True
        world.brick["x_axis"] = 6.5
        world.brick["dist"] = 98.2
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 6.1
        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "R 3% (pwm=105, power=0.315, 251ms)",
            "visible=true (>=true), xAxis_offset_abs=6.5 (=-1.4+/-0.7), dist=98.2 (=98.0+/-1.5)",
            pre_focus=pre_focus,
        )
        self.assertIn("FAILED success gates (xAxis_offset_abs=6.5mm (=-1.4+/-0.7))", diag["plain"])
        self.assertNotIn("visible=true", diag["plain"])
        self.assertNotIn("dist=98.2", diag["plain"])

    def test_auto_diag_visible_only_success_uses_success_wording(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = True
        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "FIND_WALL",
            "R 6%",
            "visible=false",
            success_override=True,
        )
        self.assertIn("[AUTO] SUCCESS:", diag["plain"])
        self.assertNotIn("FAILED success gates", diag["plain"])
        self.assertTrue(bool(diag.get("success_hit")))

    def test_emit_auto_diag_skips_stdout_when_success_hit(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = True
        buf = io.StringIO()
        with redirect_stdout(buf):
            line = telemetry_process.emit_auto_step_diagnostic(
                world,
                "FIND_WALL",
                "R 6%",
                "visible=false",
                emit=True,
                success_override=True,
            )
        self.assertIn("[AUTO] SUCCESS:", line)
        self.assertEqual(buf.getvalue(), "")

    def test_queued_auto_diag_uses_pre_focus_for_delta(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        world.brick["x_axis"] = 10.0
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        telemetry_process.queue_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "L 20%",
            "xAxis_offset_abs=10.0mm",
            pre_focus=pre_focus,
        )
        world._frame_id = 1
        world.brick["x_axis"] = 7.0
        line = telemetry_process.flush_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            force=True,
            emit=False,
        )
        self.assertIn("getting us 3.0mm closer to the success gates", line)
        self.assertEqual(world._last_auto_step_diag_line, line)

    def test_flush_auto_diag_does_not_emit_on_non_force_ticks(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                }
            }
        }
        telemetry_process.queue_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "L 20%",
            "xAxis_offset_abs=10.0mm",
        )
        world._frame_id = 1
        buf = io.StringIO()
        with redirect_stdout(buf):
            line = telemetry_process.flush_auto_step_diagnostic(
                world,
                "ALIGN_BRICK",
                force=False,
                emit=True,
            )
        self.assertIsNotNone(line)
        self.assertEqual(buf.getvalue(), "")

    def test_auto_step_action_stats_count_closer_and_backward(self):
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 100.0, "tol": 2.0},
                }
            }
        }
        telemetry_process.reset_auto_step_action_stats(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 10.0
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 7.0
        telemetry_process.emit_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "L 2%",
            "xAxis_offset_abs=10.0mm",
            emit=False,
            pre_focus=pre_focus,
        )
        world.brick["x_axis"] = 8.5
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "ALIGN_BRICK")
        world.brick["x_axis"] = 12.0
        telemetry_process.emit_auto_step_diagnostic(
            world,
            "ALIGN_BRICK",
            "R 2%",
            "xAxis_offset_abs=8.5mm",
            emit=False,
            pre_focus=pre_focus,
        )
        stats = telemetry_process.consume_auto_step_action_stats(world, "ALIGN_BRICK")
        self.assertEqual(int(stats.get("total") or 0), 2)
        self.assertEqual(int(stats.get("closer") or 0), 1)
        self.assertEqual(int(stats.get("backward") or 0), 1)

    def test_auto_step_action_stats_include_unchanged_and_reconcile_totals(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        telemetry_process.reset_auto_step_action_stats(world, "FIND_WALL2")

        # Act 1: still not visible -> unchanged
        world.brick["visible"] = False
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "FIND_WALL2")
        world.brick["visible"] = False
        telemetry_process.emit_auto_step_diagnostic(
            world,
            "FIND_WALL2",
            "R 6%",
            "visible=false",
            emit=False,
            pre_focus=pre_focus,
            success_override=False,
        )

        # Act 2: not visible -> visible -> closer
        world.brick["visible"] = False
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "FIND_WALL2")
        world.brick["visible"] = True
        telemetry_process.emit_auto_step_diagnostic(
            world,
            "FIND_WALL2",
            "R 6%",
            "visible=false",
            emit=False,
            pre_focus=pre_focus,
            success_override=True,
        )

        # Act 3: stays visible -> unchanged
        world.brick["visible"] = True
        pre_focus = telemetry_process._capture_auto_diag_focus(world, "FIND_WALL2")
        world.brick["visible"] = True
        telemetry_process.emit_auto_step_diagnostic(
            world,
            "FIND_WALL2",
            "L 6%",
            "visible=true",
            emit=False,
            pre_focus=pre_focus,
            success_override=True,
        )

        stats = telemetry_process.consume_auto_step_action_stats(world, "FIND_WALL2")
        total = int(stats.get("total") or 0)
        closer = int(stats.get("closer") or 0)
        backward = int(stats.get("backward") or 0)
        unchanged = int(stats.get("unchanged") or 0)
        unknown = int(stats.get("unknown") or 0)

        self.assertEqual(total, 3)
        self.assertEqual(closer, 1)
        self.assertEqual(backward, 0)
        self.assertEqual(unchanged, 2)
        self.assertEqual(unknown, 0)
        self.assertEqual(total, closer + backward + unchanged + unknown)


if __name__ == "__main__":
    unittest.main()
