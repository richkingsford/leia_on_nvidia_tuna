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
        self.assertIn("so I L 20%.", diag["plain"])
        self.assertIn("\ngetting us", diag["plain"])
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

    def test_auto_diag_failure_uses_multiline_result_and_word_only_red_highlight(self):
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world.brick["visible"] = True
        world.brick["confidence"] = 95.0
        world.last_visible_time = None
        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "EXIT_WALL",
            "U 1% (pwm=93, power=0.260, 250ms)",
            "visible=true",
            success_override=False,
        )
        self.assertIn(
            "\nresulting in NOT meeting the success gates (visible=true).",
            diag["plain"],
        )
        self.assertIn(
            f"{telemetry_process.COLOR_RED}FAILED{telemetry_process.COLOR_RESET} success gates",
            diag["colored"],
        )
        self.assertIn(
            f"resulting in {telemetry_process.COLOR_RED}NOT{telemetry_process.COLOR_RESET} meeting the success gates",
            diag["colored"],
        )
        self.assertNotIn(
            f"{telemetry_process.COLOR_RED}FAILED success gates{telemetry_process.COLOR_RESET}",
            diag["colored"],
        )
        self.assertNotIn(
            f"{telemetry_process.COLOR_RED}NOT meeting the success gates{telemetry_process.COLOR_RESET}",
            diag["colored"],
        )

    def test_auto_diag_appends_lite_details_in_gray_with_value_highlights(self):
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

        diag = telemetry_process._build_auto_step_diagnostic(
            world,
            "BRICK_LOCK",
            "L 3%",
            "xAxis_offset_abs=-6.0mm",
        )

        self.assertIn("\nlite gatecheck:", diag["plain"])
        self.assertIn("xAxis_offset_abs (", diag["plain"])
        self.assertIn("dist (", diag["plain"])
        self.assertIn(telemetry_process.COLOR_GRAY, diag["colored"])
        self.assertIn(f"{telemetry_process.COLOR_GREEN}(", diag["colored"])
        self.assertIn(f"{telemetry_process.COLOR_RED}(", diag["colored"])

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

    def test_success_event_lines_use_strict_boolean_gate_text(self):
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.last_visible_time = None

        lines = telemetry_process.format_success_event_lines(world, "EXIT_WALL")
        joined = " ".join(lines)
        self.assertIn("=false", joined)
        self.assertNotIn(">=false", joined)

    def test_success_event_lines_color_current_state_red_when_not_matching_gate(self):
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world.brick["visible"] = True
        world.brick["confidence"] = 95.0
        world.last_visible_time = None

        lines = telemetry_process.format_success_event_lines(world, "EXIT_WALL", colored=True)
        joined = " ".join(lines)
        self.assertIn(telemetry_process.COLOR_RED, joined)
        self.assertIn("visible=", joined)
        self.assertIn(
            f"visible={telemetry_process.COLOR_RED}true{telemetry_process.COLOR_RESET}",
            joined,
        )
        self.assertIn(" (grace=0.20s)", joined)
        self.assertNotIn(
            f"{telemetry_process.COLOR_RED}true (grace=0.20s){telemetry_process.COLOR_RESET}",
            joined,
        )

    def test_success_event_lines_color_current_state_green_when_matching_gate(self):
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.last_visible_time = None

        lines = telemetry_process.format_success_event_lines(world, "EXIT_WALL", colored=True)
        joined = " ".join(lines)
        self.assertIn(telemetry_process.COLOR_GREEN, joined)
        self.assertIn("visible=", joined)
        self.assertIn(
            f"visible={telemetry_process.COLOR_GREEN}false{telemetry_process.COLOR_RESET}",
            joined,
        )
        self.assertIn(" (grace=0.20s)", joined)
        self.assertNotIn(
            f"{telemetry_process.COLOR_GREEN}false (grace=0.20s){telemetry_process.COLOR_RESET}",
            joined,
        )

    def test_align_result_observation_formatter_uses_single_canonical_better_shape(self):
        plain, colored, color = telemetry_process._format_align_result_observation(
            "dist_err",
            -9.25,
            -10.54,
        )
        self.assertEqual(color, telemetry_process.COLOR_GREEN)
        self.assertEqual(
            plain,
            "Result: +1.29mm better than previous dist_err=-10.54mm",
        )
        self.assertIn(
            f"{telemetry_process.COLOR_BLUE_BRIGHT}Result:{telemetry_process.COLOR_RESET}",
            colored,
        )
        self.assertIn(
            f"{telemetry_process.COLOR_GREEN}+1.29mm{telemetry_process.COLOR_RESET}",
            colored,
        )
        self.assertNotIn(
            f"{telemetry_process.COLOR_GREEN}+1.29mm better than previous",
            colored,
        )

    def test_align_result_observation_formatter_uses_single_canonical_overshot_shape(self):
        plain, colored, color = telemetry_process._format_align_result_observation(
            "x_err",
            -0.80,
            +0.92,
            overshot=True,
        )
        self.assertEqual(color, telemetry_process.COLOR_RED)
        self.assertEqual(
            plain,
            "Result: overshot target; previous x_err=+0.92mm",
        )
        self.assertIn(
            f"{telemetry_process.COLOR_BLUE_BRIGHT}Result:{telemetry_process.COLOR_RESET}",
            colored,
        )
        self.assertNotIn("Result observation:", colored)

    def test_result_lite_gate_detail_colors_first_word_pink(self):
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.22, "tol": 4.0},
                }
            }
        }
        world._gatecheck_status = {
            "mode": "lite",
            "checks": 1,
            "truth_ok": False,
            "lite_collected": 1,
            "lite_required": 3,
        }
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 1
        world._gatecheck_lite_passed = False

        detail = telemetry_process._result_lite_gate_detail(world, "BRICK_LOCK")
        self.assertIsInstance(detail, dict)
        self.assertIn("lite gatecheck:", str(detail.get("plain")))
        self.assertNotIn("mode=lite", str(detail.get("plain")))
        self.assertNotIn("/=", str(detail.get("plain")))
        self.assertIn("!=", str(detail.get("plain")))
        self.assertIn(
            f"{telemetry_process.COLOR_PINK}lite{telemetry_process.COLOR_RESET}{telemetry_process.COLOR_GRAY} gatecheck:",
            str(detail.get("colored")),
        )
        self.assertIn(
            f"{telemetry_process.COLOR_WHITE}xAxis_offset_abs{telemetry_process.COLOR_RESET}",
            str(detail.get("colored")),
        )


if __name__ == "__main__":
    unittest.main()
