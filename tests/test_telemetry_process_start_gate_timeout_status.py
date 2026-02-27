import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "POSITION_BRICK": {
                "start_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.0, "tol": 1.5},
                }
            }
        }
        self.brick = {
            "visible": True,
            "dist": 90.0,
            "angle": 0.0,
            "x_axis": -9.5,
            "offset_x": -9.5,
            "y_axis": 0.0,
            "offset_y": 0.0,
            "confidence": 90.0,
        }
        self.wall = {}
        self._brick_frame_buffer = []
        self.wall_envelope = object()


class TestTelemetryProcessStartGateTimeoutStatus(unittest.TestCase):
    def test_timeout_status_lists_targets_and_colored_states(self):
        world = _DummyWorld()
        text = telemetry_process._format_start_gate_timeout_status(
            world,
            "POSITION_BRICK",
            "start_gate: origin drift 182.3mm",
        )
        self.assertIn("timeout after", text)
        self.assertIn("last blocked: origin drift 182.3mm", text)
        self.assertIn("start gates:", text)
        self.assertIn(
            f"visible target=true state={telemetry_process.COLOR_GREEN}true{telemetry_process.COLOR_RESET}",
            text,
        )
        self.assertIn(
            f"xAxis_offset_abs target=-4.0 +/- 1.5 state={telemetry_process.COLOR_RED}-9.50{telemetry_process.COLOR_RESET}",
            text,
        )

    def test_timeout_status_colors_failing_visible_state_red(self):
        world = _DummyWorld()
        world.brick["visible"] = False
        world.process_rules["POSITION_BRICK"]["start_gates"] = {"visible": {"min": True}}
        text = telemetry_process._format_start_gate_timeout_status(
            world,
            "POSITION_BRICK",
            "start_gate: visible gate",
        )
        self.assertIn(
            f"visible target=true state={telemetry_process.COLOR_RED}false{telemetry_process.COLOR_RESET}",
            text,
        )

    def test_start_gate_status_details_include_targets_and_states(self):
        world = _DummyWorld()
        text = telemetry_process._format_start_gate_status_details(world, "POSITION_BRICK")
        self.assertIsNotNone(text)
        self.assertIn("start gates:", text)
        self.assertIn(
            f"visible target=true state={telemetry_process.COLOR_GREEN}true{telemetry_process.COLOR_RESET}",
            text,
        )
        self.assertIn("xAxis_offset_abs target=-4.0 +/- 1.5", text)

    def test_wait_for_start_gates_met_log_includes_detailed_status(self):
        world = _DummyWorld()
        ok_gate = telemetry_process.telemetry_brick.GateCheck(ok=True, reasons=[])
        with patch.object(telemetry_process, "update_world_from_vision", return_value=None), \
             patch.object(telemetry_process, "GATE_STABILITY_FRAMES", 1), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(telemetry_process.telemetry_brick, "evaluate_start_gates", return_value=ok_gate), \
             patch.object(telemetry_process.telemetry_wall, "evaluate_start_gates", return_value=ok_gate), \
             patch.object(telemetry_process.telemetry_robot_module, "evaluate_start_gates", return_value=ok_gate), \
             patch("builtins.print") as mock_print:
            status = telemetry_process.wait_for_start_gates(
                world,
                object(),
                "POSITION_BRICK",
                log=True,
                allow_success=False,
            )
        self.assertEqual(status, "start")
        logged = "\n".join(str((call.args or [""])[0]) for call in mock_print.call_args_list)
        self.assertIn("[START] POSITION_BRICK start gates met", logged)
        self.assertIn("start gates:", logged)
        self.assertIn("visible target=true state=", logged)


if __name__ == "__main__":
    unittest.main()
