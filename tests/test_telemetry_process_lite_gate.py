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
            "visible": False,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 0.0,
        }
        self._smoothed_frame_history = []
        self._frame_id = 0
        self.last_visible_time = time.time()
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False


class TestTelemetryProcessLiteGate(unittest.TestCase):
    def setUp(self):
        self.prev_default = getattr(telemetry_process, "LITE_GATE_DEFAULT_UNIQUE_FRAMES", 3)
        self.prev_steps = dict(getattr(telemetry_process, "LITE_GATE_STEP_UNIQUE_FRAMES", {}))

    def tearDown(self):
        telemetry_process.LITE_GATE_DEFAULT_UNIQUE_FRAMES = self.prev_default
        telemetry_process.LITE_GATE_STEP_UNIQUE_FRAMES = self.prev_steps

    def test_apply_lite_gate_check_config_parses_step_frames(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "default_unique_smoothed_frames": 3,
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                    "POSITION_BRICK": {"enabled": True, "unique_smoothed_frames": 4},
                },
            }
        )
        self.assertEqual(telemetry_process.lite_gate_unique_frames("ALIGN_BRICK"), 3)
        self.assertEqual(telemetry_process.lite_gate_unique_frames("POSITION_BRICK"), 4)
        self.assertIsNone(telemetry_process.lite_gate_unique_frames("FIND_BRICK"))

    def test_evaluate_gate_status_uses_lite_average_for_configured_step(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                }
            }
        )
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick.update(
            {
                "visible": True,
                "dist": 80.0,
                "angle": 0.0,
                "offset_x": -2.0,
                "x_axis": -2.0,
                "confidence": 92.0,
            }
        )
        world._smoothed_frame_history = [
            {
                "frame_id": 11,
                "visible": True,
                "dist": 79.0,
                "angle": 0.0,
                "x_axis": -2.1,
                "offset_x": -2.1,
                "confidence": 90.0,
            },
            {
                "frame_id": 12,
                "visible": True,
                "dist": 80.0,
                "angle": 0.0,
                "x_axis": -2.0,
                "offset_x": -2.0,
                "confidence": 92.0,
            },
            {
                "frame_id": 13,
                "visible": True,
                "dist": 81.0,
                "angle": 0.0,
                "x_axis": -1.9,
                "offset_x": -1.9,
                "confidence": 94.0,
            },
        ]
        ok, _ = telemetry_process.evaluate_gate_status(world, "ALIGN_BRICK")
        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")
        self.assertEqual(world._gatecheck_lite_required, 3)
        self.assertEqual(world._gatecheck_lite_collected, 3)

    def test_evaluate_gate_status_falls_back_to_traditional_for_other_steps(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                }
            }
        )
        world = _DummyWorld()
        world.process_rules = {
            "FIND_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = True
        world.last_visible_time = time.time()
        ok, _ = telemetry_process.evaluate_gate_status(world, "FIND_BRICK")
        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")
        self.assertEqual(world._gatecheck_lite_required, 0)
        self.assertEqual(world._gatecheck_lite_collected, 0)


if __name__ == "__main__":
    unittest.main()
