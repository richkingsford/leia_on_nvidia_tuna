import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 100.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 95.0,
        }
        self.last_visible_time = time.time()
        self._frame_id = 1
        self._gatecheck_mode = "traditional"


class TestTelemetryProcessGatecheckFailureLogging(unittest.TestCase):
    def test_gatecheck_next_action_text_known_action(self):
        text = telemetry_process._gatecheck_next_action_text(
            "r",
            speed_score=20,
            reason="adaptive replay",
        )
        self.assertIn("continuing with next act", text)
        self.assertIn("R 20%", text)
        self.assertIn("adaptive replay", text)

    def test_gatecheck_next_action_text_unknown_action(self):
        text = telemetry_process._gatecheck_next_action_text(
            None,
            speed_score=None,
            reason="replay has no command to continue",
        )
        self.assertIn("continuing with adjustment acts", text)
        self.assertIn("replay has no command to continue", text)

    def test_gatecheck_failure_detail_includes_failed_metric_state(self):
        world = _DummyWorld()
        detail = telemetry_process._gatecheck_failure_detail(world, "EXIT_WALL")
        self.assertIn("visible", detail)
        self.assertIn("state=true", detail)
        self.assertIn("gate=false", detail)

    def test_observe_success_gatecheck_does_not_hold_on_failed_sample(self):
        world = _DummyWorld()
        tracker = telemetry_process.gate_utils.SuccessGateTracker(6, 9, 5)
        tracker.window = [True, True]
        tracker.consecutive = 2

        with mock.patch.object(telemetry_process, "evaluate_gate_status", return_value=(False, 0.0)), \
             mock.patch.object(telemetry_process, "_evaluate_instant_success_truth", return_value=True), \
             mock.patch.object(telemetry_process, "_update_success_gate_metric_tallies", return_value={}), \
             mock.patch.object(telemetry_process, "update_gatecheck_with_precheck", return_value=False), \
             mock.patch.object(
                 telemetry_process.gate_utils,
                 "should_hold_for_success_confirmation",
                 return_value=True,
             ) as hold_mock:
            result = telemetry_process.observe_success_gatecheck(
                world,
                "EXIT_WALL",
                tracker,
                phase="tail",
                log=False,
            )

        self.assertFalse(bool(result.get("effective_success_ok")))
        self.assertFalse(bool(result.get("hold_for_confirm")))
        hold_mock.assert_not_called()

    def test_lite_gate_detail_traditional_mode_uses_current_sample(self):
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world._gatecheck_status = {
            "mode": "traditional",
            "checks": 26,
            "truth_ok": False,
            "lite_required": 1,
            "lite_collected": 1,
        }
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 1
        world._gatecheck_lite_collected = 1
        world._gatecheck_lite_passed = True

        world.brick["visible"] = False
        world._smoothed_frame_history = [
            {
                "frame_id": 1,
                "visible": True,
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
                "offset_x": world.brick.get("offset_x"),
                "y_axis": 0.0,
                "offset_y": 0.0,
                "confidence": world.brick.get("confidence"),
            }
        ]

        detail = telemetry_process._auto_diag_lite_gate_detail(world, "EXIT_WALL")
        self.assertIsInstance(detail, dict)
        plain = str(detail.get("plain") or "")
        self.assertIn("mode=traditional (lite passed)", plain)
        self.assertIn("sample=wait", plain)
        self.assertIn("visible (!= Expected true & saw false)", plain)


if __name__ == "__main__":
    unittest.main()
