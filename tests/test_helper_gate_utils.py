import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_gate_utils


class TestHelperGateUtils(unittest.TestCase):
    def test_metric_value_from_measurement(self):
        measurement = {
            "visible": True,
            "angle": -10.0,
            "x_axis": 5.0,
            "offset_x": 5.0,
            "dist": 42.0,
        }
        self.assertEqual(helper_gate_utils.metric_value_from_measurement(measurement, "visible"), True)
        self.assertEqual(helper_gate_utils.metric_value_from_measurement(measurement, "angle_abs"), 10.0)
        self.assertEqual(helper_gate_utils.metric_value_from_measurement(measurement, "xAxis_offset_abs"), 5.0)
        self.assertEqual(helper_gate_utils.metric_value_from_measurement(measurement, "dist"), 42.0)

    def test_metric_error_target_tol(self):
        stats = {"target": 10.0, "tol": 2.0}
        self.assertEqual(helper_gate_utils.metric_error(11.0, stats), 0.0)
        self.assertEqual(helper_gate_utils.metric_error(13.0, stats), 1.0)

    def test_metric_error_min_max(self):
        stats = {"min": 3.0, "max": 7.0}
        self.assertEqual(helper_gate_utils.metric_error(2.0, stats), 1.0)
        self.assertEqual(helper_gate_utils.metric_error(8.0, stats), 1.0)
        self.assertEqual(helper_gate_utils.metric_error(5.0, stats), 0.0)

    def test_metric_progress_target_tol(self):
        stats = {"target": 10.0, "tol": 2.0}
        self.assertEqual(helper_gate_utils.metric_progress(10.0, stats), 1.0)
        self.assertAlmostEqual(helper_gate_utils.metric_progress(13.0, stats), 0.5, places=2)

    def test_gate_satisfied(self):
        gates = {"visible": {"min": True}, "dist": {"min": 10.0, "max": 20.0}}
        measurement = {"visible": True, "dist": 15.0}
        self.assertTrue(helper_gate_utils.gate_satisfied(measurement, gates))

    def test_step_progress(self):
        gates = {"visible": {"min": True}, "dist": {"target": 10.0, "tol": 2.0}}
        measurement = {"visible": True, "dist": 10.0}
        self.assertEqual(helper_gate_utils.step_progress(measurement, gates), 1.0)

    def test_satisfied_steps(self):
        steps = {
            "STEP_A": {"success_gates": {"visible": {"min": True}}},
            "STEP_B": {"success_gates": {"visible": {"min": False}}},
        }
        measurement = {"visible": True, "confidence": 90.0}
        satisfied = helper_gate_utils.satisfied_steps(measurement, steps)
        self.assertIn("STEP_A", satisfied)
        self.assertNotIn("STEP_B", satisfied)

    def test_visible_false_gate_fails_when_visibility_is_true_even_if_confidence_is_low(self):
        gates = {"visible": {"min": False}}
        measurement = {"visible": True, "confidence": 55.0}
        self.assertFalse(helper_gate_utils.gate_satisfied(measurement, gates))

    def test_visible_false_gate_fails_when_visibility_confidence_is_high(self):
        gates = {"visible": {"min": False}}
        measurement = {"visible": True, "confidence": 75.0}
        self.assertFalse(helper_gate_utils.gate_satisfied(measurement, gates))

    def test_visible_false_gate_passes_when_visibility_is_false(self):
        gates = {"visible": {"min": False}}
        measurement = {"visible": False, "confidence": 0.0}
        self.assertTrue(helper_gate_utils.gate_satisfied(measurement, gates))

    def test_gatecheck_tracker_status_and_stream_line(self):
        class DummyWorld:
            pass

        world = DummyWorld()
        world._frame_id = 10
        tracker = helper_gate_utils.SuccessGateTracker(
            consecutive_required=3,
            majority_window=5,
            majority_required=4,
        )
        self.assertFalse(helper_gate_utils.update_gatecheck(world, "ALIGN_BRICK", tracker, True, phase="align"))
        world._frame_id = 11
        self.assertFalse(helper_gate_utils.update_gatecheck(world, "ALIGN_BRICK", tracker, True, phase="align"))
        world._frame_id = 12
        self.assertTrue(helper_gate_utils.update_gatecheck(world, "ALIGN_BRICK", tracker, True, phase="align"))
        lines = helper_gate_utils.format_gatecheck_stream_lines(world, "ALIGN_BRICK")
        self.assertEqual(len(lines), 3)
        self.assertIn("CONSEC: 3/3 ok", lines[0])
        self.assertIn("SEEN: 3 total", lines[1])
        self.assertIn("win 3/5", lines[1])
        self.assertIn("MAJ: 3/5 pass", lines[2])
        self.assertIn("need:4", lines[2])

    def test_gatecheck_stream_line_lite_mode(self):
        class DummyWorld:
            pass

        world = DummyWorld()
        world._frame_id = 21
        world._gatecheck_mode = "lite"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 2
        tracker = helper_gate_utils.SuccessGateTracker(
            consecutive_required=1,
            majority_window=1,
            majority_required=1,
        )
        self.assertFalse(helper_gate_utils.update_gatecheck(world, "ALIGN_BRICK", tracker, False, phase="align"))
        lines = helper_gate_utils.format_gatecheck_stream_lines(world, "ALIGN_BRICK")
        self.assertEqual(len(lines), 3)
        self.assertIn("LITE: 2/3 avg-smoothed frames", lines[0])
        self.assertIn("SEEN: 1 total", lines[1])
        self.assertIn("LITE-GATE: wait", lines[2])

    def test_load_gate_checker_config_sanitizes_values(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "world_model_gate_checker.json"
            path.write_text(
                json.dumps(
                    {
                        "gate_checker": {
                            "consecutive_required": 0,
                            "majority_window": 4,
                            "majority_required": 7,
                        }
                    }
                )
            )
            cfg = helper_gate_utils.load_gate_checker_config(path)
        self.assertEqual(cfg["consecutive_required"], 1)
        self.assertEqual(cfg["majority_window"], 4)
        self.assertEqual(cfg["majority_required"], 4)

    def test_should_hold_for_success_confirmation_with_pending_majority_window(self):
        tracker = helper_gate_utils.SuccessGateTracker(
            consecutive_required=3,
            majority_window=5,
            majority_required=4,
        )
        tracker.update(True)
        self.assertTrue(
            helper_gate_utils.should_hold_for_success_confirmation(
                visible_only=True,
                tracker=tracker,
                success_met=False,
            )
        )
        tracker.update(False)
        self.assertTrue(
            helper_gate_utils.should_hold_for_success_confirmation(
                visible_only=True,
                tracker=tracker,
                success_met=False,
            )
        )
        tracker.update(False)
        tracker.update(False)
        tracker.update(False)
        self.assertFalse(
            helper_gate_utils.should_hold_for_success_confirmation(
                visible_only=True,
                tracker=tracker,
                success_met=False,
            )
        )

    def test_wait_for_fresh_frames(self):
        class DummyWorld:
            pass

        world = DummyWorld()
        world._frame_id = 0

        def _tick():
            world._frame_id += 1

        info = helper_gate_utils.wait_for_fresh_frames(
            world,
            _tick,
            required_new_frames=3,
            max_cycles=5,
        )
        self.assertEqual(info["required"], 3)
        self.assertEqual(info["advanced"], 3)

    def test_gatecheck_visible_only_passes_on_half_consecutive(self):
        class DummyWorld:
            pass

        world = DummyWorld()
        world._frame_id = 100
        tracker = helper_gate_utils.SuccessGateTracker(
            consecutive_required=12,
            majority_window=18,
            majority_required=9,
        )
        tracker.consecutive_pass_required = 6
        tracker.majority_pass_required = 9

        success_met = False
        for idx in range(6):
            world._frame_id = 100 + idx
            success_met = helper_gate_utils.update_gatecheck(
                world,
                "EXIT_WALL",
                tracker,
                True,
                phase="replay",
            )

        self.assertTrue(success_met)
        status = getattr(world, "_gatecheck_status", {}) or {}
        self.assertEqual(status.get("truth_by"), "consecutive")
        self.assertEqual(status.get("need"), 12)
        self.assertEqual(status.get("need_pass"), 6)

    def test_gatecheck_visible_only_passes_on_half_majority_without_full_window(self):
        class DummyWorld:
            pass

        world = DummyWorld()
        world._frame_id = 200
        tracker = helper_gate_utils.SuccessGateTracker(
            consecutive_required=12,
            majority_window=18,
            majority_required=9,
        )
        tracker.consecutive_pass_required = 6
        tracker.majority_pass_required = 9

        success_met = False
        for idx in range(17):
            world._frame_id = 200 + idx
            success_ok = (idx % 2) == 0
            success_met = helper_gate_utils.update_gatecheck(
                world,
                "EXIT_WALL",
                tracker,
                success_ok,
                phase="replay",
            )
            if idx < 16:
                self.assertFalse(success_met)

        self.assertTrue(success_met)
        status = getattr(world, "_gatecheck_status", {}) or {}
        self.assertEqual(status.get("truth_by"), "majority")
        self.assertEqual(status.get("window_pass"), 9)
        self.assertEqual(status.get("window_size"), 17)
        self.assertEqual(status.get("window_need_pass"), 9)


if __name__ == "__main__":
    unittest.main()
