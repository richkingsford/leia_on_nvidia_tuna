import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessFindWall2Gates(unittest.TestCase):
    def test_derive_success_gates_includes_find_wall2_x_and_y_alignment(self):
        success_segments = {
            "FIND_WALL2": [
                {
                    "states": [
                        {"timestamp": 1.0, "brick": {"visible": False, "x_axis": -20.0, "y_axis": 12.0}},
                        {"timestamp": 2.0, "brick": {"visible": True, "x_axis": -3.0, "y_axis": 2.0}},
                        {"timestamp": 3.0, "brick": {"visible": True, "x_axis": -2.0, "y_axis": 1.0}},
                        {"timestamp": 4.0, "brick": {"visible": True, "x_axis": -1.0, "y_axis": 0.5}},
                    ]
                }
            ]
        }
        step_rules = {
            "FIND_WALL2": {
                "alignment_metrics": ["visible", "xAxis_offset_abs", "y_axis"],
            }
        }
        gates = telemetry_process.derive_success_gates(
            success_segments,
            scale_by_step={},
            step_rules=step_rules,
        )
        self.assertIn("FIND_WALL2", gates)
        derived = gates["FIND_WALL2"] or {}
        visible_gate = derived.get("visible") or {}
        self.assertIs(visible_gate.get("min"), True)
        self.assertIn("xAxis_offset_abs", derived)
        self.assertIn("y_axis", derived)
        self.assertNotIn("yAxis_offset_abs", derived)

    def test_find_wall2_success_metrics_prefer_y_axis_alias(self):
        filtered = telemetry_process.success_gate_metrics_for_step(
            ["visible", "xAxis_offset_abs", "yAxis_offset_abs", "dist"],
            "FIND_WALL2",
            step_rules={},
        )
        self.assertIn("y_axis", filtered)
        self.assertNotIn("yAxis_offset_abs", filtered)

    def test_process_model_find_brick_includes_x_and_y_success_gates(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        find_brick_cfg = (steps or {}).get("FIND_BRICK") if isinstance(steps, dict) else {}
        gates = (find_brick_cfg or {}).get("success_gates") if isinstance(find_brick_cfg, dict) else {}
        self.assertIsInstance(gates, dict)
        self.assertIn("visible", gates)
        self.assertIn("xAxis_offset_abs", gates)
        self.assertIn("yAxis_offset_abs", gates)

    def test_process_model_find_topmost_brick_does_not_require_stack_booleans(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_TOPMOST_BRICK") if isinstance(steps, dict) else {}
        gates = (step_cfg or {}).get("success_gates") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(gates, dict)
        self.assertIn("inCrosshairs", gates)
        self.assertNotIn("brick_above", gates)
        self.assertNotIn("brick_below", gates)

    def test_process_model_find_topmost_brick_wall_level2_uses_non_reset_skip_policy(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_TOPMOST_BRICK_WALL") if isinstance(steps, dict) else {}
        exception_cfg = (
            (step_cfg or {}).get("topmost_crosshair_exception")
            if isinstance(step_cfg, dict)
            else {}
        )
        self.assertIsInstance(exception_cfg, dict)
        self.assertIs(exception_cfg.get("level2_require_visible_for_confirm"), False)
        self.assertIs(exception_cfg.get("level2_reset_on_skipped_observation"), False)
        self.assertIs(exception_cfg.get("level2_fail_on_skipped_observation"), False)

    def test_process_model_find_wall2_includes_y_axis_success_gate(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL2") if isinstance(steps, dict) else {}
        gates = (step_cfg or {}).get("success_gates") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(gates, dict)
        self.assertIn("y_axis", gates)
        self.assertNotIn("yAxis_offset_abs", gates)

    def test_process_model_approach_vector_brick_supply_uses_tight_y_axis_tol(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("APPROACH_VECTOR_BRICK_SUPPLY") if isinstance(steps, dict) else {}
        gates = (step_cfg or {}).get("success_gates") if isinstance(step_cfg, dict) else {}
        y_gate = (gates or {}).get("y_axis") if isinstance(gates, dict) else {}
        self.assertIsInstance(y_gate, dict)
        self.assertEqual(float(y_gate.get("tol")), 3.0)

    def test_process_model_topmost_steps_enable_crosshair_drop_completion(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        for step_name in ("FIND_TOPMOST_BRICK", "FIND_TOPMOST_BRICK_WALL"):
            with self.subTest(step=step_name):
                step_cfg = (steps or {}).get(step_name) if isinstance(steps, dict) else {}
                exception_cfg = (
                    (step_cfg or {}).get("topmost_crosshair_exception")
                    if isinstance(step_cfg, dict)
                    else {}
                )
                self.assertIsInstance(exception_cfg, dict)
                self.assertIs(exception_cfg.get("complete_on_crosshair_drop"), True)


if __name__ == "__main__":
    unittest.main()
