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

    def test_process_model_find_wall_has_no_start_ground_reset_exception(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL") if isinstance(steps, dict) else {}
        self.assertIsInstance(step_cfg, dict)
        self.assertNotIn("start_ground_reset_exception", step_cfg)

    def test_process_model_find_topmost_brick_uses_visible_and_brick_above_success_gates(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_TOPMOST_BRICK") if isinstance(steps, dict) else {}
        gates = (step_cfg or {}).get("success_gates") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(gates, dict)
        self.assertIn("visible", gates)
        self.assertIn("brick_above", gates)
        self.assertNotIn("inCrosshairs", gates)

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

    def test_process_model_find_wall2_startup_turn_and_ground_reset_config(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL2") if isinstance(steps, dict) else {}
        startup_pre = (step_cfg or {}).get("startup_pre_action_exception") if isinstance(step_cfg, dict) else {}
        ground_reset = (step_cfg or {}).get("start_ground_reset_exception") if isinstance(step_cfg, dict) else {}
        search_cycle = (step_cfg or {}).get("search_visible_false_speed_cycle") if isinstance(step_cfg, dict) else {}

        self.assertIsInstance(startup_pre, dict)
        self.assertEqual(int(startup_pre.get("score") or 0), 1)
        self.assertEqual(int(startup_pre.get("duration_override_ms") or 0), 1500)
        self.assertFalse(bool(startup_pre.get("observe_between_acts", True)))

        self.assertIsInstance(ground_reset, dict)
        self.assertEqual(str(ground_reset.get("max_acts_from_height_source") or ""), "brick_supply_height")
        self.assertEqual(int(ground_reset.get("max_acts_default") or 0), 5)
        self.assertFalse(bool(ground_reset.get("observe_between_acts", True)))

        self.assertIsInstance(search_cycle, dict)
        self.assertEqual(int((search_cycle.get("command_scores") or {}).get("l") or 0), 20)
        self.assertEqual(int((search_cycle.get("command_scores") or {}).get("b") or 0), 10)

    def test_process_model_find_wall2_visible_tiers_use_fixed_two_percent_with_duration_override(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL2") if isinstance(steps, dict) else {}
        visible_tiers = (step_cfg or {}).get("visible_only_speed_tiers") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(visible_tiers, dict)
        self.assertEqual(int(visible_tiers.get("normal") or 0), 2)
        self.assertEqual(int(visible_tiers.get("standard") or 0), 2)
        self.assertEqual(int(visible_tiers.get("fast") or 0), 2)
        self.assertEqual(int(visible_tiers.get("duration_override_ms") or 0), 1500)

    def test_process_model_seat_brick2_progress_mast_exception_uses_score_seven(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("SEAT_BRICK2") if isinstance(steps, dict) else {}
        progress_cfg = (step_cfg or {}).get("progress_mast_exception") if isinstance(step_cfg, dict) else {}
        self.assertIsInstance(progress_cfg, dict)
        self.assertEqual(int(progress_cfg.get("score") or 0), 7)

    def test_process_model_approach_vector_brick_supply_uses_tight_y_axis_tol(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("APPROACH_VECTOR_BRICK_SUPPLY") if isinstance(steps, dict) else {}
        gates = (step_cfg or {}).get("success_gates") if isinstance(step_cfg, dict) else {}
        y_gate = (gates or {}).get("y_axis") if isinstance(gates, dict) else {}
        self.assertIsInstance(y_gate, dict)
        self.assertEqual(float(y_gate.get("tol")), 3.0)

    def test_process_model_topmost_steps_disable_crosshair_drop_completion(self):
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
                self.assertIs(exception_cfg.get("complete_on_crosshair_drop"), False)

    def test_process_model_find_topmost_brick_enables_bottom_discovery_phase(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_TOPMOST_BRICK") if isinstance(steps, dict) else {}
        exception_cfg = (
            (step_cfg or {}).get("topmost_crosshair_exception")
            if isinstance(step_cfg, dict)
            else {}
        )
        bottom_cfg = (
            (exception_cfg or {}).get("bottom_brick_discovery")
            if isinstance(exception_cfg, dict)
            else {}
        )
        self.assertIsInstance(bottom_cfg, dict)
        self.assertIs(bottom_cfg.get("enabled"), True)
        self.assertEqual(int(bottom_cfg.get("consecutive_no_required")), 1)


if __name__ == "__main__":
    unittest.main()
