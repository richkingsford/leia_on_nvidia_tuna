import unittest

import telemetry_process


class TestTelemetryProcessStepControllerSelection(unittest.TestCase):
    def test_visible_only_success_gates_use_replay(self):
        rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        self.assertFalse(telemetry_process.step_uses_alignment_control("FIND_WALL", rules))

    def test_visible_plus_confidence_success_gates_use_replay(self):
        rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                    "confidence": {"min": 50.0},
                }
            }
        }
        self.assertFalse(telemetry_process.step_uses_alignment_control("FIND_WALL", rules))

    def test_nontrivial_metric_success_gates_use_align(self):
        rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                    "dist": {"target": 100.0, "tol": 5.0},
                }
            }
        }
        self.assertTrue(telemetry_process.step_uses_alignment_control("FIND_WALL", rules))


if __name__ == "__main__":
    unittest.main()
