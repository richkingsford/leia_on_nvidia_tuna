import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessFindWall2Gates(unittest.TestCase):
    def test_derive_success_gates_includes_find_wall2_visible(self):
        success_segments = {
            "FIND_WALL2": [
                {
                    "states": [
                        {"timestamp": 1.0, "brick": {"visible": False}},
                        {"timestamp": 2.0, "brick": {"visible": True}},
                        {"timestamp": 3.0, "brick": {"visible": True}},
                    ]
                }
            ]
        }
        step_rules = {
            "FIND_WALL2": {
                "alignment_metrics": ["visible"],
            }
        }
        gates = telemetry_process.derive_success_gates(
            success_segments,
            scale_by_step={},
            step_rules=step_rules,
        )
        self.assertIn("FIND_WALL2", gates)
        visible_gate = gates["FIND_WALL2"].get("visible") or {}
        self.assertIs(visible_gate.get("min"), True)


if __name__ == "__main__":
    unittest.main()
