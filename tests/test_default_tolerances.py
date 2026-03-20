import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestDefaultTolerances(unittest.TestCase):
    def test_derive_success_gates_uses_2_3_mm_min_floor(self):
        success_segments = {
            "ALIGN_BRICK": [
                {
                    "states": [
                        {
                            "timestamp": 1.0,
                            "brick": {
                                "visible": True,
                                "x_axis": -3.0,
                                "y_axis": 2.0,
                                "dist": 100.0,
                            },
                        },
                        {
                            "timestamp": 1.1,
                            "brick": {
                                "visible": True,
                                "x_axis": -2.5,
                                "y_axis": 2.4,
                                "dist": 100.8,
                            },
                        },
                    ]
                }
            ]
        }

        gates = telemetry_process.derive_success_gates(
            success_segments,
            scale_by_step={},
            step_rules={},
        )

        align_gates = gates.get("ALIGN_BRICK") or {}
        self.assertEqual((align_gates.get("yAxis_offset_abs") or {}).get("tol"), 2.3)
        self.assertEqual((align_gates.get("dist") or {}).get("tol"), 2.3)

    def test_process_model_default_success_tolerances_are_2_3_mm(self):
        model = json.loads(Path("world_model_process.json").read_text())
        steps = model.get("steps") or {}

        self.assertEqual(
            (((steps.get("BRICK_LOCK") or {}).get("locked_success_gates_by_mode") or {}).get("aruco") or {})
            .get("dist", {})
            .get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("BRICK_LOCK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}).get("yAxis_offset_abs") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("ALIGN_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("SEAT_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("POSITION_BRICK") or {}).get("success_gates") or {}).get("yAxis_offset_abs") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("POSITION_BRICK") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("SEAT_BRICK2") or {}).get("success_gates") or {}).get("x_axis") or {}).get("tol"),
            2.3,
        )
        self.assertEqual(
            (((steps.get("SEAT_BRICK2") or {}).get("success_gates") or {}).get("dist") or {}).get("tol"),
            2.3,
        )
