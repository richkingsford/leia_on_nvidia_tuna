import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


def _success_log_rows(step, states):
    rows = [
        {
            "type": "keyframe",
            "timestamp": 1.0,
            "step": step,
            "marker": "SUCCESS_START",
        }
    ]
    rows.extend(states)
    rows.append(
        {
            "type": "keyframe",
            "timestamp": 2.0,
            "step": step,
            "marker": "SUCCESS_END",
        }
    )
    return rows


class TestTelemetryProcessGateDerivationRules(unittest.TestCase):
    def test_success_gate_metrics_for_exit_wall_are_visible_only(self):
        metrics = ["angle_abs", "xAxis_offset_abs", "dist", "visible"]
        filtered = telemetry_process.success_gate_metrics_for_step(metrics, "EXIT_WALL", step_rules={})
        self.assertEqual(filtered, ["visible"])

    def test_derive_success_gates_for_exit_wall_uses_demo_visible_only(self):
        success_segments = {
            "EXIT_WALL": [
                {
                    "states": [
                        {"timestamp": 1.0, "brick": {"visible": False, "angle": 91.0, "x_axis": -121.0, "dist": 229.0}},
                        {"timestamp": 1.1, "brick": {"visible": False, "angle": 89.0, "x_axis": -120.0, "dist": 228.0}},
                        {"timestamp": 1.2, "brick": {"visible": True, "angle": 90.0, "x_axis": -122.0, "dist": 230.0}},
                    ]
                }
            ]
        }
        gates = telemetry_process.derive_success_gates(success_segments, scale_by_step={}, step_rules={})
        self.assertIn("EXIT_WALL", gates)
        self.assertEqual(set(gates["EXIT_WALL"].keys()), {"visible"})
        self.assertIs(gates["EXIT_WALL"]["visible"].get("min"), False)

    def test_update_process_model_rewrites_exit_wall_success_gates_to_visible_only(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "world_model_process.json"
            model_path.write_text(
                json.dumps(
                    {
                        "steps": {
                            "EXIT_WALL": {
                                "lock_success_gates": True,
                                "success_gates": {
                                    "visible": {"min": True},
                                    "angle_abs": {"target": 90.0, "tol": 0.0},
                                    "xAxis_offset_abs": {"target": -120.3, "tol": 1.4},
                                    "dist": {"target": 228.2, "tol": 1.5},
                                },
                            }
                        }
                    }
                )
            )
            logs = [
                (
                    Path("exit_wall_demo.json"),
                    _success_log_rows(
                        "EXIT_WALL",
                        [
                            {
                                "type": "state",
                                "timestamp": 1.1,
                                "brick": {"visible": True, "angle": 99.0, "x_axis": -135.0, "dist": 250.0},
                            },
                            {
                                "type": "state",
                                "timestamp": 1.2,
                                "brick": {"visible": True, "angle": 84.0, "x_axis": -115.0, "dist": 225.0},
                            },
                        ],
                    ),
                )
            ]
            updated = telemetry_process.update_process_model_from_demos(logs, path=model_path)
            gates = ((updated.get("steps") or {}).get("EXIT_WALL") or {}).get("success_gates") or {}
            self.assertEqual(set(gates.keys()), {"visible"})
            self.assertIs(gates.get("visible", {}).get("min"), True)

    def test_update_process_model_refreshes_unlocked_step_gates_from_demo_logs(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "world_model_process.json"
            model_path.write_text(
                json.dumps(
                    {
                        "steps": {
                            "ALIGN_BRICK": {
                                "start_gates": {"visible": {"min": False}},
                                "success_gates": {
                                    "visible": {"min": True},
                                    "xAxis_offset_abs": {"target": 500.0, "tol": 1.4},
                                    "yAxis_offset_abs": {"target": 500.0, "tol": 1.5},
                                    "dist": {"target": 999.0, "tol": 1.5},
                                },
                            }
                        }
                    }
                )
            )
            logs = [
                (
                    Path("align_demo.json"),
                    _success_log_rows(
                        "ALIGN_BRICK",
                        [
                            {
                                "type": "state",
                                "timestamp": 1.1,
                                "brick": {"visible": True, "x_axis": -3.0, "y_axis": 2.0, "dist": 130.0},
                            },
                            {
                                "type": "state",
                                "timestamp": 1.2,
                                "brick": {"visible": True, "x_axis": -2.0, "y_axis": 3.0, "dist": 132.0},
                            },
                            {
                                "type": "state",
                                "timestamp": 1.3,
                                "brick": {"visible": True, "x_axis": -1.0, "y_axis": 4.0, "dist": 134.0},
                            },
                        ],
                    ),
                )
            ]
            updated = telemetry_process.update_process_model_from_demos(logs, path=model_path)
            align_cfg = (updated.get("steps") or {}).get("ALIGN_BRICK") or {}
            self.assertEqual(align_cfg.get("start_gates"), {"visible": {"min": True}})
            gates = align_cfg.get("success_gates") or {}
            self.assertEqual(
                set(gates.keys()),
                {"visible", "xAxis_offset_abs", "yAxis_offset_abs", "dist"},
            )
            self.assertNotEqual((gates.get("dist") or {}).get("target"), 999.0)

    def test_update_process_model_uses_active_mode_locked_success_gates(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "world_model_process.json"
            model_path.write_text(
                json.dumps(
                    {
                        "steps": {
                            "BRICK_LOCK": {
                                "lock_success_gates": True,
                                "locked_success_gates_by_mode": {
                                    "aruco": {
                                        "xAxis_offset_abs": {"target": -6.22, "tol": 4.0},
                                        "dist": {"target": 154.61, "tol": 4.0},
                                    },
                                    "cyan": {
                                        "xAxis_offset_abs": {"target": -3.2, "tol": 4.0},
                                        "dist": {"target": 97.15, "tol": 4.0},
                                    },
                                },
                                "success_gates": {
                                    "xAxis_offset_abs": {"target": -999.0, "tol": 1.0},
                                    "dist": {"target": 999.0, "tol": 1.0},
                                },
                            }
                        }
                    }
                )
            )
            with mock.patch.object(
                telemetry_process,
                "_world_model_active_vision_mode",
                return_value="cyan",
            ):
                updated = telemetry_process.update_process_model_from_demos([], path=model_path)

            cfg = ((updated.get("steps") or {}).get("BRICK_LOCK") or {})
            gates = cfg.get("success_gates") or {}
            self.assertEqual((gates.get("xAxis_offset_abs") or {}).get("target"), -3.2)
            self.assertEqual((gates.get("dist") or {}).get("target"), 97.15)

    def test_update_process_model_fills_missing_active_mode_locked_success_gates_from_demos(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "world_model_process.json"
            model_path.write_text(
                json.dumps(
                    {
                        "steps": {
                            "BRICK_LOCK": {
                                "lock_success_gates": True,
                                "locked_success_gates_by_mode": {
                                    "aruco": {
                                        "xAxis_offset_abs": {"target": -6.22, "tol": 4.0},
                                        "dist": {"target": 154.61, "tol": 4.0},
                                    }
                                },
                            }
                        }
                    }
                )
            )
            logs = [
                (
                    Path("brick_lock_demo.json"),
                    _success_log_rows(
                        "BRICK_LOCK",
                        [
                            {
                                "type": "state",
                                "timestamp": 1.1,
                                "brick": {"visible": True, "x_axis": -3.0, "dist": 98.0},
                            },
                            {
                                "type": "state",
                                "timestamp": 1.2,
                                "brick": {"visible": True, "x_axis": -3.2, "dist": 97.0},
                            },
                            {
                                "type": "state",
                                "timestamp": 1.3,
                                "brick": {"visible": True, "x_axis": -3.4, "dist": 96.0},
                            },
                        ],
                    ),
                )
            ]
            with mock.patch.object(
                telemetry_process,
                "_world_model_active_vision_mode",
                return_value="cyan",
            ):
                updated = telemetry_process.update_process_model_from_demos(logs, path=model_path)

            cfg = ((updated.get("steps") or {}).get("BRICK_LOCK") or {})
            by_mode = cfg.get("locked_success_gates_by_mode") or {}
            cyan_gates = by_mode.get("cyan") or {}
            aruco_gates = by_mode.get("aruco") or {}
            self.assertTrue(cyan_gates)
            self.assertEqual((aruco_gates.get("dist") or {}).get("target"), 154.61)
            self.assertEqual(cfg.get("success_gates"), cyan_gates)


if __name__ == "__main__":
    unittest.main()
