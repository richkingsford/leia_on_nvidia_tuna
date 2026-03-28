import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestWorldModelRobotHotkeySource(unittest.TestCase):
    def test_world_model_robot_has_no_command_remap(self):
        model_path = Path(__file__).resolve().parents[1] / "world_model_robot.json"
        model = json.loads(model_path.read_text())

        self.assertNotIn(
            "command_remap",
            model,
            "Production hotkeys must not be globally inverted in world_model_robot.json.",
        )

    def test_default_loaded_hotkeys_have_no_runtime_remap(self):
        loaded = telemetry_robot._load_speed_model(telemetry_robot.ROBOT_MODEL_FILE)
        hotkeys = loaded[0]
        cmd_remap = loaded[15]

        self.assertEqual(
            cmd_remap,
            {},
            "Repo defaults should not rewrite logical hotkey commands at the wire layer.",
        )
        self.assertEqual(hotkeys["w"]["cmd"], "f")
        self.assertEqual(hotkeys["s"]["cmd"], "b")
        self.assertEqual(hotkeys["q"]["cmd"], "l")
        self.assertEqual(hotkeys["e"]["cmd"], "r")
        self.assertEqual(hotkeys["o"]["cmd"], "u")
        self.assertEqual(hotkeys["k"]["cmd"], "d")

    def test_world_model_robot_exposes_single_wire_command_map(self):
        self.assertEqual(
            telemetry_robot.ROBOT_WIRE_COMMAND_MAP,
            {
                "f": "b",
                "b": "f",
                "l": "r",
                "r": "l",
                "u": "d",
                "d": "u",
            },
        )

    def test_one_percent_discovery_lines_report_shared_floor_values(self):
        lines = telemetry_robot.one_percent_discovery_lines()

        self.assertEqual(len(lines), 4)
        self.assertIn("Forward (R): pwm=103, pwr=0.306, t=255ms", lines[0])
        self.assertIn("Backward (F): pwm=103, pwr=0.306, t=255ms", lines[1])
        self.assertIn("Left (Q): pwm=102, pwr=0.301, t=135ms", lines[2])
        self.assertIn("Right (E): pwm=102, pwr=0.301, t=65ms", lines[3])


if __name__ == "__main__":
    unittest.main()
