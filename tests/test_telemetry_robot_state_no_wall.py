import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from telemetry_robot import WorldModel


class TestTelemetryRobotStateNoWall(unittest.TestCase):
    def test_world_state_dict_excludes_wall(self):
        world = WorldModel()
        payload = world.to_dict()
        self.assertEqual(payload.get("type"), "state")
        self.assertNotIn("wall", payload)


if __name__ == "__main__":
    unittest.main()
