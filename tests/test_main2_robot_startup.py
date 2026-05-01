import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main2


class _ConnectedRobot:
    def __init__(self, *, exit_on_failure=True, serial_port=None):
        self.exit_on_failure = exit_on_failure
        self.serial_port = serial_port
        self.SERIAL_PORT = serial_port or "/dev/fake"


class TestMain2RobotStartup(unittest.TestCase):
    def test_robot_connection_is_enabled_by_default(self):
        args = main2.parse_args([])

        self.assertTrue(args.robot)

    def test_no_robot_flag_disables_robot_connection(self):
        args = main2.parse_args(["--no-robot"])

        self.assertFalse(args.robot)

    def test_robot_startup_uses_nonfatal_serial_mode(self):
        robot, status = main2._open_robot_or_none(
            enabled=True,
            require_robot=False,
            serial_port="/dev/ttyTEST0",
            robot_factory=_ConnectedRobot,
        )

        self.assertIsInstance(robot, _ConnectedRobot)
        self.assertFalse(robot.exit_on_failure)
        self.assertEqual(robot.serial_port, "/dev/ttyTEST0")
        self.assertIn("Robot connected", status)
        self.assertIn("/dev/ttyTEST0", status)

    def test_missing_robot_degrades_to_camera_only_by_default(self):
        def _missing_robot(*, exit_on_failure=True, serial_port=None):
            raise RuntimeError("no serial here")

        robot, status = main2._open_robot_or_none(
            enabled=True,
            require_robot=False,
            robot_factory=_missing_robot,
        )

        self.assertIsNone(robot)
        self.assertIn("continuing camera-only", status)
        self.assertIn("no serial here", status)

    def test_require_robot_keeps_failure_fatal(self):
        def _missing_robot(*, exit_on_failure=True, serial_port=None):
            raise RuntimeError("no serial here")

        with self.assertRaisesRegex(RuntimeError, "required but unavailable"):
            main2._open_robot_or_none(
                enabled=True,
                require_robot=True,
                robot_factory=_missing_robot,
            )


if __name__ == "__main__":
    unittest.main()
