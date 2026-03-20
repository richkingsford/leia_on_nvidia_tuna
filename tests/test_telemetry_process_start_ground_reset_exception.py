import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self, process_rules, *, brick_supply_height_bricks=None):
        self.process_rules = process_rules
        self.brick_supply_height_bricks = brick_supply_height_bricks
        self.brick_supply_height_mm = None
        self.wall_height_bricks = 1
        self.wall_height_mm = 44.0
        self.lift_height = None
        self.lift_height_source = None


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class TestTelemetryProcessStartGroundResetException(unittest.TestCase):
    def test_start_ground_reset_uses_known_supply_height_as_total_acts(self):
        world = _DummyWorld(
            {
                "FIND_WALL2": {
                    "start_ground_reset_exception": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "min_acts": 2,
                        "max_acts": 12,
                        "max_acts_from_height_source": "brick_supply_height",
                        "max_acts_default": 5,
                        "observe_between_acts": False,
                    }
                }
            },
            brick_supply_height_bricks=7,
        )
        robot = _DummyRobot()
        sent_cmds = []

        orig_pause = telemetry_process.pause_after_exception
        orig_send = telemetry_process.send_robot_command
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.pause_after_exception = lambda *_a, **_k: None
            telemetry_process.send_robot_command = (
                lambda *_args, **_kwargs: sent_cmds.append(str(_args[3])) or {}
            )
            telemetry_process.time.sleep = lambda *_a, **_k: None

            result = telemetry_process._run_start_ground_reset_exception(
                world,
                vision=object(),
                step="FIND_WALL2",
                robot=robot,
                align_silent=True,
            )
        finally:
            telemetry_process.pause_after_exception = orig_pause
            telemetry_process.send_robot_command = orig_send
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason") or ""), "ground reset max acts (no observation)")
        self.assertEqual(int(result.get("acts") or 0), 7)
        self.assertEqual(sent_cmds, ["d"] * 7)
        self.assertEqual(robot.stop_calls, 1)

    def test_start_ground_reset_defaults_to_five_acts_when_supply_height_unknown(self):
        world = _DummyWorld(
            {
                "FIND_WALL2": {
                    "start_ground_reset_exception": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "min_acts": 2,
                        "max_acts": 12,
                        "max_acts_from_height_source": "brick_supply_height",
                        "max_acts_default": 5,
                        "observe_between_acts": False,
                    }
                }
            },
            brick_supply_height_bricks=None,
        )
        robot = _DummyRobot()
        sent_cmds = []

        orig_pause = telemetry_process.pause_after_exception
        orig_send = telemetry_process.send_robot_command
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.pause_after_exception = lambda *_a, **_k: None
            telemetry_process.send_robot_command = (
                lambda *_args, **_kwargs: sent_cmds.append(str(_args[3])) or {}
            )
            telemetry_process.time.sleep = lambda *_a, **_k: None

            result = telemetry_process._run_start_ground_reset_exception(
                world,
                vision=object(),
                step="FIND_WALL2",
                robot=robot,
                align_silent=True,
            )
        finally:
            telemetry_process.pause_after_exception = orig_pause
            telemetry_process.send_robot_command = orig_send
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(str(result.get("reason") or ""), "ground reset max acts (no observation)")
        self.assertEqual(int(result.get("acts") or 0), 5)
        self.assertEqual(sent_cmds, ["d"] * 5)
        self.assertEqual(robot.stop_calls, 1)

    def test_start_ground_reset_waits_for_each_sent_pulse_duration(self):
        world = _DummyWorld(
            {
                "FIND_WALL2": {
                    "start_ground_reset_exception": {
                        "enabled": True,
                        "command": "d",
                        "score": 100,
                        "min_acts": 1,
                        "max_acts": 3,
                        "observe_between_acts": False,
                    }
                }
            },
            brick_supply_height_bricks=3,
        )
        robot = _DummyRobot()
        sleep_calls = []

        orig_pause = telemetry_process.pause_after_exception
        orig_send = telemetry_process.send_robot_command
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.pause_after_exception = lambda *_a, **_k: None
            telemetry_process.send_robot_command = (
                lambda *_args, **_kwargs: {"duration_ms": 360, "cmd_sent": "d", "pwm": 255}
            )
            telemetry_process.time.sleep = lambda seconds=0.0: sleep_calls.append(float(seconds))

            result = telemetry_process._run_start_ground_reset_exception(
                world,
                vision=object(),
                step="FIND_WALL2",
                robot=robot,
                align_silent=True,
            )
        finally:
            telemetry_process.pause_after_exception = orig_pause
            telemetry_process.send_robot_command = orig_send
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(bool(result.get("enabled")))
        self.assertTrue(bool(result.get("success")))
        self.assertEqual(int(result.get("acts") or 0), 3)
        self.assertEqual(sleep_calls, [0.36, 0.36, 0.36])
        self.assertEqual(robot.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
