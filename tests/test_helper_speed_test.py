import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_speed_test


class _FakeRobot:
    def __init__(self):
        self.calls = []
        self.stops = 0

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.calls.append((cmd, int(pwm), int(duration_ms or 0)))
        return {
            "cmd_sent": cmd,
            "pwm": int(pwm),
            "power": 0.5,
            "percent": 50,
            "duration_ms": int(duration_ms or 0),
        }

    def stop(self):
        self.stops += 1


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        seconds = max(0.0, float(seconds))
        self.sleeps.append(seconds)
        self.now += seconds


class TestHelperSpeedTest(unittest.TestCase):
    def test_default_sequence_ramps_forward_then_backward(self):
        sequence = helper_speed_test.build_speed_test_sequence()

        self.assertEqual(len(sequence), 28)
        self.assertEqual(sequence[0].cmd, "f")
        self.assertEqual(sequence[0].score, 1)
        self.assertEqual(sequence[13].cmd, "f")
        self.assertEqual(sequence[13].score, 2)
        self.assertEqual(sequence[14].cmd, "b")
        self.assertEqual(sequence[14].score, 1)
        self.assertEqual(sequence[-1].cmd, "b")
        self.assertEqual(sequence[-1].score, 2)

    def test_down_reverse_mode_ramps_forward_back_down(self):
        sequence = helper_speed_test.build_speed_test_sequence(reverse_mode="down")

        self.assertEqual(sequence[14].cmd, "f")
        self.assertEqual(sequence[14].score, 2)
        self.assertEqual(sequence[-1].cmd, "f")
        self.assertEqual(sequence[-1].score, 1)

    def test_table_includes_power_pwm_and_cadence(self):
        table = helper_speed_test.format_speed_test_table(helper_speed_test.build_speed_test_sequence()[:1])

        self.assertIn("power", table)
        self.assertIn("pwm", table)
        self.assertIn("cadence_ms", table)
        self.assertIn("150", table)

    def test_default_start_pwm_floor_is_scaled_effectively(self):
        sequence = helper_speed_test.build_speed_test_sequence()

        self.assertEqual(sequence[0].score, 1)
        self.assertGreaterEqual(sequence[0].model_pwm, 103)
        self.assertGreaterEqual(sequence[-1].model_pwm, 103)

    def test_run_execute_sends_no_intermediate_stops(self):
        robot = _FakeRobot()
        clock = _FakeClock()
        sequence = helper_speed_test.build_speed_test_sequence(phase_duration_s=0.3, interval_ms=150)

        result = helper_speed_test.run_speed_test(
            robot=robot,
            execute=True,
            sequence=sequence,
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
            log_fn=None,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(robot.stops, 2)
        self.assertEqual(len(robot.calls), len(sequence))
        self.assertEqual([call[0] for call in robot.calls], ["f", "f", "b", "b"])


if __name__ == "__main__":
    unittest.main()
