import unittest
from unittest.mock import patch

import helper_x_axis_turn_experiment as experiment
from telemetry_robot import WorldModel


class TestHelperXAxisTurnExperiment(unittest.TestCase):
    def test_parse_observe_input_accepts_direction_duration_target(self):
        parsed, err = experiment.parse_observe_while_moving_trial_input("R 1300 -25.5")

        self.assertIsNone(err)
        self.assertEqual(parsed["direction"], "r")
        self.assertEqual(parsed["duration_ms"], 1300)
        self.assertEqual(parsed["target_x_axis_mm"], -25.5)

    def test_parse_observe_input_rejects_duration_over_hard_max(self):
        parsed, err = experiment.parse_observe_while_moving_trial_input("L 5001 0")

        self.assertIsNone(parsed)
        self.assertIn("hard max of 5000 ms", err)

    def test_run_observe_trial_reports_result_x_and_percent_off_target(self):
        world = WorldModel()

        class FakeClock:
            def __init__(self):
                self.now = 0.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))

        class FakeRobot:
            def stop(self):
                return None

        class FakeVision:
            def __init__(self):
                self.rows = [
                    (True, 0.0, 100.0, -30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -20.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -15.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -15.0, 80.0, 2.0, False, False),
                ]
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        send_durations = []
        log_lines = []

        def _fake_send_robot_command(*args, **kwargs):
            duration_ms = int(kwargs.get("duration_override_ms") or 0)
            send_durations.append(duration_ms)
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": duration_ms,
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=FakeRobot(),
                world=world,
                vision=FakeVision(),
                direction="r",
                duration_ms=260,
                target_x_axis_mm=10.0,
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_x_axis_mm"], 15.0)
        self.assertEqual(result["result_target_error_mm"], 5.0)
        self.assertEqual(result["percent_off_target"], 50.0)
        self.assertEqual(send_durations, [260])
        self.assertEqual(result["continuous_duration_ms"], 260)
        self.assertEqual(result["first_visible_elapsed_ms"], 0)
        self.assertIn("[OBSERVE] 0ms: Visible; xaxis: 30.0", log_lines)
        self.assertIn("[OBSERVE] 250ms: Visible; xaxis: 20.0", log_lines)

    def test_run_observe_trial_reports_never_visible(self):
        world = WorldModel()

        class FakeClock:
            def __init__(self):
                self.now = 0.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))

        class FakeRobot:
            def stop(self):
                return None

        class FakeVision:
            def read(self):
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        def _fake_send_robot_command(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        fake_clock = FakeClock()
        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=FakeRobot(),
                world=world,
                vision=FakeVision(),
                direction="l",
                duration_ms=130,
                target_x_axis_mm=0.0,
                sample_hz=5.0,
                log_path=None,
            )

        self.assertEqual(result["status"], "never_visible")
        self.assertIsNone(result["result_x_axis_mm"])


if __name__ == "__main__":
    unittest.main()
