import threading
import unittest
from unittest.mock import patch

import helper_x_axis_turn_experiment as experiment
from telemetry_robot import WorldModel


class TestHelperXAxisTurnExperiment(unittest.TestCase):
    def test_parse_observe_input_accepts_direction_only(self):
        parsed, err = experiment.parse_observe_while_moving_trial_input("R")

        self.assertIsNone(err)
        self.assertEqual(parsed["direction"], "r")
        self.assertEqual(parsed["max_act_duration_ms"], experiment.MAX_DURATION_MS)
        self.assertEqual(parsed["target_x_axis_mm"], 0.0)

    def test_parse_observe_input_rejects_extra_params(self):
        parsed, err = experiment.parse_observe_while_moving_trial_input("L 5000 0")

        self.assertIsNone(parsed)
        self.assertIn("exactly 1 param", err)

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

        def _fake_send_robot_command_pwm(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        fake_clock = FakeClock()
        log_lines = []

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "l", "pwm": 120, "power": 0.4, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=FakeRobot(),
                world=world,
                vision=FakeVision(),
                direction="l",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
                max_acts=2,
            )

        self.assertEqual(result["status"], "never_visible")
        self.assertIsNone(result["result_x_axis_mm"])
        self.assertEqual(result["start_pose"]["offset_x"], None)
        self.assertEqual([row["duration_requested_ms"] for row in result["send_results"]], [8000, 8000])
        self.assertEqual([row["pwm"] for row in result["acts"]], [120, 122])
        self.assertTrue(any("Act 1: L" in line and "max=8000ms." in line for line in log_lines))

    def test_run_observe_trial_stops_when_target_band_is_reached(self):
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
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                return None

        class FakeVision:
            def __init__(self):
                self.rows = [
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, 10.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, 0.2, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, 0.2, 80.0, 2.0, False, False),
                ]
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        fake_robot = FakeRobot()
        log_lines = []
        sent_durations = []

        def _fake_send_robot_command_pwm(*args, **kwargs):
            sent_durations.append(int(args[6]))
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=fake_robot,
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "target_band_reached")
        self.assertEqual(result["band_hit_x_axis_mm"], -0.2)
        self.assertEqual(result["band_hit_target_error_mm"], 0.2)
        self.assertEqual(result["result_x_axis_mm"], -0.2)
        self.assertEqual(result["result_target_error_mm"], 0.2)
        self.assertEqual(result["band_hit_elapsed_ms"], 550)
        self.assertEqual(result["direction_reversals"], 0)
        self.assertEqual(sent_durations, [1200])
        self.assertTrue(all(duration <= experiment.MAX_DURATION_MS for duration in sent_durations))
        self.assertGreaterEqual(fake_robot.stop_calls, 1)
        self.assertTrue(any("Target band reached at 550ms" in line for line in log_lines))

    def test_run_observe_trial_reverses_when_it_overshoots_zero(self):
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
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                return None

        class FakeVision:
            def __init__(self):
                self.rows = [
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -5.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -5.0, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -0.1, 80.0, 2.0, False, False),
                    (True, 0.0, 100.0, -0.1, 80.0, 2.0, False, False),
                ]
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        fake_robot = FakeRobot()
        log_lines = []
        sent_cmds = []

        def _fake_send_robot_command_pwm(*args, **kwargs):
            sent_cmds.append(str(args[3]).lower())
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        def _fake_profile(cmd):
            if str(cmd).lower() == "r":
                return {"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130}
            return {"cmd": "l", "pwm": 127, "power": 0.416, "duration_ms": 130}

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            side_effect=_fake_profile,
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=fake_robot,
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "target_band_reached")
        self.assertEqual(result["direction_reversals"], 1)
        self.assertEqual(sent_cmds[:2], ["r", "l"])
        self.assertTrue(any("Overshoot detected" in line for line in log_lines))

    def test_run_observe_trial_logs_pwm_from_and_to_when_stalled(self):
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
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                ] * 32
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        log_lines = []

        def _fake_send_robot_command_pwm(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=FakeRobot(),
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
                max_acts=1,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stall_recoveries"], 1)
        self.assertTrue(
            any("Increasing R PWM from 124 to 126 for the next act." in line for line in log_lines)
        )

    def test_run_observe_trial_keeps_movement_active_after_visibility_loss(self):
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
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                return None

        class FakeVision:
            def __init__(self):
                self.rows = [
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                    (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
                    (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
                ] + [(False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)] * 64
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        fake_robot = FakeRobot()
        log_lines = []
        sent_cmds = []

        def _fake_send_robot_command_pwm(*args, **kwargs):
            sent_cmds.append(str(args[3]).lower())
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=fake_robot,
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
                max_acts=2,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "max_acts_reached")
        self.assertEqual(result["visibility_recoveries"], 2)
        self.assertEqual(sent_cmds[:2], ["r", "r"])
        self.assertEqual([row["duration_requested_ms"] for row in result["send_results"][:2]], [1200, 8000])
        self.assertEqual([row["pwm"] for row in result["acts"][:2]], [124, 126])
        self.assertEqual(fake_robot.stop_calls, 1)
        self.assertTrue(
            any(
                "Keeping movement active and increasing R PWM from 124 to 126 for the next act."
                in line
                for line in log_lines
            )
        )
        self.assertFalse(any("Pausing movement" in line for line in log_lines))

    def test_run_observe_trial_blind_crawl_keeps_ramping_pwm_up_slowly(self):
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
                self.rows = [(False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)] * 64
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        fake_robot = FakeRobot()
        log_lines = []

        def _fake_send_robot_command_pwm(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=fake_robot,
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
                max_acts=3,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "never_visible")
        self.assertEqual(result["visibility_recoveries"], 3)
        self.assertEqual([row["pwm"] for row in result["acts"]], [122, 124, 126])
        self.assertTrue(
            any(
                "Keeping movement active and increasing R PWM from 124 to 126 for the next act."
                in line
                for line in log_lines
            )
        )

    def test_run_observe_trial_honors_cancel_event(self):
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
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                return None

        class FakeVision:
            def __init__(self):
                self.rows = [
                    (True, 0.0, 100.0, 30.0, 80.0, 2.0, False, False),
                ] * 64
                self.idx = 0

            def read(self):
                row = self.rows[min(self.idx, len(self.rows) - 1)]
                self.idx += 1
                return row

        fake_clock = FakeClock()
        fake_robot = FakeRobot()
        cancel_event = threading.Event()
        log_lines = []
        cancel_state = {"calls": 0}

        def _fake_send_robot_command_pwm(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(args[6]),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        def _cancel_check():
            cancel_state["calls"] += 1
            if int(cancel_state["calls"]) >= 4:
                cancel_event.set()
            return False

        with patch.object(experiment.time, "monotonic", side_effect=fake_clock.monotonic), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), patch.object(
            experiment,
            "send_robot_command_pwm",
            side_effect=_fake_send_robot_command_pwm,
        ), patch.object(
            experiment,
            "_turn_profile_for_cmd",
            return_value={"cmd": "r", "pwm": 122, "power": 0.393, "duration_ms": 130},
        ):
            result = experiment.run_observe_while_moving_trial(
                robot=fake_robot,
                world=world,
                vision=FakeVision(),
                direction="r",
                sample_hz=5.0,
                log_path=None,
                log_fn=log_lines.append,
                cancel_event=cancel_event,
                cancel_check=_cancel_check,
                max_acts=3,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(result["send_results"]), 1)
        self.assertGreaterEqual(fake_robot.stop_calls, 1)
        self.assertEqual(result["acts"][0]["result"], "cancelled")


if __name__ == "__main__":
    unittest.main()
