import sys
import tempfile
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
            "wire_text": f"l.f.50.{int(duration_ms or 0)},r.f.50.{int(duration_ms or 0)}",
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
        baseline = helper_speed_test.build_speed_test_sequence(
            floor_pwm_scale=1.0,
            ceiling_pwm_scale=1.0,
        )
        sequence = helper_speed_test.build_speed_test_sequence()

        self.assertEqual(sequence[0].score, 1)
        self.assertEqual(sequence[0].model_pwm, baseline[0].model_pwm)
        self.assertGreater(sequence[-1].model_pwm, baseline[-1].model_pwm)

    def test_run_execute_sends_no_intermediate_stops(self):
        robot = _FakeRobot()
        clock = _FakeClock()
        sequence = helper_speed_test.build_speed_test_sequence(phase_duration_s=0.3, interval_ms=150)

        result = helper_speed_test.run_speed_test(
            robot=robot,
            execute=True,
            sequence=sequence,
            vision_sampler=lambda: {
                "found": True,
                "dist_mm": 211.5,
                "x_mm": -4.0,
                "conf": 91.0,
                "status": "unit",
                "source": "fake",
            },
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
            log_fn=None,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(robot.stops, 2)
        self.assertEqual(len(robot.calls), len(sequence))
        self.assertEqual([call[0] for call in robot.calls], ["f", "f", "b", "b"])
        self.assertEqual(result["pulses"][0]["t_ms"], 0)
        self.assertEqual([pulse["t_ms"] for pulse in result["pulses"]], [0, 150, 300, 450])

    def test_write_artifacts_includes_left_and_right_pwm_chart_points(self):
        robot = _FakeRobot()
        clock = _FakeClock()
        sequence = helper_speed_test.build_speed_test_sequence(phase_duration_s=0.15, interval_ms=150)
        result = helper_speed_test.run_speed_test(
            robot=robot,
            execute=True,
            sequence=sequence,
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
            vision_sampler=lambda: {
                "found": True,
                "dist_mm": 211.5,
                "x_mm": -4.0,
                "conf": 91.0,
                "status": "unit",
                "source": "fake",
            },
            log_fn=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = helper_speed_test.write_speed_test_artifacts(
                result,
                log_dir=tmpdir,
                run_label="unit",
            )

            self.assertTrue(Path(artifacts["json_path"]).exists())
            self.assertTrue(Path(artifacts["html_path"]).exists())
            self.assertEqual(len(artifacts["points"]), len(sequence))
            self.assertIn("left_pwm", artifacts["points"][0])
            self.assertIn("right_pwm", artifacts["points"][0])
            self.assertEqual(artifacts["points"][0]["left_pwm"], 127)
            self.assertEqual(artifacts["points"][0]["right_pwm"], 127)
            self.assertEqual(artifacts["points"][0]["dist_mm"], 211.5)
            self.assertEqual(artifacts["points"][0]["x_mm"], -4.0)
            self.assertTrue(artifacts["points"][0]["brick_found"])

    def test_chart_points_show_zero_for_stopped_wheel_action(self):
        result = {
            "ok": True,
            "execute": True,
            "pulses": [
                {
                    "phase_name": "unit",
                    "cmd": "f",
                    "t_ms": 0,
                    "score": 1,
                    "model_pwm": 160,
                    "interval_ms": 150,
                    "send_result": {
                        "pwm": 160,
                        "wire_text": "l.s,r.f.63.150",
                        "actions": [
                            {"target": "l", "action": "s", "pwm": 0, "duration_ms": 150},
                            {"target": "r", "action": "f", "pwm": 160, "duration_ms": 150},
                        ],
                    },
                }
            ],
        }

        points = helper_speed_test.speed_test_chart_points(result)

        self.assertEqual(points[0]["left_pwm"], 0)
        self.assertEqual(points[0]["right_pwm"], 160)


if __name__ == "__main__":
    unittest.main()
