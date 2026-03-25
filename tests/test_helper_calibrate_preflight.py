import unittest
from unittest.mock import patch

from calibration import helper_calibrate


class _VisionSequence:
    def __init__(self, values, *, metric="cam_h"):
        self._values = list(values)
        self._metric = str(metric)

    def read(self):
        if not self._values:
            value = 0.0
        else:
            value = self._values.pop(0)
        dist = 100.0
        offset_x = 0.0
        cam_h = 0.0
        if self._metric == "dist":
            dist = float(value)
        elif self._metric == "x_axis":
            offset_x = float(value)
        else:
            cam_h = float(value)
        return True, 0.0, dist, offset_x, 90.0, cam_h, 0.0, 0.0


class _WorldStub:
    def update_vision(self, *_args, **_kwargs):
        return None


class Check1PctSpeedMovementTests(unittest.TestCase):
    def test_preflight_escalates_scores_and_keeps_fixed_duration(self):
        vision = _VisionSequence(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.2,
                10.25,
                10.3,
            ]
        )
        world = _WorldStub()
        sent = []

        def _fake_speed_power_pwm_for_cmd(cmd, score):
            return 0.1 * float(score), 30 + int(score), int(score), 300

        def _fake_send_robot_command_pwm(_robot, _world, _step, cmd, power, pwm, duration_ms, **kwargs):
            sent.append(
                {
                    "cmd": cmd,
                    "power": power,
                    "pwm": pwm,
                    "duration_ms": duration_ms,
                    "speed_score": kwargs.get("speed_score"),
                }
            )

        with patch("telemetry_robot.speed_power_pwm_for_cmd", side_effect=_fake_speed_power_pwm_for_cmd), patch(
            "telemetry_process.send_robot_command_pwm", side_effect=_fake_send_robot_command_pwm
        ), patch.object(helper_calibrate.time, "sleep"):
            result = helper_calibrate.check_1pct_speed_movement(
                robot=object(),
                vision=vision,
                world=world,
                cmd="u",
                movement_threshold_mm=0.15,
                sample_frames=3,
                sample_timeout_s=1.5,
                observe_sleep_s=0.0,
                control_sleep_s=0.0,
                score_candidates=[1, 2],
                duration_override_ms=250,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["score_used"], 2)
        self.assertEqual(result["duration_ms"], 250)
        self.assertEqual(result["attempt_idx"], 2)
        self.assertEqual([item["speed_score"] for item in sent], [1, 2])
        self.assertEqual([item["duration_ms"] for item in sent], [250, 250])

    def test_preflight_returns_none_after_exhausting_candidates(self):
        vision = _VisionSequence(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ]
        )
        world = _WorldStub()
        sent_scores = []

        def _fake_speed_power_pwm_for_cmd(cmd, score):
            return 0.1 * float(score), 30 + int(score), int(score), 300

        def _fake_send_robot_command_pwm(_robot, _world, _step, _cmd, _power, _pwm, _duration_ms, **kwargs):
            sent_scores.append(int(kwargs.get("speed_score") or 0))

        with patch("telemetry_robot.speed_power_pwm_for_cmd", side_effect=_fake_speed_power_pwm_for_cmd), patch(
            "telemetry_process.send_robot_command_pwm", side_effect=_fake_send_robot_command_pwm
        ), patch.object(helper_calibrate.time, "sleep"):
            result = helper_calibrate.check_1pct_speed_movement(
                robot=object(),
                vision=vision,
                world=world,
                cmd="u",
                movement_threshold_mm=0.15,
                sample_frames=3,
                sample_timeout_s=1.5,
                observe_sleep_s=0.0,
                control_sleep_s=0.0,
                score_candidates=[1, 2],
                duration_override_ms=250,
            )

        self.assertIsNone(result)
        self.assertEqual(sent_scores, [1, 2])


if __name__ == "__main__":
    unittest.main()