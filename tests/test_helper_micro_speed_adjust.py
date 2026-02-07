import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot
from helper_micro_speed_adjust import micro_adjust_speed_score


class TestHelperMicroSpeedAdjust(unittest.TestCase):
    def _model(self, pwm):
        return {
            1: {
                "pwm": int(pwm),
                "power": float(telemetry_robot._pwm_to_power(int(pwm)) or 0.0),
                "duration_ms": 500,
            }
        }

    def test_micro_adjust_increases_when_delta_below_threshold(self):
        model = self._model(100)
        state = {}
        msg = None
        for value in (0.0, 0.1, 0.2, 0.3):
            msg = micro_adjust_speed_score(
                state,
                score_power_pwm=model,
                metric_value_mm=value,
                active=True,
                acts=3,
                threshold_mm=0.5,
                increase_scale=1.1,
                decrease_scale=0.9,
            )
        self.assertIsNotNone(msg)
        self.assertEqual(model[1]["pwm"], 110)

    def test_micro_adjust_decreases_when_delta_above_threshold(self):
        model = self._model(100)
        state = {}
        msg = None
        for value in (0.0, 0.3, 0.6, 0.9):
            msg = micro_adjust_speed_score(
                state,
                score_power_pwm=model,
                metric_value_mm=value,
                active=True,
                acts=3,
                threshold_mm=0.5,
                increase_scale=1.1,
                decrease_scale=0.9,
            )
        self.assertIsNotNone(msg)
        self.assertEqual(model[1]["pwm"], 90)

    def test_micro_adjust_honors_min_pwm_clamp(self):
        model = self._model(100)
        state = {}
        msg = None
        for value in (0.0, 0.3, 0.6, 0.9):
            msg = micro_adjust_speed_score(
                state,
                score_power_pwm=model,
                metric_value_mm=value,
                active=True,
                acts=3,
                threshold_mm=0.5,
                increase_scale=1.1,
                decrease_scale=0.9,
                min_pwm=95,
            )
        self.assertIsNotNone(msg)
        self.assertEqual(model[1]["pwm"], 95)


if __name__ == "__main__":
    unittest.main()

