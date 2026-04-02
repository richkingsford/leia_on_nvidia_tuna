import copy
import json
import tempfile
import unittest
from pathlib import Path

import telemetry_robot


class TestTelemetryRobotTurnBreakawayScore(unittest.TestCase):
    def setUp(self):
        self.orig_turn_path = telemetry_robot.TURN_BREAKAWAY_TEST_FILE
        self.orig_turn_map = telemetry_robot.SCORE_POWER_PWM_TURN
        self.orig_left_map = telemetry_robot.SCORE_POWER_PWM_TURN_LEFT
        self.orig_right_map = telemetry_robot.SCORE_POWER_PWM_TURN_RIGHT
        self.orig_turn_durations = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN
        self.orig_left_durations = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT
        self.orig_right_durations = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_RIGHT
        self.orig_use_turn_maps = telemetry_robot.USE_DIRECTIONAL_TURN_MAPS
        self.orig_use_turn_durations = telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS
        self.orig_cache = copy.deepcopy(telemetry_robot._BREAKAWAY_TEST_CACHE)

    def tearDown(self):
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self.orig_turn_path
        telemetry_robot.SCORE_POWER_PWM_TURN = self.orig_turn_map
        telemetry_robot.SCORE_POWER_PWM_TURN_LEFT = self.orig_left_map
        telemetry_robot.SCORE_POWER_PWM_TURN_RIGHT = self.orig_right_map
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN = self.orig_turn_durations
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT = self.orig_left_durations
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_RIGHT = self.orig_right_durations
        telemetry_robot.USE_DIRECTIONAL_TURN_MAPS = self.orig_use_turn_maps
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = self.orig_use_turn_durations
        telemetry_robot._BREAKAWAY_TEST_CACHE.clear()
        telemetry_robot._BREAKAWAY_TEST_CACHE.update(self.orig_cache)

    def _write_turn_breakaway_payload(self, recommended_score: int) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        path = Path(handle.name)
        payload = {
            "summary": {
                "by_hotkey": {
                    "q": {
                        "recommended_score": int(recommended_score),
                        "scores": [
                            {
                                "score": int(recommended_score),
                                "trial_count": 5,
                                "success_count": 5,
                                "passes": True,
                            }
                        ],
                    },
                    "e": {
                        "recommended_score": int(recommended_score),
                        "scores": [
                            {
                                "score": int(recommended_score),
                                "trial_count": 5,
                                "success_count": 5,
                                "passes": True,
                            }
                        ],
                    },
                }
            }
        }
        try:
            handle.write(json.dumps(payload))
            handle.close()
        except Exception:
            handle.close()
            raise
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _write_turn_breakaway_pwm_payload(self, *, left_pwm: int, right_pwm: int, duration_ms: int) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        path = Path(handle.name)
        payload = {
            "summary": {
                "by_hotkey": {
                    "q": {
                        "recommended_score": None,
                        "recommended_pwm": int(left_pwm),
                        "recommended_power": float(telemetry_robot.pwm_to_power(left_pwm)),
                        "recommended_duration_ms": int(duration_ms),
                        "ceiling_candidate": {
                            "movement_count": 3,
                            "valid_trial_count": 3,
                        },
                    },
                    "e": {
                        "recommended_score": None,
                        "recommended_pwm": int(right_pwm),
                        "recommended_power": float(telemetry_robot.pwm_to_power(right_pwm)),
                        "recommended_duration_ms": int(duration_ms),
                        "ceiling_candidate": {
                            "movement_count": 2,
                            "valid_trial_count": 2,
                        },
                    },
                }
            }
        }
        try:
            handle.write(json.dumps(payload))
            handle.close()
        except Exception:
            handle.close()
            raise
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _write_turn_breakaway_seed_and_ceiling_payload(
        self,
        *,
        seed_pwm: int,
        left_ceiling_pwm: int,
        right_ceiling_pwm: int,
        duration_ms: int,
    ) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        path = Path(handle.name)
        payload = {
            "summary": {
                "by_hotkey": {
                    "q": {
                        "recommended_score": None,
                        "recommended_pwm": int(seed_pwm),
                        "recommended_power": float(telemetry_robot.pwm_to_power(seed_pwm)),
                        "recommended_duration_ms": int(duration_ms),
                        "ceiling_candidate": {
                            "pwm": int(left_ceiling_pwm),
                            "power": float(telemetry_robot.pwm_to_power(left_ceiling_pwm)),
                            "duration_ms": int(duration_ms),
                            "movement_count": 3,
                            "valid_trial_count": 3,
                        },
                    },
                    "e": {
                        "recommended_score": None,
                        "recommended_pwm": int(seed_pwm),
                        "recommended_power": float(telemetry_robot.pwm_to_power(seed_pwm)),
                        "recommended_duration_ms": int(duration_ms),
                        "ceiling_candidate": {
                            "pwm": int(right_ceiling_pwm),
                            "power": float(telemetry_robot.pwm_to_power(right_ceiling_pwm)),
                            "duration_ms": int(duration_ms),
                            "movement_count": 2,
                            "valid_trial_count": 2,
                        },
                    },
                }
            }
        }
        try:
            handle.write(json.dumps(payload))
            handle.close()
        except Exception:
            handle.close()
            raise
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_speed_power_pwm_for_cmd_uses_turn_breakaway_recommendation_for_score_one(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_MAPS = True
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = True
        telemetry_robot.SCORE_POWER_PWM_TURN_LEFT = {
            1: {"pwm": 102, "power": telemetry_robot.pwm_to_power(102)},
            3: {"pwm": 140, "power": telemetry_robot.pwm_to_power(140)},
            100: {"pwm": 255, "power": telemetry_robot.pwm_to_power(255)},
        }
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT = {1: 135, 3: 320, 100: 300}
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_payload(3)

        power, pwm, score_used, duration_ms = telemetry_robot.speed_power_pwm_for_cmd("l", 1)

        self.assertEqual(int(score_used), 1)
        self.assertEqual(int(pwm), 140)
        self.assertEqual(int(duration_ms), 320)
        self.assertAlmostEqual(float(power), float(telemetry_robot.pwm_to_power(140)), places=6)

    def test_one_percent_discovery_note_reports_turn_breakaway_alias(self):
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_payload(4)

        note = telemetry_robot.one_percent_discovery_note("r", 1)

        self.assertEqual(note, "turn 1%=score 4; breakaway 5/5")

    def test_speed_power_pwm_for_cmd_uses_raw_turn_breakaway_pwm_when_available(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_MAPS = True
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = True
        telemetry_robot.SCORE_POWER_PWM_TURN_LEFT = {
            1: {"pwm": 102, "power": telemetry_robot.pwm_to_power(102)},
            100: {"pwm": 255, "power": telemetry_robot.pwm_to_power(255)},
        }
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT = {1: 135, 100: 300}
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_pwm_payload(
            left_pwm=127,
            right_pwm=122,
            duration_ms=130,
        )

        power, pwm, score_used, duration_ms = telemetry_robot.speed_power_pwm_for_cmd("l", 1)

        self.assertEqual(int(score_used), 1)
        self.assertEqual(int(pwm), 127)
        self.assertEqual(int(duration_ms), 130)
        self.assertAlmostEqual(float(power), float(telemetry_robot.pwm_to_power(127)), places=6)

    def test_breakaway_recommended_score_none_does_not_alias_to_default_score(self):
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_seed_and_ceiling_payload(
            seed_pwm=102,
            left_ceiling_pwm=127,
            right_ceiling_pwm=122,
            duration_ms=130,
        )

        self.assertIsNone(telemetry_robot.breakaway_recommended_score_for_cmd("l"))
        self.assertIsNone(telemetry_robot.breakaway_recommended_score_for_cmd("r"))

    def test_speed_power_pwm_for_cmd_uses_ceiling_candidate_when_recommendation_is_seed_only(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_MAPS = True
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = True
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_seed_and_ceiling_payload(
            seed_pwm=102,
            left_ceiling_pwm=127,
            right_ceiling_pwm=122,
            duration_ms=130,
        )

        left_power, left_pwm, left_score, left_duration_ms = telemetry_robot.speed_power_pwm_for_cmd("l", 1)
        right_power, right_pwm, right_score, right_duration_ms = telemetry_robot.speed_power_pwm_for_cmd("r", 1)

        self.assertEqual(int(left_score), 1)
        self.assertEqual(int(left_pwm), 127)
        self.assertEqual(int(left_duration_ms), 130)
        self.assertAlmostEqual(float(left_power), float(telemetry_robot.pwm_to_power(127)), places=6)
        self.assertEqual(int(right_score), 1)
        self.assertEqual(int(right_pwm), 122)
        self.assertEqual(int(right_duration_ms), 130)
        self.assertAlmostEqual(float(right_power), float(telemetry_robot.pwm_to_power(122)), places=6)

    def test_one_percent_discovery_note_reports_raw_turn_breakaway_pwm(self):
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self._write_turn_breakaway_pwm_payload(
            left_pwm=127,
            right_pwm=122,
            duration_ms=130,
        )

        note = telemetry_robot.one_percent_discovery_note("r", 1)

        self.assertEqual(note, "turn 1%=pwm 122 @ 130ms; breakaway 2/2")


if __name__ == "__main__":
    unittest.main()
