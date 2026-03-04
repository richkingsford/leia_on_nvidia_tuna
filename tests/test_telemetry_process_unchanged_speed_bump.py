import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessUnchangedSpeedBump(unittest.TestCase):
    def test_bumps_speed_for_repeated_unchanged_same_command(self):
        score, state, bumped = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=1,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=None,
        )
        self.assertTrue(bumped)
        self.assertEqual(score, 2)
        self.assertEqual(int(state.get("streak", 0)), 1)
        self.assertEqual(state.get("cmd"), "f")
        self.assertEqual(state.get("correction_type"), "distance")
        self.assertEqual(state.get("error_signature"), 1.77)

    def test_streak_advances_when_same_signature_repeats(self):
        _, state1, _ = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=1,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=None,
        )
        score2, state2, bumped2 = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=1,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=state1,
        )
        self.assertTrue(bumped2)
        self.assertEqual(score2, 3)
        self.assertEqual(int(state2.get("streak", 0)), 2)
        self.assertEqual(int(state2.get("bump_count", 0)), 2)

    def test_does_not_bump_when_not_unchanged_or_previous_cmd_differs(self):
        score_a, state_a, bumped_a = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=1,
            delta_class="closer",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state={"streak": 3, "cmd": "f", "correction_type": "distance", "error_signature": 1.77},
        )
        self.assertFalse(bumped_a)
        self.assertEqual(score_a, 1)
        self.assertEqual(int(state_a.get("streak", 0)), 0)

        score_b, state_b, bumped_b = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=1,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="b",
            state={"streak": 2, "cmd": "f", "correction_type": "distance", "error_signature": 1.77},
        )
        self.assertFalse(bumped_b)
        self.assertEqual(score_b, 1)
        self.assertEqual(int(state_b.get("streak", 0)), 0)

    def test_align_error_signature_for_distance(self):
        sig = telemetry_process._align_error_signature_for_correction(
            "distance",
            {"dist": 43.03, "dist_target": 41.26},
        )
        self.assertEqual(sig, 1.77)

    def test_bump_is_capped_at_twenty_percent(self):
        score1, state1, bumped1 = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=19,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=None,
        )
        self.assertTrue(bumped1)
        self.assertEqual(int(score1), int(telemetry_process.REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP))

        score2, _state2, bumped2 = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=25,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=state1,
        )
        self.assertFalse(bumped2)
        self.assertEqual(int(score2), int(telemetry_process.REPEATED_UNCHANGED_SPEED_BUMP_SCORE_CAP))

    def test_exhausts_after_eight_bumps(self):
        state = None
        score = 5
        for _ in range(int(telemetry_process.REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT)):
            score, state, bumped = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
                cmd="f",
                score=score,
                delta_class="unchanged",
                correction_type="distance",
                error_signature=1.77,
                previous_cmd="f",
                state=state,
            )
            self.assertTrue(bumped)

        self.assertTrue(bool((state or {}).get("exhausted")))
        score2, state2, bumped2 = telemetry_process._maybe_apply_repeated_unchanged_speed_bump(
            cmd="f",
            score=score,
            delta_class="unchanged",
            correction_type="distance",
            error_signature=1.77,
            previous_cmd="f",
            state=state,
        )
        self.assertFalse(bumped2)
        self.assertTrue(bool((state2 or {}).get("exhausted")))
        self.assertEqual(int((state2 or {}).get("bump_count", 0)), int(telemetry_process.REPEATED_UNCHANGED_SPEED_BUMP_MAX_COUNT))
        self.assertEqual(int(score2), int(score))


if __name__ == "__main__":
    unittest.main()
