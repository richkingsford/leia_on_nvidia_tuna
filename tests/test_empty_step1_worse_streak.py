import unittest

import a_follow_the_brick as follow


def _reading(dist_err: float, x_err: float) -> dict:
    return {
        "visible": True,
        "confident": True,
        "conf": 99.0,
        "dist_mm": follow._dist_target_mm() + float(dist_err),
        "x_mm": follow._x_target_mm() + float(x_err),
        "y_mm": -42.0,
    }


def _pending(dist_err: float, x_err: float, action: str = "VIRTUAL_WALL_RECOVERY_FWD") -> dict:
    return {
        "action": action,
        "cmd": "f",
        "duration_ms": 300,
        "dist_mm": follow._dist_target_mm() + float(dist_err),
        "x_mm": follow._x_target_mm() + float(x_err),
        "y_mm": -42.0,
        "dist_err": float(dist_err),
        "x_err": float(x_err),
    }


class TestEmptyStep1WorseStreak(unittest.TestCase):
    def setUp(self):
        follow._set_game_profile("empty")

    def test_step1_ready_rejects_too_close_dist_even_when_x_is_good(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "dist_axis": {
                    "win_target_mm": 175.0,
                    "win_tol_mm": 25.0,
                    "lower_win_tol_mm": 25.0,
                },
                "x_axis": {
                    "win_target_mm": 5.9,
                    "win_tol_mm": 7.0,
                },
                "win_confirmation": {
                    "min_axis_closeness_pct": 0.0,
                },
            }

            too_close = {
                "visible": True,
                "confident": True,
                "dist_mm": 76.0,
                "x_mm": 5.9,
            }
            in_band = {
                "visible": True,
                "confident": True,
                "dist_mm": 150.0,
                "x_mm": 5.9,
            }

            self.assertFalse(follow._step1_dist_x_target_ready(too_close))
            self.assertTrue(follow._step1_dist_x_target_ready(in_band))
        finally:
            follow._follow_motion_config = old_follow_motion_config

    def test_three_consecutive_worse_acts_trip_hard_stop(self):
        stats = follow._new_game_stats()

        for index in range(2):
            detail = follow._update_empty_step1_worse_act_streak(
                stats,
                _pending(70.0 + index, 50.0 + index),
                _reading(72.0 + index, 52.0 + index),
                action="VIRTUAL_WALL_RECOVERY_FWD",
            )
            self.assertEqual(detail["count"], index + 1)
            self.assertFalse(stats.get("step1_confident_worse_hard_stop", False))

        detail = follow._update_empty_step1_worse_act_streak(
            stats,
            _pending(72.0, 52.0),
            _reading(75.0, 55.0),
            action="VIRTUAL_WALL_RECOVERY_FWD",
        )

        self.assertEqual(detail["count"], 3)
        self.assertTrue(stats["step1_confident_worse_hard_stop"])

    def test_improving_act_resets_worse_streak(self):
        stats = follow._new_game_stats()

        follow._update_empty_step1_worse_act_streak(
            stats,
            _pending(70.0, 50.0),
            _reading(72.0, 52.0),
            action="VIRTUAL_WALL_RECOVERY_FWD",
        )
        detail = follow._update_empty_step1_worse_act_streak(
            stats,
            _pending(72.0, 52.0),
            _reading(68.0, 48.0),
            action="VIRTUAL_WALL_RECOVERY_FWD",
        )

        self.assertEqual(detail["count"], 0)
        self.assertFalse(stats.get("step1_confident_worse_hard_stop", False))


if __name__ == "__main__":
    unittest.main()
