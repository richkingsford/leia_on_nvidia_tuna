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

    def _patch_motion_config(self):
        old_follow_motion_config = follow._follow_motion_config
        follow._follow_motion_config = lambda: {
            "dist_axis": {
                "win_target_mm": 100.0,
                "win_tol_mm": 25.0,
                "lower_win_tol_mm": 25.0,
            },
            "x_axis": {
                "win_target_mm": 0.0,
                "win_tol_mm": 5.0,
            },
            "y_axis": {
                "enabled": False,
            },
            "win_confirmation": {
                "min_axis_closeness_pct": 0.0,
            },
        }
        return old_follow_motion_config

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

    def test_empty_step1_cluster_keeps_three_agreeing_frames(self):
        rows = [
            {"dist_mm": 210.0, "x_mm": 4.0, "y_mm": 1.0, "conf": 99.0},
            {"dist_mm": 98.0, "x_mm": 1.0, "y_mm": 1.0, "conf": 99.0},
            {"dist_mm": 101.0, "x_mm": 2.0, "y_mm": 0.0, "conf": 98.0},
            {"dist_mm": 103.0, "x_mm": 0.0, "y_mm": 2.0, "conf": 97.0},
        ]

        cluster = follow._largest_empty_reading_cluster(rows)

        self.assertEqual(len(cluster), 3)
        self.assertEqual([row["dist_mm"] for row in cluster], [98.0, 101.0, 103.0])

    def test_empty_step1_slight_x_outside_cannot_win(self):
        old_follow_motion_config = self._patch_motion_config()
        try:
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "conf": 99.0,
                    "dist_mm": 100.0,
                    "x_mm": 8.0,
                    "y_mm": 0.0,
                },
                virtual_safety_armed=False,
            )

            self.assertNotEqual(plan["kind"], "hold")
            self.assertNotEqual(plan.get("action"), "HAPPY")
        finally:
            follow._follow_motion_config = old_follow_motion_config

    def test_empty_step1_green_dist_material_x_uses_one_backoff_then_stops(self):
        old_follow_motion_config = self._patch_motion_config()
        try:
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "conf": 99.0,
                    "dist_mm": 100.0,
                    "x_mm": 20.0,
                    "y_mm": 0.0,
                },
                virtual_safety_armed=False,
            )

            self.assertEqual(plan["action"], "BCK_CLOSE_BAND_X_REAPPROACH")
            self.assertEqual(plan["reason"], "empty_s1_dist_green_x_backoff_reapproach")
            self.assertTrue(plan["allow_empty_step1_reverse_recovery"])
            self.assertEqual(
                follow._block_reverse_gap_closing_plan(plan)["action"],
                "BCK_CLOSE_BAND_X_REAPPROACH",
            )

            stats = {}
            first = follow._apply_empty_step1_close_band_backoff_budget(stats, plan)
            second = follow._apply_empty_step1_close_band_backoff_budget(stats, plan)

            self.assertEqual(first["action"], "BCK_CLOSE_BAND_X_REAPPROACH")
            self.assertEqual(second["kind"], "wait")
            self.assertEqual(second["reason"], "empty_s1_close_band_backoff_budget_stop")
        finally:
            follow._follow_motion_config = old_follow_motion_config

    def test_far_side_forward_dist_jump_is_accepted_without_ghost_probe(self):
        old_follow_motion_config = self._patch_motion_config()
        try:
            self.assertTrue(
                follow._empty_step1_far_side_forward_dist_jump_ok(
                    {"dist_mm": 140.0, "x_mm": 0.0, "y_mm": 0.0},
                    {"dist_mm": 170.0, "x_mm": 0.0, "y_mm": 0.0},
                )
            )

            stats = {
                "pending_observation": {
                    "action": "FWD_TEST",
                    "cmd": "f",
                    "duration_ms": 250,
                    "dist_mm": 140.0,
                    "x_mm": 0.0,
                    "y_mm": 0.0,
                }
            }
            detail = follow._record_observed_after_pending_act(
                stats,
                {"dist_mm": 170.0, "x_mm": 0.0, "y_mm": 0.0},
            )

            self.assertIsNotNone(detail)
            self.assertTrue(detail["observed"])
            self.assertNotIn("forward_dist_ghost_probe", stats)
            self.assertEqual(stats["forward_dist_far_side_jump_accepted_count"], 1)
        finally:
            follow._follow_motion_config = old_follow_motion_config

    def test_three_consecutive_confident_worse_windows_trip_hard_stop(self):
        stats = follow._new_game_stats()

        for index in range(3):
            detail = follow._update_empty_step1_worse_act_streak(
                stats,
                _pending(70.0 + index, 50.0 + index),
                _reading(77.0 + index, 57.0 + index),
                action="VIRTUAL_WALL_RECOVERY_FWD",
            )
            self.assertEqual(detail["count"], 0)
            self.assertEqual(detail["trend_window"], index + 1)
            self.assertFalse(stats.get("step1_confident_worse_hard_stop", False))

        for index in range(3):
            detail = follow._update_empty_step1_worse_act_streak(
                stats,
                _pending(73.0 + index, 53.0 + index),
                _reading(80.0 + index, 60.0 + index),
                action="VIRTUAL_WALL_RECOVERY_FWD",
            )
            self.assertEqual(detail["count"], index + 1)

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
