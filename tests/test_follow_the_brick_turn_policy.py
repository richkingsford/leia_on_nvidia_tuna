import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_follow_the_brick as follow


class _FakeRobot:
    def __init__(self):
        self.commands = []
        self.custom_commands = []
        self.stops = 0

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.commands.append((cmd, pwm, duration_ms))
        return {"cmd_sent": cmd, "pwm": pwm, "duration_ms": duration_ms}

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        self.custom_commands.append((cmd, list(actions), duration_ms))
        return {"cmd_sent": cmd, "actions": list(actions), "duration_ms": duration_ms}

    def stop(self):
        self.stops += 1


class TestFollowTheBrickTurnPolicy(unittest.TestCase):
    def setUp(self):
        follow._set_game_profile("empty")

    def _reading_for_gap(self, *, dist_gap_mm: float, x_gap_mm: float) -> dict:
        y_cfg = follow._follow_y_axis_config()
        return {
            "dist_mm": follow._dist_target_mm() + float(dist_gap_mm),
            "x_mm": follow._x_target_mm() + float(x_gap_mm),
            "y_mm": float(y_cfg.get("win_target_mm", 0.0)),
            "confidence": 0.99,
        }

    def test_far_low_confident_reading_is_pickup_suspect_stop(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": 301.0,
            "x_mm": follow._x_target_mm(),
            "y_mm": -79.0,
            "conf": 99.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertTrue(follow._pickup_suspected_reading(reading))
        self.assertEqual(plan["kind"], "wait")
        self.assertEqual(plan["action"], "PICKUP_SUSPECT_STOP")
        self.assertEqual(plan["reason"], "pickup_suspected_far_low")

    def test_normal_far_start_is_not_pickup_suspect(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": 323.8,
            "x_mm": follow._x_target_mm(),
            "y_mm": -32.2,
            "conf": 99.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertFalse(follow._pickup_suspected_reading(reading))
        self.assertNotEqual(plan.get("reason"), "pickup_suspected_far_low")

    def test_uses_combined_bias_until_dist_gap_is_tiny(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=100.0, x_gap_mm=11.0)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "dist_only_creep")
        self.assertFalse(plan.get("use_production_turn_curve", False))

    def test_left_of_crosshair_avoids_forward_left_bias(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=25.0, x_gap_mm=-8.0)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "dist_micro_straight")

    def test_huge_x_gap_turns_before_closing_distance(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=80.0, x_gap_mm=120.0)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["cmd"], "r")
        self.assertEqual(plan["reason"], "wide_x_before_dist")
        self.assertFalse(plan.get("use_production_turn_curve", False))

    def test_gross_y_gap_recovers_mast_when_x_is_centered(self):
        y_cfg = follow._follow_y_axis_config()
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() + 45.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": float(y_cfg.get("win_target_mm")) - 25.0,
                "confidence": 0.99,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "u")

    def test_uses_sharp_x_only_curve_when_dist_gap_is_tiny(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=0.0, x_gap_mm=20.0)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertTrue(plan.get("use_production_turn_curve"))
        self.assertEqual(plan["reason"], "sharp_x_only_tiny_dist")

    def test_near_target_dist_micro_uses_micro_drive_floor(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=13.0, x_gap_mm=0.0)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["action"], "FWD_MICRO")
        self.assertEqual(plan["duration_ms"], follow._follow_dist_approach_policy()["micro_nudge_min_effective_pulse_ms"])
        self.assertGreaterEqual(plan["pwm"], follow._pwm_floor_for_cmd("f"))

    def test_tiny_x_outside_gap_does_not_plan_turn_polish(self):
        x_gap = follow._x_tol_mm() + 1.0
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=25.0, x_gap_mm=x_gap)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "tiny_x_dist_micro_straight")

    def test_tiny_x_only_gap_backs_off_instead_of_subfloor_turn(self):
        x_gap = follow._x_tol_mm() + 1.0
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=0.0, x_gap_mm=x_gap)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "b")
        self.assertEqual(plan["reason"], "tiny_x_only_backoff_instead_of_turn")
        self.assertGreaterEqual(plan["duration_ms"], follow._min_effective_drive_duration_ms("b"))

    def test_no_observed_wheel_motion_arms_next_same_action_boost(self):
        stats = follow._new_game_stats()
        stats["pending_observation"] = {
            "action": "BIAS_L_ADAPTIVE",
            "dist_mm": 179.0,
            "x_mm": -20.0,
            "y_mm": -19.0,
            "duration_ms": 300,
        }

        result = follow._record_observed_after_pending_act(
            stats,
            {"dist_mm": 179.1, "x_mm": -20.1, "y_mm": -19.0},
        )

        self.assertFalse(result["observed"])
        self.assertFalse(stats["stall_guard_triggered"])
        self.assertEqual(stats["stall_recovery_boost"]["action"], "BIAS_L_ADAPTIVE")
        self.assertAlmostEqual(stats["stall_recovery_boost"]["scale"], 1.1)

    def test_recovery_boost_increases_drive_bias_pwm_above_floor(self):
        reading = self._reading_for_gap(dist_gap_mm=17.0, x_gap_mm=21.0)
        reading.update({"visible": True, "confident": True, "conf": 90.0, "min_confidence_pct": 75.0})
        plan = follow._follow_action_plan(reading)
        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["action"], "BIAS_R_ADAPTIVE")
        stats = follow._new_game_stats()
        stats["stall_recovery_boost"] = {"action": "BIAS_R_ADAPTIVE", "scale": 1.1}

        boosted_plan = follow._apply_stall_recovery_boost(stats, plan)
        robot = _FakeRobot()
        send_result = follow._execute_follow_action(robot, boosted_plan, reading)

        self.assertEqual(boosted_plan["recovery_boost_scale"], 1.1)
        self.assertEqual(len(robot.custom_commands), 1)
        _cmd, boosted_actions, _duration_ms = robot.custom_commands[0]
        boosted_pwms = [int(row["pwm"]) for row in boosted_actions if int(row.get("pwm", 0) or 0) > 0]
        self.assertTrue(boosted_pwms)
        self.assertTrue(
            all(
                int(row["pwm"]) >= follow._pwm_floor_for_cmd(row["action"])
                for row in boosted_actions
                if int(row.get("pwm", 0) or 0) > 0
            )
        )
        self.assertEqual(send_result["x_curve"]["recovery_boost_scale"], 1.1)

    def test_mast_wrong_way_motion_marks_spool_unreliable(self):
        stats = follow._new_game_stats()
        stats["pending_observation"] = {
            "action": "MAST_U",
            "cmd": "u",
            "dist_mm": follow._dist_target_mm(),
            "x_mm": follow._x_target_mm(),
            "y_mm": -10.0,
            "y_target_mm": 0.0,
            "duration_ms": 600,
        }

        result = follow._record_observed_after_pending_act(
            stats,
            {"dist_mm": follow._dist_target_mm(), "x_mm": follow._x_target_mm(), "y_mm": -14.0},
        )

        self.assertFalse(result["observed"])
        self.assertGreater(stats["mast_spool_unreliable_countdown"], 0)
        self.assertEqual(
            stats["mast_spool_last_unreliable_detail"]["status"],
            "target_regressed",
        )

    def test_unreliable_spool_caps_next_mast_plan(self):
        stats = follow._new_game_stats()
        stats["mast_spool_unreliable_countdown"] = 2
        plan = {
            "kind": "mast",
            "cmd": "u",
            "action": "MAST_U",
            "duration_ms": 900,
            "reason": "final_y",
        }

        capped = follow._cap_mast_plan_for_unreliable_spool(stats, plan)

        self.assertTrue(capped["spool_unreliable_capped"])
        self.assertLessEqual(
            capped["duration_ms"],
            follow._follow_y_axis_config()["spool_reversal_mast_max_ms"],
        )

    def test_close_trap_reverse_stalls_without_boost(self):
        stats = follow._new_game_stats()
        stats["pending_observation"] = {
            "action": "BCK",
            "cmd": "b",
            "dist_err": -119.0,
            "dist_mm": 97.6,
            "x_mm": 9.4,
            "y_mm": -27.8,
            "duration_ms": 180,
        }

        result = follow._record_observed_after_pending_act(
            stats,
            {"dist_mm": 96.9, "x_mm": 9.5, "y_mm": -28.4},
        )

        self.assertFalse(result["observed"])
        self.assertTrue(stats["stall_guard_triggered"])
        self.assertIsNone(stats["stall_recovery_boost"])
        self.assertTrue(stats["stall_guard_detail"]["close_trap_limited"])
        self.assertEqual(stats["stall_guard_detail"]["max_no_change_tries"], 1)

    def test_step2_precision_polishes_x_when_distance_and_y_are_ready(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 3,
            "precision_hard_max_attempts": 5,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        before = {
            "visible": True,
            "confident": True,
            "conf": 95.0,
            "dist_mm": 149.0,
            "x_mm": 18.0,
            "y_mm": -36.2,
        }
        after = dict(before, x_mm=4.0)
        readings = iter([after, after])
        old_read = follow._read_brick_measurement
        old_reset = follow._reset_follow_reading_history
        old_sleep = follow.time.sleep
        try:
            follow._read_brick_measurement = lambda _vision: next(readings)
            follow._reset_follow_reading_history = lambda *_args, **_kwargs: None
            follow.time.sleep = lambda _seconds: None
            robot = _FakeRobot()

            final, counts = follow._step2_precision_settle_to_targets(object(), robot, before, step2)
        finally:
            follow._read_brick_measurement = old_read
            follow._reset_follow_reading_history = old_reset
            follow.time.sleep = old_sleep

        self.assertEqual(counts["turn_r"], 1)
        self.assertEqual(counts["target_hit_confirmed"], 1)
        self.assertEqual(final["x_mm"], 4.0)
        self.assertEqual(len(robot.custom_commands), 1)

    def test_step2_precision_stops_when_x_polish_leaves_distance_too_close(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 3,
            "precision_hard_max_attempts": 5,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        before = {
            "visible": True,
            "confident": True,
            "conf": 95.0,
            "dist_mm": 149.0,
            "x_mm": 18.0,
            "y_mm": -36.2,
        }
        too_close = dict(before, dist_mm=138.0)
        old_read = follow._read_brick_measurement
        old_reset = follow._reset_follow_reading_history
        old_sleep = follow.time.sleep
        try:
            follow._read_brick_measurement = lambda _vision: too_close
            follow._reset_follow_reading_history = lambda *_args, **_kwargs: None
            follow.time.sleep = lambda _seconds: None
            robot = _FakeRobot()

            final, counts = follow._step2_precision_settle_to_targets(object(), robot, before, step2)
        finally:
            follow._read_brick_measurement = old_read
            follow._reset_follow_reading_history = old_reset
            follow.time.sleep = old_sleep

        self.assertEqual(counts["turn_r"], 1)
        self.assertEqual(counts["dist_too_close_safety_stop"], 1)
        self.assertEqual(final["dist_mm"], 138.0)

    def test_step2_target_rejects_frozen_xz_when_raw_distance_is_too_close(self):
        step2 = {
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        reading = {
            "confident": True,
            "xz_frozen": True,
            "dist_mm": 149.0,
            "raw_dist_mm": 138.0,
            "x_mm": 4.0,
            "raw_x_mm": 4.0,
            "y_mm": -36.2,
        }

        ready, reason, _closeness = follow._step2_targets_ready(reading, step2)

        self.assertFalse(ready)
        self.assertEqual(reason, "step2_raw_dist_too_close")

    def test_step2_target_rejects_pickup_suspect_with_raw_frozen_distance(self):
        step2 = {
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        reading = {
            "confident": True,
            "xz_frozen": True,
            "dist_mm": 149.0,
            "raw_dist_mm": 301.0,
            "x_mm": 4.0,
            "raw_x_mm": 4.0,
            "y_mm": -79.0,
        }

        ready, reason, _closeness = follow._step2_targets_ready(reading, step2)

        self.assertTrue(follow._pickup_suspected_reading(reading))
        self.assertFalse(ready)
        self.assertEqual(reason, "pickup_suspected_far_low")

    def test_step2_precision_stops_without_motion_on_pickup_suspect(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 3,
            "precision_hard_max_attempts": 5,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        reading = {
            "visible": True,
            "confident": True,
            "conf": 99.0,
            "dist_mm": 301.0,
            "x_mm": 4.0,
            "y_mm": -79.0,
        }
        robot = _FakeRobot()

        final, counts = follow._step2_precision_settle_to_targets(object(), robot, reading, step2)

        self.assertEqual(final["dist_mm"], 301.0)
        self.assertEqual(counts["pickup_suspected_stop"], 1)
        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.custom_commands), 0)
        self.assertGreaterEqual(robot.stops, 1)

    def test_step2_seat_sequence_skips_all_motion_on_pickup_suspect_start(self):
        step2 = {
            "precision_settle_enabled": True,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 10.0,
                "x_mm": 4.0,
                "x_tol_mm": 9.0,
                "y_mm": -36.2,
                "y_tol_mm": 3.0,
            },
        }
        reading = {
            "visible": True,
            "confident": True,
            "conf": 99.0,
            "dist_mm": 301.0,
            "x_mm": 4.0,
            "y_mm": -79.0,
        }
        old_read = follow._read_brick_measurement
        try:
            follow._read_brick_measurement = lambda _vision: reading
            robot = _FakeRobot()

            result = follow._run_step2_seat_sequence(object(), robot, step_cfg=step2)
        finally:
            follow._read_brick_measurement = old_read

        self.assertFalse(result["target_met"])
        self.assertEqual(result["reason"], "step2_pickup_suspected_far_low")
        self.assertEqual(result["precision_counts"]["pickup_suspected_stop"], 1)
        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.custom_commands), 0)
        self.assertGreaterEqual(robot.stops, 1)


if __name__ == "__main__":
    unittest.main()
