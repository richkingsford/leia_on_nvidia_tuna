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

    def test_mast_action_duration_capped_to_one_second(self):
        up = follow._mast_action_spec("u", duration_ms=2500)
        down = follow._mast_action_spec("d", duration_ms=1200)

        self.assertEqual(up["duration_ms"], 1000)
        self.assertEqual(down["duration_ms"], 1000)

    def test_visibility_recovery_mast_down_duration_capped_to_one_second(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "visibility_recovery": {"mast_down_duration_ms": 2500}
            }

            cfg = follow._visibility_recovery_config()
        finally:
            follow._follow_motion_config = old_follow_motion_config

        self.assertEqual(cfg["mast_down_duration_ms"], 1000)

    def test_x_sign_convention_positive_right_negative_left(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "x_axis": {"positive_error_turn_cmd": "r"}
            }

            self.assertEqual(follow._turn_cmd_to_close_x_gap(12.0), "r")
            self.assertEqual(follow._turn_cmd_to_close_x_gap(-12.0), "l")
            self.assertIsNone(follow._turn_cmd_to_close_x_gap(0.0))
        finally:
            follow._follow_motion_config = old_follow_motion_config

    def test_holding_step2_does_not_enable_backward_x_inversion_by_default(self):
        step2 = follow._default_step2_like_config()

        self.assertFalse(step2.get("precision_x_only_backward_protect_dist_enabled", False))

    def test_reset_mast_up_action_is_one_second_when_enabled(self):
        old_reset_motion_config = follow._reset_motion_config
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {"y_target_mm": -5.0},
                "mast_up": {
                    "enabled": True,
                    "min_duration_ms": 2500,
                    "max_duration_ms": 3000,
                    "pwm": 255,
                    "cmd": "u",
                    "settle_s": 0.1,
                },
            }

            action, duration_ms, _settle_s = follow._reset_mast_up_action_spec(
                reading={"y_mm": -80.0}
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config

        self.assertIsNotNone(action)
        self.assertEqual(duration_ms, 1000)
        self.assertEqual(action["duration_ms"], 1000)

    def test_reset_reverse_turn_does_not_call_mast_up(self):
        old_reset_motion_config = follow._reset_motion_config
        old_mast_up_spec = follow._reset_mast_up_action_spec
        old_send_custom = follow.guarded_send_custom_actions_pwm
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {
                    "straight_back_first": {
                        "enabled": True,
                        "duration_ms": 100,
                        "pwm": 103,
                        "mast_up_delay_fraction": 0.5,
                    }
                },
                "mast_up": {"enabled": True, "min_duration_ms": 1000, "max_duration_ms": 1000},
            }
            follow._reset_mast_up_action_spec = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("reset must not request mast-up")
            )
            follow.guarded_send_custom_actions_pwm = lambda *_args, **_kwargs: {
                "cmd_sent": "b",
                "duration_ms": _kwargs.get("duration_ms"),
            }

            result = follow._reset_reverse_turn(
                _FakeRobot(),
                "r",
                {"dist_mm": 200.0, "x_mm": 0.0, "y_mm": -80.0},
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config
            follow._reset_mast_up_action_spec = old_mast_up_spec
            follow.guarded_send_custom_actions_pwm = old_send_custom

        self.assertEqual(result["mast_up_ms"], 0)
        self.assertFalse(any(action.get("target") == "m" for action in result["actions"]))

    def test_reset_reverse_turn_obeys_total_back_budget(self):
        old_reset_motion_config = follow._reset_motion_config
        old_send_custom = follow.guarded_send_custom_actions_pwm
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {
                    "max_total_back_ms": 120,
                    "straight_back_first": {
                        "enabled": True,
                        "duration_ms": 500,
                        "pwm": 103,
                    },
                },
            }
            follow.guarded_send_custom_actions_pwm = lambda *_args, **_kwargs: {
                "cmd_sent": "b",
                "duration_ms": _kwargs.get("duration_ms"),
            }
            back_budget = follow._reset_back_budget_state({"max_total_back_ms": 120})

            result = follow._reset_reverse_turn(
                _FakeRobot(),
                "r",
                {"dist_mm": 80.0, "x_mm": 0.0, "y_mm": -25.0},
                back_budget=back_budget,
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config
            follow.guarded_send_custom_actions_pwm = old_send_custom

        self.assertEqual(result["wheel_ms"], 120)
        self.assertEqual(back_budget["used_ms"], 120)

    def test_reset_distance_cap_respects_straight_back_min_duration(self):
        old_reset_motion_config = follow._reset_motion_config
        old_send_custom = follow.guarded_send_custom_actions_pwm
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {
                    "dist_target_mm": 183.4,
                    "dist_tol_mm": 30.0,
                    "straight_back_first": {
                        "enabled": True,
                        "duration_ms": 3000,
                        "duration_min_ms": 800,
                        "duration_max_ms": 1500,
                        "pwm": 125,
                    },
                },
            }
            follow.guarded_send_custom_actions_pwm = lambda *_args, **_kwargs: {
                "cmd_sent": "b",
                "duration_ms": _kwargs.get("duration_ms"),
            }

            result = follow._reset_reverse_turn(
                _FakeRobot(),
                "r",
                {"dist_mm": 111.8, "x_mm": 9.9, "y_mm": -33.1},
                rng=type("FixedRng", (), {"uniform": lambda *_args: 1200.0})(),
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config
            follow.guarded_send_custom_actions_pwm = old_send_custom

        self.assertEqual(result["wheel_ms"], 800)

    def test_reset_step1_band_distance_still_backs_up_when_x_offset_not_ready(self):
        old_reset_motion_config = follow._reset_motion_config
        old_send_custom = follow.guarded_send_custom_actions_pwm
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {
                    "dist_target_mm": 183.4,
                    "dist_tol_mm": 30.0,
                    "target_abs_x_mm": 20.0,
                    "x_offset_min_mm": 18.0,
                    "x_offset_max_mm": 30.0,
                    "straight_back_first": {
                        "enabled": True,
                        "duration_ms": 3000,
                        "duration_min_ms": 800,
                        "duration_max_ms": 1500,
                        "pwm": 125,
                    },
                },
            }
            follow.guarded_send_custom_actions_pwm = lambda *_args, **_kwargs: {
                "cmd_sent": "b",
                "duration_ms": _kwargs.get("duration_ms"),
            }

            result = follow._reset_reverse_turn(
                _FakeRobot(),
                "r",
                {"dist_mm": 135.6, "x_mm": 14.2, "y_mm": -37.2},
                rng=type("FixedRng", (), {"uniform": lambda *_args: 1200.0})(),
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config
            follow.guarded_send_custom_actions_pwm = old_send_custom

        self.assertEqual(result["wheel_ms"], 800)

    def test_reset_straight_back_uses_configured_pwm_ceiling(self):
        old_reset_motion_config = follow._reset_motion_config
        old_send_custom = follow.guarded_send_custom_actions_pwm
        sent_actions = []
        try:
            follow._reset_motion_config = lambda: {
                "reverse_turn": {
                    "dist_target_mm": 183.4,
                    "dist_tol_mm": 30.0,
                    "straight_back_first": {
                        "enabled": True,
                        "duration_ms": 800,
                        "duration_min_ms": 800,
                        "pwm": 115,
                        "pwm_ceiling": 115,
                    },
                },
            }

            def _capture_send(*args, **kwargs):
                sent_actions.append(list(args[2]))
                return {"cmd_sent": "b", "duration_ms": kwargs.get("duration_ms")}

            follow.guarded_send_custom_actions_pwm = _capture_send

            follow._reset_reverse_turn(
                _FakeRobot(),
                "r",
                {"dist_mm": 111.8, "x_mm": 9.9, "y_mm": -33.1},
            )
        finally:
            follow._reset_motion_config = old_reset_motion_config
            follow.guarded_send_custom_actions_pwm = old_send_custom

        self.assertTrue(sent_actions)
        wheel_pwms = [
            int(action.get("pwm", 0))
            for action in sent_actions[0]
            if str(action.get("action")) in {"f", "b"}
        ]
        self.assertEqual(wheel_pwms, [115, 115])

    def test_reset_adjustment_config_has_stale_back_guard(self):
        cfg = follow._reset_adjustment_config(
            {
                "adjustment": {
                    "back_progress_min_delta_mm": 3.0,
                    "max_stale_back_attempts": 2,
                }
            }
        )

        self.assertEqual(cfg["back_progress_min_delta_mm"], 3.0)
        self.assertEqual(cfg["max_stale_back_attempts"], 2)

    def test_reset_visibility_recovery_does_not_mast_down(self):
        old_visibility_recovery_config = follow._visibility_recovery_config
        old_read = follow._read_brick_measurement
        old_sleep = follow.time.sleep
        try:
            follow._visibility_recovery_config = lambda: {
                "wait_s": 0.01,
                "poll_s": 0.01,
                "mast_down_duration_ms": 1000,
                "mast_down_pwm": 255,
            }
            follow._read_brick_measurement = lambda *_args, **_kwargs: {
                "confident": False,
                "visible": False,
                "reason": "not_visible",
            }
            follow.time.sleep = lambda _seconds: None
            robot = _FakeRobot()

            result = follow._wait_for_visibility_recovery(
                object(),
                robot,
                {"confident": False, "visible": False, "reason": "not_visible"},
                context="reset_after_motion",
                timeout_s=0.01,
                sample_s=0.01,
            )
        finally:
            follow._visibility_recovery_config = old_visibility_recovery_config
            follow._read_brick_measurement = old_read
            follow.time.sleep = old_sleep

        self.assertFalse(result["confident"])
        self.assertEqual(robot.commands, [])

    def test_robot_visibility_recovery_is_wait_only_by_default(self):
        cfg = follow._visibility_recovery_config()

        self.assertEqual(cfg["mast_down_duration_ms"], 0)

    def test_step2_seat_mast_duration_capped_to_one_second(self):
        cfg = follow._default_step2_like_config()
        cfg["seat_mast_cmd"] = "d"

        follow._apply_step2_like_config({"seat_mast_duration_ms": 2500}, cfg)

        self.assertEqual(cfg["seat_mast_duration_ms"], 1000)

    def test_low_y_inside_x_dist_gate_holds_without_mast_up(self):
        y_cfg = follow._follow_y_axis_config()
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": follow._dist_target_mm() + 43.0,
            "x_mm": follow._x_target_mm() - 0.8,
            "y_mm": float(y_cfg.get("win_target_mm")) - 18.0,
            "conf": 95.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_far_visible_low_start_closes_distance_without_mast_up(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": 313.8,
            "x_mm": 2.6,
            "y_mm": -58.8,
            "conf": 93.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertFalse(follow._pickup_suspected_reading(reading))
        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "f")
        self.assertNotIn("mast_cmd", plan)
        self.assertNotIn("MAST_U", plan["action"])

    def test_empty_step1_inband_x_uses_gentle_curve_while_closing_dist(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": follow._dist_target_mm() + 30.0,
            "x_mm": follow._x_target_mm() + 4.5,
            "y_mm": -37.0,
            "conf": 95.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["strength"], "gentle")
        self.assertIn("BIAS_", plan["action"])
        self.assertNotEqual(plan["action"], "FWD")
        self.assertLessEqual(plan["duration_ms"], 120)
        self.assertEqual(plan["reason"], "empty_s1_inband_x_gentle_curve_while_closing_dist")

    def test_empty_step1_forward_micro_is_short_near_new_dist_floor(self):
        old_dist_tol = follow._dist_tol_mm
        try:
            follow._dist_tol_mm = lambda: 5.0
            reading = {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() + 24.5,
                "x_mm": follow._x_target_mm() + 1.3,
                "y_mm": -37.0,
                "conf": 95.0,
            }

            plan = follow._follow_action_plan(reading)
        finally:
            follow._dist_tol_mm = old_dist_tol

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["action"], "FWD_MICRO")
        self.assertLessEqual(plan["duration_ms"], 100)

    def test_empty_step1_micro_dist_with_residual_x_uses_gentle_curve(self):
        old_dist_tol = follow._dist_tol_mm
        try:
            follow._dist_tol_mm = lambda: 5.0
            reading = {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() + 6.2,
                "x_mm": follow._x_target_mm() - 2.3,
                "y_mm": -37.0,
                "conf": 95.0,
            }

            plan = follow._follow_action_plan(reading)
        finally:
            follow._dist_tol_mm = old_dist_tol

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["strength"], "gentle")
        self.assertIn("BIAS_", plan["action"])
        self.assertNotEqual(plan["action"], "FWD_MICRO")
        self.assertLessEqual(plan["duration_ms"], 70)
        self.assertEqual(plan["reason"], "empty_s1_inband_x_gentle_curve_while_closing_dist")

    def test_empty_step1_learning_curve_strength_uses_gentle_for_small_x_misses(self):
        self.assertEqual(follow._empty_step1_learning_curve_strength(3.0), "gentle")
        self.assertEqual(follow._empty_step1_learning_curve_strength(5.0), "superstrong")
        self.assertEqual(follow._empty_step1_learning_curve_strength(20.0), "superstrong")

    def test_empty_step1_uses_superstrong_when_x_is_not_aligned(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=45.0, x_gap_mm=follow._x_tol_mm() + 8.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["strength"], "superstrong")
        self.assertGreaterEqual(plan["duration_ms"], follow.EMPTY_S1_LEARNING_CURVE_MS)

    def test_empty_step1_loose_tolerance_x_miss_still_turns_before_forward(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=45.0, x_gap_mm=9.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["strength"], "superstrong")
        self.assertNotEqual(plan["action"], "FWD")

    def test_reset_sharp_x_offset_uses_configured_breakaway_pwm(self):
        captured = {}
        old_send = follow._send_reset_custom_actions_pwm
        try:
            def _capture(_robot, cmd, action_specs, **kwargs):
                captured["cmd"] = cmd
                captured["actions"] = list(action_specs)
                captured["duration_ms"] = kwargs.get("duration_ms")
                return {"cmd_sent": cmd, "duration_ms": kwargs.get("duration_ms")}

            follow._send_reset_custom_actions_pwm = _capture

            follow._reset_sharp_turn_adjust(
                _FakeRobot(),
                turn_cmd="r",
                reading={"dist_mm": 150.0, "x_mm": 0.0, "confident": True},
                duration_ms=250,
                reset_cfg={
                    "low_x_extra_sharp_turn": {
                        "slower_pwm": 103,
                        "faster_pwm": 162,
                    }
                },
                reason="test",
            )
        finally:
            follow._send_reset_custom_actions_pwm = old_send

        self.assertEqual(captured["duration_ms"], 250)
        self.assertTrue(any(int(action.get("pwm", 0)) >= 160 for action in captured["actions"]))

    def test_empty_profile_near_gate_low_y_holds_without_mast_up(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 221.3,
                "x_mm": 4.1,
                "y_mm": -53.4,
                "conf": 97.0,
            }
        )

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_unreliable_spool_mast_cap_matches_regression_wish(self):
        self.assertEqual(follow._follow_y_axis_config()["spool_reversal_mast_max_ms"], 110)

    def test_too_close_recovery_does_not_attach_mast_to_backward_drive(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 119.3,
                "x_mm": -2.3,
                "y_mm": -23.0,
                "conf": 95.0,
            }
        )

        self.assertIn(plan["kind"], {"drive", "drive_bias"})
        self.assertEqual(plan["cmd"], "b")
        self.assertNotIn("mast_cmd", plan)
        self.assertNotIn("MAST_", plan["action"])

    def test_slightly_close_recovery_does_not_attach_mast_to_backward_drive(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 201.9,
                "x_mm": 8.6,
                "y_mm": -20.7,
                "conf": 93.0,
            }
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "b")
        self.assertNotIn("mast_cmd", plan)
        self.assertNotIn("MAST_", plan["action"])

    def test_zero_step1_mast_up_budget_blocks_cumulative_guard(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "y_axis": {"enabled": True, "max_step1_mast_up_ms": 0}
            }
            plan = {"kind": "mast", "cmd": "u", "duration_ms": 900}

            exceeded, used_ms, planned_ms, max_ms = follow._mast_up_budget_exceeded(
                {"step1_mast_up_ms": 900},
                plan,
            )
        finally:
            follow._follow_motion_config = old_follow_motion_config

        self.assertTrue(exceeded)
        self.assertEqual((used_ms, planned_ms, max_ms), (900, 900, 0))

    def test_zero_step1_mast_up_budget_treats_low_y_as_ok(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "dist_axis": {"win_target_mm": 216.8, "win_tol_mm": 50.0},
                "x_axis": {"win_target_mm": 5.9, "win_tol_mm": 3.0},
                "y_axis": {
                    "enabled": True,
                    "win_target_mm": -42.7,
                    "win_tol_mm": 5.0,
                    "max_step1_mast_up_ms": 0,
                },
            }
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": 216.8,
                    "x_mm": 5.9,
                    "y_mm": -60.0,
                    "conf": 99.0,
                }
            )
        finally:
            follow._follow_motion_config = old_follow_motion_config

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_zero_step1_mast_up_budget_strips_attached_up_from_drive(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "y_axis": {
                    "enabled": True,
                    "win_target_mm": -42.7,
                    "win_tol_mm": 5.0,
                    "max_step1_mast_up_ms": 0,
                },
            }
            plan = {
                "kind": "drive",
                "cmd": "f",
                "action": "FWD_MAST_U",
                "duration_ms": 200,
                "mast_cmd": "u",
                "mast_pwm": 255,
                "mast_duration_ms": 220,
            }
            capped = follow._cap_mast_up_plan_to_budget({}, plan)
        finally:
            follow._follow_motion_config = old_follow_motion_config

        self.assertEqual(capped["kind"], "drive")
        self.assertNotIn("mast_cmd", capped)
        self.assertTrue(capped["mast_up_budget_blocked"])
        self.assertIn("NO_MAST_U_BUDGET", capped["action"])

    def test_zero_step1_mast_up_budget_still_lowers_high_y(self):
        old_follow_motion_config = follow._follow_motion_config
        try:
            follow._follow_motion_config = lambda: {
                "dist_axis": {"win_target_mm": 216.8, "win_tol_mm": 50.0},
                "x_axis": {"win_target_mm": 5.9, "win_tol_mm": 3.0},
                "y_axis": {
                    "enabled": True,
                    "win_target_mm": -42.7,
                    "win_tol_mm": 5.0,
                    "max_step1_mast_up_ms": 0,
                },
            }
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": 216.8,
                    "x_mm": 5.9,
                    "y_mm": -30.0,
                    "conf": 99.0,
                }
            )
        finally:
            follow._follow_motion_config = old_follow_motion_config

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")

    def _reading_for_gap(self, *, dist_gap_mm: float, x_gap_mm: float) -> dict:
        y_cfg = follow._follow_y_axis_config()
        return {
            "dist_mm": follow._dist_target_mm() + float(dist_gap_mm),
            "x_mm": follow._x_target_mm() + float(x_gap_mm),
            "y_mm": float(y_cfg.get("win_target_mm", 0.0)),
            "confidence": 0.99,
        }

    def test_extreme_far_low_confident_reading_hits_virtual_wall_before_pickup_stop(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": 501.0,
            "x_mm": follow._x_target_mm(),
            "y_mm": -91.0,
            "conf": 99.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertTrue(follow._pickup_suspected_reading(reading))
        self.assertEqual(plan["kind"], "wait")
        self.assertEqual(plan["action"], "VIRTUAL_WALL_STOP")
        self.assertEqual(plan["reason"], "virtual_safety_dist_exceeded")

    def test_shifted_visible_low_reading_is_recoverable_not_pickup_suspect(self):
        reading = {
            "visible": True,
            "confident": True,
            "dist_mm": 299.5,
            "x_mm": follow._x_target_mm(),
            "y_mm": -60.0,
            "conf": 99.0,
        }

        plan = follow._follow_action_plan(reading)

        self.assertFalse(follow._pickup_suspected_reading(reading))
        self.assertNotEqual(plan.get("reason"), "pickup_suspected_far_low")

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
        follow._set_game_profile("holding")
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=100.0, x_gap_mm=11.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["drive_mode"], "forward")
        self.assertEqual(plan["strength"], "strong")
        self.assertEqual(plan["reason"], "holding_s1_close_dist_and_x_forward_curve")
        self.assertFalse(plan.get("use_production_turn_curve", False))

    def test_empty_close_range_width_conflict_does_not_reverse(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 79.1,
                "x_mm": -3.5,
                "y_mm": -24.5,
                "conf": 95.0,
                "vision_geometry_source": "green_edge_close_range_width",
            }
        )

        self.assertEqual(plan["kind"], "wait")
        self.assertEqual(plan["action"], "VISION_CLOSE_RANGE_CONFLICT_STOP")
        self.assertNotEqual(plan.get("cmd"), "b")

    def test_virtual_safety_wall_stops_follow_plan(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 301.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": -24.5,
                "conf": 95.0,
            }
        )

        self.assertEqual(plan["kind"], "wait")
        self.assertEqual(plan["action"], "VIRTUAL_WALL_STOP")
        self.assertEqual(plan["reason"], "virtual_safety_dist_exceeded")

    def test_reverse_gap_closing_plan_is_blocked_before_execution(self):
        plan = {
            "kind": "drive_bias",
            "cmd": "b",
            "drive_mode": "backward",
            "action": "BIAS_L_GENTLE_BACKOFF",
            "dist_err": follow._win_effective_tolerance(follow._dist_tol_mm()) + 10.0,
            "x_err": 3.7,
        }

        blocked = follow._block_reverse_gap_closing_plan(plan)

        self.assertEqual(blocked["kind"], "wait")
        self.assertEqual(blocked["action"], "REVERSE_GAP_CLOSING_BLOCKED")
        self.assertEqual(blocked["reason"], "reverse_gap_closing_blocked")
        self.assertEqual(blocked["blocked_plan"]["action"], "BIAS_L_GENTLE_BACKOFF")

    def test_reverse_bias_is_allowed_when_too_close_to_stack(self):
        plan = {
            "kind": "drive_bias",
            "cmd": "b",
            "drive_mode": "backward",
            "action": "BIAS_L_STRONG",
            "dist_err": -60.0,
            "x_err": 19.0,
        }

        self.assertIs(follow._block_reverse_gap_closing_plan(plan), plan)

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

    def test_gross_low_y_gap_holds_without_mast_up_when_x_is_centered(self):
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

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_uses_sharp_x_only_curve_when_dist_gap_is_tiny(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=0.0, x_gap_mm=20.0)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertEqual(plan["reason"], "empty_s1_dist_ok_x_only_no_forward_overshoot_one_wheel_500ms")
        self.assertTrue(plan["one_wheel_x_nudge"])
        self.assertNotIn("BACKOFF", plan["action"])

    def test_holding_dist_ok_x_gap_uses_forward_curve(self):
        follow._set_game_profile("holding")

        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=0.0, x_gap_mm=20.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["drive_mode"], "forward")
        self.assertEqual(plan["reason"], "holding_s1_dist_ok_x_forward_curve")
        self.assertNotIn("BACKOFF", plan["action"])

    def test_holding_x_noise_margin_does_not_widen_step1_win_gate(self):
        follow._set_game_profile("holding")
        reading = self._reading_for_gap(
            dist_gap_mm=0.0,
            x_gap_mm=follow._x_tol_mm() + 1.8,
        )

        plan = follow._follow_action_plan(reading, virtual_safety_armed=False)

        self.assertFalse(follow._step1_dist_x_target_ready(reading))
        self.assertNotEqual(plan.get("action"), "HAPPY")
        self.assertNotEqual(plan.get("kind"), "hold")

    def test_holding_dist_noise_margin_does_not_widen_step1_win_gate(self):
        follow._set_game_profile("holding")
        reading = self._reading_for_gap(
            dist_gap_mm=follow._dist_tol_mm() + 1.0,
            x_gap_mm=0.0,
        )

        plan = follow._follow_action_plan(reading, virtual_safety_armed=False)

        self.assertFalse(follow._step1_dist_x_target_ready(reading))
        self.assertNotEqual(plan.get("action"), "HAPPY")
        self.assertNotEqual(plan.get("kind"), "hold")

    def test_empty_too_close_x_bias_uses_bounded_back_recovery(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=-54.0, x_gap_mm=-40.0)
        )

        self.assertIn(plan["kind"], {"drive", "drive_bias"})
        self.assertEqual(plan["cmd"], "b")
        if plan["kind"] == "drive_bias":
            self.assertEqual(plan["drive_mode"], "backward")
        self.assertGreaterEqual(plan["duration_ms"], follow.EMPTY_S1_CURVE_BREAKAWAY_MIN_MS)
        self.assertLessEqual(plan["duration_ms"], 700)

    def test_drive_bias_forward_gentle_and_strong_use_leia_forward_polarity(self):
        gentle = follow._turn_bias_curve_for_drive_mode("forward", "gentle")
        strong = follow._turn_bias_curve_for_drive_mode("forward", "strong")

        gentle_actions = follow._turn_bias_actions(
            drive_mode="forward",
            turn_cmd="l",
            curve=gentle,
        )
        strong_actions = follow._turn_bias_actions(
            drive_mode="forward",
            turn_cmd="l",
            curve=strong,
        )

        self.assertEqual([row["action"] for row in gentle_actions], ["b", "f"])
        self.assertEqual([row["action"] for row in strong_actions], ["b", "f"])
        self.assertEqual(gentle_actions[0]["pwm"], gentle["inner_pwm"])
        self.assertEqual(gentle_actions[1]["pwm"], gentle["outer_pwm"])
        self.assertEqual(strong_actions[0]["pwm"], strong["inner_pwm"])
        self.assertEqual(strong_actions[1]["pwm"], strong["outer_pwm"])
        self.assertGreater(strong["outer_pwm"], gentle["outer_pwm"])

    def test_drive_bias_medium_uses_validated_duty_curve_ticks(self):
        reading = self._reading_for_gap(dist_gap_mm=35.0, x_gap_mm=-18.0)
        reading.update({"visible": True, "confident": True, "conf": 90.0, "min_confidence_pct": 75.0})
        robot = _FakeRobot()

        send_result = follow._send_drive_bias(
            robot,
            turn_cmd="r",
            drive_mode="forward",
            strength="medium",
            duration_ms=300,
            reading=reading,
            context="unit",
        )

        self.assertEqual(len(robot.custom_commands), 2)
        self.assertEqual(send_result["duty_curve"]["curve_label"], "fr_medium")
        first_actions = robot.custom_commands[0][1]
        second_actions = robot.custom_commands[1][1]
        self.assertEqual([row["action"] for row in first_actions], ["b", "f"])
        self.assertEqual([row["action"] for row in second_actions], ["b", "s"])
        self.assertGreater(first_actions[0]["pwm"], 0)
        self.assertGreater(first_actions[1]["pwm"], 0)
        self.assertGreater(second_actions[0]["pwm"], 0)
        self.assertEqual(second_actions[1]["pwm"], 0)
        self.assertEqual(first_actions[0]["pwm"], 104)
        self.assertEqual(first_actions[1]["pwm"], 104)
        self.assertEqual(second_actions[0]["pwm"], 104)

    def test_wheel_pwm_scaler_caps_to_crawl_ceiling(self):
        self.assertEqual(follow._wheel_pwm_ceiling(), 104)
        self.assertLessEqual(follow._scaled_pwm_for_cmd("f", 160), 104)
        self.assertLessEqual(follow._scaled_pwm_for_cmd("b", 160), 104)

    def test_duty_drive_bias_packets_are_capped_to_crawl_speed(self):
        reading = self._reading_for_gap(dist_gap_mm=35.0, x_gap_mm=28.0)
        reading.update({"visible": True, "confident": True, "conf": 90.0, "min_confidence_pct": 75.0})
        robot = _FakeRobot()

        follow._send_drive_bias(
            robot,
            turn_cmd="r",
            drive_mode="forward",
            strength="strong",
            duration_ms=300,
            reading=reading,
            context="unit",
        )

        self.assertTrue(robot.custom_commands)
        for _cmd, actions, _duration_ms in robot.custom_commands:
            for action in actions:
                if str(action.get("action")) in {"f", "b", "l", "r"}:
                    self.assertLessEqual(int(action.get("pwm", 0)), 104)

    def test_duty_turn_curve_config_excludes_failed_tank_curves(self):
        cfg = follow._duty_turn_curve_config()

        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["tick_ms"], 150)
        self.assertEqual(cfg["ramp_pwm_min"], 104)
        self.assertEqual(cfg["ramp_pwm_max"], 104)
        self.assertEqual(set(cfg["strengths"]), {"gentle", "medium", "strong", "superstrong"})
        self.assertNotIn("tank", cfg["strengths"])
        self.assertEqual(cfg["strengths"]["gentle"]["pwm"], 104)
        self.assertEqual(cfg["strengths"]["medium"]["pwm"], 104)
        self.assertEqual(cfg["strengths"]["strong"]["pwm"], 104)
        self.assertEqual(cfg["strengths"]["superstrong"]["pwm"], 115)
        self.assertEqual(cfg["strengths"]["superstrong"]["pwm_ceiling"], 115)

    def test_superstrong_duty_curve_uses_calibrated_extra_umph(self):
        reading = self._reading_for_gap(dist_gap_mm=45.0, x_gap_mm=80.0)
        reading.update({"visible": True, "confident": True, "conf": 95.0, "min_confidence_pct": 75.0})
        robot = _FakeRobot()

        follow._send_drive_bias(
            robot,
            turn_cmd="r",
            drive_mode="forward",
            strength="superstrong",
            duration_ms=300,
            reading=reading,
            context="unit",
        )

        self.assertTrue(robot.custom_commands)
        for _cmd, actions, _duration_ms in robot.custom_commands:
            outside = [row for row in actions if str(row.get("target")) == "l"]
            self.assertEqual(len(outside), 1)
            self.assertEqual(int(outside[0].get("pwm", 0)), 115)

    def test_empty_far_big_positive_x_uses_right_superstrong_learning_curve(self):
        reading = self._reading_for_gap(dist_gap_mm=45.0, x_gap_mm=80.0)
        reading.update({"visible": True, "confident": True, "conf": 95.0, "min_confidence_pct": 75.0})

        plan = follow._follow_action_plan(reading, virtual_safety_armed=False)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["turn_cmd"], "r")
        self.assertEqual(plan["strength"], "superstrong")
        self.assertEqual(plan["duration_ms"], follow.EMPTY_S1_LEARNING_CURVE_MS)
        self.assertTrue(plan["skip_near_target_crawl_cap"])
        self.assertTrue(plan["use_calibrated_turn_drive_curve"])

    def test_virtual_wall_recovery_with_big_x_curves_instead_of_straight(self):
        reading = self._reading_for_gap(dist_gap_mm=90.0, x_gap_mm=80.0)
        reading.update({"visible": True, "confident": True, "conf": 95.0, "min_confidence_pct": 75.0})

        plan = follow._follow_action_plan(reading, virtual_safety_armed=False)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["action"], "VIRTUAL_WALL_RECOVERY_BIAS_R_SUPERSTRONG")
        self.assertEqual(plan["turn_cmd"], "r")
        self.assertEqual(plan["strength"], "superstrong")
        self.assertEqual(plan["duration_ms"], follow.EMPTY_S1_LEARNING_CURVE_MS)

    def test_near_target_dist_micro_uses_micro_drive_floor(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=13.0, x_gap_mm=0.0)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["action"], "FWD_MICRO")
        self.assertEqual(plan["duration_ms"], follow._follow_dist_approach_policy()["micro_nudge_min_effective_pulse_ms"])
        self.assertGreaterEqual(plan["pwm"], follow._pwm_floor_for_cmd("f"))

    def test_empty_step1_crawl_to_floor_avoids_twitch_pulses(self):
        self.assertGreaterEqual(
            follow._empty_step1_crawl_to_win_floor_ms(follow._dist_tol_mm() + 2.0),
            follow.EMPTY_S1_CURVE_BREAKAWAY_MIN_MS,
        )
        self.assertGreaterEqual(
            follow._empty_step1_crawl_to_win_floor_ms(follow._dist_tol_mm() + 12.0),
            follow.EMPTY_S1_CURVE_BREAKAWAY_MIN_MS,
        )

    def test_empty_step1_near_floor_x_curve_uses_breakaway_packet(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=follow._dist_tol_mm() + 6.0, x_gap_mm=follow._x_tol_mm() + 5.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertGreaterEqual(plan["duration_ms"], follow.EMPTY_S1_CURVE_BREAKAWAY_MIN_MS)
        self.assertEqual(plan["reason"], "empty_s1_crawl_to_win_floor_x_learning_curve")

    def test_empty_step1_dist_ok_x_cleanup_uses_x_only_nudge(self):
        x_gap = follow._x_tol_mm() + 12.0
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=8.0, x_gap_mm=x_gap)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["reason"], "empty_s1_dist_ok_x_only_no_forward_overshoot_one_wheel_500ms")
        self.assertTrue(plan["one_wheel_x_nudge"])

    def test_tiny_x_outside_gap_inside_distance_band_does_not_win(self):
        x_gap = follow._x_tol_mm() + 1.0
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=25.0, x_gap_mm=x_gap)
        )

        self.assertNotEqual(plan["kind"], "hold")
        self.assertNotEqual(plan.get("action"), "HAPPY")

    def test_tiny_x_only_gap_inside_distance_band_does_not_win(self):
        x_gap = follow._x_tol_mm() + 1.0
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=0.0, x_gap_mm=x_gap)
        )

        self.assertNotEqual(plan["kind"], "hold")
        self.assertNotEqual(plan.get("action"), "HAPPY")

    def test_empty_step1_near_target_wide_x_uses_x_only_nudge(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=8.0, x_gap_mm=follow._x_tol_mm() + 12.0)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["reason"], "empty_s1_dist_ok_x_only_no_forward_overshoot_one_wheel_500ms")
        self.assertTrue(plan["one_wheel_x_nudge"])

    def test_empty_step1_in_band_x_cleanup_uses_x_only_not_forward_curve(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=-5.6, x_gap_mm=follow._x_tol_mm() + 8.0)
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["cmd"], "r")
        self.assertEqual(plan["reason"], "empty_s1_dist_ok_x_only_no_forward_overshoot_one_wheel_500ms")
        self.assertTrue(plan["one_wheel_x_nudge"])

    def test_dist_pingpong_guard_ignores_center_crossing_inside_happy_band(self):
        stats = {
            "last_distance_act_dist_err": 0.9,
            "last_distance_act_action": "BIAS_R_SUPERSTRONG_IN_BAND",
            "last_distance_act_cmd": "f",
        }
        plan = {
            "kind": "drive_bias",
            "cmd": "f",
            "distance_creep": True,
            "dist_err": -1.4,
        }

        self.assertFalse(follow._should_stop_confirm_for_dist_pingpong(stats, plan))

    def test_empty_step1_too_close_backs_up_to_reenter_band(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=-60.0, x_gap_mm=0.0)
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "b")
        self.assertEqual(plan["action"], "BCK_TOO_CLOSE")
        self.assertGreaterEqual(plan["duration_ms"], follow.EMPTY_S1_CURVE_BREAKAWAY_MIN_MS)
        self.assertLessEqual(plan["duration_ms"], 700)
        self.assertEqual(plan["reason"], "empty_s1_too_close_back_to_band")

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

    def test_empty_step1_wrong_way_uses_four_act_average_not_one_mm_jitter(self):
        stats = follow._new_game_stats()
        for index in range(4):
            stats["pending_observation"] = {
                "action": "BIAS_R_SUPERSTRONG_IN_BAND",
                "cmd": "f",
                "dist_err": 20.0,
                "x_err": 14.0,
                "dist_mm": follow._dist_target_mm() + 20.0,
                "x_mm": follow._x_target_mm() + 14.0,
                "y_mm": -30.0,
                "duration_ms": 170,
            }
            result = follow._record_observed_after_pending_act(
                stats,
                {
                    "dist_mm": follow._dist_target_mm() + 21.0 + (0.1 * index),
                    "x_mm": follow._x_target_mm() + 15.0 + (0.1 * index),
                    "y_mm": -30.0,
                },
            )

            self.assertIsNotNone(result)

        self.assertFalse(stats.get("step1_confident_worse_hard_stop", False))
        self.assertEqual(stats.get("step1_consecutive_worse_act_count"), 0)
        detail = stats.get("step1_last_worse_act_detail")
        self.assertIsInstance(detail, dict)
        self.assertLess(detail["avg_dist_regression_mm"], follow.NOISE_MARGIN_MM)
        self.assertLess(detail["avg_x_regression_mm"], follow.NOISE_MARGIN_MM)

    def test_empty_step1_wrong_way_requires_four_act_average_beyond_noise(self):
        stats = follow._new_game_stats()
        for _index in range(6):
            stats["pending_observation"] = {
                "action": "BIAS_R_SUPERSTRONG_IN_BAND",
                "cmd": "f",
                "dist_err": 20.0,
                "x_err": 14.0,
                "dist_mm": follow._dist_target_mm() + 20.0,
                "x_mm": follow._x_target_mm() + 14.0,
                "y_mm": -30.0,
                "duration_ms": 170,
            }
            follow._record_observed_after_pending_act(
                stats,
                {
                    "dist_mm": follow._dist_target_mm() + 26.0,
                    "x_mm": follow._x_target_mm() + 20.0,
                    "y_mm": -30.0,
                },
            )

        self.assertTrue(stats.get("step1_confident_worse_hard_stop"))
        detail = stats.get("step1_last_worse_act_detail")
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail["trend_window"], follow.STEP1_WRONG_WAY_TREND_WINDOW)
        self.assertGreaterEqual(detail["avg_dist_regression_mm"], follow.NOISE_MARGIN_MM)
        self.assertGreaterEqual(detail["avg_x_regression_mm"], follow.NOISE_MARGIN_MM)

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
        self.assertGreaterEqual(len(robot.custom_commands), 1)
        boosted_actions = [
            row
            for _cmd, actions, _duration_ms in robot.custom_commands
            for row in actions
        ]
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

    def test_step2_precision_lowers_high_mast_before_distance(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 1,
            "precision_hard_max_attempts": 1,
            "precision_settle_s": 0.0,
            "precision_drive_min_pulse_ms": 80,
            "precision_drive_max_pulse_ms": 180,
            "precision_mast_pulse_ms": 250,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 5.0,
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
            "dist_mm": 180.0,
            "x_mm": 4.0,
            "y_mm": -28.0,
        }
        after = dict(before, y_mm=-36.2)
        readings = iter([after])
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

        self.assertEqual(counts["fwd"], 0)
        self.assertEqual(counts["mast_d"], 1)
        self.assertEqual(final["y_mm"], -36.2)
        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "d")

    def test_step2_precision_raises_low_mast_before_distance(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 1,
            "precision_hard_max_attempts": 1,
            "precision_settle_s": 0.0,
            "precision_mast_pulse_ms": 250,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 5.0,
                "x_mm": 5.9,
                "x_tol_mm": 3.0,
                "y_mm": -42.7,
                "y_tol_mm": 5.0,
            },
        }
        before = {
            "visible": True,
            "confident": True,
            "conf": 95.0,
            "dist_mm": 202.7,
            "x_mm": 5.9,
            "y_mm": -59.6,
        }
        after = dict(before, y_mm=-48.0)
        readings = iter([after])
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

        self.assertEqual(counts["mast_u"], 1)
        self.assertEqual(counts["bck"], 0)
        self.assertEqual(final["y_mm"], -48.0)
        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "u")

    def test_step2_precision_distance_only_ignores_low_y(self):
        step2 = {
            "precision_settle_enabled": True,
            "precision_max_attempts": 3,
            "precision_hard_max_attempts": 5,
            "precision_settle_s": 0.0,
            "precision_drive_min_pulse_ms": 80,
            "precision_drive_max_pulse_ms": 180,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": 149.0,
                "dist_tol_mm": 5.0,
                "x_mm": None,
                "x_tol_mm": None,
                "y_mm": None,
                "y_tol_mm": None,
            },
        }
        before = {
            "visible": True,
            "confident": True,
            "conf": 95.0,
            "dist_mm": 168.0,
            "x_mm": 6.7,
            "y_mm": -40.0,
        }
        after = dict(before, dist_mm=149.0, y_mm=-36.2)
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

        self.assertEqual(counts["fwd"], 1)
        self.assertEqual(counts["mast_u"], 0)
        self.assertEqual(final["dist_mm"], 149.0)
        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(len(robot.custom_commands), 0)

    def test_empty_step3_is_distance_only_slow_crawl(self):
        follow._set_game_profile("empty")

        cfg = follow._follow_step3_config()
        targets = cfg["targets"]

        self.assertEqual(cfg["nickname"], "crawl dist")
        self.assertFalse(cfg["recovery_creep_enabled"])
        self.assertEqual(cfg["precision_drive_max_pulse_ms"], 85)
        self.assertEqual(cfg["precision_drive_min_pulse_ms"], 45)
        self.assertEqual(cfg["precision_max_attempts"], 30)
        self.assertIsNotNone(targets["dist_mm"])
        self.assertEqual(targets["dist_tol_mm"], 5.0)
        self.assertIsNone(targets["x_mm"])
        self.assertIsNone(targets["x_tol_mm"])
        self.assertIsNone(targets["y_mm"])
        self.assertIsNone(targets["y_tol_mm"])

    def test_empty_step2_requires_x_y_alignment_with_distance(self):
        follow._set_game_profile("empty")

        cfg = follow._follow_step2_config()
        targets = cfg["targets"]

        self.assertEqual(cfg["nickname"], "close dist+x+y")
        self.assertEqual(targets["dist_mm"], 149.0)
        self.assertEqual(targets["dist_tol_mm"], 5.0)
        self.assertEqual(targets["x_mm"], follow._x_target_mm())
        self.assertEqual(targets["x_tol_mm"], follow._x_tol_mm())
        self.assertEqual(targets["y_mm"], -18.0)
        self.assertEqual(targets["y_tol_mm"], 5.0)

        ready, _reason, closeness = follow._step2_targets_ready(
            {
                "visible": True,
                "confident": True,
                "conf": 95.0,
                "dist_mm": 149.0,
                "x_mm": 15.0,
                "y_mm": targets["y_mm"],
            },
            cfg,
        )

        self.assertFalse(ready)
        self.assertIsInstance(closeness, dict)
        self.assertLess(closeness["x_target_closeness_pct"], 100.0)

        ready, _reason, closeness = follow._step2_targets_ready(
            {
                "visible": True,
                "confident": True,
                "conf": 95.0,
                "dist_mm": 149.0,
                "x_mm": targets["x_mm"],
                "y_mm": -42.7,
            },
            cfg,
        )

        self.assertFalse(ready)
        self.assertIsInstance(closeness, dict)
        self.assertLess(closeness["y_target_closeness_pct"], 100.0)

    def test_step2_x_polish_uses_committed_turn_when_x_is_far(self):
        old_curve = follow._production_turn_curve_for_reading
        try:
            follow._production_turn_curve_for_reading = lambda **_kwargs: (
                {"curve_name": "test", "curve_value_mm": 14.0},
                620,
            )

            plan = follow._x_only_turn_plan(
                reading={"dist_mm": 149.0, "x_mm": 20.0, "y_mm": -32.0},
                turn_cmd="r",
                drive_mode="backward",
                strength="micro",
                dist_err=0.0,
                x_err=14.0,
                x_outside=11.0,
                dist_outside=0.0,
                y_plan=None,
                reason="step2_precision_x_polish",
                use_production_curve=True,
            )
        finally:
            follow._production_turn_curve_for_reading = old_curve

        self.assertGreaterEqual(plan["duration_ms"], 300)

    def test_empty_reset_distance_matches_step1_target(self):
        follow._set_game_profile("empty")

        reset_cfg = follow._reset_motion_config()["reverse_turn"]

        self.assertEqual(reset_cfg["dist_target_mm"], follow._dist_target_mm())

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
            "y_mm": -91.0,
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
            "y_mm": -91.0,
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
            "y_mm": -91.0,
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

    def test_empty_step1_predictive_momentum_continues_straight_when_curve_would_enter_band(self):
        stats = follow._new_game_stats()
        stats["last_x_momentum"] = {
            "after_x_err": 18.0,
            "x_err_reduction_mm": 8.0,
            "turn_cmd": "r",
            "strength": "superstrong",
        }
        plan = {
            "kind": "drive_bias",
            "cmd": "f",
            "turn_cmd": "r",
            "drive_mode": "forward",
            "strength": "superstrong",
            "action": "BIAS_R_SUPERSTRONG",
            "dist_err": 35.0,
            "x_err": 11.0,
            "duration_ms": follow.EMPTY_S1_LEARNING_CURVE_MS,
        }

        out = follow._apply_empty_step1_predictive_x_momentum(
            stats,
            plan,
            self._reading_for_gap(dist_gap_mm=35.0, x_gap_mm=11.0),
        )

        self.assertEqual(out["kind"], "drive")
        self.assertEqual(out["cmd"], "f")
        self.assertEqual(out["action"], "PREDICTIVE_X_BRAKE_FWD")
        self.assertIsNone(stats["last_x_momentum"])

    def test_empty_step1_predictive_momentum_downshifts_superstrong_before_overcorrection(self):
        stats = follow._new_game_stats()
        stats["last_x_momentum"] = {
            "after_x_err": 35.0,
            "x_err_reduction_mm": 8.0,
            "turn_cmd": "r",
            "strength": "superstrong",
        }
        plan = {
            "kind": "drive_bias",
            "cmd": "f",
            "turn_cmd": "r",
            "drive_mode": "forward",
            "strength": "superstrong",
            "action": "BIAS_R_SUPERSTRONG",
            "dist_err": 35.0,
            "x_err": 25.0,
            "duration_ms": follow.EMPTY_S1_LEARNING_CURVE_MS,
        }

        out = follow._apply_empty_step1_predictive_x_momentum(
            stats,
            plan,
            self._reading_for_gap(dist_gap_mm=35.0, x_gap_mm=25.0),
        )

        self.assertEqual(out["kind"], "drive_bias")
        self.assertEqual(out["strength"], "medium")
        self.assertEqual(out["action"], "BIAS_R_MEDIUM")
        self.assertEqual(out["reason"], "empty_s1_predictive_x_momentum_downshift")
        self.assertIsNone(stats["last_x_momentum"])

    def test_empty_step1_predictive_dist_momentum_shortens_forward_act(self):
        stats = follow._new_game_stats()
        stats["last_dist_momentum"] = {
            "cmd": "f",
            "after_dist_err": 25.0,
            "dist_err_reduction_mm": 30.0,
            "duration_ms": 250,
        }
        plan = {
            "kind": "drive",
            "cmd": "f",
            "action": "FWD",
            "dist_err": 20.0,
            "x_err": 0.0,
            "duration_ms": 250,
            "distance_creep": True,
            "reason": "dist_only_creep",
        }

        out = follow._apply_empty_step1_predictive_dist_momentum(
            stats,
            plan,
            self._reading_for_gap(dist_gap_mm=20.0, x_gap_mm=0.0),
        )

        self.assertEqual(out["kind"], "drive")
        self.assertEqual(out["cmd"], "f")
        self.assertLess(out["duration_ms"], plan["duration_ms"])
        self.assertEqual(out["reason"], "dist_only_creep_predictive_dist_brake")
        self.assertIsNone(stats["last_dist_momentum"])

    def test_empty_step1_settles_after_large_x_bias_before_straight_forward(self):
        stats = follow._new_game_stats()
        stats["last_x_momentum"] = {
            "action": "BIAS_R_SUPERSTRONG",
            "turn_cmd": "r",
            "strength": "superstrong",
            "drive_mode": "forward",
            "before_x_err": 61.4,
            "after_x_err": 23.3,
            "abs_before_x_err": 61.4,
            "abs_after_x_err": 23.3,
            "x_err_reduction_mm": 38.1,
            "duration_ms": 320,
        }
        plan = {
            "kind": "drive",
            "cmd": "f",
            "action": "FWD",
            "dist_err": 93.9,
            "x_err": 4.2,
            "duration_ms": 270,
            "distance_creep": True,
            "reason": "dist_only_creep",
        }

        out = follow._apply_empty_step1_x_bias_settle_gate(
            stats,
            plan,
            self._reading_for_gap(dist_gap_mm=93.9, x_gap_mm=4.2),
        )

        self.assertEqual(out["kind"], "wait")
        self.assertEqual(out["action"], "EMPTY_S1_X_BIAS_SETTLE_READ")
        self.assertEqual(out["reason"], "empty_s1_settle_after_large_x_bias_before_next_commit")
        self.assertIs(follow._apply_empty_step1_x_bias_settle_gate(stats, plan, {}), plan)

    def test_empty_step1_near_floor_large_x_caps_superstrong_to_strong(self):
        plan = follow._follow_action_plan(
            self._reading_for_gap(dist_gap_mm=follow._dist_tol_mm() + 6.0, x_gap_mm=follow._x_tol_mm() + 24.0)
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["strength"], "strong")
        self.assertEqual(plan["reason"], "empty_s1_crawl_to_win_floor_x_learning_curve")

    def test_empty_step1_post_action_wait_includes_learning_settle(self):
        plan = {
            "kind": "drive",
            "cmd": "f",
            "duration_ms": 100,
            "distance_creep": True,
        }

        wait_s = follow._post_action_wait_s(plan, {"duration_ms": 100})

        self.assertGreaterEqual(wait_s, 0.1 + follow.EMPTY_S1_LEARNING_POST_ACT_SETTLE_S)


if __name__ == "__main__":
    unittest.main()
