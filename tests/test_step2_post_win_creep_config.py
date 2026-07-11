import unittest
from unittest import mock

import a_follow_the_brick as follow


class TestStep2PostWinCreepConfig(unittest.TestCase):
    def test_empty_profile_preserves_post_win_forward_creep(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()

        self.assertEqual(cfg.get("post_win_forward_creep_ms"), 940)
        self.assertEqual(cfg.get("post_win_forward_creep_pwm"), 103)
        self.assertFalse(cfg.get("post_win_forward_creep_read_after"))

    def test_holding_profile_preserves_pre_place_forward_creep(self):
        follow._set_game_profile("holding")
        cfg = follow._follow_step2_config()

        self.assertTrue(cfg.get("blind_mast_only"))
        self.assertEqual(cfg.get("seat_mast_duration_ms"), 3773)
        self.assertEqual(cfg.get("pre_place_forward_creep_ms"), 1400)
        self.assertEqual(cfg.get("pre_place_forward_creep_pwm"), 103)

    def test_holding_step2_nudges_forward_before_blind_lower(self):
        follow._set_game_profile("holding")
        order = []
        nudge_send = {"cmd_sent": "f", "duration_ms": 1400, "pwm": 103}

        def fake_nudge(_vision, _robot, reading, step2):
            order.append("nudge")
            self.assertEqual(step2.get("pre_place_forward_creep_ms"), 1400)
            self.assertEqual(step2.get("pre_place_forward_creep_pwm"), 103)
            return {"confident": True, "reason": "after_forward_nudge"}, nudge_send

        def fake_lower(_robot, *, reading=None, step_cfg=None, label="holding_s3_lower"):
            order.append("lower")
            self.assertEqual((reading or {}).get("reason"), "after_forward_nudge")
            self.assertTrue((step_cfg or {}).get("blind_mast_only"))
            self.assertEqual(label, "step2")
            return {
                "success": True,
                "target_met": True,
                "reason": "step2_blind_lower_no_post_read",
                "reading": reading,
            }

        with mock.patch.object(
            follow,
            "_run_holding_step1_forward_nudge",
            side_effect=fake_nudge,
        ), mock.patch.object(
            follow,
            "_run_holding_blind_lower_sequence",
            side_effect=fake_lower,
        ):
            result = follow._run_step2_seat_sequence(
                object(),
                object(),
                initial_reading={"confident": True, "reason": "holding_step1_win"},
            )

        self.assertEqual(order, ["nudge", "lower"])
        self.assertTrue(result["success"])
        self.assertEqual(result["pre_place_forward_creep_result"], nudge_send)
        self.assertEqual(result["pre_place_forward_creep_ms"], 1400)

    def test_holding_forward_nudge_sends_and_waits_for_configured_duration(self):
        follow._set_game_profile("holding")
        events = []

        def fake_send(_robot, cmd, pwm, *, duration_ms, reading, context):
            events.append(("send", cmd, pwm, duration_ms, context))
            return {"cmd_sent": cmd, "pwm": pwm, "duration_ms": duration_ms}

        with mock.patch.object(
            follow,
            "guarded_send_command_pwm",
            side_effect=fake_send,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=lambda seconds: events.append(("sleep", seconds)),
        ), mock.patch.object(
            follow,
            "_stop_robot",
            side_effect=lambda _robot: events.append(("stop",)),
        ):
            _reading, result = follow._run_holding_step1_forward_nudge(
                object(),
                object(),
                {"confident": True},
                follow._follow_step2_config(),
            )

        self.assertEqual(result["cmd_sent"], "f")
        self.assertEqual(
            events,
            [
                ("send", "f", 103, 1400, "reset_holding_step1_pre_place_forward_creep"),
                ("sleep", 1.4),
                ("stop",),
            ],
        )

    def test_holding_retreat_uses_full_mirrored_turn_reset(self):
        follow._set_game_profile("holding")
        cfg = follow._follow_step3_config()

        self.assertTrue(cfg.get("holding_blind_reset_enabled"))
        self.assertEqual(cfg.get("holding_blind_back_crawl_ms"), 3000)
        self.assertEqual(cfg.get("post_lift_back_crawl_ms"), 1500)
        self.assertEqual(cfg.get("post_lift_turn_ms"), 2160)
        self.assertEqual(cfg.get("holding_blind_turn_ms"), 1800)
        self.assertEqual(cfg.get("holding_blind_turn_pause_ms"), 500)

    def test_post_lift_reset_sequence_item_uses_holding_transition_reset(self):
        sequence, errors = follow._normalize_custom_sequence("post_lift_reset,s1")

        self.assertEqual(errors, [])
        self.assertEqual(sequence, ["holding_transition_reset", "step1"])

        follow._set_game_profile("empty")
        calls = []

        def fake_transition(_vision, _robot, step3):
            calls.append(
                {
                    "profile": follow._active_game_profile(),
                    "back_ms": step3.get("post_lift_back_crawl_ms"),
                    "turn_ms": step3.get("post_lift_turn_ms"),
                    "pause_ms": step3.get("holding_blind_turn_pause_ms"),
                }
            )
            return {
                "success": True,
                "soft_reset_complete": True,
                "reason": "post_lift_holding_transition_mirrored_turn_complete",
                "reading": {"confident": True, "conf": 100.0},
            }

        with mock.patch.object(
            follow,
            "_run_post_lift_holding_transition_reset",
            side_effect=fake_transition,
        ):
            result = follow._run_custom_sequence(
                object(),
                object(),
                ["holding_transition_reset"],
                duration_s=0.1,
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            calls,
            [{"profile": "holding", "back_ms": 1500, "turn_ms": 2160, "pause_ms": 500}],
        )

    def test_post_lift_transition_backs_away_before_full_random_turn_and_mirror(self):
        events = []

        def fake_back(_robot, cmd, pwm, *, duration_ms, reading, context):
            events.append(("back", cmd, duration_ms, context))
            return {"cmd_sent": cmd, "pwm": pwm, "duration_ms": duration_ms}

        def fake_turn(_robot, logical_cmd, **kwargs):
            events.append(
                (
                    "turn",
                    logical_cmd,
                    kwargs["drive_mode"],
                    kwargs["turn_cmd"],
                    kwargs["duration_ms"],
                )
            )
            return {"cmd_sent": logical_cmd, "duration_ms": kwargs["duration_ms"]}

        with mock.patch.object(follow.random, "choice", return_value="l"), mock.patch.object(
            follow,
            "guarded_send_command_pwm",
            side_effect=fake_back,
        ), mock.patch.object(
            follow,
            "_send_duty_curve_sequence",
            side_effect=fake_turn,
        ), mock.patch.object(
            follow,
            "_stop_robot",
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=lambda seconds: events.append(("sleep", seconds)),
        ):
            result = follow._run_post_lift_holding_transition_reset(
                object(),
                object(),
                {
                    "post_lift_back_crawl_ms": 1500,
                    "post_lift_turn_ms": 2160,
                    "holding_blind_turn_pause_ms": 500,
                },
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["turn_cmd"], "l")
        self.assertEqual(result["mirror_turn_cmd"], "r")
        self.assertEqual(result["back_crawl_ms"], 1500)
        self.assertEqual(
            events,
            [
                ("back", "b", 1500, "reset_post_lift_holding_transition_back_crawl"),
                ("sleep", 1.5),
                ("turn", "b", "backward", "l", 2160),
                ("sleep", 0.5),
                ("turn", "f", "forward", "r", 2160),
            ],
        )

    def test_empty_profile_enables_step0_two_blind_turns(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step0_config()

        self.assertTrue(cfg.get("enabled"))
        self.assertEqual(cfg.get("nickname"), "2 blind turns")
        self.assertEqual(cfg.get("turn_ms"), 1500)
        self.assertEqual(cfg.get("pause_ms"), 0)
        self.assertEqual(cfg.get("strength"), "superstrong")
        self.assertEqual(cfg.get("first_drive_mode"), "backward")
        self.assertEqual(cfg.get("mirror_drive_mode"), "forward")

    def test_e2e_pre_empty_step1_crawls_right_until_visible(self):
        readings = [
            {"visible": False, "confident": False, "reason": "brick_not_confident"},
            {"visible": False, "confident": False, "reason": "brick_not_confident"},
            {
                "visible": True,
                "confident": True,
                "conf": 92.0,
                "dist_mm": 140.0,
                "x_mm": 25.0,
            },
        ]
        sends = []

        def fake_send(_robot, cmd, actions, *, duration_ms, reading, context):
            sends.append((cmd, actions, duration_ms, context))
            return {"cmd_sent": cmd, "duration_ms": duration_ms}

        with mock.patch.object(
            follow,
            "_read_brick_measurement",
            side_effect=readings,
        ), mock.patch.object(
            follow,
            "_gap_crawl_config",
            return_value={"turn_min_seg_ms": 220, "turn_pwm": 133},
        ), mock.patch.object(
            follow,
            "_pwm_floor_for_cmd",
            return_value=133,
        ), mock.patch.object(
            follow,
            "guarded_send_custom_actions_pwm",
            side_effect=fake_send,
        ), mock.patch.object(
            follow,
            "_stop_robot",
        ), mock.patch.object(
            follow,
            "_reset_follow_reading_history",
        ), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_e2e_pre_empty_step1_right_visibility_crawl(
                object(),
                object(),
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["camera_reads"], 2)
        self.assertEqual(result["refresh_count"], 1)
        self.assertEqual(len(sends), 1)
        self.assertTrue(all(send[0] == "r" and send[2] == 220 for send in sends))
        self.assertEqual(
            sends[0][1],
            [
                {"target": "l", "action": "b", "pwm": 133, "duration_ms": 220},
                {"target": "r", "action": "s", "pwm": 0, "duration_ms": 0},
            ],
        )

    def test_e2e_pre_empty_step1_right_crawl_has_five_second_cap(self):
        clock = {"now": 0.0}

        def fake_sleep(seconds):
            clock["now"] += float(seconds)

        with mock.patch.object(
            follow,
            "_read_brick_measurement",
            return_value={"visible": False, "confident": False, "reason": "brick_not_confident"},
        ), mock.patch.object(
            follow,
            "_gap_crawl_config",
            return_value={"turn_min_seg_ms": 220, "turn_pwm": 133},
        ), mock.patch.object(
            follow,
            "_pwm_floor_for_cmd",
            return_value=133,
        ), mock.patch.object(
            follow,
            "guarded_send_custom_actions_pwm",
            return_value={"cmd_sent": "r", "duration_ms": 220},
        ), mock.patch.object(
            follow,
            "_stop_robot",
        ), mock.patch.object(
            follow,
            "_reset_follow_reading_history",
        ), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=lambda: clock["now"],
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_sleep,
        ):
            result = follow._run_e2e_pre_empty_step1_right_visibility_crawl(
                object(),
                object(),
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "empty_step1_pre_visibility_right_crawl_timeout")
        self.assertEqual(result["max_duration_s"], 5.0)
        self.assertGreaterEqual(clock["now"], 5.0)

    def test_empty_lift_skips_post_lift_camera_pause(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step4_config()

        self.assertTrue(cfg.get("fixed_lift_only"))
        self.assertFalse(cfg.get("fixed_lift_read_after"))
        self.assertEqual(cfg.get("lift_settle_s"), 0.0)

    def test_e2e_empty_step2_noise_grace_can_handoff_to_lift(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        targets = cfg.get("targets")
        reading = {
            "confident": True,
            "dist_mm": targets.get("dist_mm"),
            "x_mm": float(targets.get("x_mm")) + float(targets.get("x_tol_mm")) + 1.0,
            "visible": True,
            "conf": 90.0,
        }
        result = {
            "success": True,
            "target_met": False,
            "reason": "step2_visibility_recovered_targets_scored",
            "reading": reading,
        }

        self.assertTrue(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_final_visibility_loss_assumes_ready_for_lift(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        result = {
            "success": True,
            "target_met": False,
            "reason": "step2_unconfirmed_no_final_visibility",
            "reading": {
                "visible": False,
                "confident": False,
                "reason": "brick_not_confident",
            },
        }

        self.assertTrue(follow._e2e_empty_step2_lost_only_final_visibility(result))
        self.assertTrue(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_target_hit_losing_stopped_confirmation_assumes_ready(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        result = {
            "success": True,
            "target_met": False,
            "reason": "step2_precision_target_hit_unconfirmed_after_stop",
            "reading": {
                "visible": False,
                "confident": False,
                "reason": "brick_not_confident",
            },
            "precision_counts": {
                "target_hit_stop": 1,
                "target_hit_confirm_failed": 1,
            },
        }

        self.assertTrue(follow._e2e_empty_step2_lost_only_final_visibility(result))
        self.assertTrue(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_measured_drift_after_stop_still_blocks_lift(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        result = {
            "success": True,
            "target_met": False,
            "reason": "step2_precision_target_drifted_after_stop:step2_x_outside",
            "reading": {"visible": True, "confident": True, "dist_mm": 120.0, "x_mm": 30.0},
        }

        self.assertFalse(follow._e2e_empty_step2_lost_only_final_visibility(result))
        self.assertFalse(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_other_unconfirmed_failure_still_blocks_lift(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        result = {
            "success": True,
            "target_met": False,
            "reason": "step2_step_timeout",
            "reading": {"visible": False, "confident": False},
        }

        self.assertFalse(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_blocked_creep_does_not_handoff_to_lift(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()
        targets = cfg.get("targets")
        reading = {
            "confident": True,
            "dist_mm": targets.get("dist_mm"),
            "x_mm": targets.get("x_mm"),
            "visible": True,
            "conf": 90.0,
        }
        result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": reading,
            "post_win_forward_creep_result": {"blocked": True, "reason": "virtual_safety_dist_exceeded"},
        }

        self.assertFalse(follow._e2e_empty_step2_ready_for_lift(result, cfg))

    def test_e2e_empty_step2_not_confirmed_action_uses_real_reason(self):
        action = follow._e2e_empty_step2_not_confirmed_action(
            {"reason": "step2_visibility_recovered_targets_scored"}
        )

        self.assertEqual(
            action,
            "EMPTY_STEP2_NOT_CONFIRMED:step2_visibility_recovered_targets_scored",
        )
        self.assertNotIn("NEEDS_WORK", action)

    def test_custom_empty_step2_assumes_final_visibility_loss_and_runs_pickup_nudge(self):
        follow._set_game_profile("empty")
        step2_result = {
            "success": True,
            "target_met": False,
            "reason": "step2_unconfirmed_no_final_visibility",
            "reading": {"visible": False, "confident": False},
        }
        creep_send = {"cmd_sent": "f", "duration_ms": 940, "pwm": 103}

        with mock.patch.object(
            follow,
            "_run_step2_seat_sequence",
            return_value=step2_result,
        ), mock.patch.object(
            follow,
            "_run_post_win_forward_creep",
            return_value=({"confident": True}, creep_send),
        ) as pickup_nudge, mock.patch.object(
            follow,
            "_format_game_results_table",
            return_value="results",
        ):
            result = follow._run_custom_sequence(
                object(),
                object(),
                ["step2"],
                duration_s=1.0,
            )

        self.assertTrue(result["success"])
        detail = result["items"][0]["result"]
        self.assertTrue(detail["target_met"])
        self.assertTrue(detail["custom_assumed_complete_after_final_visibility_loss"])
        self.assertEqual(detail["original_reason"], "step2_unconfirmed_no_final_visibility")
        self.assertEqual(detail["post_win_forward_creep_result"], creep_send)
        self.assertTrue(pickup_nudge.call_args.args[2]["confident"])

    def test_e2e_holding_step1_requires_recorded_wheel_motion_before_lower(self):
        no_motion_win = {"win_count": 1, "sent_act_counts": {}}
        moved_win = {"win_count": 1, "sent_act_counts": {"GAPCRAWL_FWD": 1}}

        self.assertTrue(follow._e2e_step1_complete("empty", no_motion_win))
        self.assertFalse(follow._e2e_step1_complete("holding", no_motion_win))
        self.assertTrue(follow._e2e_step1_complete("holding", moved_win))

    def test_e2e_holding_retreat_is_final_motion_without_extra_reset(self):
        step1_results = [
            {
                "win_count": 1,
                "follow_attempt_count": 1,
                "sent_act_counts": {"GAPCRAWL_FWD": 1},
                "last_step1_win": {
                    "visible": True,
                    "confident": True,
                    "conf": 92.0,
                    "dist_mm": 96.0,
                    "x_mm": 0.0,
                },
            },
            {
                "win_count": 1,
                "follow_attempt_count": 1,
                "sent_act_counts": {"GAPCRAWL_FWD": 1},
                "last_step1_win": {
                    "visible": True,
                    "confident": True,
                    "conf": 92.0,
                    "dist_mm": 90.0,
                    "x_mm": 0.0,
                },
            },
        ]
        step2_results = [
            {
                "success": True,
                "target_met": True,
                "reason": "step2_target_hit_confirmed",
                "reading": {"confident": True, "dist_mm": 88.0, "x_mm": 0.0},
            },
            {
                "success": True,
                "target_met": True,
                "reason": "step2_blind_lower_no_post_read",
                "reading": {"confident": True, "dist_mm": 90.0, "x_mm": 0.0},
            },
        ]

        with mock.patch.object(
            follow,
            "_run_empty_step0_two_blind_turns",
            return_value={"success": True, "target_met": True},
        ), mock.patch.object(
            follow,
            "_run_e2e_pre_empty_step1_right_visibility_crawl",
            return_value={"success": True, "attempts": 0},
        ), mock.patch.object(
            follow,
            "_follow_loop",
            side_effect=step1_results,
        ) as follow_loop, mock.patch.object(
            follow,
            "_format_game_results_table",
            return_value="results",
        ), mock.patch.object(
            follow,
            "_run_step2_seat_sequence",
            side_effect=step2_results,
        ), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value={"success": True, "target_met": True, "holding": True},
        ), mock.patch.object(
            follow,
            "_run_post_lift_holding_transition_reset",
            return_value={"success": True, "soft_reset_complete": True},
        ), mock.patch.object(
            follow,
            "_run_step3_retreat_sequence",
            return_value={"success": True, "soft_reset_complete": True},
        ), mock.patch.object(
            follow,
            "_run_reset_sequence",
        ) as extra_reset:
            result = follow._run_e2e_trial_gapcrawl_production(
                object(),
                object(),
                duration_s=1.0,
            )

        self.assertEqual(result["last_action"], "E2E_CYCLES_DONE")
        self.assertTrue(result["holding_retreat_is_final_reset"])
        self.assertFalse(follow_loop.call_args_list[0].kwargs["require_step1_motion_before_win"])
        self.assertTrue(follow_loop.call_args_list[1].kwargs["require_step1_motion_before_win"])
        extra_reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
