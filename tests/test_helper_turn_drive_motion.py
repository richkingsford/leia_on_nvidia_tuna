import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_turn_drive_motion
import telemetry_robot


class TestHelperTurnDriveMotion(unittest.TestCase):
    def test_build_forward_right_plan_uses_forward_arc_actions(self):
        plan = helper_turn_drive_motion.build_turn_drive_motion_plan(
            cmd="r",
            score=1,
            hold_duration_ms=120,
            pwm_override=200,
            profile_override={
                "drive_mode": "forward",
                "inner_ratio": 0.5,
                "outer_ratio": 1.0,
                "action_note": "TURN+FWD",
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["drive_mode"], "forward")
        self.assertEqual(plan["actions"], [
            {"target": "l", "action": "b", "pwm": 200},
            {"target": "r", "action": "f", "pwm": 100},
        ])

    def test_build_backward_left_plan_uses_backward_arc_actions(self):
        plan = helper_turn_drive_motion.build_turn_drive_motion_plan(
            cmd="l",
            score=1,
            hold_duration_ms=140,
            pwm_override=180,
            profile_override={
                "drive_mode": "backward",
                "inner_ratio": 0.25,
                "outer_ratio": 1.0,
                "action_note": "TURN+BWD",
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertEqual(plan["actions"], [
            {"target": "l", "action": "f", "pwm": 45},
            {"target": "r", "action": "b", "pwm": 180},
        ])

    def test_align_turn_drive_assist_request_selects_forward_profile_when_too_far(self):
        request = helper_turn_drive_motion.align_turn_drive_assist_request(
            {
                "align_policy": {
                    "x_axis_turn_drive_assist": {
                        "enabled": True,
                        "require_dist_outside_gate": True,
                        "forward_profile": "forward_pivot",
                        "backward_profile": "backward_pivot",
                    }
                }
            },
            cmd="l",
            local_gate_status={
                "visible": True,
                "dist_required": True,
                "dist_within_tol": False,
                "dist_err": 18.0,
                "dist": 130.0,
                "dist_target": 111.45,
            },
        )

        self.assertEqual(
            request,
            {"drive_mode": "forward", "profile_name": "forward_pivot"},
        )

    def test_align_turn_drive_assist_request_selects_backward_profile_when_too_close(self):
        request = helper_turn_drive_motion.align_turn_drive_assist_request(
            {
                "align_policy": {
                    "x_axis_turn_drive_assist": {
                        "enabled": True,
                        "require_dist_outside_gate": True,
                        "forward_profile": "forward_pivot",
                        "backward_profile": "backward_pivot",
                    }
                }
            },
            cmd="r",
            local_gate_status={
                "visible": True,
                "dist_required": True,
                "dist_within_tol": False,
                "dist_err": 6.0,
                "dist": 105.0,
                "dist_target": 111.45,
            },
        )

        self.assertEqual(
            request,
            {"drive_mode": "backward", "profile_name": "backward_pivot"},
        )

    def test_align_turn_drive_assist_request_skips_when_dist_is_already_in_gate(self):
        request = helper_turn_drive_motion.align_turn_drive_assist_request(
            {
                "align_policy": {
                    "x_axis_turn_drive_assist": {
                        "enabled": True,
                        "require_dist_outside_gate": True,
                        "forward_profile": "forward_pivot",
                        "backward_profile": "backward_pivot",
                    }
                }
            },
            cmd="r",
            local_gate_status={
                "visible": True,
                "dist_required": True,
                "dist_within_tol": True,
                "dist_err": 0.8,
                "dist": 111.2,
                "dist_target": 111.45,
            },
        )

        self.assertIsNone(request)

    def test_align_turn_drive_assist_request_requires_explicit_direction_profile(self):
        request = helper_turn_drive_motion.align_turn_drive_assist_request(
            {
                "align_policy": {
                    "x_axis_turn_drive_assist": {
                        "enabled": True,
                        "require_dist_outside_gate": True,
                        "forward_profile": "forward_pivot",
                    }
                }
            },
            cmd="r",
            local_gate_status={
                "visible": True,
                "dist_required": True,
                "dist_within_tol": False,
                "dist_err": 6.0,
                "dist": 105.0,
                "dist_target": 111.45,
            },
        )

        self.assertIsNone(request)

    def test_build_plan_can_use_drive_capable_duration_from_profile(self):
        turn_duration_ms = telemetry_robot.speed_power_pwm_for_cmd("r", 1)[3]
        drive_duration_ms = telemetry_robot.speed_power_pwm_for_cmd("f", 1)[3]

        plan = helper_turn_drive_motion.build_turn_drive_motion_plan(
            cmd="r",
            score=1,
            hold_duration_ms=None,
            profile_override={
                "drive_mode": "forward",
                "inner_ratio": 0.0,
                "outer_ratio": 1.0,
                "duration_mode": "max_turn_drive",
                "action_note": "TURN+FWD",
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["duration_mode"], "max_turn_drive")
        self.assertEqual(plan["duration_ms"], max(int(turn_duration_ms), int(drive_duration_ms)))


if __name__ == "__main__":
    unittest.main()
