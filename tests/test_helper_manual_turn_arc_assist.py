import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_manual_turn_arc_assist


class _DummyRobot:
    def __init__(self):
        self.calls = []

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        self.calls.append((cmd, list(actions), duration_ms))
        return {
            "cmd_sent": cmd,
            "wire_text": "custom",
            "pwm": max(int(action.get("pwm") or 0) for action in actions),
            "power": 1.0,
            "duration_ms": int(duration_ms or 0),
        }


class TestHelperManualTurnArcAssist(unittest.TestCase):
    def test_build_plan_for_micro_right_turn_stops_inner_tread(self):
        plan = helper_manual_turn_arc_assist.build_manual_turn_arc_plan(
            hotkey="e",
            cmd="r",
            score=1,
            hold_duration_ms=135,
            pwm_override=132,
            config={
                "enabled": True,
                "hotkey_profiles": {
                    "e": {"inner_ratio": 0.0, "outer_ratio": 1.0},
                },
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["outer_pwm"], 132)
        self.assertEqual(plan["inner_pwm"], 0)
        self.assertEqual(
            plan["actions"],
            [
                {"target": "l", "action": "b", "pwm": 132},
                {"target": "r", "action": "f", "pwm": 0},
            ],
        )

    def test_build_plan_for_right_arc_uses_outer_left_inner_right(self):
        plan = helper_manual_turn_arc_assist.build_manual_turn_arc_plan(
            hotkey="e",
            cmd="r",
            score=1,
            hold_duration_ms=120,
            pwm_override=200,
            config={
                "enabled": True,
                "hotkey_profiles": {
                    "e": {"inner_ratio": 0.5, "outer_ratio": 1.0},
                },
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["outer_pwm"], 200)
        self.assertEqual(plan["inner_pwm"], 100)
        self.assertEqual(
            plan["actions"],
            [
                {"target": "l", "action": "b", "pwm": 200},
                {"target": "r", "action": "f", "pwm": 100},
            ],
        )

    def test_build_plan_for_left_arc_uses_inner_left_outer_right(self):
        plan = helper_manual_turn_arc_assist.build_manual_turn_arc_plan(
            hotkey="a",
            cmd="l",
            score=25,
            hold_duration_ms=300,
            pwm_override=180,
            config={
                "enabled": True,
                "hotkey_profiles": {
                    "a": {"inner_ratio": 0.25, "outer_ratio": 1.0},
                },
            },
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(
            plan["actions"],
            [
                {"target": "l", "action": "b", "pwm": 45},
                {"target": "r", "action": "f", "pwm": 180},
            ],
        )

    def test_execute_plan_uses_robot_custom_action_send(self):
        robot = _DummyRobot()

        result = helper_manual_turn_arc_assist.execute_manual_turn_arc_plan(
            robot=robot,
            hotkey="e",
            cmd="r",
            score=1,
            hold_duration_ms=125,
            pwm_override=160,
            config={
                "enabled": True,
                "hotkey_profiles": {
                    "e": {"inner_ratio": 0.75, "outer_ratio": 1.0},
                },
            },
        )

        self.assertEqual(len(robot.calls), 1)
        cmd, actions, duration_ms = robot.calls[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(duration_ms, 125)
        self.assertEqual(actions[0]["pwm"], 160)
        self.assertEqual(actions[1]["pwm"], 120)
        self.assertEqual(result["cmd_sent"], "r")
        self.assertIn("manual_turn_arc_assist", result)


if __name__ == "__main__":
    unittest.main()
