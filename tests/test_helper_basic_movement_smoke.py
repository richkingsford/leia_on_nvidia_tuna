import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_basic_movement_smoke as smoke


class _FakeRobot:
    def __init__(self):
        self.calls = []

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.calls.append(("command_pwm", cmd, int(pwm), int(duration_ms or 0)))
        return {
            "cmd_sent": cmd,
            "pwm": int(pwm),
            "duration_ms": int(duration_ms or 0),
            "wire_text": f"{cmd}:{int(pwm)}:{int(duration_ms or 0)}",
        }

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        action_list = [dict(action) for action in actions]
        self.calls.append(("custom", cmd, action_list, int(duration_ms or 0)))
        return {
            "cmd_sent": cmd,
            "pwm": max(int(action.get("pwm") or 0) for action in action_list),
            "duration_ms": int(duration_ms or 0),
            "wire_text": "custom",
            "actions": action_list,
        }

    def stop(self):
        self.calls.append(("stop",))


class TestHelperBasicMovementSmoke(unittest.TestCase):
    def test_build_sequence_uses_old_slow_reference_hotkeys(self):
        sequence = smoke.build_basic_movement_sequence()

        self.assertEqual([pulse.reference_hotkey for pulse in sequence], ["r", "f", "q", "e", "o", "k"])
        self.assertEqual([pulse.cmd for pulse in sequence], ["f", "b", "l", "r", "u", "d"])
        self.assertTrue(all(pulse.score == 1 for pulse in sequence))
        self.assertTrue(all(pulse.pwm > 0 for pulse in sequence))
        self.assertTrue(all(pulse.duration_ms > 0 for pulse in sequence))

    def test_turn_reference_hotkeys_use_world_model_arc_profiles(self):
        sequence = smoke.build_basic_movement_sequence()
        q_pulse = sequence[2]
        e_pulse = sequence[3]

        self.assertEqual(q_pulse.send_mode, smoke.SEND_MODE_TURN_ARC)
        self.assertEqual(e_pulse.send_mode, smoke.SEND_MODE_TURN_ARC)
        self.assertEqual(
            q_pulse.turn_arc_plan["actions"],
            [
                {"target": "l", "action": "b", "pwm": 0},
                {"target": "r", "action": "f", "pwm": q_pulse.pwm},
            ],
        )
        self.assertEqual(
            e_pulse.turn_arc_plan["actions"],
            [
                {"target": "l", "action": "b", "pwm": e_pulse.pwm},
                {"target": "r", "action": "f", "pwm": 0},
            ],
        )

    def test_dry_run_does_not_require_robot_or_sleep(self):
        sleeps = []
        logs = []

        result = smoke.run_basic_movement_smoke_test(
            execute=False,
            sleep_fn=sleeps.append,
            log_fn=logs.append,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["execute"])
        self.assertEqual(len(result["pulses"]), 6)
        self.assertEqual(sleeps, [])
        self.assertTrue(any("[DRY-RUN]" in line for line in logs))

    def test_execute_sends_six_pulses_and_stops_between_moves(self):
        robot = _FakeRobot()

        result = smoke.run_basic_movement_smoke_test(
            robot=robot,
            execute=True,
            pause_s=0.01,
            settle_s=0.01,
            sleep_fn=lambda _seconds: None,
            log_fn=lambda _message: None,
        )

        send_calls = [call for call in robot.calls if call[0] in {"command_pwm", "custom"}]
        self.assertEqual(len(send_calls), 6)
        self.assertEqual(send_calls[0][0:2], ("command_pwm", "f"))
        self.assertEqual(send_calls[1][0:2], ("command_pwm", "b"))
        self.assertEqual(send_calls[2][0:2], ("custom", "l"))
        self.assertEqual(send_calls[3][0:2], ("custom", "r"))
        self.assertEqual(send_calls[4][0:2], ("command_pwm", "u"))
        self.assertEqual(send_calls[5][0:2], ("command_pwm", "d"))
        self.assertGreaterEqual(len([call for call in robot.calls if call[0] == "stop"]), 7)
        self.assertTrue(result["execute"])


if __name__ == "__main__":
    unittest.main()
