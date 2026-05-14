import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_turn_test


class TestHelperTurnTest(unittest.TestCase):
    def test_sequence_builds_forward_right_then_back_left_mirror(self):
        sequence = helper_turn_test.build_turn_test_sequence()

        self.assertEqual(len(sequence), 40)
        first = sequence[0]
        self.assertEqual(first.phase_name, "forward+right")
        self.assertEqual(first.logical_cmd, "r")
        self.assertEqual(first.drive_mode, "forward")
        self.assertEqual(first.outer_target, "l")
        self.assertEqual(first.inner_target, "r")

        mirror = sequence[20]
        self.assertEqual(mirror.phase_name, "back+left")
        self.assertEqual(mirror.logical_cmd, "l")
        self.assertEqual(mirror.drive_mode, "backward")
        self.assertEqual(mirror.outer_target, "r")
        self.assertEqual(mirror.inner_target, "l")
        self.assertEqual(mirror.inner_direction, "backward")

    def test_inner_tread_holds_still_then_reverses_to_outer_speed(self):
        sequence = helper_turn_test.build_turn_test_sequence()
        forward_phase = sequence[:20]

        self.assertEqual(forward_phase[7].inner_direction, "forward")
        self.assertEqual(forward_phase[8].inner_direction, "stop")
        self.assertEqual(forward_phase[9].inner_direction, "stop")
        self.assertEqual(forward_phase[10].inner_direction, "backward")
        self.assertEqual(forward_phase[-1].inner_pwm, forward_phase[-1].outer_pwm)

        back_phase = sequence[20:]
        self.assertEqual(back_phase[8].inner_direction, "stop")
        self.assertEqual(back_phase[10].inner_direction, "forward")
        self.assertEqual(back_phase[-1].inner_pwm, back_phase[-1].outer_pwm)

    def test_planned_pulses_use_model_safe_floor_duration(self):
        sequence = helper_turn_test.build_turn_test_sequence()

        self.assertTrue(all(pulse.duration_ms >= 250 for pulse in sequence))
        self.assertTrue(all(pulse.outer_pwm >= pulse.inner_pwm for pulse in sequence))
        self.assertTrue(all(action["target"] in {"l", "r"} for pulse in sequence for action in pulse.actions))

    def test_format_keeps_operator_level_motion_intent(self):
        line = helper_turn_test.format_turn_test_pulse(helper_turn_test.build_turn_test_sequence()[0])

        self.assertIn("cmd=FORWARD+RIGHT", line)
        self.assertIn("left forward", line)
        self.assertIn("right forward", line)
        self.assertNotIn("wire", line.lower())


if __name__ == "__main__":
    unittest.main()
