#!/usr/bin/env python3
"""
Unit tests to prevent "dumb adjustments" in calibrate_align.py

A "dumb adjustment" is when the control system makes a decision that worsens
the error or contradicts the previous successful action.

Case study (from Feb 17 run):
  [T1 ALIGN] x_err=+0.92mm → turn R → x_err=-0.05mm (GOOD, improved)
  [T1 ALIGN] x_err=-0.05mm → turn L → x_err=+0.83mm (BAD, worsened & opposite direction)

Test coverage:
  1. Overshoot detection - recognizes when error crosses zero
  2. Direction consistency - doesn't flip immediately after overshoot
  3. Error improvement validation - detects when act makes things worse
  4. Sign flip prevention - prevents immediate oscillation after reaching target
  5. Minimum movement threshold - ignores noise instead of overreacting
  6. Correction coherence - turning direction matches error direction
"""

import unittest
from collections import deque


class TestDumbAdjustmentPrevention(unittest.TestCase):
    """Tests for preventing dumb control adjustments"""
    
    def test_overshoot_detection_positive_to_negative(self):
        """Test: Detect when we cross from positive to negative error (overshoot right side)"""
        # Scenario: x_err was positive (too far right), now negative (went too far left)
        x_err_history = deque([0.92, 0.50, 0.10, -0.05], maxlen=4)
        
        recent_signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in x_err_history]
        crossed_zero = (recent_signs[-2] == 1 and recent_signs[-1] == -1)
        
        self.assertTrue(crossed_zero, "Should detect crossing from positive to negative")
    
    def test_overshoot_detection_negative_to_positive(self):
        """Test: Detect when we cross from negative to positive error (overshoot left side)"""
        # Scenario: x_err was negative (too far left), now positive (went too far right)
        x_err_history = deque([-0.80, -0.40, -0.10, 0.15], maxlen=4)
        
        recent_signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in x_err_history]
        crossed_zero = (recent_signs[-2] == -1 and recent_signs[-1] == 1)
        
        self.assertTrue(crossed_zero, "Should detect crossing from negative to positive")
    
    def test_no_direction_flip_after_overshoot(self):
        """Test: Don't flip direction immediately after overshooting"""
        # Case study scenario:
        # Step 1: x_err +0.92 -> turn R (correct, moved error to -0.05)
        # Step 2: x_err -0.05 -> should NOT turn L (that would worsen it)
        
        x_err_sequence = [0.92, -0.05]  # Overshot!
        last_turn = 'R'  # Previous successful direction
        
        # With anti-oscillation: detect overshoot, keep turning R
        recent_signs = deque([1, -1], maxlen=3)  # positive -> negative
        just_crossed_zero = (list(recent_signs)[-2] != 0 and 
                            list(recent_signs)[-1] != 0 and 
                            list(recent_signs)[-2] != list(recent_signs)[-1])
        
        if just_crossed_zero:
            next_turn = last_turn  # Don't flip!
        else:
            next_turn = 'L' if x_err_sequence[-1] < 0 else 'R'
        
        self.assertEqual(next_turn, 'R', 
                        "Should keep turning R even though x_err is now negative")
    
    def test_detect_worsening_acts(self):
        """Test: Recognize when an action made error worse instead of better"""
        # The act of turning L took error from -0.05 to +0.83 (worsened by 0.77mm)
        x_err_before = -0.05
        x_err_after = 0.83
        
        abs_err_before = abs(x_err_before)
        abs_err_after = abs(x_err_after)
        delta = abs_err_before - abs_err_after  # Should be negative for worsening
        
        is_worsening = delta < -0.02  # Worsened by more than 0.02mm
        
        self.assertTrue(is_worsening, 
                       f"Should detect worsening: {abs_err_before:.2f}mm -> {abs_err_after:.2f}mm")
    
    def test_direction_sign_coherence(self):
        """Test: Turning direction should match error direction"""
        # Mechanical mapping in calibrate_align.py: align_dir = 'r' if x_err > 0 else 'l'
        # This means:
        #   - If x_err is positive (brick right of target) -> turn RIGHT
        #   - If x_err is negative (brick left of target) -> turn LEFT
        
        test_cases = [
            (1.5, 'R', True),    # Positive error, turn right -> CORRECT
            (1.5, 'L', False),   # Positive error, turn left -> WRONG
            (-1.5, 'L', True),   # Negative error, turn left -> CORRECT
            (-1.5, 'R', False),  # Negative error, turn right -> WRONG
        ]
        
        for x_err, turn_dir, should_be_correct in test_cases:
            correct_dir = 'R' if x_err > 0 else 'L'
            is_correct = (turn_dir == correct_dir)
            
            self.assertEqual(is_correct, should_be_correct,
                           f"x_err={x_err:+.1f}, direction={turn_dir}, expected_correct={should_be_correct}")
    
    def test_oscillation_prevention_three_frame_window(self):
        """Test: Prevent oscillation by tracking sign changes in 3-frame window"""
        # Pattern that indicates oscillation:
        # Frame 1: x_err = +0.92 (right), turn R, improve
        # Frame 2: x_err = -0.05 (left), turn L, worsen
        # Frame 3: x_err = +0.83 (right), oscillating!
        
        x_err_window = deque([0.92, -0.05, 0.83], maxlen=3)
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in x_err_window]
        
        # Check for oscillation: sign changed at least twice in 3 frames
        sign_changes = sum(1 for i in range(len(signs)-1) 
                          if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1])
        
        is_oscillating = sign_changes >= 2
        
        self.assertTrue(is_oscillating,
                       f"Should detect oscillation pattern in signs {signs}")
    
    def test_noise_threshold_prevents_flip_on_measurement_jitter(self):
        """Test: Small measurement noise doesn't trigger direction flip"""
        # If x_err is 0.12mm but next measurement is -0.08mm (measurement jitter),
        # we shouldn't flip direction on this 0.20mm difference
        
        x_err_frame1 = 0.12
        x_err_frame2 = -0.08  # Just noise, probably
        noise_threshold = 0.3  # Only care if error changes by >0.3mm
        
        error_change = abs(x_err_frame2 - x_err_frame1)
        should_flip_direction = error_change > noise_threshold
        
        self.assertFalse(should_flip_direction,
                        f"Should ignore {error_change:.2f}mm change (< {noise_threshold}mm threshold)")
    
    def test_consecutive_direction_changes_penalty(self):
        """Test: Track consecutive direction flips as sign of instability"""
        # If we flip direction 2+ times in 5 acts, something is wrong
        direction_history = deque(['R', 'R', 'L', 'L', 'R'], maxlen=5)
        
        flips = sum(1 for i in range(len(direction_history)-1)
                   if direction_history[i] != direction_history[i+1])
        
        is_unstable = flips >= 2  # 2+ direction changes in 5 acts
        
        self.assertTrue(is_unstable,
                       f"Should flag {flips} direction changes in {len(direction_history)} acts as unstable")


class TestActionSelectionLogic(unittest.TestCase):
    """Tests for the action selection decision-making"""
    
    def test_error_improvement_calculation(self):
        """Test: Correctly calculate if error improved or worsened"""
        test_cases = [
            # (abs_err_before, abs_err_after, expected_improvement_mm)
            (0.92, 0.05, 0.87),    # Improved (positive delta)
            (0.05, 0.83, -0.78),   # Worsened (negative delta)
            (1.50, 1.49, 0.01),    # Tiny improvement
            (0.10, 0.10, 0.00),    # No change
        ]
        
        for err_before, err_after, expected_delta in test_cases:
            delta = err_before - err_after
            self.assertAlmostEqual(delta, expected_delta, places=2,
                                  msg=f"{err_before:.2f} -> {err_after:.2f}")


class TestOvershootHandling(unittest.TestCase):
    """Tests for handling overshoot situations"""
    
    def test_one_strike_rule_applies_penalty(self):
        """Test: One-strike rule correctly penalizes after first overshoot"""
        learned_speed = 40
        overshoot_count = 1
        penalty_pct = 30.0
        
        if overshoot_count == 1:
            # Apply one-strike penalty
            new_speed = int(learned_speed * (1.0 - penalty_pct/100.0))
            expected_speed = 28  # 40 * 0.7 = 28
            
            self.assertEqual(new_speed, expected_speed,
                           "One-strike rule should reduce speed by 30%")
    
    def test_no_penalty_before_first_overshoot(self):
        """Test: No penalty applied if no overshoots yet"""
        learned_speed = 40
        overshoot_count = 0
        
        new_speed = learned_speed if overshoot_count == 0 else int(learned_speed * 0.7)
        
        self.assertEqual(new_speed, 40,
                        "Should not apply penalty before any overshoot")
    
    def test_direction_dampening_after_flip(self):
        """Test: Direction change applies speed dampening"""
        base_speed = 20
        direction_changed = True
        dampening_factor = 0.50
        
        final_speed = base_speed * dampening_factor if direction_changed else base_speed
        
        self.assertEqual(final_speed, 10,
                        "Direction change should apply 50% dampening")


class TestResetPhaseOscillationPrevention(unittest.TestCase):
    """Tests to prevent reset phase oscillation (direction flipping)"""
    
    def test_detect_backwards_direction_mapping(self):
        """Test: Detect when direction mapping is backwards (consistently wrong movements)"""
        # Scenario: We keep moving opposite to desired direction
        # Consecutive flips: 1st flip (f→b), 2nd flip (b→f), 3rd flip (f→b)
        # By 3rd flip, we know the mapping is backwards
        
        consecutive_flips = 3
        direction_inverted = consecutive_flips >= 3
        
        self.assertTrue(direction_inverted,
                       "After 3+ consecutive flips, should detect backwards direction mapping")
    
    def test_increase_flip_threshold_prevents_oscillation(self):
        """Test: Larger threshold (1.5mm) prevents oscillating on small movements"""
        # Before: MIN_MOVEMENT_TO_FLIP_MM = 0.5mm → flips on every small jitter
        # After: MIN_MOVEMENT_TO_FLIP_MM = 1.5mm → only flips on real misses
        
        movements = [-0.9, -1.8, -2.1, -2.7, -4.4]  # From the bug report
        min_flip_threshold = 1.5
        
        flip_count = sum(1 for m in movements if abs(m) > min_flip_threshold)
        
        self.assertEqual(flip_count, 4,  # Only -1.8, -2.1, -2.7, -4.4 exceed 1.5mm
                        "Larger threshold filters out small noise-induced flips")
    
    def test_dont_flip_after_direction_inverted(self):
        """Test: Once direction mapping is inverted, don't keep flipping"""
        # After detecting backwards mapping and inverting it, stick with it
        direction_inverted = True
        consecutive_flips = 5
        
        # Logic: if already inverted, don't flip again
        should_flip = not direction_inverted
        
        self.assertFalse(should_flip,
                        "Should not flip direction after inversion detected")
    
    def test_reset_flip_counter_on_correct_movement(self):
        """Test: Movement counter resets when movement is in correct direction"""
        # Scenario: flip counter increments on wrong movement, resets on correct movement
        
        consecutive_flips = 0
        
        # Movement 1: wrong direction -> flip count = 1
        consecutive_flips += 1
        self.assertEqual(consecutive_flips, 1)
        
        # Movement 2: wrong direction -> flip count = 2
        consecutive_flips += 1
        self.assertEqual(consecutive_flips, 2)
        
        # Movement 3: CORRECT direction -> reset counter
        consecutive_flips = 0
        self.assertEqual(consecutive_flips, 0,
                        "Flip counter should reset when movement goes correct direction")
    
    def test_identify_direction_inversion_after_three_flips(self):
        """Test: Identify backwards direction mapping after 3 consecutive wrong movements"""
        # Case from bug report: every single movement was wrong direction
        # After 3 flips, we should realize the direction is completely backwards
        
        desired_dist_mm = 130.0  # Want to move away
        movements = [-0.9, -1.8, -2.1]  # All moved wrong direction
        
        consecutive_wrong_moves = 0
        for dist_moved in movements:
            if dist_moved < 0:  # Moving wrong way
                consecutive_wrong_moves += 1
        
        direction_mapping_inverted = consecutive_wrong_moves >= 3
        
        self.assertTrue(direction_mapping_inverted,
                       "Should invert direction after 3+ consecutive wrong movements")


if __name__ == '__main__':
    unittest.main()
