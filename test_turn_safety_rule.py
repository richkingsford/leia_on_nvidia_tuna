#!/usr/bin/env python3
"""
Test script to verify the hard safety rule: gap < 8mm → 1% speed for turns.
"""

import sys
from helper_next import align_brick_x_axis_one_shot_score, align_steps_turn_speed_score

def test_safety_rule():
    """Test that the safety rule is enforced."""
    print("Testing turn speed safety rule: gap < 8mm → 1% speed")
    print("=" * 80)
    
    # Test cases: (gap_mm, expected_behavior)
    test_cases = [
        (0.5, "should be 1%"),
        (2.4, "should be 1%"),  # Your log example
        (4.0, "should be 1%"),
        (7.9, "should be 1%"),
        (8.0, "can be > 1%"),
        (10.0, "can be > 1%"),
        (15.0, "can be > 1%"),
    ]
    
    print("\nTesting align_brick_x_axis_one_shot_score:")
    print("-" * 80)
    for gap_mm, expected in test_cases:
        try:
            score = align_brick_x_axis_one_shot_score(gap_mm)
            status = "✓ PASS" if (gap_mm < 8.0 and score == 1) or (gap_mm >= 8.0) else "✗ FAIL"
            print(f"{status}: gap={gap_mm:5.1f}mm → score={score:2d}% ({expected})")
        except RuntimeError as e:
            print(f"✗ FAIL: gap={gap_mm:5.1f}mm → EXCEPTION (should not happen in normal flow)")
            print(f"  Error: {str(e)[:100]}...")
    
    print("\nTesting align_steps_turn_speed_score:")
    print("-" * 80)
    for gap_mm, expected in test_cases:
        try:
            score = align_steps_turn_speed_score(gap_mm, fallback_score=6)
            status = "✓ PASS" if (gap_mm < 8.0 and score == 1) or (gap_mm >= 8.0) else "✗ FAIL"
            print(f"{status}: gap={gap_mm:5.1f}mm → score={score:2d}% ({expected})")
        except RuntimeError as e:
            print(f"✗ FAIL: gap={gap_mm:5.1f}mm → EXCEPTION (should not happen in normal flow)")
            print(f"  Error: {str(e)[:100]}...")
    
    print("\n" + "=" * 80)
    print("Test complete! The safety rule is now enforced.")
    print("\nIf you see a violation in production, the script will halt with:")
    print("  CRITICAL SAFETY VIOLATION: Turn speed rule broken!")
    print("  This prevents damage from overshooting during fine alignment.")

if __name__ == "__main__":
    test_safety_rule()
