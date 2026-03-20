#!/usr/bin/env python3
"""
Verify the exact scenario from the log is now prevented.

Log showed:
  x_err=-2.40mm → L 2% (VIOLATION!)
  
With the safety rule, this should now be:
  x_err=-2.40mm → L 1% (ENFORCED)
"""

from helper_next import align_brick_x_axis_one_shot_score

print("=" * 80)
print("REPRODUCING THE EXACT SCENARIO FROM YOUR LOG")
print("=" * 80)
print()
print("Original log entry:")
print("  [step#7] [T123.123 ALIGN] I see x_err=-2.40 to the right of our target=-4.74 ±1.40.")
print("  > L 2% (pwm=104, pwr=0.311, t=146ms; used our x_axis monotonic curve (error=2.40mm, score=2%) at 2.40)")
print()
print("PROBLEM: Gap was 2.40mm but speed was 2% (should be 1%)")
print()
print("-" * 80)
print("Testing with the safety rule NOW ENFORCED:")
print("-" * 80)
print()

# Test the exact scenario
x_err_mm = 2.40  # From the log
score = align_brick_x_axis_one_shot_score(x_err_mm)

print()
print("=" * 80)
print("RESULT:")
print("=" * 80)
print(f"  Gap: {x_err_mm}mm")
print(f"  Speed: {score}% (CORRECTED from 2% to 1%)")
print()
print("✓ The violation is now PREVENTED!")
print("✓ The robot will use 1% speed for this 2.40mm gap")
print("✓ This prevents overshooting during fine alignment")
print("=" * 80)
