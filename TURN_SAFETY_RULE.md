# Turn Speed Safety Rule Implementation

## Problem
The log showed a violation of a critical safety rule:
```
[step#7] [T123.123 ALIGN] I see x_err=-2.40 to the right of our target=-4.74 ±1.40.  
> L 2% (pwm=104, pwr=0.311, t=146ms; used our x_axis monotonic curve (error=2.40mm, score=2%) at 2.40)
```

**VIOLATION**: Gap was 2.40mm but speed was 2%. The rule states: **"whenever our gap is less than 8mm, we must go only 1% speed."**

## Solution Implemented

### 1. Hard Safety Check Function
Added `_enforce_turn_safety_rule()` in `helper_next.py`:
- Checks if x-axis gap < 8mm
- If gap < 8mm and score > 1%, it:
  - Prints a clear warning message
  - Auto-corrects the score to 1%
  - Can optionally halt the script (strict_mode=True)

### 2. Integration Points
The safety check is now enforced at TWO critical locations:

#### A. `align_brick_x_axis_one_shot_score()`
- This is the monotonic curve function that was producing 2% for 2.4mm gap
- Now enforces: gap < 8mm → 1% speed
- Used by: ALIGN_BRICK step and calibration

#### B. `align_steps_turn_speed_score()`
- This is the shared turn bands function
- Also enforces: gap < 8mm → 1% speed  
- Used by: POSITION_BRICK and other alignment steps

### 3. Constants
```python
ALIGN_TURN_SAFETY_GAP_MM = 8.0      # Threshold for safety rule
ALIGN_TURN_SAFETY_MIN_SCORE = 1     # Required speed when gap < 8mm
```

## How It Works

### Normal Operation (Auto-Correct Mode)
When a violation is detected:
```
================================================================================
CRITICAL SAFETY VIOLATION: Turn speed rule broken!
================================================================================
RULE: When x-axis gap < 8.0mm, speed MUST be 1%

VIOLATION DETAILS:
  Gap: 2.40mm (< 8.0mm threshold)
  Proposed speed: 2% (ILLEGAL - correcting to 1%)
  Context: align_brick_x_axis_one_shot_score

This is a hard safety rule to prevent overshooting during fine alignment.
Auto-correcting to 1% to enforce safety rule.
================================================================================
```

The system:
1. Detects the violation
2. Prints a clear warning
3. Auto-corrects to 1% speed
4. Continues operation safely

### Strict Mode (Optional)
If you want the script to HALT on violations instead of auto-correcting:
```python
_enforce_turn_safety_rule(x_err_mm, score, context="...", strict_mode=True)
```

This will raise a `RuntimeError` and stop execution immediately.

## Testing

Run the test script to verify:
```bash
python3 test_turn_safety_rule.py
```

Expected output:
- All gaps < 8mm are forced to 1% speed
- All gaps >= 8mm can use higher speeds
- Violations are detected and corrected with clear warnings

## Why This Matters

**The Problem**: When the robot is very close to the target (< 8mm), even a small turn at 2% speed can overshoot the target, causing:
- Ping-pong oscillation
- Inability to converge
- Potential damage from repeated overshooting

**The Solution**: By enforcing 1% speed for gaps < 8mm, we ensure:
- Fine control during final alignment
- No overshooting when close to target
- Stable convergence to the goal position

## Example Scenarios

### Before (Violation)
```
Gap: 2.40mm → Speed: 2% → OVERSHOOT → Gap: -3.10mm → Speed: 2% → OVERSHOOT → ...
```

### After (Enforced)
```
Gap: 2.40mm → Speed: 1% (enforced) → Gap: 1.80mm → Speed: 1% → Gap: 0.50mm → SUCCESS
```

## Future Enhancements

If you want even stricter enforcement:
1. Set `strict_mode=True` to halt on violations
2. Adjust `ALIGN_TURN_SAFETY_GAP_MM` threshold (currently 8mm)
3. Add similar checks for forward/backward distance movements
4. Add checks for lift movements (y-axis)

## Files Modified

1. `helper_next.py`:
   - Added `ALIGN_TURN_SAFETY_GAP_MM` constant
   - Added `ALIGN_TURN_SAFETY_MIN_SCORE` constant
   - Added `_enforce_turn_safety_rule()` function
   - Modified `align_brick_x_axis_one_shot_score()` to enforce rule
   - Modified `align_steps_turn_speed_score()` to enforce rule

2. `test_turn_safety_rule.py`:
   - New test script to verify enforcement

## Summary

The hard safety rule is now enforced at the source. Any code path that calculates turn speeds will automatically:
1. Check if gap < 8mm
2. Detect if proposed speed > 1%
3. Print a clear warning
4. Auto-correct to 1% speed
5. Continue safely

**The violation you saw in the log will no longer happen.** The system will catch it and correct it before the command is sent to the robot.
