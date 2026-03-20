# Align Phase Simplification for Accuracy-First Approach

## Summary
Simplified the ALIGN_BRICK phase in `calibrate_align.py` to prioritize **accuracy and reliability over speed**, eliminating distance-only bias and gate overshoots.

## Changes Made

### 1. Removed Complex Decision Logic
**Deleted (~200 lines of code):**
- Coarse vs Precision mode threshold decision (based on error magnitude)
- Stall detection and escalation logic
- Direction change tracking and dampening
- Settling zone multiplier (0.4x when near target)
- Proportional control scaling (0.5-1.0x based on error ratio)
- Overshoot penalty exponential decay (0.65^n, floored at 0.2x-0.3x)
- Direction change dampening (0.50x scale when direction flips)
- Feature annotation for diagnostics

### 2. Unified to Minimal Conservative Speeds
**New approach (8 lines):**
```python
if correction_type == "distance":
    # Always use score 3 (~10% power) for distance movements
    event = self.send_coarse_command(align_dir, score=3, ...)
else:
    # Always use 3% precision bump for x-axis movements  
    event = self.send_precision_command(align_dir, bump_pct=3.0, ...)
```

**Key principle:** No speed adaptation, no scaling, no features. Just minimal reliable movement.

## Architecture Principle (Enforced)
- Helper modules must contain generalized, reusable logic only.
- Step-specific behavior must be represented in world model files (`world_model_*.json`), not hardcoded by step name inside helper modules.
- Any step-specific helper exception requires explicit human leadership approval and documentation.

### 3. Simplified Console Output
**Before:**
```
[ALIGN] worst_metric=dist, ...many feature flags... settling(0.40x), overshoot_penalty(0.65x), proportional(0.75x)
```

**Now:**
```
[ALIGN] worst_metric=dist, error=+2.35mm, cmd=f
[ALIGN] worst_metric=x_axis, error=+0.89mm, cmd=l
```

Only shows: worst metric, numerical error, command direction.

### 4. Reinstated Essential Safety Limits
**Added back (before any movement command):**
- Check if overshoot count ≥ MAX_OVERSHOOTS_PER_TRIAL (8) → stop trial
- Check if adjust count ≥ MAX_ADJUSTS_PER_TRIAL (20) → stop trial

## Expected Behavior Changes

### ✅ Improvements
1. **Worst-metric rule is now truly followed**
   - Will see alternating distance and x-axis corrections
   - No more distance-only bias
   - Correct metric (biggest gap) is always addressed

2. **Zero gate overshoots**
   - Minimal speeds (score 3, bump 3%) make overshooting nearly impossible
   - Won't penetrate success gate boundaries (±1.5mm for dist, ±1.4mm for x-axis)

3. **Performance degradation is now visible**
   - No hidden dampening factors masking actual speed
   - Slow convergence is honest signal to adjust learning rate if needed
   - No "stall escalation" fooling the system into faster movement

4. **Cleaner diagnostics**
   - Console output is minimal and focused
   - No feature flags to interpret
   - Easier to spot oscillation or patterns

### ⚠️ Trade-offs
1. **Slower convergence**
   - Score 3 and 3% bumps are very conservative
   - Trials will take longer (expected: 10-20 seconds → 20-40 seconds)
   - This is acceptable trade for reliability

2. **More acts per trial**
   - Using 20 acts limit (vs previous lower limits with dampening)
   - Expected: 15-20 acts vs 8-12 with faster moving

3. **Stall detection removed**
   - No automatic speed escalation if stuck
   - If truly stuck, manual human inspection of the gate/target

## Testing
- **Test coverage:** 17 tests in `test_calibrate_align_dumb_adjustments.py`
- **Status:** ✅ All 17 tests PASS
- **Test focus:** Dumb adjustment prevention, one-gap-per-act rule, gate respect

## Configuration
**No changes needed!** Using same gate definitions:
- ALIGN_BRICK gate: xAxis ±1.4mm, distance ±1.5mm
- Both metrics must pass simultaneously
- SuccessGateTracker: 12 consecutive OR 14/26 majority

## Next Steps (If Needed)
1. **If still overshooting:** Score 3 and bump 3% can be reduced to 2 and 2%
2. **If convergence stalls:** Check gate target values are correct
3. **If fast path needed:** Once accuracy proved, add back dampening factors one-by-one

## Files Modified
- `calibrate_align.py`: Align phase logic (lines ~2620-2680)

## Validation
Run alignment trials:
```bash
python calibrate_align.py
```

Expected observations:
- Trial logs show alternating "worst_metric=dist" and "worst_metric=x_axis" 
- No movements beyond gate boundaries
- Every 8+ successful trials in row = gate passing reliably
