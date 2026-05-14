# Brick Visibility Motion Safety Rule

## Rule
Leia must not send any non-stop movement command unless the current fresh brick
reading is confidently visible.

For this rule, "confidently visible" means:
- the detector says `visible=true`
- the same fresh reading has confidence greater than or equal to the configured
  `brick_visibility_motion_safety.min_confidence_pct`

The default and current configured threshold is 75%.

## Enforcement
Reusable enforcement lives in `helper_brick_visibility_safety.py`.

Motion code that has access to vision must send motor commands through:
- `guarded_send_command_pwm(...)`
- `guarded_send_custom_actions_pwm(...)`

Those helpers stop the robot and refuse to send non-stop movement when the
current reading is missing, not visible, missing confidence, or below the
confidence threshold.

Stop commands are still allowed without brick visibility.

## Configuration
The threshold lives in `world_model_robot.json`:

```json
"brick_visibility_motion_safety": {
  "min_confidence_pct": 75.0
}
```

## Follow-The-Brick Integration
`a_follow_the_brick.py` reads a fresh detector sample each loop, normalizes it
through the safety helper, and only moves when that sample passes the rule. It
does not reuse held or stale brick readings.
