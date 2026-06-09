# Brick Vision Model Lockdown

Date: 2026-06-08

## What Broke

The livestream/debug vision path was letting the held-brick detector choose the distance model frame by frame. A false "holding" read could switch the stream into the holding target model and apply holding distance calibration, which made the distance jump to 90 mm even while we were practicing empty Step 1.

The game script was already safer when forced to the empty profile, but the livestream was not. That made debugging painful because the view could show the wrong model even when the step at hand was an empty step.

We hit a second version of the same class of bug after painting the stack green and lowering the camera: the detector switched between the normal native-rectangle distance model and green-edge close-range models (`green_edge_top_strip_width` / `green_edge_painted_column_width`). A real pose around 250 mm could suddenly read as about 90 mm. That is not acceptable for motion.

## Golden Rule

The active game step chooses one vision model and stays there. A noisy frame-level holding guess or green-edge measurement may be shown as a diagnostic, but it must not switch the robot-facing distance model unless the step context explicitly allows it.

For the current empty game, the robot-facing model is locked to the native rectangle stack model. Do not auto-switch to green-edge top-strip, painted-column, or held-brick calibrated distance while playing empty steps.

## Step Policy

- `empty_s1`, `empty_s2`: native rectangle stack model only. Holding masks, holding distance calibration, and green-edge close-range distance overrides are blocked.
- `empty_s3` / lift: keep the same model as far as vision is available; if the camera loses the stack because Leia is close or carrying a brick, the game may continue only through explicit blind scripted steps.
- `holding_s1`, `holding_s2`, `holding_s3`: held-brick mask model is allowed. The mask lock may keep the held-brick exclusion stable through brief detector flicker.

## Forbidden Empty-Game Sources

When the livestream or command-line read is being used for empty S1/S2 motion, these geometry sources are not allowed as robot-facing truth:

- `green_edge_top_strip_width`
- `green_edge_painted_column_width`
- `green_edge_close_range_width`
- holding-calibrated distance reads

Healthy empty-mode reads should stay on a native-rectangle source such as:

- `native_rect_width_fallback_only`
- `native_rect_width_robust_median`

If the model source flips frame to frame, stop and fix vision before trusting wins.

## Expected Livestream Text

For empty Step 1 practice, start the stream with:

```bash
python3 -u livestream_crown_vision.py --stream-host 0.0.0.0 --stream-port 8766 --vision-context empty_s1
```

Healthy debug text should include:

```text
[VISION] CONTEXT: Empty Step 1/2: normal stack model locked
HOLDING: false (...) raw=true/... model=blocked
```

It is okay for `raw=true` to appear. That means the side detector noticed green at the top. It is not okay for empty Step 1/2 to show `model=allowed`, a `holding calibrated: ... -> 90mm` line, or green-edge geometry as the active distance source.

## Regression Tests

Run these on Leia before trusting a future brick vision change:

```bash
python3 -m unittest tests/test_helper_holding_brick.py tests/test_livestream_vision_context.py
```

These tests protect the current fix:

- Thin top-edge green slivers do not count as holding.
- Short broad top-edge regions do not count as holding.
- A single top nub does not count as holding.
- The holding mask lock resets when an empty context blocks the holding model.
- Empty contexts preserve raw holding diagnostics but never activate holding masking or calibration.
- Holding contexts still use mask lock behavior for brief flicker.

## Quick Diagnosis

If distance jumps between the real estimate and about 90 mm during empty Step 1:

1. Check `/text` on the livestream.
2. Confirm the context is `empty_s1` or `empty_s2`.
3. Confirm the holding line says `model=blocked`.
4. Confirm there is no `holding calibrated` line.
5. Confirm geometry does not say `green_edge_top_strip_width`, `green_edge_painted_column_width`, or `green_edge_close_range_width`.
6. If the stream is in a holding context, restart it with `--vision-context empty_s1`.

## Current Code Lock

`helper_brick_detector_native_oak.py` keeps the green-edge helper available for future calibration, but the active native-rectangle path does not call it as a primary result or fallback. This is deliberate. Do not re-enable that switch without a step-scoped model selector and an operator-visible proof that distance does not jump across models.
