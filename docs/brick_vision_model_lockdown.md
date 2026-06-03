# Brick Vision Model Lockdown

Date: 2026-06-03

## What Broke

The livestream/debug vision path was letting the held-brick detector choose the distance model frame by frame. A false "holding" read could switch the stream into the holding target model and apply holding distance calibration, which made the distance jump to 90 mm even while we were practicing empty Step 1.

The game script was already safer when forced to the empty profile, but the livestream was not. That made debugging painful because the view could show the wrong model even when the step at hand was an empty step.

## Golden Rule

The active game step chooses the vision model. A noisy frame-level holding guess may be shown as a diagnostic, but it must not switch models unless the step context allows it.

## Step Policy

- `empty_s1`, `empty_s2`: normal stack model only. Holding masks and holding distance calibration are blocked.
- `empty_s3`: close stack model. Still no held-brick mask unless the game is explicitly in a holding context.
- `holding_s1`, `holding_s2`, `holding_s3`: held-brick mask model is allowed. The mask lock may keep the held-brick exclusion stable through brief detector flicker.

## Expected Livestream Text

For empty Step 1 practice, start the stream with:

```bash
python3 -u livestream_crown_vision.py --stream-host 0.0.0.0 --stream-port 5000 --vision-context empty_s1
```

Healthy debug text should include:

```text
[VISION] CONTEXT: Empty Step 1/2: normal stack model locked
HOLDING: false (...) raw=true/... model=blocked
```

It is okay for `raw=true` to appear. That means the side detector noticed green at the top. It is not okay for empty Step 1/2 to show `model=allowed` or a `holding calibrated: ... -> 90mm` line.

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

If distance jumps between the real estimate and 90 mm during empty Step 1:

1. Check `/text` on the livestream.
2. Confirm the context is `empty_s1` or `empty_s2`.
3. Confirm the holding line says `model=blocked`.
4. Confirm there is no `holding calibrated` line.
5. If the stream is in a holding context, restart it with `--vision-context empty_s1`.
