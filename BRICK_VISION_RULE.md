# Brick Vision Rule

## Purpose
Leia tracks bright green/cyan LEGO brick faces with the OAK-D camera using a
TensorRT YOLO model plus HSV color confirmation. The operating goal is practical
runtime tracking: accept obvious bright green brick rectangles and stacks, while
blocking unsupported background false positives.

## Current Runtime Profile
The default livestream and follow-the-brick profile is `tight_color`.

Current profile intent:
- Use the TensorRT model as the primary proposal source.
- Use wide green HSV bounds because the real bricks are bright green, not only
  narrow cyan.
- Allow trusted model boxes only after the box contains strong green HSV
  evidence.
- Require cyan/green shape evidence before returning a visible brick.
- Keep the far-suspect color-only path disabled for livestream/follow runtime.

Core settings:
```json
{
  "confidence": 0.20,
  "hsv_lower": [60, 15, 8],
  "hsv_upper": [93, 255, 255],
  "hsv_cyan_coverage_min": 0.05,
  "hsv_min_area_ratio": 0.03,
  "shape_gate_mode": "shape_match",
  "conf_gate_pct": 50.0,
  "trust_detector_boxes": true,
  "require_cyan_shape": true,
  "far_suspect_enabled": false
}
```

`livestream_crown_vision.py` and `a_follow_the_brick.py` must stay aligned on
this profile unless there is an explicit reason to diverge.

## Detection Paths
There are three brick-detection paths in the detector.

1. TensorRT YOLO boxes

YOLO proposes candidate boxes. In the active runtime profile, YOLO confidence
must meet the configured threshold, and the accepted target still needs green
HSV/shape support. The detector may use the model box as the final target box,
but only after the box contains enough green pixels.

2. Green HSV plus shape matching

HSV segmentation finds green/cyan pixels inside YOLO boxes or, for close-up
cases, in the full frame. The detector then tries brick-face geometry such as
shape matching, trapezoid/cutout evidence, stack splitting, and bbox tightening.
This is the normal path for the livestream when it reports:

```text
SEARCH: target locked (HSV)
VISIBLE: true
```

3. Far-suspect color fallback

The detector can mark a tiny green blob as a suspected far brick. This path is
useful for experiments but is disabled in the livestream and follow runtime
because it caused serious false positives. If it is ever re-enabled, it must be
treated as operator-facing "suspected" evidence, not as a motion-ready brick.

## Why This Balance Matters
The detector previously failed in two opposite ways:
- Too permissive: weak green blobs and low-confidence boxes became false
  positives.
- Too strict: obvious bright green brick stacks were rejected because the
  profile required tight color/cutout evidence.

The current balance keeps far-suspect disabled, but restores the practical path
that worked well: model proposal plus wide green HSV plus shape support.

Do not fix false positives by requiring negative-cutout-only gates for normal
runtime. Stacked bricks can merge into one obvious rectangle and still fail
cutout-only matching. The stable runtime default is `shape_match`.

## Operator Checks
Use the livestream at:

```bash
python3 livestream_crown_vision.py
```

Expected healthy telemetry when a brick stack is directly in front of the
camera:

```text
PROFILE: 2 Tight Color
TRUST: model
SEARCH: target locked (HSV)
TOP CONF: 70%+
VISIBLE: true
CONF: 85%
```

If the page loads but the camera image is missing, check `/video_feed` directly:

```bash
curl -s -m 3 http://127.0.0.1:5000/video_feed -o /tmp/crown_video_probe.bin
```

A healthy feed contains repeated JPEG frames. The browser can hold a stale
multipart connection after restarts; reopening the page or hard-refreshing
usually fixes the client side.

## Code Locations
- `helper_brick_detector_yolo.py`: shared detector pipeline and runtime tuning.
- `livestream_crown_vision.py`: livestream profile selection and operator
  telemetry.
- `a_follow_the_brick.py`: follow-the-brick runtime profile and robot-facing
  vision reads.
- `BRICK_VISIBILITY_SAFETY_RULE.md`: movement safety rule that consumes these
  detector results.

## Change Rule
When tuning brick vision:
- Keep livestream and follow profile semantics aligned.
- Do not let bare YOLO boxes drive visibility without green/shape evidence.
- Do not enable far-suspect for robot motion.
- Verify with `/text` and `/video_feed` before running the robot.
- Preserve `BRICK_VISIBILITY_SAFETY_RULE.md`: movement still requires a fresh,
  confident visible brick.
