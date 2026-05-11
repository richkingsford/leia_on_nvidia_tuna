#!/usr/bin/env python3
"""Discover brick HSV ranges from the live camera OR a saved image file.

Usage:
  python3 probe_hsv.py               # live camera
  python3 probe_hsv.py ~/bricks.jpg  # from image file

Clusters the vivid-green pixels with k-means and prints ready-to-paste
CYAN_SHADE_HEXES and TIGHT/BALANCED/WIDE HSV tuples.
"""
from __future__ import annotations
import time, sys
import cv2
import numpy as np

WARMUP   = 20
CAPTURE  = 10
LOOP_S   = 0.07
K        = 4

# Pre-filter: vivid teal-green only (weeds out gray backgrounds)
PRE_H_LO, PRE_H_HI = 50, 135
PRE_S_MIN           = 120
PRE_V_MIN           = 50


def _hsv_to_hex(h, s, v):
    px = np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return f"{int(r):02X}{int(g):02X}{int(b):02X}"


def _range(centers, h_margin, s_margin, v_margin, s_floor=0):
    h, s, v = centers[:, 0], centers[:, 1], centers[:, 2]
    lo = (max(0, int(h.min()) - h_margin),
          max(s_floor, int(s.min()) - s_margin),
          max(0, int(v.min()) - v_margin))
    hi = (min(179, int(h.max()) + h_margin), 255, 255)
    return lo, hi


def _collect_from_image(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: could not read image: {path}")
        sys.exit(1)
    print(f"Loaded image: {frame.shape[1]}×{frame.shape[0]}  ({path})")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = (H >= PRE_H_LO) & (H <= PRE_H_HI) & (S >= PRE_S_MIN) & (V >= PRE_V_MIN)
    px = hsv[mask]
    print(f"  {px.shape[0]:,} candidate pixels (H {PRE_H_LO}-{PRE_H_HI}, S≥{PRE_S_MIN}, V≥{PRE_V_MIN})")
    if px.shape[0] < 50:
        print("ERROR: too few vivid pixels — is the brick visible and well-lit?")
        sys.exit(1)
    return px.astype(np.float32)


def _collect_from_camera() -> np.ndarray:
    from helper_brick_detector_yolo import BrickDetector
    print("Starting BrickDetector…")
    vision = BrickDetector(debug=False)

    print(f"Warming up ({WARMUP} frames)…", flush=True)
    for _ in range(WARMUP):
        try: vision.read()
        except Exception: pass
        time.sleep(LOOP_S)

    print(f"Sampling {CAPTURE} frames — keep bricks in view…", flush=True)
    pools = []
    attempts = 0
    while len(pools) < CAPTURE and attempts < CAPTURE * 5:
        attempts += 1
        try: vision.read()
        except Exception:
            time.sleep(LOOP_S); continue
        frame = getattr(vision, "raw_frame", None)
        if frame is None or frame.size == 0:
            time.sleep(LOOP_S); continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        mask = (H >= PRE_H_LO) & (H <= PRE_H_HI) & (S >= PRE_S_MIN) & (V >= PRE_V_MIN)
        px = hsv[mask]
        if px.shape[0] >= 50:
            idx = np.random.choice(px.shape[0], min(px.shape[0], 2000), replace=False)
            pools.append(px[idx])
            print(f"  frame {len(pools)}/{CAPTURE}: {px.shape[0]:,} candidate pixels")
        time.sleep(LOOP_S)

    try: vision.close()
    except Exception: pass

    if not pools:
        print("\nERROR: no vivid pixels found — make sure bricks are in view.")
        sys.exit(1)
    return np.vstack(pools).astype(np.float32)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        pool = _collect_from_image(image_path)
    else:
        pool = _collect_from_camera()

    pool = np.vstack(pool).astype(np.float32)
    print(f"\nClustering {len(pool):,} pixels…")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    k = min(K, len(pool))
    _, labels, centers = cv2.kmeans(pool, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order  = np.argsort(-counts)
    centers, counts = centers[order].astype(float), counts[order]
    total = counts.sum()

    print(f"\n── Clusters ──────────────────────────────────────")
    candidates = []
    for i, (c, n) in enumerate(zip(centers, counts)):
        h, s, v = c
        hex_code = _hsv_to_hex(h, s, v)
        pct = 100.0 * n / total
        marker = ""
        # Keep vivid teal-green (H=55-95, S≥130) — filters out blue-gray backgrounds
        if 55 <= h <= 95 and s >= 130:
            candidates.append(c)
            marker = " ✓ brick"
        print(f"  Cluster {i+1}: H={h:.1f} S={s:.1f} V={v:.1f}  #{hex_code}  ({pct:.1f}%){marker}")

    if not candidates:
        # Fallback: highest-saturation cluster
        candidates = [centers[np.argmax(centers[:, 1])]]
        print("  (no vivid-teal cluster; using highest-saturation cluster as fallback)")

    kept = np.vstack(candidates)
    hexes = tuple(_hsv_to_hex(*c) for c in kept)

    tight_lo,    tight_hi    = _range(kept, h_margin=4,  s_margin=40,  v_margin=30,  s_floor=65)
    balanced_lo, balanced_hi = _range(kept, h_margin=8,  s_margin=80,  v_margin=55,  s_floor=45)
    wide_lo,     wide_hi     = _range(kept, h_margin=14, s_margin=130, v_margin=80,  s_floor=20)

    print(f"\n── Paste into helper_brick_detector_yolo.py ──────")
    print(f"\nCYAN_SHADE_HEXES = (")
    for h in hexes:
        print(f'    "{h}",')
    print(")")
    print(f"\n# Camera-calibrated HSV ranges (painted bricks, OAK camera)")
    print(f"CYAN_HSV_TIGHT_LOWER: tuple[int, int, int] = {tight_lo}")
    print(f"CYAN_HSV_TIGHT_UPPER: tuple[int, int, int] = {tight_hi}")
    print(f"CYAN_HSV_BALANCED_LOWER: tuple[int, int, int] = {balanced_lo}")
    print(f"CYAN_HSV_BALANCED_UPPER: tuple[int, int, int] = {balanced_hi}")
    print(f"CYAN_HSV_WIDE_LOWER: tuple[int, int, int] = {wide_lo}")
    print(f"CYAN_HSV_WIDE_UPPER: tuple[int, int, int] = {wide_hi}")

    print(f"\n── Self-check: coverage on sampled pixels ─────────")
    for name, lo, hi in [("TIGHT", tight_lo, tight_hi),
                          ("BALANCED", balanced_lo, balanced_hi),
                          ("WIDE", wide_lo, wide_hi)]:
        lo_a = np.array(lo, dtype=np.float64)
        hi_a = np.array(hi, dtype=np.float64)
        hits = np.all((pool >= lo_a) & (pool <= hi_a), axis=1).sum()
        print(f"  {name:8s} {hits:,} / {len(pool):,} px  ({100*hits/len(pool):.1f}%)")


if __name__ == "__main__":
    main()
