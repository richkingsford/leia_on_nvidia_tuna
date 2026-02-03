"""
GrabCut-Based Brick Detector - Drop-in replacement for helper_brick_vision.py
==============================================================================

Replaces the HSV + notch-fingerprint pipeline with a GrabCut-based detector
that works WITHOUT ArUco markers and WITHOUT scene-specific color tuning.

Same interface as the original BrickDetector:
    read() → (found, angle, dist, offset_x, confidence, cam_height,
              brick_above, brick_below)

Usage:
    # In autobuild.py or any consumer, replace:
    #   from helper_brick_vision import BrickDetector
    # with:
    #   from helper_brick_vision_grabcut import BrickDetector
"""

import cv2
import numpy as np
import math
import os
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the core detector from brick_detector.py (same directory or parent)
# ---------------------------------------------------------------------------
import sys

_BRICK_DETECTOR_DIR = str(Path(__file__).resolve().parent.parent)
if _BRICK_DETECTOR_DIR not in sys.path:
    sys.path.insert(0, _BRICK_DETECTOR_DIR)

from brick_detector import (
    BrickDetector as _CoreDetector,
    DetectorConfig as _CoreConfig,
    BrickDetection,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"

# Known brick dimensions (mm) — from world_model_brick.json
BRICK_WIDTH_MM = 64.0
BRICK_HEIGHT_MM = 48.0
BRICK_DEPTH_MM = 20.0

# Camera defaults (overridden at runtime from actual frame size)
DEFAULT_FRAME_W = 640
DEFAULT_FRAME_H = 480


class BrickDetector:
    """
    GrabCut-based brick detector.  Drop-in replacement for the original
    BrickDetector in helper_brick_vision.py.

    Provides the same read() interface expected by telemetry_brick.py
    and autobuild.py.
    """

    def __init__(self, debug=True, save_folder=None, speed_optimize=False):
        self.debug = debug
        self.speed_optimize = speed_optimize
        self.log = logging.getLogger("BrickVisionGC")

        # Headless detection
        self.headless = True
        if self.debug and (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self.headless = False

        self.save_folder = save_folder
        self.current_frame = None
        self.raw_frame = None
        if self.save_folder and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # Camera setup — try indices 0–3
        self.cap = None
        for idx in range(4):
            self.log.info("Attempting camera %d...", idx)
            temp = cv2.VideoCapture(idx)
            if temp.isOpened():
                self.log.info("Connected to camera %d (%s)", idx,
                              temp.getBackendName())
                self.cap = temp
                break
            temp.release()

        if self.cap is None:
            self.log.error("No camera found (0–3). Using dummy.")
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_H)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        # Warm up
        for _ in range(5):
            self.cap.read()

        # Camera intrinsics (simple pinhole: focal = width px)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.camera_center_offset_px = 0.0

        # Load calibration offset from world model if available
        self._load_world_model()

        # Core detector (GrabCut pipeline)
        work_h = 360 if speed_optimize else 480
        cfg = _CoreConfig(
            work_height=work_h,
            temporal_alpha=0.70,
        )
        self._det = _CoreDetector(cfg)

        # 3D model points for PnP (brick face corners in mm)
        hw = BRICK_WIDTH_MM / 2.0
        hh = BRICK_HEIGHT_MM / 2.0
        self._face_3d = np.array([
            [-hw, -hh, 0],
            [ hw, -hh, 0],
            [ hw,  hh, 0],
            [-hw,  hh, 0],
        ], dtype=np.float64)

    def _load_world_model(self):
        """Load camera calibration offset from world_model_brick.json."""
        try:
            import json
            with open(WORLD_MODEL_BRICK_FILE, "r") as f:
                wm = json.load(f)
            self.camera_center_offset_px = (
                wm.get("calibration", {}).get("camera_center_offset_px", 0.0)
            )
        except (FileNotFoundError, json.JSONDecodeError):
            self.camera_center_offset_px = 0.0

    def _init_camera_matrix(self, w, h):
        focal = float(w)
        self.camera_matrix = np.array([
            [focal, 0, w / 2.0],
            [0, focal, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core: process a single frame
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Process one BGR frame.  Returns the same 9-tuple as the original:
            (found, angle, dist, offset_x, display_frame, confidence,
             cam_height, brick_above, brick_below)
        """
        fh, fw = frame.shape[:2]
        if self.camera_matrix is None:
            self._init_camera_matrix(fw, fh)

        detection = self._det.detect(frame)

        found = False
        angle = 0.0
        dist = 0.0
        offset_x = 0.0
        confidence = 0.0
        cam_height = 0.0
        brick_above = False
        brick_below = False

        if detection is not None and detection.confidence > 0.10:
            found = True
            confidence = detection.confidence * 100.0   # 0–100 scale
            angle = detection.angle_deg                  # already [-90,+90]

            # Distance from apparent brick width in pixels
            _, _, bw, _ = detection.bbox
            if bw > 0:
                focal = self.camera_matrix[0, 0]
                dist = (BRICK_WIDTH_MM * focal) / bw

                # Lateral offset (mm) from camera center
                cx_px = detection.center[0]
                cal_cx = (fw / 2.0) + self.camera_center_offset_px
                offset_x = (cx_px - cal_cx) * (dist / focal)

            # --- PnP refinement (if face polygon looks reasonable) --------
            poly = detection.face_polygon
            if poly is not None and len(poly) == 4:
                # Order polygon corners: TL, TR, BR, BL
                ordered = self._order_corners(poly.astype(np.float64))
                success, rvec, tvec = cv2.solvePnP(
                    self._face_3d, ordered,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if success:
                    dist = float(np.linalg.norm(tvec))
                    R, _ = cv2.Rodrigues(rvec)
                    brick_x = R[:, 0]
                    pnp_angle = -math.degrees(
                        math.atan2(brick_x[2], brick_x[0])
                    )
                    # Sanity: only use PnP angle if it's within reason
                    if abs(pnp_angle) <= 90:
                        angle = pnp_angle

                    cam_world = -np.matrix(R).T @ np.matrix(tvec)
                    cam_height = float(cam_world[1, 0])

                    focal = self.camera_matrix[0, 0]
                    mm_fix = (self.camera_center_offset_px * dist) / focal
                    offset_x = tvec[0][0] - mm_fix

        # Draw debug overlay
        display = self._det.draw(frame, detection)

        return (found, angle, dist, offset_x, display,
                confidence, cam_height, brick_above, brick_below)

    @staticmethod
    def _order_corners(pts):
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.
        Needed for consistent PnP correspondence.
        """
        # Sort by y first (top two, bottom two)
        s = pts[pts[:, 1].argsort()]
        top = s[:2]
        bot = s[2:]
        # Within each pair, sort by x
        tl, tr = top[top[:, 0].argsort()]
        bl, br = bot[bot[:, 0].argsort()]
        return np.array([tl, tr, br, bl], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API — matches original BrickDetector.read()
    # ------------------------------------------------------------------

    def read(self):
        """
        Read one frame from the camera, detect the brick, return metrics.

        Returns:
            (found, angle, dist, offset_x, confidence,
             cam_height, brick_above, brick_below)

        On camera failure: returns (False, 0, -1, 0, 0, 0, False, False)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, 0, -1, 0, 0, 0, False, False

        self.raw_frame = frame.copy()

        (found, angle, dist, offset_x, display,
         confidence, cam_height, brick_above, brick_below) = self.process_frame(frame)

        self.current_frame = display

        # Save screenshot if enabled
        if self.save_folder is not None:
            ts = int(time.time() * 1000)
            cv2.imwrite(
                os.path.join(self.save_folder, f"frame_{ts}.jpg"), display
            )

        # Debug display
        if self.debug and not self.headless and not self.speed_optimize:
            cv2.imshow("Brick Vision (GrabCut)", display)
            cv2.waitKey(1)

        # Noise filter: same 25% threshold as original
        reliable = found and confidence > 25
        if not reliable:
            brick_above = False
            brick_below = False

        return (reliable, angle, dist, offset_x,
                confidence, cam_height, brick_above, brick_below)

    # ------------------------------------------------------------------
    # Utility — matches original interface
    # ------------------------------------------------------------------

    def save_frame(self, filename):
        if self.current_frame is not None:
            cv2.imwrite(filename, self.current_frame)
            return True
        return False

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    det = BrickDetector(save_folder=None)
    print("Running GrabCut brick detector (Ctrl+C to stop)...")
    try:
        while True:
            found, angle, dist, off_x, conf, cam_h, above, below = det.read()
            if found:
                print(f"  FOUND  angle={angle:.1f}°  dist={dist:.0f}mm  "
                      f"offset={off_x:.1f}mm  conf={conf:.0f}%")
    except KeyboardInterrupt:
        pass
    finally:
        det.close()
