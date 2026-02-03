#!/usr/bin/env python3
"""
brick_detector.py - Brick Face Detection for Pick-and-Place Robotics
=====================================================================

Detects interlocking concrete bricks WITHOUT ArUco markers using
classical computer vision. Designed for Jetson Orin Nano.

Detection Pipeline:
    1. Morphological-gradient texture density → position-weighted center-of-mass
       to estimate a bounding box around the brick
    2. GrabCut foreground extraction using the seed rectangle
    3. Morphological cleanup (close + open)
    4. Contour extraction with multi-criteria geometric scoring
       (rectangularity, solidity, extent, position, area)
    5. Temporal EMA smoothing for stable video tracking

Usage:
    # Live camera + livestream viewer (default, no args required)
    python3 brick_detector.py

    # Live camera without livestream (local OpenCV window)
    python3 brick_detector.py --no-stream --show

    # Play video with live brick overlay
    python3 brick_detector.py --input video.mp4

    # Export annotated frames as images
    python3 brick_detector.py --input video.mp4 --mode images --output ./out/

    # Process a single image
    python3 brick_detector.py --input photo.png --mode image --output ./out/

    # Tune for Orin Nano (lower resolution, faster)
    python3 brick_detector.py --input video.mp4 --work-height 360
"""

import cv2
import numpy as np
import logging
import argparse
import sys
import os
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from helper_manual_config import load_manual_training_config
from helper_streaming import start_stream_server

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """All tunable parameters in one place. No magic numbers in the pipeline."""

    # -- Processing resolution (height in pixels) --
    # Lower = faster (good for Orin Nano). 540 is a good balance.
    work_height: int = 540

    # -- Bilateral filter (edge-preserving denoise) --
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # -- Texture analysis (local variance via box filter) --
    texture_kernel: int = 21          # Kernel size for local variance

    # -- Gabor anisotropy (isotropic vs directional texture) --
    gabor_ksize: int = 31
    gabor_sigma: float = 4.0
    gabor_lambd: float = 10.0
    gabor_gamma: float = 0.5
    gabor_n_orientations: int = 4
    gabor_aniso_thresh: float = 1.8   # Anisotropy ratio above this → directional

    # -- Morphological cleanup --
    morph_close_size: int = 51        # Large close to fill brick interior
    morph_open_size: int = 11         # Open to remove noise blobs

    # -- Contour scoring thresholds --
    min_area_ratio: float = 0.005     # Min contour area / frame area
    max_area_ratio: float = 0.55      # Max contour area / frame area
    min_solidity: float = 0.35        # Contour area / convex hull area
    min_extent: float = 0.20          # Contour area / bounding rect area
    min_score: float = 0.10           # Reject detections below this

    # -- Temporal smoothing (video) --
    temporal_alpha: float = 0.65      # Weight for current frame (1.0=no smoothing)
    max_center_jump: int = 200        # Max px jump before resetting tracker

    # -- Drawing --
    poly_color: Tuple[int, int, int] = (0, 255, 0)
    face_color: Tuple[int, int, int] = (0, 200, 255)
    center_color: Tuple[int, int, int] = (0, 255, 255)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    poly_thickness: int = 3
    center_radius: int = 8
    font_scale: float = 0.7
    font_thickness: int = 2
    overlay_alpha: float = 0.18       # Transparency of face fill


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class BrickDetection:
    contour: np.ndarray               # Full contour of detected brick region
    face_polygon: np.ndarray          # 4-point polygon of the primary face
    center: Tuple[int, int]           # (x, y) center of the face
    confidence: float                 # 0.0 – 1.0
    area: float                       # Contour area in original-resolution px²
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) bounding rect
    angle_deg: float = 0.0            # Brick rotation in degrees [-90, +90]


# ---------------------------------------------------------------------------
# Brick Detector
# ---------------------------------------------------------------------------

class BrickDetector:
    """
    Classical-CV brick detector.  No deep learning, no GPU required
    (though CUDA-accelerated OpenCV will help on Orin Nano).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.log = logging.getLogger("BrickDetector")
        self._prev: Optional[BrickDetection] = None
        self._frame_idx: int = 0
        self._gabor_kernels = self._build_gabor_bank()

    # -- Gabor filter bank (built once) ------------------------------------

    def _build_gabor_bank(self) -> List[np.ndarray]:
        kernels = []
        for i in range(self.cfg.gabor_n_orientations):
            theta = i * np.pi / self.cfg.gabor_n_orientations
            kern = cv2.getGaborKernel(
                (self.cfg.gabor_ksize, self.cfg.gabor_ksize),
                self.cfg.gabor_sigma,
                theta,
                self.cfg.gabor_lambd,
                self.cfg.gabor_gamma,
                0,
                ktype=cv2.CV_32F,
            )
            kern /= kern.sum() + 1e-7
            kernels.append(kern)
        return kernels

    # -- Preprocessing ------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self.cfg.work_height / h
        new_w = int(w * scale)
        small = cv2.resize(frame, (new_w, self.cfg.work_height),
                           interpolation=cv2.INTER_AREA)
        return small, scale

    # -- Texture features ---------------------------------------------------

    def _morph_grad_density(self, gray: np.ndarray) -> np.ndarray:
        """
        Morphological gradient blurred into a texture-density map.
        Responds well to porous brick surfaces (random small edges)
        and brick boundaries (strong transitions).
        """
        k = self.cfg.texture_kernel
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, elem)
        density = cv2.blur(grad.astype(np.float32), (k, k))
        return density

    def _gabor_anisotropy(self, gray: np.ndarray) -> np.ndarray:
        """
        Ratio of max Gabor response to mean Gabor response.
        High → directional texture (ground).  Low → isotropic (brick pores).
        """
        responses = []
        gf = gray.astype(np.float32)
        for kern in self._gabor_kernels:
            resp = cv2.filter2D(gf, cv2.CV_32F, kern)
            responses.append(np.abs(resp))
        stack = np.stack(responses, axis=-1)
        max_resp = stack.max(axis=-1)
        mean_resp = stack.mean(axis=-1) + 1e-6
        anisotropy = max_resp / mean_resp
        return anisotropy

    # -- Mask generation ----------------------------------------------------

    def _texture_seed_rect(self, gray_smooth: np.ndarray,
                           h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Estimate a bounding box around the brick using position-weighted
        texture center-of-mass.  Returns (x, y, w, h) or None.
        """
        density = self._morph_grad_density(gray_smooth)
        thresh = np.percentile(density, 80)
        ys, xs = np.where(density > thresh)
        if len(xs) < 20:
            return None

        # Weight by position: prefer center-lower
        x_off = np.abs(xs - w / 2.0) / (w / 2.0)        # 0=center 1=edge
        y_norm = ys / float(h)                             # 0=top 1=bottom
        weights = (1.0 - 0.5 * x_off) * np.clip(y_norm / 0.25, 0, 1)
        weights += 1e-9

        cx = int(np.average(xs, weights=weights))
        cy = int(np.average(ys, weights=weights))

        # Bounding box sized ~45% of the shorter dimension
        half = int(min(h, w) * 0.28)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        bw, bh = x2 - x1, y2 - y1
        if bw < 30 or bh < 30:
            return None
        return (x1, y1, bw, bh)

    def _build_brick_mask(self, gray_smooth: np.ndarray,
                          small_bgr: np.ndarray) -> np.ndarray:
        """
        Build a binary mask using GrabCut with texture-seeded initialization.

        Pipeline:
          1. Position-weighted texture analysis → rough brick center
          2. Build bounding rect for GrabCut
          3. GrabCut foreground extraction (3 iterations)
          4. Morphological cleanup
        """
        h, w = gray_smooth.shape

        seed = self._texture_seed_rect(gray_smooth, h, w)
        if seed is None:
            return np.zeros((h, w), dtype=np.uint8)

        rx, ry, rw, rh = seed
        gc_mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(small_bgr, gc_mask, (rx, ry, rw, rh),
                        bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            self.log.debug("GrabCut failed, returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Light close + open
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_k)
        return fg

    # -- Candidate scoring --------------------------------------------------

    def _score_contour(self, c: np.ndarray,
                       frame_area: int,
                       frame_shape: Tuple[int, int]) -> float:
        area = cv2.contourArea(c)
        if area < 1:
            return 0.0

        ratio = area / frame_area
        if ratio < self.cfg.min_area_ratio or ratio > self.cfg.max_area_ratio:
            return 0.0

        # Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < self.cfg.min_solidity:
            return 0.0

        # Extent
        _, _, bw, bh = cv2.boundingRect(c)
        extent = area / (bw * bh) if bw * bh > 0 else 0
        if extent < self.cfg.min_extent:
            return 0.0

        # Rectangularity (contour area vs minimum-area-rect area)
        min_rect = cv2.minAreaRect(c)
        mar_area = min_rect[1][0] * min_rect[1][1]
        rectangularity = area / mar_area if mar_area > 0 else 0

        # Position heuristic: bricks tend to be in the lower-center region
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            fh, fw = frame_shape
            x_off = abs(cx - fw / 2) / (fw / 2)    # 0 = center, 1 = edge
            y_norm = cy / fh                         # 0 = top, 1 = bottom
            pos_score = (1.0 - 0.5 * x_off) * min(y_norm / 0.25, 1.0)
        else:
            pos_score = 0.5

        # Area reasonableness (broad peak around 3-25% of frame)
        area_score = 1.0 - abs(ratio - 0.12) / 0.15
        area_score = max(0.0, min(1.0, area_score))

        score = (
            0.25 * rectangularity
            + 0.20 * solidity
            + 0.20 * pos_score
            + 0.20 * area_score
            + 0.15 * extent
        )
        return float(score)

    # -- Main detection entry -----------------------------------------------

    def detect(self, frame: np.ndarray) -> Optional[BrickDetection]:
        self._frame_idx += 1

        # Resize for speed
        small, scale = self._resize(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_smooth = cv2.bilateralFilter(
            gray,
            self.cfg.bilateral_d,
            self.cfg.bilateral_sigma_color,
            self.cfg.bilateral_sigma_space,
        )

        # Build candidate mask
        mask = self._build_brick_mask(gray_smooth, small)

        # Extract contours and pick best
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self.log.debug("Frame %d: no contours", self._frame_idx)
            return None

        fh, fw = small.shape[:2]
        frame_area = fh * fw

        best_c, best_score = None, 0.0
        for c in cnts:
            s = self._score_contour(c, frame_area, (fh, fw))
            if s > best_score:
                best_score = s
                best_c = c

        if best_c is None or best_score < self.cfg.min_score:
            self.log.debug("Frame %d: no candidate above min_score (best=%.3f)",
                           self._frame_idx, best_score)
            return None

        # Scale back to original resolution
        contour_orig = (best_c.astype(np.float64) / scale).astype(np.int32)

        # Face polygon = minimum-area rectangle (best rectangle fit)
        min_rect = cv2.minAreaRect(contour_orig)
        face_poly = cv2.boxPoints(min_rect).astype(np.int32)

        # --- Angle extraction from minAreaRect -------------------------
        # minAreaRect returns ((cx,cy), (w,h), angle)
        # OpenCV convention: angle is in [-90, 0) for the first edge.
        # Normalize so the WIDER side defines 0° and range is [-90, +90].
        (_, _), (rect_w, rect_h), rect_angle = min_rect
        if rect_w < rect_h:
            angle_deg = rect_angle + 90.0
        else:
            angle_deg = rect_angle
        # Clamp to [-90, +90]
        if angle_deg > 90.0:
            angle_deg -= 180.0
        if angle_deg < -90.0:
            angle_deg += 180.0

        # Center via moments
        M = cv2.moments(contour_orig)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(contour_orig)
            cx, cy = bx + bw // 2, by + bh // 2

        # Temporal smoothing
        if self._prev is not None:
            dx = abs(cx - self._prev.center[0])
            dy = abs(cy - self._prev.center[1])
            if dx + dy < self.cfg.max_center_jump:
                a = self.cfg.temporal_alpha
                cx = int(a * cx + (1 - a) * self._prev.center[0])
                cy = int(a * cy + (1 - a) * self._prev.center[1])
                angle_deg = a * angle_deg + (1 - a) * self._prev.angle_deg

        det = BrickDetection(
            contour=contour_orig,
            face_polygon=face_poly,
            center=(cx, cy),
            confidence=best_score,
            area=float(cv2.contourArea(contour_orig)),
            bbox=cv2.boundingRect(contour_orig),
            angle_deg=round(angle_deg, 1),
        )
        self._prev = det
        self.log.debug("Frame %d: detected (%.2f) at (%d,%d) area=%.0f",
                       self._frame_idx, best_score, cx, cy, det.area)
        return det

    def reset(self) -> None:
        """Clear temporal state (call between unrelated videos)."""
        self._prev = None
        self._frame_idx = 0

    # -- Drawing overlay ----------------------------------------------------

    def draw(self, frame: np.ndarray,
             det: Optional[BrickDetection]) -> np.ndarray:
        out = frame.copy()
        cfg = self.cfg
        fh, fw = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if det is None:
            cv2.putText(out, "No brick detected", (10, 30),
                        font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return out

        # --- Contour outline (tight boundary) ---
        cv2.drawContours(out, [det.contour], -1, cfg.poly_color,
                         cfg.poly_thickness, cv2.LINE_AA)

        # --- Face rectangle outline ---
        cv2.drawContours(out, [det.face_polygon], -1, cfg.face_color,
                         cfg.poly_thickness, cv2.LINE_AA)

        # --- Transparent face fill ---
        overlay = out.copy()
        cv2.fillPoly(overlay, [det.face_polygon], cfg.poly_color)
        cv2.addWeighted(overlay, cfg.overlay_alpha, out,
                        1.0 - cfg.overlay_alpha, 0, out)

        # --- Center dot ---
        cx, cy = det.center
        cv2.circle(out, (cx, cy), cfg.center_radius, cfg.center_color, -1,
                   cv2.LINE_AA)
        cv2.circle(out, (cx, cy), cfg.center_radius + 2, (0, 0, 0), 2,
                   cv2.LINE_AA)

        # --- Center label ---
        label = f"({cx}, {cy})"
        (tw, th), baseline = cv2.getTextSize(label, font,
                                             cfg.font_scale, cfg.font_thickness)
        tx = cx + cfg.center_radius + 8
        ty = cy + th // 2
        # Keep label inside frame
        if tx + tw + 4 > fw:
            tx = cx - cfg.center_radius - tw - 8
        if ty + 4 > fh:
            ty = fh - 4
        if ty - th - 4 < 0:
            ty = th + 4
        cv2.rectangle(out, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 6),
                       (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), font, cfg.font_scale,
                    cfg.text_color, cfg.font_thickness, cv2.LINE_AA)

        # --- Angle indicator line from center ---
        length = max(det.bbox[2], det.bbox[3]) // 3
        rad = np.radians(det.angle_deg)
        ax = int(cx + length * np.cos(rad))
        ay = int(cy - length * np.sin(rad))
        cv2.line(out, (cx, cy), (ax, ay), (0, 0, 255), 3, cv2.LINE_AA)

        # --- Top-left HUD ---
        hud_lines = [
            f"Conf: {det.confidence:.2f}",
            f"Angle: {det.angle_deg:.1f} deg",
            f"BBox: {det.bbox[2]}x{det.bbox[3]}",
        ]
        for i, line in enumerate(hud_lines):
            y = 28 + i * 28
            cv2.putText(out, line, (10, y), font, 0.65,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), font, 0.65,
                        cfg.text_color, 1, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
# Video / Image Processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self, detector: BrickDetector, input_path: str,
                 mode: str = "video", output_dir: Optional[str] = None):
        self.detector = detector
        self.input_path = input_path
        self.mode = mode
        self.output_dir = output_dir
        self.log = logging.getLogger("VideoProcessor")

    def _ensure_output_dir(self) -> str:
        d = self.output_dir or os.path.join(
            os.path.dirname(self.input_path), "brick_detector_output"
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _is_image(self) -> bool:
        ext = os.path.splitext(self.input_path)[1].lower()
        return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

    def process_single_image(self) -> None:
        """Process one image, display and optionally save."""
        frame = cv2.imread(self.input_path)
        if frame is None:
            self.log.error("Cannot read image: %s", self.input_path)
            sys.exit(1)
        det = self.detector.detect(frame)
        vis = self.detector.draw(frame, det)

        out_dir = self._ensure_output_dir()
        base = os.path.splitext(os.path.basename(self.input_path))[0]
        out_path = os.path.join(out_dir, f"{base}_detected.png")
        cv2.imwrite(out_path, vis)
        self.log.info("Saved: %s", out_path)

        cv2.imshow("Brick Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self) -> None:
        """
        Process a video file.
        mode='video'  → play with live overlay (press Q/ESC to quit)
        mode='images' → export every Nth frame as annotated PNG
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.log.error("Cannot open video: %s", self.input_path)
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.info("Video: %dx%d @ %.1ffps, %d frames", w, h, fps, total)

        self.detector.reset()

        # For image-export mode
        out_dir = None
        export_every = max(1, int(fps / 2))  # ~2 images per second
        if self.mode == "images":
            out_dir = self._ensure_output_dir()
            base = os.path.splitext(os.path.basename(self.input_path))[0]
            self.log.info("Exporting frames to %s (every %d frames)",
                          out_dir, export_every)

        # For video-write mode
        writer = None
        if self.mode == "video_out":
            out_dir = self._ensure_output_dir()
            base = os.path.splitext(os.path.basename(self.input_path))[0]
            out_path = os.path.join(out_dir, f"{base}_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            self.log.info("Writing video to %s", out_path)

        frame_delay = max(1, int(1000 / fps))
        idx = 0
        detect_count = 0
        t_start = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            det = self.detector.detect(frame)
            if det is not None:
                detect_count += 1

            vis = self.detector.draw(frame, det)

            # Frame counter overlay
            elapsed = time.monotonic() - t_start
            proc_fps = idx / elapsed if elapsed > 0 else 0
            counter = f"Frame {idx}/{total}  ({proc_fps:.1f} fps)"
            cv2.putText(vis, counter, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, counter, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)

            if self.mode == "video":
                cv2.imshow("Brick Detection (Q/ESC to quit)", vis)
                key = cv2.waitKey(frame_delay) & 0xFF
                if key in (ord("q"), 27):
                    break

            if self.mode == "images" and idx % export_every == 0:
                fname = f"{base}_f{idx:04d}.png"
                cv2.imwrite(os.path.join(out_dir, fname), vis)

            if writer is not None:
                writer.write(vis)

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.monotonic() - t_start
        self.log.info(
            "Done: %d/%d frames detected (%.1f%%), %.1fs (%.1f fps)",
            detect_count, idx,
            100 * detect_count / max(idx, 1),
            elapsed,
            idx / max(elapsed, 1e-9),
        )

    def run(self) -> None:
        if self._is_image() or self.mode == "image":
            self.process_single_image()
        else:
            self.process_video()


# ---------------------------------------------------------------------------
# Live Camera Processor
# ---------------------------------------------------------------------------

class CameraProcessor:
    def __init__(
        self,
        detector: BrickDetector,
        camera_index: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        stream: bool = True,
        show: bool = False,
        stream_host: str = "127.0.0.1",
        stream_port: int = 5000,
        stream_fps: int = 10,
        stream_jpeg_quality: int = 85,
        max_frames: int = 0,
    ):
        self.detector = detector
        self.camera_index = int(camera_index)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.stream = bool(stream)
        self.show = bool(show)
        self.stream_host = str(stream_host)
        self.stream_port = int(stream_port)
        self.stream_fps = int(stream_fps)
        self.stream_jpeg_quality = int(stream_jpeg_quality)
        self.max_frames = max(0, int(max_frames))
        self.log = logging.getLogger("CameraProcessor")

        self.stream_server = None
        self.stream_state = {
            "frame": None,
            "lock": threading.Lock(),
        }

    def _set_stream_frame(self, frame: np.ndarray) -> None:
        with self.stream_state["lock"]:
            self.stream_state["frame"] = frame

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        if not cap.isOpened():
            self.log.error(
                "Cannot open camera index %d. Check /dev/video* and permissions.",
                self.camera_index,
            )
            sys.exit(1)

        for _ in range(3):
            cap.grab()

        if self.stream:
            self.stream_server, stream_url = start_stream_server(
                self.stream_state,
                title="Robot Leia - Brick Detector",
                header="Robot Leia - Brick Detector",
                footer="Press Ctrl+C in terminal to stop.",
                host=self.stream_host,
                port=self.stream_port,
                fps=self.stream_fps,
                jpeg_quality=self.stream_jpeg_quality,
            )
            self.log.info("Livestream started at %s", stream_url)
        else:
            self.log.info("Livestream disabled (--no-stream).")

        if self.show:
            self.log.info("Local OpenCV window enabled (Q/ESC to quit).")

        self.detector.reset()

        idx = 0
        detect_count = 0
        t_start = time.monotonic()
        quit_requested = False

        try:
            while not quit_requested:
                ret, frame = cap.read()
                if not ret:
                    self.log.warning("Camera read failed; stopping.")
                    break

                idx += 1
                det = self.detector.detect(frame)
                if det is not None:
                    detect_count += 1

                vis = self.detector.draw(frame, det)

                elapsed = max(1e-9, time.monotonic() - t_start)
                proc_fps = idx / elapsed
                counter = f"Frame {idx}  ({proc_fps:.1f} fps)"
                cv2.putText(
                    vis,
                    counter,
                    (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    counter,
                    (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                if self.stream:
                    self._set_stream_frame(vis)

                if self.show:
                    cv2.imshow("Brick Detection Camera (Q/ESC to quit)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        quit_requested = True

                if self.max_frames and idx >= self.max_frames:
                    quit_requested = True

        except KeyboardInterrupt:
            self.log.info("Stopping (Ctrl+C).")
        finally:
            cap.release()
            if self.stream_server:
                self.stream_server.stop()
            cv2.destroyAllWindows()

        elapsed = time.monotonic() - t_start
        self.log.info(
            "Done: %d/%d frames detected (%.1f%%), %.1fs (%.1f fps)",
            detect_count, idx, 100 * detect_count / max(idx, 1),
            elapsed, idx / max(elapsed, 1e-9),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    stream_cfg = load_manual_training_config()
    default_stream_host = stream_cfg.get("stream_host", "127.0.0.1")
    default_stream_port = int(stream_cfg.get("stream_port", 5000))
    default_stream_fps = int(stream_cfg.get("stream_fps", 10))
    default_stream_jpeg_quality = int(stream_cfg.get("stream_jpeg_quality", 85))

    p = argparse.ArgumentParser(
        description="Brick face detector for pick-and-place robotics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", default=None,
        help="Optional input video/image path. If omitted, uses live camera.",
    )
    p.add_argument(
        "--mode", "-m", default="camera",
        choices=["camera", "video", "images", "image", "video_out"],
        help="camera (default) uses live camera, others process --input media.",
    )
    p.add_argument("--output", "-o", default=None,
                   help="Output directory for exported images/video")
    p.add_argument("--camera-index", type=int, default=0,
                   help="Camera index for live mode (default: 0)")
    p.add_argument("--camera-width", type=int, default=640,
                   help="Camera capture width in live mode")
    p.add_argument("--camera-height", type=int, default=480,
                   help="Camera capture height in live mode")
    p.add_argument("--stream", dest="stream", action="store_true",
                   help="Enable livestream server in live mode")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Disable livestream server in live mode")
    p.set_defaults(stream=True)
    p.add_argument("--stream-host", default=default_stream_host,
                   help="Livestream host")
    p.add_argument("--stream-port", type=int, default=default_stream_port,
                   help="Livestream port")
    p.add_argument("--stream-fps", type=int, default=default_stream_fps,
                   help="Livestream frames per second")
    p.add_argument("--stream-jpeg-quality", type=int, default=default_stream_jpeg_quality,
                   help="Livestream JPEG quality (1..100)")
    p.add_argument("--show", action="store_true",
                   help="Show local OpenCV preview window in live mode")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames in live mode (0 = run until stopped)")
    p.add_argument("--work-height", type=int, default=540,
                   help="Processing resolution height (lower=faster)")
    p.add_argument("--no-gabor", action="store_true",
                   help="Disable Gabor anisotropy filter (faster, less accurate)")
    p.add_argument("--temporal-alpha", type=float, default=0.65,
                   help="Temporal smoothing weight (1.0=no smoothing)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = DetectorConfig(
        work_height=args.work_height,
        temporal_alpha=args.temporal_alpha,
    )
    # If --no-gabor, raise the aniso threshold so it never filters anything
    if args.no_gabor:
        cfg.gabor_aniso_thresh = 999.0

    detector = BrickDetector(cfg)

    run_camera_mode = args.mode == "camera" or not args.input

    if run_camera_mode:
        processor = CameraProcessor(
            detector=detector,
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            stream=args.stream,
            show=args.show,
            stream_host=args.stream_host,
            stream_port=args.stream_port,
            stream_fps=args.stream_fps,
            stream_jpeg_quality=args.stream_jpeg_quality,
            max_frames=args.max_frames,
        )
        processor.run()
        return

    if not os.path.isfile(args.input):
        logging.error("Input file not found: %s", args.input)
        sys.exit(1)

    file_mode = "video" if args.mode == "camera" else args.mode
    processor = VideoProcessor(detector, args.input, file_mode, args.output)
    processor.run()


if __name__ == "__main__":
    main()
