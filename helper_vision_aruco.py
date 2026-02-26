import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

@dataclass
class BrickPose:
    found: bool
    position: Tuple[float, float, float]  # x, y, z (mm)
    orientation: Tuple[float, float, float]  # roll, pitch, yaw (degrees)
    confidence: float
    marker_id: int
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None # (4, 2) image points
    brickAbove: bool = False
    brickBelow: bool = False

class ArucoBrickVision:
    def __init__(self, marker_size_mm: float = 20.0, debug: bool = True):
        self.marker_size = marker_size_mm
        self.debug = debug
        self.camera_matrix = None
        self.dist_coeffs = None
        self.debug_frame = None
        self.current_frame = None
        self.raw_frame = None
        self.cap = None
        self.camera_index = None
        
        # Stacking stability
        self.stack_history = deque(maxlen=6)
        
        # ArUco Setup
        # Use a stricter detector for reliable full-marker reads on 6x6 codes, plus a
        # looser fallback detector for edge/clipped candidates used by stack heuristics.
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = self._build_aruco_params(loose=False)
        self.aruco_loose_params = self._build_aruco_params(loose=True)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.loose_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_loose_params)
        self.last_detect_pass = "raw"
        
        # 3D Marker Points (Marker center is origin of its 3D space)
        # Assuming marker is flat on the brick face (Z=0)
        s = marker_size_mm / 2.0
        self.marker_points_3d = np.array([
            [-s,  s, 0], # Top Left
            [ s,  s, 0], # Top Right
            [ s, -s, 0], # Bottom Right
            [-s, -s, 0]  # Bottom Left
        ], dtype=np.float32)

        # Brick Dimensions (mm)
        self.BRICK_H = 48.0
        
        # Smoothing
        self.ALPHA = 0.2
        self.last_pose = None

    def _build_aruco_params(self, loose: bool) -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()

        # 6x6 markers are denser than 4x4 markers, so subpixel corner refinement helps
        # the decode and pose estimate remain stable at the same print size.
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        if hasattr(params, "cornerRefinementMinAccuracy"):
            params.cornerRefinementMinAccuracy = 0.01

        # Sweep more adaptive threshold window sizes to handle live-stream lighting shifts.
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 4

        # 6x6 codes benefit from a little more canonical sampling density.
        params.perspectiveRemovePixelPerCell = 6
        params.minCornerDistanceRate = 0.05
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 1

        if loose:
            # Loose profile keeps support for clipped/partial candidates.
            params.minMarkerPerimeterRate = 0.005
            params.polygonalApproxAccuracyRate = 0.08
            params.minSideLengthCanonicalImg = 8
        else:
            # Primary profile favors stable full detections for the new 6x6 markers.
            params.minMarkerPerimeterRate = 0.008
            params.polygonalApproxAccuracyRate = 0.05
            params.minSideLengthCanonicalImg = 16

        return params

    def _detect_markers_with_fallback(self, gray: np.ndarray):
        def _quad_bbox(quad):
            try:
                pts = np.asarray(quad, dtype=np.float32)
            except Exception:
                return None
            if pts.ndim == 3 and pts.shape[0] == 1:
                pts = pts[0]
            if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
                return None
            x, y, w, h = cv2.boundingRect(np.round(pts[:4]).astype(np.int32))
            return (float(x), float(y), float(max(1, w)), float(max(1, h)))

        def _bbox_overlap_frac(box_a, box_b):
            ax, ay, aw, ah = box_a
            bx, by, bw, bh = box_b
            ax2 = ax + aw
            ay2 = ay + ah
            bx2 = bx + bw
            by2 = by + bh
            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = float(iw * ih)
            area_a = max(1.0, float(aw * ah))
            return inter / area_a

        def _boxes_same_marker(box_a, box_b):
            if box_a is None or box_b is None:
                return False
            ov = max(_bbox_overlap_frac(box_a, box_b), _bbox_overlap_frac(box_b, box_a))
            if ov >= 0.35:
                return True
            ax, ay, aw, ah = box_a
            bx, by, bw, bh = box_b
            acx = ax + (aw * 0.5)
            acy = ay + (ah * 0.5)
            bcx = bx + (bw * 0.5)
            bcy = by + (bh * 0.5)
            center_dist = float(np.hypot(acx - bcx, acy - bcy))
            size_ref = max(8.0, min(max(aw, ah), max(bw, bh)))
            return center_dist <= (0.35 * float(size_ref))

        # For 6x6 markers near the frame edges, one pass often decodes only the center
        # marker and leaves off-center markers as rejected candidates. Run all detector
        # passes and merge confirmed quads by geometry (not ID), because stack markers
        # intentionally reuse the same ID.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)

        pass_results = []
        for label, detector, img in (
            ("raw", self.detector, gray),
            ("clahe", self.detector, clahe_gray),
            ("raw_loose", self.loose_detector, gray),
            ("clahe_loose", self.loose_detector, clahe_gray),
        ):
            corners_i, ids_i, rejected_i = detector.detectMarkers(img)
            pass_results.append((label, corners_i, ids_i, rejected_i))

        merged_corners = []
        merged_corner_boxes = []
        merged_ids = []
        primary_label = None
        merged_from_later = False

        for label, corners_i, ids_i, _rejected_i in pass_results:
            if ids_i is None or corners_i is None or len(ids_i) <= 0:
                continue
            if primary_label is None:
                primary_label = str(label)
            for idx in range(min(len(corners_i), len(ids_i))):
                quad = corners_i[idx]
                bbox = _quad_bbox(quad)
                if bbox is None:
                    continue
                duplicate = any(_boxes_same_marker(bbox, existing) for existing in merged_corner_boxes)
                if duplicate:
                    continue
                try:
                    marker_id = int(ids_i[idx][0])
                except Exception:
                    try:
                        marker_id = int(ids_i[idx])
                    except Exception:
                        marker_id = 0
                merged_corners.append(quad)
                merged_corner_boxes.append(bbox)
                merged_ids.append(marker_id)
                if primary_label is not None and str(label) != str(primary_label):
                    merged_from_later = True

        if merged_ids:
            best_rejected = None
            best_rejected_count = -1
            for _label, _corners, _ids, rejected_i in pass_results:
                count_i = len(rejected_i) if rejected_i is not None else 0
                if count_i > best_rejected_count:
                    best_rejected = rejected_i
                    best_rejected_count = int(count_i)
            if primary_label is None:
                primary_label = "raw"
            self.last_detect_pass = f"{primary_label}+merge" if merged_from_later else str(primary_label)
            return merged_corners, np.asarray(merged_ids, dtype=np.int32).reshape(-1, 1), best_rejected

        # No confirmed markers found; return whichever pass produced the richest rejected set
        # so the partial-stack fallback still has candidates to inspect.
        best = max(
            pass_results,
            key=lambda item: len(item[3]) if item[3] is not None else 0,
        )
        self.last_detect_pass = best[0]
        return best[1], best[2], best[3]

    def _candidate_camera_indices(self, preferred_index: Optional[int]) -> List[int]:
        candidates: List[int] = []

        if preferred_index is not None:
            candidates.append(int(preferred_index))

        env_index = os.getenv("LEIA_CAMERA_INDEX")
        if env_index:
            try:
                candidates.append(int(env_index))
            except ValueError:
                pass

        candidates.extend([0, 1, 2, 3])

        deduped: List[int] = []
        for idx in candidates:
            if idx not in deduped:
                deduped.append(idx)
        return deduped

    def _open_camera(self, index: int, width: int, height: int):
        backend_pref = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        for backend in (backend_pref, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
            if cap is not None:
                cap.release()
            if backend == cv2.CAP_ANY:
                break
        return None

    def init_camera(self, width: int = 640, height: int = 480, camera_index: Optional[int] = None):
        if self.cap is None or not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            tried_indices: List[int] = []
            for idx in self._candidate_camera_indices(camera_index):
                tried_indices.append(idx)
                cap = self._open_camera(idx, width, height)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.camera_index = idx
                    print(f"[VISION] ArUco camera opened on index {idx}")
                    break

            if self.cap is None or not self.cap.isOpened():
                self.cap = None
                self.camera_index = None
                print(f"[VISION] Unable to open camera. Tried indices: {tried_indices}")

        # Approximate camera matrix if none provided
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.init_camera()
            if self.cap is None or not self.cap.isOpened():
                return False, 0, -1, 0, 0, 0, False, False

        # _open_camera() already requests a 1-frame capture buffer. Extra grab()
        # calls here can block for additional frame periods on some webcams/V4L2
        # backends, which multiplies act-to-act latency because the control loop
        # reads vision several times between robot commands.
        ret, frame = self.cap.read()
        if not ret:
            return False, 0, -1, 0, 0, 0, False, False

        self.raw_frame = frame.copy()
        pose = self.process(frame)
        self.current_frame = self.debug_frame if self.debug_frame is not None else frame
        
        if not pose.found:
            return False, 0, 0, 0, 0, 0, False, False

        # found, angle, dist, offset_x, conf, cam_h, above, below
        return (
            True, 
            pose.orientation[1], # Yaw-ish
            pose.position[2],    # Z dist
            pose.position[0],    # X offset
            pose.confidence,
            pose.position[1],    # Y height (cam relative)
            pose.brickAbove,
            pose.brickBelow
        )

    def close(self):
        if self.cap:
            self.cap.release()

    def process(self, frame: np.ndarray) -> BrickPose:
        if self.camera_matrix is None:
            self.init_camera(frame.shape[1], frame.shape[0])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detect_markers_with_fallback(gray)
        
        if self.debug:
            self.debug_frame = frame.copy()
            # Removed drawDetectedMarkers (dots)
            def _norm_pts(quad):
                try:
                    pts = np.asarray(quad, dtype=np.float32)
                except Exception:
                    return None
                if pts.ndim == 3 and pts.shape[0] == 1:
                    pts = pts[0]
                if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
                    return None
                return pts

            def _bbox_overlap_frac(box_a, box_b):
                ax, ay, aw, ah = box_a
                bx, by, bw, bh = box_b
                ax2 = ax + max(0, aw)
                ay2 = ay + max(0, ah)
                bx2 = bx + max(0, bw)
                by2 = by + max(0, bh)
                ix1 = max(ax, bx)
                iy1 = max(ay, by)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = float(iw * ih)
                area_a = float(max(1, aw) * max(1, ah))
                return inter / area_a if area_a > 0 else 0.0

            def _draw_poly(pts, color, thickness=2):
                poly = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.debug_frame, [poly], True, color, int(thickness), cv2.LINE_AA)

            def _order_quad(pts):
                if pts is None:
                    return None
                pts = np.asarray(pts, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
                    return None
                pts4 = pts[:4].copy()
                s = pts4.sum(axis=1)
                d = (pts4[:, 1] - pts4[:, 0])
                ordered = np.zeros((4, 2), dtype=np.float32)
                ordered[0] = pts4[np.argmin(s)]  # top-left
                ordered[2] = pts4[np.argmax(s)]  # bottom-right
                ordered[1] = pts4[np.argmin(d)]  # top-right
                ordered[3] = pts4[np.argmax(d)]  # bottom-left
                return ordered

            def _marker_like_bw_partial(pts):
                # Strengthen orange highlights using the known ArUco structure:
                # square-ish, high contrast, black/white only, dark outer border,
                # and multiple binary transitions through the interior pattern.
                try:
                    quad = _order_quad(pts)
                    if quad is None:
                        return False
                    warp_n = 56
                    dst = np.array(
                        [[0, 0], [warp_n - 1, 0], [warp_n - 1, warp_n - 1], [0, warp_n - 1]],
                        dtype=np.float32,
                    )
                    M = cv2.getPerspectiveTransform(quad, dst)
                    patch = cv2.warpPerspective(gray, M, (warp_n, warp_n), flags=cv2.INTER_LINEAR)
                    if patch is None or patch.size == 0:
                        return False

                    p5, p95 = np.percentile(patch, (5, 95))
                    if float(p95 - p5) < 40.0:
                        return False

                    _thr, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bw01 = (bw > 0).astype(np.uint8)
                    white_frac = float(np.mean(bw01))
                    if not (0.08 <= white_frac <= 0.92):
                        return False

                    inner_m = max(4, int(warp_n // 6))
                    if (warp_n - 2 * inner_m) <= 4:
                        return False
                    inner = bw01[inner_m:warp_n - inner_m, inner_m:warp_n - inner_m]
                    inner_white = float(np.mean(inner))
                    if not (0.05 <= inner_white <= 0.95):
                        return False

                    t = max(2, int(warp_n // 10))
                    border_black = [
                        1.0 - float(np.mean(bw01[:t, :])),
                        1.0 - float(np.mean(bw01[-t:, :])),
                        1.0 - float(np.mean(bw01[:, :t])),
                        1.0 - float(np.mean(bw01[:, -t:])),
                    ]
                    dark_sides = sum(1 for v in border_black if v >= 0.55)
                    ring_mask = np.zeros_like(bw01, dtype=np.uint8)
                    ring_mask[:t, :] = 1
                    ring_mask[-t:, :] = 1
                    ring_mask[:, :t] = 1
                    ring_mask[:, -t:] = 1
                    ring_black_ratio = 1.0 - float(np.mean(bw01[ring_mask == 1]))
                    if ring_black_ratio < 0.45 and dark_sides < 2:
                        return False

                    row = bw01[warp_n // 2, :]
                    col = bw01[:, warp_n // 2]
                    transitions = int(np.count_nonzero(row[1:] != row[:-1])) + int(np.count_nonzero(col[1:] != col[:-1]))
                    if transitions < 4:
                        return False

                    return True
                except Exception:
                    return False

            confirmed_contours = []
            confirmed_bboxes = []
            confirmed_short_sides = []

            if corners is not None:
                for quad in corners:
                    pts = _norm_pts(quad)
                    if pts is None:
                        continue
                    _draw_poly(pts, (0, 255, 0), thickness=2)  # bright green
                    confirmed_contours.append(pts.reshape((-1, 1, 2)).astype(np.float32))
                    confirmed_bboxes.append(cv2.boundingRect(np.round(pts).astype(np.int32)))
                    rect = cv2.minAreaRect(pts)
                    rw, rh = rect[1]
                    if rw > 0 and rh > 0:
                        confirmed_short_sides.append(float(min(rw, rh)))

            if rejected is not None:
                h, w = frame.shape[:2]
                frame_area = float(max(1, h * w))
                min_area_px = max(40.0, frame_area * 0.00005)
                max_area_px = frame_area * 0.25
                ref_short_side = float(np.median(confirmed_short_sides)) if confirmed_short_sides else None

                for quad in rejected:
                    pts = _norm_pts(quad)
                    if pts is None:
                        continue

                    area = abs(float(cv2.contourArea(pts)))
                    if area < min_area_px or area > max_area_px:
                        continue

                    rect = cv2.minAreaRect(pts)
                    (cx, cy), (rw, rh), _ = rect
                    if rw <= 0 or rh <= 0:
                        continue
                    short_side = float(min(rw, rh))
                    long_side = float(max(rw, rh))
                    if short_side < 6.0:
                        continue
                    if ref_short_side is not None and short_side < max(6.0, ref_short_side * 0.18):
                        # Suppress tiny rejected quads inside a confirmed marker (cell artifacts).
                        continue

                    aspect = long_side / max(short_side, 1e-3)
                    if aspect > 2.4:
                        continue

                    rect_area = float(rw * rh)
                    if rect_area <= 1.0:
                        continue
                    fill_ratio = area / rect_area
                    if fill_ratio < 0.45:
                        continue
                    if not _marker_like_bw_partial(pts):
                        continue

                    center_pt = (float(cx), float(cy))
                    inside_confirmed = False
                    for contour in confirmed_contours:
                        try:
                            if cv2.pointPolygonTest(contour, center_pt, False) >= 0:
                                inside_confirmed = True
                                break
                        except Exception:
                            continue
                    if inside_confirmed:
                        continue

                    cand_bbox = cv2.boundingRect(np.round(pts).astype(np.int32))
                    if any(_bbox_overlap_frac(cand_bbox, box) > 0.35 for box in confirmed_bboxes):
                        continue

                    box_pts = cv2.boxPoints(rect)
                    _draw_poly(box_pts, (0, 165, 255), thickness=2)  # bright orange

        best_marker = None
        
        if ids is not None:
            # Find centermost marker
            min_dist = float('inf')
            img_center = (frame.shape[1] / 2, frame.shape[0] / 2)
            
            poses = []
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                c = corners[i][0]
                
                # Estimate pose
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_points_3d, c, self.camera_matrix, self.dist_coeffs
                )
                
                if success:
                    # image space distance to center
                    mcx = np.mean(c[:, 0])
                    mcy = np.mean(c[:, 1])
                    dist = np.sqrt((mcx - img_center[0])**2 + (mcy - img_center[1])**2)
                    
                    rmat, _ = cv2.Rodrigues(rvec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    
                    pose = BrickPose(
                        found=True,
                        position=(tvec[0][0], tvec[1][0], tvec[2][0]),
                        orientation=angles,
                        confidence=100.0,
                        marker_id=marker_id,
                        rvec=rvec,
                        tvec=tvec,
                        corners=c
                    )
                    poses.append(pose)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_marker = pose

            # Detect stacking
            if best_marker:
                best_mcx = best_mcy = None
                best_size_px = None
                if best_marker.corners is not None:
                    best_mcx = float(np.mean(best_marker.corners[:, 0]))
                    best_mcy = float(np.mean(best_marker.corners[:, 1]))
                    best_size_px = float(np.linalg.norm(best_marker.corners[0] - best_marker.corners[2]) / 1.414)
                focal = float(self.camera_matrix[0, 0]) if self.camera_matrix is not None else 0.0
                best_dist_mm = max(1e-3, float(best_marker.position[2]))
                expected_shift_px = (focal * (self.BRICK_H / best_dist_mm)) if focal > 0 else 0.0

                # 1. Check verified markers
                verified_stack_hits = 0
                for pose in poses:
                    # Compare against the selected center-most marker instance.
                    # Marker IDs can repeat across bricks, so ID equality is not a safe
                    # way to skip only the reference brick.
                    if pose is best_marker:
                        continue
                    dz_signed = float(pose.position[2] - best_marker.position[2])
                    dz = abs(dz_signed)
                    dy = float(pose.position[1] - best_marker.position[1])
                    dx = abs(float(pose.position[0] - best_marker.position[0]))
                    yz_sep = float(np.hypot(dy, dz_signed))

                    # In low-camera / pitched views, the 48mm brick-height offset rotates into
                    # both camera Y and Z. Use the combined YZ separation instead of only dy.
                    stack_like_3d = (
                        dx < 35.0 and
                        20.0 < yz_sep < 90.0 and
                        dz < 85.0
                    )

                    mcx = mcy = None
                    idx_px = idy_px = None
                    stack_like_img = True
                    if pose.corners is not None and best_mcx is not None and best_mcy is not None:
                        mcx = float(np.mean(pose.corners[:, 0]))
                        mcy = float(np.mean(pose.corners[:, 1]))
                        idx_px = abs(mcx - best_mcx)
                        idy_px = float(best_mcy - mcy)  # positive => other marker is above

                        size_px = float(np.linalg.norm(pose.corners[0] - pose.corners[2]) / 1.414)
                        ref_size_px = max(best_size_px or 0.0, size_px, 1.0)
                        lateral_tol_px = max(12.0, ref_size_px * 1.25)
                        min_vertical_shift_px = max(8.0, ref_size_px * 0.35)
                        # Keep a wide upper bound because camera pitch compresses/expands
                        # apparent vertical spacing in image space.
                        max_vertical_shift_px = max(ref_size_px * 4.5, expected_shift_px * 2.0, 40.0)
                        stack_like_img = (
                            idx_px <= lateral_tol_px and
                            min_vertical_shift_px <= abs(idy_px) <= max_vertical_shift_px
                        )

                    if stack_like_3d and stack_like_img:
                        verified_stack_hits += 1
                        if idy_px is not None:
                            if idy_px > 0:
                                best_marker.brickAbove = True
                            elif idy_px < 0:
                                best_marker.brickBelow = True
                        else:
                            if dy < 0:
                                best_marker.brickAbove = True
                            elif dy > 0:
                                best_marker.brickBelow = True

                # 2. Check rejected markers for partial bricks
                # Disabled for stack booleans: `brickAbove` / `brickBelow` should reflect
                # only confirmed (green) ArUco markers so the HUD booleans match what the
                # operator sees highlighted. If only one marker is confirmed, both remain
                # false here and the higher-level stack gate checker will continue waiting.
                use_rejected_stack_fallback = False
                if use_rejected_stack_fallback and rejected and (not best_marker.brickAbove or not best_marker.brickBelow):
                    dist_mm = best_dist_mm
                    expected_shift_px = focal * (self.BRICK_H / dist_mm)
                    expected_size_px = focal * (self.marker_size / dist_mm)
                    rejected_above_best = None
                    rejected_below_best = None
                    
                    # Project center to image space
                    best_img_pts, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float32), 
                                                       best_marker.rvec, best_marker.tvec, 
                                                       self.camera_matrix, self.dist_coeffs)
                    bcx, bcy = best_img_pts[0][0]

                    for rej in rejected:
                        corners = rej[0]
                        mcx = np.mean(corners[:, 0])
                        # Bottom-most point of this candidate (highest Y value in pixels)
                        max_y = np.max(corners[:, 1])
                        min_y = np.min(corners[:, 1])
                        
                        size_px = np.linalg.norm(corners[0] - corners[2]) / 1.414 
                        
                        # For clipped markers, size will be smaller. 
                        # Allow tiny size if it's right at the edge.
                        is_at_edge = (min_y < 10 or max_y > frame.shape[0] - 10)
                        min_size_f = 0.2 if is_at_edge else 0.4
                        if not (expected_size_px * min_size_f < size_px < expected_size_px * 1.6): continue
                        
                        idx = abs(mcx - bcx)
                        # idy based on bottom edge vs center
                        # expected shift to bottom edge is roughly 38mm (48 - 10)
                        dy_bottom = bcy - max_y # Correct: if max_y is above bcy, dy is positive
                        # expected_shift_bottom = focal * (38 / dist_mm)
                        # But simpler: use center if size is okay, or just check alignment
                        
                        lateral_tol_px = max(expected_shift_px * 0.7, (best_size_px or expected_size_px) * 1.25, 12.0)
                        if idx < lateral_tol_px: # Loosened alignment for pitched views
                            # Check "Above" (Positive idy)
                            # center-center shift is 48mm. 
                            # If clipped at top, max_y (bottom edge) is the most reliable anchor.
                            # expected max_y for upper brick: bcy - (focal * (38 / dist_mm))
                            idy_center = bcy - np.mean(corners[:, 1])
                            
                            # Shifted check: if center is visible, use center.
                            # If clipped, idy_center will be lower than expected center-center shift.
                            # But max_y should still be at least ~28mm above center.
                            dist_to_bottom = bcy - max_y
                            expected_dist_to_bottom = focal * (38.0 / dist_mm)
                            
                            min_above_shift = max(expected_size_px * 0.35, 8.0)
                            max_above_shift = max(expected_dist_to_bottom * 1.8, expected_size_px * 5.0, 40.0)
                            if (min_above_shift < dist_to_bottom < max_above_shift):
                                above_err = abs(float(dist_to_bottom) - float(expected_dist_to_bottom))
                                above_score = (
                                    float(above_err) / max(8.0, float(expected_size_px)),
                                    float(idx) / max(1.0, float(lateral_tol_px)),
                                )
                                if rejected_above_best is None or above_score < rejected_above_best:
                                    rejected_above_best = above_score
                            
                            # Check "Below" (Negative dy)
                            dist_to_top = bcy - min_y # Negative if below
                            expected_dist_to_top = focal * (-38.0 / dist_mm)
                            
                            min_below_shift = -max(expected_size_px * 0.35, 8.0)
                            max_below_shift = -max(expected_dist_to_bottom * 1.8, expected_size_px * 5.0, 40.0)
                            if (max_below_shift < dist_to_top < min_below_shift):
                                below_err = abs(float(dist_to_top) - float(expected_dist_to_top))
                                below_score = (
                                    float(below_err) / max(8.0, float(expected_size_px)),
                                    float(idx) / max(1.0, float(lateral_tol_px)),
                                )
                                if rejected_below_best is None or below_score < rejected_below_best:
                                    rejected_below_best = below_score

                    # Rejected (orange) quads are only a fallback when we had zero verified
                    # stack hits this frame. In that mode, asserting both sides from partials
                    # is too prone to false positives; keep only the stronger side.
                    if rejected_above_best is not None and rejected_below_best is not None:
                        if rejected_above_best < rejected_below_best:
                            if not best_marker.brickAbove:
                                best_marker.brickAbove = True
                        elif rejected_below_best < rejected_above_best:
                            if not best_marker.brickBelow:
                                best_marker.brickBelow = True
                    else:
                        if rejected_above_best is not None and not best_marker.brickAbove:
                            best_marker.brickAbove = True
                        if rejected_below_best is not None and not best_marker.brickBelow:
                            best_marker.brickBelow = True

        if best_marker:
            if self.last_pose:
                # Smoothing
                p = []
                for i in range(3):
                    p.append(self.ALPHA * best_marker.position[i] + (1 - self.ALPHA) * self.last_pose.position[i])
                best_marker.position = tuple(p)
            self.last_pose = best_marker
            return best_marker
        else:
            return BrickPose(False, (0, 0, 0), (0, 0, 0), 0.0, -1)
