"""Fresh, testable visual-servo controller for Leia's direct Uno protocol.

This module intentionally contains no camera or serial code.  It consumes
timestamped x/dist measurements and returns differential tread commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import atan2, cos
from statistics import median


@dataclass(frozen=True)
class ServoConfig:
    # Coordinate/wiring switches are centralized here.  The current Uno sketch
    # uses positive left/right values for forward tread motion.  The measured
    # The detector defines positive x as brick-to-image-right.  With the Uno
    # sketch, positive left/right values are forward; standard differential
    # drive therefore needs a negative turn sign to steer toward +x.
    invert_x: bool = False
    invert_left_motor: bool = False
    invert_right_motor: bool = False
    swap_left_right_motors: bool = False
    turn_sign: float = -1.0
    target_dist_mm: float = 100.0
    x_tolerance_mm: float = 5.0
    dist_tolerance_mm: float = 10.0
    min_dist_mm: float = 35.0
    max_dist_mm: float = 2000.0
    min_confidence: float = 0.65
    max_x_jump_mm: float = 80.0
    max_dist_jump_mm: float = 180.0
    max_measurement_age_s: float = 0.20
    filter_window: int = 3
    smoothing_alpha: float = 0.35
    kp_dist: float = 0.18
    kd_dist: float = 0.03
    kp_heading: float = 42.0
    kd_heading: float = 5.0
    left_min_moving_pct: float = 25.0
    right_min_moving_pct: float = 25.0
    max_tread_pct: float = 35.0
    max_command_change_per_second: float = 80.0
    max_reversal_change_per_second: float = 55.0
    super_sharp_x_mm: float = 40.0
    sharp_x_mm: float = 35.0
    semi_gentle_x_mm: float = 20.0
    gentle_x_mm: float = 10.0
    super_sharp_inner_pct: float = 0.0
    sharp_inner_pct: float = 25.0
    semi_gentle_inner_pct: float = 30.0
    gentle_inner_pct: float = 30.0
    super_gentle_inner_pct: float = 35.0
    acquisition_frames: int = 3
    aligned_frames: int = 5


@dataclass(frozen=True)
class BrickMeasurement:
    x_mm: float
    dist_mm: float
    confidence: float
    timestamp: float


@dataclass(frozen=True)
class DriveCommand:
    left_pct: float
    right_pct: float
    state: str
    valid: bool
    rejection_reason: str = ""


class TargetState(str, Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    ALIGNED = "ALIGNED"


class VisualServoController:
    def __init__(self, config: ServoConfig = ServoConfig()):
        self.config = config
        self.state = TargetState.SEARCHING
        self._history: deque[BrickMeasurement] = deque(maxlen=config.filter_window)
        self._filtered_x = None
        self._filtered_dist = None
        self._previous = None
        self._x_rate = 0.0
        self._dist_rate = 0.0
        self._previous_command = (0.0, 0.0)
        self._previous_command_time = None
        self._good_frames = 0
        self._aligned_frames = 0

    def _reset_tracking(self) -> None:
        self._history.clear()
        self._filtered_x = None
        self._filtered_dist = None
        self._previous = None
        self._x_rate = 0.0
        self._dist_rate = 0.0
        self._good_frames = 0
        self._aligned_frames = 0
        self._previous_command = (0.0, 0.0)
        self._previous_command_time = None

    def _reject_reason(self, m: BrickMeasurement, now: float) -> str:
        c = self.config
        if m.confidence < c.min_confidence:
            return "low_confidence"
        if m.timestamp > now + 0.05:
            return "future_timestamp"
        if not (c.min_dist_mm <= m.dist_mm <= c.max_dist_mm):
            return "distance_out_of_range"
        if now - m.timestamp > c.max_measurement_age_s:
            return "stale"
        if self._history:
            prev = self._history[-1]
            dt = max(0.001, m.timestamp - prev.timestamp)
            if abs(m.x_mm - prev.x_mm) > c.max_x_jump_mm + 250.0 * dt:
                return "x_jump"
            if abs(m.dist_mm - prev.dist_mm) > c.max_dist_jump_mm + 600.0 * dt:
                return "dist_jump"
        return ""

    def _slew(self, requested: float, previous: float, dt: float) -> float:
        limit = self.config.max_command_change_per_second * max(dt, 0.001)
        if requested * previous < 0:
            limit = self.config.max_reversal_change_per_second * max(dt, 0.001)
        delta = max(-limit, min(limit, requested - previous))
        return previous + delta

    def _compensate(self, value: float, minimum: float) -> float:
        if value == 0.0:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * max(abs(value), minimum)

    def _arc_commands(self, drive: float, x_error: float) -> tuple[float, float]:
        """Choose a forward/reverse arc from lateral error magnitude.

        Positive x is image-right.  A negative x error therefore makes the
        left tread the inner/slow tread under the configured turn sign.
        Keeping both treads on the same drive sign avoids an uncalibrated
        in-place pivot at crawl power.
        """
        c = self.config
        magnitude = abs(x_error)
        if magnitude >= c.super_sharp_x_mm:
            inner = c.super_sharp_inner_pct
        elif magnitude >= c.sharp_x_mm:
            inner = c.sharp_inner_pct
        elif magnitude >= c.semi_gentle_x_mm:
            inner = c.semi_gentle_inner_pct
        elif magnitude >= c.gentle_x_mm:
            inner = c.gentle_inner_pct
        else:
            inner = c.super_gentle_inner_pct
        outer = c.max_tread_pct
        sign = 1.0 if drive >= 0.0 else -1.0
        if x_error < 0.0:
            return sign * inner, sign * outer
        return sign * outer, sign * inner

    def update(self, measurement: BrickMeasurement | None, now: float) -> DriveCommand:
        if measurement is None:
            self.state = TargetState.UNCERTAIN
            self._reset_tracking()
            return DriveCommand(0.0, 0.0, self.state.value, False, "no_measurement")

        reason = self._reject_reason(measurement, now)
        if reason:
            self.state = TargetState.UNCERTAIN
            self._reset_tracking()
            return DriveCommand(0.0, 0.0, self.state.value, False, reason)

        self._history.append(measurement)
        xs = [item.x_mm for item in self._history]
        ds = [item.dist_mm for item in self._history]
        median_x, median_dist = median(xs), median(ds)
        a = self.config.smoothing_alpha
        self._filtered_x = median_x if self._filtered_x is None else a * median_x + (1 - a) * self._filtered_x
        self._filtered_dist = median_dist if self._filtered_dist is None else a * median_dist + (1 - a) * self._filtered_dist

        if self._previous is not None:
            dt = measurement.timestamp - self._previous[2]
            if 0.005 <= dt <= 0.25:
                raw_x_rate = (self._filtered_x - self._previous[0]) / dt
                raw_dist_rate = (self._filtered_dist - self._previous[1]) / dt
                self._x_rate = 0.25 * raw_x_rate + 0.75 * self._x_rate
                self._dist_rate = 0.25 * raw_dist_rate + 0.75 * self._dist_rate
        self._previous = (self._filtered_x, self._filtered_dist, measurement.timestamp)
        self._good_frames += 1
        if self._good_frames < self.config.acquisition_frames:
            self.state = TargetState.ACQUIRING
            return DriveCommand(0.0, 0.0, self.state.value, True)

        x_error = -self._filtered_x if self.config.invert_x else self._filtered_x
        dist_error = self._filtered_dist - self.config.target_dist_mm
        aligned = abs(x_error) <= self.config.x_tolerance_mm and abs(dist_error) <= self.config.dist_tolerance_mm
        self._aligned_frames = self._aligned_frames + 1 if aligned else 0
        if self._aligned_frames >= self.config.aligned_frames:
            self.state = TargetState.ALIGNED
            self._previous_command = (0.0, 0.0)
            return DriveCommand(0.0, 0.0, self.state.value, True)

        self.state = TargetState.TRACKING
        heading = atan2(x_error, max(self._filtered_dist, self.config.min_dist_mm))
        v = (self.config.kp_dist * dist_error) - (self.config.kd_dist * self._dist_rate)
        x_rate = -self._x_rate if self.config.invert_x else self._x_rate
        omega = self.config.turn_sign * ((self.config.kp_heading * heading) - (self.config.kd_heading * x_rate / max(self._filtered_dist, 1.0)))
        v *= max(0.0, cos(heading))
        # The decision system uses x magnitude to select an arc family.  The
        # heading term still influences the selected side, while distance
        # error supplies the common forward/reverse direction.
        if abs(v) < 0.5 and abs(x_error) > self.config.x_tolerance_mm:
            v = 1.0 if dist_error >= 0.0 else -1.0
        left, right = self._arc_commands(v, x_error)
        left = self._compensate(left, self.config.left_min_moving_pct)
        right = self._compensate(right, self.config.right_min_moving_pct)
        if self.config.invert_left_motor:
            left = -left
        if self.config.invert_right_motor:
            right = -right
        if self.config.swap_left_right_motors:
            left, right = right, left
        peak = max(abs(left), abs(right), 1.0)
        if peak > self.config.max_tread_pct:
            scale = self.config.max_tread_pct / peak
            left, right = left * scale, right * scale
        dt = now - (self._previous_command_time if self._previous_command_time is not None else now)
        self._previous_command_time = now
        left = self._slew(left, self._previous_command[0], dt)
        right = self._slew(right, self._previous_command[1], dt)
        self._previous_command = (left, right)
        return DriveCommand(left, right, self.state.value, True)
