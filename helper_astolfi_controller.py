#!/usr/bin/env python3
"""Coordinated Astolfi-style differential-drive controller helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class AstolfiGains:
    kp_d: float
    kd_d: float
    kp_h: float
    kd_h: float


@dataclass
class AstolfiState:
    prev_dist_err: float | None = None
    prev_head_err: float | None = None
    d_dist_err: float = 0.0
    d_head_err: float = 0.0


def overdamped_pd_gains(
    *,
    damping_ratio: float,
    dist_settle_time_s: float,
    heading_settle_time_s: float,
) -> AstolfiGains:
    """Return PD gains from a slightly-overdamped second-order target model."""
    zeta = max(1.01, float(damping_ratio))
    dist_ts = max(0.1, float(dist_settle_time_s))
    head_ts = max(0.1, float(heading_settle_time_s))
    omega_d = 4.0 / (zeta * dist_ts)
    omega_h = 4.0 / (zeta * head_ts)
    return AstolfiGains(
        kp_d=omega_d * omega_d,
        kd_d=2.0 * zeta * omega_d,
        kp_h=omega_h * omega_h,
        kd_h=2.0 * zeta * omega_h,
    )


def _filtered_derivative(previous: float | None, current: float, prior_d: float, dt_s: float, alpha: float) -> float:
    if previous is None:
        return 0.0
    raw = (float(current) - float(previous)) / max(1e-6, float(dt_s))
    a = max(0.0, min(1.0, float(alpha)))
    return (a * raw) + ((1.0 - a) * float(prior_d))


def astolfi_wheel_command(
    *,
    x_off_mm: float,
    distance_mm: float,
    stop_offset_mm: float,
    bearing_y_offset_mm: float | None,
    dt_s: float,
    gains: AstolfiGains,
    state: AstolfiState,
    derivative_alpha: float = 0.35,
    linear_limit: float = 1.0,
    angular_limit: float = 0.75,
    wheel_limit: float = 1.0,
    max_dist_derivative_m_s: float | None = None,
    max_head_derivative_rad_s: float | None = None,
) -> dict:
    """Compute coupled left/right commands in normalized -1..1 units."""
    dist_err = float(distance_mm) - float(stop_offset_mm)
    y_for_bearing = float(bearing_y_offset_mm) if bearing_y_offset_mm is not None else max(1.0, float(distance_mm))
    head_err = math.atan2(-float(x_off_mm), max(1.0, y_for_bearing))
    d_dist = _filtered_derivative(
        state.prev_dist_err,
        dist_err,
        state.d_dist_err,
        dt_s,
        derivative_alpha,
    )
    d_head = _filtered_derivative(
        state.prev_head_err,
        head_err,
        state.d_head_err,
        dt_s,
        derivative_alpha,
    )
    state.prev_dist_err = float(dist_err)
    state.prev_head_err = float(head_err)
    state.d_dist_err = float(d_dist)
    state.d_head_err = float(d_head)

    # Scale distance to meters for gain sanity; heading is already radians.
    dist_m = float(dist_err) / 1000.0
    d_dist_m = float(d_dist) / 1000.0
    if max_dist_derivative_m_s is not None:
        d_limit = abs(float(max_dist_derivative_m_s))
        d_dist_m = max(-d_limit, min(d_limit, d_dist_m))
    if max_head_derivative_rad_s is not None:
        h_limit = abs(float(max_head_derivative_rad_s))
        d_head = max(-h_limit, min(h_limit, float(d_head)))
    v = ((float(gains.kp_d) * dist_m) + (float(gains.kd_d) * d_dist_m)) * math.cos(head_err)
    omega = (float(gains.kp_h) * head_err) + (float(gains.kd_h) * d_head)
    v = max(-float(linear_limit), min(float(linear_limit), float(v)))
    omega = max(-float(angular_limit), min(float(angular_limit), float(omega)))

    left = v - omega
    right = v + omega
    peak = max(abs(left), abs(right), 1e-9)
    limit = max(0.01, float(wheel_limit))
    scale = 1.0
    if peak > limit:
        scale = limit / peak
        left *= scale
        right *= scale
    return {
        "dist_err_mm": float(dist_err),
        "head_err_rad": float(head_err),
        "d_dist_err_mm_s": float(d_dist),
        "d_head_err_rad_s": float(d_head),
        "v": float(v),
        "omega": float(omega),
        "left": float(left),
        "right": float(right),
        "saturation_scale": float(scale),
    }
