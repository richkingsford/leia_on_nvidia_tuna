#!/usr/bin/env python3
"""Generate a compact static SVG plot for a trial manifest."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


SVG_WIDTH = 1220
SVG_HEIGHT = 760
PLOT_LEFT = 88
PLOT_TOP = 118
PLOT_RIGHT = 860
PLOT_BOTTOM = 618
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT
PLOT_HEIGHT = PLOT_BOTTOM - PLOT_TOP
TURN_RESULT_EPSILON_MM = 0.25


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_text(value, digits: int = 3) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _color_for_result(value: float) -> str:
    if float(value) > TURN_RESULT_EPSILON_MM:
        return "#0d7c66"
    if float(value) < -TURN_RESULT_EPSILON_MM:
        return "#bf3b2b"
    return "#c1811f"


def _forward_only_rows(manifest: dict) -> list[dict]:
    curve = dict(manifest.get("curve") or {})
    setup_phase = dict(curve.get("setup_phase") or {})
    setup_pair = dict(setup_phase.get("motor_pair") or {})
    expected_curve_pair = str(curve.get("curve_pair_label") or "").strip()
    adaptive_curve_pairs = bool(str(curve.get("turn_selection_policy") or "").strip()) or str(
        curve.get("sequence_style") or ""
    ).strip().lower().startswith("adaptive_")
    setup_range_cfg = dict(curve.get("setup_turn_duration_range_ms") or {})
    setup_min_ms = _coerce_float(setup_range_cfg.get("min_ms"))
    setup_max_ms = _coerce_float(setup_range_cfg.get("max_ms"))
    default_left = _coerce_float(setup_pair.get("left_motor_pwm"))
    default_right = _coerce_float(setup_pair.get("right_motor_pwm"))
    rows = []

    source_rows = list(manifest.get("trials_backwards") or [])
    if not source_rows:
        source_rows = list(manifest.get("trials") or [])
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        row_status = str(row.get("status") or "").strip().lower()
        if row_status and row_status != "completed":
            continue
        if row.get("usable") is not True:
            continue
        row_curve_pair = str(row.get("curvePair") or "").strip()
        if (
            not adaptive_curve_pairs
            and expected_curve_pair
            and row_curve_pair
            and row_curve_pair != expected_curve_pair
        ):
            continue

        measured_duration = _coerce_float(row.get("setupDurationMs"))
        if measured_duration is None:
            measured_duration = _coerce_float(row.get("measuredDurationMs"))
        if measured_duration is None:
            measured_duration = _coerce_float(row.get("measured_duration"))
        if measured_duration is None:
            continue
        if setup_min_ms is not None and measured_duration < float(setup_min_ms):
            continue
        if setup_max_ms is not None and measured_duration > float(setup_max_ms):
            continue

        x_gap_closed = _coerce_float(row.get("xGapClosed"))
        if x_gap_closed is None:
            x_gap_closed = _coerce_float(row.get("xDelta"))
        dist_delta = _coerce_float(row.get("distDelta"))
        if x_gap_closed is None or dist_delta is None:
            continue

        forward_left = _coerce_float(row.get("setupLeftMotorPwr"))
        if forward_left is None:
            forward_left = _coerce_float(row.get("setupPwm"))
        if forward_left is None:
            forward_left = default_left
        forward_right = _coerce_float(row.get("setupRightMotorPwr"))
        if forward_right is None:
            forward_right = default_right
        if forward_left is None or forward_right is None:
            continue

        avg_forward_pwm = (float(forward_left) + float(forward_right)) / 2.0
        rows.append(
            {
                "trial": int(row.get("trial") or 0),
                "measured_duration_ms": int(round(float(measured_duration))),
                "avg_forward_pwm": float(avg_forward_pwm),
                "forward_left_pwm": int(round(float(forward_left))),
                "forward_right_pwm": int(round(float(forward_right))),
                "x_gap_closed_mm": abs(float(x_gap_closed)),
                "dist_delta_mm": float(dist_delta),
                "start_dist_mm": float(_coerce_float(row.get("startDist")) or 0.0),
                "start_conf_pct": float(_coerce_float(row.get("startConf")) or 0.0),
                "end_conf_pct": float(_coerce_float(row.get("endConf")) or 0.0),
                "curve_pair": str(row.get("curvePair") or ""),
            }
        )

    rows.sort(key=lambda item: (item["measured_duration_ms"], item["trial"]))
    return rows


def _scale_builder(min_val: float, max_val: float, out_low: float, out_high: float):
    if abs(max_val - min_val) < 1e-9:
        midpoint = (out_low + out_high) / 2.0

        def _constant(_value: float) -> float:
            return midpoint

        return _constant

    ratio = (out_high - out_low) / (max_val - min_val)

    def _scale(value: float) -> float:
        return out_low + (value - min_val) * ratio

    return _scale


def _bubble_radius(dist_delta_mm: float, *, max_abs_dist_mm: float) -> float:
    magnitude = abs(float(dist_delta_mm))
    if max_abs_dist_mm <= 0.0:
        return 12.0
    return 9.0 + (magnitude / max_abs_dist_mm) * 18.0


def _tick_values(min_val: float, max_val: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if abs(max_val - min_val) < 1e-9:
        return [float(min_val)]
    if count == 1:
        return [float(min_val)]
    step = (max_val - min_val) / float(count - 1)
    return [float(min_val + (step * idx)) for idx in range(count)]


def _build_svg(manifest: dict, *, source_path: Path) -> str:
    curve = dict(manifest.get("curve") or {})
    rows = _forward_only_rows(manifest)
    if not rows:
        raise ValueError("No complete plottable rows found in manifest.")

    x_values = [row["measured_duration_ms"] for row in rows]
    y_values = [row["x_gap_closed_mm"] for row in rows]
    max_abs_dist = max(abs(row["dist_delta_mm"]) for row in rows)

    x_pad = max(1.0, (max(x_values) - min(x_values)) * 0.08)
    y_min_raw = min(min(y_values), 0.0)
    y_max_raw = max(max(y_values), 0.0)
    y_pad = max(0.18, (y_max_raw - y_min_raw) * 0.22)

    x_min = min(x_values) - x_pad
    x_max = max(x_values) + x_pad
    y_min = y_min_raw - y_pad
    y_max = y_max_raw + y_pad

    x_scale = _scale_builder(x_min, x_max, PLOT_LEFT, PLOT_RIGHT)
    y_scale = _scale_builder(y_min, y_max, PLOT_BOTTOM, PLOT_TOP)

    polyline_points = " ".join(
        f"{x_scale(row['measured_duration_ms']):.1f},{y_scale(row['x_gap_closed_mm']):.1f}"
        for row in rows
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">trial duration plot</title>",
        "<desc id=\"desc\">Forward setup duration in milliseconds on the horizontal axis, forward-only xDelta in millimeters on the vertical axis, and circle diameter representing absolute distance traveled.</desc>",
        "<defs>",
        "  <style>",
        "    .title { fill: #18242c; font: 900 34px 'Arial', sans-serif; letter-spacing: -0.03em; }",
        "    .subtitle { fill: #5a6770; font: 14px 'Arial', sans-serif; }",
        "    .axis-label { fill: #5a6770; font: 700 13px 'Arial', sans-serif; letter-spacing: 0.08em; text-transform: uppercase; }",
        "    .tick { fill: #5a6770; font: 12px 'Arial', sans-serif; }",
        "    .panel-title { fill: #18242c; font: 700 15px 'Arial', sans-serif; letter-spacing: 0.06em; text-transform: uppercase; }",
        "    .panel-body { fill: #18242c; font: 13px 'Arial', sans-serif; }",
        "    .panel-muted { fill: #5a6770; font: 12px 'Arial', sans-serif; }",
        "  </style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#f4efe4" />',
        '<rect x="18" y="18" width="1184" height="724" rx="28" fill="#fffaf2" stroke="#d8cdb7" stroke-width="1.5" />',
        f'<text x="{PLOT_LEFT}" y="56" class="title">Forward Duration vs Forward X Delta</text>',
        f'<text x="{PLOT_LEFT}" y="82" class="subtitle">{_escape(manifest.get("name") or source_path.stem)} • {_escape(curve.get("curve_pair_label") or "")}</text>',
        f'<text x="{PLOT_LEFT}" y="100" class="subtitle">X axis is setup forward duration (ms). Dot diameter = |distDelta| in mm.</text>',
        f'<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" rx="22" fill="#fff7eb" stroke="#eadfca" stroke-width="1.2" />',
    ]

    for tick_val in _tick_values(y_min, y_max, 7):
        y = y_scale(tick_val)
        stroke = "#9f9688" if abs(tick_val) < 1e-9 else "#ddd1be"
        dash = "" if abs(tick_val) < 1e-9 else ' stroke-dasharray="5 8"'
        width = "1.5" if abs(tick_val) < 1e-9 else "1"
        svg.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash} />'
        )
        svg.append(
            f'<text x="{PLOT_LEFT - 14}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_round_text(tick_val, 2)}</text>'
        )

    x_ticks = _tick_values(x_min, x_max, 7)
    for tick_val in x_ticks:
        x = x_scale(tick_val)
        svg.append(
            f'<line x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}" '
            f'stroke="#efe4d2" stroke-width="1" />'
        )
        svg.append(
            f'<line x1="{x:.1f}" y1="{PLOT_BOTTOM}" x2="{x:.1f}" y2="{PLOT_BOTTOM + 8}" '
            f'stroke="#8f8372" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{PLOT_BOTTOM + 28}" text-anchor="end" transform="rotate(-32 {x:.1f} {PLOT_BOTTOM + 28})" '
            f'class="tick">{int(round(float(tick_val)))} ms</text>'
        )

    svg.extend(
        [
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="#897d6d" stroke-width="1.4" />',
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" stroke="#897d6d" stroke-width="1.4" />',
            f'<text x="{(PLOT_LEFT + PLOT_RIGHT) / 2:.1f}" y="{PLOT_BOTTOM + 74}" text-anchor="middle" class="axis-label">Forward Setup Duration (ms)</text>',
            f'<text x="26" y="{(PLOT_TOP + PLOT_BOTTOM) / 2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 26 {(PLOT_TOP + PLOT_BOTTOM) / 2:.1f})">Forward xDelta (mm)</text>',
            f'<polyline points="{polyline_points}" fill="none" stroke="#0d5c63" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.45" />',
        ]
    )

    for row in rows:
        x = x_scale(row["measured_duration_ms"])
        y = y_scale(row["x_gap_closed_mm"])
        radius = _bubble_radius(row["dist_delta_mm"], max_abs_dist_mm=max_abs_dist)
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{_color_for_result(row["x_gap_closed_mm"])}" '
            'fill-opacity="0.82" stroke="#18242c" stroke-opacity="0.28" stroke-width="1.4" />'
        )

    panel_x = 902
    panel_y = 108
    panel_w = 268
    panel_h = 516
    svg.extend(
        [
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="22" fill="#f8f0e2" stroke="#e2d3bd" stroke-width="1.2" />',
            f'<text x="{panel_x + 18}" y="{panel_y + 30}" class="panel-title">Legend</text>',
            f'<circle cx="{panel_x + 26}" cy="{panel_y + 58}" r="7" fill="#0d7c66" /><text x="{panel_x + 42}" y="{panel_y + 62}" class="panel-body">positive xDelta</text>',
            f'<circle cx="{panel_x + 26}" cy="{panel_y + 84}" r="7" fill="#bf3b2b" /><text x="{panel_x + 42}" y="{panel_y + 88}" class="panel-body">negative xDelta</text>',
            f'<circle cx="{panel_x + 26}" cy="{panel_y + 110}" r="7" fill="#c1811f" /><text x="{panel_x + 42}" y="{panel_y + 114}" class="panel-body">near zero</text>',
            f'<text x="{panel_x + 18}" y="{panel_y + 156}" class="panel-title">Bubble Diameter</text>',
        ]
    )

    bubble_legend_vals = [max_abs_dist * 0.3, max_abs_dist * 0.65, max_abs_dist]
    bubble_legend_vals = [round(val, 1) for val in bubble_legend_vals]
    bubble_base_y = panel_y + 194
    bubble_x = panel_x + 42
    for idx, value in enumerate(bubble_legend_vals):
        radius = _bubble_radius(value, max_abs_dist_mm=max_abs_dist)
        cy = bubble_base_y + (idx * 62)
        svg.append(
            f'<circle cx="{bubble_x}" cy="{cy}" r="{radius:.1f}" fill="#0d5c63" fill-opacity="0.12" stroke="#0d5c63" stroke-width="1.4" />'
        )
        svg.append(
            f'<text x="{panel_x + 92}" y="{cy + 4}" class="panel-body">{_round_text(value, 1)} mm</text>'
        )

    stats = dict(manifest.get("curve_stats") or {})
    stat_rows = [
        ("usable trials", stats.get("usable_trials")),
        ("close rate", f"{stats.get('x_gap_close_rate_pct')}%"),
        ("median xDelta", f"{stats.get('median_x_gap_closed_mm')} mm"),
        ("median dist traveled", f"{stats.get('median_dist_traveled_mm')} mm"),
        ("production worthy", str(stats.get("production_worthy"))),
    ]
    svg.append(f'<text x="{panel_x + 18}" y="{panel_y + 402}" class="panel-title">Current Stats</text>')
    for idx, (label, value) in enumerate(stat_rows):
        y = panel_y + 430 + (idx * 24)
        svg.append(f'<text x="{panel_x + 18}" y="{y}" class="panel-muted">{_escape(label)}</text>')
        svg.append(f'<text x="{panel_x + 250}" y="{y}" text-anchor="end" class="panel-body">{_escape(value)}</text>')

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def generate_plot(manifest_path: Path, output_path: Path | None = None) -> Path:
    manifest = json.loads(Path(manifest_path).read_text())
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(manifest_path).with_name(f"{Path(manifest_path).stem}_plot.svg")
    )
    destination.write_text(_build_svg(manifest, source_path=Path(manifest_path).resolve()))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a forward-only SVG plot for a trial manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a compact trial manifest JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional SVG output path.")
    args = parser.parse_args()
    output_path = generate_plot(args.manifest, args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
