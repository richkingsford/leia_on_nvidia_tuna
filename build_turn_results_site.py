#!/usr/bin/env python3
"""Build a small SVG-only results site for turn calibration runs."""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRIALS_DIR = ROOT / "trials"
RESULTS_DIR = TRIALS_DIR / "results"
ASSETS_DIR = RESULTS_DIR / "assets"
SWEEP_JSON = TRIALS_DIR / "balanced_turn_sweep_72_rerun.json"
PRODUCTION_JSON = TRIALS_DIR / "turn_duration_production_curves.json"
GUARD_JSON = TRIALS_DIR / "balanced_turn_red_purple_rerun.json"

PHASES = ("forward_left", "back_right", "forward_right", "back_left")
PHASE_LABELS = {
    "forward_left": "forward+left",
    "back_right": "back+right",
    "forward_right": "forward+right",
    "back_left": "back+left",
}
PHASE_COLORS = {
    "forward_left": "#2563eb",
    "back_right": "#dc2626",
    "forward_right": "#16a34a",
    "back_left": "#9333ea",
}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _median_points(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        duration = _coerce_float(row.get("measuredDurationMs"))
        x_delta = _coerce_float(row.get("xDelta"))
        if duration is None or x_delta is None:
            continue
        grouped[int(round(duration))].append(abs(float(x_delta)))
    points = []
    for duration in sorted(grouped):
        points.append(
            {
                "duration_ms": int(duration),
                "x_mm": float(statistics.median(grouped[duration])),
                "samples": int(len(grouped[duration])),
            }
        )
    return points


def _line_chart_svg(sweep: dict) -> str:
    rows = [
        row for row in list(sweep.get("trials") or [])
        if isinstance(row, dict) and row.get("usable") is True
    ]
    by_phase = {
        phase: _median_points([row for row in rows if str(row.get("plannedPhase")) == phase])
        for phase in PHASES
    }
    all_points = [point for points in by_phase.values() for point in points]
    min_x = min((point["duration_ms"] for point in all_points), default=150)
    max_x = max((point["duration_ms"] for point in all_points), default=350)
    max_y = max((point["x_mm"] for point in all_points), default=3.0)
    max_y = max(1.0, round(float(max_y) + 0.5, 1))

    width, height = 680, 360
    left, right, top, bottom = 52, 18, 28, 46
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        if max_x <= min_x:
            return left
        return left + ((float(value) - min_x) / (max_x - min_x)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (float(value) / max_y)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Turn x travel by duration</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="52" y="19" font-family="system-ui, sans-serif" font-size="15" font-weight="700" fill="#111827">x travel by act duration</text>',
    ]
    for tick in range(0, 5):
        y_val = (max_y / 4.0) * tick
        y = sy(y_val)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="10" y="{y+4:.1f}" font-family="system-ui, sans-serif" font-size="10" fill="#6b7280">{y_val:.1f}</text>')
    for x_val in range(int(min_x), int(max_x) + 1, 50):
        x = sx(x_val)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x-12:.1f}" y="{height-20}" font-family="system-ui, sans-serif" font-size="10" fill="#6b7280">{x_val}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#6b7280"/>')

    legend_x = 430
    legend_y = 18
    for idx, phase in enumerate(PHASES):
        color = PHASE_COLORS[phase]
        points = by_phase[phase]
        if points:
            poly_points = " ".join(f'{sx(point["duration_ms"]):.1f},{sy(point["x_mm"]):.1f}' for point in points)
            parts.append(f'<polyline points="{poly_points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
            for point in points:
                parts.append(f'<circle cx="{sx(point["duration_ms"]):.1f}" cy="{sy(point["x_mm"]):.1f}" r="2.4" fill="{color}" opacity="0.78"/>')
        y = legend_y + (idx * 18)
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 10}" y="{y + 4}" font-family="system-ui, sans-serif" font-size="11" fill="#111827">{_escape(PHASE_LABELS[phase])}</text>')

    parts.append(f'<text x="{width/2-44:.1f}" y="{height-4}" font-family="system-ui, sans-serif" font-size="11" fill="#374151">duration ms</text>')
    parts.append('<text x="4" y="180" transform="rotate(-90 4 180)" font-family="system-ui, sans-serif" font-size="11" fill="#374151">x mm</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def _status_svg(curves: dict) -> str:
    phase_curves = curves.get("phase_curves") if isinstance(curves.get("phase_curves"), dict) else {}
    width, height = 520, 210
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Production status</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="18" y="24" font-family="system-ui, sans-serif" font-size="15" font-weight="700" fill="#111827">curve status</text>',
    ]
    for idx, phase in enumerate(PHASES):
        curve = phase_curves.get(phase) if isinstance(phase_curves.get(phase), dict) else {}
        status = str(curve.get("status") or "unknown")
        samples = int(curve.get("used_sample_count") or curve.get("sample_count") or 0)
        outliers = int(curve.get("outlier_count") or 0)
        mono = curve.get("monotonic_score_pct")
        y = 54 + idx * 36
        ready = status == "provisional_production"
        fill = "#dcfce7" if ready else "#fee2e2"
        stroke = "#16a34a" if ready else "#dc2626"
        text = "provisional" if ready else "rerun"
        parts.append(f'<text x="18" y="{y+5}" font-family="system-ui, sans-serif" font-size="12" fill="#111827">{_escape(PHASE_LABELS[phase])}</text>')
        parts.append(f'<rect x="142" y="{y-12}" width="96" height="22" rx="4" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="155" y="{y+4}" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="{stroke}">{text}</text>')
        parts.append(f'<text x="260" y="{y+5}" font-family="system-ui, sans-serif" font-size="11" fill="#4b5563">{samples} used, {outliers} outliers, monotonic {_escape(mono)}%</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def _guard_svg(guard: dict) -> str:
    rows = [row for row in list(guard.get("trials") or []) if isinstance(row, dict)]
    first = rows[0] if rows else {}
    start = _coerce_float(first.get("startDist")) or 70.414
    delta = _coerce_float(first.get("distDelta")) or 79.812
    after = start + delta
    guard_mm = _coerce_float(((guard.get("curve") or {}).get("dist_zone_guard_mm"))) or 30.0
    low, high = start - guard_mm, start + guard_mm
    min_v = max(0.0, min(low, start, after) - 12.0)
    max_v = max(high, start, after) + 12.0
    width, height = 520, 150
    left, right = 42, 24
    y = 78

    def sx(value: float) -> float:
        return left + ((float(value) - min_v) / (max_v - min_v)) * (width - left - right)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Distance guard stop</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="18" y="24" font-family="system-ui, sans-serif" font-size="15" font-weight="700" fill="#111827">distance guard</text>',
        f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#9ca3af" stroke-width="2"/>',
        f'<rect x="{sx(low):.1f}" y="{y-15}" width="{sx(high)-sx(low):.1f}" height="30" fill="#dcfce7" stroke="#16a34a" opacity="0.9"/>',
        f'<line x1="{sx(start):.1f}" y1="{y-26}" x2="{sx(start):.1f}" y2="{y+26}" stroke="#111827" stroke-width="2"/>',
        f'<circle cx="{sx(after):.1f}" cy="{y}" r="7" fill="#dc2626"/>',
        f'<text x="{sx(start)-28:.1f}" y="{y+44}" font-family="system-ui, sans-serif" font-size="11" fill="#111827">start {start:.1f}</text>',
        f'<text x="{sx(after)-36:.1f}" y="{y-34}" font-family="system-ui, sans-serif" font-size="11" fill="#dc2626">after {after:.1f}</text>',
        f'<text x="18" y="136" font-family="system-ui, sans-serif" font-size="11" fill="#4b5563">stopped after trial 1: drift {delta:.1f}mm &gt; guard {guard_mm:.1f}mm</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Turn Calibration Results</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #111827; }
    main { max-width: 980px; margin: 0 auto; padding: 22px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 18px; }
    h1 { font-size: 22px; line-height: 1.1; margin: 0; letter-spacing: 0; }
    .meta { color: #4b5563; font-size: 13px; }
    section { background: #fff; border: 1px solid #d9dde5; border-radius: 6px; padding: 12px; margin: 12px 0; }
    img { display: block; width: 100%; height: auto; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 0.66fr); gap: 12px; align-items: start; }
    @media (max-width: 760px) { main { padding: 12px; } header, .grid { display: block; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Turn Calibration Results</h1>
      <div class="meta">SVG-only dashboard</div>
    </header>
    <section>
      <img src="assets/turn_duration_curves.svg" alt="Turn x travel by act duration">
    </section>
    <div class="grid">
      <section>
        <img src="assets/production_status.svg" alt="Curve production status">
      </section>
      <section>
        <img src="assets/dist_guard.svg" alt="Distance guard stop summary">
      </section>
    </div>
  </main>
</body>
</html>
"""


def main() -> int:
    sweep = _load_json(SWEEP_JSON)
    curves = _load_json(PRODUCTION_JSON)
    guard = _load_json(GUARD_JSON)

    _write(ASSETS_DIR / "turn_duration_curves.svg", _line_chart_svg(sweep))
    _write(ASSETS_DIR / "production_status.svg", _status_svg(curves))
    _write(ASSETS_DIR / "dist_guard.svg", _guard_svg(guard))
    _write(RESULTS_DIR / "index.html", _index_html())
    print(f"Saved {RESULTS_DIR / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
