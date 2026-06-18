#!/usr/bin/env python3
"""Pin the beautiful no-camera speed trial in the speed-test website."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path


LOG_DIR = Path("logs/speed_tests")
SOURCE_STEM = "20260603_155947_physical_2p4_forward_stop_backward_no_camera"
LOCKED_STEM = "LOCKED_beautiful_no_camera_104_to_160_20260603_155947"


def _load_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    src_html = LOG_DIR / f"{SOURCE_STEM}.html"
    src_json = LOG_DIR / f"{SOURCE_STEM}.json"
    dst_html = LOG_DIR / f"{LOCKED_STEM}.html"
    dst_json = LOG_DIR / f"{LOCKED_STEM}.json"
    if not src_html.exists() or not src_json.exists():
        raise SystemExit(f"Missing source trial artifacts: {src_html} / {src_json}")

    shutil.copy2(src_html, dst_html)
    shutil.copy2(src_json, dst_json)
    data = _load_meta(dst_json)
    points = data.get("points") if isinstance(data, dict) else []
    pwm_values = []
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            for key in ("left_pwm", "right_pwm"):
                try:
                    pwm_values.append(int(point.get(key)))
                except (TypeError, ValueError):
                    pass
    pwm_text = ""
    if pwm_values:
        pwm_text = f"{min(pwm_values)} -> {max(pwm_values)} actual PWM"

    cards = [
        {
            "title": "Beautiful No-Camera Straight Ramp",
            "subtitle": "Pinned because the physical motion looked smooth.",
            "html": dst_html.name,
            "json": dst_json.name,
            "details": pwm_text or "actual PWM captured in JSON",
        }
    ]
    body = "\n".join(
        f"""
        <article>
          <h2>{html.escape(card['title'])}</h2>
          <p>{html.escape(card['subtitle'])}</p>
          <p><strong>{html.escape(card['details'])}</strong></p>
          <a href="{html.escape(card['html'])}">Open chart</a>
          <a href="{html.escape(card['json'])}">Open JSON log</a>
        </article>
        """
        for card in cards
    )
    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leia Speed-Test Locked Trials</title>
  <style>
    body {{ margin: 0; font: 15px system-ui, -apple-system, Segoe UI, sans-serif; background: #f5f7f9; color: #111827; }}
    header {{ background: #143447; color: white; padding: 18px 22px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ padding: 18px 22px; max-width: 980px; }}
    article {{ background: white; border: 1px solid #d6dde5; border-radius: 6px; padding: 16px; margin-bottom: 12px; }}
    h2 {{ margin: 0 0 6px; font-size: 18px; }}
    p {{ margin: 4px 0 8px; }}
    a {{ display: inline-block; margin-right: 14px; color: #075985; font-weight: 700; }}
  </style>
</head>
<body>
  <header><h1>Leia Speed-Test Locked Trials</h1></header>
  <main>{body}</main>
</body>
</html>
"""
    (LOG_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(dst_html)
    print(dst_json)
    print(LOG_DIR / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
