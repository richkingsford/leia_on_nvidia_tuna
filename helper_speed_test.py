#!/usr/bin/env python3
"""Build and run Leia's straight-line speed ramp test."""

from __future__ import annotations

import datetime as _dt
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import telemetry_robot as telemetry_robot_module


DEFAULT_PHASE_DURATION_S = 2.0
DEFAULT_INTERVAL_MS = 150
DEFAULT_START_SCORE = 1
DEFAULT_END_SCORE = 2
DEFAULT_FLOOR_PWM_SCALE = 1.00
DEFAULT_CEILING_PWM_SCALE = 1.50
DEFAULT_START_PWM_SCALE = DEFAULT_FLOOR_PWM_SCALE
DEFAULT_LOG_DIR = "logs/speed_tests"


@dataclass(frozen=True)
class SpeedTestPulse:
    phase_name: str
    cmd: str
    phase_step: int
    phase_steps: int
    t_ms: int
    interval_ms: int
    score: int
    model_power: float
    model_pwm: int
    model_duration_ms: int


def _positive_int(value, fallback: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback)
    return parsed if parsed > 0 else int(fallback)


def _phase_steps(phase_duration_s: float, interval_ms: int) -> int:
    try:
        phase_ms = max(1.0, float(phase_duration_s) * 1000.0)
    except (TypeError, ValueError):
        phase_ms = float(DEFAULT_PHASE_DURATION_S * 1000.0)
    return max(1, int(math.ceil(float(phase_ms) / float(max(1, int(interval_ms))))))


def _score_for_step(step_idx: int, phase_steps: int, *, start_score: int, end_score: int) -> int:
    if int(phase_steps) <= 1:
        return int(telemetry_robot_module.normalize_speed_score(end_score))
    frac = float(step_idx) / float(int(phase_steps) - 1)
    raw = float(start_score) + (float(end_score) - float(start_score)) * frac
    return int(telemetry_robot_module.normalize_speed_score(raw))


def _uno_floor_pwm(pwm: int | float | None) -> int:
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        pwm_val = 0
    pwm_val = int(telemetry_robot_module.clamp_pwm(pwm_val))
    if pwm_val <= 0:
        return 0
    percent = int(math.ceil(float(pwm_val) * 100.0 / float(max(1, int(telemetry_robot_module.MAX_PWM)))))
    percent = max(1, min(100, int(percent)))
    return int((int(percent) * int(telemetry_robot_module.MAX_PWM)) / 100)


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scaled_pwm_for_score(cmd: str, score: int, scale: float) -> tuple[float, int, int, int]:
    power, pwm, score_used, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, score)
    pwm = _uno_floor_pwm(pwm)
    try:
        scaled_pwm = int(round(float(pwm) * max(0.0, float(scale))))
    except (TypeError, ValueError):
        scaled_pwm = int(pwm)
    scaled_pwm = int(telemetry_robot_module.clamp_pwm(scaled_pwm))
    scaled_power = telemetry_robot_module.pwm_to_power(int(scaled_pwm)) or float(power)
    return float(scaled_power), int(scaled_pwm), int(score_used), int(duration_ms)


def _pulse_for_step(
    *,
    phase_name: str,
    cmd: str,
    step_idx: int,
    phase_steps: int,
    interval_ms: int,
    start_score: int,
    end_score: int,
    floor_pwm_scale: float,
    ceiling_pwm_scale: float,
) -> SpeedTestPulse:
    score = _score_for_step(
        int(step_idx),
        int(phase_steps),
        start_score=int(start_score),
        end_score=int(end_score),
    )
    high_score = max(int(start_score), int(end_score))
    start_scale = ceiling_pwm_scale if int(start_score) >= int(high_score) else floor_pwm_scale
    end_scale = ceiling_pwm_scale if int(end_score) >= int(high_score) else floor_pwm_scale
    _start_power, start_pwm, _start_score_used, duration_ms = _scaled_pwm_for_score(cmd, start_score, start_scale)
    _end_power, end_pwm, _end_score_used, _end_duration_ms = _scaled_pwm_for_score(cmd, end_score, end_scale)
    if int(phase_steps) <= 1:
        frac = 1.0
    else:
        frac = float(step_idx) / float(int(phase_steps) - 1)
    pwm = int(round(float(start_pwm) + (float(end_pwm) - float(start_pwm)) * float(frac)))
    pwm = int(telemetry_robot_module.clamp_pwm(pwm))
    power = telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0
    return SpeedTestPulse(
        phase_name=str(phase_name),
        cmd=str(cmd),
        phase_step=int(step_idx + 1),
        phase_steps=int(phase_steps),
        t_ms=int(step_idx) * int(interval_ms),
        interval_ms=int(interval_ms),
        score=int(score),
        model_power=float(power),
        model_pwm=int(pwm),
        model_duration_ms=int(duration_ms),
    )


def build_speed_test_sequence(
    *,
    phase_duration_s: float = DEFAULT_PHASE_DURATION_S,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    end_score: int = DEFAULT_END_SCORE,
    start_pwm_scale: float = DEFAULT_START_PWM_SCALE,
    floor_pwm_scale: float | None = None,
    ceiling_pwm_scale: float = DEFAULT_CEILING_PWM_SCALE,
    reverse_mode: str = "backward",
) -> list[SpeedTestPulse]:
    interval = _positive_int(interval_ms, DEFAULT_INTERVAL_MS)
    steps = _phase_steps(phase_duration_s, interval)
    ceiling_score = telemetry_robot_module.normalize_speed_score(end_score, default=DEFAULT_END_SCORE)
    floor_scale = float(start_pwm_scale if floor_pwm_scale is None else floor_pwm_scale)
    phases = [
        ("forward ramp up", "f", DEFAULT_START_SCORE, ceiling_score),
    ]
    reverse_key = str(reverse_mode or "backward").strip().lower()
    if reverse_key == "down":
        phases.append(("forward ramp down", "f", ceiling_score, DEFAULT_START_SCORE))
    else:
        phases.append(("backward ramp up", "b", DEFAULT_START_SCORE, ceiling_score))

    sequence: list[SpeedTestPulse] = []
    for phase_name, cmd, start_score, end_score in phases:
        for step_idx in range(int(steps)):
            sequence.append(
                _pulse_for_step(
                    phase_name=phase_name,
                    cmd=cmd,
                    step_idx=step_idx,
                    phase_steps=steps,
                    interval_ms=interval,
                    start_score=start_score,
                    end_score=end_score,
                    floor_pwm_scale=float(floor_scale),
                    ceiling_pwm_scale=float(ceiling_pwm_scale),
                )
            )
    return sequence


def _table_row(
    pulse: SpeedTestPulse,
    *,
    global_step: int,
    send_result: dict | None = None,
) -> list[str]:
    sent_power = pulse.model_power
    sent_pwm = pulse.model_pwm
    sent_ms = pulse.interval_ms
    percent = ""
    if isinstance(send_result, dict):
        try:
            sent_power = float(send_result.get("power"))
        except (TypeError, ValueError):
            sent_power = pulse.model_power
        try:
            sent_pwm = int(round(float(send_result.get("pwm"))))
        except (TypeError, ValueError):
            sent_pwm = pulse.model_pwm
        try:
            sent_ms = int(round(float(send_result.get("duration_ms"))))
        except (TypeError, ValueError):
            sent_ms = pulse.interval_ms
        if send_result.get("percent") is not None:
            percent = str(int(send_result.get("percent")))
    return [
        str(int(global_step)),
        str(int(pulse.t_ms)),
        pulse.phase_name,
        pulse.cmd.upper(),
        str(int(pulse.score)),
        f"{float(sent_power):.3f}",
        str(int(sent_pwm)),
        percent,
        str(int(pulse.interval_ms)),
        str(int(sent_ms)),
    ]


def format_speed_test_table(
    pulses: Iterable[SpeedTestPulse],
    *,
    records: Iterable[dict] | None = None,
) -> str:
    pulse_list = list(pulses)
    record_list = list(records or [])
    headers = ["#", "t_ms", "phase", "cmd", "score", "power", "pwm", "pct", "cadence_ms", "sent_ms"]
    rows = [headers]
    for idx, pulse in enumerate(pulse_list):
        send_result = None
        if idx < len(record_list) and isinstance(record_list[idx], dict):
            send_result = record_list[idx].get("send_result")
        rows.append(_table_row(pulse, global_step=idx + 1, send_result=send_result))

    widths = [0 for _ in headers]
    for row in rows:
        for col_idx, value in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(str(value)))

    def fmt(row: list[str]) -> str:
        return "  ".join(str(value).rjust(widths[col_idx]) for col_idx, value in enumerate(row))

    lines = [fmt(rows[0]), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows[1:])
    return "\n".join(lines)


def _wheel_pwm_from_send_result(send_result: dict | None, wheel: str, fallback_pwm: int | float | None) -> int:
    fallback = _positive_int(fallback_pwm, 0)
    if not isinstance(send_result, dict):
        return fallback

    wheel_key = "l" if str(wheel).lower().startswith("left") else "r"
    actions = send_result.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            if str(action.get("target") or "").strip().lower() != wheel_key:
                continue
            if str(action.get("action") or "").strip().lower() == "s":
                return 0
            try:
                return int(round(float(action.get("pwm"))))
            except (TypeError, ValueError):
                break

    wire_text = str(send_result.get("wire_text") or "")
    for chunk in wire_text.split(","):
        parts = chunk.strip().split(".")
        if len(parts) >= 2 and parts[0] == wheel_key and parts[1] == "s":
            return 0
        if len(parts) < 4 or parts[0] != wheel_key:
            continue
        try:
            percent = int(round(float(parts[2])))
        except (TypeError, ValueError):
            continue
        percent = max(0, min(100, percent))
        return int((int(percent) * int(telemetry_robot_module.MAX_PWM)) / 100)

    try:
        return int(round(float(send_result.get("pwm"))))
    except (TypeError, ValueError):
        return fallback


def _normalize_vision_sample(sample) -> dict:
    if sample is None:
        return {"found": False, "dist_mm": None, "x_mm": None, "conf": None}
    if isinstance(sample, dict):
        return {
            "found": bool(sample.get("found")),
            "dist_mm": _float_or_none(sample.get("dist_mm", sample.get("dist"))),
            "x_mm": _float_or_none(sample.get("x_mm", sample.get("offset_x"))),
            "y_mm": _float_or_none(sample.get("y_mm", sample.get("offset_y"))),
            "conf": _float_or_none(sample.get("conf", sample.get("confidence"))),
            "status": sample.get("status"),
            "source": sample.get("source"),
            "error": sample.get("error"),
        }
    if isinstance(sample, (list, tuple)) and len(sample) >= 5:
        found = bool(sample[0])
        return {
            "found": found,
            "dist_mm": _float_or_none(sample[2]) if found else None,
            "x_mm": _float_or_none(sample[3]) if found else None,
            "y_mm": _float_or_none(sample[5]) if found and len(sample) >= 6 else None,
            "conf": _float_or_none(sample[4]) if found else None,
            "status": None,
            "source": None,
            "error": None,
        }
    return {"found": False, "dist_mm": None, "x_mm": None, "conf": None, "error": "unsupported_sample"}


def speed_test_chart_points(result: dict) -> list[dict]:
    points = []
    for idx, record in enumerate(result.get("pulses") or []):
        if not isinstance(record, dict):
            continue
        send_result = record.get("send_result")
        vision = _normalize_vision_sample(record.get("vision"))
        fallback_pwm = record.get("model_pwm", 0)
        interval_ms = _positive_int(record.get("interval_ms"), DEFAULT_INTERVAL_MS)
        t_ms = _positive_int(record.get("t_ms"), idx * interval_ms)
        points.append(
            {
                "step": int(idx + 1),
                "t_ms": int(t_ms),
                "phase": str(record.get("phase_name") or ""),
                "cmd": str(record.get("cmd") or "").upper(),
                "score": int(record.get("score") or 0),
                "left_pwm": _wheel_pwm_from_send_result(send_result, "left", fallback_pwm),
                "right_pwm": _wheel_pwm_from_send_result(send_result, "right", fallback_pwm),
                "brick_found": bool(vision.get("found")),
                "dist_mm": vision.get("dist_mm"),
                "x_mm": vision.get("x_mm"),
                "conf": vision.get("conf"),
                "vision_status": vision.get("status"),
                "vision_source": vision.get("source"),
            }
        )
    return points


def _render_speed_test_chart_html(*, title: str, points: list[dict], json_name: str) -> str:
    points_json = json.dumps(points, separators=(",", ":"))
    title_json = json.dumps(str(title))
    source_json = json.dumps(str(json_name))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, -apple-system, Segoe UI, sans-serif; background: #f5f7f9; color: #111827; }}
    header {{ background: #143447; color: white; padding: 18px 22px; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    main {{ padding: 18px 22px 28px; }}
    .panel {{ max-width: 1080px; background: white; border: 1px solid #d6dde5; border-radius: 6px; padding: 16px; }}
    canvas {{ width: 100%; height: 360px; display: block; }}
    #distChart {{ height: 300px; margin-top: 8px; }}
    h2 {{ font-size: 16px; margin: 18px 0 8px; }}
    .legend {{ display: flex; gap: 18px; align-items: center; margin: 8px 0 0; }}
    .swatch {{ display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 6px; }}
    .note {{ color: #475569; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4), th:nth-child(9), td:nth-child(9) {{ text-align: left; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div>Source JSON: <code id="source"></code></div>
  </header>
  <main>
    <section class="panel">
      <h2>Wheel PWM</h2>
      <canvas id="pwmChart" width="1040" height="360"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:#2563eb"></span>Left wheel PWM</span>
        <span><span class="swatch" style="background:#16a34a"></span>Right wheel PWM</span>
      </div>
      <h2>Brick Distance</h2>
      <canvas id="distChart" width="1040" height="300"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:#dc2626"></span>Brick dist mm</span>
      </div>
      <div id="distNote" class="note"></div>
      <table>
        <thead><tr><th>#</th><th>ms</th><th>phase</th><th>cmd</th><th>left pwm</th><th>right pwm</th><th>dist mm</th><th>x mm</th><th>vision</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
  <script>
    const title = {title_json};
    const source = {source_json};
    const points = {points_json};
    document.getElementById("source").textContent = source;

    const pad = {{ left: 62, right: 18, top: 18, bottom: 48 }};
    const maxT = Math.max(1, ...points.map(p => p.t_ms));
    const finiteNumber = value => Number.isFinite(Number(value));

    function drawAxes(ctx, canvas, yLabel, yMin, yMax, tickStep) {{
      const width = canvas.width;
      const height = canvas.height;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const x = t => pad.left + (Number(t) / maxT) * plotW;
      const y = v => pad.top + plotH - ((Number(v) - yMin) / Math.max(1, yMax - yMin)) * plotH;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#d8e0e8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.stroke();

      ctx.fillStyle = "#475569";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let tick = yMin; tick <= yMax + 0.0001; tick += tickStep) {{
        const yy = y(tick);
        ctx.strokeStyle = Math.abs(tick) < 0.0001 ? "#9aa7b5" : "#edf1f5";
        ctx.beginPath();
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(pad.left + plotW, yy);
        ctx.stroke();
        ctx.fillText(String(Math.round(tick)), pad.left - 8, yy);
      }}

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i <= 5; i += 1) {{
        const t = Math.round((maxT / 5) * i);
        const xx = x(t);
        ctx.strokeStyle = "#edf1f5";
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, pad.top + plotH);
        ctx.stroke();
        ctx.fillText(String(t), xx, pad.top + plotH + 10);
      }}

      ctx.save();
      ctx.translate(16, pad.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = "#111827";
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
      ctx.fillText("timestamp (ms)", pad.left + plotW / 2, height - 20);
      return {{ x, y }};
    }}

    function drawLine(ctx, mapper, key, color) {{
      const valid = points.filter(p => finiteNumber(p[key]));
      if (!valid.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      let open = false;
      points.forEach(p => {{
        if (!finiteNumber(p[key])) {{
          open = false;
          return;
        }}
        const xx = mapper.x(p.t_ms);
        const yy = mapper.y(p[key]);
        if (!open) {{
          ctx.moveTo(xx, yy);
          open = true;
        }} else {{
          ctx.lineTo(xx, yy);
        }}
      }});
      ctx.stroke();
      ctx.fillStyle = color;
      valid.forEach(p => {{
        ctx.beginPath();
        ctx.arc(mapper.x(p.t_ms), mapper.y(p[key]), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }});
    }}

    const pwmCanvas = document.getElementById("pwmChart");
    const pwmCtx = pwmCanvas.getContext("2d");
    const maxPwm = Math.max(120, ...points.flatMap(p => [p.left_pwm, p.right_pwm]));
    const yMax = Math.ceil((maxPwm + 10) / 10) * 10;
    const pwmMap = drawAxes(pwmCtx, pwmCanvas, "PWM", 0, yMax, 20);
    drawLine(pwmCtx, pwmMap, "left_pwm", "#2563eb");
    drawLine(pwmCtx, pwmMap, "right_pwm", "#16a34a");

    const distCanvas = document.getElementById("distChart");
    const distCtx = distCanvas.getContext("2d");
    const distValues = points.map(p => Number(p.dist_mm)).filter(Number.isFinite);
    const distMin = distValues.length ? Math.max(0, Math.floor((Math.min(...distValues) - 20) / 10) * 10) : 0;
    const distMax = distValues.length ? Math.ceil((Math.max(...distValues) + 20) / 10) * 10 : 100;
    const distStep = Math.max(10, Math.ceil((distMax - distMin) / 5 / 10) * 10);
    const distMap = drawAxes(distCtx, distCanvas, "brick dist (mm)", distMin, distMax, distStep);
    drawLine(distCtx, distMap, "dist_mm", "#dc2626");
    const missCount = points.filter(p => !finiteNumber(p.dist_mm)).length;
    document.getElementById("distNote").textContent = distValues.length
      ? `${{distValues.length}} distance sample(s), ${{missCount}} miss(es).`
      : "No brick-distance samples were available for this run.";

    const rows = document.getElementById("rows");
    rows.innerHTML = points.map(p => {{
      const dist = finiteNumber(p.dist_mm) ? Number(p.dist_mm).toFixed(1) : "miss";
      const x = finiteNumber(p.x_mm) ? Number(p.x_mm).toFixed(1) : "";
      const vision = p.brick_found ? `found ${{finiteNumber(p.conf) ? Number(p.conf).toFixed(0) + "%" : ""}}` : (p.vision_status || "miss");
      return `<tr><td>${{p.step}}</td><td>${{p.t_ms}}</td><td>${{p.phase}}</td><td>${{p.cmd}}</td><td>${{p.left_pwm}}</td><td>${{p.right_pwm}}</td><td>${{dist}}</td><td>${{x}}</td><td>${{vision}}</td></tr>`;
    }}).join("");
  </script>
</body>
</html>
"""


def write_speed_test_artifacts(
    result: dict,
    *,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    run_label: str | None = None,
    metadata: dict | None = None,
) -> dict:
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_label or "speed_test"))
    stem = f"{timestamp}_{safe_label}"
    points = speed_test_chart_points(result)
    payload = {
        "metadata": dict(metadata or {}),
        "ok": bool(result.get("ok")),
        "execute": bool(result.get("execute")),
        "points": points,
        "pulses": result.get("pulses") or [],
    }
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(
        _render_speed_test_chart_html(
            title=f"Leia Speed Test PWM - {safe_label}",
            points=points,
            json_name=json_path.name,
        ),
        encoding="utf-8",
    )
    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
        "points": points,
    }


def run_speed_test(
    *,
    robot=None,
    execute: bool = False,
    sequence: Iterable[SpeedTestPulse] | None = None,
    vision_sampler: Callable[[], dict | tuple | None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    log_fn: Callable[[str], None] | None = print,
) -> dict:
    pulses = list(sequence) if sequence is not None else build_speed_test_sequence()
    records = []
    if execute and robot is None:
        raise ValueError("execute=True requires a robot instance")

    def emit(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    if execute:
        emit("[SPEED] Sending initial stop, then starting continuous ramp.")
        robot.stop()
    else:
        emit("[DRY-RUN] Planned speed table:")
        emit(format_speed_test_table(pulses))

    start_s = monotonic_fn()
    cadence_s = (float(pulses[0].interval_ms) / 1000.0) if pulses else 0.0
    for idx, pulse in enumerate(pulses):
        send_result = None
        if execute:
            send_result = robot.send_command_pwm(
                pulse.cmd,
                int(pulse.model_pwm),
                duration_ms=int(pulse.interval_ms),
            )
        vision_sample = None
        if callable(vision_sampler):
            try:
                vision_sample = _normalize_vision_sample(vision_sampler())
            except Exception as exc:
                vision_sample = {
                    "found": False,
                    "dist_mm": None,
                    "x_mm": None,
                    "conf": None,
                    "error": str(exc),
                }
        records.append(
            {
                "phase_name": pulse.phase_name,
                "cmd": pulse.cmd,
                "t_ms": int(idx) * int(pulse.interval_ms),
                "phase_t_ms": pulse.t_ms,
                "phase_step": pulse.phase_step,
                "phase_steps": pulse.phase_steps,
                "score": pulse.score,
                "model_power": pulse.model_power,
                "model_pwm": pulse.model_pwm,
                "interval_ms": pulse.interval_ms,
                "send_result": send_result,
                "vision": vision_sample,
            }
        )
        next_start_s = float(start_s) + (float(idx + 1) * float(cadence_s))
        remaining_s = float(next_start_s) - float(monotonic_fn())
        if remaining_s > 0.0:
            sleep_fn(remaining_s)

    if execute:
        robot.stop()
        emit("[SPEED] Final stop sent. Actual sent table:")
        emit(format_speed_test_table(pulses, records=records))

    return {
        "ok": True,
        "execute": bool(execute),
        "pulses": records,
    }
