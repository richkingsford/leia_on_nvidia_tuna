#!/usr/bin/env python3
"""Crawl forward/backward while recording live brick distance readings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

from alignmentTest import CameraTelemetryReader, TelemetrySample, read_stream_sample
from helper_basic_movement_smoke import build_basic_movement_sequence, format_basic_movement_pulse
from helper_robot_control import Robot


DEFAULT_STREAM_URL = "http://127.0.0.1:5000"
DEFAULT_TOTAL_S = 2.0
DEFAULT_SAMPLE_INTERVAL_S = 0.10
DEFAULT_MIN_CONFIDENCE_PCT = 40.0


@dataclass
class DistRecord:
    phase: str
    t_s: float
    found: bool
    dist_mm: float | None
    delta_mm: float | None
    x_mm: float | None
    y_mm: float | None
    confidence_pct: float | None
    note: str = ""


def _make_reader(args):
    source = str(args.source or "auto").strip().lower()
    if source in {"auto", "stream"}:
        try:
            read_stream_sample(args.stream_url, timeout_s=args.stream_timeout_s)
            return (lambda: read_stream_sample(args.stream_url, timeout_s=args.stream_timeout_s)), None, "stream"
        except Exception:
            if source == "stream":
                raise
    reader = CameraTelemetryReader()
    return reader.read, reader.close, "camera"


def _pulse_map():
    return {pulse.reference_hotkey: pulse for pulse in build_basic_movement_sequence(["r", "f"])}


def _send_pulse(robot: Robot, pulse) -> dict:
    return robot.send_command_pwm(
        pulse.cmd,
        int(pulse.pwm),
        duration_ms=int(pulse.duration_ms),
    )


def _fmt_record(record: DistRecord) -> str:
    if not record.found or record.dist_mm is None:
        conf = "N/A" if record.confidence_pct is None else f"{record.confidence_pct:.0f}%"
        return f"[DIST] {record.phase:<8} t={record.t_s:4.2f}s missing conf={conf} {record.note}".rstrip()
    delta = "base" if record.delta_mm is None else f"{record.delta_mm:+6.1f}mm"
    conf = "N/A" if record.confidence_pct is None else f"{record.confidence_pct:.0f}%"
    return (
        f"[DIST] {record.phase:<8} t={record.t_s:4.2f}s "
        f"dist={record.dist_mm:7.1f}mm d={delta} "
        f"x={record.x_mm:+6.1f}mm y={record.y_mm:+6.1f}mm conf={conf}"
    )


def _record_sample(phase: str, t0: float, read_fn, baseline: float | None, min_confidence_pct: float) -> DistRecord:
    now = time.monotonic()
    try:
        sample: TelemetrySample = read_fn()
    except Exception as exc:
        return DistRecord(
            phase=phase,
            t_s=float(now - t0),
            found=False,
            dist_mm=None,
            delta_mm=None,
            x_mm=None,
            y_mm=None,
            confidence_pct=None,
            note=str(exc),
        )
    conf = float(sample.confidence_pct)
    found = bool(sample.found) and conf >= float(min_confidence_pct)
    dist = float(sample.dist_mm) if found else None
    delta = None if baseline is None or dist is None else float(dist - baseline)
    return DistRecord(
        phase=phase,
        t_s=float(now - t0),
        found=found,
        dist_mm=dist,
        delta_mm=delta,
        x_mm=float(sample.x_mm) if found else None,
        y_mm=float(sample.y_mm) if found else None,
        confidence_pct=conf,
        note="" if found else "low_conf_or_not_visible",
    )


def _summarize_phase(records: list[DistRecord], phase: str) -> str:
    rows = [row for row in records if row.phase == phase and row.found and row.dist_mm is not None]
    if len(rows) < 2:
        return f"[DIST] {phase:<8} summary: not enough usable samples ({len(rows)})"
    deltas = [
        float(rows[idx].dist_mm) - float(rows[idx - 1].dist_mm)
        for idx in range(1, len(rows))
    ]
    abs_deltas = [abs(value) for value in deltas]
    start = float(rows[0].dist_mm)
    end = float(rows[-1].dist_mm)
    sign_flips = 0
    last_sign = 0
    for value in deltas:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign and last_sign and sign != last_sign:
            sign_flips += 1
        if sign:
            last_sign = sign
    return (
        f"[DIST] {phase:<8} summary: samples={len(rows)} "
        f"dist {start:.1f}->{end:.1f}mm total={end - start:+.1f}mm "
        f"median_step={statistics.median(deltas):+.1f}mm "
        f"median_abs_step={statistics.median(abs_deltas):.1f}mm "
        f"max_abs_step={max(abs_deltas):.1f}mm sign_flips={sign_flips}"
    )


def _crawl_phase(args, robot: Robot, read_fn, phase: str, pulse, baseline: float | None) -> list[DistRecord]:
    records: list[DistRecord] = []
    total_s = max(0.1, float(args.total_s))
    sample_interval_s = max(0.02, float(args.sample_interval_s))
    pulse_s = max(0.01, float(pulse.duration_ms) / 1000.0)
    end_at = time.monotonic() + total_s
    t0 = time.monotonic()
    next_sample_at = t0
    next_pulse_at = t0
    print(format_basic_movement_pulse(pulse, prefix=f"[DIST] {phase}"))

    while time.monotonic() < end_at:
        now = time.monotonic()
        if now >= next_pulse_at:
            _send_pulse(robot, pulse)
            next_pulse_at = now + pulse_s
        if now >= next_sample_at:
            record = _record_sample(phase, t0, read_fn, baseline, args.min_confidence_pct)
            records.append(record)
            print(_fmt_record(record), flush=True)
            if baseline is None and record.found and record.dist_mm is not None:
                baseline = float(record.dist_mm)
            next_sample_at = now + sample_interval_s
        time.sleep(min(0.01, max(0.0, end_at - time.monotonic())))

    robot.stop()
    return records


def _run(args) -> dict:
    pulses = _pulse_map()
    forward = pulses["r"]
    backward = pulses["f"]
    if args.dry_run:
        print(format_basic_movement_pulse(forward, prefix="[DRY-RUN] forward"))
        print(format_basic_movement_pulse(backward, prefix="[DRY-RUN] backward"))
        return {"ok": True, "dry_run": True, "records": []}

    read_fn, close_reader, source_used = _make_reader(args)
    robot = Robot(exit_on_failure=False, serial_port=args.serial_port)
    records: list[DistRecord] = []
    try:
        print(f"[DIST] telemetry source: {source_used}")
        print(
            f"[DIST] crawl windows: forward={float(args.total_s):.1f}s, "
            f"backward={float(args.total_s):.1f}s, sample_interval={float(args.sample_interval_s):.2f}s"
        )
        robot.stop()
        time.sleep(max(0.0, float(args.pre_settle_s)))

        baseline_record = _record_sample("baseline", time.monotonic(), read_fn, None, args.min_confidence_pct)
        print(_fmt_record(baseline_record), flush=True)
        baseline = baseline_record.dist_mm if baseline_record.found else None

        records.extend(_crawl_phase(args, robot, read_fn, "forward", forward, baseline))
        time.sleep(max(0.0, float(args.between_settle_s)))
        records.extend(_crawl_phase(args, robot, read_fn, "backward", backward, baseline))

        print(_summarize_phase(records, "forward"))
        print(_summarize_phase(records, "backward"))
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        if callable(close_reader):
            close_reader()

    return {
        "ok": True,
        "dry_run": False,
        "telemetry_source": source_used,
        "records": [asdict(record) for record in records],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Slowly crawl forward/backward while logging brick distance jitter.",
    )
    parser.add_argument("--source", choices=("auto", "stream", "camera"), default="auto")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--stream-timeout-s", type=float, default=1.0)
    parser.add_argument("--total-s", type=float, default=DEFAULT_TOTAL_S)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    parser.add_argument("--pre-settle-s", type=float, default=0.25)
    parser.add_argument("--between-settle-s", type=float, default=0.35)
    parser.add_argument("--min-confidence-pct", type=float, default=DEFAULT_MIN_CONFIDENCE_PCT)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = _run(args)
    except KeyboardInterrupt:
        print("\n[DIST] Interrupted.")
        return 130
    except Exception as exc:
        print(f"[DIST] ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
