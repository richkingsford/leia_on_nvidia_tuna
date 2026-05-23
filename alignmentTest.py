#!/usr/bin/env python3
"""Verify brick telemetry direction after small model-defined robot nudges.

The movement pulses are sourced from world_model_robot.json via
helper_basic_movement_smoke/telemetry_robot.  No PWM or wire motion values live
in this script.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from helper_basic_movement_smoke import (
    SEND_MODE_TURN_ARC,
    build_basic_movement_sequence,
    format_basic_movement_pulse,
)
from helper_robot_control import Robot


DEFAULT_STREAM_URL = "http://127.0.0.1:5000"
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_SAMPLE_INTERVAL_S = 0.12
DEFAULT_PRE_SETTLE_S = 0.25
DEFAULT_POST_SETTLE_S = 0.55
DEFAULT_MIN_CONFIDENCE_PCT = 50.0
DEFAULT_MIN_DELTA_MM = 1.0


@dataclass(frozen=True)
class TelemetrySample:
    found: bool
    dist_mm: float
    x_mm: float
    y_mm: float
    confidence_pct: float
    source: str


@dataclass(frozen=True)
class AlignmentCheck:
    name: str
    hotkey: str
    metric: str
    expected_sign: int
    expected_text: str


ALIGNMENT_CHECKS = (
    AlignmentCheck("forward", "r", "dist_mm", -1, "dist gets smaller"),
    AlignmentCheck("backward", "f", "dist_mm", 1, "dist gets larger"),
    AlignmentCheck("left", "q", "x_mm", -1, "x gets smaller"),
    AlignmentCheck("right", "e", "x_mm", 1, "x gets larger"),
    AlignmentCheck("up", "o", "y_mm", 1, "y gets larger"),
    AlignmentCheck("down", "k", "y_mm", -1, "y gets smaller"),
)


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _num_from_text(text: str) -> float | None:
    match = _NUM_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _stream_text_payload(stream_url: str, timeout_s: float) -> dict:
    base = str(stream_url or DEFAULT_STREAM_URL).rstrip("/")
    with urllib.request.urlopen(f"{base}/text", timeout=float(timeout_s)) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def read_stream_sample(stream_url: str, *, timeout_s: float = 1.0) -> TelemetrySample:
    payload = _stream_text_payload(stream_url, timeout_s)
    lines = payload.get("lines")
    if not isinstance(lines, list):
        lines = []

    visible = False
    dist = x_axis = y_axis = conf = None
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        upper = text.upper()
        if upper.startswith("VISIBLE:"):
            visible = "TRUE" in upper
        elif upper.startswith("DIST:"):
            dist = _num_from_text(text)
        elif upper.startswith("X-AXIS:"):
            x_axis = _num_from_text(text)
        elif upper.startswith("Y-AXIS:"):
            y_axis = _num_from_text(text)
        elif upper.startswith("CONF:"):
            conf = _num_from_text(text)

    if dist is None or x_axis is None or y_axis is None:
        raise RuntimeError("stream telemetry missing DIST/X-AXIS/Y-AXIS")
    if conf is None:
        conf = 0.0
    return TelemetrySample(
        found=bool(visible),
        dist_mm=float(dist),
        x_mm=float(x_axis),
        y_mm=float(y_axis),
        confidence_pct=float(conf),
        source="stream",
    )


class CameraTelemetryReader:
    def __init__(self) -> None:
        from helper_brick_detector_native_oak import BrickDetector
        from livestream_crown_vision import CROWN_PROFILE_TUNING

        self._vision = BrickDetector(debug=False)
        self._vision.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))

    def close(self) -> None:
        close_fn = getattr(self._vision, "close", None)
        if callable(close_fn):
            close_fn()
            return
        cap = getattr(self._vision, "cap", None)
        release_fn = getattr(cap, "release", None)
        if callable(release_fn):
            release_fn()

    def read(self) -> TelemetrySample:
        result = self._vision.read()
        found, _angle, dist, x_axis, conf, y_axis, _brick_above, _brick_below = result
        return TelemetrySample(
            found=bool(found),
            dist_mm=float(dist),
            x_mm=float(x_axis),
            y_mm=float(y_axis),
            confidence_pct=float(conf),
            source="camera",
        )


def _median_sample(samples: list[TelemetrySample]) -> TelemetrySample:
    if not samples:
        raise ValueError("no telemetry samples")
    return TelemetrySample(
        found=all(sample.found for sample in samples),
        dist_mm=float(statistics.median(sample.dist_mm for sample in samples)),
        x_mm=float(statistics.median(sample.x_mm for sample in samples)),
        y_mm=float(statistics.median(sample.y_mm for sample in samples)),
        confidence_pct=float(statistics.median(sample.confidence_pct for sample in samples)),
        source=samples[-1].source,
    )


def sample_telemetry(
    read_fn: Callable[[], TelemetrySample],
    *,
    count: int,
    interval_s: float,
    min_confidence_pct: float,
) -> TelemetrySample:
    wanted = max(1, int(count))
    interval = max(0.0, float(interval_s))
    min_conf = max(0.0, float(min_confidence_pct))
    usable: list[TelemetrySample] = []
    misses: list[str] = []

    for idx in range(wanted):
        try:
            sample = read_fn()
        except Exception as exc:
            misses.append(str(exc))
        else:
            if not sample.found:
                misses.append("not visible")
            elif float(sample.confidence_pct) < min_conf:
                misses.append(f"confidence {sample.confidence_pct:.1f}<{min_conf:.1f}")
            else:
                usable.append(sample)
        if idx + 1 < wanted and interval > 0.0:
            time.sleep(interval)

    if not usable:
        detail = "; ".join(misses[-3:]) if misses else "no samples"
        raise RuntimeError(f"no usable brick telemetry ({detail})")
    return _median_sample(usable)


def _make_reader(args):
    source = str(args.source or "auto").strip().lower()
    if source in {"auto", "stream"}:
        try:
            read_stream_sample(args.stream_url, timeout_s=args.stream_timeout_s)
            return (lambda: read_stream_sample(args.stream_url, timeout_s=args.stream_timeout_s)), None, "stream"
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
            if source == "stream":
                raise
    reader = CameraTelemetryReader()
    return reader.read, reader.close, "camera"


def _pulse_map(use_turn_arc_profiles: bool):
    sequence = build_basic_movement_sequence(
        [check.hotkey for check in ALIGNMENT_CHECKS],
        use_turn_arc_profiles=bool(use_turn_arc_profiles),
    )
    return {pulse.reference_hotkey: pulse for pulse in sequence}


def _send_pulse(robot: Robot, pulse):
    if pulse.send_mode == SEND_MODE_TURN_ARC and isinstance(pulse.turn_arc_plan, dict):
        return robot.send_custom_actions_pwm(
            pulse.cmd,
            pulse.turn_arc_plan.get("actions") or [],
            duration_ms=int(pulse.duration_ms),
        )
    return robot.send_command_pwm(
        pulse.cmd,
        int(pulse.pwm),
        duration_ms=int(pulse.duration_ms),
    )


def _metric(sample: TelemetrySample, metric: str) -> float:
    return float(getattr(sample, metric))


def _fmt_sample(sample: TelemetrySample) -> str:
    return (
        f"dist={sample.dist_mm:.1f}mm "
        f"x={sample.x_mm:+.1f}mm "
        f"y={sample.y_mm:+.1f}mm "
        f"conf={sample.confidence_pct:.0f}%"
    )


def _run(args) -> dict:
    pulses = _pulse_map(use_turn_arc_profiles=not bool(args.no_turn_arc))
    if args.dry_run:
        for check in ALIGNMENT_CHECKS:
            print(format_basic_movement_pulse(pulses[check.hotkey], prefix="[DRY-RUN]"))
        return {"ok": True, "dry_run": True, "checks": []}

    read_fn, close_reader, source_used = _make_reader(args)
    robot = Robot(exit_on_failure=False, serial_port=args.serial_port)
    records = []

    try:
        print(f"[ALIGN] telemetry source: {source_used}")
        robot.stop()
        time.sleep(max(0.0, float(args.pre_settle_s)))

        for check in ALIGNMENT_CHECKS:
            pulse = pulses[check.hotkey]
            print(format_basic_movement_pulse(pulse, prefix="[ALIGN]"))
            before = sample_telemetry(
                read_fn,
                count=args.samples,
                interval_s=args.sample_interval_s,
                min_confidence_pct=args.min_confidence_pct,
            )

            send_result = _send_pulse(robot, pulse)
            wait_s = max(
                0.0,
                (float(pulse.duration_ms) / 1000.0) + float(args.post_settle_s),
            )
            time.sleep(wait_s)
            robot.stop()
            time.sleep(max(0.0, float(args.pre_settle_s)))

            after = sample_telemetry(
                read_fn,
                count=args.samples,
                interval_s=args.sample_interval_s,
                min_confidence_pct=args.min_confidence_pct,
            )

            delta = _metric(after, check.metric) - _metric(before, check.metric)
            signed_delta = float(delta) * float(check.expected_sign)
            ok = signed_delta >= float(args.min_delta_mm)
            status = "PASS" if ok else "FAIL"
            metric_name = check.metric.replace("_mm", "")
            print(
                f"[ALIGN] {status} {check.name:<8} expected {check.expected_text}: "
                f"{metric_name} delta={delta:+.2f}mm | "
                f"before {_fmt_sample(before)} -> after {_fmt_sample(after)}"
            )
            if bool(args.show_wire) and isinstance(send_result, dict) and send_result.get("wire_text"):
                print(f"[WIRE DEBUG] {send_result['wire_text']}")

            record = {
                "name": check.name,
                "hotkey": check.hotkey,
                "cmd": pulse.cmd,
                "metric": check.metric,
                "expected_sign": check.expected_sign,
                "delta_mm": float(delta),
                "ok": bool(ok),
                "before": before.__dict__,
                "after": after.__dict__,
                "pulse": {
                    "reference_hotkey": pulse.reference_hotkey,
                    "cmd": pulse.cmd,
                    "score": pulse.score,
                    "pwm": pulse.pwm,
                    "duration_ms": pulse.duration_ms,
                    "send_mode": pulse.send_mode,
                },
            }
            records.append(record)
            if not ok and bool(args.stop_on_fail):
                break

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
        "ok": all(bool(record.get("ok")) for record in records),
        "dry_run": False,
        "telemetry_source": source_used,
        "checks": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move Leia in each direction and verify brick telemetry sign changes.",
    )
    parser.add_argument("--source", choices=("auto", "stream", "camera"), default="auto")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--stream-timeout-s", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    parser.add_argument("--pre-settle-s", type=float, default=DEFAULT_PRE_SETTLE_S)
    parser.add_argument("--post-settle-s", type=float, default=DEFAULT_POST_SETTLE_S)
    parser.add_argument("--min-confidence-pct", type=float, default=DEFAULT_MIN_CONFIDENCE_PCT)
    parser.add_argument("--min-delta-mm", type=float, default=DEFAULT_MIN_DELTA_MM)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--no-turn-arc", action="store_true", help="Use raw q/e world-model rows instead of manual turn-arc profiles.")
    parser.add_argument("--show-wire", action="store_true", help="Print raw Uno wire payloads after each logical command.")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the final result as JSON.")
    args = parser.parse_args()

    try:
        result = _run(args)
    except KeyboardInterrupt:
        print("\n[ALIGN] Interrupted.")
        return 130
    except Exception as exc:
        print(f"[ALIGN] ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
