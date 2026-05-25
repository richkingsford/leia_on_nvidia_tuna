#!/usr/bin/env python3
"""Calibrate mast Y motion with paired up/down pulses."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

from alignmentTest import CameraTelemetryReader, TelemetrySample, read_stream_sample
from a_follow_the_brick import _follow_y_axis_config, _scaled_pwm_for_cmd
from helper_robot_control import Robot


MAX_SINGLE_DIRECTION_MS = 3000
DEFAULT_PULSE_MS = 1000
DEFAULT_CYCLES = 2
DEFAULT_SETTLE_S = 0.45
DEFAULT_SAMPLE_COUNT = 7
DEFAULT_SAMPLE_INTERVAL_S = 0.08
DEFAULT_MIN_CONFIDENCE_PCT = 40.0
DEFAULT_STREAM_URL = "http://127.0.0.1:5000"
DEFAULT_MAX_DIST_DRIFT_MM = 35.0
DEFAULT_MAX_X_DRIFT_MM = 35.0


@dataclass(frozen=True)
class YProfile:
    name: str
    up_pwm: int
    down_pwm: int
    pulse_ms: int
    cycles: int


@dataclass
class YSample:
    found: bool
    y_mm: float | None
    dist_mm: float | None
    x_mm: float | None
    confidence_pct: float | None
    source: str


@dataclass
class YCycle:
    profile: str
    cycle: int
    pulse_ms: int
    up_pwm: int
    down_pwm: int
    before: YSample
    after_up: YSample
    after_down: YSample
    up_delta_mm: float | None
    down_delta_mm: float | None
    up_mm_per_100ms: float | None
    down_mm_per_100ms: float | None
    quality_ok: bool
    quality_reason: str


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


def _profile_from_spec(spec: str, fallback_cycles: int) -> YProfile:
    parts = [part.strip() for part in str(spec or "").split(":")]
    if len(parts) not in {3, 4, 5}:
        raise ValueError(
            "--profile must be name:pwm:pulse_ms[:cycles] or name:up_pwm:down_pwm:pulse_ms[:cycles]"
        )
    name = parts[0] or "profile"
    if len(parts) == 3:
        up_pwm = down_pwm = int(round(float(parts[1])))
        pulse_ms = int(round(float(parts[2])))
        cycles = int(fallback_cycles)
    elif len(parts) == 4:
        up_pwm = down_pwm = int(round(float(parts[1])))
        pulse_ms = int(round(float(parts[2])))
        cycles = int(round(float(parts[3])))
    else:
        up_pwm = int(round(float(parts[1])))
        down_pwm = int(round(float(parts[2])))
        pulse_ms = int(round(float(parts[3])))
        cycles = int(round(float(parts[4])))
    if pulse_ms <= 0 or pulse_ms > MAX_SINGLE_DIRECTION_MS:
        raise ValueError(f"profile {name!r} pulse_ms must be 1..{MAX_SINGLE_DIRECTION_MS}")
    if cycles <= 0:
        raise ValueError(f"profile {name!r} cycles must be positive")
    return YProfile(
        name=str(name),
        up_pwm=max(1, min(255, int(up_pwm))),
        down_pwm=max(1, min(255, int(down_pwm))),
        pulse_ms=int(pulse_ms),
        cycles=int(cycles),
    )


def _default_profiles(y_cfg: dict, fallback_cycles: int) -> list[YProfile]:
    fast_pwm = int(_scaled_pwm_for_cmd("u", y_cfg.get("mast_pwm", 255)))
    return [
        YProfile("precision", 40, 40, 1000, 1),
        YProfile("slow", 80, 80, 800, 1),
        YProfile("medium", 140, 140, 700, 1),
        YProfile("down_precision_140", 80, 140, 700, 1),
        YProfile("down_precision_180", 80, 180, 600, 1),
        YProfile("down_precision_220", 80, 220, 500, 1),
        YProfile("fast", fast_pwm, fast_pwm, 1000, max(1, int(fallback_cycles))),
    ]


def _profiles_for_args(args, y_cfg: dict) -> list[YProfile]:
    if args.profile:
        return [_profile_from_spec(spec, int(args.cycles)) for spec in args.profile]
    if args.single:
        pulse_ms = int(round(float(args.pulse_ms)))
        if pulse_ms <= 0 or pulse_ms > MAX_SINGLE_DIRECTION_MS:
            raise ValueError(f"--pulse-ms must be 1..{MAX_SINGLE_DIRECTION_MS}")
        up_pwm = int(args.up_pwm) if args.up_pwm is not None else int(_scaled_pwm_for_cmd("u", y_cfg.get("mast_pwm", 255)))
        down_pwm = int(args.down_pwm) if args.down_pwm is not None else int(_scaled_pwm_for_cmd("d", y_cfg.get("mast_pwm", 255)))
        return [
            YProfile(
                "single",
                max(1, min(255, int(up_pwm))),
                max(1, min(255, int(down_pwm))),
                int(pulse_ms),
                max(1, int(args.cycles)),
            )
        ]
    return _default_profiles(y_cfg, int(args.cycles))


def _usable_sample(sample: TelemetrySample, min_confidence_pct: float) -> bool:
    return bool(sample.found) and float(sample.confidence_pct) >= float(min_confidence_pct)


def _sample_once(read_fn, *, min_confidence_pct: float) -> YSample:
    sample = read_fn()
    if not _usable_sample(sample, min_confidence_pct):
        return YSample(
            found=False,
            y_mm=None,
            dist_mm=None,
            x_mm=None,
            confidence_pct=float(sample.confidence_pct),
            source=str(sample.source),
        )
    return YSample(
        found=True,
        y_mm=float(sample.y_mm),
        dist_mm=float(sample.dist_mm),
        x_mm=float(sample.x_mm),
        confidence_pct=float(sample.confidence_pct),
        source=str(sample.source),
    )


def _sample_y(read_fn, *, count: int, interval_s: float, min_confidence_pct: float) -> YSample:
    samples: list[YSample] = []
    for idx in range(max(1, int(count))):
        try:
            sample = _sample_once(read_fn, min_confidence_pct=min_confidence_pct)
        except Exception:
            sample = YSample(False, None, None, None, None, "error")
        if sample.found:
            samples.append(sample)
        if idx + 1 < int(count):
            time.sleep(max(0.0, float(interval_s)))
    if not samples:
        return YSample(False, None, None, None, None, "none")
    return YSample(
        found=True,
        y_mm=float(statistics.median(float(s.y_mm) for s in samples if s.y_mm is not None)),
        dist_mm=float(statistics.median(float(s.dist_mm) for s in samples if s.dist_mm is not None)),
        x_mm=float(statistics.median(float(s.x_mm) for s in samples if s.x_mm is not None)),
        confidence_pct=float(statistics.median(float(s.confidence_pct) for s in samples if s.confidence_pct is not None)),
        source=samples[-1].source,
    )


def _fmt_sample(sample: YSample) -> str:
    if not sample.found:
        conf = "N/A" if sample.confidence_pct is None else f"{sample.confidence_pct:.0f}%"
        return f"missing conf={conf}"
    return (
        f"y={sample.y_mm:+.2f}mm dist={sample.dist_mm:.1f}mm "
        f"x={sample.x_mm:+.1f}mm conf={sample.confidence_pct:.0f}%"
    )


def _delta(a: YSample, b: YSample) -> float | None:
    if not a.found or not b.found or a.y_mm is None or b.y_mm is None:
        return None
    return float(b.y_mm) - float(a.y_mm)


def _rate(delta_mm: float | None, pulse_ms: int) -> float | None:
    if delta_mm is None or int(pulse_ms) <= 0:
        return None
    return float(delta_mm) * 100.0 / float(pulse_ms)


def _sample_drift(a: YSample, b: YSample, attr: str) -> float | None:
    if not a.found or not b.found:
        return None
    av = getattr(a, attr)
    bv = getattr(b, attr)
    if av is None or bv is None:
        return None
    return abs(float(bv) - float(av))


def _cycle_quality(args, before: YSample, after_up: YSample, after_down: YSample) -> tuple[bool, str]:
    if not before.found or not after_up.found or not after_down.found:
        return False, "missing_sample"
    max_dist = float(args.max_dist_drift_mm)
    max_x = float(args.max_x_drift_mm)
    checks = (
        ("dist_up", _sample_drift(before, after_up, "dist_mm"), max_dist),
        ("dist_down", _sample_drift(after_up, after_down, "dist_mm"), max_dist),
        ("x_up", _sample_drift(before, after_up, "x_mm"), max_x),
        ("x_down", _sample_drift(after_up, after_down, "x_mm"), max_x),
    )
    bad = [
        f"{name}={value:.1f}>{limit:.1f}"
        for name, value, limit in checks
        if value is not None and float(value) > float(limit)
    ]
    if bad:
        return False, ",".join(bad)
    return True, "ok"


def _send_mast(robot: Robot, cmd: str, pwm: int, pulse_ms: int) -> None:
    if int(pulse_ms) > MAX_SINGLE_DIRECTION_MS:
        raise ValueError(f"pulse_ms {pulse_ms} exceeds {MAX_SINGLE_DIRECTION_MS}ms safety cap")
    robot.send_command_pwm(str(cmd), int(pwm), duration_ms=int(pulse_ms))
    time.sleep(float(pulse_ms) / 1000.0)
    robot.stop()


def _run(args) -> dict:
    y_cfg = _follow_y_axis_config()
    profiles = _profiles_for_args(args, y_cfg)

    if args.dry_run:
        for profile in profiles:
            print(
                f"[Y-CAL] dry run: profile={profile.name} cycles={profile.cycles} "
                f"pulse={profile.pulse_ms}ms up_pwm={profile.up_pwm} down_pwm={profile.down_pwm}"
            )
        return {"ok": True, "dry_run": True, "cycles": []}

    read_fn, close_reader, source_used = _make_reader(args)
    robot = Robot(exit_on_failure=False, serial_port=args.serial_port)
    records: list[YCycle] = []
    unmatched_up_ms = 0
    cleanup_down_pwm = profiles[0].down_pwm

    try:
        print(
            f"[Y-CAL] source={source_used} profiles={len(profiles)} "
            f"max_one_direction={MAX_SINGLE_DIRECTION_MS}ms",
            flush=True,
        )
        robot.stop()
        time.sleep(max(0.0, float(args.settle_s)))

        for profile in profiles:
            print(
                f"[Y-CAL] profile {profile.name}: cycles={profile.cycles} "
                f"pulse={profile.pulse_ms}ms up_pwm={profile.up_pwm} down_pwm={profile.down_pwm}",
                flush=True,
            )
            for cycle in range(1, int(profile.cycles) + 1):
                before = _sample_y(
                    read_fn,
                    count=args.samples,
                    interval_s=args.sample_interval_s,
                    min_confidence_pct=args.min_confidence_pct,
                )
                print(f"[Y-CAL] {profile.name} cycle {cycle} before   {_fmt_sample(before)}", flush=True)

                print(f"[Y-CAL] {profile.name} cycle {cycle} UP {profile.pulse_ms}ms", flush=True)
                cleanup_down_pwm = int(profile.down_pwm)
                _send_mast(robot, "u", profile.up_pwm, profile.pulse_ms)
                unmatched_up_ms += int(profile.pulse_ms)
                time.sleep(max(0.0, float(args.settle_s)))
                after_up = _sample_y(
                    read_fn,
                    count=args.samples,
                    interval_s=args.sample_interval_s,
                    min_confidence_pct=args.min_confidence_pct,
                )
                print(f"[Y-CAL] {profile.name} cycle {cycle} after up {_fmt_sample(after_up)}", flush=True)

                print(f"[Y-CAL] {profile.name} cycle {cycle} DOWN {profile.pulse_ms}ms", flush=True)
                _send_mast(robot, "d", profile.down_pwm, profile.pulse_ms)
                unmatched_up_ms -= int(profile.pulse_ms)
                time.sleep(max(0.0, float(args.settle_s)))
                after_down = _sample_y(
                    read_fn,
                    count=args.samples,
                    interval_s=args.sample_interval_s,
                    min_confidence_pct=args.min_confidence_pct,
                )
                print(f"[Y-CAL] {profile.name} cycle {cycle} after dn {_fmt_sample(after_down)}", flush=True)

                up_delta = _delta(before, after_up)
                down_delta = _delta(after_up, after_down)
                quality_ok, quality_reason = _cycle_quality(args, before, after_up, after_down)
                record = YCycle(
                    profile=profile.name,
                    cycle=cycle,
                    pulse_ms=profile.pulse_ms,
                    up_pwm=profile.up_pwm,
                    down_pwm=profile.down_pwm,
                    before=before,
                    after_up=after_up,
                    after_down=after_down,
                    up_delta_mm=up_delta,
                    down_delta_mm=down_delta,
                    up_mm_per_100ms=_rate(up_delta, profile.pulse_ms),
                    down_mm_per_100ms=_rate(down_delta, profile.pulse_ms),
                    quality_ok=bool(quality_ok),
                    quality_reason=str(quality_reason),
                )
                records.append(record)
                print(
                    f"[Y-CAL] {profile.name} cycle {cycle} delta: "
                    f"up={up_delta if up_delta is not None else 'N/A'}mm "
                    f"({record.up_mm_per_100ms if record.up_mm_per_100ms is not None else 'N/A'} mm/100ms), "
                    f"down={down_delta if down_delta is not None else 'N/A'}mm "
                    f"({record.down_mm_per_100ms if record.down_mm_per_100ms is not None else 'N/A'} mm/100ms) "
                    f"quality={record.quality_reason}",
                    flush=True,
                )
    finally:
        if unmatched_up_ms > 0:
            try:
                safe_ms = min(int(unmatched_up_ms), MAX_SINGLE_DIRECTION_MS)
                print(f"[Y-CAL] cleanup matching DOWN {safe_ms}ms for unmatched UP", flush=True)
                _send_mast(robot, "d", cleanup_down_pwm, safe_ms)
            except Exception:
                pass
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

    summaries = {}
    for profile in profiles:
        profile_records = [r for r in records if r.profile == profile.name and r.quality_ok]
        up_rates = [r.up_mm_per_100ms for r in profile_records if r.up_mm_per_100ms is not None]
        down_rates = [r.down_mm_per_100ms for r in profile_records if r.down_mm_per_100ms is not None]
        summaries[profile.name] = {
            "samples": len(profile_records),
            "median_up_mm_per_100ms": statistics.median(up_rates) if up_rates else None,
            "median_down_mm_per_100ms": statistics.median(down_rates) if down_rates else None,
        }
        up_text = "N/A" if not up_rates else f"{statistics.median(up_rates):+.2f}"
        down_text = "N/A" if not down_rates else f"{statistics.median(down_rates):+.2f}"
        print(
            f"[Y-CAL] summary {profile.name}: good_cycles={len(profile_records)} "
            f"up={up_text} mm/100ms down={down_text} mm/100ms",
            flush=True,
        )
    return {
        "ok": True,
        "dry_run": False,
        "source": source_used,
        "cycles": [asdict(r) for r in records],
        "summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate mast y movement with safe paired up/down pulses.")
    parser.add_argument("--source", choices=("auto", "stream", "camera"), default="auto")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--stream-timeout-s", type=float, default=1.0)
    parser.add_argument("--pulse-ms", type=int, default=DEFAULT_PULSE_MS)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    parser.add_argument("--min-confidence-pct", type=float, default=DEFAULT_MIN_CONFIDENCE_PCT)
    parser.add_argument("--max-dist-drift-mm", type=float, default=DEFAULT_MAX_DIST_DRIFT_MM)
    parser.add_argument("--max-x-drift-mm", type=float, default=DEFAULT_MAX_X_DRIFT_MM)
    parser.add_argument("--up-pwm", type=int, default=None)
    parser.add_argument("--down-pwm", type=int, default=None)
    parser.add_argument(
        "--profile",
        action="append",
        help=(
            "Add a calibration profile. Format: name:pwm:pulse_ms[:cycles] "
            "or name:up_pwm:down_pwm:pulse_ms[:cycles]. Can be repeated."
        ),
    )
    parser.add_argument("--single", action="store_true", help="Use only --pulse-ms/--cycles/--up-pwm/--down-pwm instead of the default sweep.")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = _run(args)
    except KeyboardInterrupt:
        print("\n[Y-CAL] Interrupted.", flush=True)
        return 130
    except Exception as exc:
        print(f"[Y-CAL] ERROR: {exc}", flush=True)
        return 1
    if args.json:
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
