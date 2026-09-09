#!/usr/bin/env python3
"""Find Leia's lowest consistent tread power.

Each power is pulsed several times in forward and reverse, with a pause between
pulses. Judge a setting as usable only if every pulse breaks away consistently.
Dry-run is the default; add --execute when Leia is clear to move.
"""

from __future__ import annotations

import argparse
import time

import serial


BAUD_RATE = 115200
DEFAULT_DURATION_MS = 750
DEFAULT_REPEATS = 3
DEFAULT_REST_S = 0.75


def send_packet(port: serial.Serial, left: int, right: int) -> None:
    port.write(f"<{left},{right},0>".encode("ascii"))
    port.flush()


def pulse(
    port: serial.Serial | None,
    label: str,
    left: int,
    right: int,
    duration_ms: int,
    rest_s: float,
) -> None:
    print(f"{label}: <{left},{right},0> for {duration_ms} ms")
    if port is not None:
        send_packet(port, left, right)
        time.sleep(duration_ms / 1000.0)
        send_packet(port, 0, 0)
        time.sleep(rest_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Leia tread breakaway power.")
    parser.add_argument("--execute", action="store_true", help="Actually move Leia.")
    parser.add_argument("--port", default="/dev/leia-uno", help="Arduino serial device.")
    parser.add_argument("--min-power", type=int, default=25)
    parser.add_argument("--max-power", type=int, default=35)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--duration-ms", type=int, default=DEFAULT_DURATION_MS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rest-s", type=float, default=DEFAULT_REST_S)
    args = parser.parse_args()

    if args.step <= 0 or not 0 <= args.min_power <= args.max_power <= 100:
        parser.error("require 0 <= min-power <= max-power <= 100 and step > 0")
    if args.duration_ms <= 0 or args.repeats <= 0 or args.rest_s < 0:
        parser.error("duration-ms and repeats must be positive; rest-s cannot be negative")

    powers = range(args.min_power, args.max_power + 1, args.step)
    port: serial.Serial | None = None
    try:
        if args.execute:
            print(f"Opening {args.port} at {BAUD_RATE} baud...")
            port = serial.Serial(args.port, BAUD_RATE, timeout=1)
            time.sleep(2.0)

        print("Test each power in both directions. Record the lowest setting that moves on all repeats.")
        for power in powers:
            print(f"\n--- PAIRED SCENARIO: {power}% ---")
            for repeat in range(1, args.repeats + 1):
                pulse(
                    port,
                    f"forward power {power}% ({repeat}/{args.repeats})",
                    power,
                    power,
                    args.duration_ms,
                    args.rest_s,
                )
                pulse(
                    port,
                    f"reverse power {power}% ({repeat}/{args.repeats})",
                    -power,
                    -power,
                    args.duration_ms,
                    args.rest_s,
                )

        print("\nBreakaway test complete.")
        if not args.execute:
            print("Dry run: no commands were sent. Re-run with --execute when clear.")
        return 0
    finally:
        if port is not None:
            try:
                send_packet(port, 0, 0)
            finally:
                port.close()


if __name__ == "__main__":
    raise SystemExit(main())
