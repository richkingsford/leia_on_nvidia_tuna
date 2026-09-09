#!/usr/bin/env python3
"""Small smoke test for the Leia Arduino Uno motor interface.

The test is dry-run by default. Add --execute only when Leia is clear to move.
"""

from __future__ import annotations

import argparse
import time

import serial


BAUD_RATE = 115200
FORWARD_MS = 2000
BACK_MS = 2000
MAST_MS = 500
TREAD_POWER = 100


def send_packet(port: serial.Serial, left: int, right: int, mast: int) -> None:
    packet = f"<{left},{right},{mast}>".encode("ascii")
    port.write(packet)
    port.flush()


def pulse(
    port: serial.Serial | None,
    name: str,
    left: int,
    right: int,
    mast: int,
    duration_ms: int,
) -> None:
    print(f"{name}: <{left},{right},{mast}> for {duration_ms} ms")
    if port is None:
        return
    send_packet(port, left, right, mast)
    time.sleep(duration_ms / 1000.0)
    send_packet(port, 0, 0, 0)
    time.sleep(0.15)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Leia's forward/back/mast smoke test.")
    parser.add_argument("--execute", action="store_true", help="Actually move Leia.")
    parser.add_argument("--port", default="/dev/leia-uno", help="Arduino serial device.")
    parser.add_argument("--treads-only", action="store_true", help="Skip the mast pulses.")
    args = parser.parse_args()

    port: serial.Serial | None = None
    try:
        if args.execute:
            print(f"Opening {args.port} at {BAUD_RATE} baud...")
            port = serial.Serial(args.port, BAUD_RATE, timeout=1)
            # Allow the Uno to reset when the serial port opens.
            time.sleep(2.0)

        pulse(port, "Forward", TREAD_POWER, TREAD_POWER, 0, FORWARD_MS)
        pulse(port, "Back", -TREAD_POWER, -TREAD_POWER, 0, BACK_MS)
        if not args.treads_only:
            pulse(port, "Mast up", 0, 0, 40, MAST_MS)
            pulse(port, "Mast down", 0, 0, -40, MAST_MS)
        print("Smoke test complete.")
        if not args.execute:
            print("Dry run: no commands were sent. Re-run with --execute when clear.")
        return 0
    finally:
        if port is not None:
            try:
                send_packet(port, 0, 0, 0)
            finally:
                port.close()


if __name__ == "__main__":
    raise SystemExit(main())
