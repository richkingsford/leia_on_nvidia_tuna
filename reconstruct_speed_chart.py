#!/usr/bin/env python3
"""Reconstruct the latest physical speed-test chart from captured terminal output."""

from helper_speed_test import write_speed_test_artifacts


def main() -> int:
    records = []

    def add(phase: str, cmd: str, pwm: int, percent: int, idx: int) -> None:
        wire_text = (
            f"l.f.{percent}.150,r.f.{percent}.150"
            if cmd == "f"
            else f"l.f.{percent}.150,r.b.{percent}.150"
        )
        records.append(
            {
                "phase_name": phase,
                "cmd": cmd,
                "t_ms": idx * 150,
                "phase_t_ms": (idx % 10) * 150,
                "phase_step": (idx % 10) + 1,
                "phase_steps": 10,
                "score": 1 if pwm <= 104 else 2,
                "model_power": 0.0,
                "model_pwm": pwm,
                "interval_ms": 150,
                "send_result": {
                    "cmd_sent": cmd,
                    "pwm": pwm,
                    "power": 0.0,
                    "percent": percent,
                    "duration_ms": 150,
                    "wire_text": wire_text,
                },
            }
        )

    idx = 0
    for _ in range(10):
        add("physical forward ramp up", "f", 104, 41, idx)
        idx += 1
    for _ in range(10):
        add("physical forward ramp down", "f", 107, 42, idx)
        idx += 1
    for _ in range(5):
        add("physical backward ramp only", "b", 104, 41, idx)
        idx += 1
    for _ in range(5):
        add("physical backward ramp only", "b", 107, 42, idx)
        idx += 1

    artifacts = write_speed_test_artifacts(
        {"ok": True, "execute": True, "pulses": records},
        log_dir="logs/speed_tests",
        run_label="reconstructed_physical_runs",
        metadata={
            "note": "Reconstructed from Codex terminal output after the physical forward/down and backward-only runs.",
        },
    )
    print(artifacts["json_path"])
    print(artifacts["html_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
