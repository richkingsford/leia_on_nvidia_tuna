#!/usr/bin/env python3
import argparse
import json
import sys
import termios
import tty
import time
from pathlib import Path

from helper_robot_control import Robot
from telemetry_robot import ROBOT_MODEL_FILE, DEFAULT_SPEED_MODEL, interp_pwm_for_score


CMD_KEYS = ("f", "b", "l", "r")
CMD_LABELS = {
    "f": "Forward",
    "b": "Reverse",
    "l": "Turn Left",
    "r": "Turn Right",
}

SPEED_MAP_KEY_DRIVE = "score_power_pwm_drive"
SPEED_MAP_KEY_TURN = "score_power_pwm_turn"


def _get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return ch + seq
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _load_model(path):
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                model = data
        except (OSError, json.JSONDecodeError):
            model = DEFAULT_SPEED_MODEL

    def _coerce_pwm(value, fallback):
        if value is None:
            value = fallback
        try:
            pwm = int(round(float(value)))
        except (TypeError, ValueError):
            pwm = int(round(float(fallback)))
        return max(0, min(255, pwm))

    hotkeys = model.get("hotkey_speed_scores") or {}
    score_seconds = model.get("speed_score_seconds") or {}

    legacy_map = model.get("score_power_pwm")
    drive_map = model.get(SPEED_MAP_KEY_DRIVE) or {}
    turn_map = model.get(SPEED_MAP_KEY_TURN) or {}
    if isinstance(legacy_map, dict):
        if not drive_map:
            drive_map = legacy_map
        if not turn_map:
            turn_map = legacy_map

    if not hotkeys:
        hotkeys = DEFAULT_SPEED_MODEL["hotkey_speed_scores"]
    if not drive_map:
        drive_map = DEFAULT_SPEED_MODEL.get(SPEED_MAP_KEY_DRIVE) or {}
    if not turn_map:
        turn_map = DEFAULT_SPEED_MODEL.get(SPEED_MAP_KEY_TURN) or {}
    if not score_seconds:
        score_seconds = DEFAULT_SPEED_MODEL.get("speed_score_seconds") or {}

    min_pwm = _coerce_pwm(model.get("min_pwm"), DEFAULT_SPEED_MODEL.get("min_pwm", 0))
    max_pwm = _coerce_pwm(model.get("max_pwm"), DEFAULT_SPEED_MODEL.get("max_pwm", 255))
    if max_pwm < min_pwm:
        min_pwm, max_pwm = max_pwm, min_pwm

    model["hotkey_speed_scores"] = dict(hotkeys)
    model[SPEED_MAP_KEY_DRIVE] = dict(drive_map)
    model[SPEED_MAP_KEY_TURN] = dict(turn_map)
    model["speed_score_seconds"] = dict(score_seconds)
    model["min_pwm"] = int(min_pwm)
    model["max_pwm"] = int(max_pwm)
    return model


def _save_model(path, model):
    path.write_text(json.dumps(model, indent=2))


def _run_demo(robot, cmd, pwm, duration_s, duration_ms=None):
    act_ms = robot.CMD_DURATION if duration_ms is None else int(duration_ms)
    interval = max(0.05, act_ms / 1000.0)
    start = time.time()
    while time.time() - start < duration_s:
        if hasattr(robot, "send_command_pwm"):
            robot.send_command_pwm(cmd, pwm, duration_ms=act_ms)
        else:
            robot.send_command(cmd, pwm / 255.0, duration_ms=act_ms)
        time.sleep(interval)
    robot.stop()


def _coerce_int(value, default):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _clamp_pwm(pwm, max_pwm):
    return max(0, min(int(max_pwm), _coerce_int(pwm, 0)))


def _default_pwm_from_score(score_map, score, fallback, max_pwm):
    entry = score_map.get(str(int(score))) if isinstance(score_map, dict) else None
    if not isinstance(entry, dict):
        return _clamp_pwm(fallback, max_pwm)
    return _clamp_pwm(entry.get("pwm", fallback), max_pwm)


def _pwm_to_power(pwm, min_pwm, max_pwm):
    try:
        raw = float(pwm)
    except (TypeError, ValueError):
        return 0.0
    raw = max(0.0, min(255.0, raw))
    if raw <= 0.0:
        return 0.0
    try:
        min_val = float(min_pwm)
        max_val = float(max_pwm)
    except (TypeError, ValueError):
        return 0.0
    min_val = max(0.0, min(255.0, min_val))
    max_val = max(0.0, min(255.0, max_val))
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    span = max(1.0, max_val - min_val)
    p = (raw - min_val) / span
    return max(0.0, min(1.0, p))


def _set_score_entry(score_map, score, pwm, *, min_pwm, max_pwm):
    pwm_val = _clamp_pwm(pwm, max_pwm)
    score_map[str(int(score))] = {
        "power": round(_pwm_to_power(pwm_val, min_pwm, max_pwm), 3),
        "pwm": int(pwm_val),
    }


def _hw_cmd(cmd, cmd_remap):
    if isinstance(cmd_remap, dict) and cmd in cmd_remap:
        return str(cmd_remap.get(cmd) or cmd)
    return cmd


def _print_minmax_menu(model_path, cmd_remap, cmd_speed, *, speed_map_key, speed_map_label):
    slow_pwm = cmd_speed.get("slow_pwm", 0)
    fast_pwm = cmd_speed.get("fast_pwm", 0)
    pwm_3 = interp_pwm_for_score(3, slow_pwm, fast_pwm) or 0
    pwm_50 = interp_pwm_for_score(50, slow_pwm, fast_pwm) or 0
    print("\nLeia Speed Endpoint Calibration")
    print("------------------------------")
    print(f"Model: {model_path}")
    print(f"This writes ONLY speed scores 1 and 100 in {speed_map_key} ({speed_map_label}).")
    print("Intermediate scores (e.g. 3, 50) are computed with a straight-line curve.")
    print("")
    print(f"Endpoints: 1% pwm={int(slow_pwm)} | 100% pwm={int(fast_pwm)}")
    print(f"Derived:   3% pwm={int(pwm_3)} | 50% pwm={int(pwm_50)}")
    print("")
    print("Demo commands (uses current selected PWM):")
    for cmd in CMD_KEYS:
        label = CMD_LABELS.get(cmd, cmd)
        hw = _hw_cmd(cmd, cmd_remap)
        print(f"  {cmd.upper()} = {label:9s} (hw {hw.upper()})")
    print("")


def _speed_map_key_for_cmd(cmd):
    return SPEED_MAP_KEY_TURN if cmd in ("l", "r") else SPEED_MAP_KEY_DRIVE


def _speed_map_label_for_cmd(cmd):
    return "turn (l/r)" if cmd in ("l", "r") else "drive (f/b)"


def main():
    parser = argparse.ArgumentParser(description="Leia speed calibration (1/100 endpoints)")
    parser.add_argument("--model", default=str(ROBOT_MODEL_FILE))
    parser.add_argument(
        "--pwm-step",
        "--step",
        dest="pwm_step",
        type=float,
        default=1.0,
        help="PWM adjustment per arrow key (rounded, min 1)",
    )
    parser.add_argument("--duration", type=float, default=0.75, help="demo duration in seconds")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = _load_model(model_path)
    cmd_remap = model.get("command_remap") if isinstance(model, dict) else {}

    min_pwm = model.get("min_pwm", DEFAULT_SPEED_MODEL.get("min_pwm", 0))
    max_pwm = model.get("max_pwm", DEFAULT_SPEED_MODEL.get("max_pwm", 255))
    min_pwm = _coerce_int(min_pwm, 0)
    max_pwm = _coerce_int(max_pwm, 255)
    min_pwm = max(0, min(255, int(min_pwm)))
    max_pwm = max(0, min(255, int(max_pwm)))
    if max_pwm < min_pwm:
        min_pwm, max_pwm = max_pwm, min_pwm

    drive_map = model.get(SPEED_MAP_KEY_DRIVE) if isinstance(model, dict) else None
    if not isinstance(drive_map, dict):
        drive_map = {}
        model[SPEED_MAP_KEY_DRIVE] = drive_map
    turn_map = model.get(SPEED_MAP_KEY_TURN) if isinstance(model, dict) else None
    if not isinstance(turn_map, dict):
        turn_map = {}
        model[SPEED_MAP_KEY_TURN] = turn_map

    drive_slow_pwm = _default_pwm_from_score(drive_map, 1, min_pwm, max_pwm)
    drive_fast_pwm = _default_pwm_from_score(drive_map, 100, max_pwm, max_pwm)
    if drive_fast_pwm < drive_slow_pwm:
        drive_slow_pwm, drive_fast_pwm = drive_fast_pwm, drive_slow_pwm
    endpoints_drive = {1: drive_slow_pwm, 100: drive_fast_pwm}

    turn_slow_pwm = _default_pwm_from_score(turn_map, 1, min_pwm, max_pwm)
    turn_fast_pwm = _default_pwm_from_score(turn_map, 100, max_pwm, max_pwm)
    if turn_fast_pwm < turn_slow_pwm:
        turn_slow_pwm, turn_fast_pwm = turn_fast_pwm, turn_slow_pwm
    endpoints_turn = {1: turn_slow_pwm, 100: turn_fast_pwm}

    base_step_pwm = max(1, _coerce_int(args.pwm_step, 1))

    robot = Robot()
    try:
        demo_cmd = "f"
        target_score = 1  # 1 | 100
        pwm = int(endpoints_drive[target_score])
        step_pwm = base_step_pwm

        while True:
            speed_map_key = _speed_map_key_for_cmd(demo_cmd)
            speed_map_label = _speed_map_label_for_cmd(demo_cmd)
            endpoints = endpoints_turn if speed_map_key == SPEED_MAP_KEY_TURN else endpoints_drive

            cmd_speed = {"slow_pwm": endpoints[1], "fast_pwm": endpoints[100]}
            _print_minmax_menu(
                model_path,
                cmd_remap,
                cmd_speed,
                speed_map_key=speed_map_key,
                speed_map_label=speed_map_label,
            )

            cmd = demo_cmd
            label = CMD_LABELS.get(cmd, cmd)
            hw = _hw_cmd(cmd, cmd_remap)
            pwm = _clamp_pwm(pwm, max_pwm)
            power = _pwm_to_power(pwm, min_pwm, max_pwm)

            print(f"\n[{label}] cmd {cmd.upper()} (hw {hw.upper()})")
            print(f"Editing: {target_score}% ({speed_map_label}) | step {step_pwm} | pwm {pwm} | power {power:.3f}")
            print("Keys: d demo | ↑/↓ pwm | ←/→ step | t toggle 1/100 | f/b/l/r demo cmd | s save | x quit")
            print("Input: ", end="", flush=True)
            ch = _get_key()
            print("")
            if not ch:
                continue
            if ch in ("x", "X", "\x03"):
                break
            if ch in ("?", "h", "H"):
                print("Controls:")
                print("  d = demo this movement at current pwm")
                print("  ↑/↓ = adjust pwm")
                print("  ←/→ = adjust step size")
                print("  t = toggle endpoint (1% / 100%)")
                print("  f/b/l/r = switch demo command")
                print("  s = save endpoints to world_model_robot.json")
                print("  x = quit")
                continue
            if ch in ("t", "T"):
                target_score = 100 if target_score == 1 else 1
                pwm = int(endpoints[target_score])
                continue
            if ch == "\x1b[A":  # up arrow
                pwm = _clamp_pwm(pwm + step_pwm, max_pwm)
                continue
            if ch == "\x1b[B":  # down arrow
                pwm = _clamp_pwm(pwm - step_pwm, max_pwm)
                continue
            if ch == "\x1b[C":  # right arrow
                step_pwm = min(50, int(step_pwm) + 1)
                continue
            if ch == "\x1b[D":  # left arrow
                step_pwm = max(1, int(step_pwm) - 1)
                continue
            if ch in ("d", "D", " "):
                _run_demo(robot, cmd, pwm, args.duration, duration_ms=robot.CMD_DURATION)
                continue
            if ch in ("s", "S"):
                endpoints[target_score] = int(_clamp_pwm(pwm, max_pwm))
                slow_pwm = int(endpoints[1])
                fast_pwm = int(endpoints[100])
                if slow_pwm > fast_pwm:
                    slow_pwm, fast_pwm = fast_pwm, slow_pwm
                    endpoints[1] = slow_pwm
                    endpoints[100] = fast_pwm
                speed_map_key = _speed_map_key_for_cmd(demo_cmd)
                score_map = model.get(speed_map_key) if isinstance(model, dict) else None
                if not isinstance(score_map, dict):
                    score_map = {}
                    model[speed_map_key] = score_map
                score_map.clear()
                _set_score_entry(score_map, 1, slow_pwm, min_pwm=min_pwm, max_pwm=max_pwm)
                _set_score_entry(score_map, 100, fast_pwm, min_pwm=min_pwm, max_pwm=max_pwm)
                try:
                    model.pop("cmd_speed_minmax_pwm", None)
                except Exception:
                    pass
                try:
                    model.pop("score_power_pwm", None)
                except Exception:
                    pass
                _save_model(model_path, model)
                print(f"Saved endpoints ({speed_map_key}): 1% pwm={slow_pwm}, 100% pwm={fast_pwm}")
                continue
            cmd_candidate = ch.lower()
            if cmd_candidate in CMD_KEYS:
                prev_map_key = _speed_map_key_for_cmd(demo_cmd)
                demo_cmd = cmd_candidate
                step_pwm = base_step_pwm
                next_map_key = _speed_map_key_for_cmd(demo_cmd)
                if next_map_key != prev_map_key:
                    endpoints = endpoints_turn if next_map_key == SPEED_MAP_KEY_TURN else endpoints_drive
                    pwm = int(endpoints[target_score])
                continue
            print(f"Unknown key: {ch!r}")
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
