#!/usr/bin/env python3
import argparse
import json
import sys
import termios
import tty
import time
from pathlib import Path

from helper_robot_control import Robot
from telemetry_robot import ROBOT_MODEL_FILE, DEFAULT_SPEED_MODEL, MIN_PWM, MAX_PWM, interp_pwm_for_score, pwm_to_power


CMD_KEYS = ("f", "b", "l", "r")
CMD_LABELS = {
    "f": "Forward",
    "b": "Reverse",
    "l": "Turn Left",
    "r": "Turn Right",
}


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
    hotkeys = model.get("hotkey_speed_scores") or {}
    score_map = model.get("score_power_pwm") or {}
    score_seconds = model.get("speed_score_seconds") or {}
    if not hotkeys:
        hotkeys = DEFAULT_SPEED_MODEL["hotkey_speed_scores"]
    if not score_map:
        score_map = DEFAULT_SPEED_MODEL["score_power_pwm"]
    if not score_seconds:
        score_seconds = DEFAULT_SPEED_MODEL.get("speed_score_seconds") or {}
    model["hotkey_speed_scores"] = dict(hotkeys)
    model["score_power_pwm"] = dict(score_map)
    model["speed_score_seconds"] = dict(score_seconds)
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


def _clamp_pwm(pwm):
    return max(0, min(MAX_PWM, _coerce_int(pwm, 0)))


def _default_pwm_from_score(score_map, score, fallback):
    entry = score_map.get(str(int(score))) if isinstance(score_map, dict) else None
    if not isinstance(entry, dict):
        return _clamp_pwm(fallback)
    return _clamp_pwm(entry.get("pwm", fallback))


def _set_score_entry(score_map, score, pwm):
    pwm_val = _clamp_pwm(pwm)
    score_map[str(int(score))] = {
        "power": round(pwm_to_power(pwm_val) or 0.0, 3),
        "pwm": int(pwm_val),
    }


def _hw_cmd(cmd, cmd_remap):
    if isinstance(cmd_remap, dict) and cmd in cmd_remap:
        return str(cmd_remap.get(cmd) or cmd)
    return cmd


def _print_minmax_menu(model_path, cmd_remap, cmd_speed):
    slow_pwm = cmd_speed.get("slow_pwm", 0)
    fast_pwm = cmd_speed.get("fast_pwm", 0)
    pwm_3 = interp_pwm_for_score(3, slow_pwm, fast_pwm) or 0
    pwm_50 = interp_pwm_for_score(50, slow_pwm, fast_pwm) or 0
    print("\nLeia Speed Endpoint Calibration")
    print("------------------------------")
    print(f"Model: {model_path}")
    print("This writes ONLY speed scores 1 and 100 in score_power_pwm.")
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
    score_map = model["score_power_pwm"]
    slow_pwm = _default_pwm_from_score(score_map, 1, MIN_PWM)
    fast_pwm = _default_pwm_from_score(score_map, 100, MAX_PWM)
    if fast_pwm < slow_pwm:
        slow_pwm, fast_pwm = fast_pwm, slow_pwm
    endpoints = {1: slow_pwm, 100: fast_pwm}

    base_step_pwm = max(1, _coerce_int(args.pwm_step, 1))

    robot = Robot()
    try:
        demo_cmd = "f"
        target_score = 1  # 1 | 100
        pwm = int(endpoints[target_score])
        step_pwm = base_step_pwm

        while True:
            cmd_speed = {"slow_pwm": endpoints[1], "fast_pwm": endpoints[100]}
            _print_minmax_menu(model_path, cmd_remap, cmd_speed)

            cmd = demo_cmd
            label = CMD_LABELS.get(cmd, cmd)
            hw = _hw_cmd(cmd, cmd_remap)
            pwm = _clamp_pwm(pwm)
            power = pwm_to_power(pwm) or 0.0

            print(f"\n[{label}] cmd {cmd.upper()} (hw {hw.upper()})")
            print(f"Editing: {target_score}% | step {step_pwm} | pwm {pwm} | power {power:.3f}")
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
                pwm = _clamp_pwm(pwm + step_pwm)
                continue
            if ch == "\x1b[B":  # down arrow
                pwm = _clamp_pwm(pwm - step_pwm)
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
                endpoints[target_score] = int(_clamp_pwm(pwm))
                slow_pwm = int(endpoints[1])
                fast_pwm = int(endpoints[100])
                if slow_pwm > fast_pwm:
                    slow_pwm, fast_pwm = fast_pwm, slow_pwm
                    endpoints[1] = slow_pwm
                    endpoints[100] = fast_pwm
                score_map = model.get("score_power_pwm") if isinstance(model, dict) else None
                if not isinstance(score_map, dict):
                    score_map = {}
                    model["score_power_pwm"] = score_map
                score_map.clear()
                _set_score_entry(score_map, 1, slow_pwm)
                _set_score_entry(score_map, 100, fast_pwm)
                try:
                    model.pop("cmd_speed_minmax_pwm", None)
                except Exception:
                    pass
                _save_model(model_path, model)
                print(f"Saved endpoints: 1% pwm={slow_pwm}, 100% pwm={fast_pwm}")
                continue
            cmd_candidate = ch.lower()
            if cmd_candidate in CMD_KEYS:
                demo_cmd = cmd_candidate
                step_pwm = base_step_pwm
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
