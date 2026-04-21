#!/usr/bin/env python3
"""Helper calibration launcher for motion and telemetry workflows.

This helper can run as a standalone script or be invoked from the main
manual-training tool as an interactive calibration mode. Motion-duration
calibrations and telemetry-value calibrations both live here so operators can
stay in one menu and one livestream.
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import helper_random_back_turn_experiment
from calibration import helper_calibrate_dist
from calibration.helper_calibrate import get_shared_calibration_context, get_shared_stream_runtime, use_shared_stream_runtime
from calibration import helper_calibrate_motion
from calibration import helper_calibrate_speed
from calibration import helper_calibrate_x_dist
from calibration import helper_calibrate_x
from calibration import helper_calibrate_x_axis
from calibration import helper_calibrate_y
import helper_calibrate_telemetry
import helper_manual_drive_breakaway_test
import helper_manual_turn_breakaway_test
from helper_manual_config import load_manual_training_config
from helper_stream_server import format_stream_url


ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
ANSI_RESET = "\033[0m"


def _orange_text(text: str) -> str:
    return f"{ANSI_ORANGE_BRIGHT}{str(text)}{ANSI_RESET}"


def _default_stream_url() -> str:
    cfg = load_manual_training_config()
    host = str(cfg.get("stream_host", "127.0.0.1"))
    try:
        port = int(cfg.get("stream_port", 5000))
    except (TypeError, ValueError):
        port = 5000
    return str(format_stream_url(host, port))


TRIALS_DIR_DEFAULT = Path(__file__).resolve().parent / "trials"


@dataclass(frozen=True)
class CalibrateOption:
    key: str
    label: str
    runner: Callable[[], int | None]
    borrow_manual_runtime: bool = False


OPTIONS: tuple[CalibrateOption, ...] = (
    CalibrateOption("x", "X-axis duration curve calibration", helper_calibrate_x.main),
    CalibrateOption("y", "Y-axis duration curve calibration", helper_calibrate_y.main),
    CalibrateOption("dist", "Distance duration curve calibration", helper_calibrate_dist.main),
    CalibrateOption("x-dist", "Combined one-shot X+Distance duration probe", helper_calibrate_x_dist.main),
    CalibrateOption("telemetry", "Telemetry value calibration (dist/x/y)", helper_calibrate_telemetry.main),
    CalibrateOption("dist-guided", "Distance curve calibration (guided checkpoints)", lambda: run_guided_distance_calibration()),
    CalibrateOption("breakaway", "Drive breakaway test (find first reliable low-speed score)", helper_manual_drive_breakaway_test.main),
    CalibrateOption("turn-breakaway", "Turn breakaway test (find the slowest raw turn PWM that still moves)", helper_manual_turn_breakaway_test.main),
    CalibrateOption("speed", "Speed endpoint calibration", helper_calibrate_speed.main),
    CalibrateOption("motion", "Motion tick conversion calibration", helper_calibrate_motion.main),
    CalibrateOption("x-axis-legacy", "Legacy X-axis turn-score learning experiment", helper_calibrate_x_axis.main),
)


def _parse_checkpoint_list(raw_text: str) -> list[float]:
    values: list[float] = []
    for token in str(raw_text or "").replace(";", ",").split(","):
        text = str(token or "").strip()
        if not text:
            continue
        number = float(text)
        if number <= 0:
            raise ValueError("Checkpoint distances must be positive millimeters.")
        values.append(float(number))
    if not values:
        raise ValueError("No checkpoint distances provided.")
    return values


def _default_runs_dir_for_vision(vision: str) -> str:
    vision_key = str(vision or "").strip().lower()
    suffix = "aruco" if vision_key == "aruco" else "cyan"
    return str(Path(__file__).resolve().parent / f"Runs - {suffix}")


def run_guided_distance_calibration() -> int:
    parser = argparse.ArgumentParser(
        description="Guided multi-checkpoint distance duration curve calibration.",
        add_help=True,
    )
    parser.add_argument(
        "--checkpoints-mm",
        type=str,
        default="120,140,160,180,200,220",
        help="Comma-separated target distance checkpoints in mm.",
    )
    parser.add_argument(
        "--trials-per-checkpoint",
        type=int,
        default=12,
        help="Number of trials to run at each checkpoint.",
    )
    parser.add_argument("--vision", choices=["leia", "yolo", "aruco"], default="leia")
    parser.add_argument("--speed-score", type=int, default=5)
    parser.add_argument("--min-duration-ms", type=int, default=200)
    parser.add_argument("--max-duration-ms", type=int, default=400)
    parser.add_argument("--observe-samples", type=int, default=7)
    parser.add_argument("--observe-timeout-s", type=float, default=2.8)
    parser.add_argument("--post-act-settle-s", type=float, default=0.10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--allow-direction-mismatch", action="store_true")
    args, passthrough = parser.parse_known_args(sys.argv[1:])

    try:
        checkpoints_mm = _parse_checkpoint_list(args.checkpoints_mm)
    except ValueError as exc:
        print(f"[CALIBRATE_DIST_GUIDED] Invalid --checkpoints-mm: {exc}")
        return 2

    trials_per_checkpoint = max(1, int(args.trials_per_checkpoint))
    output_dir = Path(args.output_dir or _default_runs_dir_for_vision(args.vision))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[CALIBRATE_DIST_GUIDED] Starting guided distance duration calibration.")
    print(
        "[CALIBRATE_DIST_GUIDED] "
        f"checkpoints_mm={[int(round(v)) if float(v).is_integer() else v for v in checkpoints_mm]} "
        f"trials_per_checkpoint={trials_per_checkpoint} vision={args.vision}"
    )
    print("[CALIBRATE_DIST_GUIDED] For each checkpoint: place robot, press Enter, then automated probe runs.")
    print("[CALIBRATE_DIST_GUIDED] Type 'q' then Enter at any checkpoint to stop early.")

    failures: list[tuple[float, int]] = []
    original_argv = list(sys.argv)
    try:
        total = len(checkpoints_mm)
        for idx, checkpoint_mm in enumerate(checkpoints_mm, start=1):
            label_mm = int(round(checkpoint_mm)) if float(checkpoint_mm).is_integer() else round(checkpoint_mm, 2)
            print("")
            print(f"[CALIBRATE_DIST_GUIDED] Checkpoint {idx}/{total}: target {label_mm}mm")
            print(
                f"[CALIBRATE_DIST_GUIDED] Move robot to about {label_mm}mm from the brick, "
                "keep brick visible, then press Enter to run this checkpoint."
            )
            ready = input("Ready? [Enter=run, q=quit]: ").strip().lower()
            if ready in ("q", "quit", "exit"):
                print("[CALIBRATE_DIST_GUIDED] Stopped by operator.")
                break

            stem = f"calibrate_dist_{str(label_mm).replace('.', 'p')}mm"
            results_path = output_dir / f"{stem}.json"
            run_args = [
                "calibrate:dist-guided-step",
                "--trials",
                str(trials_per_checkpoint),
                "--vision",
                str(args.vision),
                "--target-dist-mm",
                str(float(checkpoint_mm)),
                "--speed-score",
                str(int(args.speed_score)),
                "--min-duration-ms",
                str(int(args.min_duration_ms)),
                "--max-duration-ms",
                str(int(args.max_duration_ms)),
                "--observe-samples",
                str(int(args.observe_samples)),
                "--observe-timeout-s",
                str(float(args.observe_timeout_s)),
                "--post-act-settle-s",
                str(float(args.post_act_settle_s)),
                "--results-file",
                str(results_path),
            ]
            if bool(args.show_plot):
                run_args.append("--show-plot")
            if bool(args.allow_direction_mismatch):
                run_args.append("--allow-direction-mismatch")
            run_args.extend(list(passthrough))

            print(f"[CALIBRATE_DIST_GUIDED] Running automated probe at {label_mm}mm...")
            sys.argv = run_args
            step_code = int(helper_calibrate_dist.main() or 0)
            if step_code != 0:
                failures.append((float(checkpoint_mm), int(step_code)))
                print(
                    f"[CALIBRATE_DIST_GUIDED] Checkpoint {label_mm}mm ended with code={step_code}. "
                    f"Saved partial output to {results_path}"
                )
            else:
                print(
                    f"[CALIBRATE_DIST_GUIDED] Completed {label_mm}mm. "
                    f"Results: {results_path}"
                )
    finally:
        sys.argv = original_argv

    if failures:
        print("[CALIBRATE_DIST_GUIDED] Completed with checkpoint failures:")
        for distance_mm, code in failures:
            label = int(round(distance_mm)) if float(distance_mm).is_integer() else round(distance_mm, 2)
            print(f"  - {label}mm (code={code})")
        return 1

    print("[CALIBRATE_DIST_GUIDED] Guided distance duration calibration complete.")
    return 0


def _load_trial_manifest(path: Path) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _trial_file_paths(trials_dir: Path | None = None) -> list[Path]:
    root = Path(trials_dir) if trials_dir is not None else TRIALS_DIR_DEFAULT
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file()
            and not str(path.name).startswith(".")
            and str(path.suffix).lower() == ".json"
        ],
        key=lambda path: str(path.name).lower(),
    )


def _run_trial_manifest_file(path: Path) -> int:
    manifest = _load_trial_manifest(path)
    if not isinstance(manifest, dict):
        print(f"[CALIBRATE] Invalid trial file: {path}")
        return 2
    shared_context = get_shared_calibration_context() or {}
    robot = shared_context.get("robot")
    world = shared_context.get("world")
    vision = shared_context.get("vision")
    vision_mode = str(shared_context.get("vision_mode") or "").strip() or None
    curve_cfg = helper_random_back_turn_experiment._trial_manifest_curve_config(manifest)
    created_robot = False
    if robot is None:
        robot = helper_random_back_turn_experiment.Robot()
        created_robot = True
    if world is None:
        world = helper_random_back_turn_experiment.WorldModel()
        world.step_state = helper_random_back_turn_experiment.StepState.ALIGN_BRICK
    try:
        result = helper_random_back_turn_experiment.run_close_dist_x_axis_one_act_experiment(
            robot=robot,
            world=world,
            vision=vision,
            vision_mode=vision_mode,
            yolo_model_path=None,
            score=int(curve_cfg.get("score_pct") or helper_random_back_turn_experiment.DEFAULT_SCORE),
            trials=int(
                len(helper_random_back_turn_experiment._trial_manifest_trials_list(manifest))
                or curve_cfg.get("trial_count")
                or helper_random_back_turn_experiment.DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS
            ),
            phase_duration_ms=int(
                curve_cfg.get("measured_phase_duration_ms")
                or helper_random_back_turn_experiment.DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_DURATION_MS
            ),
            phase=str(curve_cfg.get("phase") or helper_random_back_turn_experiment.DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_PHASE),
            strength=str(
                curve_cfg.get("turn_strength")
                or helper_random_back_turn_experiment.DEFAULT_ALTERNATING_TURN_DRIVE_STRENGTH
            ),
            distance_band_mm=float(
                curve_cfg.get("distance_band_mm")
                or helper_random_back_turn_experiment.DEFAULT_ALTERNATING_TURN_DRIVE_DISTANCE_BAND_MM
            ),
            observe_timeout_s=float(helper_random_back_turn_experiment.DEFAULT_OBSERVE_TIMEOUT_S),
            relaxed_timeout_s=float(helper_random_back_turn_experiment.DEFAULT_RELAXED_TIMEOUT_S),
            setup_forward_range_ms=curve_cfg.get("setup_turn_duration_range_ms"),
            trial_manifest=manifest,
            trials_path=Path(path),
            log_path=None,
        )
        print(json.dumps(result, indent=2))
        return 0 if bool(result.get("ok")) else 1
    finally:
        if created_robot:
            try:
                robot.close()
            except Exception:
                pass


def _trial_file_options(trials_dir: Path | None = None) -> tuple[CalibrateOption, ...]:
    rows: list[CalibrateOption] = []
    for path in _trial_file_paths(trials_dir):
        manifest = _load_trial_manifest(path)
        stem = str(path.stem)
        if isinstance(manifest, dict):
            curve_cfg = helper_random_back_turn_experiment._trial_manifest_curve_config(manifest)
            trial_count = int(
                len(helper_random_back_turn_experiment._trial_manifest_trials_list(manifest))
                or curve_cfg.get("trial_count")
                or helper_random_back_turn_experiment.DEFAULT_CLOSE_DIST_X_AXIS_ONE_ACT_TRIALS
            )
            suffix = f" ({trial_count} trials)"
        else:
            suffix = " (invalid file)"
        rows.append(
            CalibrateOption(
                key=str(stem),
                label=f"Trial File: {stem}{suffix}",
                runner=(lambda trial_path=Path(path): _run_trial_manifest_file(trial_path)),
                borrow_manual_runtime=True,
            )
        )
    return tuple(rows)


def _menu_options() -> tuple[CalibrateOption, ...]:
    return tuple(OPTIONS) + tuple(_trial_file_options())


def _print_menu() -> None:
    options = _menu_options()
    base_count = len(OPTIONS)
    print("\nCalibration Options")
    print("-------------------")
    print("Run motion-duration, breakaway, and telemetry-value calibrations.\n")
    for index, option in enumerate(options, start=1):
        if index == base_count + 1:
            print("\nTrial Files")
            print("-----------")
        print(f"  {index}. {option.label} [{option.key}]")
    if len(options) == base_count:
        print("\nTrial Files")
        print("-----------")
        print(f"  none found in {TRIALS_DIR_DEFAULT}")
    print("  q. Quit")


def _resolve_choice(text: str) -> CalibrateOption | None:
    token = str(text or "").strip().lower()
    if not token:
        return None
    for index, option in enumerate(_menu_options(), start=1):
        if token in (str(index), str(option.key).lower()):
            return option
    return None


def _pick_interactive() -> CalibrateOption | None:
    while True:
        _print_menu()
        choice = input("Select calibration to run: ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            return None
        selected = _resolve_choice(choice)
        if selected is not None:
            return selected
        print(f"Unknown selection: {choice!r}. Please choose a number, key, or q.")


def _has_cli_flag(args: list[str], flag: str) -> bool:
    target = str(flag)
    for item in args:
        text = str(item or "")
        if text == target or text.startswith(f"{target}="):
            return True
    return False


def _prompt_int_value(prompt: str, *, minimum: int = 0) -> int:
    while True:
        raw = input(str(prompt)).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < int(minimum):
            print(f"Please enter a number >= {int(minimum)}.")
            continue
        return int(value)


def _interactive_trial_args(option: CalibrateOption, passthrough_args: list[str]) -> list[str]:
    key = str(option.key).strip().lower()
    if key not in {"x", "y", "dist"}:
        return []
    if _has_cli_flag(passthrough_args, "--trials") or _has_cli_flag(passthrough_args, "--repeat-trials"):
        return []

    print("\nTrial Setup")
    print("-----------")
    num_trials = _prompt_int_value("How many trials should I run? ", minimum=1)
    return [
        "--trials",
        str(int(num_trials)),
    ]


def _run_selected(option: CalibrateOption, passthrough_args: list[str]) -> int:
    original_argv = list(sys.argv)
    try:
        sys.argv = [f"calibrate:{option.key}"] + list(passthrough_args)
        result = option.runner()
        if result is None:
            return 0
        return int(result)
    finally:
        sys.argv = original_argv


def _restore_tty_line_input_mode() -> None:
    """Best-effort restore for terminals left in raw/no-echo mode."""
    stream = getattr(sys, "stdin", None)
    if stream is None:
        return
    try:
        if not stream.isatty():
            return
        fd = stream.fileno()
        attrs = termios.tcgetattr(fd)
        iflag = int(attrs[0])
        lflag = int(attrs[3])
        # Ensure Enter maps CR->NL so input() receives a completed line.
        attrs[0] = (iflag | termios.ICRNL) & ~termios.INLCR & ~termios.IGNCR
        attrs[3] = lflag | termios.ICANON | termios.ECHO | termios.ISIG
        cc = attrs[6]
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        attrs[6] = cc
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        # Keep launcher resilient across non-POSIX shells or redirected stdin.
        return


def _print_stream_banner(shared_stream_url: str | None = None) -> None:
    active_url = str(shared_stream_url or "").strip()
    if not active_url:
        _shared_state, active_url = get_shared_stream_runtime()
    if not active_url:
        active_url = _default_stream_url()
    print(f"[CALIBRATE] Livestream URL: {_orange_text(active_url)}")


def pick_interactive_option() -> CalibrateOption | None:
    _restore_tty_line_input_mode()
    return _pick_interactive()


def run_option(
    option: CalibrateOption,
    *,
    passthrough_args: list[str] | None = None,
    shared_stream_state: dict | None = None,
    shared_stream_url: str | None = None,
    shared_context: dict | None = None,
) -> int:
    run_args = list(passthrough_args or [])
    run_args.extend(_interactive_trial_args(option, list(run_args)))
    print(f"Running: {option.label}")
    with use_shared_stream_runtime(
        stream_state=shared_stream_state,
        stream_url=shared_stream_url,
        context=shared_context,
    ):
        return _run_selected(option, run_args)


def run_interactive_session(
    *,
    passthrough_args: list[str] | None = None,
    show_banner: bool = True,
    shared_stream_state: dict | None = None,
    shared_stream_url: str | None = None,
    shared_context: dict | None = None,
) -> int:
    if bool(show_banner):
        _print_stream_banner(shared_stream_url=shared_stream_url)

    base_args = list(passthrough_args or [])
    exit_code = 0
    while True:
        selected = pick_interactive_option()
        if selected is None:
            print("[CALIBRATE] Leaving calibration mode.")
            return int(exit_code)

        step_code = run_option(
            selected,
            passthrough_args=list(base_args),
            shared_stream_state=shared_stream_state,
            shared_stream_url=shared_stream_url,
            shared_context=shared_context,
        )
        exit_code = int(step_code) if int(step_code) != 0 else int(exit_code)
        if int(step_code) == 0:
            print(f"[CALIBRATE] Completed: {selected.label}")
        else:
            print(f"[CALIBRATE] {selected.label} ended with code={int(step_code)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive calibration launcher for robot calibrations")
    parser.add_argument(
        "--choice",
        type=str,
        default=None,
        help="Optional non-interactive selection key or menu number.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available options and exit.",
    )
    args, passthrough_args = parser.parse_known_args()

    if bool(args.list):
        _print_menu()
        return 0

    _print_stream_banner()

    if args.choice is not None:
        selected = _resolve_choice(str(args.choice))
        if selected is None:
            print(f"Unknown --choice value: {args.choice!r}")
            _print_menu()
            return 2
        print(f"Running: {selected.label}")
        return _run_selected(selected, passthrough_args)

    return run_interactive_session(passthrough_args=list(passthrough_args), show_banner=False)


if __name__ == "__main__":
    raise SystemExit(main())
