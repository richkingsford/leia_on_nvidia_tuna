#!/usr/bin/env python3
"""Visualize ALIGN calibration trials as a rotated animated line chart.

The chart is rotated 90 degrees compared to a typical time-series view:
- Horizontal axis: x-axis error in mm (center line at 0)
- Vertical axis: adjustment index (top to bottom)

One colored line per trial is animated sequentially. Each trial consumes a fixed
animation duration (default 6 seconds), and each adjustment appears as one dot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

_ACTIVE_ANIMATION = None
_ACTIVE_FIGURE = None
DEFAULT_HEADLESS_GIF = Path("align_trials.gif")
DEFAULT_FINAL_PNG = Path("align_trials_final.png")
DEFAULT_PER_TRIAL_GIF_DIR = Path("align_trials_by_trial")


def _is_noninteractive_backend() -> bool:
    backend = str(matplotlib.get_backend() or "").lower()
    non_interactive_tokens = (
        "agg",
        "pdf",
        "svg",
        "ps",
        "cairo",
        "template",
    )
    return any(token in backend for token in non_interactive_tokens)


@dataclass
class TrialSeries:
    trial: int
    x_err_mm: List[float]


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_event_log(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected event list JSON in {path}, got {type(payload).__name__}. "
            "Use world_model_align.json (event log), not aggregate results JSON."
        )
    return payload


def build_trial_series(events: List[dict]) -> List[TrialSeries]:
    starts: Dict[int, float] = {}
    align_acts: Dict[int, List[tuple]] = {}

    for row in events:
        if not isinstance(row, dict):
            continue
        row_type = row.get("type")
        trial = row.get("trial")
        if trial is None:
            continue
        trial_id = _to_int(trial, -1)
        if trial_id < 0:
            continue

        if row_type == "trial_target_snapshot":
            start_x = row.get("start_x_err_mm")
            if start_x is not None:
                starts[trial_id] = _to_float(start_x, 0.0)
            continue

        if row_type == "act" and str(row.get("phase", "")).lower() == "align":
            x_err = row.get("x_err_mm")
            if x_err is None:
                continue
            act_idx = _to_int(row.get("act"), 0)
            ts = _to_float(row.get("timestamp"), 0.0)
            align_acts.setdefault(trial_id, []).append((act_idx, ts, _to_float(x_err, 0.0)))

    trial_ids = sorted(set(list(starts.keys()) + list(align_acts.keys())))
    output: List[TrialSeries] = []
    for trial_id in trial_ids:
        acts = align_acts.get(trial_id, [])
        if not acts and trial_id not in starts:
            continue

        acts_sorted = sorted(acts, key=lambda r: (r[0], r[1]))
        points = [x for _, _, x in acts_sorted]
        if trial_id in starts:
            points = [float(starts[trial_id])] + points
        elif points:
            points = [float(points[0])] + points

        if len(points) < 2:
            points = points + points

        output.append(TrialSeries(trial=trial_id, x_err_mm=points))

    return output


def animate_trials(
    trials: List[TrialSeries],
    seconds_per_trial: float,
    fps: int,
    title: str,
    output_path: Optional[Path] = None,
    final_png_path: Optional[Path] = None,
) -> None:
    if not trials:
        raise ValueError("No trial data found to visualize.")

    fps = max(1, int(fps))
    seconds_per_trial = max(0.1, float(seconds_per_trial))
    frames_per_trial = max(1, int(round(seconds_per_trial * fps)))
    total_frames = frames_per_trial * len(trials)

    max_points = max(len(t.x_err_mm) for t in trials)
    max_abs_x = max(max(abs(x) for x in t.x_err_mm) for t in trials)
    x_lim = max(1.0, max_abs_x * 1.10)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Calibrate Align Trial Visualizer")

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, max_points - 1)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1.6, linestyle="--", alpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("x-axis error (mm)")
    ax.set_ylabel("adjustment index (top = start)")

    color_map = plt.get_cmap("tab20")
    lines = []
    scatters = []
    final_markers = []
    for idx, trial in enumerate(trials):
        color = color_map(idx % 20)
        (line,) = ax.plot([], [], color=color, linewidth=2.0, alpha=0.95, label=f"Trial {trial.trial}")
        scatter = ax.scatter([], [], color=[color], s=22, alpha=0.95)
        final_marker = ax.scatter([], [], color=[color], s=70, alpha=1.0, edgecolors="white", linewidths=0.9)
        lines.append(line)
        scatters.append(scatter)
        final_markers.append(final_marker)

    legend_cols = 2 if len(trials) > 8 else 1
    ax.legend(loc="upper right", ncol=legend_cols, frameon=True, fontsize=9)
    title_artist = ax.set_title(title)

    y_vectors = [list(range(len(t.x_err_mm))) for t in trials]

    def _set_empty(idx: int):
        lines[idx].set_data([], [])
        scatters[idx].set_offsets([[math.nan, math.nan]])
        final_markers[idx].set_offsets([[math.nan, math.nan]])

    def _set_visible_prefix(idx: int, count: int):
        xs = trials[idx].x_err_mm[:count]
        ys = y_vectors[idx][:count]
        lines[idx].set_data(xs, ys)
        scatters[idx].set_offsets(list(zip(xs, ys)))
        if count > 0:
            final_markers[idx].set_offsets([[xs[-1], ys[-1]]])
        else:
            final_markers[idx].set_offsets([[math.nan, math.nan]])

    def update(frame_idx: int):
        trial_slot = min(len(trials) - 1, frame_idx // frames_per_trial)
        local_frame = frame_idx % frames_per_trial
        progress = (local_frame + 1) / float(frames_per_trial)

        for idx in range(len(trials)):
            if idx < trial_slot:
                _set_visible_prefix(idx, len(trials[idx].x_err_mm))
            elif idx > trial_slot:
                _set_empty(idx)
            else:
                n_points = len(trials[idx].x_err_mm)
                show_count = max(1, int(math.ceil(progress * n_points)))
                _set_visible_prefix(idx, show_count)

        current_trial = trials[trial_slot]
        title_artist.set_text(
            f"{title} | Trial {current_trial.trial} ({trial_slot + 1}/{len(trials)})"
        )

        artists = [title_artist]
        artists.extend(lines)
        artists.extend(scatters)
        artists.extend(final_markers)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000.0 / float(fps),
        blit=False,
        repeat=True,
    )

    global _ACTIVE_ANIMATION, _ACTIVE_FIGURE
    _ACTIVE_ANIMATION = anim
    _ACTIVE_FIGURE = fig

    def _save_final_png_if_requested():
        if final_png_path is None:
            return
        update(total_frames - 1)
        plt.tight_layout()
        fig.savefig(str(final_png_path), dpi=160)
        print(f"Saved final frame PNG to {final_png_path}")

    if output_path is not None:
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        print(f"Saved animation to {output_path}")
        _save_final_png_if_requested()
    else:
        plt.tight_layout()
        plt.show(block=True)
        _save_final_png_if_requested()

    return anim


def export_per_trial_gifs(
    trials: List[TrialSeries],
    seconds_per_trial: float,
    fps: int,
    title: str,
    output_dir: Path,
) -> None:
    if not trials:
        raise ValueError("No trial data found to visualize.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(fps))
    seconds_per_trial = max(0.1, float(seconds_per_trial))
    frames_per_trial = max(1, int(round(seconds_per_trial * fps)))

    max_points = max(len(t.x_err_mm) for t in trials)
    max_abs_x = max(max(abs(x) for x in t.x_err_mm) for t in trials)
    x_lim = max(1.0, max_abs_x * 1.10)
    y_vectors = [list(range(len(t.x_err_mm))) for t in trials]
    color_map = plt.get_cmap("tab20")

    for target_idx, target_trial in enumerate(trials):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(0, max_points - 1)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=1.6, linestyle="--", alpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlabel("x-axis error (mm)")
        ax.set_ylabel("adjustment index (top = start)")

        lines = []
        scatters = []
        final_markers = []
        for idx, trial in enumerate(trials):
            color = color_map(idx % 20)
            (line,) = ax.plot([], [], color=color, linewidth=2.0, alpha=0.95, label=f"Trial {trial.trial}")
            scatter = ax.scatter([], [], color=[color], s=22, alpha=0.95)
            final_marker = ax.scatter([], [], color=[color], s=70, alpha=1.0, edgecolors="white", linewidths=0.9)
            lines.append(line)
            scatters.append(scatter)
            final_markers.append(final_marker)

        legend_cols = 2 if len(trials) > 8 else 1
        ax.legend(loc="upper right", ncol=legend_cols, frameon=True, fontsize=9)
        title_artist = ax.set_title("")

        def _set_empty(idx: int):
            lines[idx].set_data([], [])
            scatters[idx].set_offsets([[math.nan, math.nan]])
            final_markers[idx].set_offsets([[math.nan, math.nan]])

        def _set_visible_prefix(idx: int, count: int):
            xs = trials[idx].x_err_mm[:count]
            ys = y_vectors[idx][:count]
            lines[idx].set_data(xs, ys)
            scatters[idx].set_offsets(list(zip(xs, ys)))
            if count > 0:
                final_markers[idx].set_offsets([[xs[-1], ys[-1]]])
            else:
                final_markers[idx].set_offsets([[math.nan, math.nan]])

        def update(frame_idx: int):
            progress = (frame_idx + 1) / float(frames_per_trial)

            for idx in range(len(trials)):
                if idx < target_idx:
                    _set_visible_prefix(idx, len(trials[idx].x_err_mm))
                elif idx > target_idx:
                    _set_empty(idx)
                else:
                    n_points = len(trials[idx].x_err_mm)
                    show_count = max(1, int(math.ceil(progress * n_points)))
                    _set_visible_prefix(idx, show_count)

            title_artist.set_text(
                f"{title} | Trial {target_trial.trial} ({target_idx + 1}/{len(trials)})"
            )

            artists = [title_artist]
            artists.extend(lines)
            artists.extend(scatters)
            artists.extend(final_markers)
            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames_per_trial,
            interval=1000.0 / float(fps),
            blit=False,
            repeat=True,
        )

        output_path = output_dir / f"align_trial_{int(target_trial.trial):02d}.gif"
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        print(f"Saved trial GIF {target_idx + 1}/{len(trials)}: {output_path}")
        plt.close(fig)


def save_final_static_png(trials: List[TrialSeries], title: str, output_png: Path) -> None:
    if not trials:
        raise ValueError("No trial data found to visualize.")

    max_points = max(len(t.x_err_mm) for t in trials)
    max_abs_x = max(max(abs(x) for x in t.x_err_mm) for t in trials)
    x_lim = max(1.0, max_abs_x * 1.10)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, max_points - 1)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1.6, linestyle="--", alpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("x-axis error (mm)")
    ax.set_ylabel("adjustment index (top = start)")
    ax.set_title(f"{title} | All trials complete")

    color_map = plt.get_cmap("tab20")
    for idx, trial in enumerate(trials):
        color = color_map(idx % 20)
        ys = list(range(len(trial.x_err_mm)))
        ax.plot(trial.x_err_mm, ys, color=color, linewidth=2.0, alpha=0.95, label=f"Trial {trial.trial}")
        ax.scatter(trial.x_err_mm, ys, color=[color], s=18, alpha=0.9)
        ax.scatter([trial.x_err_mm[-1]], [ys[-1]], color=[color], s=70, alpha=1.0, edgecolors="white", linewidths=0.9)

    legend_cols = 2 if len(trials) > 8 else 1
    ax.legend(loc="upper right", ncol=legend_cols, frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    print(f"Saved final frame PNG to {output_png}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate ALIGN calibration trial trajectories from world_model_align.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("world_model_align.json"),
        help="Path to event log JSON (default: world_model_align.json)",
    )
    parser.add_argument(
        "--seconds-per-trial",
        type=float,
        default=5.0,
        help="Animation duration per trial in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ALIGN calibration: x-axis trajectory by trial",
        help="Plot title",
    )
    parser.add_argument(
        "--save-gif",
        type=Path,
        default=None,
        help="Optional output GIF path. If omitted, opens interactive window.",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=DEFAULT_FINAL_PNG,
        help="Final-frame PNG path (default: align_trials_final.png).",
    )
    parser.add_argument(
        "--no-save-png",
        action="store_true",
        help="Disable final-frame PNG export.",
    )
    parser.add_argument(
        "--per-trial-gifs",
        action="store_true",
        help="Export one GIF per trial with prior trials shown static and current trial animated.",
    )
    parser.add_argument(
        "--save-gif-dir",
        type=Path,
        default=DEFAULT_PER_TRIAL_GIF_DIR,
        help="Output directory for --per-trial-gifs (default: align_trials_by_trial).",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Export static all-trials PNG only (no animation/GIF).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.save_gif
    if (not bool(args.per_trial_gifs)) and output_path is None and _is_noninteractive_backend():
        output_path = DEFAULT_HEADLESS_GIF
        print(
            f"[viz] Non-interactive Matplotlib backend detected ({matplotlib.get_backend()}). "
            f"Auto-saving animation to {output_path}."
        )

    final_png_path = None if bool(args.no_save_png) else args.save_png

    events = load_event_log(args.input)
    trials = build_trial_series(events)

    if bool(args.static_only):
        if final_png_path is None:
            print("[viz] --static-only requested but PNG export is disabled; nothing to save.")
            return None
        save_final_static_png(trials=trials, title=str(args.title), output_png=final_png_path)
        return None

    if bool(args.per_trial_gifs):
        export_per_trial_gifs(
            trials=trials,
            seconds_per_trial=float(args.seconds_per_trial),
            fps=int(args.fps),
            title=str(args.title),
            output_dir=args.save_gif_dir,
        )
        if final_png_path is not None:
            save_final_static_png(trials=trials, title=str(args.title), output_png=final_png_path)
        return None

    anim = animate_trials(
        trials=trials,
        seconds_per_trial=float(args.seconds_per_trial),
        fps=int(args.fps),
        title=str(args.title),
        output_path=output_path,
        final_png_path=final_png_path,
    )
    return anim


if __name__ == "__main__":
    main()
