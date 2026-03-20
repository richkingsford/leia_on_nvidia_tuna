#!/usr/bin/env python3
"""
Animated 2D plot of robot brick observations (x_axis vs y_axis) from camera perspective.
Green rectangle shows success gate boundaries.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Load data
with open("Runs - aruco/Mar 13 217pm.json", "r") as f:
    data = json.load(f)

# Extract state observations (only the "decision point" state right before each action)
observations = []
for i, entry in enumerate(data):
    if entry.get("type") == "action":
        # Walk backward to find the state immediately before this action
        for j in range(i - 1, -1, -1):
            prev = data[j]
            if prev.get("type") == "state" and prev.get("brick", {}).get("visible"):
                brick = prev["brick"]
                x = brick.get("x_axis")
                y = brick.get("y_axis")
                if x is not None and y is not None:
                    observations.append({
                        "x": float(x),
                        "y": float(y),
                        "t": float(prev["timestamp"]),
                        "action": entry.get("command", ""),
                    })
                break

print(f"{len(observations)} observation checkpoints")

# Success gates from world model
X_TARGET, X_TOL = -4.74, 1.4   # x band: -6.14 to -3.34
Y_TARGET, Y_TOL = 2.5, 1.5     # y band:  1.0  to  4.0

gate_x_min = X_TARGET - X_TOL
gate_x_max = X_TARGET + X_TOL
gate_y_min = Y_TARGET - Y_TOL
gate_y_max = Y_TARGET + Y_TOL

x_vals = [o["x"] for o in observations]
y_vals = [o["y"] for o in observations]

# Figure setup
fig, ax = plt.subplots(figsize=(10, 8))
pad = 3
ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)

# Success gate rectangle (green, 1px solid)
gate_rect = patches.Rectangle(
    (gate_x_min, gate_y_min),
    gate_x_max - gate_x_min,
    gate_y_max - gate_y_min,
    linewidth=1, edgecolor="green", facecolor="green", alpha=0.08,
)
ax.add_patch(gate_rect)
# Redraw border on top
gate_border = patches.Rectangle(
    (gate_x_min, gate_y_min),
    gate_x_max - gate_x_min,
    gate_y_max - gate_y_min,
    linewidth=1, edgecolor="green", facecolor="none",
)
ax.add_patch(gate_border)

# Target crosshair
ax.plot(X_TARGET, Y_TARGET, "g+", markersize=14, markeredgewidth=1.5, zorder=5)

ax.set_xlabel("X-axis (mm) — left/right", fontsize=11)
ax.set_ylabel("Y-axis (mm) — up/down", fontsize=11)
ax.set_title("Mar 13 2:17pm — Brick Alignment Sequence (camera view)", fontsize=13)
ax.grid(True, alpha=0.2)

# Plot elements
trail_line, = ax.plot([], [], "-", color="#888", alpha=0.4, linewidth=1)
trail_dots, = ax.plot([], [], "o", color="#4488cc", markersize=4, alpha=0.5)
current_dot, = ax.plot([], [], "o", color="red", markersize=9, zorder=10)
info_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
)

trail_x, trail_y = [], []

def init():
    trail_line.set_data([], [])
    trail_dots.set_data([], [])
    current_dot.set_data([], [])
    info_text.set_text("")
    return trail_line, trail_dots, current_dot, info_text

def animate(frame):
    obs = observations[frame]
    x, y = obs["x"], obs["y"]
    trail_x.append(x)
    trail_y.append(y)

    trail_line.set_data(trail_x, trail_y)
    trail_dots.set_data(trail_x, trail_y)
    current_dot.set_data([x], [y])

    elapsed = obs["t"] - observations[0]["t"]
    x_in = gate_x_min <= x <= gate_x_max
    y_in = gate_y_min <= y <= gate_y_max
    x_status = "✓ green" if x_in else "✗ red"
    y_status = "✓ green" if y_in else "✗ red"

    info_text.set_text(
        f"T{frame + 1}/{len(observations)}  +{elapsed:.1f}s\n"
        f"x={x:+.2f}mm  ({x_status})\n"
        f"y={y:.2f}mm  ({y_status})\n"
        f"action → {obs['action']}"
    )
    return trail_line, trail_dots, current_dot, info_text

anim = FuncAnimation(
    fig, animate, init_func=init,
    frames=len(observations),
    interval=400,  # 400ms per checkpoint
    blit=True, repeat=True,
)

out = "robot_movement_animation.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=3, dpi=120)
print(f"Done → {out}")
