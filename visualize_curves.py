#!/usr/bin/env python3
"""Visualize all calibration curves - distance traveled vs duration."""
import json
import matplotlib.pyplot as plt
import numpy as np

def load_curve(path):
    """Load curve data from JSON file."""
    with open(path) as f:
        return json.load(f)

def plot_all_curves():
    """Create plots showing distance traveled vs duration for all curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Robot Calibration Curves - Distance vs Duration', fontsize=16, fontweight='bold')
    
    # Load all curve files
    left_right = load_curve('world_model_left_right_curve.json')
    up_down = load_curve('world_model_up_down_curve.json')
    forward_backward = load_curve('world_model_forward_backward_curve.json')
    
    # Plot 1: X-axis (left/right) at 20% speed
    ax1 = axes[0, 0]
    lr_calib = left_right['aruco_marker_calibration']
    lr_speed = lr_calib['speed_score_pct']
    lr_ref_dist = lr_calib['reference_distance_mm']
    
    l_cmd = lr_calib['by_cmd']['l']
    r_cmd = lr_calib['by_cmd']['r']
    
    durations = np.linspace(lr_calib['duration_range_ms'][0], lr_calib['duration_range_ms'][1], 50)
    l_distances = l_cmd['intercept_mm'] + l_cmd['slope_mm_per_ms'] * durations
    r_distances = r_cmd['intercept_mm'] + r_cmd['slope_mm_per_ms'] * durations
    
    ax1.plot(durations, l_distances, 'b-', label=f"Left ({l_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax1.plot(durations, r_distances, 'r-', label=f"Right ({r_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax1.set_xlabel('Duration (ms)', fontsize=11)
    ax1.set_ylabel('Distance Traveled (mm)', fontsize=11)
    ax1.set_title(f'x_axis curve at {lr_ref_dist:.0f}mm distance at {lr_speed}% speed', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Y-axis (up/down) at 1% speed
    ax2 = axes[0, 1]
    ud_calib = up_down['aruco_marker_calibration']
    ud_speed = ud_calib['speed_score_pct']
    ud_ref_dist = ud_calib['reference_distance_mm']
    
    u_cmd = ud_calib['by_cmd']['u']
    d_cmd = ud_calib['by_cmd']['d']
    
    durations = np.linspace(ud_calib['duration_range_ms'][0], ud_calib['duration_range_ms'][1], 50)
    u_distances = u_cmd['intercept_mm'] + u_cmd['slope_mm_per_ms'] * durations
    d_distances = d_cmd['intercept_mm'] + d_cmd['slope_mm_per_ms'] * durations
    
    ax2.plot(durations, u_distances, 'g-', label=f"Up ({u_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax2.plot(durations, d_distances, 'm-', label=f"Down ({d_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax2.set_xlabel('Duration (ms)', fontsize=11)
    ax2.set_ylabel('Distance Traveled (mm)', fontsize=11)
    ax2.set_title(f'y_axis curve at {ud_ref_dist:.0f}mm distance at {ud_speed}% speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Distance (forward/backward) at 5% speed
    ax3 = axes[1, 0]
    fb_calib = forward_backward['aruco_marker_calibration']
    fb_speed = fb_calib['speed_score_pct']
    fb_ref_dist = fb_calib['reference_distance_mm']
    
    f_cmd = fb_calib['by_cmd']['f']
    b_cmd = fb_calib['by_cmd']['b']
    
    durations = np.linspace(fb_calib['duration_range_ms'][0], fb_calib['duration_range_ms'][1], 50)
    f_distances = f_cmd['intercept_mm'] + f_cmd['slope_mm_per_ms'] * durations
    b_distances = b_cmd['intercept_mm'] + b_cmd['slope_mm_per_ms'] * durations
    
    ax3.plot(durations, f_distances, 'b-', label=f"Forward ({f_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax3.plot(durations, b_distances, 'r-', label=f"Backward ({b_cmd['motion_rate_mm_per_sec']:.1f} mm/s)", linewidth=2)
    ax3.set_xlabel('Duration (ms)', fontsize=11)
    ax3.set_ylabel('Distance Traveled (mm)', fontsize=11)
    ax3.set_title(f'distance curve at {fb_ref_dist:.1f}mm distance at {fb_speed}% speed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Axis', 'Speed', 'Ref Dist', 'Duration Range', 'Rate'],
        ['X (left)', f'{lr_speed}%', f'{lr_ref_dist:.0f}mm', f"{lr_calib['duration_range_ms'][0]}-{lr_calib['duration_range_ms'][1]}ms", f"{l_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
        ['X (right)', f'{lr_speed}%', f'{lr_ref_dist:.0f}mm', f"{lr_calib['duration_range_ms'][0]}-{lr_calib['duration_range_ms'][1]}ms", f"{r_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
        ['Y (up)', f'{ud_speed}%', f'{ud_ref_dist:.0f}mm', f"{ud_calib['duration_range_ms'][0]}-{ud_calib['duration_range_ms'][1]}ms", f"{u_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
        ['Y (down)', f'{ud_speed}%', f'{ud_ref_dist:.0f}mm', f"{ud_calib['duration_range_ms'][0]}-{ud_calib['duration_range_ms'][1]}ms", f"{d_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
        ['Dist (fwd)', f'{fb_speed}%', f'{fb_ref_dist:.1f}mm', f"{fb_calib['duration_range_ms'][0]}-{fb_calib['duration_range_ms'][1]}ms", f"{f_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
        ['Dist (bwd)', f'{fb_speed}%', f'{fb_ref_dist:.1f}mm', f"{fb_calib['duration_range_ms'][0]}-{fb_calib['duration_range_ms'][1]}ms", f"{b_cmd['motion_rate_mm_per_sec']:.1f} mm/s"],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.15, 0.12, 0.15, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Calibration Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('robot_calibration_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved robot_calibration_curves.png")

if __name__ == '__main__':
    plot_all_curves()
