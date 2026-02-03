
import sys
import os
from pathlib import Path

# Add current directory to path so we can import telemetry_robot
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "cub-queso"))

from telemetry_robot import WorldModel, MotionEvent

def test_asymmetry():
    print("--- Verifying Turning Asymmetry ---")
    wm = WorldModel()
    
    print(f"Loaded Turn Efficiency L: {wm.turn_efficiency_l}")
    print(f"Loaded Turn Efficiency R: {wm.turn_efficiency_r}")
    
    # Expected values from world_model_robot.json: 88.04, 59.25
    assert wm.turn_efficiency_l == 88.04
    assert wm.turn_efficiency_r == 59.25
    print("SUCCESS: Efficiency values loaded correctly.")
    
    # Test update_from_motion
    # Left Turn
    wm.theta = 0.0
    # Create a left turn event: 50% power for 1 second
    # power_ratio = 127/255 ~= 0.5
    # rot_pulse = deg_per_sec (90) * 0.5 * 1.0 * 0.5 * (88.04/100) = 45 * 0.5 * 0.8804 = 19.809
    event_l = MotionEvent("left_turn", power=127, duration_ms=1000)
    wm.update_from_motion(event_l)
    print(f"Theta after 1s Left Turn @ 50% power: {wm.theta:.3f}")
    
    # Reset and Right Turn
    wm.theta = 0.0
    event_r = MotionEvent("right_turn", power=127, duration_ms=1000)
    wm.update_from_motion(event_r)
    print(f"Theta after 1s Right Turn @ 50% power: {wm.theta:.3f}")
    
    # Check asymmetry in results
    if abs(wm.theta) < 19.809 * 0.8: # Should be roughly 19.8 vs 13.3
         print("Asymmetry detected and applied.")
    else:
         print("Something is wrong, no asymmetry detected in theta change.")

if __name__ == "__main__":
    try:
        test_asymmetry()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
