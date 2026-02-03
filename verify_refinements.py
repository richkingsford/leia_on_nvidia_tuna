
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "cub-queso"))

import telemetry_brick
from telemetry_robot import WorldModel, MotionEvent, SPEED_SCORE_LEVELS
import telemetry_process

def test_display():
    print("\n--- Verifying Action Display Simplified ---")
    wm = WorldModel()
    # Mocking record_action_display call
    telemetry_process.record_action_display(wm, "test", "l", 0.22, 1)
    display = wm._last_action_display
    print(f"Display string for L 1%: '{display}'")
    
    assert display == "L 1%"
    print("SUCCESS: Action display is simplified.")

if __name__ == "__main__":
    try:
        test_display()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
