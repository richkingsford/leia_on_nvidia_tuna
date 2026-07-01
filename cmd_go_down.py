#!/usr/bin/env python3
"""Simple command to move the mast down for a specified duration."""

import sys
import time
from helper_robot_control import Robot

def main():
    duration_ms = 2600  # 2.6 seconds
    speed = 1.0  # Full speed (0.0 to 1.0)
    
    try:
        robot = Robot(exit_on_failure=True)
        print(f"Sending mast down command for {duration_ms}ms...")
        robot.send_command("d", speed, duration_ms=duration_ms)
        print(f"Command sent. Waiting {duration_ms/1000:.1f}s for completion...")
        time.sleep(duration_ms / 1000.0)
        robot.stop()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
