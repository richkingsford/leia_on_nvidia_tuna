import time
import statistics
from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from telemetry_robot import manual_speed_for_cmd

def main():
    robot = Robot()
    vision = ArucoBrickVision(debug=False)
    
    print("--- TURN VERIFICATION TEST WITH SEARCH ---")
    
    def get_x(samples=10, timeout=2.0):
        poses = []
        start_t = time.time()
        while len(poses) < samples and (time.time() - start_t < timeout):
            found, _, _, off_x, _, _, _, _ = vision.read()
            if found:
                poses.append(off_x)
            time.sleep(0.05)
        if not poses: return None
        return statistics.median(poses)

    def find_brick():
        print("[SEARCH] Searching for brick...")
        for i in range(1, 10):
            scan_dir = 'l' if i % 2 == 0 else 'r'
            duration = 0.2 + (i * 0.1)
            print(f"  Pulse {scan_dir} for {duration:.1f}s")
            robot.send_command(scan_dir, 0.4)
            time.sleep(duration)
            robot.stop()
            time.sleep(1.0)
            if get_x(samples=2):
                print("[SEARCH] Found brick!")
                return True
        return False

    if get_x() is None:
        if not find_brick():
            print("[CRITICAL] Could not find brick after search. Aborting.")
            robot.close()
            return

    for cmd in ['l', 'r']:
        print(f"\nTesting command: {cmd}")
        x_before = get_x()
        if x_before is None:
            if not find_brick(): break
            x_before = get_x()
        
        print(f"  X Before: {x_before:.1f}mm")
        
        # Move
        robot.send_command(cmd, 0.4) # Moderate power
        print(f"  Moving {cmd}...")
        time.sleep(0.5)
        robot.stop()
        
        time.sleep(1.0) # Settle
        x_after = get_x()
        
        if x_after is None:
            print(f"  [ERROR] Brick lost after '{cmd}'!")
        else:
            change = x_after - x_before
            print(f"  X After: {x_after:.1f}mm")
            print(f"  Change: {change:+.1f}mm")
            
            if change > 2.0:
                print(f"  Result: '{cmd}' moved brick RIGHT (+)")
            elif change < -2.0:
                print(f"  Result: '{cmd}' moved brick LEFT (-)")
            else:
                print(f"  Result: '{cmd}' produced NO SIGNIFICANT MOVEMENT (jitter?)")
                
    robot.close()

if __name__ == "__main__":
    main()
