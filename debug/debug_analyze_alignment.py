
import json

def analyze_logs(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    l_ks = []
    r_ks = []
    
    for entry in data:
        # Some entries have align_cmd.dir, others might have align_dir
        # Based on setup_calibrate_align.py, log_trial uses alignment direction
        # But wait, looking at the JSON, it varies.
        
        k_obs = entry.get("K_observed")
        if k_obs is None or k_obs <= 0:
            continue
            
        align_dir = None
        if "align_cmd" in entry:
            align_dir = entry["align_cmd"].get("dir")
        elif "align_dir" in entry:
            align_dir = entry["align_dir"]
        elif "dir" in entry:
            align_dir = entry["dir"]
            
        if align_dir == 'l':
            l_ks.append(k_obs)
        elif align_dir == 'r':
            r_ks.append(k_obs)
            
    if not l_ks or not r_ks:
        print("Not enough data to compare.")
        return
        
    avg_l = sum(l_ks) / len(l_ks)
    avg_r = sum(r_ks) / len(r_ks)
    
    print(f"Left Turn Avg K: {avg_l:.2f} ({len(l_ks)} samples)")
    print(f"Right Turn Avg K: {avg_r:.2f} ({len(r_ks)} samples)")
    
    if avg_l > avg_r:
        ratio = avg_l / avg_r
        print(f"Left is {ratio:.2f}x faster than Right.")
    else:
        ratio = avg_r / avg_l
        print(f"Right is {ratio:.2f}x faster than Left.")

if __name__ == "__main__":
    analyze_logs("world_model_align.json")
