import statistics, time, sys
from collections import Counter
import a_follow_the_brick as A

true_mm = sys.argv[1] if len(sys.argv) > 1 else "?"
v = A.BrickDetector(debug=False)
v.set_runtime_tuning(**dict(A.CROWN_PROFILE_TUNING))
A._set_game_profile("holding")
A._warmup(v)
inner = getattr(v, "_detector", v)

def attr(obj, name):
    try:
        return float(getattr(obj, name))
    except Exception:
        return None

rep, raw, calib, srcs, widths = [], [], [], [], []
for _ in range(40):
    try:
        r = A._read_brick_measurement(v, jump_guard=False)
    except Exception as exc:
        print("read error:", exc); break
    um = r.get("unmasked_target_reading") or r
    if bool(um.get("confident")):
        try:
            rep.append(float(um.get("dist_mm")))
        except Exception:
            pass
        rv = attr(inner, "last_raw_dist") or attr(v, "last_raw_dist")
        cv = attr(inner, "last_bbox_dist") or attr(v, "last_bbox_dist")
        if rv is not None: raw.append(rv)
        if cv is not None: calib.append(cv)
        srcs.append(str(um.get("vision_geometry_source") or r.get("vision_geometry_source") or "?"))
    time.sleep(0.05)
    if len(rep) >= 15:
        break

def med(xs):
    return f"{statistics.median(xs):.1f}" if xs else "N/A"

if rep:
    print(f"TRUE={true_mm}mm  n={len(rep)}  reported_dist_med={med(rep)}mm  "
          f"RAW_width_dist_med={med(raw)}mm  calibrated_med={med(calib)}mm  "
          f"src={Counter(srcs).most_common()}")
else:
    print(f"TRUE={true_mm}mm  no confident frames")
try:
    v.close()
except Exception:
    pass
