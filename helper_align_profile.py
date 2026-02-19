import json
from pathlib import Path

ALIGN_PROFILE_FILENAME = "world_model_align_profile.json"
ALIGN_PROFILE_CANDIDATE_FILENAME = "world_model_align_profile_candidate.json"
ALIGN_RESULTS_FILENAME = "world_model_align_results.json"


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_align_profile_from_results_payload(results_payload):
    if not isinstance(results_payload, dict):
        return None

    trials = results_payload.get("trials")
    if not isinstance(trials, list) or not trials:
        return None

    valid_trials = [row for row in trials if isinstance(row, dict)]
    if not valid_trials:
        return None

    success_trials = [row for row in valid_trials if bool(row.get("success"))]
    total = len(valid_trials)
    success_count = len(success_trials)
    success_rate = (float(success_count) / float(total)) if total > 0 else 0.0

    total_adjusts = sum(max(0, _safe_int(row.get("adjust_count"), 0)) for row in valid_trials)
    total_overshoot = sum(max(0, _safe_int(row.get("overshoot_acts"), 0)) for row in valid_trials)
    overshoot_rate = (float(total_overshoot) / float(max(1, total_adjusts)))

    success_adjusts = sorted(max(0, _safe_int(row.get("adjust_count"), 0)) for row in success_trials)
    if success_adjusts:
        mid = len(success_adjusts) // 2
        if len(success_adjusts) % 2 == 0:
            median_adjusts = float(success_adjusts[mid - 1] + success_adjusts[mid]) / 2.0
        else:
            median_adjusts = float(success_adjusts[mid])
    else:
        median_adjusts = float(max(0, _safe_int(results_payload.get("config", {}).get("max_adjusts_per_trial"), 20)))

    learned_scores = [max(1, _safe_int(row.get("learned_speed_score"), 1)) for row in success_trials]
    learned_scores.sort()
    if learned_scores:
        mid = len(learned_scores) // 2
        if len(learned_scores) % 2 == 0:
            median_learned_score = float(learned_scores[mid - 1] + learned_scores[mid]) / 2.0
        else:
            median_learned_score = float(learned_scores[mid])
    else:
        median_learned_score = 10.0

    lost_vision_failures = 0
    for row in valid_trials:
        if bool(row.get("success")):
            continue
        reason = str(row.get("reason") or "").strip().lower()
        if reason in {"lost_vision", "pre_reset_failed_vision", "reset_failed_vision"}:
            lost_vision_failures += 1

    turn_speed_scale = 1.0
    dist_speed_scale = 1.0

    if success_rate >= 0.85:
        turn_speed_scale += 0.05
        dist_speed_scale += 0.05
    elif success_rate < 0.75:
        turn_speed_scale -= 0.08
        dist_speed_scale -= 0.08

    if overshoot_rate < 0.02:
        turn_speed_scale += 0.04
    elif overshoot_rate > 0.06:
        turn_speed_scale -= 0.12

    if median_adjusts > 16.0:
        dist_speed_scale -= 0.08
    elif median_adjusts < 10.0:
        dist_speed_scale += 0.04

    if lost_vision_failures > 0:
        dist_speed_scale -= 0.06

    turn_speed_scale = float(_clamp(turn_speed_scale, 0.75, 1.20))
    dist_speed_scale = float(_clamp(dist_speed_scale, 0.75, 1.20))

    # Keep auto-step speed conservative enough for precision gates.
    max_speed_score = int(round(_clamp(median_learned_score + 5.0, 8.0, 20.0)))

    profile = {
        "source": "calibrate_align",
        "source_run_id": _safe_int(results_payload.get("run_id"), 0),
        "generated_at": _safe_float(results_payload.get("finished_at") or results_payload.get("timestamp") or 0.0, 0.0),
        "turn_speed_scale": float(turn_speed_scale),
        "dist_speed_scale": float(dist_speed_scale),
        "max_speed_score": int(max_speed_score),
        "stats": {
            "trials": int(total),
            "success_rate": float(success_rate),
            "overshoot_rate": float(overshoot_rate),
            "median_success_adjusts": float(median_adjusts),
            "lost_vision_failures": int(lost_vision_failures),
            "median_learned_speed_score": float(median_learned_score),
        },
    }
    return profile


def profile_from_results_file(results_path):
    path = Path(results_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return build_align_profile_from_results_payload(payload)


def load_align_profile(workspace_root=None):
    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parent
    profile_path = root / ALIGN_PROFILE_FILENAME
    if profile_path.exists():
        try:
            payload = json.loads(profile_path.read_text())
            if isinstance(payload, dict) and payload:
                return payload
        except Exception:
            pass

    # Fallback: derive profile from latest calibration results.
    return profile_from_results_file(root / ALIGN_RESULTS_FILENAME)


def save_align_profile(profile, workspace_root=None):
    if not isinstance(profile, dict) or not profile:
        return None
    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parent
    profile_path = root / ALIGN_PROFILE_FILENAME
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    return profile_path


def _profile_quality_score(profile):
    if not isinstance(profile, dict):
        return float("-inf")
    stats = profile.get("stats") if isinstance(profile.get("stats"), dict) else {}
    success_rate = _safe_float(stats.get("success_rate"), 0.0)
    overshoot_rate = _safe_float(stats.get("overshoot_rate"), 0.0)
    median_adjusts = _safe_float(stats.get("median_success_adjusts"), 100.0)
    lost_vision = _safe_float(stats.get("lost_vision_failures"), 0.0)
    trials = _safe_float(stats.get("trials"), 0.0)

    score = 0.0
    score += 100.0 * success_rate
    score -= 40.0 * overshoot_rate
    score -= 0.8 * median_adjusts
    score -= 3.0 * lost_vision
    if trials < 8.0:
        score -= 5.0
    return float(score)


def promote_align_profile_if_better(candidate_profile, workspace_root=None, min_improvement=0.25):
    if not isinstance(candidate_profile, dict) or not candidate_profile:
        return {
            "promoted": False,
            "reason": "invalid_candidate",
            "candidate_path": None,
            "champion_path": None,
            "candidate_score": None,
            "champion_score": None,
        }

    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parent
    candidate_path = root / ALIGN_PROFILE_CANDIDATE_FILENAME
    champion_path = root / ALIGN_PROFILE_FILENAME
    candidate_path.write_text(json.dumps(candidate_profile, indent=2) + "\n")

    champion_profile = None
    if champion_path.exists():
        try:
            payload = json.loads(champion_path.read_text())
            if isinstance(payload, dict) and payload:
                champion_profile = payload
        except Exception:
            champion_profile = None

    candidate_score = _profile_quality_score(candidate_profile)
    champion_score = _profile_quality_score(champion_profile)

    promote = champion_profile is None or (candidate_score >= (champion_score + float(min_improvement)))
    reason = "new_champion" if champion_profile is None else "better_score" if promote else "kept_existing"

    if promote:
        champion_path.write_text(json.dumps(candidate_profile, indent=2) + "\n")

    return {
        "promoted": bool(promote),
        "reason": str(reason),
        "candidate_path": str(candidate_path),
        "champion_path": str(champion_path),
        "candidate_score": float(candidate_score),
        "champion_score": (None if champion_profile is None else float(champion_score)),
    }


def inject_align_profile_into_learned_rules(learned_rules, profile):
    base = dict(learned_rules) if isinstance(learned_rules, dict) else {}
    if not isinstance(profile, dict) or not profile:
        return base

    for step_key in ("ALIGN_BRICK", "POSITION_BRICK"):
        step_rules = base.get(step_key)
        if not isinstance(step_rules, dict):
            step_rules = {}
        step_rules["calibration_profile"] = dict(profile)
        base[step_key] = step_rules
    return base
