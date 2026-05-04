import json
from pathlib import Path

VISION_MODEL_FILE = Path(__file__).resolve().parent / "world_model_vision.json"
LEGACY_DEMOS_DIR = Path(__file__).resolve().parent / "demos"

VISION_MODE_ARUCO = "aruco"
VISION_MODE_CYAN = "cyan"

_MODE_ALIASES = {
    "aruco": VISION_MODE_ARUCO,
    "marker": VISION_MODE_ARUCO,
    "markers": VISION_MODE_ARUCO,
    "crown": VISION_MODE_CYAN,
    "crown_brick": VISION_MODE_CYAN,
    "crown_bricks": VISION_MODE_CYAN,
    "cyan": VISION_MODE_CYAN,
    "yolo": VISION_MODE_CYAN,
    "markerless": VISION_MODE_CYAN,
}

DEFAULT_VISION_MODEL = {
    "active_mode": VISION_MODE_CYAN,
    "demos_by_mode": {
        VISION_MODE_CYAN: "demos - cyan",
    },
}


def normalize_vision_mode(value, fallback=VISION_MODE_CYAN):
    key = str(value or "").strip().lower()
    mode = _MODE_ALIASES.get(key)
    if mode:
        return mode
    fb = str(fallback or "").strip().lower()
    mode = _MODE_ALIASES.get(fb)
    if mode:
        return mode
    return VISION_MODE_CYAN


def load_vision_model(path=VISION_MODEL_FILE):
    cfg = {
        "active_mode": DEFAULT_VISION_MODEL["active_mode"],
        "demos_by_mode": dict(DEFAULT_VISION_MODEL["demos_by_mode"]),
    }
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        raw = {}
    if isinstance(raw, dict):
        active_raw = raw.get("active_mode")
        cfg["active_mode"] = normalize_vision_mode(active_raw, fallback=cfg["active_mode"])
        demos_raw = raw.get("demos_by_mode")
        if isinstance(demos_raw, dict):
            merged = dict(cfg["demos_by_mode"])
            for mode_key, value in demos_raw.items():
                mode = normalize_vision_mode(mode_key, fallback="")
                if mode not in (VISION_MODE_ARUCO, VISION_MODE_CYAN):
                    continue
                value_text = str(value or "").strip()
                if value_text:
                    merged[mode] = value_text
            cfg["demos_by_mode"] = merged
    return cfg


def active_vision_mode(path=VISION_MODEL_FILE):
    cfg = load_vision_model(path=path)
    return normalize_vision_mode(cfg.get("active_mode"), fallback=VISION_MODE_CYAN)


def demos_dir_for_mode(mode=None, *, path=VISION_MODEL_FILE):
    cfg = load_vision_model(path=path)
    mode_norm = normalize_vision_mode(mode, fallback=cfg.get("active_mode"))
    demos_map = cfg.get("demos_by_mode") if isinstance(cfg, dict) else {}
    demo_path_raw = None
    if isinstance(demos_map, dict):
        demo_path_raw = demos_map.get(mode_norm)
    if not demo_path_raw:
        demo_path_raw = DEFAULT_VISION_MODEL["demos_by_mode"][VISION_MODE_CYAN]
    demos_dir = Path(demo_path_raw)
    if not demos_dir.is_absolute():
        demos_dir = Path(__file__).resolve().parent / demos_dir
    # Compatibility fallback: support historical cyan-mode dataset names while
    # the operator-facing mode is now described as crown bricks.
    if not demos_dir.exists() and mode_norm == VISION_MODE_CYAN:
        for legacy_name in ("demos - cyan", "Cyan demos"):
            legacy_cyan_dir = Path(__file__).resolve().parent / legacy_name
            if legacy_cyan_dir.exists():
                return legacy_cyan_dir
    # Compatibility fallback for environments still using the legacy `demos/`
    # folder and no mode-specific directories yet.
    if not demos_dir.exists() and mode_norm == VISION_MODE_ARUCO and LEGACY_DEMOS_DIR.exists():
        return LEGACY_DEMOS_DIR
    return demos_dir


def demos_dirs_by_mode(*, path=VISION_MODEL_FILE):
    return {
        VISION_MODE_ARUCO: demos_dir_for_mode(VISION_MODE_ARUCO, path=path),
        VISION_MODE_CYAN: demos_dir_for_mode(VISION_MODE_CYAN, path=path),
    }
