# AGENTS.md

## Architecture and Clean Code Rules (Leia)
- Leia lays bricks using world model JSON files and demo logs in `demos/`.
- Robot motion must use the movements defined in `world_model_robot.json` only (no overrides or conflicts).
- Robot motion must only occur after an explicit, current gap assessment and a chosen correction plan; never send movement commands while sampling/observing frames (no keepalive motion during sensing).
- `telemetry*` files contain robot/interaction logic.
- `debug*` files are never imported or used by other scripts.
- `helper*` files are helpers and should be used whenever possible.
- `setup_manual_training.py` is a key file but must not contain robot behavior logic (use helpers).
- `autobuild.py` is a key file but must not contain robot behavior logic (use helpers).
- Minimize duplicate code.
- Exceptions and overrides are a last resort; use only when necessary and explain why.
