# AGENTS.md

## Architecture and Clean Code Rules (Leia)
- Leia lays bricks using world model JSON files and demo logs in `demos/`.
- Robot motion must use the movements defined in `world_model_robot.json` only (no overrides or conflicts).
- Robot motion must only occur after an explicit, current gap assessment and a chosen correction plan; never send movement commands while sampling/observing frames (no keepalive motion during sensing).
- Hard rule: Full success gate confirmation (consecutive/majority gatecheck tracker/logs) must not start for any step until the lite gate precheck has passed and the current effective success-gate sample is passing; before that, remain in lite gate mode only.
- Operator-facing command logs/displays must report the logical motion command (physical intent). Raw wire/remapped commands must only appear in explicit wire/debug logs. Follow "do what we say and say what we do" at the operator level.
- `telemetry*` files contain robot/interaction logic.
- `debug*` files are never imported or used by other scripts.
- `helper*` files are helpers and should be used whenever possible.
- `setup_manual_training.py` is a key file but must not contain robot behavior logic (use helpers).
- `autobuild.py` is a key file but must not contain robot behavior logic (use helpers).
- Minimize duplicate code.
- Exceptions and overrides are a last resort; use only when necessary and explain why.
