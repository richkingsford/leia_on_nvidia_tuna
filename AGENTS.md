# AGENTS.md

## Architecture and Clean Code Rules (Leia)...
- Leia lays bricks using world model JSON files and demo logs in `demos/`.
- Robot motion must use the movements defined in `world_model_robot.json` only (no overrides or conflicts).
- Hard rule: Full success gate confirmation (consecutive/majority gatecheck tracker/logs) must not start for any step until the lite gate precheck has passed and the current effective success-gate sample is passing; before that, remain in lite gate mode only.
- Hard rule: Lite gatecheck truth, full gatecheck truth, and operator-facing gate logs must use identical metric semantics (including directionality and tolerance behavior). Never allow logs to report a gate as passing/failing under different logic than runtime gate evaluation.
- Operator-facing command logs/displays must report the logical motion command (physical intent). Raw wire/remapped commands must only appear in explicit wire/debug logs. Follow "do what we say and say what we do" at the operator level.
- `telemetry*` files contain robot/interaction logic.
- `debug*` files are never imported or used by other scripts.
- `helper*` files are helpers and should be used whenever possible.
- Generalized/reusable logic belongs in `helper*` files and must remain step-agnostic by default.
- Step-specific behavior in helpers is prohibited unless explicitly approved by human leadership and documented as an exception with rationale.
- Step-custom details (for example: start gates, success gates, metric directions, speed policies, focus thresholds) must live in world model JSON files, not hardcoded helper branches.
- `a_MAIN.py` is the primary runner and must not contain robot behavior logic (use helpers).
- Minimize duplicate code.
- Exceptions and overrides are a last resort; use only when necessary and explain why.
