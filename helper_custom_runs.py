#!/usr/bin/env python3
"""Helpers for manual-training custom run sequences."""

from __future__ import annotations


CUSTOM_RUN_RESET_STEP = "__custom_run_reset__"
CUSTOM_RUN_RESET_ALIASES = frozenset({"r", "reset"})

CUSTOM_RUN_RESET_MIN_DIST_MM = 120.0
CUSTOM_RUN_RESET_BACKOFF_SCORE = 50
CUSTOM_RUN_RESET_TURN_SCORE = 1
CUSTOM_RUN_RESET_MAX_BACKOFF_ACTS = 24
CUSTOM_RUN_RESET_MAX_TURN_ACTS = 240
CUSTOM_RUN_RESET_INVISIBLE_CONFIRM_READS = 3
CUSTOM_RUN_RESET_CONFIRM_READ_GAP_S = 0.03
CUSTOM_RUN_RESET_POST_ACT_SETTLE_S = 0.05


def is_custom_run_reset_step(step) -> bool:
    return str(step or "").strip().lower() == str(CUSTOM_RUN_RESET_STEP)


def resolve_custom_run_step_token(token, *, resolve_step_token_fn):
    key = str(token or "").strip().lower()
    if not key:
        return None
    if key in CUSTOM_RUN_RESET_ALIASES:
        return CUSTOM_RUN_RESET_STEP
    if not callable(resolve_step_token_fn):
        return None
    return resolve_step_token_fn(key)


def parse_custom_run_steps_csv(steps_csv, *, resolve_step_token_fn):
    step_tokens = [token.strip() for token in str(steps_csv or "").split(",") if token.strip()]
    if not step_tokens:
        return None, "[CUSTOM RUNS] No step codes provided."

    steps = []
    for token in step_tokens:
        obj = resolve_custom_run_step_token(
            token,
            resolve_step_token_fn=resolve_step_token_fn,
        )
        if obj is None:
            return None, f"[CUSTOM RUNS] Unknown step token '{token}'."
        steps.append(obj)
    return steps, None


def custom_run_step_code(step, *, step_code_for_obj_fn):
    if is_custom_run_reset_step(step):
        return "r"
    if not callable(step_code_for_obj_fn):
        return "?"
    try:
        code = step_code_for_obj_fn(step)
    except Exception:
        code = None
    return str(code or "?")


def custom_run_step_name(step):
    if is_custom_run_reset_step(step):
        return "reset"
    return str(getattr(step, "value", step)).lower()
