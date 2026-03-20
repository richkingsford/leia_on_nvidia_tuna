"""Helpers for estimating stack height from inCrosshairs bands."""

from __future__ import annotations


def summarize_crosshair_stack_samples(samples, *, initial_bricks=0) -> dict:
    try:
        initial_bricks = max(0, int(initial_bricks or 0))
    except (TypeError, ValueError):
        initial_bricks = 0
    normalized = [bool(value) for value in list(samples or []) if value is not None]
    bands = []
    for value in normalized:
        if not bands or bool(bands[-1]["state"]) != bool(value):
            bands.append({"state": bool(value), "samples": 1})
        else:
            bands[-1]["samples"] = int(bands[-1]["samples"]) + 1

    true_indices = [idx for idx, band in enumerate(bands) if bool(band.get("state"))]
    if not true_indices:
        return {
            "samples_count": len(normalized),
            "band_count": len(bands),
            "bands": bands,
            "brick_bands": 0,
            "internal_gap_bands": 0,
            "leading_gap_observed": bool(bands and not bool(bands[0].get("state"))),
            "trailing_gap_observed": bool(bands and not bool(bands[-1].get("state"))),
            "initial_bricks": int(initial_bricks),
            "estimated_bricks": None,
            "complete_scan": False,
        }

    first_true_idx = int(true_indices[0])
    last_true_idx = int(true_indices[-1])
    stack_bands = bands[first_true_idx:last_true_idx + 1]
    brick_bands = sum(1 for band in stack_bands if bool(band.get("state")))
    internal_gap_bands = sum(1 for band in stack_bands if not bool(band.get("state")))
    leading_gap_observed = any(not bool(band.get("state")) for band in bands[:first_true_idx])
    trailing_gap_observed = any(not bool(band.get("state")) for band in bands[last_true_idx + 1:])
    estimated_bricks = max(int(brick_bands), int(internal_gap_bands) + 1)
    estimated_bricks = int(estimated_bricks) + int(initial_bricks)
    return {
        "samples_count": len(normalized),
        "band_count": len(bands),
        "bands": bands,
        "brick_bands": int(brick_bands),
        "internal_gap_bands": int(internal_gap_bands),
        "leading_gap_observed": bool(leading_gap_observed),
        "trailing_gap_observed": bool(trailing_gap_observed),
        "initial_bricks": int(initial_bricks),
        "estimated_bricks": int(estimated_bricks),
        "complete_scan": bool(leading_gap_observed and trailing_gap_observed and brick_bands > 0),
    }
