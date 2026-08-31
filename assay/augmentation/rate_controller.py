"""Token counting and rate verification utilities for augmentation rate enforcement."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*\b|\b\d+(?:\.\d+)?\b')


def get_unique_tokens(text: str) -> set[str]:
    """Extract unique tokens (words and numbers) from text, case-insensitive."""
    return {tok.lower() for tok in _TOKEN_PATTERN.findall(text)}


def count_unique_tokens(text: str) -> int:
    return len(get_unique_tokens(text))


def compute_token_rate(original: str, perturbed: str) -> float:
    """Compute the fraction of original unique tokens that were changed."""
    orig_tokens = get_unique_tokens(original)
    if not orig_tokens:
        return 0.0
    pert_tokens = get_unique_tokens(perturbed)
    changed = orig_tokens - pert_tokens
    return len(changed) / len(orig_tokens)


def verify_token_rate(
    original: str,
    perturbed: str,
    target_rate: float,
    tolerance_tokens: int = 1,
    tolerance_pct: float = 0.005,
) -> tuple[bool, float, int]:
    """Verify that the actual token change rate is within tolerance of target.

    Returns:
        (is_ok, actual_rate, diff) where diff = actual_changed - expected_changed.
    """
    orig_tokens = get_unique_tokens(original)
    n_tokens = len(orig_tokens)
    if n_tokens == 0:
        return (True, 0.0, 0)

    pert_tokens = get_unique_tokens(perturbed)
    changed = orig_tokens - pert_tokens
    actual_changed = len(changed)
    expected_changed = round(n_tokens * target_rate)

    diff = actual_changed - expected_changed
    actual_rate = actual_changed / n_tokens

    is_ok = (
        abs(diff) <= tolerance_tokens
        or abs(actual_rate - target_rate) <= tolerance_pct
    )
    return (is_ok, actual_rate, diff)
