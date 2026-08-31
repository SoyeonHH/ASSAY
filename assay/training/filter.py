"""Direction-aware filtering for DPO pair construction.

Representational pairs: higher score = better (less biased) → chosen.
Error pairs: lower score = better (more error-sensitive) → chosen.
"""

from __future__ import annotations


def filter_repr_pair(
    score_a: float,
    score_b: float,
    anchor: float,
    min_delta: float = 0.5,
    anchor_tolerance: float = 1.5,
) -> tuple[bool, float | None, float | None]:
    """Filter a representational pair. Higher score = chosen.

    Args:
        score_a: Score from model A.
        score_b: Score from model B.
        anchor: Anchor score for this sample (Eq. 5 in paper).
        min_delta: Minimum score difference to form a pair.
        anchor_tolerance: Maximum distance of chosen score below anchor.

    Returns:
        (is_valid, score_chosen, score_rejected)
        is_valid is False if pair should be discarded.
    """
    if score_a == score_b:
        return False, None, None

    if score_a > score_b:
        s_chosen, s_rejected = score_a, score_b
    else:
        s_chosen, s_rejected = score_b, score_a

    delta = s_chosen - s_rejected
    if delta < min_delta:
        return False, None, None

    if s_chosen < anchor - anchor_tolerance:
        return False, None, None

    return True, s_chosen, s_rejected


def filter_error_pair(
    score_a: float,
    score_b: float,
    anchor: float,
    min_delta: float = 0.5,
    anchor_tolerance: float = 2.0,
) -> tuple[bool, float | None, float | None]:
    """Filter an error pair. Lower score = chosen (more error-sensitive).

    Args:
        score_a: Score from model A.
        score_b: Score from model B.
        anchor: Anchor score for this sample (Eq. 5 in paper).
        min_delta: Minimum score difference to form a pair.
        anchor_tolerance: Maximum distance of rejected score below anchor.

    Returns:
        (is_valid, score_chosen, score_rejected)
        is_valid is False if pair should be discarded.
    """
    if score_a == score_b:
        return False, None, None

    if score_a < score_b:
        s_chosen, s_rejected = score_a, score_b
    else:
        s_chosen, s_rejected = score_b, score_a

    delta = s_rejected - s_chosen
    if delta < min_delta:
        return False, None, None

    if s_rejected < anchor - anchor_tolerance:
        return False, None, None

    return True, s_chosen, s_rejected
