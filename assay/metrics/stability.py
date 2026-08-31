"""Stability metric — AUC of per-rate stability curve.

Stability measures false-negative avoidance: for each Equivalence Rewriting output ŷ,
the score is *stable* when |J(x,y*) - J(x,ŷ)| ≤ τ.
"""

from __future__ import annotations

import statistics

from assay.metrics.validity import _trapz_auc, TAU


def compute_stability(
    anchors: list[float | None],
    repr_rate_scores: dict[float, list[float | None]],
    tau: float = TAU,
) -> tuple[float | None, dict]:
    """Compute Stability = AUC of per-rate stability curve.

    Args:
        anchors: Per-sample anchor scores (model's own ref-free score on GT).
        repr_rate_scores: {rate: [scores]} for Equivalence Rewriting outputs.
        tau: Stability threshold τ (default 0.5).

    Returns:
        (stability_auc, per_rate_details)
    """
    per_rate = {}
    for rate in sorted(repr_rate_scores):
        rate_scores = repr_rate_scores[rate]
        valid = [(a, s) for a, s in zip(anchors, rate_scores)
                 if a is not None and s is not None]
        n_parsed = sum(1 for s in rate_scores if s is not None)

        if valid:
            n_inv = sum(1 for a, s in valid if abs(s - a) <= tau)
            per_rate[str(rate)] = {
                "mean_score": round(statistics.mean(s for _, s in valid), 4),
                "mean_anchor": round(statistics.mean(a for a, _ in valid), 4),
                "stability": round(n_inv / len(valid), 4),
                "n_correct": n_inv,
                "n_matched": len(valid),
                "n_parsed": n_parsed,
                "n_total": len(rate_scores),
            }

    if len(per_rate) < 2:
        return None, per_rate

    rates_sorted = sorted(per_rate.keys(), key=float)
    x = [float(r) for r in rates_sorted]
    y = [per_rate[r]["stability"] for r in rates_sorted]
    stability = _trapz_auc(x, y)
    return stability, per_rate
