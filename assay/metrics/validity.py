"""Validity metric — AUC of per-rate detection curve.

Validity measures false-positive detection: for each Error Injection output ỹ,
the score is *detected* when J(x,y*) - J(x,ỹ) > τ.
"""

from __future__ import annotations

import statistics

TAU = 0.5
DEFAULT_RATES = [0.01, 0.02, 0.05, 0.10, 0.15]


def _trapz_auc(x: list[float], y: list[float]) -> float | None:
    """Trapezoidal AUC, normalized to [0, 1] by dividing by x-range."""
    assert len(x) == len(y)
    if len(x) < 2:
        return None
    area = sum(0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1]) for i in range(len(x) - 1))
    return round(area / (x[-1] - x[0]), 4)


def compute_validity(
    anchors: list[float | None],
    error_rate_scores: dict[float, list[float | None]],
    tau: float = TAU,
) -> tuple[float | None, dict]:
    """Compute Validity = AUC of per-rate detection curve.

    Args:
        anchors: Per-sample anchor scores (model's own ref-free score on GT).
        error_rate_scores: {rate: [scores]} for Error Injection outputs.
        tau: Detection threshold τ (default 0.5).

    Returns:
        (validity_auc, per_rate_details)
    """
    per_rate = {}
    for rate in sorted(error_rate_scores):
        rate_scores = error_rate_scores[rate]
        valid = [
            (a, s) for a, s in zip(anchors, rate_scores)
            if a is not None and s is not None
        ]
        n_parsed = sum(1 for s in rate_scores if s is not None)

        if valid:
            n_det = sum(1 for a, s in valid if a - s > tau)
            per_rate[str(rate)] = {
                "mean_score": round(statistics.mean(s for _, s in valid), 4),
                "mean_anchor": round(statistics.mean(a for a, _ in valid), 4),
                "mean_drop": round(statistics.mean(a - s for a, s in valid), 4),
                "detection_rate": round(n_det / len(valid), 4),
                "n_correct": n_det,
                "n_matched": len(valid),
                "n_parsed": n_parsed,
                "n_total": len(rate_scores),
            }

    if len(per_rate) < 2:
        return None, per_rate

    rates_sorted = sorted(per_rate.keys(), key=float)
    x = [float(r) for r in rates_sorted]
    y = [per_rate[r]["detection_rate"] for r in rates_sorted]
    validity = _trapz_auc(x, y)
    return validity, per_rate
