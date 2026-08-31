"""Accuracy metric — AUC of per-rate combined correctness.

Accuracy pools all binary decisions (detected for τ⁻, stable for τ⁺)
at each rate ρ and computes the trapezoidal AUC of the resulting curve.
"""

from __future__ import annotations

from assay.metrics.validity import _trapz_auc


def compute_accuracy(
    validity_per_rate: dict,
    stability_per_rate: dict,
) -> float | None:
    """Accuracy = AUC of combined correctness rate over ρ.

    At each rate ρ present in both dicts:
        C(ρ) = (n_correct_err + n_correct_repr) / (n_matched_err + n_matched_repr)

    Returns None if fewer than 2 shared rates exist.
    """
    common_rates = sorted(
        set(validity_per_rate.keys()) & set(stability_per_rate.keys()),
        key=float,
    )
    if len(common_rates) < 2:
        return None

    x = [float(r) for r in common_rates]
    y = []
    for r in common_rates:
        v = validity_per_rate[r]
        s = stability_per_rate[r]
        n_correct = v["n_correct"] + s["n_correct"]
        n_total = v["n_matched"] + s["n_matched"]
        y.append(n_correct / n_total if n_total > 0 else 0.0)

    return _trapz_auc(x, y)
