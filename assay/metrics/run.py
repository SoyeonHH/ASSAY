"""CLI entry point for computing ASSAY metrics.

Usage:
    python -m assay.metrics.run --model gpt-4o \
        --gt_dir results/groundtruth \
        --error_dir results/error \
        --repr_dir results/repr

    python -m assay.metrics.run --model all
"""

from __future__ import annotations

import json
import os

import fire

from assay.judge.score_parser import extract_overall_score
from assay.metrics.validity import compute_validity
from assay.metrics.stability import compute_stability
from assay.metrics.accuracy import compute_accuracy

SUFFIX_STANDARD = ".jsonl"
ERROR_RATES = [0.01, 0.02, 0.05, 0.10, 0.15]
REPR_RATES = [0.01, 0.02, 0.05, 0.10, 0.15]


def _load_scores(filepath: str) -> tuple[list[float | None], int]:
    """Load JSONL → list of float|None (one per line)."""
    scores = []
    failures = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                scores.append(None)
                failures += 1
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                scores.append(None)
                failures += 1
                continue
            score = extract_overall_score(r.get("judge_result", ""))
            if score is None:
                failures += 1
            scores.append(score)
    return scores, failures


def _discover_models(gt_dir: str) -> dict[str, str]:
    """Scan GT directory and return {model_name: suffix}."""
    models = {}
    for fname in sorted(os.listdir(gt_dir)):
        if os.path.isdir(os.path.join(gt_dir, fname)):
            continue
        if fname.endswith(SUFFIX_STANDARD):
            model_name = fname[: -len(SUFFIX_STANDARD)]
            models[model_name] = SUFFIX_STANDARD
    return models


def _find_and_load(base_dir: str, subdir: str, model_name: str, suffix: str):
    path = os.path.join(base_dir, subdir, f"{model_name}{suffix}")
    if not os.path.exists(path):
        return None
    return _load_scores(path)


def main(
    model: str = "all",
    gt_dir: str = "results/groundtruth",
    error_dir: str = "results/error",
    repr_dir: str = "results/repr",
    output_dir: str = "results/metrics",
    dataset_prefix: str = "test_high_impact",
):
    """Compute ASSAY metrics for judge evaluation.

    Args:
        model: Model name (e.g. "gpt-4o") or "all" for batch processing.
        gt_dir: Directory containing ground-truth judge results.
        error_dir: Directory containing error-augmented judge results.
        repr_dir: Directory containing equivalence-rewritten judge results.
        output_dir: Directory for output JSON files.
        dataset_prefix: Dataset name prefix for subdirectory naming.
    """
    all_models = _discover_models(gt_dir)

    if model == "all":
        targets = all_models
    elif model in all_models:
        targets = {model: all_models[model]}
    else:
        print(f"Model '{model}' not found. Available:")
        for m in sorted(all_models):
            print(f"  {m}")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for model_name, suffix in sorted(targets.items()):
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        # GT anchor scores
        gt_path = os.path.join(gt_dir, f"{model_name}{suffix}")
        anchors, gt_fail = _load_scores(gt_path)
        n_samples = len(anchors)
        n_anchors = sum(1 for a in anchors if a is not None)
        print(f"  GT anchors: {n_anchors}/{n_samples} parsed")

        # Error augmentation scores (multi-rate)
        error_rate_scores = {}
        for rate in ERROR_RATES:
            rate_str = f"{int(rate * 100)}pct"
            subdir = f"{dataset_prefix}_all_error_injection_{rate_str}"
            result = _find_and_load(error_dir, subdir, model_name, suffix)
            if result:
                scores, fail = result
                error_rate_scores[rate] = scores
                n_p = sum(1 for s in scores if s is not None)
                print(f"  Error {rate_str}: {n_p}/{len(scores)} parsed")

        # Repr augmentation scores (multi-rate)
        repr_rate_scores = {}
        for rate in REPR_RATES:
            rate_str = f"{int(rate * 100)}pct"
            subdir = f"{dataset_prefix}_all_equivalence_rewriting_{rate_str}"
            result = _find_and_load(repr_dir, subdir, model_name, suffix)
            if result:
                scores, fail = result
                repr_rate_scores[rate] = scores
                n_p = sum(1 for s in scores if s is not None)
                print(f"  Repr {rate_str}: {n_p}/{len(scores)} parsed")

        # Compute metrics
        validity, err_per_rate = compute_validity(anchors, error_rate_scores)
        stability_val, repr_per_rate = compute_stability(anchors, repr_rate_scores)
        accuracy = compute_accuracy(err_per_rate, repr_per_rate)

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n  Validity (↑):  {_fmt(validity)}")
        print(f"  Stability (↑): {_fmt(stability_val)}")
        print(f"  Accuracy (↑):  {_fmt(accuracy)}")

        # Save
        result_obj = {
            "model": model_name,
            "n_samples": n_samples,
            "n_anchors_parsed": n_anchors,
            "validity": validity,
            "stability": stability_val,
            "accuracy": accuracy,
            "validity_per_rate": err_per_rate,
            "stability_per_rate": repr_per_rate,
        }

        out_path = os.path.join(output_dir, f"{model_name}_metrics.json")
        with open(out_path, "w") as f:
            json.dump(result_obj, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

        all_results[model_name] = result_obj

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"{'Model':<35s} {'Validity':>10s} {'Stability':>10s} {'Accuracy':>10s}")
        print(f"{'-' * 70}")
        for name in sorted(all_results):
            r = all_results[name]
            v = f"{r['validity']:.4f}" if r["validity"] is not None else "N/A"
            s = f"{r['stability']:.4f}" if r["stability"] is not None else "N/A"
            a = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
            print(f"{name:<35s} {v:>10s} {s:>10s} {a:>10s}")


if __name__ == "__main__":
    fire.Fire(main)
