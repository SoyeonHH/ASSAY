"""Build DPO dataset for reference-free judge debiasing.

Anchor-Consensus pair construction from multi-model judge results.

Usage:
    python -m assay.training.build_dpo_pairs \
        --judge_results_root results \
        --output_dir dpo_data
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from itertools import combinations

import fire
import pandas as pd

from assay.training.filter import filter_repr_pair, filter_error_pair

# Anchor datasets: representational augmentation at 5 rates
ANCHOR_DATASETS = [
    "all_equivalence_rewriting_1pct",
    "all_equivalence_rewriting_2pct",
    "all_equivalence_rewriting_5pct",
    "all_equivalence_rewriting_10pct",
    "all_equivalence_rewriting_15pct",
]

ERROR_DATASETS = [
    "all_error_injection_1pct",
    "all_error_injection_2pct",
    "all_error_injection_5pct",
    "all_error_injection_10pct",
    "all_error_injection_15pct",
    "element_substitution_100pct",
    "numerical_perturbation_100pct",
    "equipment_substitution_100pct",
    "action_antonym_100pct",
]

REPR_DATASETS = [
    *ANCHOR_DATASETS,
    "llm_to_formula_100pct",
    "llm_to_name_100pct",
    "llm_to_iupac_100pct",
]

USER_PROMPT_REF_FREE = """Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}"""


# ── Score Extraction ────────────────────────────────────────────────────────

def _safe_eval_arithmetic(expr: str) -> float | None:
    expr = expr.strip()
    if not re.match(r'^[0-9.+\-*/() ]+$', expr):
        return None
    try:
        result = eval(expr)  # noqa: S307 — guarded by strict whitelist
        return float(result)
    except Exception:
        return None


def extract_score(model_name: str, text: str) -> dict | None:
    """Extract scores from judge output."""
    pattern = r'```json\s*(\{[^`]+\})\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1)

    # Try direct parse
    try:
        scores = json.loads(json_str)
        if "overall_score" in scores:
            overall = scores.get("overall_score")
            if overall is not None:
                val = float(overall)
                if 0.5 <= val <= 5.5:
                    return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Parse individual score keys with regex for arithmetic handling
    score_keys = [
        "materials_appropriateness_score",
        "equipment_appropriateness_score",
        "procedure_completeness_score",
        "procedure_feasibility_score",
        "characterization_appropriateness_score",
        "overall_score",
    ]
    scores = {}
    for key in score_keys:
        key_pattern = rf'"{key}"\s*:\s*(.+?)(?:\s*[,\n}}])'
        key_match = re.search(key_pattern, json_str)
        if key_match:
            val_str = key_match.group(1).strip().rstrip(',')
            try:
                scores[key] = float(val_str)
            except ValueError:
                if '=' in val_str:
                    val_str = val_str.split('=')[-1].strip()
                    try:
                        scores[key] = float(val_str)
                        continue
                    except ValueError:
                        pass
                result = _safe_eval_arithmetic(val_str)
                if result is not None:
                    scores[key] = round(result, 2)

    if "overall_score" in scores:
        overall = scores["overall_score"]
        if 0.5 <= overall <= 5.5:
            return scores
    return None


# ── Data Loading ────────────────────────────────────────────────────────────

def load_judge_results(
    dataset_name: str,
    category: str,
    models: dict[str, str],
    judge_results_root: str,
    dataset_prefix: str,
) -> dict[str, dict[int, dict]]:
    """Load judge results for a dataset across multiple models.

    Mirrors the layout written by assay.judge.evaluate:
    <judge_results_root>/<dataset_prefix>_<dataset_name>/<model file>.
    """
    dataset_dir = os.path.join(judge_results_root, f"{dataset_prefix}_{dataset_name}")

    results = {}
    for model_name, filename in models.items():
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            continue

        model_results = {}
        parse_failures = 0
        total = 0

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total += 1
                sid = record["sample_id"]
                judge_result = record.get("judge_result", "")
                scores = extract_score(model_name, judge_result)

                if scores is None or "overall_score" not in scores:
                    parse_failures += 1
                    continue

                overall = scores["overall_score"]
                try:
                    overall = float(overall)
                except (ValueError, TypeError):
                    parse_failures += 1
                    continue

                model_results[sid] = {
                    "score": overall,
                    "judge_result": judge_result,
                    "record": record,
                }

        results[model_name] = model_results

    return results


# ── Anchor Score Computation ────────────────────────────────────────────────

def compute_anchors(
    judge_results_root: str,
    dataset_prefix: str,
    models: dict[str, str],
) -> dict[int, float]:
    """Compute per-sample anchor scores from Equivalence Rewriting consensus (Eq. 5)."""
    print("Computing anchor scores...")
    sample_scores = defaultdict(list)

    for dataset_name in ANCHOR_DATASETS:
        results = load_judge_results(
            dataset_name, "represent", models,
            judge_results_root, dataset_prefix,
        )
        for model_results in results.values():
            for sid, entry in model_results.items():
                sample_scores[sid].append(entry["score"])

    anchors = {}
    for sid, scores in sample_scores.items():
        anchors[sid] = statistics.median(scores)

    if anchors:
        score_values = list(anchors.values())
        print(
            f"  Anchor stats: {len(anchors)} samples, "
            f"median={statistics.median(score_values):.2f}, "
            f"mean={statistics.mean(score_values):.2f}"
        )
    return anchors


# ── Response Normalization ──────────────────────────────────────────────────

def normalize_response(model_name: str, text: str) -> str:
    """Normalize judge response to 'reasoning → JSON' format."""
    pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if len(matches) > 1:
        # Remove duplicate JSON blocks
        text = text[:matches[1].start()].rstrip()
    return text


# ── Pair Construction ───────────────────────────────────────────────────────

def build_pairs_for_dataset(
    dataset_name: str,
    category: str,
    anchors: dict[int, float],
    judge_results_root: str,
    dataset_prefix: str,
    models: dict[str, str],
    min_delta: float,
    max_per_sample: int,
) -> list[dict]:
    """Build DPO pairs for a single dataset."""
    print(f"  Processing {category}/{dataset_name}...")

    results = load_judge_results(
        dataset_name, category, models,
        judge_results_root, dataset_prefix,
    )
    if not results:
        return []

    model_names = list(results.keys())
    if not model_names:
        return []

    common_sids = set(results[model_names[0]].keys())
    for mname in model_names[1:]:
        common_sids &= set(results[mname].keys())
    common_sids = {sid for sid in common_sids if sid in anchors}

    pairs = []
    for sid in common_sids:
        anchor = anchors[sid]
        sample_pairs = []

        for m_a, m_b in combinations(model_names, 2):
            score_a = results[m_a][sid]["score"]
            score_b = results[m_b][sid]["score"]

            if category == "represent":
                is_valid, s_chosen, s_rejected = filter_repr_pair(
                    score_a, score_b, anchor, min_delta=min_delta,
                )
                if not is_valid:
                    continue
                m_chosen = m_a if score_a >= score_b else m_b
                m_rejected = m_b if score_a >= score_b else m_a
                delta = s_chosen - s_rejected
            else:
                is_valid, s_chosen, s_rejected = filter_error_pair(
                    score_a, score_b, anchor, min_delta=min_delta,
                )
                if not is_valid:
                    continue
                m_chosen = m_a if score_a <= score_b else m_b
                m_rejected = m_b if score_a <= score_b else m_a
                delta = s_rejected - s_chosen

            record = results[m_chosen][sid]["record"]
            perturbation_meta = record.get("perturbation_meta", {})

            sample_pairs.append({
                "sample_id": sid,
                "score_chosen": s_chosen,
                "score_rejected": s_rejected,
                "score_delta": delta,
                "anchor_score": anchor,
                "chosen_model": m_chosen,
                "rejected_model": m_rejected,
                "chosen_response": results[m_chosen][sid]["judge_result"],
                "rejected_response": results[m_rejected][sid]["judge_result"],
                "contribution": record["contribution"],
                "prediction": record["prediction"],
                "Material_Name": record.get("Material_Name", ""),
                "domain": record.get("domain", ""),
                "process": record.get("process", ""),
                "perturbation_type": perturbation_meta.get("type", dataset_name),
                "perturbation_category": category,
                "perturbation_rate": perturbation_meta.get("rate_requested", 1.0),
                "dataset_name": dataset_name,
            })

        sample_pairs.sort(key=lambda x: x["score_delta"], reverse=True)
        pairs.extend(sample_pairs[:max_per_sample])

    print(f"    {len(pairs)} pairs from {len(common_sids)} samples")
    return pairs


# ── DPO Formatting ─────────────────────────────────────────────────────────

def format_dpo_row(pair: dict, system_prompt: str) -> dict:
    chosen_text = normalize_response(pair["chosen_model"], pair["chosen_response"])
    rejected_text = normalize_response(pair["rejected_model"], pair["rejected_response"])

    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": USER_PROMPT_REF_FREE.format(
                objective=pair["contribution"],
                prediction=pair["prediction"],
            ),
        },
    ]

    chosen = [{"role": "assistant", "content": chosen_text}]
    rejected = [{"role": "assistant", "content": rejected_text}]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "score_chosen": pair["score_chosen"],
        "score_rejected": pair["score_rejected"],
        "score_delta": pair["score_delta"],
        "anchor_score": pair["anchor_score"],
        "sample_id": pair["sample_id"],
        "Material_Name": pair["Material_Name"],
        "domain": pair["domain"],
        "process": pair["process"],
        "perturbation_type": pair["perturbation_type"],
        "perturbation_category": pair["perturbation_category"],
        "perturbation_rate": pair["perturbation_rate"],
        "chosen_model": pair["chosen_model"],
        "rejected_model": pair["rejected_model"],
        "dataset_name": pair["dataset_name"],
    }


# ── Dedup, Split, Export ────────────────────────────────────────────────────

def content_hash(row: dict) -> str:
    key = (
        row["prompt"][1]["content"]
        + row["chosen"][0]["content"]
        + row["rejected"][0]["content"]
    )
    return hashlib.sha256(key.encode()).hexdigest()


def dedup_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        h = content_hash(row)
        if h not in seen:
            seen.add(h)
            deduped.append(row)
    return deduped


def split_and_export(
    rows: list[dict],
    output_dir: str,
    val_fraction: float,
    seed: int,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    try:
        from sklearn.model_selection import train_test_split
        categories = [r["perturbation_category"] for r in rows]
        train_rows, val_rows = train_test_split(
            rows, test_size=val_fraction, random_state=seed, stratify=categories,
        )
    except ImportError:
        import random
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - val_fraction))
        train_rows, val_rows = shuffled[:split_idx], shuffled[split_idx:]

    train_df = _rows_to_dataframe(train_rows)
    val_df = _rows_to_dataframe(val_rows)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\n  Exported: {train_path} ({len(train_df)} rows)")
    print(f"  Exported: {val_path} ({len(val_df)} rows)")

    return {"train": len(train_df), "validation": len(val_df)}


def _rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ["prompt", "chosen", "rejected"]:
        if col in df.columns:
            df[col] = df[col].apply(json.dumps, ensure_ascii=False)
    return df


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def main(
    output_dir: str = "dpo_dataset",
    min_score_delta: float = 0.5,
    max_pairs_per_sample: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
    judge_results_root: str = "judge_results",
    dataset_prefix: str = "train_2000",
    prompt_file: str = "assay/judge/prompt.txt",
    models: str = "",
    exclude_datasets: str = "",
    include_only_category: str = "",
):
    """Build DPO dataset from multi-model judge results.

    Args:
        output_dir: Output directory for parquet files.
        min_score_delta: Minimum score difference for pair construction.
        max_pairs_per_sample: Maximum pairs per sample per dataset.
        val_fraction: Fraction of data for validation split.
        seed: Random seed for reproducibility.
        judge_results_root: Root directory of judge results.
        dataset_prefix: Dataset name prefix (e.g., train_2000).
        prompt_file: Path to judge prompt file for DPO prompt construction.
        models: Comma-separated model_name:filename pairs
            (e.g. "Qwen3-8B:Qwen3-8B.jsonl,gpt-4o:gpt-4o.jsonl").
        exclude_datasets: Comma-separated dataset names to exclude.
        include_only_category: Only include "error" or "represent" (empty = both).
    """
    print("=" * 60)
    print("Building DPO Dataset (Anchor-Consensus)")
    print("=" * 60)

    # Parse models from CLI
    if not models:
        print("Error: --models is required. Provide comma-separated model_name:filename pairs.")
        return

    pair_models = {}
    for entry in models.split(","):
        entry = entry.strip()
        if ":" in entry:
            name, fname = entry.split(":", 1)
            pair_models[name.strip()] = fname.strip()
        else:
            pair_models[entry] = f"{entry}.jsonl"

    print(f"  Models: {list(pair_models.keys())}")
    print(f"  min_score_delta: {min_score_delta}")
    print(f"  max_pairs_per_sample: {max_pairs_per_sample}")
    print()

    with open(prompt_file) as f:
        system_prompt = f.read().strip()

    excluded = set(s.strip() for s in exclude_datasets.split(",") if s.strip())

    # Step 1: Compute anchors
    anchors = compute_anchors(judge_results_root, dataset_prefix, pair_models)
    print()

    # Step 2: Build pairs
    all_pairs = []

    if include_only_category in ("", "represent"):
        print("Building representational pairs...")
        for ds in REPR_DATASETS:
            if ds in excluded:
                continue
            pairs = build_pairs_for_dataset(
                ds, "represent", anchors,
                judge_results_root, dataset_prefix, pair_models,
                min_score_delta, max_pairs_per_sample,
            )
            all_pairs.extend(pairs)

    if include_only_category in ("", "error"):
        print("\nBuilding error pairs...")
        for ds in ERROR_DATASETS:
            if ds in excluded:
                continue
            pairs = build_pairs_for_dataset(
                ds, "error", anchors,
                judge_results_root, dataset_prefix, pair_models,
                min_score_delta, max_pairs_per_sample,
            )
            all_pairs.extend(pairs)

    print(f"\nTotal raw pairs: {len(all_pairs)}")

    # Step 3: Format into DPO rows
    dpo_rows = [format_dpo_row(p, system_prompt) for p in all_pairs]

    # Step 4: Dedup
    before_dedup = len(dpo_rows)
    dpo_rows = dedup_rows(dpo_rows)
    print(f"Dedup: {before_dedup} → {len(dpo_rows)}")

    if not dpo_rows:
        print("No pairs generated.")
        return

    # Step 5: Split and export
    split_and_export(dpo_rows, output_dir, val_fraction, seed)
    print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
