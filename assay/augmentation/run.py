"""CLI entry point for the augmentation system.

Usage:
    python -m assay.augmentation.run \
        --input_file examples/sample_input.jsonl \
        --type element_substitution \
        --rates '[0.05, 0.10, 0.15]'
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from assay.augmentation.base import PerturbationResult
from assay.augmentation.rate_controller import count_unique_tokens, compute_token_rate

# Available augmentation types (lazy-loaded)
_TYPE_MAP = {
    # Error Injection (ỹ)
    "element_substitution": ("assay.augmentation.error_injection", "ElementSubstitution"),
    "numerical_perturbation": ("assay.augmentation.error_injection", "NumericalPerturbation"),
    "equipment_substitution": ("assay.augmentation.error_injection", "EquipmentSubstitution"),
    "action_antonym": ("assay.augmentation.error_injection", "ActionAntonym"),
    "all_error_injection": ("assay.augmentation.combined", "CombinedErrorInjection"),
    # Equivalence Rewriting (ŷ)
    "all_equivalence_rewriting": ("assay.augmentation.equivalence_rewriting", "LLMRepresentationalPerturbation"),
    "llm_to_formula": ("assay.augmentation.equivalence_rewriting", "LLMToFormula"),
    "llm_to_name": ("assay.augmentation.equivalence_rewriting", "LLMToName"),
    "llm_to_iupac": ("assay.augmentation.equivalence_rewriting", "LLMToIUPAC"),
    "cross_lingual": ("assay.augmentation.equivalence_rewriting", "CrossLingual"),
}


def _get_perturber(type_name: str):
    if type_name not in _TYPE_MAP:
        raise ValueError(
            f"Unknown augmentation type: {type_name}\n"
            f"Available: {', '.join(sorted(_TYPE_MAP))}"
        )
    module_path, class_name = _TYPE_MAP[type_name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def _load_records(input_file: str) -> list[dict]:
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _generate_single(
    input_file: str,
    perturbation_type: str,
    rate: float,
    output_dir: Path,
    max_changes: int | None = None,
    seed: int = 42,
) -> Path:
    perturber = _get_perturber(perturbation_type)
    records = _load_records(input_file)

    rate_pct = int(rate * 100)
    stem = Path(input_file).stem
    out_path = output_dir / f"{stem}_{perturbation_type}_{rate_pct}pct.jsonl"

    output_records = []
    for rec in records:
        recipe = rec.get("recipe", "")
        result: PerturbationResult = perturber.apply(
            recipe, rate=rate, max_changes=max_changes, seed=seed
        )

        out_rec = {
            "sample_id": rec.get("sample_id") or rec.get("id"),
            "id": rec.get("id"),
            "contribution": rec.get("contribution"),
            "recipe": recipe,
            "prediction": result.perturbed_text,
            "Material_Name": rec.get("Material_Name"),
            "process": rec.get("process"),
            "domain": rec.get("domain"),
            "perturbation_meta": {
                "type": result.perturbation_type,
                "category": result.category,
                "rate_requested": rate,
                "rate_actual": round(compute_token_rate(recipe, result.perturbed_text), 6),
                "unique_tokens": count_unique_tokens(recipe),
                "num_changes": len(result.changes),
                "changes": [
                    {
                        "original": c.original,
                        "replacement": c.replacement,
                        "position": c.start,
                        **c.metadata,
                    }
                    for c in result.changes
                ],
                "seed": seed,
            },
        }
        output_records.append(out_rec)

    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Created {out_path.name} ({len(output_records)} records, "
          f"{perturbation_type} @ {rate_pct}%)")
    return out_path


def generate(
    input_file: str = "examples/sample_input.jsonl",
    type: str = "element_substitution",
    rate: float | None = None,
    rates: list[float] | str | None = None,
    max_changes: int | None = None,
    seed: int = 42,
    output_dir: str = "outputs/augmentation",
    data_dir: str | None = None,
):
    """Generate augmented JSONL files.

    Args:
        input_file: Path to input JSONL
        type: Augmentation type name (see --help for available types)
        rate: Single augmentation rate (0.0-1.0)
        rates: List of rates, e.g. '[0.05, 0.10, 0.15]'
        max_changes: Maximum number of changes per sample
        seed: Random seed
        output_dir: Output directory
        data_dir: Swap dictionary directory. Defaults to the materials science
            configuration; pass assay/augmentation/data_bio for biology.
    """
    if data_dir is not None:
        from assay.domain import set_data_dir
        set_data_dir(data_dir)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(rates, str):
        rates = json.loads(rates)

    if rates:
        for r in rates:
            _generate_single(input_file, type, r, out_dir,
                             max_changes=max_changes, seed=seed)
    else:
        effective_rate = rate if rate is not None else 0.1
        _generate_single(input_file, type, effective_rate, out_dir,
                         max_changes=max_changes, seed=seed)


if __name__ == "__main__":
    fire.Fire(generate)
