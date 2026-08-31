"""Combined error injection — apply all 4 error types in a single pass."""

from __future__ import annotations

import random
import re

from assay.augmentation.base import BasePerturbation, Change, PerturbationResult
from assay.augmentation.rate_controller import count_unique_tokens, verify_token_rate
from assay.augmentation.error_injection import (
    ElementSubstitution,
    NumericalPerturbation,
    EquipmentSubstitution,
    ActionAntonym,
    MULTIPLIERS,
    format_number,
    _resolve_replacement_chains,
)

_ERROR_TYPES = {
    "element_substitution": ElementSubstitution,
    "numerical_perturbation": NumericalPerturbation,
    "equipment_substitution": EquipmentSubstitution,
    "action_antonym": ActionAntonym,
}


class CombinedErrorInjection(BasePerturbation):
    """Apply all 4 error injection types simultaneously in one pass.

    Pools targets from all constituent types, deduplicates overlapping targets,
    and distributes changes evenly across types based on the requested rate.
    """

    name = "all_error_injection"
    category = "error"

    def __init__(self):
        self._instances = {name: cls() for name, cls in _ERROR_TYPES.items()}

    def _pool_targets(self, text: str, type_order: list[str]) -> list[dict]:
        all_targets: list[dict] = []
        seen_originals: set[str] = set()
        seen_positions: set[int] = set()

        for tname in type_order:
            perturber = self._instances[tname]
            targets = perturber.detect_targets(text)

            for t in targets:
                t["_perturb_type"] = tname
                is_positional = tname == "numerical_perturbation"

                pos_range = range(t["start"], t["end"])
                if any(p in seen_positions for p in pos_range):
                    continue

                if not is_positional:
                    orig_key = t["original"].lower()
                    if orig_key in seen_originals:
                        continue
                    seen_originals.add(orig_key)
                    for gr in t.get("group_replacements", []):
                        seen_originals.add(gr["original"].lower())

                seen_positions.update(pos_range)
                all_targets.append(t)

        return all_targets

    def detect_targets(self, text: str) -> list[dict]:
        return self._pool_targets(text, list(self._instances.keys()))

    def _prepare_replacements(self, targets: list[dict], rng: random.Random) -> None:
        for t in targets:
            tname = t.get("_perturb_type", "")

            if tname == "element_substitution" and "alternatives" in t:
                t["replacement"] = rng.choice(t["alternatives"])

            elif tname == "numerical_perturbation" and "unit_type" in t:
                multiplier = rng.choice(
                    MULTIPLIERS.get(t["unit_type"], [2.0, 0.5])
                )
                new_value = t["value"] * multiplier
                if t["unit_type"] == "percentage" and new_value > 100:
                    new_value = 99.9
                if new_value < 0:
                    new_value = abs(new_value)
                new_number_str = format_number(new_value)
                spacing_match = re.match(r'[\d,.]+(\s*)', t["original"])
                spacing = spacing_match.group(1) if spacing_match else ""
                t["replacement"] = new_number_str + spacing + t["unit"]
                t["multiplier"] = multiplier

    def _build_priority_list(
        self,
        all_targets: list[dict],
        rng: random.Random,
    ) -> list[dict]:
        """Build a fixed priority list via type-balanced round-robin interleaving.

        The list order is determined solely by the seed (via rng), so taking
        a prefix of length n₁ is always a subset of a prefix of length n₂
        when n₁ < n₂.  This guarantees the nesting property across rates.
        """
        by_type: dict[str, list[dict]] = {}
        for t in all_targets:
            by_type.setdefault(t["_perturb_type"], []).append(t)

        type_keys = sorted(by_type.keys())
        rng_order = random.Random(rng.randint(0, 2**32))
        rng_order.shuffle(type_keys)
        for key in type_keys:
            rng_order.shuffle(by_type[key])

        priority: list[dict] = []
        max_len = max(len(v) for v in by_type.values())
        for i in range(max_len):
            for key in type_keys:
                if i < len(by_type[key]):
                    priority.append(by_type[key][i])
        return priority

    def _apply_selected(
        self, text: str, selected: list[dict]
    ) -> tuple[str, list[Change]]:
        positional = [
            t for t in selected if t.get("_perturb_type") == "numerical_perturbation"
        ]
        wordboundary = [
            t for t in selected if t.get("_perturb_type") != "numerical_perturbation"
        ]

        equip_targets = [t for t in wordboundary
                         if t.get("_perturb_type") == "equipment_substitution"]
        if equip_targets:
            _resolve_replacement_chains(equip_targets)

        group_targets: list[dict] = []
        for t in wordboundary:
            for gr in t.pop("group_replacements", []):
                group_targets.append(gr)

        for t in selected:
            t["source_type"] = t.pop("_perturb_type", "unknown")

        result = text
        all_changes: list[Change] = []

        if positional:
            result, changes = self._apply_positional_replacements(result, positional)
            all_changes.extend(changes)

        if wordboundary:
            preserve_case = self.category != "representational"
            result, changes = self._apply_replacements(result, wordboundary, preserve_case=preserve_case)
            all_changes.extend(changes)

        if group_targets:
            result, changes = self._apply_replacements(result, group_targets, preserve_case=False)
            all_changes.extend(changes)

        return result, all_changes

    def apply(
        self,
        text: str,
        rate: float = 1.0,
        max_changes: int | None = None,
        seed: int = 42,
    ) -> PerturbationResult:
        rng = random.Random(seed)

        available_types = list(self._instances.keys())
        rng.shuffle(available_types)

        all_targets = self._pool_targets(text, available_types)

        if not all_targets:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                changes=[],
                perturbation_type=self.name,
                category=self.category,
                rate=rate,
                seed=seed,
            )

        self._prepare_replacements(all_targets, rng)

        priority_list = self._build_priority_list(all_targets, rng)

        n_unique = count_unique_tokens(text)
        n_total = max(1, round(n_unique * rate))
        if max_changes is not None:
            n_total = min(n_total, max_changes)
        n_total = min(n_total, len(priority_list))

        max_retries = 3
        best_result = None
        best_changes = None
        best_diff = float("inf")

        for _attempt in range(max_retries):
            selected = [{**t} for t in priority_list[:n_total]]
            result_text, changes = self._apply_selected(text, selected)

            is_ok, _actual_rate, diff = verify_token_rate(
                text, result_text, rate
            )

            if is_ok or abs(diff) < abs(best_diff):
                best_result = result_text
                best_changes = changes
                best_diff = diff

            if is_ok:
                break

            n_total = max(1, n_total - diff)
            n_total = min(n_total, len(priority_list))

        return PerturbationResult(
            original_text=text,
            perturbed_text=best_result,
            changes=best_changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )
