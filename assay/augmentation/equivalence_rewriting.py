"""Equivalence Rewriting operators (ŷ): Formula, Name, IUPAC, Cross-lingual.

Each class substitutes semantically equivalent representations of chemical entities.
"""

from __future__ import annotations

import os
import random
import re

from litellm import completion

from assay.augmentation.base import BasePerturbation, Change, PerturbationResult
from assay.augmentation.llm_convert import LLMChemicalConverter
from assay.augmentation.pubchem import PubChemClient
from assay.augmentation.rate_controller import count_unique_tokens, verify_token_rate
from assay.ner.chemical_ner import LLMChemicalNER

_ner: LLMChemicalNER | None = None
_pubchem: PubChemClient | None = None
_converter: LLMChemicalConverter | None = None


def _get_ner() -> LLMChemicalNER:
    global _ner
    if _ner is None:
        _ner = LLMChemicalNER()
    return _ner


def _get_pubchem() -> PubChemClient:
    global _pubchem
    if _pubchem is None:
        _pubchem = PubChemClient()
    return _pubchem


def _get_converter() -> LLMChemicalConverter:
    global _converter
    if _converter is None:
        _converter = LLMChemicalConverter()
    return _converter


class LLMRepresentationalPerturbation(BasePerturbation):
    """Representational perturbation using LLM-based NER + PubChem lookups."""

    name = "all_equivalence_rewriting"
    category = "representational"

    @staticmethod
    def _resolve_abbreviations(text: str, entities: list[dict]) -> dict[str, str]:
        abbrev_map: dict[str, str] = {}
        abbrevs = [e["text"] for e in entities if e["type"] == "abbreviation"]

        for abbr in abbrevs:
            esc = re.escape(abbr)
            match = re.search(r'([^\n]{3,80}?)\s*\(' + esc + r'\)', text)
            if match:
                full_name = match.group(1).strip().lstrip("- *")
                if len(full_name) > len(abbr) and re.search(r'[a-zA-Z]{2,}', full_name):
                    abbrev_map[abbr.lower()] = full_name

        return abbrev_map

    def _pubchem_lookup(
        self, ent_text: str, ent_type: str, abbrev_map: dict[str, str],
        pubchem: PubChemClient,
    ) -> tuple[str | None, str | None, str | None]:
        lookup_name = ent_text
        resolved_full = abbrev_map.get(ent_text.lower()) if ent_type == "abbreviation" else None

        if resolved_full:
            props_full = pubchem.get_properties(resolved_full)
            if props_full:
                lookup_name = resolved_full
            else:
                if re.search(r'\bpoly', resolved_full, re.IGNORECASE):
                    return None, None, None

        props = pubchem.get_properties(lookup_name)
        formula = props.get("MolecularFormula") if props else None
        iupac = props.get("IUPACName") if props else None
        common_name = pubchem.get_common_name(lookup_name)
        return formula, iupac, common_name

    def _build_all_candidates(self, text: str) -> list[dict]:
        ner = _get_ner()
        pubchem = _get_pubchem()

        entities = ner.detect(text)
        if not entities:
            return []

        abbrev_map = self._resolve_abbreviations(text, entities)

        seen: dict[str, dict] = {}
        for ent in entities:
            key = ent["text"].lower()
            if key not in seen:
                seen[key] = ent

        candidates = []
        for ent in seen.values():
            ent_text = ent["text"]
            ent_type = ent["type"]
            start = ent["start"]
            end = ent["end"]

            formula, iupac, common_name = self._pubchem_lookup(
                ent_text, ent_type, abbrev_map, pubchem,
            )

            lookup_source = "pubchem"
            if formula is None and iupac is None and common_name is None:
                converter = _get_converter()
                resolved = abbrev_map.get(ent_text.lower()) if ent_type == "abbreviation" else None
                llm_result = converter.convert(ent_text, ent_type, full_name=resolved)
                formula = llm_result.get("formula")
                iupac = llm_result.get("iupac")
                common_name = llm_result.get("common_name")
                if any(v is not None for v in (formula, iupac, common_name)):
                    lookup_source = "llm"

            if ent_type in ("compound_name", "abbreviation", "iupac_name"):
                if formula and formula.lower() != ent_text.lower():
                    candidates.append({
                        "original": ent_text,
                        "replacement": formula,
                        "start": start,
                        "end": end,
                        "transform_type": "to_formula",
                        "entity_type": ent_type,
                        "lookup_source": lookup_source,
                    })

            if ent_type in ("formula", "abbreviation"):
                if common_name and len(common_name) <= 60 and common_name.lower() != ent_text.lower():
                    candidates.append({
                        "original": ent_text,
                        "replacement": common_name,
                        "start": start,
                        "end": end,
                        "transform_type": "to_name",
                        "entity_type": ent_type,
                        "lookup_source": lookup_source,
                    })

            if iupac and len(iupac) <= 80 and iupac.lower() != ent_text.lower():
                candidates.append({
                    "original": ent_text,
                    "replacement": iupac,
                    "start": start,
                    "end": end,
                    "transform_type": "to_iupac",
                    "entity_type": ent_type,
                    "lookup_source": lookup_source,
                })

        return candidates

    def detect_targets(self, text: str) -> list[dict]:
        candidates = self._build_all_candidates(text)
        seen: set[str] = set()
        targets = []
        for c in candidates:
            key = c["original"].lower()
            if key not in seen:
                seen.add(key)
                targets.append(c)
        return targets

    def apply(
        self,
        text: str,
        rate: float = 1.0,
        max_changes: int | None = None,
        seed: int = 42,
    ) -> PerturbationResult:
        candidates = self._build_all_candidates(text)
        if not candidates:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                changes=[],
                perturbation_type=self.name,
                category=self.category,
                rate=rate,
                seed=seed,
            )

        by_entity: dict[str, list[dict]] = {}
        for c in candidates:
            key = c["original"].lower()
            if key not in by_entity:
                by_entity[key] = []
            by_entity[key].append(c)

        rng = random.Random(seed)
        entity_keys = list(by_entity.keys())
        rng.shuffle(entity_keys)

        # Build full priority list once (rate-independent)
        priority_list: list[dict] = []
        for key in entity_keys:
            cand = rng.choice(by_entity[key])
            priority_list.append(cand)

        # Rate-dependent: select prefix by token budget
        n_unique_tokens = count_unique_tokens(text)
        target_changed_tokens = round(n_unique_tokens * rate)
        if max_changes is not None:
            target_changed_tokens = min(target_changed_tokens, max_changes)

        def _select_prefix(budget: int) -> list[dict]:
            selected: list[dict] = []
            accumulated = 0
            for cand in priority_list:
                if accumulated >= budget:
                    break
                selected.append(cand)
                accumulated += count_unique_tokens(cand["original"])
            return selected

        best_result = None
        best_diff = float("inf")

        for _attempt in range(3):
            selected = _select_prefix(target_changed_tokens)

            if not selected:
                break

            perturbed, changes = self._apply_replacements(text, selected, preserve_case=False)

            ok, actual_rate, diff = verify_token_rate(text, perturbed, rate)

            if best_result is None or abs(diff) < best_diff:
                best_result = (perturbed, changes, actual_rate)
                best_diff = abs(diff)

            if ok:
                break

            target_changed_tokens = max(1, target_changed_tokens - diff)

        if best_result is None:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                changes=[],
                perturbation_type=self.name,
                category=self.category,
                rate=rate,
                seed=seed,
            )

        perturbed, changes, actual_rate = best_result
        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            changes=changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )


# ── Directional subclasses ──────────────────────────────────────────────────

class _LLMDirectional(LLMRepresentationalPerturbation):
    _transform_filter: str = ""

    def _build_all_candidates(self, text: str) -> list[dict]:
        candidates = super()._build_all_candidates(text)
        return [c for c in candidates if c["transform_type"] == self._transform_filter]


class LLMToFormula(_LLMDirectional):
    name = "llm_to_formula"
    _transform_filter = "to_formula"


class LLMToName(_LLMDirectional):
    name = "llm_to_name"
    _transform_filter = "to_name"


class LLMToIUPAC(_LLMDirectional):
    name = "llm_to_iupac"
    _transform_filter = "to_iupac"


# ── Cross-lingual ───────────────────────────────────────────────────────────

_LANGUAGE_PROMPTS = {
    "korean": "한국어",
    "japanese": "日本語",
    "chinese": "中文",
    "german": "Deutsch",
    "french": "français",
    "spanish": "español",
    "portuguese": "português",
    "russian": "русский",
    "arabic": "العربية",
    "hindi": "हिन्दी",
}


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: export OPENROUTER_API_KEY=your_key"
        )
    return key


def _translate(text: str, target_language: str, model: str = "openai/gpt-4.1-mini") -> str:
    os.environ["OPENROUTER_API_KEY"] = _get_api_key()

    lang_name = _LANGUAGE_PROMPTS.get(target_language, target_language)
    system_prompt = (
        f"You are a materials science translator. Translate the following synthesis recipe "
        f"into {lang_name}. Preserve all chemical formulas, numbers, units, and "
        f"measurement values exactly as they are (do not translate them). "
        f"Translate only the natural language parts (descriptions, instructions, headers). "
        f"Maintain the original formatting (markdown, bullet points, etc.)."
    )

    response = completion(
        model=f"openrouter/{model}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


class CrossLingual(BasePerturbation):
    name = "cross_lingual"
    category = "representational"

    def detect_targets(self, text: str) -> list[dict]:
        return [{"original": text, "replacement": "", "start": 0, "end": len(text)}]

    def apply(self, text: str, rate: float = 1.0, max_changes: int | None = None,
              seed: int = 42, target_language: str = "korean",
              model: str = "openai/gpt-4.1-mini") -> PerturbationResult:
        translated = _translate(text, target_language, model=model)

        changes = [Change(
            original="[full text]",
            replacement=f"[translated to {target_language}]",
            start=0,
            end=len(text),
            category=self.category,
            metadata={"target_language": target_language, "model": model},
        )]

        return PerturbationResult(
            original_text=text,
            perturbed_text=translated,
            changes=changes,
            perturbation_type=self.name,
            category=self.category,
            rate=1.0,
            seed=seed,
        )
