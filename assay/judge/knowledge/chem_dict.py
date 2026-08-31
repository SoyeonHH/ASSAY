"""Chemical dictionary construction for the ChemDict knowledge augmentation baseline.

Entities in the candidate recipe are identified with both NER prompts, then
resolved against the swap dictionaries and the PubChem / LLM conversion caches.
The result is a four-section reference block (compounds, elements, equipment,
actions) prepended to the judge prompt at inference time.
"""

from __future__ import annotations

import json

from assay.domain import get_data_dir
from assay.ner.chemical_ner import LLMChemicalNER
from assay.ner.error_ner import get_error_ner


class ChemDictBuilder:
    """Build a chemical reference dictionary from a candidate recipe."""

    def __init__(self):
        self._loaded = False

    def _load_json(self, name: str, default):
        path = get_data_dir() / name
        if not path.exists():
            return default
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _ensure_loaded(self):
        if self._loaded:
            return

        periodic_table = self._load_json("periodic_table.json", {}).get("groups", {})
        self._equipment_rules = self._load_json("equipment_rules.json", {}).get("rules", {})
        action_pairs = self._load_json("action_antonyms.json", {}).get("pairs", [])

        # PubChem and LLM conversion caches are written by the Equivalence
        # Rewriting operators; ChemDict reads them without issuing new lookups.
        self._pubchem_cache = self._load_json("pubchem_cache.json", {})
        self._llm_convert_cache = self._load_json("llm_convert_cache.json", {})

        self._chem_ner = LLMChemicalNER()
        self._error_ner = get_error_ner()

        self._element_to_group: dict[str, tuple[str, list[str]]] = {}
        for group_name, group_info in periodic_table.items():
            elements = group_info.get("elements", [])
            desc = group_info.get("description", group_name)
            for el in elements:
                self._element_to_group[el] = (desc, elements)

        self._action_lookup: dict[str, str] = {}
        for pair in action_pairs:
            a, b = pair["a"], pair["b"]
            for form in ("base", "past", "gerund"):
                if a.get(form):
                    self._action_lookup[a[form].lower()] = b.get("base", "")
                if b.get(form):
                    self._action_lookup[b[form].lower()] = a.get("base", "")

        self._loaded = True

    def _lookup_compound(self, entity_text: str) -> dict | None:
        text_lower = entity_text.lower()

        key = f"props:{text_lower}"
        if key in self._pubchem_cache:
            return self._pubchem_cache[key]

        # LLM conversion cache keys are "<entity>|<type>|<full_name>"
        for cache_key, val in self._llm_convert_cache.items():
            if cache_key.split("|")[0] == text_lower:
                return val

        return None

    @staticmethod
    def _format_compound_entry(entity_text: str, info: dict) -> str:
        parts = [f"- {entity_text}:"]
        formula = info.get("MolecularFormula") or info.get("formula")
        iupac = info.get("IUPACName") or info.get("iupac")
        common = info.get("common_name")
        if formula:
            parts.append(f"Formula={formula}")
        if iupac:
            parts.append(f"IUPAC={iupac}")
        if common:
            parts.append(f"Common Name={common}")
        return " ".join(parts) if len(parts) > 1 else ""

    def build_context(self, prediction_text: str, max_chars: int = 2000) -> str:
        """Return the reference dictionary for a candidate recipe, or "" if empty."""
        self._ensure_loaded()

        chem_entities = self._chem_ner.detect(prediction_text)
        error_entities = self._error_ner.detect(prediction_text)

        compounds_section = []
        elements_section = []
        equipment_section = []
        actions_section = []

        seen_compounds = set()
        for ent in chem_entities:
            text = ent["text"]
            if text in seen_compounds:
                continue
            seen_compounds.add(text)
            info = self._lookup_compound(text)
            if info:
                line = self._format_compound_entry(text, info)
                if line:
                    compounds_section.append(line)

        seen_elements = set()
        seen_equipment = set()
        seen_actions = set()

        for ent in error_entities:
            etype = ent["type"]
            text = ent["text"]

            if etype == "element" and text not in seen_elements:
                seen_elements.add(text)
                if text in self._element_to_group:
                    desc, group_elements = self._element_to_group[text]
                    others = [e for e in group_elements if e != text][:4]
                    elements_section.append(
                        f"- {text}: {desc} ({', '.join([text] + others)})"
                    )

            elif etype == "equipment" and text.lower() not in seen_equipment:
                seen_equipment.add(text.lower())
                text_lower = text.lower()
                alt = self._equipment_rules.get(text_lower)
                if not alt:
                    for rule_name, rule_alt in self._equipment_rules.items():
                        if text_lower in rule_name or rule_name in text_lower:
                            alt = rule_alt
                            break
                if alt:
                    equipment_section.append(
                        f"- {text}: Equipment (alternative: {alt})"
                    )

            elif etype == "action_verb" and text.lower() not in seen_actions:
                seen_actions.add(text.lower())
                antonym = self._action_lookup.get(text.lower())
                if antonym:
                    actions_section.append(
                        f"- {text}: Synthesis action (antonym: {antonym})"
                    )

        parts = ["Chemical Reference Dictionary:"]
        for header, section in [
            ("Compounds:", compounds_section),
            ("Elements:", elements_section),
            ("Equipment:", equipment_section),
            ("Actions:", actions_section),
        ]:
            if section:
                parts.append(header)
                parts.extend(section)

        if len(parts) == 1:
            return ""

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars].rsplit("\n", 1)[0]
        return result
