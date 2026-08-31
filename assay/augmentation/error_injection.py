"""Error Injection operators (ỹ): Element, Numerical, Equipment, Action.

Each class injects scientifically invalid modifications into synthesis recipes.
"""

from __future__ import annotations

import json
import random
import re

from assay.augmentation.base import BasePerturbation, PerturbationResult
from assay.domain import get_data_dir

# ── Element Substitution ────────────────────────────────────────────────────

_ELEMENT_TO_GROUP: dict[str, str] = {}
_GROUP_ELEMENTS: dict[str, list[str]] = {}
_loaded_elements = False


def _ensure_elements_loaded() -> None:
    global _ELEMENT_TO_GROUP, _GROUP_ELEMENTS, _loaded_elements
    if _loaded_elements:
        return
    with open(get_data_dir() / "periodic_table.json") as f:
        pt_data = json.load(f)
    _ELEMENT_TO_GROUP = {}
    _GROUP_ELEMENTS = {}
    for group_name, group_info in pt_data["groups"].items():
        elements = group_info["elements"]
        _GROUP_ELEMENTS[group_name] = elements
        for elem in elements:
            _ELEMENT_TO_GROUP[elem] = group_name
    _loaded_elements = True


class ElementSubstitution(BasePerturbation):
    name = "element_substitution"
    category = "error"

    def detect_targets(self, text: str) -> list[dict]:
        from assay.ner.error_ner import get_error_ner

        _ensure_elements_loaded()
        entities = get_error_ner().detect(text)
        element_entities = [e for e in entities if e["type"] == "element"]

        targets = []
        seen_elements: set[str] = set()
        for entity in element_entities:
            symbol = entity["text"]
            if symbol not in _ELEMENT_TO_GROUP or symbol in seen_elements:
                continue
            seen_elements.add(symbol)

            group_name = _ELEMENT_TO_GROUP[symbol]
            alternatives = [e for e in _GROUP_ELEMENTS[group_name] if e != symbol]
            if not alternatives:
                continue

            targets.append({
                "original": symbol,
                "replacement": alternatives[0],
                "start": entity["start"],
                "end": entity["end"],
                "element_group": group_name,
                "alternatives": alternatives,
            })

        return targets

    def apply(self, text: str, rate: float = 0.1, max_changes: int | None = None,
              seed: int = 42) -> PerturbationResult:
        rng = random.Random(seed)
        targets = self.detect_targets(text)

        for t in targets:
            t["replacement"] = rng.choice(t["alternatives"])

        selected = self._select_targets(targets, rate, max_changes, seed)
        perturbed, changes = self._apply_replacements(text, selected)

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            changes=changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )


# ── Numerical Perturbation ──────────────────────────────────────────────────

_UNIT_TO_TYPE: dict[str, str] = {
    "°C": "temperature", "°F": "temperature", "K": "temperature",
    "℃": "temperature",
    "hours": "time", "hour": "time", "hrs": "time", "hr": "time",
    "minutes": "time", "minute": "time", "mins": "time", "min": "time",
    "seconds": "time", "second": "time", "secs": "time", "sec": "time",
    "days": "time", "day": "time", "h": "time", "s": "time",
    "mg": "mass", "g": "mass", "kg": "mass", "μg": "mass", "µg": "mass",
    "mL": "volume", "ml": "volume", "μL": "volume", "µL": "volume",
    "L": "volume", "µl": "volume", "μl": "volume",
    "mM": "concentration", "M": "concentration", "mol/L": "concentration",
    "mmol/L": "concentration", "μM": "concentration", "µM": "concentration",
    "mol%": "concentration", "wt%": "concentration", "vol%": "concentration",
    "mg/mL": "concentration", "g/L": "concentration",
    "nm": "length", "μm": "length", "µm": "length", "mm": "length",
    "cm": "length", "m": "length", "Å": "length",
    "rpm": "speed", "sccm": "speed", "mV/s": "speed", "V/s": "speed",
    "mA": "speed", "A": "speed",
    "atm": "pressure", "Pa": "pressure", "kPa": "pressure",
    "MPa": "pressure", "GPa": "pressure", "bar": "pressure",
    "mbar": "pressure", "Torr": "pressure", "psi": "pressure",
    "%": "percentage",
    "W": "power", "kW": "power", "mW": "power",
}

MULTIPLIERS = {
    "temperature": [2.0, 0.5, 1.5],
    "time": [10.0, 0.1, 5.0],
    "mass": [10.0, 0.1, 3.0],
    "volume": [10.0, 0.1, 5.0],
    "concentration": [10.0, 0.1, 5.0],
    "length": [10.0, 0.1, 100.0],
    "speed": [10.0, 0.1, 5.0],
    "pressure": [10.0, 0.1, 5.0],
    "percentage": [2.0, 0.5, 0.1],
    "power": [10.0, 0.1, 5.0],
}

_NUM_UNIT_RE = re.compile(r'^([\d,]+(?:\.\d+)?)\s*(.+)$')


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.4g}"


def _parse_numerical_entity(entity_text: str) -> tuple[float | None, str | None, str | None]:
    m = _NUM_UNIT_RE.match(entity_text.strip())
    if not m:
        return None, None, None

    number_str = m.group(1).replace(',', '')
    unit_str = m.group(2).strip()

    try:
        value = float(number_str)
    except ValueError:
        return None, None, None

    unit_type = _UNIT_TO_TYPE.get(unit_str)
    if unit_type is None:
        for known_unit, ut in _UNIT_TO_TYPE.items():
            if unit_str.lower() == known_unit.lower():
                unit_type = ut
                unit_str = known_unit
                break

    if unit_type is None:
        lower = unit_str.lower()
        if "celsius" in lower or "fahrenheit" in lower:
            unit_type = "temperature"

    return value, unit_str, unit_type


class NumericalPerturbation(BasePerturbation):
    name = "numerical_perturbation"
    category = "error"

    def detect_targets(self, text: str) -> list[dict]:
        from assay.ner.error_ner import get_error_ner

        entities = get_error_ner().detect(text)
        num_entities = [e for e in entities if e["type"] == "numerical_value"]

        targets = []
        occupied: set[int] = set()

        for entity in num_entities:
            value, unit, unit_type = _parse_numerical_entity(entity["text"])
            if value is None or value == 0 or unit_type is None:
                continue

            for match in re.finditer(re.escape(entity["text"]), text):
                pos_range = range(match.start(), match.end())
                if any(p in occupied for p in pos_range):
                    continue
                occupied.update(pos_range)

                targets.append({
                    "original": match.group(0),
                    "replacement": "",
                    "start": match.start(),
                    "end": match.end(),
                    "value": value,
                    "unit": unit,
                    "unit_type": unit_type,
                })

        return targets

    def apply(self, text: str, rate: float = 0.1, max_changes: int | None = None,
              seed: int = 42) -> PerturbationResult:
        rng = random.Random(seed)
        targets = self.detect_targets(text)

        for t in targets:
            unit_type = t["unit_type"]
            multiplier = rng.choice(MULTIPLIERS.get(unit_type, [2.0, 0.5]))
            new_value = t["value"] * multiplier

            if unit_type == "percentage" and new_value > 100:
                new_value = 99.9
            if new_value < 0:
                new_value = abs(new_value)

            new_number_str = format_number(new_value)
            original = t["original"]
            spacing_match = re.match(r'[\d,.]+(\s*)', original)
            spacing = spacing_match.group(1) if spacing_match else ''
            t["replacement"] = new_number_str + spacing + t["unit"]
            t["multiplier"] = multiplier

        selected = self._select_targets(targets, rate, max_changes, seed)
        perturbed, changes = self._apply_positional_replacements(text, selected)

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            changes=changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )


# ── Equipment Substitution ──────────────────────────────────────────────────

_RULES: dict[str, str] = {}
_SORTED_EQUIPMENT: list[str] = []
_RULES_LOWER: dict[str, str] = {}
_FULL_NAMES: dict[str, str] = {}
_FULL_NAMES_LOWER: dict[str, str] = {}
_FULL_NAME_TO_ABBREV: dict[str, str] = {}
_loaded_equipment = False


def _ensure_equipment_loaded() -> None:
    global _RULES, _SORTED_EQUIPMENT, _RULES_LOWER
    global _FULL_NAMES, _FULL_NAMES_LOWER, _FULL_NAME_TO_ABBREV, _loaded_equipment
    if _loaded_equipment:
        return
    with open(get_data_dir() / "equipment_rules.json") as f:
        equipment_data = json.load(f)
    _RULES = equipment_data["rules"]
    _SORTED_EQUIPMENT = sorted(_RULES.keys(), key=len, reverse=True)
    _RULES_LOWER = {k.lower(): v for k, v in _RULES.items()}
    _FULL_NAMES = equipment_data.get("full_names", {})
    _FULL_NAMES_LOWER = {k.lower(): v for k, v in _FULL_NAMES.items()}
    _FULL_NAME_TO_ABBREV = {v.lower(): k for k, v in _FULL_NAMES.items()}
    _loaded_equipment = True


def _lookup_equipment_rule(detected_text: str) -> str | None:
    _ensure_equipment_loaded()
    text_lower = detected_text.lower()
    if text_lower in _RULES_LOWER:
        return _RULES_LOWER[text_lower]

    best_match = None
    best_len = 0
    for equip in _SORTED_EQUIPMENT:
        equip_lower = equip.lower()
        if equip_lower in text_lower and len(equip_lower) > best_len:
            best_match = equip
            best_len = len(equip_lower)
    if best_match:
        return _RULES[best_match]

    for equip in reversed(_SORTED_EQUIPMENT):
        if text_lower in equip.lower():
            return _RULES[equip]
    return None


def _split_words(s: str) -> set[str]:
    return set(re.split(r'[\s\-]+', s.lower()))


def _find_full_name_in_text(abbrev: str, text: str, ner_entities: list[dict]) -> str | None:
    _ensure_equipment_loaded()
    canonical = _FULL_NAMES_LOWER.get(abbrev.lower())
    if not canonical:
        return None

    canon_words = canonical.split()
    flex_pattern = r'\b' + r'[\s\-]+'.join(re.escape(w) for w in canon_words) + r'\b'
    m = re.search(flex_pattern, text, flags=re.IGNORECASE)
    if m:
        return m.group()

    canon_word_set = _split_words(canonical)
    for ent in ner_entities:
        ent_word_set = _split_words(ent["text"])
        overlap = canon_word_set & ent_word_set
        if len(overlap) >= len(canon_word_set) - 1 and len(canon_word_set) >= 2:
            ent_pattern = r'\b' + re.escape(ent["text"]) + r'\b'
            if re.search(ent_pattern, text, flags=re.IGNORECASE):
                return ent["text"]
    return None


def _derive_full_name_replacement(abbrev_replacement: str) -> str:
    _ensure_equipment_loaded()
    full = _FULL_NAMES_LOWER.get(abbrev_replacement.lower())
    if full:
        return full
    return abbrev_replacement.title()


def _resolve_replacement_chains(selected: list[dict]) -> None:
    """Resolve cascading replacement chains in-place."""
    orig_to_target = {t["original"].lower(): t for t in selected}
    for t in selected:
        original_replacement = t["replacement"]
        repl_lower = t["replacement"].lower()
        seen = {t["original"].lower()}
        while repl_lower in orig_to_target and repl_lower not in seen:
            seen.add(repl_lower)
            t["replacement"] = orig_to_target[repl_lower]["replacement"]
            repl_lower = t["replacement"].lower()
        if t["replacement"] != original_replacement and "group_replacements" in t:
            for gr in t["group_replacements"]:
                gr["replacement"] = _derive_full_name_replacement(t["replacement"])


class EquipmentSubstitution(BasePerturbation):
    name = "equipment_substitution"
    category = "error"

    def detect_targets(self, text: str) -> list[dict]:
        from assay.ner.error_ner import get_error_ner

        _ensure_equipment_loaded()
        entities = get_error_ner().detect(text)
        equip_entities = [e for e in entities if e["type"] == "equipment"]

        targets = []
        seen_equipment: set[str] = set()
        grouped_full_names: set[str] = set()

        for entity in equip_entities:
            equip_lower = entity["text"].lower()
            replacement = _lookup_equipment_rule(equip_lower)
            if replacement is None or equip_lower in seen_equipment:
                continue
            seen_equipment.add(equip_lower)

            target = {
                "original": entity["text"],
                "replacement": replacement,
                "start": entity["start"],
                "end": entity["end"],
                "original_key": equip_lower,
            }

            full_name_match = _find_full_name_in_text(entity["text"], text, equip_entities)
            if full_name_match:
                full_name_replacement = _derive_full_name_replacement(replacement)
                target["group_replacements"] = [{
                    "original": full_name_match,
                    "replacement": full_name_replacement,
                }]
                grouped_full_names.add(full_name_match.lower())

            targets.append(target)

        for entity in equip_entities:
            equip_lower = entity["text"].lower()
            if equip_lower in seen_equipment or equip_lower in grouped_full_names:
                continue

            abbrev = _FULL_NAME_TO_ABBREV.get(equip_lower)
            if abbrev:
                replacement = _lookup_equipment_rule(abbrev)
                if replacement is not None:
                    full_name_replacement = _derive_full_name_replacement(replacement)
                    seen_equipment.add(equip_lower)
                    targets.append({
                        "original": entity["text"],
                        "replacement": full_name_replacement,
                        "start": entity["start"],
                        "end": entity["end"],
                        "original_key": equip_lower,
                    })
                    continue

            replacement = _lookup_equipment_rule(equip_lower)
            if replacement is not None and equip_lower not in seen_equipment:
                seen_equipment.add(equip_lower)
                targets.append({
                    "original": entity["text"],
                    "replacement": replacement,
                    "start": entity["start"],
                    "end": entity["end"],
                    "original_key": equip_lower,
                })

        return targets

    def apply(self, text: str, rate: float = 0.1, max_changes: int | None = None,
              seed: int = 42) -> PerturbationResult:
        targets = self.detect_targets(text)
        selected = self._select_targets(targets, rate, max_changes, seed)
        _resolve_replacement_chains(selected)

        primary_targets = []
        group_targets = []
        for t in selected:
            primary_targets.append({k: v for k, v in t.items() if k != "group_replacements"})
            for gr in t.get("group_replacements", []):
                group_targets.append(gr)

        result = text
        all_changes = []

        if primary_targets:
            result, changes = self._apply_replacements(result, primary_targets)
            all_changes.extend(changes)

        if group_targets:
            result, changes = self._apply_replacements(result, group_targets, preserve_case=False)
            all_changes.extend(changes)

        return PerturbationResult(
            original_text=text,
            perturbed_text=result,
            changes=all_changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )


# ── Action Antonym ──────────────────────────────────────────────────────────

_ANTONYM_MAP: dict[str, str] = {}
_loaded_actions = False


def _ensure_actions_loaded() -> None:
    global _ANTONYM_MAP, _loaded_actions
    if _loaded_actions:
        return
    with open(get_data_dir() / "action_antonyms.json") as f:
        antonym_data = json.load(f)
    _ANTONYM_MAP = {}
    for pair in antonym_data["pairs"]:
        a = pair["a"]
        b = pair["b"]
        for form_key in ("base", "past", "gerund"):
            if form_key in a and form_key in b:
                _ANTONYM_MAP[a[form_key].lower()] = b[form_key]
                _ANTONYM_MAP[b[form_key].lower()] = a[form_key]
    _loaded_actions = True


class ActionAntonym(BasePerturbation):
    name = "action_antonym"
    category = "error"

    def detect_targets(self, text: str) -> list[dict]:
        from assay.ner.error_ner import get_error_ner

        _ensure_actions_loaded()
        entities = get_error_ner().detect(text)
        action_entities = [e for e in entities if e["type"] == "action_verb"]

        targets = []
        seen_words: set[str] = set()
        for entity in action_entities:
            word_lower = entity["text"].lower()
            if word_lower not in _ANTONYM_MAP or word_lower in seen_words:
                continue
            seen_words.add(word_lower)

            targets.append({
                "original": entity["text"],
                "replacement": _ANTONYM_MAP[word_lower],
                "start": entity["start"],
                "end": entity["end"],
                "antonym_pair": f"{word_lower} <-> {_ANTONYM_MAP[word_lower]}",
            })

        return targets

    def apply(self, text: str, rate: float = 0.1, max_changes: int | None = None,
              seed: int = 42) -> PerturbationResult:
        targets = self.detect_targets(text)
        selected = self._select_targets(targets, rate, max_changes, seed)
        perturbed, changes = self._apply_replacements(text, selected)

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            changes=changes,
            perturbation_type=self.name,
            category=self.category,
            rate=rate,
            seed=seed,
        )
