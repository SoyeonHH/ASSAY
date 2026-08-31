"""LLM-based chemical entity conversion with persistent caching.

Fallback for PubChem-unresolvable entities (polymers, complex compounds,
trade names). Uses GPT-4.1-mini via OpenRouter to generate alternative
representations (formula, IUPAC, common name).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from litellm import completion

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_CACHE = _DATA_DIR / "llm_convert_cache.json"

_CONVERT_SYSTEM_PROMPT = """\
You are a materials science chemistry expert. Given a chemical entity, \
provide its alternative representations.

For each field:
- "formula": Molecular formula in Hill notation. For polymers, give the \
repeat unit (e.g., C2H4O for PEO).
- "iupac": IUPAC systematic name.
- "common_name": Most widely recognized trivial name.

CRITICAL: Return null for any field you are not confident about.
Return JSON: {"formula": ..., "iupac": ..., "common_name": ...}"""


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: export OPENROUTER_API_KEY=your_key"
        )
    return key


def _validate_formula(value: str | None, original: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not re.match(r'^[A-Z][A-Za-z0-9()\[\]·.+-]{0,49}$', value):
        return None
    if value.lower() == original.lower():
        return None
    return value


def _validate_iupac(value: str | None, original: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not (3 <= len(value) <= 120):
        return None
    if value.lower() == original.lower():
        return None
    return value


def _validate_common_name(value: str | None, original: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not (2 <= len(value) <= 60):
        return None
    if value.lower() == original.lower():
        return None
    return value


class LLMChemicalConverter:
    """LLM-based chemical entity converter with persistent caching."""

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        cache_file: str | Path | None = None,
    ):
        self.model = model
        self.cache_file = Path(cache_file) if cache_file else _DEFAULT_CACHE
        self._cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cache_key(entity: str, entity_type: str, full_name: str | None) -> str:
        parts = [entity.lower(), entity_type.lower()]
        parts.append(full_name.lower() if full_name else "")
        return "|".join(parts)

    def _call_llm(self, entity: str, entity_type: str, full_name: str | None) -> dict:
        os.environ["OPENROUTER_API_KEY"] = _get_api_key()

        user_parts = [f"Entity: {entity}", f"Type: {entity_type}"]
        if full_name:
            user_parts.append(f"Full name: {full_name}")
        user_msg = "\n".join(user_parts)

        response = completion(
            model=f"openrouter/{self.model}",
            messages=[
                {"role": "system", "content": _CONVERT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {}

    def convert(
        self,
        entity: str,
        entity_type: str = "compound_name",
        full_name: str | None = None,
    ) -> dict[str, str | None]:
        """Convert a chemical entity to alternative representations.

        Returns dict with keys: formula, iupac, common_name (each may be None).
        """
        key = self._cache_key(entity, entity_type, full_name)

        if key in self._cache:
            raw = self._cache[key]
        else:
            raw = self._call_llm(entity, entity_type, full_name)
            self._cache[key] = raw
            self._save_cache()

        return {
            "formula": _validate_formula(raw.get("formula"), entity),
            "iupac": _validate_iupac(raw.get("iupac"), entity),
            "common_name": _validate_common_name(raw.get("common_name"), entity),
        }

    @property
    def cache_size(self) -> int:
        return len(self._cache)
