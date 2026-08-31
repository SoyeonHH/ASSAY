"""LLM-based Chemical Named Entity Recognition with persistent caching.

Uses GPT-4.1-mini via OpenRouter to extract chemical entities from text,
classifying them as formula / compound_name / abbreviation / iupac_name.
Results are cached by SHA-256 hash of input text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from litellm import completion

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_CACHE = _DATA_DIR / "ner_cache.json"

_NER_SYSTEM_PROMPT = """\
You are a materials science NER system. Extract ALL chemical entities from the given synthesis recipe text.

For each entity, classify its type as one of:
- "formula": Chemical formulas like H2O, SiO2, TiO2, NaCl, Fe3O4, CH3OH
- "compound_name": Common chemical names like "ethanol", "titanium dioxide", "sodium hydroxide"
- "abbreviation": Chemical abbreviations like "TEOS", "PVA", "CTAB", "DMF", "DMSO"
- "iupac_name": IUPAC systematic names like "tetraethyl orthosilicate", "poly(vinyl alcohol)"

Rules:
- Extract every distinct chemical entity mentioned in the text
- Include elements when used as chemical substances (e.g., "gold", "silicon", "Ar" as argon gas)
- Include solvents, precursors, reagents, catalysts, dopants, substrates, and products
- DO NOT include: equipment names (furnace, autoclave), techniques (XRD, SEM, TEM), units (mL, mg, nm), or generic terms (solution, mixture, sample, substrate when not a specific material)
- DO NOT include process parameters (temperature, pressure, time values)
- If the same entity appears multiple times, include it only ONCE

Return a JSON object with a single key "entities" containing an array of objects, each with "text" (exact string as it appears) and "type" fields.

Example output:
{"entities": [{"text": "TiO2", "type": "formula"}, {"text": "ethanol", "type": "compound_name"}, {"text": "TEOS", "type": "abbreviation"}]}"""


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: export OPENROUTER_API_KEY=your_key"
        )
    return key


class LLMChemicalNER:
    """LLM-based chemical entity extractor with SHA-256 caching."""

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        cache_file: str | Path | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.cache_file = Path(cache_file) if cache_file else _DEFAULT_CACHE
        self._system_prompt = system_prompt or _NER_SYSTEM_PROMPT
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
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _call_llm(self, text: str) -> list[dict]:
        os.environ["OPENROUTER_API_KEY"] = _get_api_key()

        response = completion(
            model=f"openrouter/{self.model}",
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=4096,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
            entities = parsed.get("entities", [])
            return [
                {"text": e["text"], "type": e["type"]}
                for e in entities
                if isinstance(e, dict) and "text" in e and "type" in e
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _locate_entities(self, text: str, entities: list[dict]) -> list[dict]:
        located = []
        for entity in entities:
            ent_text = entity["text"]
            pattern = re.escape(ent_text)
            if re.match(r'\w', ent_text):
                pattern = r'\b' + pattern
            if re.search(r'\w$', ent_text):
                pattern = pattern + r'\b'

            match = re.search(pattern, text)
            if match:
                located.append({
                    "text": entity["text"],
                    "type": entity["type"],
                    "start": match.start(),
                    "end": match.end(),
                })
        return located

    def detect(self, text: str) -> list[dict]:
        """Detect chemical entities in text, using cache when available.

        Returns list of dicts with keys: text, type, start, end.
        """
        cache_key = self._hash_text(text)

        if cache_key in self._cache:
            raw_entities = self._cache[cache_key]
            return self._locate_entities(text, raw_entities)

        raw_entities = self._call_llm(text)
        self._cache[cache_key] = raw_entities
        self._save_cache()

        return self._locate_entities(text, raw_entities)

    def detect_batch(self, texts: list[str]) -> list[list[dict]]:
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    @property
    def cache_size(self) -> int:
        return len(self._cache)
