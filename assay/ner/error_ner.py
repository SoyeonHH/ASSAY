"""Error entity NER — detects elements, numerical values, equipment, and action verbs.

The system prompt and the cache file are read from the configured domain data
directory (``assay.domain``). A domain that ships its own ``error_ner_prompt.txt``
overrides the default materials science prompt below.
"""

from __future__ import annotations

from assay.domain import get_data_dir
from assay.ner.chemical_ner import LLMChemicalNER

_ERROR_NER_SYSTEM_PROMPT = """\
You are a materials science NER system. Extract ALL relevant entities from the given synthesis recipe text.

For each entity, classify its type as one of:
- "element": Chemical element symbols used as substances or in recognizable positions within formulas. Include elements at word boundaries or recognizable formula boundaries (e.g., "Li" in LiTFSI, "P" in P(EO)14, "Ti" in TiO2, "Na", "Fe", "Ar"). Do NOT extract elements buried inside compound formulas where they are inseparable (e.g., do not extract "O" from H2O or "Si" from SiO2). Focus on elements that could be independently substituted.
- "numerical_value": Number+unit combinations representing measurable quantities. Include the number AND unit as a single entity. Examples: "500 °C", "10 hours", "5 mg", "0.1 M", "20 nm", "3000 rpm", "5%", "2 mL", "150 W", "1 atm".
- "equipment": Laboratory equipment, instruments, and apparatus. Examples: "furnace", "autoclave", "centrifuge", "SEM", "XRD", "TGA", "ball mill", "glovebox", "hot plate", "magnetic stirrer", "oven", "mortar", "pestle".
- "action_verb": Synthesis procedure verbs in their exact inflected form as they appear in the text. Examples: "heated", "stirring", "dissolved", "calcined", "dried", "added", "mixed", "annealed", "sintered", "washed", "filtered", "cooled", "ground", "dispersed".

Rules:
- Extract every distinct entity mentioned in the text
- If the same entity text appears multiple times, include it only ONCE
- For numerical_value: always include both the number and its unit together as one entity
- For action_verb: extract the verb in its exact form (past tense, gerund, etc.)
- For element: only extract when the element symbol is clearly identifiable and could be substituted

Return a JSON object with a single key "entities" containing an array of objects, each with "text" (exact string as it appears) and "type" fields.

Example output:
{"entities": [{"text": "Ti", "type": "element"}, {"text": "500 °C", "type": "numerical_value"}, {"text": "furnace", "type": "equipment"}, {"text": "heated", "type": "action_verb"}]}"""

_error_ner_instance: LLMChemicalNER | None = None


def get_error_ner() -> LLMChemicalNER:
    """Get or create the error NER singleton for the active domain."""
    global _error_ner_instance
    if _error_ner_instance is None:
        data_dir = get_data_dir()
        prompt_file = data_dir / "error_ner_prompt.txt"
        system_prompt = (
            prompt_file.read_text(encoding="utf-8").strip()
            if prompt_file.exists()
            else _ERROR_NER_SYSTEM_PROMPT
        )
        _error_ner_instance = LLMChemicalNER(
            system_prompt=system_prompt,
            cache_file=data_dir / "ner_error_cache.json",
        )
    return _error_ner_instance
