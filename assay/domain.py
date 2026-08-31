"""Domain configuration for the augmentation system.

Selects the directory holding the swap dictionaries and the domain-specific
NER prompt, so the same augmentation code runs on materials synthesis recipes
(``assay/augmentation/data``) or biological protocols
(``assay/augmentation/data_bio``).
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "augmentation" / "data"
_data_dir: Path = _DEFAULT_DATA_DIR


def set_data_dir(path: str | Path) -> None:
    """Set the global data directory for swap dictionaries and NER prompts."""
    global _data_dir
    _data_dir = Path(path).resolve()
    reset_singletons()


def get_data_dir() -> Path:
    return _data_dir


def reset_singletons() -> None:
    """Clear cached dictionaries and NER instances after a domain change."""
    from assay.ner import error_ner
    error_ner._error_ner_instance = None

    from assay.augmentation import error_injection
    error_injection._loaded_elements = False
    error_injection._loaded_equipment = False
    error_injection._loaded_actions = False
