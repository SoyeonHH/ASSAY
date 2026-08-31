"""Base classes and dataclasses for the augmentation system."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Change:
    """A single change made during augmentation."""
    original: str
    replacement: str
    start: int
    end: int
    category: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PerturbationResult:
    """Result of applying an augmentation to text."""
    original_text: str
    perturbed_text: str
    changes: list[Change]
    perturbation_type: str
    category: str
    rate: float | None
    seed: int


class BasePerturbation(ABC):
    """Abstract base class for all augmentation types."""

    name: str = ""
    category: str = ""

    @abstractmethod
    def detect_targets(self, text: str) -> list[dict]:
        """Detect all possible augmentation targets in the text.

        Returns list of dicts with at least:
            - 'original': the matched text
            - 'replacement': the proposed replacement
            - 'start': start position in text
            - 'end': end position in text
        """

    @abstractmethod
    def apply(self, text: str, rate: float = 1.0, max_changes: int | None = None,
              seed: int = 42) -> PerturbationResult:
        """Apply augmentation to text at the given rate."""

    def _select_targets(self, targets: list[dict], rate: float,
                        max_changes: int | None, seed: int) -> list[dict]:
        if not targets:
            return []

        rng = random.Random(seed)
        shuffled = list(targets)
        rng.shuffle(shuffled)
        n = max(1, int(len(shuffled) * rate))
        if max_changes is not None:
            n = min(n, max_changes)
        n = min(n, len(shuffled))
        return shuffled[:n]

    def _apply_replacements(
        self, text: str, selected: list[dict], preserve_case: bool = True,
    ) -> tuple[str, list[Change]]:
        """Apply replacements using word-boundary substitution.

        Replaces ALL occurrences of each selected target.
        """
        changes: list[Change] = []
        result = text

        for target in selected:
            original = target["original"]
            replacement = target["replacement"]
            metadata = {k: v for k, v in target.items()
                        if k not in ("original", "replacement", "start", "end")}

            pattern = r'\b' + re.escape(original) + r'\b'
            match_positions = [(m.start(), m.end()) for m in re.finditer(pattern, result, flags=re.IGNORECASE)]

            if not match_positions:
                continue

            if preserve_case:
                is_abbrev = replacement.isupper()

                def _replacer(match: re.Match) -> str:
                    if is_abbrev:
                        return replacement
                    matched = match.group()
                    if matched.isupper():
                        return replacement.upper()
                    elif matched[0].isupper():
                        return replacement[0].upper() + replacement[1:].lower() if len(replacement) > 1 else replacement.upper()
                    else:
                        return replacement.lower()

                new_result = re.sub(pattern, _replacer, result, flags=re.IGNORECASE)
            else:
                new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

            changes.append(Change(
                original=original,
                replacement=replacement,
                start=match_positions[0][0],
                end=match_positions[0][1],
                category=self.category,
                metadata=metadata,
            ))

            result = new_result

        return result, changes

    def _apply_positional_replacements(self, text: str, selected: list[dict]) -> tuple[str, list[Change]]:
        """Apply replacements at specific positions (for numerical augmentations)."""
        changes: list[Change] = []
        sorted_targets = sorted(selected, key=lambda t: t["start"], reverse=True)

        result = text
        for target in sorted_targets:
            start = target["start"]
            end = target["end"]
            replacement = target["replacement"]
            metadata = {k: v for k, v in target.items()
                        if k not in ("original", "replacement", "start", "end")}

            actual = result[start:end]
            if actual != target["original"]:
                continue

            result = result[:start] + replacement + result[end:]
            changes.append(Change(
                original=target["original"],
                replacement=replacement,
                start=start,
                end=end,
                category=self.category,
                metadata=metadata,
            ))

        return result, changes
