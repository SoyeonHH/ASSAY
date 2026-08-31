"""Score extraction from judge output text.

Handles multiple output formats: markdown-fenced JSON, raw JSON,
JudgeLRM <answer> tags, and arithmetic expressions in scores.
"""

from __future__ import annotations

import json
import re


def _safe_eval_arithmetic(expr: str) -> str | None:
    """Evaluate simple arithmetic expressions containing only numbers and +-*/()."""
    cleaned = expr.strip()
    eq_match = re.search(r'=\s*([\d.]+)\s*$', cleaned)
    if eq_match:
        return eq_match.group(1)
    cleaned = cleaned.rstrip("=").strip()
    if not cleaned or not re.match(r'^[\d.+\-*/() \t]+$', cleaned):
        return None
    try:
        result = eval(cleaned, {"__builtins__": {}})  # noqa: S307
        return str(round(float(result), 2))
    except Exception:
        return None


def _fix_arithmetic_in_json(text: str) -> str:
    """Replace arithmetic expressions in JSON score values with computed results."""
    text = re.sub(r"\([^)]+\)\s*/\s*[\d.]+\s*=\s*([\d.]+)", r"\1", text)
    text = re.sub(
        r'":\s*([\d.+\-*/() =\t]+?)(\s*[,}\n])',
        lambda m: f'": {_safe_eval_arithmetic(m.group(1)) or m.group(1)}{m.group(2)}',
        text,
    )
    return text


def extract_overall_score(judge_result: str) -> float | None:
    """Extract overall_score from judge output text.

    Tries multiple strategies:
    1. ```json {...} ``` fenced block
    2. ``` {...} ``` fenced block (no language tag)
    3. Raw JSON object containing "overall_score"
    4. JudgeLRM <answer>N</answer> format (1-10 scale → 1-5)
    """
    if not judge_result:
        return None

    for pattern in [
        r"```json\s*(\{[^`]+\})\s*```",
        r"```\s*(\{[^`]+\})\s*```",
    ]:
        match = re.search(pattern, judge_result, re.DOTALL)
        if match:
            raw = _fix_arithmetic_in_json(match.group(1))
            try:
                scores = json.loads(raw)
                overall = scores.get("overall_score")
                if overall is not None:
                    return float(overall)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    match = re.search(r'(\{[^{}]*"overall_score"[^{}]*\})', judge_result, re.DOTALL)
    if match:
        raw = _fix_arithmetic_in_json(match.group(1))
        try:
            scores = json.loads(raw)
            overall = scores.get("overall_score")
            if overall is not None:
                return float(overall)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    match = re.search(r"<answer>\s*(\d+)\s*</answer>", judge_result)
    if match:
        return round(float(match.group(1)) / 2, 1)

    return None


def extract_all_scores(judge_result: str) -> dict | None:
    """Extract all scores from judge output text.

    Returns dict with all score keys, or None if parsing fails.
    """
    if not judge_result:
        return None

    for pattern in [
        r"```json\s*(\{[^`]+\})\s*```",
        r"```\s*(\{[^`]+\})\s*```",
    ]:
        match = re.search(pattern, judge_result, re.DOTALL)
        if match:
            raw = _fix_arithmetic_in_json(match.group(1))
            try:
                scores = json.loads(raw)
                if "overall_score" in scores:
                    return scores
            except json.JSONDecodeError:
                continue

    match = re.search(r'(\{[^{}]*"overall_score"[^{}]*\})', judge_result, re.DOTALL)
    if match:
        raw = _fix_arithmetic_in_json(match.group(1))
        try:
            scores = json.loads(raw)
            if "overall_score" in scores:
                return scores
        except json.JSONDecodeError:
            pass

    # JudgeLRM format
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", judge_result)
    if match:
        score_10 = float(match.group(1))
        return {"overall_score": round(score_10 / 2, 1)}

    return None
