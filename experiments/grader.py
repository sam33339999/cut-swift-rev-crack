"""Gold-answer grading for visible replies. No network and no model calls."""

from __future__ import annotations

import re

ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*answer\s*[:：]\s*(-?\d{1,3}(?:,\d{3})+|-?\d+)\s*$"
)


def extract_answer(content: str | None) -> str | None:
    """Return the integer on the last ANSWER line, with commas removed."""
    if not content:
        return None
    matches = list(ANSWER_RE.finditer(content.strip()))
    if not matches:
        return None
    return matches[-1].group(1).replace(",", "")


def is_correct(content: str | None, gold: str, truncated: bool) -> bool:
    """Correct only when the reply was not truncated and the final ANSWER matches.

    Truncated replies stay in the accuracy denominator; callers count them as
    wrong by using this function, rather than dropping the row.
    """
    if truncated:
        return False
    return extract_answer(content) == str(gold)
