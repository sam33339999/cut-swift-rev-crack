"""Method 45 rewards. Wrong answers score 0. Correct answers get a small length bonus and a review penalty.

The length term uses the mean thinking length of the correct samples in the same group.
The review term is the fraction of thinking tokens after the first formed answer that equals the gold integer.
A bare intermediate number does not count.
"""

from __future__ import annotations

import re

_FORMED = re.compile(
    r"(?:\\boxed\{(?P<boxed>[+-]?\d+)"
    r"|the answer is\s+(?P<phrase>[+-]?\d+)"
    r"|answer\s*:\s*(?P<colon>[+-]?\d+)"
    r"|答案是\s*(?P<zh>[+-]?\d+))",
    re.IGNORECASE,
)


def thinking_text(text: str) -> str:
    end = text.find("</think>")
    if end == -1:
        return text
    return text[:end]


def first_gold_end(thinking: str, gold: str) -> int | None:
    target = str(int(gold))
    for match in _FORMED.finditer(thinking):
        raw = next(group for group in match.groups() if group)
        if str(int(raw)) == target:
            return match.end()
    return None


def review_fraction(reasoning: str, gold: str, tokenize) -> float:
    thinking = thinking_text(reasoning or "")
    token_ids = list(tokenize(thinking))
    if not token_ids:
        return 0.0
    cut = first_gold_end(thinking, gold)
    if cut is None:
        return 0.0
    before = list(tokenize(thinking[:cut]))
    after = max(0, len(token_ids) - len(before))
    return after / len(token_ids)


def group_rewards(rows: list[dict], tokenize) -> list[float]:
    """rows need correct, gold, reasoning, and reasoning_tokens."""
    correct_lengths = [
        int(row["reasoning_tokens"])
        for row in rows
        if row.get("correct") and row.get("reasoning_tokens") is not None
    ]
    mean_correct = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0
    rewards = []
    for row in rows:
        if not row.get("correct"):
            rewards.append(0.0)
            continue
        think = int(row.get("reasoning_tokens") or 0)
        if mean_correct <= 0:
            length_term = 0.0
        else:
            length_term = 1.0 - (think / mean_correct)
        review = review_fraction(row.get("reasoning") or "", str(row["gold"]), tokenize)
        rewards.append(1.0 + 0.1 * length_term - 0.1 * review)
    return rewards
