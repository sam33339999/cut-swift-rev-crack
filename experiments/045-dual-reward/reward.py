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


PROBE_PERIOD_S = 30
_ANS_LOOP = re.compile(r"(?:ANS){8,}")
_ANSWER_LINE = re.compile(r"ANSWER:\s*[+-]?\d+")


def advantage_mask(pieces: list[str]) -> list[float]:
    """1 inside the think span and on the </think> token. 0 on ANSWER: tokens.

    After </think> the visible answer is masked off, including a bare ANS fragment.
    """
    mask: list[float] = []
    closed = False
    for piece in pieces:
        if closed or _answer_piece(piece):
            mask.append(0.0)
            if "</think>" in piece:
                closed = True
            continue
        mask.append(1.0)
        if "</think>" in piece:
            closed = True
    return mask


def _answer_piece(piece: str) -> bool:
    compact = piece.replace(" ", "")
    if compact in {"ANS", "ANSWER", "ANSWER:"}:
        return True
    return "ANSWER:" in piece


def trainable_post_answer_span(
    token_ids: list[int],
    prompt_len: int,
    first_correct_index: int | None,
    max_len: int,
) -> list[int] | None:
    """Keep a long sequence when the post-answer span can still be trained.

    first_correct_index is an index into the completion, after prompt_len.
    A sequence that fits is returned whole. A longer sequence without a cut is skipped.
    """
    if len(token_ids) <= max_len:
        return list(token_ids)
    if first_correct_index is None:
        return None
    start = prompt_len + first_correct_index
    if start >= len(token_ids):
        return None
    span = token_ids[start:start + max_len]
    if not span:
        return None
    return span


def is_ans_collapse(text: str, content: str = "", truncated: bool = False) -> bool:
    """True for the failed e01 pattern: a repeated ANS fragment and no answer line."""
    blob = text or ""
    visible = content or ""
    if not _ANS_LOOP.search(blob):
        return False
    if _ANSWER_LINE.search(visible):
        return False
    return True


def probe_decision(text: str, content: str = "", truncated: bool = False) -> str:
    """Return discard or keep. No user input."""
    if is_ans_collapse(text, content, truncated):
        return "discard"
    return "keep"


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
