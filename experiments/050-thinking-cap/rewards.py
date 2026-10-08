"""Thinking-Cap reward, adapted from khudgins/ornith-thinking-cap.

reward = correctness
       - (lambda * normalized thinking length, only if correct)
       + format bonus

A wrong answer cannot outscore a correct one: the length term is at most
lambda (0.05), and the format bonus is 0.1, while correctness is 1.0.
"""

from __future__ import annotations

import re

_THINK_PAIR = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_CLOSE = re.compile(r"(.*?)</think>", re.DOTALL)
_NUMBER = re.compile(r"-?\d+\.?\d*")


def extract_number(text: str) -> str | None:
    nums = _NUMBER.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def compare_number(actual: str | None, expected: str, tol: float = 0.01) -> bool:
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tol
    except ValueError:
        return actual.strip() == expected.strip()


def split_think(text: str) -> tuple[str, str]:
    match = _THINK_PAIR.search(text)
    if match:
        return match.group(1), text[match.end() :]
    match = _THINK_CLOSE.search(text)
    if match:
        return match.group(1), text[match.end() :]
    return "", text


def reasoning_tokens(think: str, tokenize) -> int:
    if not think.strip():
        return 0
    return len(list(tokenize(think)))


def score_completion(
    text: str,
    gold: str,
    tokenize,
    lam: float = 1e-3,
    max_think_tokens: int = 2048,
    min_think_tokens: int = 32,
    format_bonus: float = 0.1,
) -> dict:
    think, answer = split_think(text)
    correct = compare_number(extract_number(answer), gold)
    correctness = 1.0 if correct else 0.0
    if correct:
        used = max(reasoning_tokens(think, tokenize), min_think_tokens)
        length_penalty = -lam * min(1.0, used / max_think_tokens)
    else:
        length_penalty = 0.0
    has_think = bool(_THINK_PAIR.search(text) or _THINK_CLOSE.search(text))
    fmt = format_bonus if has_think and answer.strip() else 0.0
    return {
        "correct": correct,
        "correctness": correctness,
        "length_penalty": length_penalty,
        "format_bonus": fmt,
        "total": correctness + length_penalty + fmt,
    }
