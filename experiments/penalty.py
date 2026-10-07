"""First-tier stall phrases from recommended.json, as vLLM bad_words sequences.

vLLM bans a phrase by setting the last token's logit to -inf only after the
preceding tokens of that phrase have just been produced. A leading space is
added as a second sequence when it changes the first token and keeps the
length. Capitalized sentence-start forms are separate strings because a
capital letter is not inserted automatically.
"""

from __future__ import annotations

import json
from pathlib import Path


def first_tier_phrases(recommended: dict) -> list[str]:
    phrases: list[str] = []
    for item in recommended["first"]:
        surface = item["surface"].strip()
        if not surface:
            raise ValueError("empty first-tier surface")
        phrases.append(surface)
        capital = surface[:1].upper() + surface[1:]
        if capital != surface:
            phrases.append(capital)
    if len(phrases) < 2:
        raise ValueError("expected the four first-tier phrases")
    return list(dict.fromkeys(phrases))


def load_first_tier_phrases(path: Path) -> list[str]:
    recommended = json.loads(path.read_text(encoding="utf-8"))
    return first_tier_phrases(recommended)


def bad_word_token_ids(tokenizer, phrases: list[str]) -> list[list[int]]:
    """Same selection rule as vLLM SamplingParams.update_from_tokenizer."""
    sequences: list[list[int]] = []
    for phrase in phrases:
        for add_prefix_space in (False, True):
            prefix = " " if add_prefix_space else ""
            prompt = prefix + phrase.lstrip()
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not token_ids:
                if not add_prefix_space:
                    raise ValueError(f"phrase tokenized to nothing: {phrase!r}")
                continue
            if (not add_prefix_space) or (
                token_ids[0] != sequences[-1][0] and len(token_ids) == len(sequences[-1])
            ):
                sequences.append(list(token_ids))
    return sequences
