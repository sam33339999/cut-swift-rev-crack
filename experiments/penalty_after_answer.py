"""Ban the 002 phrases only after a formed answer has appeared in thinking.

This is a vLLM V2 logits processor. It is a no-op unless the request sets
``vllm_xargs.penalty_after_answer`` to 1. It does not register ``bad_words``,
so the phrases stay available until the detector trips.

The online detector is a cheap text rule, not the offline answer probe.
It looks only at text before ``</think>``. A bare integer does not count.
"""

from __future__ import annotations

import re
from pathlib import Path

from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
RECOMMENDED = ROOT / "experiments" / "001-baseline-and-mine" / "recommended.json"
FLAG = "penalty_after_answer"

# A committed answer, not an intermediate quantity. Case is ignored.
_FORMED_ANSWER = re.compile(
    r"(?:\\boxed\{[+-]?\d|the answer is\s+[+-]?\d|answer\s*:\s*[+-]?\d|答案是\s*[+-]?\d)",
    re.IGNORECASE,
)


def thinking_text(text: str) -> str:
    end = text.find("</think>")
    if end == -1:
        return text
    return text[:end]


def formed_answer_seen(text: str) -> bool:
    return _FORMED_ANSWER.search(thinking_text(text)) is not None


def completing_token_ids(output_ids: list[int], sequences: list[list[int]]) -> list[int]:
    """Token ids to mask when output already ends with a phrase prefix.

    Same last-token rule as vLLM bad_words: the completing token is masked
    only after the preceding tokens have just been produced.
    """
    banned: list[int] = []
    seen: set[int] = set()
    for sequence in sequences:
        if not sequence:
            continue
        prefix = sequence[:-1]
        if len(output_ids) < len(prefix):
            continue
        if prefix and output_ids[-len(prefix) :] != prefix:
            continue
        token_id = sequence[-1]
        if token_id not in seen:
            seen.add(token_id)
            banned.append(token_id)
    return banned


class PenaltyAfterAnswerProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states) -> None:
        from transformers import AutoTokenizer

        from experiments.penalty import bad_word_token_ids, load_first_tier_phrases

        self.req_states = req_states
        self.enabled: dict[int, bool] = {}
        self.armed: dict[int, bool] = {}
        model = vllm_config.model_config.model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.phrases = load_first_tier_phrases(RECOMMENDED)
        self.sequences = bad_word_token_ids(self.tokenizer, self.phrases)

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        extra = sampling_params.extra_args or {}
        if FLAG not in extra:
            return
        if extra[FLAG] not in (0, 1):
            raise ValueError("penalty_after_answer must be 0 or 1")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        extra = sampling_params.extra_args or {}
        enabled = extra.get(FLAG) == 1
        self.enabled[req_idx] = enabled
        self.armed[req_idx] = False
        return enabled

    def _output_ids(self, slot: int) -> list[int]:
        prompt_len = int(self.req_states.prompt_len.np[slot])
        total_len = int(self.req_states.total_len.gpu[slot].item())
        if total_len <= prompt_len:
            return []
        return self.req_states.all_token_ids.gpu[slot, prompt_len:total_len].tolist()

    def apply(self, logits, ctx: LogitsContext):
        n_rows = logits.shape[0]
        if n_rows == len(ctx.idx_mapping_np):
            slots = [int(slot) for slot in ctx.idx_mapping_np]
        else:
            slots = [int(slot) for slot in ctx.expanded_idx_mapping.tolist()]
        for row, slot in enumerate(slots):
            if not self.enabled.get(slot):
                continue
            output_ids = self._output_ids(slot)
            if not self.armed.get(slot):
                text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                if formed_answer_seen(text):
                    self.armed[slot] = True
            if not self.armed.get(slot):
                continue
            for token_id in completing_token_ids(output_ids, self.sequences):
                logits[row, token_id] = float("-inf")
        return logits
