"""The after-answer penalty stays off until a formed answer, then bans phrase endings."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.penalty import bad_word_token_ids, load_first_tier_phrases
from experiments.penalty_after_answer import (
    FLAG,
    RECOMMENDED,
    PenaltyAfterAnswerProcessor,
    completing_token_ids,
    formed_answer_seen,
)
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
)

MODEL = "/content/models/ornith-1.5-9B"


class _Tensor:
    def __init__(self, values):
        self.gpu = values if isinstance(values, torch.Tensor) else torch.tensor(values)


class _Host:
    def __init__(self, values):
        self.np = np.array(values)


class PenaltyAfterAnswerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        cls.phrases = load_first_tier_phrases(RECOMMENDED)
        cls.sequences = bad_word_token_ids(cls.tokenizer, cls.phrases)
        spaced = cls.tokenizer.encode(" reconsider", add_special_tokens=False)
        cls.reconsider = next(seq for seq in cls.sequences if seq[-1] == spaced[0])

    def test_detector_ignores_intermediate_numbers(self):
        self.assertFalse(formed_answer_seen("n = 405 is still short of 100 zeros."))
        self.assertFalse(formed_answer_seen("Let me reconsider the total."))
        self.assertTrue(formed_answer_seen("So the answer is 405."))
        self.assertTrue(formed_answer_seen("Answer: 401."))
        self.assertTrue(formed_answer_seen(r"\boxed{405}"))
        self.assertTrue(formed_answer_seen("答案是 12"))
        self.assertFalse(
            formed_answer_seen("still working.</think>\nThe answer is 405.")
        )
        self.assertTrue(
            formed_answer_seen("The answer is 405.</think>\nlet me reconsider")
        )

    def test_phrase_ban_needs_the_prefix(self):
        prefix = self.reconsider[:-1]
        self.assertEqual(
            completing_token_ids(prefix, self.sequences).count(self.reconsider[-1]),
            1,
        )
        self.assertNotIn(
            self.reconsider[-1],
            completing_token_ids(prefix[:-1], self.sequences),
        )

    def test_processor_is_the_v2_interface_and_is_request_gated(self):
        self.assertTrue(issubclass(PenaltyAfterAnswerProcessor, LogitsProcessor))
        PenaltyAfterAnswerProcessor.validate_params(SimpleNamespace(extra_args=None))
        PenaltyAfterAnswerProcessor.validate_params(
            SimpleNamespace(extra_args={FLAG: 1})
        )
        with self.assertRaises(ValueError):
            PenaltyAfterAnswerProcessor.validate_params(
                SimpleNamespace(extra_args={FLAG: -2})
            )

    def test_logits_stay_open_until_the_answer_then_the_phrase_is_masked(self):
        prefix = self.reconsider[:-1]
        before = self.tokenizer.encode("n = 405. ", add_special_tokens=False) + prefix
        after = self.tokenizer.encode("The answer is 405. ", add_special_tokens=False) + prefix
        processor = self._processor(before)
        params = SimpleNamespace(extra_args={FLAG: 1})
        self.assertTrue(processor.add_request(0, params))
        open_logits = self._apply(processor, before)
        self.assertFalse(torch.isneginf(open_logits[0, self.reconsider[-1]]).item())

        processor = self._processor(after)
        processor.add_request(0, params)
        masked = self._apply(processor, after)
        self.assertTrue(torch.isneginf(masked[0, self.reconsider[-1]]).item())
        self.assertFalse(torch.isneginf(masked[0, 0]).item())

        quiet = self._processor(after)
        self.assertFalse(quiet.add_request(0, SimpleNamespace(extra_args=None)))
        untouched = self._apply(quiet, after)
        self.assertFalse(torch.isneginf(untouched[0, self.reconsider[-1]]).item())

    def _processor(self, output_ids: list[int]) -> PenaltyAfterAnswerProcessor:
        width = len(output_ids) + 4
        tokens = torch.zeros((1, width), dtype=torch.int64)
        tokens[0, : len(output_ids)] = torch.tensor(output_ids)
        req_states = SimpleNamespace(
            prompt_len=_Host([0]),
            total_len=_Tensor([len(output_ids)]),
            all_token_ids=_Tensor(tokens),
        )
        config = SimpleNamespace(model_config=SimpleNamespace(model=MODEL))
        return PenaltyAfterAnswerProcessor(config, req_states)

    def _apply(self, processor, output_ids: list[int]) -> torch.Tensor:
        vocab = max(token for seq in processor.sequences for token in seq) + 1
        logits = torch.zeros((1, vocab), dtype=torch.float32)
        ctx = LogitsContext(
            expanded_idx_mapping=torch.tensor([0]),
            idx_mapping=torch.tensor([0]),
            idx_mapping_np=np.array([0]),
            expanded_local_pos=torch.tensor([0]),
            input_ids=torch.tensor([output_ids[-1]]),
            pos=torch.tensor([0]),
            seq_lens_upper_bound_np=np.array([len(output_ids)]),
        )
        return processor.apply(logits, ctx)


if __name__ == "__main__":
    unittest.main()
