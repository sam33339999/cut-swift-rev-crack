"""The phrase penalty is built from recommended.json and this model's tokenizer."""

import unittest
from pathlib import Path

import importlib.util

from experiments.penalty import bad_word_token_ids, load_first_tier_phrases

ROOT = Path(__file__).resolve().parents[1]
RECOMMENDED = ROOT / "experiments" / "001-baseline-and-mine" / "recommended.json"
MODEL = "/content/models/ornith-1.5-9B"


class PenaltyTest(unittest.TestCase):
    def test_scoring_is_refused_before_reload(self):
        path = Path(__file__).resolve().parent / "002-phrase-penalty" / "run_round.py"
        spec = importlib.util.spec_from_file_location("round_002", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(RuntimeError):
            module.refuse_unless_reloaded()
    def test_first_tier_phrases_come_from_the_file(self):
        phrases = load_first_tier_phrases(RECOMMENDED)
        for surface in ("let me reconsider", "to be safe", "make sure", "let me recheck"):
            self.assertIn(surface, phrases)
            self.assertIn(surface[:1].upper() + surface[1:], phrases)
        self.assertNotIn("wait", phrases)
        self.assertNotIn("sure", phrases)

    def test_reconsider_sequence_ends_on_the_spaced_token(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        phrases = load_first_tier_phrases(RECOMMENDED)
        sequences = bad_word_token_ids(tokenizer, phrases)
        spaced = tokenizer.encode(" reconsider", add_special_tokens=False)
        self.assertEqual(len(spaced), 1)
        reconsider_seqs = [seq for seq in sequences if seq[-1] == spaced[0]]
        self.assertGreaterEqual(len(reconsider_seqs), 2)
        self.assertTrue(any(seq[-3:] == tokenizer.encode("let me reconsider", add_special_tokens=False) for seq in sequences))


if __name__ == "__main__":
    unittest.main()
