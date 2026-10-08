"""Method 45 reward: wrong scores 0, and a bare number is not a formed answer."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARD = ROOT / "experiments" / "045-dual-reward" / "reward.py"
MAKE = ROOT / "experiments" / "045-dual-reward" / "make_problems.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Reward45Test(unittest.TestCase):
    def test_wrong_is_zero_and_review_starts_at_the_formed_gold(self):
        reward = load(REWARD, "reward45")
        rows = [
            {
                "correct": False,
                "gold": "12",
                "reasoning": "So the answer is 12. Let me recheck the total.",
                "reasoning_tokens": 10,
            },
            {
                "correct": True,
                "gold": "12",
                "reasoning": "n = 12 is only a factor. So the answer is 12. extra",
                "reasoning_tokens": 20,
            },
            {
                "correct": True,
                "gold": "12",
                "reasoning": "So the answer is 12.",
                "reasoning_tokens": 10,
            },
        ]
        scores = reward.group_rewards(rows, tokenize=list)
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], 0.0)
        cut = reward.first_gold_end("n = 12 is only a factor. So the answer is 12.", "12")
        self.assertIsNotNone(cut)
        self.assertIn("answer is 12", "n = 12 is only a factor. So the answer is 12."[:cut])

    def test_training_questions_do_not_repeat_eval_sets(self):
        made = load(MAKE, "make45")
        payload = made.build_payload()
        self.assertGreaterEqual(len(payload["problems"]), 16)
        self.assertTrue({"easy", "hard"} <= {item["basket"] for item in payload["problems"]})


class MaskAndProbeTest(unittest.TestCase):
    def test_advantage_is_zero_on_answer_tokens_and_open_inside_think(self):
        reward = load(REWARD, "reward45mask")
        pieces = ["17", " × ", "23", " = ", "391", "</think>", "ANSWER", ":", " 391"]
        mask = reward.advantage_mask(pieces)
        self.assertEqual(mask[pieces.index("</think>")], 1.0)
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[pieces.index("ANSWER")], 0.0)
        self.assertEqual(mask[pieces.index(":")], 0.0)
        self.assertEqual(mask[-1], 0.0)
        self.assertEqual(reward.advantage_mask(["ANS"]), [0.0])

    def test_over_long_sequence_keeps_the_post_answer_span(self):
        reward = load(REWARD, "reward45span")
        token_ids = list(range(20))
        kept = reward.trainable_post_answer_span(token_ids, prompt_len=4, first_correct_index=6, max_len=10)
        self.assertIsNotNone(kept)
        self.assertEqual(kept, list(range(10, 20)))
        self.assertIsNone(
            reward.trainable_post_answer_span(token_ids, prompt_len=4, first_correct_index=None, max_len=10)
        )
        self.assertEqual(
            reward.trainable_post_answer_span(list(range(5)), prompt_len=1, first_correct_index=1, max_len=10),
            list(range(5)),
        )

    def test_probe_discards_the_ans_loop_and_keeps_a_normal_answer(self):
        reward = load(REWARD, "reward45probe")
        loop = "17 × 23 = 391" + "ANS" * 40
        self.assertEqual(reward.probe_decision(loop, content="", truncated=True), "discard")
        self.assertTrue(reward.is_ans_collapse(loop, "", True))
        normal = "17 × 23 = 391.\nANSWER: 391"
        self.assertEqual(reward.probe_decision(normal, content="ANSWER: 391", truncated=False), "keep")
        self.assertEqual(reward.PROBE_PERIOD_S, 30)
        self.assertEqual(reward.probe_decision.__code__.co_varnames[:3], ("text", "content", "truncated"))


if __name__ == "__main__":
    unittest.main()
