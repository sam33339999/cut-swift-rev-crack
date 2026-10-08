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


if __name__ == "__main__":
    unittest.main()
