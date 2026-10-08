"""The Thinking-Cap gate: no wrong answer outscores a correct one."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "experiments" / "050-thinking-cap" / "rewards.py"


def load():
    spec = importlib.util.spec_from_file_location("thinking_cap_rewards", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ThinkingCapRewardTest(unittest.TestCase):
    def test_wrong_scores_below_every_correct_answer(self):
        reward = load()

        def tokenize(text: str) -> list[str]:
            return list(text)

        wrong = reward.score_completion("17 times 23 is 999</think>\nANSWER: 999", "391", tokenize, lam=0.05)
        correct_long = reward.score_completion(
            ("step " * 500) + "</think>\nANSWER: 391",
            "391",
            tokenize,
            lam=0.05,
        )
        correct_short = reward.score_completion("391</think>\nANSWER: 391", "391", tokenize, lam=0.05)
        self.assertFalse(wrong["correct"])
        self.assertTrue(correct_long["correct"])
        self.assertTrue(correct_short["correct"])
        self.assertLess(wrong["total"], correct_long["total"])
        self.assertLess(wrong["total"], correct_short["total"])
        self.assertLess(correct_long["total"], correct_short["total"])
        self.assertEqual(wrong["length_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
