"""The next-round knobs stay one change each, and NoWait does not ban split tokens."""

import unittest

from experiments.next_rounds import NOWAIT_WORDS, PROMPT_SHORT, nowait_logit_bias, round_specs


class FakeTokenizer:
    def __init__(self):
        self.table = {
            " reconsider": [37781],
            "wait": [11158],
            " wait": [3655],
            "Wait": [13784],
            " Wait": [13428],
            "hmm": [71, 3693],
            " hmm": [84485],
            "Hmm": [77264],
            " Hmm": [85152],
            "alternatively": [39075, 7638],
            " alternatively": [66073],
            "Alternatively": [88842],
            " Alternatively": [37201],
        }

    def encode(self, text, add_special_tokens=False):
        if text not in self.table:
            raise KeyError(text)
        return list(self.table[text])


class NextRoundsTest(unittest.TestCase):
    def test_prompt_is_the_documented_sentence(self):
        self.assertEqual(PROMPT_SHORT, "推理寫短，想到答案就停。")

    def test_nowait_bias_keeps_only_whole_tokens(self):
        bias = nowait_logit_bias(FakeTokenizer(), penalty=-2.0)
        self.assertEqual(set(bias), {"11158", "3655", "13784", "13428", "84485", "77264", "85152", "66073", "88842", "37201"})
        self.assertTrue(all(value == -2.0 for value in bias.values()))
        self.assertNotIn("71", bias)

    def test_each_round_changes_one_thing(self):
        specs = round_specs(FakeTokenizer())
        names = [spec["name"] for spec in specs]
        self.assertEqual(
            names,
            [
                "003-no-thinking",
                "004-prompt-short",
                "005-logit-reconsider-2",
                "006-nowait-2",
                "007-think-budget-2048",
                "008-think-budget-512",
            ],
        )
        no_thinking = specs[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        self.assertIs(no_thinking, False)
        self.assertEqual(specs[1]["system"], PROMPT_SHORT)
        self.assertEqual(specs[1]["extra_body"], {})
        self.assertEqual(specs[2]["extra_body"], {"logit_bias": {"37781": -2.0}})
        self.assertEqual(specs[4]["extra_body"], {"thinking_token_budget": 2048})
        self.assertEqual(specs[5]["extra_body"], {"thinking_token_budget": 512})
        self.assertEqual(NOWAIT_WORDS, ("wait", "Wait", "hmm", "Hmm", "alternatively", "Alternatively"))


if __name__ == "__main__":
    unittest.main()
