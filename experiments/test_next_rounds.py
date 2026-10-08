"""The next-round knobs stay one change each, and NoWait does not ban split tokens."""

import unittest

from experiments.next_rounds import (
    NOWAIT_WORDS,
    PROMPT_ANSWER_FIRST,
    PROMPT_ANSWER_FIRST_EN,
    PROMPT_BAN_STALL,
    PROMPT_BAN_STALL_EN,
    PROMPT_CHAIN_OF_DRAFT,
    PROMPT_EASY,
    PROMPT_HARD,
    PROMPT_SHORT,
    PROMPT_SHORT_EN,
    baseline_runner,
    compact_rows,
    nowait_logit_bias,
    round_specs,
    upsert_compare_row,
    write_record,
)


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
                "009-answer-first",
                "010-chain-of-draft",
                "013-ban-stall-words",
                "014-text-budget-256",
                "015-text-budget-512",
                "016-text-budget-1024",
                "017-text-budget-2048",
                "018-budget-by-difficulty",
                "019-prompt-short-en",
                "020-ban-stall-en",
                "021-answer-first-en",
                "022-qwen38-low-template",
                "023-qwen-sharp-terse",
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
        answer_first = specs[6]
        draft = specs[7]
        self.assertEqual(answer_first["system"], PROMPT_ANSWER_FIRST)
        self.assertEqual(answer_first["extra_body"], {})
        self.assertEqual(draft["system"], PROMPT_CHAIN_OF_DRAFT)
        self.assertEqual(draft["extra_body"], {})
        self.assertTrue(draft["split_tokens"])
        self.assertNotIn("bad_words", answer_first["extra_body"])
        self.assertNotIn("logit_bias", draft["extra_body"])
        ban = specs[8]
        self.assertEqual(ban["system"], PROMPT_BAN_STALL)
        self.assertEqual(ban["extra_body"], {})
        budgets = specs[9:13]
        self.assertEqual(
            [spec["user_note"] for spec in budgets],
            [
                "思考不得超過 256 個 token。",
                "思考不得超過 512 個 token。",
                "思考不得超過 1024 個 token。",
                "思考不得超過 2048 個 token。",
            ],
        )
        self.assertTrue(all(spec["extra_body"] == {} and spec["system"] is None for spec in budgets))
        by_basket = specs[13]["system_by_basket"]
        self.assertEqual(by_basket, {"easy": PROMPT_EASY, "hard": PROMPT_HARD})
        self.assertNotIn("thinking_token_budget", budgets[0]["extra_body"])
        by_name = {spec["name"]: spec for spec in specs}
        english = [
            by_name["019-prompt-short-en"],
            by_name["020-ban-stall-en"],
            by_name["021-answer-first-en"],
        ]
        self.assertEqual(
            [spec["system"] for spec in english],
            [PROMPT_SHORT_EN, PROMPT_BAN_STALL_EN, PROMPT_ANSWER_FIRST_EN],
        )
        self.assertTrue(all(spec["extra_body"] == {} for spec in english))
        self.assertNotEqual(english[0]["system"], PROMPT_SHORT)
        self.assertNotEqual(english[2]["system"], PROMPT_ANSWER_FIRST)
        low = by_name["022-qwen38-low-template"]
        self.assertIsNone(low["system"])
        self.assertEqual(low["extra_body"], {})
        self.assertTrue(str(low["chat_template"]).endswith("022-qwen38-low-template/chat_template.jinja"))
        sharp = by_name["023-qwen-sharp-terse"]
        self.assertIsNone(sharp["system"])
        self.assertEqual(sharp["extra_body"], {})
        self.assertTrue(str(sharp["chat_template"]).endswith("023-qwen-sharp-terse/chat_template.jinja"))

    def test_budget_note_stays_before_the_answer_suffix(self):
        prompt = baseline_runner.user_prompt("What is 2+2?", "\n\nANSWER", "思考不得超過 256 個 token。")
        self.assertEqual(prompt, "What is 2+2?\n\n思考不得超過 256 個 token。\n\nANSWER")
        self.assertEqual(baseline_runner.user_prompt("Q", "\nS"), "Q\nS")

    def test_chain_of_draft_record_splits_token_kinds(self):
        import tempfile
        from pathlib import Path

        rows = []
        for basket, gold, reasoning, answer in (
            ("easy", "1", 10, 4),
            ("hard", "2", 100, 8),
        ):
            rows.append(
                {
                    "problem_id": basket,
                    "basket": basket,
                    "seed": 101,
                    "gold": gold,
                    "content": f"ANSWER: {gold}",
                    "truncated": False,
                    "error": None,
                    "reasoning_tokens_api": reasoning,
                    "answer_tokens_api": answer,
                }
            )
        spec = {
            "name": "010-chain-of-draft",
            "method": "4",
            "rule": "分開記",
            "system": PROMPT_CHAIN_OF_DRAFT,
            "extra_body": {},
            "split_tokens": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_record(
                Path(tmp),
                spec,
                rows,
                {"ready_at": "t", "model": "Ornith-1.5-9B", "max_model_len": 20480},
            )
            text = (Path(tmp) / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("思考 token 和答案 token", text)
        self.assertIn("答案中位數", text)
        self.assertEqual(summary["answer_hard"]["median"], 8)
        self.assertEqual(summary["answer_easy"]["mean"], 4)

    def test_compare_row_is_replaced_not_duplicated(self):
        import tempfile
        from pathlib import Path

        spec = {
            "name": "009-answer-first",
            "rule": "x",
            "knob_label": "系統提示「先給答案，再用最多三句驗證。」",
        }
        summary = {
            "easy": {"n_correct": 23, "n": 24, "reasoning_median": 30},
            "hard": {
                "n_correct": 30,
                "n": 32,
                "n_truncated": 0,
                "reasoning_mean": 1000.0,
                "reasoning_median": 700.0,
            },
            "delta_hard": {
                "reasoning_median": -188.5,
                "reasoning_mean": -200.2,
                "n_correct": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "COMPARE.md"
            path.write_text(
                "| 輪 |\n| --- |\n| 008 | old |\n\n| 004 那一句 | 留出 |\n\n後記\n",
                encoding="utf-8",
            )
            upsert_compare_row(path, spec, summary)
            upsert_compare_row(path, spec, summary)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("| 009 |"), 1)
        self.assertIn("先給答案", text)
        self.assertIn("後記", text)
        self.assertLess(text.index("| 009 |"), text.index("| 004 那一句 |"))
        self.assertIn("| 004 那一句 | 留出 |", text)

    def test_compact_keeps_the_successful_row(self):
        rows = compact_rows(
            [
                {"problem_id": "e01", "seed": 101, "error": "timeout"},
                {"problem_id": "e01", "seed": 101, "error": None, "ok": True},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["error"])
        self.assertTrue(rows[0]["ok"])


if __name__ == "__main__":
    unittest.main()
