#!/usr/bin/env python3
"""Traditional Chinese verifiable items for the short-thinking LoRA.

Numbers and wording do not repeat the English dev set, the held-out set,
or the earlier English method-45 training questions.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("problems.json")
ROOT = Path(__file__).resolve().parents[2]
BLOCKED = [
    ROOT / "experiments/001-baseline-and-mine/problems.json",
    ROOT / "experiments/011-heldout/problems.json",
    ROOT / "experiments/045-dual-reward/problems.json",
]
SUFFIX = (
    "\n\n推理結束後，可見回答的最後一行必須正好是：\n"
    "ANSWER: <整數>\n"
    "這一行後面不要再寫任何字。整數不要加逗號、單位或文字。"
)


def _math():
    path = ROOT / "experiments/045-dual-reward/make_problems.py"
    spec = importlib.util.spec_from_file_location("math45", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_payload() -> dict:
    math45 = _math()
    problems = []
    pairs = [(26, 19), (34, 12), (38, 15), (44, 8), (28, 17), (32, 15), (46, 5), (52, 4)]
    for index, (a, b) in enumerate(pairs, start=1):
        problems.append(
            {
                "id": f"zh{index:02d}",
                "basket": "easy",
                "answer": str(a * b),
                "verify": f"{a}*{b}",
                "question": f"{a} 乘以 {b} 是多少？",
            }
        )
    start = len(problems) + 1
    for offset, exp in enumerate((16, 17, 18, 19)):
        problems.append(
            {
                "id": f"zh{start + offset:02d}",
                "basket": "easy",
                "answer": str(2**exp),
                "verify": f"2**{exp}",
                "question": f"2 的 {exp} 次方是多少？",
            }
        )
    hard = [
        (25, math45.smallest_n_with_zeros, "n! 至少有 25 個結尾的 0 時，最小的正整數 n 是多少？"),
        (35, math45.smallest_n_with_zeros, "n! 至少有 35 個結尾的 0 時，最小的正整數 n 是多少？"),
        (250, math45.count_div_3_or_5_not_7, "在 1 到 250 之間，有多少個正整數可以被 3 或 5 整除，但不能被 7 整除？"),
        (350, math45.count_div_3_or_5_not_7, "在 1 到 350 之間，有多少個正整數可以被 3 或 5 整除，但不能被 7 整除？"),
        (35, math45.primes_below, "嚴格小於 35 的所有質數，加起來是多少？"),
        (70, math45.primes_below, "嚴格小於 70 的所有質數，加起來是多少？"),
        (36, math45.consecutive_count, "把 36 寫成至少兩個連續正整數的和，有幾種寫法？"),
        (48, math45.consecutive_count, "把 48 寫成至少兩個連續正整數的和，有幾種寫法？"),
    ]
    start = len(problems) + 1
    for offset, (arg, fn, question) in enumerate(hard):
        problems.append(
            {
                "id": f"zh{start + offset:02d}",
                "basket": "hard",
                "answer": str(fn(arg)),
                "verify": f"{fn.__name__}({arg})",
                "question": question,
            }
        )
    assert len({item["question"] for item in problems}) == len(problems)
    assert len({item["id"] for item in problems}) == len(problems)
    blocked = set()
    for path in BLOCKED:
        if path.exists():
            for item in json.loads(path.read_text(encoding="utf-8"))["problems"]:
                blocked.add(item["question"].strip())
    overlap = blocked.intersection(item["question"].strip() for item in problems)
    assert not overlap, overlap
    assert all(item["answer"] == str(int(item["answer"])) for item in problems)
    assert math.prod(pairs[0]) == int(problems[0]["answer"]) or pairs[0][0] * pairs[0][1] == int(problems[0]["answer"])
    return {
        "version": 1,
        "split": "train-zh-short",
        "language": "zh-Hant",
        "suffix": SUFFIX,
        "note": "Traditional Chinese verifiable items. Not the English dev or held-out sets.",
        "problems": problems,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['problems'])} to {OUT}")


if __name__ == "__main__":
    main()
