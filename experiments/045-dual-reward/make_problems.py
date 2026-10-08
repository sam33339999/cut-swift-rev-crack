#!/usr/bin/env python3
"""Training items for method 45. New constants, programmatic gold.

These questions are not the 14 dev items and not the 14 held-out items.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("problems.json")
DEV = Path(__file__).resolve().parents[1] / "001-baseline-and-mine" / "problems.json"
HELD = Path(__file__).resolve().parents[1] / "011-heldout" / "problems.json"

SUFFIX = (
    "\n\nThe visible reply, after reasoning, must end with exactly one line:\n"
    "ANSWER: <integer>\n"
    "Do not write anything after that line. "
    "Write the integer without commas, units, or words."
)


def trailing_zeros(n: int) -> int:
    zeros = 0
    power = 5
    while power <= n:
        zeros += n // power
        power *= 5
    return zeros


def smallest_n_with_zeros(target: int) -> int:
    n = 1
    while trailing_zeros(n) < target:
        n += 1
    return n


def count_div_3_or_5_not_7(limit: int) -> int:
    def c(k: int) -> int:
        return limit // k

    return (c(3) + c(5) - c(15)) - (c(21) + c(35) - c(105))


def primes_below(limit: int) -> int:
    primes = [
        n
        for n in range(2, limit)
        if all(n % d != 0 for d in range(2, int(math.sqrt(n)) + 1))
    ]
    return sum(primes)


def consecutive_count(n: int) -> int:
    ways = 0
    length = 2
    while length * (length + 1) // 2 <= n:
        remainder = n - length * (length - 1) // 2
        if remainder > 0 and remainder % length == 0 and remainder // length >= 1:
            ways += 1
        length += 1
    return ways


def largest_prime_factor(n: int) -> int:
    factor = n
    largest = 1
    d = 2
    while d * d <= factor:
        while factor % d == 0:
            largest = d
            factor //= d
        d += 1 if d == 2 else 2
    if factor > 1:
        largest = factor
    return largest


def build_payload() -> dict:
    problems = []
    easy_pairs = [(29, 13), (37, 11), (41, 9), (43, 6), (27, 14), (33, 8), (39, 7), (23, 16)]
    for index, (a, b) in enumerate(easy_pairs, start=1):
        problems.append(
            {
                "id": f"tr{index:02d}",
                "basket": "easy",
                "answer": str(a * b),
                "verify": f"{a}*{b}",
                "question": f"What is {a} times {b}?",
            }
        )
    for index, exp in enumerate((11, 13, 14, 15), start=len(problems) + 1):
        problems.append(
            {
                "id": f"tr{index:02d}",
                "basket": "easy",
                "answer": str(2**exp),
                "verify": f"2**{exp}",
                "question": f"What is 2 to the power {exp}?",
            }
        )
    for index, (a, b) in enumerate(((48, 18), (105, 30), (64, 24), (91, 35)), start=len(problems) + 1):
        problems.append(
            {
                "id": f"tr{index:02d}",
                "basket": "easy",
                "answer": str(math.gcd(a, b)),
                "verify": f"gcd({a},{b})",
                "question": f"What is the greatest common divisor of {a} and {b}?",
            }
        )
    hard_specs = [
        ("smallest n with at least 20 trailing zeros", 20, smallest_n_with_zeros, "Find the smallest positive integer n such that n! has at least 20 trailing zeros."),
        ("smallest n with at least 30 trailing zeros", 30, smallest_n_with_zeros, "Find the smallest positive integer n such that n! has at least 30 trailing zeros."),
        ("smallest n with at least 40 trailing zeros", 40, smallest_n_with_zeros, "Find the smallest positive integer n such that n! has at least 40 trailing zeros."),
        ("inclusion up to 200", 200, count_div_3_or_5_not_7, "How many positive integers n ≤ 200 are divisible by 3 or by 5, but not divisible by 7?"),
        ("inclusion up to 300", 300, count_div_3_or_5_not_7, "How many positive integers n ≤ 300 are divisible by 3 or by 5, but not divisible by 7?"),
        ("inclusion up to 800", 800, count_div_3_or_5_not_7, "How many positive integers n ≤ 800 are divisible by 3 or by 5, but not divisible by 7?"),
        ("2^40 mod 1000", 40, lambda n: pow(2, n, 1000), "Find 2^40 modulo 1000. The answer is the remainder, an integer from 0 to 999."),
        ("2^70 mod 1000", 70, lambda n: pow(2, n, 1000), "Find 2^70 modulo 1000. The answer is the remainder, an integer from 0 to 999."),
        ("primes below 30", 30, primes_below, "Find the sum of all prime numbers strictly less than 30."),
        ("primes below 40", 40, primes_below, "Find the sum of all prime numbers strictly less than 40."),
        ("primes below 80", 80, primes_below, "Find the sum of all prime numbers strictly less than 80."),
        ("consecutive 30", 30, consecutive_count, "A consecutive representation of a positive integer N is a sequence of two or more consecutive positive integers that add up to N. How many consecutive representations does 30 have?"),
        ("consecutive 75", 75, consecutive_count, "A consecutive representation of a positive integer N is a sequence of two or more consecutive positive integers that add up to N. How many consecutive representations does 75 have?"),
        ("consecutive 90", 90, consecutive_count, "A consecutive representation of a positive integer N is a sequence of two or more consecutive positive integers that add up to N. How many consecutive representations does 90 have?"),
        ("factor 3^7+2^7", 3**7 + 2**7, largest_prime_factor, "Find the largest prime factor of 3^7 + 2^7."),
        ("factor 3^9+2^9", 3**9 + 2**9, largest_prime_factor, "Find the largest prime factor of 3^9 + 2^9."),
    ]
    for index, (verify, arg, fn, question) in enumerate(hard_specs, start=len(problems) + 1):
        problems.append(
            {
                "id": f"tr{index:02d}",
                "basket": "hard",
                "answer": str(fn(arg)),
                "verify": verify,
                "question": question,
            }
        )
    questions = [item["question"].strip() for item in problems]
    assert len(questions) == len(set(questions))
    blocked = set()
    for path in (DEV, HELD):
        if path.exists():
            for item in json.loads(path.read_text(encoding="utf-8"))["problems"]:
                blocked.add(item["question"].strip())
    overlap = blocked.intersection(questions)
    assert not overlap, overlap
    assert any(item["basket"] == "easy" for item in problems)
    assert any(item["basket"] == "hard" for item in problems)
    return {
        "version": 1,
        "model": "Ornith-1.5-9B",
        "split": "train-045",
        "suffix": SUFFIX,
        "note": "Method 45 training prompts. Not the dev set and not the held-out set.",
        "problems": problems,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['problems'])} problems to {OUT}")


if __name__ == "__main__":
    main()
