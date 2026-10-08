#!/usr/bin/env python3
"""Held-out items. Same task families as the dev set, different constants.

Gold answers are computed here and checked again by experiments/test_heldout.py.
This file does not read the dev-set questions, and nothing here is used to
edit a prompt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("problems.json")

SUFFIX = (
    "\n\nThe visible reply, after reasoning, must end with exactly one line:\n"
    "ANSWER: <integer>\n"
    "Do not write anything after that line. "
    "Write the integer without commas, units, or words."
)


def n_divisors(n: int) -> int:
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1 if i * i == n else 2
        i += 1
    return count


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

    union = c(3) + c(5) - c(15)
    also_7 = c(21) + c(35) - c(105)
    return union - also_7


def is_squareful(n: int) -> bool:
    d = 2
    rest = n
    while d * d <= rest:
        if rest % d == 0:
            count = 0
            while rest % d == 0:
                rest //= d
                count += 1
            if count >= 2:
                return True
        d += 1 if d == 2 else 2
    return False


def squareful_pair_sum(limit: int) -> tuple[list[int], int]:
    ns = [n for n in range(1, limit) if is_squareful(n) and is_squareful(n + 1)]
    return ns, sum(ns)


def consecutive_representations(n: int) -> list[list[int]]:
    ways = []
    length = 2
    while length * (length + 1) // 2 <= n:
        remainder = n - length * (length - 1) // 2
        if remainder > 0 and remainder % length == 0:
            start = remainder // length
            if start >= 1:
                ways.append(list(range(start, start + length)))
        length += 1
    return ways


def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def primes_below(limit: int) -> list[int]:
    return [
        n
        for n in range(2, limit)
        if all(n % d != 0 for d in range(2, int(n**0.5) + 1))
    ]


def pythagorean_triple_count(total: int) -> int:
    count = 0
    for a in range(1, total):
        for b in range(a + 1, total - a):
            c = total - a - b
            if c <= b:
                continue
            if a * a + b * b == c * c:
                count += 1
    return count


def build_payload() -> dict:
    assert 19 * 31 == 589
    assert 2**12 == 4096
    assert math.gcd(96, 36) == 12
    assert n_divisors(48) == 10
    assert 50 * 51 // 2 == 1275
    assert pow(13, 3, 7) == 6

    n_zeros = smallest_n_with_zeros(50)
    assert trailing_zeros(n_zeros - 1) < 50
    assert trailing_zeros(n_zeros) >= 50

    count_500 = count_div_3_or_5_not_7(500)
    assert count_500 == (500 // 3 + 500 // 5 - 500 // 15) - (
        500 // 21 + 500 // 35 - 500 // 105
    )

    composite = 3**8 + 2**8
    factors = prime_factors(composite)
    largest = max(factors)
    assert math.prod(factors) == composite
    assert largest > 1

    squareful_ns, squareful_sum = squareful_pair_sum(60)
    last_three = pow(2, 90, 1000)
    ways = consecutive_representations(45)
    for way in ways:
        assert sum(way) == 45 and len(way) >= 2
    triple_count = pythagorean_triple_count(120)
    prime_sum = sum(primes_below(60))

    problems = [
        {
            "id": "te01",
            "basket": "easy",
            "answer": "589",
            "verify": "19*31",
            "question": "What is 19 times 31?",
        },
        {
            "id": "te02",
            "basket": "easy",
            "answer": "4096",
            "verify": "2**12",
            "question": "What is 2 to the power 12?",
        },
        {
            "id": "te03",
            "basket": "easy",
            "answer": "12",
            "verify": "gcd(96, 36)",
            "question": "What is the greatest common divisor of 96 and 36?",
        },
        {
            "id": "te04",
            "basket": "easy",
            "answer": "10",
            "verify": "divisor count of 48 = (4+1)*(1+1)",
            "question": "How many positive divisors does 48 have?",
        },
        {
            "id": "te05",
            "basket": "easy",
            "answer": "1275",
            "verify": "50*51/2",
            "question": "What is the sum of the integers from 1 through 50?",
        },
        {
            "id": "te06",
            "basket": "easy",
            "answer": "6",
            "verify": "pow(13, 3, 7)",
            "question": "What is the remainder when 13 cubed is divided by 7?",
        },
        {
            "id": "th01",
            "basket": "hard",
            "answer": str(n_zeros),
            "verify": "smallest n with v_5(n!) >= 50",
            "question": (
                "Find the smallest positive integer n such that n! "
                "has at least 50 trailing zeros."
            ),
        },
        {
            "id": "th02",
            "basket": "hard",
            "answer": str(count_500),
            "verify": "inclusion-exclusion up to 500",
            "question": (
                "How many positive integers n ≤ 500 are divisible by 3 or by 5, "
                "but not divisible by 7?"
            ),
        },
        {
            "id": "th03",
            "basket": "hard",
            "answer": str(triple_count),
            "verify": "count of a<b<c, a+b+c=120, a^2+b^2=c^2",
            "question": (
                "How many Pythagorean triples of positive integers (a, b, c) "
                "satisfy a < b < c, a squared plus b squared equal to c squared, "
                "and a + b + c = 120?"
            ),
        },
        {
            "id": "th04",
            "basket": "hard",
            "answer": str(largest),
            "verify": f"3**8+2**8={composite}, factors={factors}",
            "question": "Find the largest prime factor of 3^8 + 2^8.",
        },
        {
            "id": "th05",
            "basket": "hard",
            "answer": str(squareful_sum),
            "verify": f"n in {squareful_ns}",
            "question": (
                "Call a positive integer squareful when some perfect square "
                "greater than 1 divides it. Find the sum of all positive integers "
                "n < 60 such that both n and n+1 are squareful."
            ),
        },
        {
            "id": "th06",
            "basket": "hard",
            "answer": str(last_three),
            "verify": "pow(2, 90, 1000)",
            "question": (
                "Find 2^90 modulo 1000. "
                "The answer is the remainder, an integer from 0 to 999."
            ),
        },
        {
            "id": "th07",
            "basket": "hard",
            "answer": str(len(ways)),
            "verify": "starts " + "; ".join("+".join(map(str, way)) for way in ways),
            "question": (
                "A consecutive representation of a positive integer N is a sequence "
                "of two or more consecutive positive integers that add up to N. "
                "How many consecutive representations does 45 have?"
            ),
        },
        {
            "id": "th08",
            "basket": "hard",
            "answer": str(prime_sum),
            "verify": f"{len(primes_below(60))} primes below 60",
            "question": "Find the sum of all prime numbers strictly less than 60.",
        },
    ]
    assert {item["basket"] for item in problems} == {"easy", "hard"}
    assert len({item["id"] for item in problems}) == len(problems)
    return {
        "version": 1,
        "model": "Ornith-1.5-9B",
        "split": "heldout",
        "prompt_language": "en",
        "suffix": SUFFIX,
        "note": (
            "Same families as experiments/001-baseline-and-mine/problems.json, "
            "different constants. Not used to choose prompts or knobs."
        ),
        "problems": problems,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for problem in payload["problems"]:
        print(f"{problem['id']} {problem['basket']:4} {problem['answer']:>8}  {problem['verify']}")


if __name__ == "__main__":
    main()
