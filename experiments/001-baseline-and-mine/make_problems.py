#!/usr/bin/env python3
"""Build the dev set and prove each gold answer before any model call."""

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


def count_div_3_or_5_not_7(limit: int = 1000) -> int:
    def c(k: int) -> int:
        return limit // k

    union = c(3) + c(5) - c(15)
    also_7 = c(21) + c(35) - c(105)
    return union - also_7


def pythagorean_products(total: int = 1000) -> list[tuple[int, int, int, int]]:
    found = []
    for a in range(1, total):
        for b in range(a + 1, total - a):
            c = total - a - b
            if c <= b:
                continue
            if a * a + b * b == c * c:
                found.append((a, b, c, a * b * c))
    return found


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


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def is_squareful(n: int) -> bool:
    """True when some perfect square greater than 1 divides n."""
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


def squareful_pair_sum(limit: int = 100) -> tuple[list[int], int]:
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


def primes_below(limit: int) -> list[int]:
    sieve = [True] * limit
    sieve[0:2] = [False, False]
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit:step] = [False] * len(range(start, limit, step))
    return [i for i, keep in enumerate(sieve) if keep]


def main() -> None:
    assert math.gcd(84, 30) == 6
    assert n_divisors(36) == 9
    assert 100 * 101 // 2 == 5050
    assert pow(17, 3, 5) == 3
    assert 17 * 23 == 391
    assert 2**10 == 1024

    n_zeros = smallest_n_with_zeros(100)
    assert n_zeros == 405
    assert trailing_zeros(404) == 99
    assert trailing_zeros(405) == 100

    assert count_div_3_or_5_not_7() == 401

    triples = pythagorean_products()
    assert triples == [(200, 375, 425, 31_875_000)]

    composite = 3**12 + 2**12
    assert composite == 535_537
    factors = prime_factors(composite)
    largest = max(factors)
    assert largest == 5521
    assert is_prime(5521)
    assert math.prod(factors) == composite

    squareful_ns, squareful_sum = squareful_pair_sum()
    assert squareful_ns == [8, 24, 27, 44, 48, 49, 63, 75, 80, 98, 99]
    assert squareful_sum == 615

    last_three = pow(2, 100, 1000)
    ways = consecutive_representations(100)
    for way in ways:
        assert sum(way) == 100 and len(way) >= 2
    prime_sum = sum(primes_below(100))
    # Cross-check the sieve against trial division.
    trial = [
        n
        for n in range(2, 100)
        if all(n % d != 0 for d in range(2, int(n**0.5) + 1))
    ]
    assert trial == primes_below(100)

    problems = [
        {
            "id": "e01",
            "basket": "easy",
            "answer": "391",
            "verify": "17*23",
            "question": "What is 17 times 23?",
        },
        {
            "id": "e02",
            "basket": "easy",
            "answer": "1024",
            "verify": "2**10",
            "question": "What is 2 to the power 10?",
        },
        {
            "id": "e03",
            "basket": "easy",
            "answer": "6",
            "verify": "gcd(84, 30)",
            "question": "What is the greatest common divisor of 84 and 30?",
        },
        {
            "id": "e04",
            "basket": "easy",
            "answer": "9",
            "verify": "divisor count of 36 = (2+1)*(2+1)",
            "question": "How many positive divisors does 36 have?",
        },
        {
            "id": "e05",
            "basket": "easy",
            "answer": "5050",
            "verify": "100*101/2",
            "question": "What is the sum of the integers from 1 through 100?",
        },
        {
            "id": "e06",
            "basket": "easy",
            "answer": "3",
            "verify": "pow(17, 3, 5)",
            "question": "What is the remainder when 17 cubed is divided by 5?",
        },
        {
            "id": "h01",
            "basket": "hard",
            "answer": str(n_zeros),
            "verify": "smallest n with v_5(n!) >= 100; v_5(404)=99, v_5(405)=100",
            "question": (
                "Find the smallest positive integer n such that n! "
                "has at least 100 trailing zeros."
            ),
        },
        {
            "id": "h02",
            "basket": "hard",
            "answer": "401",
            "verify": "|A∪B| - |(A∪B)∩C| for multiples of 3, 5, 7 up to 1000",
            "question": (
                "How many positive integers n ≤ 1000 are divisible by 3 or by 5, "
                "but not divisible by 7?"
            ),
        },
        {
            "id": "h03",
            "basket": "hard",
            "answer": "31875000",
            "verify": "only triple is 200, 375, 425; product 31875000",
            "question": (
                "There is exactly one Pythagorean triple of positive integers "
                "(a, b, c) with a < b < c, a squared plus b squared equal to c squared, "
                "and a + b + c = 1000. Find the product a*b*c."
            ),
        },
        {
            "id": "h04",
            "basket": "hard",
            "answer": str(largest),
            "verify": f"3**12+2**12={composite}, factors={factors}",
            "question": "Find the largest prime factor of 3^12 + 2^12.",
        },
        {
            "id": "h05",
            "basket": "hard",
            "answer": str(squareful_sum),
            "verify": f"n in {squareful_ns}",
            "question": (
                "Call a positive integer squareful when some perfect square "
                "greater than 1 divides it. Find the sum of all positive integers "
                "n < 100 such that both n and n+1 are squareful."
            ),
        },
        {
            "id": "h06",
            "basket": "hard",
            "answer": str(last_three),
            "verify": "pow(2, 100, 1000)",
            "question": (
                "Find 2^100 modulo 1000. "
                "The answer is the remainder, an integer from 0 to 999."
            ),
        },
        {
            "id": "h07",
            "basket": "hard",
            "answer": str(len(ways)),
            "verify": "starts " + "; ".join("+".join(map(str, w)) for w in ways),
            "question": (
                "A consecutive representation of a positive integer N is a sequence "
                "of two or more consecutive positive integers that add up to N. "
                "How many consecutive representations does 100 have?"
            ),
        },
        {
            "id": "h08",
            "basket": "hard",
            "answer": str(prime_sum),
            "verify": f"{len(trial)} primes below 100, trial division matches the sieve",
            "question": "Find the sum of all prime numbers strictly less than 100.",
        },
    ]
    baskets = {p["basket"] for p in problems}
    assert baskets == {"easy", "hard"}
    assert len({p["id"] for p in problems}) == len(problems)
    payload = {
        "version": 1,
        "model": "Ornith-1.5-9B",
        "prompt_language": "en",
        "suffix": SUFFIX,
        "problems": problems,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for p in problems:
        print(f"{p['id']} {p['basket']:4} {p['answer']:>10}  {p['verify']}")


if __name__ == "__main__":
    main()
