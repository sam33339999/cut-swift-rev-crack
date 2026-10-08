"""Held-out items are computed twice and do not repeat the dev set."""

import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments" / "001-baseline-and-mine" / "problems.json"
MODULE = ROOT / "experiments" / "011-heldout" / "make_problems.py"


def load_module():
    spec = importlib.util.spec_from_file_location("heldout_problems", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def trailing_zeros(n: int) -> int:
    zeros = 0
    power = 5
    while power <= n:
        zeros += n // power
        power *= 5
    return zeros


def smallest_n(target: int) -> int:
    n = 1
    while trailing_zeros(n) < target:
        n += 1
    return n


def squareful(n: int) -> bool:
    d = 2
    rest = n
    while d * d <= rest:
        if rest % (d * d) == 0:
            return True
        if rest % d == 0:
            while rest % d == 0:
                rest //= d
        d += 1 if d == 2 else 2
    return False


class HeldoutTest(unittest.TestCase):
    def test_items_do_not_repeat_the_dev_set_and_golds_recompute(self):
        payload = load_module().build_payload()
        problems = payload["problems"]
        dev = json.loads(DEV.read_text(encoding="utf-8"))["problems"]
        dev_questions = {item["question"].strip() for item in dev}
        dev_ids = {item["id"] for item in dev}
        self.assertTrue({"easy", "hard"} <= {item["basket"] for item in problems})
        self.assertEqual(len(problems), 14)
        by_id = {}
        for item in problems:
            self.assertNotIn(item["question"].strip(), dev_questions)
            self.assertNotIn(item["id"], dev_ids)
            by_id[item["id"]] = item["answer"]

        self.assertEqual(by_id["te01"], str(19 * 31))
        self.assertEqual(by_id["te02"], str(2**12))
        self.assertEqual(by_id["te03"], str(math.gcd(96, 36)))
        self.assertEqual(by_id["te04"], "10")
        self.assertEqual(by_id["te05"], str(50 * 51 // 2))
        self.assertEqual(by_id["te06"], str(pow(13, 3, 7)))

        n50 = smallest_n(50)
        self.assertLess(trailing_zeros(n50 - 1), 50)
        self.assertGreaterEqual(trailing_zeros(n50), 50)
        self.assertEqual(by_id["th01"], str(n50))
        self.assertEqual(
            by_id["th02"],
            str((500 // 3 + 500 // 5 - 500 // 15) - (500 // 21 + 500 // 35 - 500 // 105)),
        )
        triples = 0
        for a in range(1, 120):
            for b in range(a + 1, 120 - a):
                c = 120 - a - b
                if b < c and a * a + b * b == c * c:
                    triples += 1
        self.assertEqual(by_id["th03"], str(triples))
        composite = 3**8 + 2**8
        factor = composite
        largest = 1
        d = 2
        while d * d <= factor:
            while factor % d == 0:
                largest = d
                factor //= d
            d += 1 if d == 2 else 2
        if factor > 1:
            largest = factor
        self.assertEqual(by_id["th04"], str(largest))
        pair_sum = sum(n for n in range(1, 60) if squareful(n) and squareful(n + 1))
        self.assertEqual(by_id["th05"], str(pair_sum))
        self.assertEqual(by_id["th06"], str(pow(2, 90, 1000)))
        ways = 0
        length = 2
        while length * (length + 1) // 2 <= 45:
            remainder = 45 - length * (length - 1) // 2
            if remainder > 0 and remainder % length == 0 and remainder // length >= 1:
                ways += 1
            length += 1
        self.assertEqual(by_id["th07"], str(ways))
        primes = [
            n
            for n in range(2, 60)
            if all(n % d != 0 for d in range(2, int(math.sqrt(n)) + 1))
        ]
        self.assertEqual(by_id["th08"], str(sum(primes)))


if __name__ == "__main__":
    unittest.main()
