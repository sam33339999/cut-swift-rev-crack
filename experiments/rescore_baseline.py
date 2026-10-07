#!/usr/bin/env python3
"""Re-score baseline 001 with the shipped grader. Prints the basket totals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.grader import extract_answer, is_correct
from experiments.summary import basket_stats
BASELINE = ROOT / "experiments" / "001-baseline-and-mine" / "results.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASELINE
    rows = load(path)
    errors = [row for row in rows if row.get("error")]
    mismatches = []
    for row in rows:
        if row.get("error"):
            continue
        fresh = is_correct(row.get("content"), row["gold"], bool(row.get("truncated")))
        if fresh != bool(row.get("strict_correct")):
            mismatches.append(row["problem_id"] + ":" + str(row["seed"]))
    easy = basket_stats(rows, "easy")
    hard = basket_stats(rows, "hard")
    print(f"file {path}")
    print(f"rows {len(rows)} errors {len(errors)} stored_mismatches {len(mismatches)}")
    print(f"easy {easy['n_correct']}/{easy['n']} truncated {easy['n_truncated']}")
    print(f"hard {hard['n_correct']}/{hard['n']} truncated {hard['n_truncated']}")
    watched = {
        ("h03", 102): False,
        ("h05", 103): False,
    }
    for row in rows:
        key = (row["problem_id"], int(row["seed"]))
        if key not in watched:
            continue
        fresh = is_correct(row.get("content"), row["gold"], bool(row.get("truncated")))
        extracted = extract_answer(row.get("content"))
        print(
            f"{row['problem_id']} seed {row['seed']} correct={fresh} "
            f"extracted={extracted} truncated={bool(row.get('truncated'))}"
        )
        watched[key] = True
    missing = [f"{pid}:{seed}" for (pid, seed), seen in watched.items() if not seen]
    if missing:
        raise SystemExit(f"missing watched rows: {missing}")
    if mismatches:
        raise SystemExit(f"rescore disagrees with stored flags: {mismatches}")
    if easy["n_correct"] != 24 or easy["n"] != 24 or hard["n_correct"] != 30 or hard["n"] != 32:
        raise SystemExit("baseline totals are not 24/24 easy and 30/32 hard")


if __name__ == "__main__":
    main()
