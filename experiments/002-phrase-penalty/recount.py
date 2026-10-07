#!/usr/bin/env python3
"""Recount round 002 from the jsonl and require it to match summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.summary import basket_stats

ROUND = Path(__file__).resolve().parent


def main() -> None:
    rows = [
        json.loads(line)
        for line in (ROUND / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((ROUND / "summary.json").read_text(encoding="utf-8"))
    results = (ROUND / "RESULTS.md").read_text(encoding="utf-8")
    errors = [row for row in rows if row.get("error")]
    easy = basket_stats(rows, "easy")
    hard = basket_stats(rows, "hard")
    ready_at = summary["reload"]["ready_at"]
    early = [
        f"{row['problem_id']}:{row['seed']}"
        for row in rows
        if row.get("request_started") and row["request_started"] < ready_at
    ]
    print(f"rows {len(rows)} errors {len(errors)}")
    print(
        f"easy {easy['n_correct']}/{easy['n']} truncated {easy['n_truncated']} "
        f"mean {easy['reasoning_mean']:.4f} median {easy['reasoning_median']:.4f}"
    )
    print(
        f"hard {hard['n_correct']}/{hard['n']} truncated {hard['n_truncated']} "
        f"mean {hard['reasoning_mean']:.4f} median {hard['reasoning_median']:.4f}"
    )
    print(f"reload {summary['reload']['model']} at {ready_at}")
    print(f"phrases {summary['knob']['phrases']}")
    checks = [
        len(rows) == 56,
        len(errors) == 0,
        easy == summary["easy"],
        hard == summary["hard"],
        not early,
        "bad_words" in results,
        "Ornith-1.5-9B" in results,
        f"{easy['n_correct']}/{easy['n']}" in results,
        f"{hard['n_correct']}/{hard['n']}" in results,
        f"{easy['reasoning_mean']:.4f}" in results,
        f"{hard['reasoning_mean']:.4f}" in results,
        f"{easy['reasoning_median']:.4f}" in results,
        f"{hard['reasoning_median']:.4f}" in results,
    ]
    if not all(checks):
        raise SystemExit(
            f"recount mismatch easy={easy} hard={hard} early={early} summary_easy={summary['easy']}"
        )
    print("recount matches the round record")


if __name__ == "__main__":
    main()
