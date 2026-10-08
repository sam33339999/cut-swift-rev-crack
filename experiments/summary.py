"""Accuracy and reasoning-length stats shared by the round writeup and the recount."""

from __future__ import annotations

import statistics

from experiments.grader import is_correct


def scored_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("error")]


def row_correct(row: dict) -> bool:
    return is_correct(row.get("content"), row["gold"], bool(row.get("truncated")))


def basket_stats(rows: list[dict], basket: str) -> dict:
    chosen = [row for row in scored_rows(rows) if row["basket"] == basket]
    lengths = [
        row["reasoning_tokens_api"]
        for row in chosen
        if row.get("reasoning_tokens_api") is not None
    ]
    n = len(chosen)
    n_correct = sum(1 for row in chosen if row_correct(row))
    n_truncated = sum(1 for row in chosen if row.get("truncated"))
    mean = sum(lengths) / len(lengths) if lengths else None
    median = statistics.median(lengths) if lengths else None
    return {
        "basket": basket,
        "n": n,
        "n_correct": n_correct,
        "n_truncated": n_truncated,
        "accuracy": n_correct / n if n else None,
        "reasoning_mean": mean,
        "reasoning_median": median,
        "n_lengths": len(lengths),
    }


def field_mean_median(rows: list[dict], basket: str, field: str) -> tuple[float | None, float | None]:
    chosen = [row for row in scored_rows(rows) if row["basket"] == basket]
    values = [row[field] for row in chosen if isinstance(row.get(field), (int, float))]
    if not values:
        return None, None
    return sum(values) / len(values), statistics.median(values)


def compare(current: dict, baseline: dict) -> dict:
    return {
        "n_correct": current["n_correct"] - baseline["n_correct"],
        "n_truncated": current["n_truncated"] - baseline["n_truncated"],
        "reasoning_mean": current["reasoning_mean"] - baseline["reasoning_mean"],
        "reasoning_median": current["reasoning_median"] - baseline["reasoning_median"],
    }


def judgement(hard_delta: dict, hard_baseline: dict) -> str:
    """Do not call a small or unsigned length gap a win."""
    base_median = hard_baseline["reasoning_median"]
    median_delta = hard_delta["reasoning_median"]
    correct_delta = hard_delta["n_correct"]
    shortened = median_delta < 0 and abs(median_delta) >= 0.15 * base_median
    if shortened and correct_delta >= -1:
        return (
            "難題思考中位數至少少了 15%，答對筆數沒有少超過 1。"
            "這是開發集上的一次比較，不是測試集結論。"
        )
    if shortened and correct_delta <= -2:
        return "難題中位數變短，但答對少了 2 筆以上。不把這輪記成可用的縮短。"
    if median_delta < 0:
        return "難題中位數有變短，但幅度小於 15%。不把這輪記成有效縮短。"
    return "難題思考中位數沒有變短。不把這輪記成有效縮短。"
