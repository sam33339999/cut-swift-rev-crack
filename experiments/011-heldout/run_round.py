#!/usr/bin/env python3
"""Held-out check: empty prompt, then the 004 sentence. Nothing else changes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.next_rounds import PROMPT_SHORT, sample_round  # noqa: E402
from experiments.server_ctl import reload_server  # noqa: E402

ROUND = Path(__file__).resolve().parent
PROBLEMS = ROUND / "problems.json"


def _make_problems():
    spec = importlib.util.spec_from_file_location("heldout_problems", ROUND / "make_problems.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    payload = module.build_payload()
    PROBLEMS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def heldout_judgement(hard_delta: dict, hard_baseline: dict) -> str:
    base_median = hard_baseline["reasoning_median"]
    median_delta = hard_delta["reasoning_median"]
    correct_delta = hard_delta["n_correct"]
    shortened = median_delta < 0 and abs(median_delta) >= 0.15 * base_median
    if shortened and correct_delta >= -1:
        return (
            "004 的縮短在這份留出題上還在："
            "難題思考中位數至少少 15%，答對沒有少超過 1。"
        )
    if shortened:
        return "004 的縮短在這份留出題上不在：中位數少了，但答對少了超過 1。"
    return "004 的縮短在這份留出題上不在：難題思考中位數沒有少到 15%。"


def baseline_note(hard_delta: dict, hard_baseline: dict) -> str:
    return "這是留出題的空提示對照。縮短與否只看同一份題上的 004 提示。"


def reload_ready(log_path: Path) -> dict:
    body, ready_at, pid = reload_server(log_path)
    payload = json.loads(body)
    model = payload["data"][0]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (log_path.parent / "reload.json").write_text(body, encoding="utf-8")
    return {
        "ready_at": ready_at,
        "model": model["id"],
        "max_model_len": model.get("max_model_len"),
        "pid": pid,
        "stopped_and_reloaded_by": "experiments/011-heldout/run_round.py",
    }


def write_overview(payload: dict, baseline: dict, prompted: dict) -> None:
    lines = [
        "# 011 留出測試集",
        "",
        "題目和開發集 14 題不重複。同一類題、不同數字，標準答案由 `make_problems.py` 算出，並由 `experiments/test_heldout.py` 再用另一段程式核對。這份題沒有拿來回改提示。",
        "",
        "只跑兩輪：空提示，以及 004 的系統提示「推理寫短，想到答案就停。」解碼和 001 相同，seed 101–104。每一輪抽樣前都重載。",
        "",
        "| id | 籃 | 答案 | 核對 |",
        "| --- | --- | --- | --- |",
    ]
    for problem in payload["problems"]:
        lines.append(
            f"| {problem['id']} | {problem['basket']} | {problem['answer']} | {problem['verify']} |"
        )
    lines.extend(
        [
            "",
            "## 兩輪",
            "",
            f"空提示：簡單題 {baseline['easy']['n_correct']}/{baseline['easy']['n']}，"
            f"難題 {baseline['hard']['n_correct']}/{baseline['hard']['n']}，"
            f"難題思考中位數 {baseline['hard']['reasoning_median']:.4f}。",
            "",
            f"004 提示：簡單題 {prompted['easy']['n_correct']}/{prompted['easy']['n']}，"
            f"難題 {prompted['hard']['n_correct']}/{prompted['hard']['n']}，"
            f"難題思考中位數 {prompted['hard']['reasoning_median']:.4f}。",
            "",
            prompted["judgement"],
            "",
        ]
    )
    (ROUND / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROUND / "summary.json").write_text(
        json.dumps({"baseline": baseline, "prompt_short": prompted}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one(spec: dict, problems_ready: int, conclude) -> dict:
    round_dir = ROUND / spec["dirname"]
    ready = reload_ready(round_dir / "logs" / "vllm.log")
    summary = None
    for _ in range(2):
        summary = sample_round(
            spec,
            ready,
            workers=4,
            timeout=600,
            problems_path=PROBLEMS,
            round_dir=round_dir,
            baseline_results=ROUND / "baseline" / "results.jsonl",
            conclude=conclude,
        )
        if summary is not None and summary["n_errors"] == 0 and summary["n_rows"] == problems_ready:
            return summary
    raise RuntimeError(f"{spec['name']} did not finish with {problems_ready} clean rows")


def main() -> None:
    import os

    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    payload = _make_problems()
    expected = len(payload["problems"]) * 4
    baseline_spec = {
        "name": "011-heldout-baseline",
        "dirname": "baseline",
        "method": "8，留出題",
        "rule": "留出題，空系統提示，思考打開。解碼和 001 相同。",
        "system": None,
        "extra_body": {},
        "compare": False,
    }
    prompt_spec = {
        "name": "011-heldout-prompt-short",
        "dirname": "prompt-short",
        "method": "1，留出題",
        "rule": "留出題上只加 004 那一句系統提示。其他不變。",
        "system": PROMPT_SHORT,
        "extra_body": {},
        "compare": False,
    }
    baseline = run_one(baseline_spec, expected, baseline_note)
    prompted = run_one(prompt_spec, expected, heldout_judgement)
    write_overview(payload, baseline, prompted)
    print(prompted["judgement"], flush=True)


if __name__ == "__main__":
    main()
