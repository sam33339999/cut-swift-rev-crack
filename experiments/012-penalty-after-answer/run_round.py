#!/usr/bin/env python3
"""Penalize the 002 phrases only after a formed answer appears in thinking.

The server is reloaded with the V2 logits processor. Requests that omit
vllm_xargs.penalty_after_answer are unchanged. This script never sends
bad_words.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.next_rounds import (  # noqa: E402
    COMPARE_PATH,
    load_rows,
    sample_round,
    upsert_compare_row,
)
from experiments.penalty import load_first_tier_phrases  # noqa: E402
from experiments.penalty_after_answer import FLAG, RECOMMENDED, formed_answer_seen  # noqa: E402
from experiments.server_ctl import reload_server  # noqa: E402

ROUND = Path(__file__).resolve().parent
PROCESSOR = "experiments.penalty_after_answer:PenaltyAfterAnswerProcessor"


def main() -> None:
    os.environ["VLLM_LOGITS_PROCESSORS"] = PROCESSOR
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    phrases = load_first_tier_phrases(RECOMMENDED)
    spec = {
        "name": "012-penalty-after-answer",
        "method": "14",
        "rule": (
            "思考區出現成形答案之前不罰。成形答案指「the answer is」、「answer:」、"
            "「\\boxed{」或「答案是」後面緊接整數，而且只看 </think> 之前的文字。"
            "出現之後才擋 002 的四個片語（含句首大寫）的下一個 token："
            + "、".join(phrases)
            + "。不是全程 bad_words。"
        ),
        "system": None,
        "extra_body": {"vllm_xargs": {FLAG: 1}},
        "knob_label": "答完才罰四個空轉片語",
        "compare": True,
    }
    if "bad_words" in spec["extra_body"]:
        raise RuntimeError("012 must not send bad_words")
    log_path = ROUND / "logs" / "vllm.log"
    body, ready_at, pid = reload_server(log_path)
    payload = json.loads(body)
    model = payload["data"][0]
    ready = {
        "ready_at": ready_at,
        "model": model["id"],
        "max_model_len": model.get("max_model_len"),
        "pid": pid,
        "stopped_and_reloaded_by": "experiments/012-penalty-after-answer/run_round.py",
        "logits_processors": PROCESSOR,
    }
    (ROUND / "logs").mkdir(parents=True, exist_ok=True)
    (ROUND / "logs" / "reload.json").write_text(body, encoding="utf-8")
    summary = None
    for _ in range(2):
        summary = sample_round(spec, ready, workers=4, timeout=600)
        if summary is not None and summary["n_errors"] == 0 and summary["n_rows"] == 56:
            break
    if summary is None or summary["n_errors"] or summary["n_rows"] != 56:
        raise RuntimeError("012 did not finish with 56 clean rows")
    rows = load_rows(ROUND / "results.jsonl")
    armed = 0
    scored = 0
    for row in rows:
        if row.get("error"):
            continue
        scored += 1
        if formed_answer_seen(row.get("reasoning") or ""):
            armed += 1
    summary["answer_armed"] = armed
    summary["answer_scored"] = scored
    (ROUND / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    extra = (
        "\n## 成形答案有沒有出現\n\n"
        f"用和線上相同的規則回看思考區：{armed}/{scored} 筆出現成形答案，罰分有機會打開。\n"
        "沒有出現的那幾筆，這輪和基線的差別只會是抽樣波動。\n"
    )
    results = ROUND / "RESULTS.md"
    results.write_text(results.read_text(encoding="utf-8") + extra, encoding="utf-8")
    upsert_compare_row(COMPARE_PATH, spec, summary)
    print(summary["judgement"], flush=True)
    print(f"armed {armed}/{scored}", flush=True)


if __name__ == "__main__":
    main()
