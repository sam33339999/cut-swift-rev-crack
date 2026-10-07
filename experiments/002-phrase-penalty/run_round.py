#!/usr/bin/env python3
"""Reload vLLM, then score the dev set with the first-tier phrase penalty.

--reload-only stops, starts, and writes nothing but the models response to stdout.
A scored request is refused until that reload has returned Ornith-1.5-9B.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.penalty import bad_word_token_ids, load_first_tier_phrases
from experiments.server_ctl import reload_server
from experiments.summary import basket_stats, compare, judgement

ROUND = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "experiments" / "001-baseline-and-mine"
PROBLEMS = BASELINE_DIR / "problems.json"
RECOMMENDED = BASELINE_DIR / "recommended.json"
BASELINE_RESULTS = BASELINE_DIR / "results.jsonl"
MODEL = "/content/models/ornith-1.5-9B"

_baseline = importlib.util.spec_from_file_location(
    "baseline_runner", BASELINE_DIR / "run_baseline.py"
)
baseline_runner = importlib.util.module_from_spec(_baseline)
assert _baseline.loader is not None
_baseline.loader.exec_module(baseline_runner)

READY: dict | None = None


def refuse_unless_reloaded() -> dict:
    if READY is None:
        raise RuntimeError("refusing to score before this run reloads Ornith-1.5-9B")
    return READY


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_record(
    rows: list[dict],
    phrases: list[str],
    sequences: list[list[int]],
    ready: dict,
) -> None:
    easy = basket_stats(rows, "easy")
    hard = basket_stats(rows, "hard")
    base_rows = load_rows(BASELINE_RESULTS)
    base_easy = basket_stats(base_rows, "easy")
    base_hard = basket_stats(base_rows, "hard")
    delta_easy = compare(easy, base_easy)
    delta_hard = compare(hard, base_hard)
    summary = {
        "experiment": "002-phrase-penalty",
        "knob": {
            "mechanism": "vllm bad_words",
            "rule": (
                "The completing token of each first-tier phrase is set to logit -inf "
                "only after that phrase's preceding tokens. A leading-space sequence is "
                "included when it changes the first token. Sentence-start capitals are "
                "separate strings."
            ),
            "phrases": phrases,
            "token_sequences": sequences,
        },
        "reload": ready,
        "easy": easy,
        "hard": hard,
        "baseline_easy": base_easy,
        "baseline_hard": base_hard,
        "delta_easy": delta_easy,
        "delta_hard": delta_hard,
        "judgement": judgement(delta_hard, base_hard),
        "n_rows": len(rows),
        "n_errors": sum(1 for row in rows if row.get("error")),
    }
    (ROUND / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 002 片語懲罰",
        "",
        "這一輪在抽樣前先由本程式關掉 port 8000 上的 vLLM，再用 `start_server.sh` 重新載入 Ornith-1.5-9B。",
        f"重新載入完成時間：{ready['ready_at']}。models id：{ready['model']}。max_model_len：{ready['max_model_len']}。",
        "",
        "## 旋鈕",
        "",
        "只動解碼。用 vLLM 的 `bad_words` 封鎖 `recommended.json` 第一檔的四個片語。前綴對上之後，最後一個 token 的 logit 設為負無窮。沒有改提示詞，沒有改抽樣參數，沒有壓 `wait` 或單獨的 `sure`。",
        "",
        "片語：`" + "`, `".join(phrases) + "`。",
        "",
        "實際 token 序列：",
        "",
        "```json",
        json.dumps(sequences),
        "```",
        "",
        "## 分數",
        "",
        "正確的定義和基線相同：沒有截斷，而且可見回答最後一個 `ANSWER:` 整數等於標準答案。截斷留在分母裡。",
        "",
        "| 籃 | 答對 | 截斷 | 思考平均 | 思考中位數 | 相對基線答對 | 相對基線平均 | 相對基線中位數 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, block, delta in (
        ("easy", easy, delta_easy),
        ("hard", hard, delta_hard),
    ):
        lines.append(
            f"| {label} | {block['n_correct']}/{block['n']} | {block['n_truncated']} | "
            f"{block['reasoning_mean']:.4f} | {block['reasoning_median']:.4f} | "
            f"{delta['n_correct']:+d} | {delta['reasoning_mean']:+.4f} | "
            f"{delta['reasoning_median']:+.4f} |"
        )
    lines.extend(
        [
            "",
            summary["judgement"],
            "",
            "基線數字由 `experiments/001-baseline-and-mine/results.jsonl` 用同一支統計函式重算，不是手抄。",
            "",
        ]
    )
    (ROUND / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def sample(url: str, out: Path, phrases: list[str], workers: int, timeout: int) -> None:
    refuse_unless_reloaded()
    suffix, problems = baseline_runner.load_problems(PROBLEMS)
    seeds = [101, 102, 103, 104]
    have = baseline_runner.done_keys(out)
    jobs = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if (problem["id"], seed) not in have
    ]
    knob = {"mechanism": "vllm bad_words", "phrases": phrases}
    extra = {"bad_words": phrases}
    print(f"jobs {len(jobs)} already_done {len(have)} out {out}", flush=True)
    if not jobs:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def write(row: dict) -> None:
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def work(problem: dict, seed: int) -> dict:
        started = time.time()
        try:
            return baseline_runner.call_one(
                url,
                problem,
                suffix,
                seed,
                8192,
                timeout,
                experiment="002-phrase-penalty",
                extra_body=extra,
                knob=knob,
            )
        except Exception as exc:  # noqa: BLE001 — stored, then the batch continues
            return baseline_runner.error_row(
                problem,
                seed,
                8192,
                exc,
                started,
                experiment="002-phrase-penalty",
                knob=knob,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, problem, seed) for problem, seed in jobs]
        for future in as_completed(futures):
            row = future.result()
            write(row)
            print(
                f"{row['problem_id']} seed={row['seed']} "
                f"correct={int(bool(row.get('strict_correct')))} "
                f"trunc={int(bool(row.get('truncated')))} "
                f"reason_tok={row.get('reasoning_tokens_api')} "
                f"extracted={row.get('extracted_answer')} "
                f"err={row.get('error')}",
                flush=True,
            )


def main() -> None:
    global READY
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    args = parser.parse_args()
    log_path = ROUND / "logs" / "vllm.log"
    body, ready_at, pid = reload_server(log_path)
    payload = json.loads(body)
    model = payload["data"][0]
    READY = {
        "ready_at": ready_at,
        "model": model["id"],
        "max_model_len": model.get("max_model_len"),
        "pid": pid,
        "stopped_and_reloaded_by": "experiments/002-phrase-penalty/run_round.py",
    }
    (ROUND / "logs").mkdir(parents=True, exist_ok=True)
    (ROUND / "logs" / "reload.json").write_text(body, encoding="utf-8")
    if args.reload_only:
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
        return
    phrases = load_first_tier_phrases(RECOMMENDED)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sequences = bad_word_token_ids(tokenizer, phrases)
    out = ROUND / "results.jsonl"
    sample(args.url, out, phrases, args.workers, args.timeout)
    rows = load_rows(out)
    write_record(rows, phrases, sequences, READY)
    print(
        f"wrote {ROUND / 'RESULTS.md'} rows={len(rows)} "
        f"errors={sum(1 for row in rows if row.get('error'))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
