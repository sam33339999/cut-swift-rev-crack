"""Further single-knob rounds on the same dev set.

Each round reloads vLLM before it scores anything. The template has no
separate low/medium/high thinking budget, so method 9's effort levels are
not distinct from the baseline; only enable_thinking false changes the prompt.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.server_ctl import reload_server
from experiments.summary import basket_stats, compare, judgement

BASELINE_DIR = ROOT / "experiments" / "001-baseline-and-mine"
PROBLEMS = BASELINE_DIR / "problems.json"
BASELINE_RESULTS = BASELINE_DIR / "results.jsonl"
MODEL = "/content/models/ornith-1.5-9B"
SEEDS = [101, 102, 103, 104]

_baseline = importlib.util.spec_from_file_location(
    "baseline_runner", BASELINE_DIR / "run_baseline.py"
)
baseline_runner = importlib.util.module_from_spec(_baseline)
assert _baseline.loader is not None
_baseline.loader.exec_module(baseline_runner)

PROMPT_SHORT = "推理寫短，想到答案就停。"
NOWAIT_WORDS = ("wait", "Wait", "hmm", "Hmm", "alternatively", "Alternatively")


def nowait_logit_bias(tokenizer, penalty: float = -2.0) -> dict[str, float]:
    """Whole-token forms only, including the leading-space id. Split pieces are skipped."""
    bias: dict[str, float] = {}
    for word in NOWAIT_WORDS:
        for text in (word, " " + word):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                bias[str(ids[0])] = penalty
    return bias


def round_specs(tokenizer) -> list[dict]:
    reconsider = tokenizer.encode(" reconsider", add_special_tokens=False)
    if len(reconsider) != 1:
        raise RuntimeError(f"expected one token for ' reconsider', got {reconsider}")
    return [
        {
            "name": "003-no-thinking",
            "method": "55，以及第 9 條在這個模板上唯一不同的一檔",
            "rule": (
                "chat_template_kwargs enable_thinking=false。"
                "模板會預填空的思考區。low/medium/high 不會改這個模板，所以不另開一輪。"
            ),
            "system": None,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {
            "name": "004-prompt-short",
            "method": "1",
            "rule": "系統提示只加文件第 1 條那一句，其餘解碼不變。",
            "system": PROMPT_SHORT,
            "extra_body": {},
        },
        {
            "name": "005-logit-reconsider-2",
            "method": "12",
            "rule": "只對帶前置空格的 reconsider 這一個 token 加 logit -2。不是第 002 輪的片語硬封鎖。",
            "system": None,
            "extra_body": {"logit_bias": {str(reconsider[0]): -2.0}},
        },
        {
            "name": "006-nowait-2",
            "method": "11 的第一組詞，懲罰用第 12 條的 -2",
            "rule": "wait / hmm / alternatively 的完整 token，含前置空格和大小寫，logit -2。切開的 hmm 不罰。",
            "system": None,
            "extra_body": {"logit_bias": nowait_logit_bias(tokenizer)},
        },
        {
            "name": "007-think-budget-2048",
            "method": "10",
            "rule": "thinking_token_budget=2048。思考到 2048 就收束，答案區仍可生成。8192 和這次的輸出上限重合，不另跑。",
            "system": None,
            "extra_body": {"thinking_token_budget": 2048},
        },
        {
            "name": "008-think-budget-512",
            "method": "10",
            "rule": "thinking_token_budget=512。和第 007 輪只差上限。",
            "system": None,
            "extra_body": {"thinking_token_budget": 512},
        },
    ]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_record(round_dir: Path, spec: dict, rows: list[dict], ready: dict) -> None:
    easy = basket_stats(rows, "easy")
    hard = basket_stats(rows, "hard")
    base_rows = load_rows(BASELINE_RESULTS)
    base_easy = basket_stats(base_rows, "easy")
    base_hard = basket_stats(base_rows, "hard")
    delta_easy = compare(easy, base_easy)
    delta_hard = compare(hard, base_hard)
    knob = {
        "method": spec["method"],
        "rule": spec["rule"],
        "system": spec["system"],
        "extra_body": spec["extra_body"],
    }
    summary = {
        "experiment": spec["name"],
        "knob": knob,
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
    (round_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {spec['name']}",
        "",
        f"方法：{spec['method']}。",
        "",
        "這一輪在抽樣前先由本程式關掉 port 8000 上的 vLLM，再用 `start_server.sh` 重新載入 Ornith-1.5-9B。",
        f"重新載入完成時間：{ready['ready_at']}。models id：{ready['model']}。max_model_len：{ready['max_model_len']}。",
        "",
        "## 旋鈕",
        "",
        spec["rule"],
        "",
        "```json",
        json.dumps(knob, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分數",
        "",
        "正確的定義和基線相同：沒有截斷，而且可見回答最後一個 `ANSWER:` 整數等於標準答案。截斷留在分母裡。",
        "",
        "| 籃 | 答對 | 截斷 | 思考平均 | 思考中位數 | 相對基線答對 | 相對基線平均 | 相對基線中位數 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, block, delta in (("easy", easy, delta_easy), ("hard", hard, delta_hard)):
        lines.append(
            f"| {label} | {block['n_correct']}/{block['n']} | {block['n_truncated']} | "
            f"{block['reasoning_mean']:.4f} | {block['reasoning_median']:.4f} | "
            f"{delta['n_correct']:+d} | {delta['reasoning_mean']:+.4f} | "
            f"{delta['reasoning_median']:+.4f} |"
        )
    lines.extend(["", summary["judgement"], ""])
    (round_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def sample_round(spec: dict, ready: dict, workers: int, timeout: int) -> None:
    if ready.get("model") != "Ornith-1.5-9B":
        raise RuntimeError("refusing to score before a reload that serves Ornith-1.5-9B")
    round_dir = ROOT / "experiments" / spec["name"]
    out = round_dir / "results.jsonl"
    suffix, problems = baseline_runner.load_problems(PROBLEMS)
    have = baseline_runner.done_keys(out)
    jobs = [
        (problem, seed)
        for problem in problems
        for seed in SEEDS
        if (problem["id"], seed) not in have
    ]
    print(f"{spec['name']} jobs {len(jobs)} already_done {len(have)}", flush=True)
    if not jobs:
        rows = load_rows(out)
        if len(rows) >= 56 and not any(row.get("error") for row in rows):
            write_record(round_dir, spec, rows, ready)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    knob = {"method": spec["method"], "rule": spec["rule"], "extra_body": spec["extra_body"]}

    def write(row: dict) -> None:
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def work(problem: dict, seed: int) -> dict:
        started = time.time()
        try:
            return baseline_runner.call_one(
                "http://127.0.0.1:8000/v1/chat/completions",
                problem,
                suffix,
                seed,
                8192,
                timeout,
                experiment=spec["name"],
                extra_body=spec["extra_body"] or None,
                knob=knob,
                system=spec["system"],
            )
        except Exception as exc:  # noqa: BLE001 — stored, then the batch continues
            return baseline_runner.error_row(
                problem,
                seed,
                8192,
                exc,
                started,
                experiment=spec["name"],
                knob=knob,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, problem, seed) for problem, seed in jobs]
        for future in as_completed(futures):
            row = future.result()
            write(row)
            print(
                f"{spec['name']} {row['problem_id']} seed={row['seed']} "
                f"correct={int(bool(row.get('strict_correct')))} "
                f"trunc={int(bool(row.get('truncated')))} "
                f"reason_tok={row.get('reasoning_tokens_api')} "
                f"extracted={row.get('extracted_answer')} err={row.get('error')}",
                flush=True,
            )
    rows = load_rows(out)
    write_record(round_dir, spec, rows, ready)


def run_all(workers: int = 4, timeout: int = 600) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    specs = round_specs(tokenizer)
    for spec in specs:
        round_dir = ROOT / "experiments" / spec["name"]
        existing = load_rows(round_dir / "results.jsonl")
        if len(existing) >= 56 and not any(row.get("error") for row in existing):
            print(f"{spec['name']} already complete, skip", flush=True)
            continue
        body, ready_at, pid = reload_server(round_dir / "logs" / "vllm.log")
        payload = json.loads(body)
        model = payload["data"][0]
        ready = {
            "ready_at": ready_at,
            "model": model["id"],
            "max_model_len": model.get("max_model_len"),
            "pid": pid,
            "stopped_and_reloaded_by": "experiments/next_rounds.py",
        }
        (round_dir / "logs").mkdir(parents=True, exist_ok=True)
        (round_dir / "logs" / "reload.json").write_text(body, encoding="utf-8")
        sample_round(spec, ready, workers, timeout)
        print(f"{spec['name']} record written", flush=True)


if __name__ == "__main__":
    run_all()
