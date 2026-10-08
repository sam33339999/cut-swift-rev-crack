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
from experiments.summary import basket_stats, compare, field_mean_median, judgement

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
PROMPT_ANSWER_FIRST = "先給答案，再用最多三句驗證。"
PROMPT_CHAIN_OF_DRAFT = "逐步推理，每一步最多五個詞。"
PROMPT_BAN_STALL = "不要寫 wait、alternatively、hmm、再檢查一次、再想一次。"
PROMPT_EASY = "用短推理"
PROMPT_HARD = "可以想完，但不要重複檢查。"
PROMPT_SHORT_EN = "Keep the reasoning short. Stop once you think of the answer."
PROMPT_BAN_STALL_EN = "Do not write wait, alternatively, hmm, check again, or think again."
PROMPT_ANSWER_FIRST_EN = "Give the answer first, then verify it in at most three sentences."
QWEN38_LOW_SENTENCE = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)
QWEN38_LOW_TEMPLATE = ROOT / "experiments" / "022-qwen38-low-template" / "chat_template.jinja"
SHARP_TERSE_TEMPLATE = ROOT / "experiments" / "023-qwen-sharp-terse" / "chat_template.jinja"
TEXT_BUDGETS = (256, 512, 1024, 2048)
NOWAIT_WORDS = ("wait", "Wait", "hmm", "Hmm", "alternatively", "Alternatively")
COMPARE_PATH = ROOT / "experiments" / "COMPARE.md"


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
        {
            "name": "009-answer-first",
            "method": "3",
            "rule": "系統提示只加「先給答案，再用最多三句驗證。」其餘解碼和 001 相同。",
            "system": PROMPT_ANSWER_FIRST,
            "extra_body": {},
            "knob_label": "系統提示「先給答案，再用最多三句驗證。」",
        },
        {
            "name": "010-chain-of-draft",
            "method": "4",
            "rule": "系統提示只加「逐步推理，每一步最多五個詞。」思考 token 和答案 token 分開記。",
            "system": PROMPT_CHAIN_OF_DRAFT,
            "extra_body": {},
            "split_tokens": True,
            "knob_label": "系統提示「逐步推理，每一步最多五個詞。」",
        },
        {
            "name": "013-ban-stall-words",
            "method": "2",
            "rule": "系統提示只加「不要寫 wait、alternatively、hmm、再檢查一次、再想一次。」其餘解碼和 001 相同。",
            "system": PROMPT_BAN_STALL,
            "extra_body": {},
            "knob_label": "系統提示「不要寫 wait、alternatively、hmm、再檢查一次、再想一次。」",
        },
        *[
            {
                "name": f"0{14 + index}-text-budget-{limit}",
                "method": "5",
                "rule": f"題目後面只加「思考不得超過 {limit} 個 token。」不是思考區硬截斷。",
                "system": None,
                "user_note": f"思考不得超過 {limit} 個 token。",
                "extra_body": {},
                "knob_label": f"題目後加「思考不得超過 {limit} 個 token。」",
            }
            for index, limit in enumerate(TEXT_BUDGETS)
        ],
        {
            "name": "018-budget-by-difficulty",
            "method": "6",
            "rule": "簡單題系統提示「用短推理」，難題系統提示「可以想完，但不要重複檢查。」其他不變。",
            "system": None,
            "system_by_basket": {"easy": PROMPT_EASY, "hard": PROMPT_HARD},
            "extra_body": {},
            "knob_label": "簡單題「用短推理」，難題「可以想完，但不要重複檢查。」",
        },
        {
            "name": "019-prompt-short-en",
            "method": "7，對應第 1 條",
            "rule": "第 7 條的英文版，對應 004。系統提示只換成英文那一句，中文版不重跑。",
            "system": PROMPT_SHORT_EN,
            "extra_body": {},
            "knob_label": "英文系統提示「Keep the reasoning short. Stop once you think of the answer.」",
        },
        {
            "name": "020-ban-stall-en",
            "method": "7，對應第 2 條",
            "rule": "第 7 條的英文版，對應 013。系統提示只換成英文那一句。",
            "system": PROMPT_BAN_STALL_EN,
            "extra_body": {},
            "knob_label": "英文系統提示「Do not write wait, alternatively, hmm, check again, or think again.」",
        },
        {
            "name": "021-answer-first-en",
            "method": "7，對應第 3 條",
            "rule": "第 7 條的英文版，對應 009。系統提示只換成英文那一句。",
            "system": PROMPT_ANSWER_FIRST_EN,
            "extra_body": {},
            "knob_label": "英文系統提示「Give the answer first, then verify it in at most three sentences.」",
        },
        {
            "name": "022-qwen38-low-template",
            "method": "模板，不在第 1–62 條",
            "rule": (
                "只換 chat template。在 Ornith 原模板的無工具路徑加上 Qwen3.8 的 low 那一句。"
                "思考預填仍是 <think>。沒有系統提示，解碼和 001 相同。"
            ),
            "system": None,
            "extra_body": {},
            "chat_template": str(QWEN38_LOW_TEMPLATE),
            "knob_label": "Qwen3.8 low 模板句",
        },
        {
            "name": "023-qwen-sharp-terse",
            "method": "模板，不在第 1–62 條",
            "rule": (
                "只換 chat template。在 Ornith 原模板的無工具路徑加上 Qwen Sharp 思考開啟時的 terseness 那一段。"
                "不含 froggeric 的工具修正，也不含 022 的 low 句子。思考預填仍是 <think>。沒有系統提示，解碼和 001 相同。"
            ),
            "system": None,
            "extra_body": {},
            "chat_template": str(SHARP_TERSE_TEMPLATE),
            "knob_label": "Qwen Sharp terseness 模板段",
        },
    ]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_rows(rows: list[dict]) -> list[dict]:
    """Keep the last successful row for each problem and seed.

    A transport error stays only when that key never succeeded. Successful
    rows are not overwritten by an earlier failure.
    """
    order: list[tuple[str, int]] = []
    success: dict[tuple[str, int], dict] = {}
    failure: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["problem_id"], int(row["seed"]))
        if key not in success and key not in failure:
            order.append(key)
        if row.get("error"):
            failure[key] = row
        else:
            success[key] = row
    return [success.get(key) or failure[key] for key in order]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _fmt_delta(value: float) -> str:
    rounded = int(round(value))
    if rounded > 0:
        return f"+{rounded}"
    if rounded < 0:
        return "−" + str(abs(rounded))
    return "0"


def _fmt_median(value: float | None) -> str:
    if value is None:
        return "—"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def compare_row(spec: dict, summary: dict) -> str:
    easy = summary["easy"]
    hard = summary["hard"]
    delta = summary["delta_hard"]
    label = spec.get("knob_label") or spec["rule"]
    easy_cell = f"{easy['n_correct']}/{easy['n']}，中位數 {_fmt_median(easy['reasoning_median'])}"
    relative = (
        f"中位數 {_fmt_delta(delta['reasoning_median'])}，"
        f"平均 {_fmt_delta(delta['reasoning_mean'])}，"
        f"答對 {_fmt_delta(delta['n_correct'])}"
    )
    number = spec["name"].split("-", 1)[0]
    return (
        f"| {number} | {label} | {easy_cell} | {hard['n_correct']}/{hard['n']} | "
        f"{hard['n_truncated']} | {hard['reasoning_mean']:.1f} | "
        f"{hard['reasoning_median']:.1f} | {relative} |"
    )


def _dev_table_row(line: str) -> bool:
    """A dev-set score row looks like '| 009 | ...', not '| 004 那一句 |'."""
    parts = line.split("|")
    if len(parts) < 3:
        return False
    return len(parts[1].strip()) == 3 and parts[1].strip().isdigit()


def upsert_compare_row(path: Path, spec: dict, summary: dict) -> None:
    """Insert or replace this round's line in the dev-set table."""
    row = compare_row(spec, summary)
    number = spec["name"].split("-", 1)[0]
    prefix = f"| {number} |"
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    insert_at = None
    for index, line in enumerate(lines):
        if line.startswith(prefix) and _dev_table_row(line):
            lines[index] = row
            replaced = True
            break
        if _dev_table_row(line):
            insert_at = index + 1
    if not replaced:
        if insert_at is None:
            raise RuntimeError(f"no dev-set table in {path}")
        lines.insert(insert_at, row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_record(
    round_dir: Path,
    spec: dict,
    rows: list[dict],
    ready: dict,
    baseline_results: Path = BASELINE_RESULTS,
    conclude=judgement,
) -> dict:
    easy = basket_stats(rows, "easy")
    hard = basket_stats(rows, "hard")
    base_rows = load_rows(baseline_results)
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
    if spec.get("user_note"):
        knob["user_note"] = spec["user_note"]
    if spec.get("system_by_basket"):
        knob["system_by_basket"] = spec["system_by_basket"]
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
        "judgement": conclude(delta_hard, base_hard),
        "n_rows": len(rows),
        "n_errors": sum(1 for row in rows if row.get("error")),
    }
    if spec.get("split_tokens"):
        for basket in ("easy", "hard"):
            mean, median = field_mean_median(rows, basket, "answer_tokens_api")
            summary[f"answer_{basket}"] = {"mean": mean, "median": median}
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
    if spec.get("split_tokens"):
        lines.extend(
            [
                "",
                "## 思考 token 和答案 token",
                "",
                "思考 token 是 API 的 `reasoning_tokens`。答案 token 是 `completion_tokens` 減去思考 token。比較長度仍只看思考 token。",
                "",
                "| 籃 | 思考平均 | 思考中位數 | 答案平均 | 答案中位數 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for basket, block in (("easy", easy), ("hard", hard)):
            answer = summary[f"answer_{basket}"]
            lines.append(
                f"| {basket} | {block['reasoning_mean']:.4f} | {block['reasoning_median']:.4f} | "
                f"{answer['mean']:.4f} | {answer['median']:.4f} |"
            )
    lines.extend(["", summary["judgement"], ""])
    (round_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def sample_round(
    spec: dict,
    ready: dict,
    workers: int,
    timeout: int,
    problems_path: Path = PROBLEMS,
    round_dir: Path | None = None,
    baseline_results: Path = BASELINE_RESULTS,
    conclude=judgement,
    seeds: list[int] | None = None,
) -> dict | None:
    if ready.get("model") != "Ornith-1.5-9B":
        raise RuntimeError("refusing to score before a reload that serves Ornith-1.5-9B")
    round_dir = round_dir or (ROOT / "experiments" / spec["name"])
    out = round_dir / "results.jsonl"
    suffix, problems = baseline_runner.load_problems(problems_path)
    seed_list = list(seeds or SEEDS)
    have = baseline_runner.done_keys(out)
    expected = len(problems) * len(seed_list)
    jobs = [
        (problem, seed)
        for problem in problems
        for seed in seed_list
        if (problem["id"], seed) not in have
    ]
    print(f"{spec['name']} jobs {len(jobs)} already_done {len(have)}", flush=True)
    if not jobs:
        rows = compact_rows(load_rows(out))
        write_rows(out, rows)
        if len(rows) >= expected and not any(row.get("error") for row in rows):
            return write_record(
                round_dir,
                spec,
                rows,
                ready,
                baseline_results=baseline_results,
                conclude=conclude,
            )
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def system_for(problem: dict) -> str | None:
        by_basket = spec.get("system_by_basket")
        if by_basket:
            return by_basket[problem["basket"]]
        return spec["system"]

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
                knob={
                    "method": spec["method"],
                    "rule": spec["rule"],
                    "extra_body": spec["extra_body"],
                    "system": system_for(problem),
                    "user_note": spec.get("user_note"),
                },
                system=system_for(problem),
                user_note=spec.get("user_note"),
                request_model=spec.get("request_model") or "Ornith-1.5-9B",
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
    rows = compact_rows(load_rows(out))
    write_rows(out, rows)
    if any(row.get("error") for row in rows) or len(rows) < expected:
        return None
    return write_record(
        round_dir,
        spec,
        rows,
        ready,
        baseline_results=baseline_results,
        conclude=conclude,
    )


def _complete(rows: list[dict], expected: int = 56) -> bool:
    if any(row.get("error") for row in rows):
        return False
    keys = {(row["problem_id"], int(row["seed"])) for row in rows}
    return len(rows) == expected and len(keys) == expected


def run_all(workers: int = 4, timeout: int = 600, only: list[str] | None = None) -> None:
    import os

    from transformers import AutoTokenizer

    # Prompt rounds must not inherit a logits processor from an earlier shell.
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    os.environ.pop("CHAT_TEMPLATE", None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    specs = round_specs(tokenizer)
    if only:
        wanted = set(only)
        known = {spec["name"] for spec in specs}
        missing = wanted - known
        if missing:
            raise SystemExit(f"unknown round: {sorted(missing)}")
        specs = [spec for spec in specs if spec["name"] in wanted]
    for spec in specs:
        round_dir = ROOT / "experiments" / spec["name"]
        existing = load_rows(round_dir / "results.jsonl")
        if _complete(existing):
            print(f"{spec['name']} already complete, skip", flush=True)
            continue
        if spec.get("chat_template"):
            os.environ["CHAT_TEMPLATE"] = spec["chat_template"]
        else:
            os.environ.pop("CHAT_TEMPLATE", None)
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
        summary = None
        for attempt in range(2):
            summary = sample_round(spec, ready, workers, timeout)
            if summary is not None and summary["n_errors"] == 0 and summary["n_rows"] == 56:
                break
            print(f"{spec['name']} attempt {attempt + 1} left transport errors", flush=True)
        if summary is None or summary["n_errors"] or summary["n_rows"] != 56:
            raise RuntimeError(f"{spec['name']} did not finish with 56 clean rows")
        if spec.get("compare", True):
            upsert_compare_row(COMPARE_PATH, spec, summary)
        print(f"{spec['name']} record written", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Comma-separated round directory names.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    chosen = [item for item in args.only.split(",") if item]
    run_all(workers=args.workers, timeout=args.timeout, only=chosen or None)
