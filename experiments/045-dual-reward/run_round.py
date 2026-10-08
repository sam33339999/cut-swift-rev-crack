#!/usr/bin/env python3
"""Method 45: sample a new training set, one GRPO-style LoRA update, then score.

Training prompts do not include the Sharp terseness text. After training, the
dev set is scored twice: LoRA alone, then LoRA plus the 023 template.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.grader import is_correct  # noqa: E402
from experiments.next_rounds import sample_round  # noqa: E402
from experiments.server_ctl import reload_server, stop_listeners  # noqa: E402
import experiments.next_rounds as rounds  # noqa: E402

ROUND = Path(__file__).resolve().parent
DEV = ROOT / "experiments" / "001-baseline-and-mine" / "problems.json"
SHARP = ROOT / "experiments" / "023-qwen-sharp-terse" / "chat_template.jinja"
ADAPTER = ROUND / "adapter"
SEEDS = [201, 202, 203, 204]
LORA_NAME = "m45"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clear_server_overrides() -> None:
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LORA_MODULES", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)


def reload_ready(log_path: Path, stopped_by: str) -> dict:
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
        "stopped_and_reloaded_by": stopped_by,
    }


def sample_rollouts(problems: list[dict], suffix: str) -> None:
    out = ROUND / "rollouts.jsonl"
    have = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                have.add((row["problem_id"], int(row["seed"])))
    jobs = [
        (problem, seed)
        for problem in problems
        for seed in SEEDS
        if (problem["id"], seed) not in have
    ]
    print(f"rollout jobs {len(jobs)} already {len(have)}", flush=True)
    if not jobs:
        return
    clear_server_overrides()
    reload_ready(ROUND / "logs" / "rollout-vllm.log", "experiments/045-dual-reward/run_round.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    for problem, seed in jobs:
        row = rounds.baseline_runner.call_one(
            "http://127.0.0.1:8000/v1/chat/completions",
            problem,
            suffix,
            seed,
            8192,
            600,
            experiment="045-train",
        )
        if row.get("error"):
            row["correct"] = False
            row["reasoning_tokens"] = 0
        else:
            row["correct"] = is_correct(row.get("content"), row["gold"], bool(row.get("truncated")))
            row["reasoning_tokens"] = row.get("reasoning_tokens_api") or 0
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"rollout {problem['id']} seed={seed} correct={int(row['correct'])} "
            f"reason_tok={row['reasoning_tokens']}",
            flush=True,
        )


def score_dev(dirname: str, spec_name: str, label: str, use_sharp: bool) -> dict:
    clear_server_overrides()
    os.environ["VLLM_LORA_MODULES"] = f"{LORA_NAME}={ADAPTER}"
    os.environ["VLLM_MAX_LORA_RANK"] = "16"
    if use_sharp:
        os.environ["CHAT_TEMPLATE"] = str(SHARP)
    spec = {
        "name": spec_name,
        "method": "45",
        "rule": label,
        "system": None,
        "extra_body": {},
        "request_model": LORA_NAME,
        "knob_label": label,
        "compare": True,
    }
    ready = reload_ready(ROUND / dirname / "logs" / "vllm.log", "experiments/045-dual-reward/run_round.py")
    summary = None
    for _ in range(2):
        summary = sample_round(
            spec,
            ready,
            workers=4,
            timeout=600,
            problems_path=DEV,
            round_dir=ROUND / dirname,
        )
        if summary is not None and summary["n_errors"] == 0 and summary["n_rows"] == 56:
            rounds.upsert_compare_row(rounds.COMPARE_PATH, spec, summary)
            return summary
    raise RuntimeError(f"{dirname} did not finish with 56 clean rows")


def main() -> None:
    maker = _load(ROUND / "make_problems.py", "make45")
    payload = maker.build_payload()
    (ROUND / "problems.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sample_rollouts(payload["problems"], payload["suffix"])
    if not (ADAPTER / "adapter_model.safetensors").exists():
        print("stopping server before LoRA", flush=True)
        stop_listeners()
        subprocess.run(
            [sys.executable, str(ROUND / "train_lora.py")],
            check=True,
        )
    else:
        print("adapter already present, skip training", flush=True)
    lora = score_dev(
        "lora-only",
        "045-lora-only",
        "第 45 條 LoRA，訓練提示沒有 Sharp",
        use_sharp=False,
    )
    stacked = score_dev(
        "lora-plus-sharp",
        "046-lora-plus-sharp",
        "第 45 條 LoRA 再加 023 的 Sharp 模板",
        use_sharp=True,
    )
    text = (
        "# 045 雙獎勵 LoRA\n\n"
        "訓練題是新的常數，不是開發集，也不是留出題。訓練提示沒有 Sharp。\n"
        "一組 4 條，兩輪更新，KL 係數 0.01。答錯獎勵是 0。\n\n"
        f"只掛 LoRA：難題 {lora['hard']['n_correct']}/{lora['hard']['n']}，"
        f"思考中位數 {lora['hard']['reasoning_median']:.4f}。\n\n"
        f"LoRA 加 Sharp：難題 {stacked['hard']['n_correct']}/{stacked['hard']['n']}，"
        f"思考中位數 {stacked['hard']['reasoning_median']:.4f}。\n\n"
        "Sharp 單獨的那格是 023，不重跑。\n\n"
        f"{lora['judgement']}\n\n"
        f"疊加 Sharp：{stacked['judgement']}\n"
    )
    (ROUND / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
