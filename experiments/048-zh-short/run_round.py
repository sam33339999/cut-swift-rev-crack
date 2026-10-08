#!/usr/bin/env python3
"""Sample Traditional Chinese math, train the short-thinking LoRA, then score.

Rollouts use the base model. The new adapter is served as zhshort.
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
from experiments.server_ctl import reload_server, stop_listeners  # noqa: E402
import experiments.next_rounds as rounds  # noqa: E402

ROUND = Path(__file__).resolve().parent
DEV = ROOT / "experiments/001-baseline-and-mine/problems.json"
ADAPTER = ROUND / "adapter"
SEEDS = [301, 302, 303, 304]
LORA_NAME = "zhshort"
BASE_WEIGHT = Path("/content/models/ornith-1.5-9B/model.safetensors.index.json")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_math(problems: list[dict], suffix: str) -> None:
    out = ROUND / "rollouts.jsonl"
    have = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                have.add((row["problem_id"], int(row["seed"])))
    jobs = [(problem, seed) for problem in problems for seed in SEEDS if (problem["id"], seed) not in have]
    print(f"zh rollout jobs {len(jobs)} already {len(have)}", flush=True)
    if not jobs:
        return
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LORA_MODULES", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    reload_server(ROUND / "logs" / "rollout-vllm.log")
    for problem, seed in jobs:
        row = rounds.baseline_runner.call_one(
            "http://127.0.0.1:8000/v1/chat/completions",
            problem,
            suffix,
            seed,
            8192,
            600,
            experiment="048-zh-train",
            request_model="Ornith-1.5-9B",
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
            f"zh {problem['id']} seed={seed} correct={int(row['correct'])} reason_tok={row['reasoning_tokens']}",
            flush=True,
        )


def serve() -> dict:
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    os.environ["VLLM_LORA_MODULES"] = f"{LORA_NAME}={ADAPTER}"
    os.environ["VLLM_MAX_LORA_RANK"] = "16"
    body, ready_at, pid = reload_server(ROUND / "logs" / "vllm.log")
    payload = json.loads(body)
    model = payload["data"][0]
    return {
        "ready_at": ready_at,
        "model": model["id"],
        "max_model_len": model.get("max_model_len"),
        "pid": pid,
        "stopped_and_reloaded_by": "experiments/048-zh-short/run_round.py",
    }


def chinese_turns(probe) -> None:
    turns = [
        ("用三句話說明 Redis 和 Memcached 怎麼選。", "那快取失效策略你會怎麼做？"),
        ("請用繁體中文解釋什麼是冪等。", "那付款重送時你會怎麼擋？"),
    ]
    out = ROUND / "zh-chat.jsonl"
    for index, (first, second) in enumerate(turns, start=1):
        first_row = rounds.baseline_runner.call_one(
            "http://127.0.0.1:8000/v1/chat/completions",
            {"id": f"chat{index}a", "basket": "chat", "answer": "0", "question": first},
            "",
            101,
            512,
            180,
            experiment="048-zh-chat",
            request_model=LORA_NAME,
        )
        if probe.discarded or reward_decision(first_row):
            raise SystemExit("probe discarded the adapter on a Chinese turn")
        reply = first_row.get("content") or ""
        row = second_turn(first, reply, second, probe)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"first": first_row, "second": row}, ensure_ascii=False) + "\n")
        content = row.get("content") or ""
        print(
            f"zh-chat {index} content_chars={len(content)} trunc={row.get('truncated')} "
            f"reason_tok={row.get('reasoning_tokens_api')}",
            flush=True,
        )
        if not content.strip() or row.get("truncated"):
            raise SystemExit(f"Chinese turn {index} did not produce a finished answer")


def reward_decision(row: dict) -> bool:
    spec = importlib.util.spec_from_file_location(
        "reward45zh", ROOT / "experiments/045-dual-reward/reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.probe_decision(row.get("reasoning") or "", row.get("content") or "", bool(row.get("truncated"))) == "discard"


def second_turn(first: str, reply: str, second: str, probe) -> dict:
    import json as _json
    import time
    import urllib.request

    body = {
        "model": LORA_NAME,
        "messages": [
            {"role": "user", "content": first},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": second},
        ],
        "max_tokens": 512,
        "seed": 101,
        "chat_template_kwargs": {"enable_thinking": True},
        **rounds.baseline_runner.SAMPLING,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = _json.load(response)
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    if reward_decision({"reasoning": reasoning, "content": content, "truncated": choice.get("finish_reason") == "length"}):
        probe.discarded = True
    return {
        "content": content,
        "reasoning": reasoning,
        "truncated": choice.get("finish_reason") == "length",
        "reasoning_tokens_api": ((data.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> None:
    before = BASE_WEIGHT.stat().st_mtime
    maker = _load(ROUND / "make_problems.py", "zh_problems")
    payload = maker.build_payload()
    (ROUND / "problems.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample_math(payload["problems"], payload["suffix"])
    if not (ADAPTER / "adapter_model.safetensors").exists():
        print("stopping server before Chinese LoRA", flush=True)
        stop_listeners()
        subprocess.run([sys.executable, str(ROUND / "train.py")], check=True)
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model mtime changed")
    ready = serve()
    retry = _load(ROOT / "experiments/045-dual-reward/run_retry.py", "run_retry_for_zh")
    probe = retry.CollapseProbe(ROUND / "probe.log")
    probe.start()
    try:
        suffix, problems = rounds.baseline_runner.load_problems(DEV)
        canary = next(item for item in problems if item["id"] == "e01")
        sample = rounds.baseline_runner.call_one(
            "http://127.0.0.1:8000/v1/chat/completions",
            canary,
            suffix,
            101,
            512,
            180,
            experiment="048-canary",
            request_model=LORA_NAME,
        )
        print(
            f"canary correct={sample.get('strict_correct')} trunc={sample.get('truncated')} "
            f"chars={len(sample.get('reasoning') or '')}",
            flush=True,
        )
        if reward_decision(sample):
            raise SystemExit("probe discarded the adapter on the canary")
        summary = rounds.sample_round(
            {
                "name": "048-zh-short",
                "method": "45+zh",
                "rule": "繁體可評分題用第 45 條獎勵，繁體對話只做下一個字。優勢只在思考區。",
                "system": None,
                "extra_body": {},
                "request_model": LORA_NAME,
                "knob_label": "繁體中文加短思考 LoRA",
            },
            ready,
            workers=4,
            timeout=600,
            problems_path=DEV,
            round_dir=ROUND / "dev",
        )
        if summary is None or summary["n_errors"] or summary["n_rows"] != 56:
            raise SystemExit("dev score did not finish cleanly")
        print(summary["judgement"], flush=True)
        if summary["judgement"].startswith("難題思考中位數至少少了 15%"):
            rounds.upsert_compare_row(
                rounds.COMPARE_PATH,
                {"name": "048-zh-short", "knob_label": "繁體中文加短思考 LoRA"},
                summary,
            )
        chinese_turns(probe)
        (ROUND / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        probe.close()
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model mtime changed after scoring")


if __name__ == "__main__":
    main()
