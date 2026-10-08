#!/usr/bin/env python3
"""Train the bounded Thinking-Cap LoRA, probe it, then score the dev set.

The failed 045 adapter and the 049 adapter are not loaded. General chat stays
on the base model name unless this round's own notes say otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.grader import extract_answer, is_correct  # noqa: E402
from experiments.next_rounds import (  # noqa: E402
    COMPARE_PATH,
    SEEDS,
    upsert_compare_row,
    write_record,
)
from experiments.server_ctl import reload_server, stop_listeners  # noqa: E402
import experiments.next_rounds as rounds  # noqa: E402

ROUND = Path(__file__).resolve().parent
DEV = ROOT / "experiments" / "001-baseline-and-mine" / "problems.json"
ADAPTER = ROUND / "adapter"
BASE_WEIGHT = Path("/content/models/ornith-1.5-9B/model.safetensors.index.json")
LORA_NAME = "m50cap"
BASE_NAME = "Ornith-1.5-9B"
URL = "http://127.0.0.1:8000/v1/chat/completions"
PY = os.environ.get("TRAIN_PYTHON", "/content/.venv/bin/python")
ZH_ITEMS = [
    {"id": "s1", "turns": ["我國國慶日是幾月幾號？用繁體中文一句話。"]},
    {
        "id": "s2",
        "turns": ["把這句改短，用繁體中文一句話：我們目前正在進行一項非常重要的系統升級工作。"],
    },
    {
        "id": "m4",
        "turns": ["用三句話說明 Redis 和 Memcached 怎麼選。", "那快取失效策略你會怎麼做？"],
    },
]


def _reward45():
    spec = importlib.util.spec_from_file_location("reward45", ROOT / "experiments" / "045-dual-reward" / "reward.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _reward45()
_TOKENIZER = None


def _count_tokens(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("/content/models/ornith-1.5-9B", trust_remote_code=True)
    if not text:
        return 0
    return len(_TOKENIZER(text, add_special_tokens=False).input_ids)


def wait_gpu_free(limit_mib: int = 2000, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        used = int(probe.stdout.strip().splitlines()[0])
        if used < limit_mib:
            return
        time.sleep(2)
    raise SystemExit(f"GPU still above {limit_mib} MiB")


class CollapseProbe:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.inflight: dict[str, dict[str, str]] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.discarded = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self._record("start")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.wait(reward.PROBE_PERIOD_S):
            self._record("tick")
            if self.discarded:
                break

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def set_partial(self, key: str, text: str, content: str) -> None:
        with self.lock:
            self.inflight[key] = {"text": text, "content": content}
        if reward.probe_decision(text, content, False) == "discard":
            self.discarded = True

    def clear(self, key: str) -> None:
        with self.lock:
            self.inflight.pop(key, None)

    def _record(self, kind: str) -> None:
        with self.lock:
            text = "\n".join(item["text"] for item in self.inflight.values())
            content = "\n".join(item["content"] for item in self.inflight.values())
        decision = reward.probe_decision(text, content, False)
        if decision == "discard":
            self.discarded = True
        row = {"t": time.time(), "kind": kind, "decision": decision, "chars": len(text)}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"probe {kind} decision={decision} chars={len(text)}", flush=True)


def stream_one(problem: dict, suffix: str, seed: int, probe: CollapseProbe, experiment: str) -> dict:
    prompt = problem["question"] + suffix
    body = {
        "model": LORA_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        **rounds.baseline_runner.SAMPLING,
    }
    key = f"{problem['id']}-{seed}"
    reasoning: list[str] = []
    content: list[str] = []
    usage = {}
    finish = None
    started = time.time()
    probe.set_partial(key, "", "")
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                if probe.discarded:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                piece = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece:
                    reasoning.append(piece)
                if delta.get("content"):
                    content.append(delta["content"])
                probe.set_partial(key, "".join(reasoning), "".join(content))
                if probe.discarded:
                    break
    except Exception as exc:  # noqa: BLE001
        probe.clear(key)
        return rounds.baseline_runner.error_row(problem, seed, 8192, exc, started, experiment=experiment)
    probe.clear(key)
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    details = (usage.get("completion_tokens_details") or {}) if usage else {}
    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _count_tokens(reasoning_text)
    completion_tokens = usage.get("completion_tokens") if usage else None
    answer_tokens = None
    if reasoning_tokens is not None and completion_tokens is not None:
        answer_tokens = completion_tokens - reasoning_tokens
    truncated = finish == "length" or probe.discarded
    return {
        "schema_version": 1,
        "experiment": experiment,
        "problem_id": problem["id"],
        "basket": problem["basket"],
        "seed": seed,
        "gold": problem["answer"],
        "prompt": prompt,
        "sampling": {**rounds.baseline_runner.SAMPLING, "max_tokens": 8192, "enable_thinking": True},
        "request_started": rounds.baseline_runner.now(),
        "request_finished": rounds.baseline_runner.now(),
        "wall_s": round(time.time() - started, 3),
        "http_status": 200,
        "error": None,
        "finish_reason": finish,
        "usage": usage,
        "reasoning_tokens_api": reasoning_tokens,
        "answer_tokens_api": answer_tokens,
        "reasoning": reasoning_text,
        "content": content_text,
        "extracted_answer": extract_answer(content_text),
        "strict_correct": is_correct(content_text, problem["answer"], truncated),
        "truncated": truncated,
        "knob": {"method": "thinking-cap", "request_model": LORA_NAME},
        "gold_in_reasoning": rounds.baseline_runner.gold_in_text(reasoning_text, problem["answer"]),
        "gold_in_content": rounds.baseline_runner.gold_in_text(content_text, problem["answer"]),
    }


def unfinished_canary(sample: dict) -> bool:
    reasoning = sample.get("reasoning") or ""
    content = sample.get("content") or ""
    return bool(sample.get("truncated")) and not content.strip() and "</think>" not in reasoning


def discard_adapter(reason: str) -> None:
    note = ROUND / "FAILURE.md"
    note.write_text(
        "# 050 探針丟掉 adapter\n\n"
        f"{reason}\n\n"
        "沒有拿這份權重去評開發集。通用聊天仍用基座 `Ornith-1.5-9B`。\n",
        encoding="utf-8",
    )
    if ADAPTER.exists():
        target = ROUND / "adapter-discarded"
        if target.exists():
            target = ROUND / f"adapter-discarded-{int(time.time())}"
        ADAPTER.rename(target)
        print(f"adapter discarded to {target}", flush=True)
    stop_listeners()
    os.environ.pop("VLLM_LORA_MODULES", None)


def serve() -> dict:
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    os.environ["VLLM_LORA_MODULES"] = f"{LORA_NAME}={ADAPTER}"
    os.environ["VLLM_MAX_LORA_RANK"] = "16"
    log_path = ROUND / "logs" / "vllm.log"
    body, ready_at, pid = reload_server(log_path, boot_timeout_s=600)
    payload = json.loads(body)
    names = [item.get("id") for item in payload["data"]]
    if BASE_NAME not in names or LORA_NAME not in names:
        raise SystemExit(f"expected {BASE_NAME} and {LORA_NAME}, got {names}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (log_path.parent / "reload.json").write_text(body, encoding="utf-8")
    model = next(item for item in payload["data"] if item["id"] == BASE_NAME)
    return {
        "ready_at": ready_at,
        "model": model["id"],
        "models": names,
        "max_model_len": model.get("max_model_len"),
        "pid": pid,
        "stopped_and_reloaded_by": "experiments/050-thinking-cap/run_round.py",
    }


def complete(model_name: str, messages: list[dict], max_tokens: int, seed: int) -> dict:
    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True},
        **rounds.baseline_runner.SAMPLING,
    }
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode())
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _count_tokens(reasoning)
    return {
        "model": model_name,
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": reasoning_tokens,
        "reasoning": reasoning,
        "content": content,
        "closed": "</think>" in reasoning or bool(content.strip()),
        "visible": bool(content.strip()),
        "has_cjk": any("\u4e00" <= char <= "\u9fff" for char in content),
        "ans_loop": reward.is_ans_collapse(reasoning, content, choice.get("finish_reason") == "length"),
    }


def zh_probe() -> list[dict]:
    rows = []
    for model_name in (BASE_NAME, LORA_NAME):
        for item in ZH_ITEMS:
            messages: list[dict] = []
            for turn, user in enumerate(item["turns"]):
                messages.append({"role": "user", "content": user})
                try:
                    row = complete(model_name, messages, 400, 101)
                    row["error"] = None
                except Exception as exc:  # noqa: BLE001
                    row = {"model": model_name, "error": str(exc), "visible": False, "turn": turn, "id": item["id"]}
                row["id"] = item["id"]
                row["turn"] = turn
                rows.append(row)
                print(
                    f"zh {model_name} {item['id']} turn={turn} visible={row.get('visible')} "
                    f"tokens={row.get('reasoning_tokens')}",
                    flush=True,
                )
                messages.append({"role": "assistant", "content": row.get("content") or ""})
    (ROUND / "zh_probe.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    before = BASE_WEIGHT.stat().st_mtime
    if not (ADAPTER / "adapter_model.safetensors").exists():
        stop_listeners()
        wait_gpu_free()
        subprocess.run([PY, "-u", str(ROUND / "train.py")], check=True)
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model weight index mtime changed")
    suffix, problems = rounds.baseline_runner.load_problems(DEV)
    probe = CollapseProbe(ROUND / "probe.log")
    probe.start()
    canary = next(item for item in problems if item["id"] == "e01")
    try:
        ready = serve()
        sample = stream_one(canary, suffix, 101, probe, "050-canary")
        (ROUND / "canary.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"canary correct={sample.get('strict_correct')} trunc={sample.get('truncated')} "
            f"chars={len(sample.get('reasoning') or '')}",
            flush=True,
        )
        bad = probe.discarded or reward.probe_decision(
            sample.get("reasoning") or "", sample.get("content") or "", bool(sample.get("truncated"))
        ) == "discard"
        if bad or unfinished_canary(sample):
            discard_adapter("e01 出現 ANS 迴圈，或思考沒有結束而且可見回答是空的。")
            raise SystemExit("probe discarded the adapter on the canary")
        zh_probe()
        out = ROUND / "results.jsonl"
        have = rounds.baseline_runner.done_keys(out)
        jobs = [(problem, seed) for problem in problems for seed in SEEDS if (problem["id"], seed) not in have]
        print(f"dev jobs {len(jobs)}", flush=True)
        rows_lock = threading.Lock()

        def work(job):
            problem, seed = job
            return stream_one(problem, suffix, seed, probe, "050-thinking-cap")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, job) for job in jobs]
            for future in as_completed(futures):
                if probe.discarded:
                    break
                row = future.result()
                with rows_lock:
                    with out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"050 {row['problem_id']} seed={row['seed']} correct={int(bool(row.get('strict_correct')))} "
                    f"trunc={int(bool(row.get('truncated')))} reason_tok={row.get('reasoning_tokens_api')}",
                    flush=True,
                )
        if probe.discarded:
            discard_adapter("開發集抽樣途中探針看到 ANS 迴圈。")
            raise SystemExit("probe discarded the adapter during dev scoring")
    finally:
        probe.close()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line.strip()]
    if any(reward.is_ans_collapse(row.get("reasoning") or "", row.get("content") or "", bool(row.get("truncated"))) for row in rows):
        discard_adapter("評分列裡有 ANS 迴圈。")
        raise SystemExit("scored rows contain an ANS loop")
    summary = write_record(
        ROUND,
        {
            "name": "050-thinking-cap",
            "method": "thinking-cap",
            "rule": "Thinking-Cap 獎勵接到 Ornith-1.5-9B。一輪、學習率 2e-6、KL 0.04、優勢只在思考區。不是通用繁體訓練。",
            "system": None,
            "extra_body": {},
            "knob_label": "Thinking-Cap 獎勵，一輪 LoRA",
        },
        rows,
        ready,
    )
    print(summary["judgement"], flush=True)
    if summary["n_errors"] == 0 and summary["n_rows"] == 56:
        upsert_compare_row(
            COMPARE_PATH,
            {"name": "050-thinking-cap", "knob_label": "Thinking-Cap 獎勵，一輪 LoRA"},
            summary,
        )
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model weight index mtime changed after scoring")


if __name__ == "__main__":
    main()
