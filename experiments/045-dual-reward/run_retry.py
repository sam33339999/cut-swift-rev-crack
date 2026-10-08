#!/usr/bin/env python3
"""Retry method 45: milder LoRA, then score the dev set under a 30s ANS probe.

The failed adapter at experiments/045-dual-reward/adapter is not loaded or served.
Training prompts are the existing rollouts, which do not include Sharp text.
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
ADAPTER = ROUND / "retry" / "adapter"
FAILED = ROUND / "adapter"
LORA_NAME = "m45retry"
BASE_WEIGHT = Path("/content/models/ornith-1.5-9B/model.safetensors.index.json")
URL = "http://127.0.0.1:8000/v1/chat/completions"


def _reward():
    spec = importlib.util.spec_from_file_location("reward45", ROUND / "reward.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _reward()
_TOKENIZER = None


def _count_tokens(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("/content/models/ornith-1.5-9B", trust_remote_code=True)
    if not text:
        return 0
    return len(_TOKENIZER(text, add_special_tokens=False).input_ids)


class CollapseProbe:
    """Recheck in-progress text on the shipped period. Discard needs no person."""

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
        row = {
            "t": time.time(),
            "kind": kind,
            "decision": decision,
            "chars": len(text),
            "period_s": reward.PROBE_PERIOD_S,
        }
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
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
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
        "knob": {"method": "45", "request_model": LORA_NAME},
        "gold_in_reasoning": rounds.baseline_runner.gold_in_text(reasoning_text, problem["answer"]),
        "gold_in_content": rounds.baseline_runner.gold_in_text(content_text, problem["answer"]),
    }


def discard_adapter() -> None:
    if ADAPTER.exists():
        target = ROUND / "retry" / "adapter-discarded"
        if target.exists():
            target = ROUND / "retry" / f"adapter-discarded-{int(time.time())}"
        ADAPTER.rename(target)
        print(f"adapter discarded to {target}", flush=True)


def serve() -> dict:
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    os.environ["VLLM_LORA_MODULES"] = f"{LORA_NAME}={ADAPTER}"
    os.environ["VLLM_MAX_LORA_RANK"] = "16"
    log_path = ROUND / "retry" / "logs" / "vllm.log"
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
        "stopped_and_reloaded_by": "experiments/045-dual-reward/run_retry.py",
    }


def main() -> None:
    before = BASE_WEIGHT.stat().st_mtime
    if not (ADAPTER / "adapter_model.safetensors").exists():
        stop_listeners()
        subprocess.run([sys.executable, str(ROUND / "train_retry.py")], check=True)
    after = BASE_WEIGHT.stat().st_mtime
    if before != after:
        raise SystemExit("base model weight index mtime changed")
    if ADAPTER.resolve() == FAILED.resolve():
        raise SystemExit("refusing to serve the failed adapter")
    suffix, problems = rounds.baseline_runner.load_problems(DEV)
    probe = CollapseProbe(ROUND / "retry" / "probe.log")
    probe.start()
    canary = next(item for item in problems if item["id"] == "e01")
    ready = serve()
    if ready["model"] != "Ornith-1.5-9B":
        raise SystemExit(f"unexpected served model {ready['model']}")
    try:
        sample = stream_one(canary, suffix, 101, probe, "047-canary")
        print(
            f"canary correct={sample.get('strict_correct')} trunc={sample.get('truncated')} "
            f"chars={len(sample.get('reasoning') or '')}",
            flush=True,
        )
        if probe.discarded or reward.probe_decision(sample.get("reasoning") or "", sample.get("content") or "", bool(sample.get("truncated"))) == "discard":
            discard_adapter()
            raise SystemExit("probe discarded the adapter on the canary")
        out = ROUND / "retry" / "results.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        have = rounds.baseline_runner.done_keys(out)
        jobs = [(problem, seed) for problem in problems for seed in SEEDS if (problem["id"], seed) not in have]
        print(f"dev jobs {len(jobs)}", flush=True)
        rows_lock = threading.Lock()

        def work(job):
            problem, seed = job
            return stream_one(problem, suffix, seed, probe, "047-method45-retry")

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
                    f"047 {row['problem_id']} seed={row['seed']} correct={int(bool(row.get('strict_correct')))} "
                    f"trunc={int(bool(row.get('truncated')))} reason_tok={row.get('reasoning_tokens_api')}",
                    flush=True,
                )
        if probe.discarded:
            discard_adapter()
            raise SystemExit("probe discarded the adapter during dev scoring")
    finally:
        probe.close()
    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if any(reward.is_ans_collapse(row.get("reasoning") or "", row.get("content") or "", bool(row.get("truncated"))) for row in rows):
        discard_adapter()
        raise SystemExit("scored rows contain an ANS loop")
    summary = write_record(
        ROUND / "retry",
        {
            "name": "047-method45-retry",
            "method": "45",
            "rule": "第 45 條第二次 LoRA。優勢只在思考區和 </think>。訓練提示沒有 Sharp。",
            "system": None,
            "extra_body": {},
        },
        rows,
        ready,
    )
    print(summary["judgement"], flush=True)
    success = summary["judgement"].startswith("難題思考中位數至少少了 15%")
    if success and summary["n_errors"] == 0 and summary["n_rows"] == 56:
        upsert_compare_row(
            COMPARE_PATH,
            {
                "name": "047-method45-retry",
                "knob_label": "第 45 條 LoRA，優勢只在思考區",
            },
            summary,
        )
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model weight index mtime changed after scoring")


if __name__ == "__main__":
    main()
