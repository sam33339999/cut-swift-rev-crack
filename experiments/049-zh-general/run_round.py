#!/usr/bin/env python3
"""Train the general Traditional Chinese short-think LoRA and score the fixed chats."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.server_ctl import reload_server, stop_listeners  # noqa: E402

ROUND = Path(__file__).resolve().parent
ADAPTER = ROUND / "adapter"
LORA_NAME = "zhgeneral"
BASE_WEIGHT = Path("/content/models/ornith-1.5-9B/model.safetensors.index.json")
URL = "http://127.0.0.1:8000/v1/chat/completions"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _load(ROOT / "experiments/045-dual-reward/reward.py", "reward49")
prepare = _load(ROUND / "prepare_data.py", "prepare49")


class Probe:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.text = ""
        self.content = ""
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.discarded = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._write("start")
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.wait(reward.PROBE_PERIOD_S):
            self._write("tick")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def update(self, text: str, content: str) -> None:
        with self.lock:
            self.text = text
            self.content = content
        if reward.probe_decision(text, content, False) == "discard":
            self.discarded = True
        if text and "</think>" not in text and len(text) > 500:
            self.discarded = True

    def _write(self, kind: str) -> None:
        with self.lock:
            text, content = self.text, self.content
        decision = "discard" if self.discarded else reward.probe_decision(text, content, False)
        if decision == "discard":
            self.discarded = True
        row = {"t": time.time(), "kind": kind, "decision": decision, "chars": len(text)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"probe {kind} decision={decision} chars={len(text)}", flush=True)


def serve() -> None:
    os.environ.pop("CHAT_TEMPLATE", None)
    os.environ.pop("VLLM_LOGITS_PROCESSORS", None)
    os.environ["VLLM_LORA_MODULES"] = f"{LORA_NAME}={ADAPTER}"
    os.environ["VLLM_MAX_LORA_RANK"] = "16"
    body, _, _ = reload_server(ROUND / "logs" / "vllm.log")
    model = json.loads(body)["data"][0]["id"]
    if model != "Ornith-1.5-9B":
        raise SystemExit(f"unexpected model {model}")


def chat(messages: list[dict], probe: Probe) -> dict:
    body = {
        "model": LORA_NAME,
        "messages": messages,
        "max_tokens": 480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    reasoning: list[str] = []
    content: list[str] = []
    usage = {}
    finish = None
    with urllib.request.urlopen(request, timeout=180) as response:
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
            probe.update("".join(reasoning), "".join(content))
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    details = (usage.get("completion_tokens_details") or {}) if usage else {}
    return {
        "reasoning": reasoning_text,
        "content": content_text,
        "truncated": finish == "length",
        "reasoning_tokens": details.get("reasoning_tokens"),
        "finish_reason": finish,
    }


def run_eval(probe: Probe) -> dict:
    payload = json.loads((ROUND / "eval_zh.json").read_text(encoding="utf-8"))
    results = []
    for item in payload["items"]:
        messages = []
        turns = []
        for turn in item["turns"]:
            messages.append({"role": "user", "content": turn})
            row = chat(messages, probe)
            turns.append(row)
            if probe.discarded or not (row["content"] or "").strip():
                results.append({"id": item["id"], "size": item["size"], "turns": turns, "ok": False})
                return {"items": results, "discarded": probe.discarded}
            messages.append({"role": "assistant", "content": row["content"]})
        tokens = [turn["reasoning_tokens"] for turn in turns if isinstance(turn["reasoning_tokens"], int)]
        results.append({"id": item["id"], "size": item["size"], "turns": turns, "ok": True, "reasoning_tokens": tokens})
    flat = [token for item in results for token in item.get("reasoning_tokens") or []]
    median = statistics.median(flat) if flat else None
    passed = (
        not probe.discarded
        and len(results) == 16
        and all(item["ok"] and not any(turn["truncated"] for turn in item["turns"]) for item in results)
        and median is not None
        and median <= 80
    )
    return {"items": results, "discarded": probe.discarded, "reasoning_median": median, "passed": passed}


def main() -> None:
    before = BASE_WEIGHT.stat().st_mtime
    prepare.main()
    if not (ADAPTER / "adapter_model.safetensors").exists():
        stop_listeners()
        subprocess.run([sys.executable, str(ROUND / "train.py")], check=True)
    if BASE_WEIGHT.stat().st_mtime != before:
        raise SystemExit("base model changed")
    serve()
    probe = Probe(ROUND / "probe.log")
    probe.start()
    try:
        report = run_eval(probe)
    finally:
        probe.close()
    (ROUND / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 049 通用繁體、按題目大小縮短思考",
        "",
        "語料是 yentinglin/twllm-data 的繁體對話。思考依小、中、大三檔寫成固定短句，回答是原本的繁體正文。",
        "沒有數學獎勵，沒有 Sharp，沒有用英文開發集當及格線。",
        "",
        f"思考 token 中位數：{report.get('reasoning_median')}",
        f"探針丟棄：{report.get('discarded')}",
        f"這組固定題是否通過：{report.get('passed')}",
        "",
    ]
    for item in report["items"]:
        last = item["turns"][-1]
        lines.append(
            f"- {item['id']} {item['size']} ok={item['ok']} "
            f"content_chars={len(last.get('content') or '')} "
            f"reason_tok={last.get('reasoning_tokens')} trunc={last.get('truncated')}"
        )
    (ROUND / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
