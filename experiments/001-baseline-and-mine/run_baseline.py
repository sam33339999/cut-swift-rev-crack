#!/usr/bin/env python3
"""Sample the locked baseline decoding setup on the dev set.

One row per problem and seed is appended to the jsonl as soon as it returns,
so a killed run can be resumed. Transport failures are stored and are not
treated as wrong answers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from experiments.grader import extract_answer, is_correct

ROOT = Path(__file__).resolve().parent
DEFAULT_PROBLEMS = ROOT / "problems.json"
DEFAULT_OUT = ROOT / "results.jsonl"

SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_problems(path: Path) -> tuple[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["suffix"], payload["problems"]


def done_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        keys.add((row["problem_id"], int(row["seed"])))
    return keys


def gold_in_text(text: str | None, gold: str) -> bool:
    if not text:
        return False
    return re.search(rf"(?<!\d){re.escape(gold)}(?!\d)", text) is not None


def post_json(url: str, body: dict, timeout: int) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc


def call_one(
    url: str,
    problem: dict,
    suffix: str,
    seed: int,
    max_tokens: int,
    timeout: int,
    experiment: str = "001-baseline",
    extra_body: dict | None = None,
    knob: dict | None = None,
    system: str | None = None,
) -> dict:
    prompt = problem["question"] + suffix
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": "Ornith-1.5-9B",
        "messages": messages,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True},
        **SAMPLING,
    }
    if extra_body:
        body.update(extra_body)
    started = time.time()
    started_iso = now()
    status, data = post_json(url, body, timeout)
    wall = time.time() - started
    choice = data["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning") or ""
    content = message.get("content") or ""
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    completion_tokens = usage.get("completion_tokens")
    answer_tokens = None
    if reasoning_tokens is not None and completion_tokens is not None:
        answer_tokens = completion_tokens - reasoning_tokens
    extracted = extract_answer(content)
    truncated = choice.get("finish_reason") == "length"
    return {
        "schema_version": 1,
        "experiment": experiment,
        "problem_id": problem["id"],
        "basket": problem["basket"],
        "seed": seed,
        "gold": problem["answer"],
        "prompt": prompt,
        "sampling": {**SAMPLING, "max_tokens": max_tokens, "enable_thinking": True},
        "request_started": started_iso,
        "request_finished": now(),
        "wall_s": round(wall, 3),
        "http_status": status,
        "error": None,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "reasoning_tokens_api": reasoning_tokens,
        "answer_tokens_api": answer_tokens,
        "reasoning": reasoning,
        "content": content,
        "extracted_answer": extracted,
        "strict_correct": is_correct(content, problem["answer"], truncated),
        "truncated": truncated,
        "knob": knob,
        "gold_in_reasoning": gold_in_text(reasoning, problem["answer"]),
        "gold_in_content": gold_in_text(content, problem["answer"]),
    }


def error_row(
    problem: dict,
    seed: int,
    max_tokens: int,
    exc: Exception,
    started: float,
    experiment: str = "001-baseline",
    knob: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "knob": knob,
        "problem_id": problem["id"],
        "basket": problem["basket"],
        "seed": seed,
        "gold": problem["answer"],
        "sampling": {**SAMPLING, "max_tokens": max_tokens, "enable_thinking": True},
        "request_started": now(),
        "request_finished": now(),
        "wall_s": round(time.time() - started, 3),
        "error": str(exc),
        "strict_correct": False,
        "truncated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="101,102,103,104")
    parser.add_argument("--ids", default="", help="Comma-separated problem ids. Empty means all.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    args = parser.parse_args()

    suffix, problems = load_problems(args.problems)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.ids.strip():
        wanted = {x.strip() for x in args.ids.split(",")}
        problems = [p for p in problems if p["id"] in wanted]
        missing = wanted - {p["id"] for p in problems}
        if missing:
            raise SystemExit(f"unknown problem ids: {sorted(missing)}")
    have = done_keys(args.out)
    jobs = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if (problem["id"], seed) not in have
    ]
    print(f"jobs {len(jobs)} already_done {len(have)} out {args.out}", flush=True)
    if not jobs:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def write(row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with lock:
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def work(problem: dict, seed: int) -> dict:
        started = time.time()
        try:
            return call_one(args.url, problem, suffix, seed, args.max_tokens, args.timeout)
        except Exception as exc:  # noqa: BLE001 — recorded, then the batch continues
            return error_row(problem, seed, args.max_tokens, exc, started)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
                f"wall={row.get('wall_s')} "
                f"err={row.get('error')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
