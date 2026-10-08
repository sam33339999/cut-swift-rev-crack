#!/usr/bin/env python3
"""Build a small Traditional Chinese chat set with a task-size think budget."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROUND = Path(__file__).resolve().parent
PARQUET = (
    "/root/.cache/huggingface/hub/datasets--yentinglin--twllm-data/snapshots/"
    "027f9f20c0d80d3695e1477c874ca9f102eccb68/20250216/"
    "train-00000-of-00001-5fcf805680823132.parquet"
)
EVAL_PATH = ROUND / "eval_zh.json"
OUT = ROUND / "train.jsonl"
LIMIT = 600
SEED = 49

THINK = {
    "small": ("這是小題。", "小題，直接回答。"),
    "medium": ("這是中題。先講結論，再補必要步驟。", "中題。用兩三句把要點說完。"),
    "large": ("這是大題。先定範圍，再列取捨，不重寫第二遍。", "大題。列出取捨後就停，不重算。"),
}
SMALL_WORDS = ("一句", "是什麼", "什麼是", "幾號", "改成", "定義", "翻譯")
LARGE_WORDS = ("系統設計", "架構", "策略", "分析", "比較", "排障", "故障", "多租戶", "設計")


def task_size(user_text: str, earlier_users: int) -> str:
    if any(word in user_text for word in LARGE_WORDS) or len(user_text) > 180 or earlier_users >= 2:
        return "large"
    if any(word in user_text for word in SMALL_WORDS) or len(user_text) < 24:
        return "small"
    return "medium"


def think_for(size: str, index: int) -> str:
    options = THINK[size]
    return options[index % len(options)]


def normalize(messages: list[dict]) -> list[dict] | None:
    out = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role not in {"user", "assistant"} or not content:
            return None
        out.append({"role": role, "content": content})
    if len(out) < 2 or out[0]["role"] != "user" or out[-1]["role"] != "assistant":
        return None
    return out


def has_chinese(text: str) -> bool:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text) >= 8


def eval_user_texts() -> set[str]:
    if not EVAL_PATH.exists():
        return set()
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    texts = set()
    for item in payload["items"]:
        texts.add(item["turns"][0].strip())
        for turn in item["turns"]:
            texts.add(turn.strip())
    return texts


def build_rows() -> list[dict]:
    import pyarrow.parquet as pq

    conversations = pq.read_table(PARQUET, columns=["conversations"]).column("conversations").to_pylist()
    blocked = eval_user_texts()
    multi = []
    single = []
    for index, raw in enumerate(conversations):
        messages = normalize(raw)
        if messages is None:
            continue
        if not has_chinese(messages[0]["content"]):
            continue
        users = [message["content"] for message in messages if message["role"] == "user"]
        if users[0].strip() in blocked or any(turn in blocked for turn in users):
            continue
        # One example per assistant reply, so a later turn sees the earlier dialogue.
        history: list[dict] = []
        assistant_index = 0
        for message in messages:
            if message["role"] != "assistant":
                history.append(message)
                continue
            answer = message["content"]
            if "ANSWER:" in answer or len(answer) < 8 or len(answer) > 1200:
                history.append(message)
                assistant_index += 1
                continue
            last_user = history[-1]["content"]
            earlier = sum(item["role"] == "user" for item in history[:-1])
            size = task_size(last_user, earlier)
            row = {
                "id": f"{index}-{assistant_index}",
                "size": size,
                "think": think_for(size, index + assistant_index),
                "history": history,
                "answer": answer,
                "multi": earlier > 0,
            }
            (multi if row["multi"] else single).append(row)
            history.append(message)
            assistant_index += 1
    random.seed(SEED)
    random.shuffle(multi)
    random.shuffle(single)
    chosen = multi[:400] + single[:200]
    random.shuffle(chosen)
    return chosen[:LIMIT]


def main() -> None:
    rows = build_rows()
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sizes = {}
    multi = 0
    for row in rows:
        sizes[row["size"]] = sizes.get(row["size"], 0) + 1
        multi += int(row["multi"])
    print(json.dumps({"rows": len(rows), "multi": multi, "sizes": sizes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
