#!/usr/bin/env python3
"""Traditional Chinese short-thinking LoRA.

Verifiable Chinese math uses the method-45 reward, with advantage only inside
<think> and on </think>. Dialogue rows are ordinary next-token loss on a short
Chinese think plus the Traditional Chinese reply. No Sharp text.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

ROUND = Path(__file__).resolve().parent
ROOT = ROUND.parents[1]
MODEL = "/content/models/ornith-1.5-9B"
ADAPTER = ROUND / "adapter"
KL_COEF = 0.1
LR = 1e-5
GRAD_CLIP = 1.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_retry = _load(ROOT / "experiments/045-dual-reward/train_retry.py", "train_retry_zh")
reward_mod = train_retry.reward_mod


def load_math_groups() -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in (ROUND / "rollouts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        key = row["problem_id"]
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    return [groups[key] for key in order]


def sft_batch(tokenizer, messages: list[dict]) -> tuple[torch.Tensor, int] | None:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    target = "先給結論。\n</think>\n\n" + messages[-1]["content"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if not target_ids or len(prompt_ids) + len(target_ids) > train_retry.MAX_LEN:
        return None
    return torch.tensor(prompt_ids + target_ids, dtype=torch.long), len(prompt_ids)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def tokenize(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    math_rows = []
    for group in load_math_groups():
        rewards = reward_mod.group_rewards(group, tokenize)
        mean = sum(rewards) / len(rewards)
        var = sum((item - mean) ** 2 for item in rewards) / len(rewards)
        std = var**0.5
        for row, score in zip(group, rewards):
            advantage = 0.0 if std < 1e-6 else (score - mean) / std
            math_rows.append((row, advantage))
    dialogues = [
        json.loads(line)
        for line in (ROUND / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    math_updates = 0
    sft_updates = 0
    skipped = 0

    def step(ids, prompt_len, advantage: float | None) -> None:
        ids = ids.cuda()
        with torch.no_grad(), model.disable_adapter():
            ref = train_retry.completion_logprobs(model, ids, prompt_len)
        policy = train_retry.completion_logprobs(model, ids, prompt_len)
        width = min(policy.shape[0], ref.shape[0])
        policy = policy[:width]
        ref = ref[:width]
        kl = torch.exp(policy - ref) - (policy - ref) - 1
        if advantage is None:
            loss = -policy.mean() + KL_COEF * kl.mean()
        else:
            pieces = [tokenizer.decode([int(token)]) for token in ids[prompt_len:prompt_len + width].tolist()]
            mask = torch.tensor(reward_mod.advantage_mask(pieces)[:width], dtype=torch.float32, device=ids.device)
            weight = mask.sum().clamp_min(1.0)
            loss = -(advantage * mask * policy).sum() / weight + KL_COEF * kl.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for row, advantage in math_rows:
        built = train_retry.build_example(tokenizer, row)
        if built is None or built.get("skip"):
            skipped += 1
            continue
        try:
            step(built["ids"], built["prompt_len"], advantage)
            math_updates += 1
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
    for item in dialogues:
        built = sft_batch(tokenizer, item["messages"])
        if built is None:
            skipped += 1
            continue
        ids, prompt_len = built
        try:
            step(ids, prompt_len, None)
            sft_updates += 1
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    stats = {
        "math_updates": math_updates,
        "sft_updates": sft_updates,
        "skipped": skipped,
        "epochs": 1,
        "lr": LR,
        "kl_coef": KL_COEF,
        "grad_clip": GRAD_CLIP,
        "adapter": str(ADAPTER),
        "language": "zh-Hant",
    }
    (ROUND / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
