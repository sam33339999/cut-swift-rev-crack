#!/usr/bin/env python3
"""One GRPO update of method 45 on saved rollouts.

The base model is the reference. A KL penalty keeps this single update from
walking away from it. Sharp text is not in these prompts.
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


def _load_reward():
    spec = importlib.util.spec_from_file_location("reward45", ROUND / "reward.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward_mod = _load_reward()

MODEL = "/content/models/ornith-1.5-9B"
ROLLOUTS = ROUND / "rollouts.jsonl"
ADAPTER = ROUND / "adapter"
KL_COEF = 0.01
LR = 1e-4
EPOCHS = 2
MAX_LEN = 4096
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]


def load_rollouts() -> list[dict]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in ROLLOUTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        key = row["problem_id"]
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    return [groups[key] for key in order]


def build_ids(tokenizer, row: dict) -> tuple[torch.Tensor, int] | None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    reasoning = (row.get("reasoning") or "").strip("\n")
    content = row.get("content") or ""
    target = reasoning + "\n</think>\n\n" + content
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if not target_ids:
        return None
    if len(prompt_ids) + len(target_ids) > MAX_LEN:
        return None
    ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
    return ids, len(prompt_ids)


def completion_logprobs(model, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    outputs = model(input_ids=ids.unsqueeze(0), use_cache=False)
    logits = outputs.logits[0, prompt_len - 1 : -1]
    target = ids[prompt_len:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(1, target.unsqueeze(1)).squeeze(1)


def main() -> None:
    groups = load_rollouts()
    if not groups:
        raise SystemExit("no rollouts")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def tokenize(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    scored = []
    for group in groups:
        rewards = reward_mod.group_rewards(group, tokenize)
        mean = sum(rewards) / len(rewards)
        var = sum((item - mean) ** 2 for item in rewards) / len(rewards)
        std = var**0.5
        for row, score in zip(group, rewards):
            advantage = 0.0 if std < 1e-6 else (score - mean) / std
            scored.append((row, score, advantage))

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
    used = 0
    skipped = 0
    for _epoch in range(EPOCHS):
        for row, _score, advantage in scored:
            built = build_ids(tokenizer, row)
            if built is None:
                skipped += 1
                continue
            ids, prompt_len = built
            ids = ids.cuda()
            try:
                with torch.no_grad(), model.disable_adapter():
                    ref = completion_logprobs(model, ids, prompt_len)
                policy = completion_logprobs(model, ids, prompt_len)
                kl = torch.exp(policy - ref) - (policy - ref) - 1
                loss = -(advantage * policy).mean() + KL_COEF * kl.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                used += 1
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                continue
            del ids
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    stats = {
        "groups": len(groups),
        "sequences": len(scored),
        "updates": used,
        "skipped": skipped,
        "epochs": EPOCHS,
        "kl_coef": KL_COEF,
        "lr": LR,
        "max_len": MAX_LEN,
        "targets": TARGETS,
        "mean_reward": sum(item[1] for item in scored) / len(scored),
    }
    (ROUND / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
