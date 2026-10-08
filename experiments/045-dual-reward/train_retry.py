#!/usr/bin/env python3
"""One milder method-45 pass. Advantage stays inside <think> and on </think>.

Does not read or write experiments/045-dual-reward/adapter, and does not write
the base model directory.
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
MODEL = "/content/models/ornith-1.5-9B"
ROLLOUTS = ROUND / "rollouts.jsonl"
ADAPTER = ROUND / "retry" / "adapter"
KL_COEF = 0.1
LR = 1e-5
EPOCHS = 1
GRAD_CLIP = 1.0
MAX_LEN = 4096
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
SHARP_MARK = "Answer directly, after thinking."


def _load_reward():
    spec = importlib.util.spec_from_file_location("reward45", ROUND / "reward.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward_mod = _load_reward()


def load_groups() -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in ROLLOUTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        if SHARP_MARK in (row.get("prompt") or ""):
            raise SystemExit("training prompt contains the Sharp terseness text")
        key = row["problem_id"]
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    return [groups[key] for key in order]


def completion_logprobs(model, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    outputs = model(input_ids=ids.unsqueeze(0), use_cache=False)
    logits = outputs.logits[0, prompt_len - 1 : -1]
    target = ids[prompt_len:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(1, target.unsqueeze(1)).squeeze(1)


def build_example(tokenizer, row: dict) -> dict | None:
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
    cut_char = reward_mod.first_gold_end(reasoning, str(row["gold"]))
    first_index = None
    if cut_char is not None:
        first_index = len(tokenizer(reasoning[:cut_char], add_special_tokens=False).input_ids)
    full = prompt_ids + target_ids
    over_long = len(full) > MAX_LEN
    chosen = reward_mod.trainable_post_answer_span(full, len(prompt_ids), first_index, MAX_LEN)
    if chosen is None:
        return {"skip": "length", "over_long": over_long}
    if over_long:
        context = prompt_ids[: min(len(prompt_ids), max(1, MAX_LEN // 4))]
        room = MAX_LEN - len(context)
        span = chosen[:room]
        ids = context + span
        prompt_len = len(context)
        pieces = [tokenizer.decode([token]) for token in span]
    else:
        ids = chosen
        prompt_len = len(prompt_ids)
        pieces = [tokenizer.decode([token]) for token in target_ids]
    mask = reward_mod.advantage_mask(pieces)
    if len(mask) != len(ids) - prompt_len:
        mask = mask[: len(ids) - prompt_len]
        if len(mask) < len(ids) - prompt_len:
            mask = mask + [0.0] * ((len(ids) - prompt_len) - len(mask))
    return {
        "skip": None,
        "over_long": over_long,
        "ids": torch.tensor(ids, dtype=torch.long),
        "prompt_len": prompt_len,
        "mask": mask,
    }


def main() -> None:
    if ADAPTER.resolve() == (ROUND / "adapter").resolve():
        raise SystemExit("refusing to write the failed adapter directory")
    groups = load_groups()
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
    kept_long = 0
    skipped_for_length = 0
    for _epoch in range(EPOCHS):
        for row, _score, advantage in scored:
            built = build_example(tokenizer, row)
            if built is None or built.get("skip"):
                skipped += 1
                if built and built.get("skip") == "length":
                    skipped_for_length += 1
                continue
            if built["over_long"]:
                kept_long += 1
            ids = built["ids"].cuda()
            prompt_len = built["prompt_len"]
            mask = torch.tensor(built["mask"], dtype=torch.float32, device=ids.device)
            try:
                with torch.no_grad(), model.disable_adapter():
                    ref = completion_logprobs(model, ids, prompt_len)
                policy = completion_logprobs(model, ids, prompt_len)
                width = min(policy.shape[0], ref.shape[0], mask.shape[0])
                policy = policy[:width]
                ref = ref[:width]
                mask = mask[:width]
                kl = torch.exp(policy - ref) - (policy - ref) - 1
                weight = mask.sum().clamp_min(1.0)
                loss = -(advantage * mask * policy).sum() / weight + KL_COEF * kl.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
        "skipped_for_length": skipped_for_length,
        "kept_long": kept_long,
        "epochs": EPOCHS,
        "kl_coef": KL_COEF,
        "lr": LR,
        "grad_clip": GRAD_CLIP,
        "max_len": MAX_LEN,
        "adapter": str(ADAPTER),
        "mean_reward": sum(item[1] for item in scored) / max(len(scored), 1),
    }
    out = ROUND / "retry" / "train_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
