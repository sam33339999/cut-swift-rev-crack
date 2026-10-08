#!/usr/bin/env python3
"""SFT a general Traditional Chinese LoRA with a short, size-tagged think."""

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
LR = 1e-5
KL_COEF = 0.1
GRAD_CLIP = 1.0
MAX_LEN = 2048
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]


def _load_retry():
    spec = importlib.util.spec_from_file_location(
        "train_retry_ref", ROOT / "experiments/045-dual-reward/train_retry.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


retry = _load_retry()


def examples(tokenizer):
    rows = []
    for line in (ROUND / "train.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    built = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["history"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        target = row["think"] + "\n</think>\n\n" + row["answer"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids or len(prompt_ids) + len(target_ids) > MAX_LEN:
            continue
        built.append((torch.tensor(prompt_ids + target_ids, dtype=torch.long), len(prompt_ids)))
    return built


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    batches = examples(tokenizer)
    if not batches:
        raise SystemExit("no training rows")
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
    updates = 0
    skipped = 0
    for ids, prompt_len in batches:
        ids = ids.cuda()
        try:
            with torch.no_grad(), model.disable_adapter():
                ref = retry.completion_logprobs(model, ids, prompt_len)
            policy = retry.completion_logprobs(model, ids, prompt_len)
            width = min(policy.shape[0], ref.shape[0])
            policy = policy[:width]
            ref = ref[:width]
            kl = torch.exp(policy - ref) - (policy - ref) - 1
            loss = -policy.mean() + KL_COEF * kl.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    stats = {
        "updates": updates,
        "skipped": skipped,
        "rows": len(batches),
        "epochs": 1,
        "lr": LR,
        "kl_coef": KL_COEF,
        "grad_clip": GRAD_CLIP,
        "adapter": str(ADAPTER),
    }
    (ROUND / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
