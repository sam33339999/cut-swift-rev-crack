#!/usr/bin/env python3
"""One offline Thinking-Cap update on the local Ornith-1.5-9B.

Reads the existing method-45 rollouts. Does not sample, does not write the
base checkpoint, and does not write the 045 or 049 adapters.
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
MODEL = Path("/content/models/ornith-1.5-9B")
ROLLOUTS = ROOT / "experiments" / "045-dual-reward" / "rollouts.jsonl"
ADAPTER = ROUND / "adapter"
BASE_INDEX = MODEL / "model.safetensors.index.json"
KL_COEF = 0.04
LR = 2e-6
EPOCHS = 1
GRAD_CLIP = 1.0
MAX_LEN = 4096
LAM = 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
SHARP_MARK = "Answer directly, after thinking."
FORBIDDEN = [
    ROOT / "experiments" / "045-dual-reward" / "adapter",
    ROOT / "experiments" / "045-dual-reward" / "retry" / "adapter",
    ROOT / "experiments" / "049-zh-general" / "adapter",
    MODEL,
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _load(ROUND / "rewards.py", "thinking_cap_rewards")
reward45 = _load(ROOT / "experiments" / "045-dual-reward" / "reward.py", "reward45_mask")


def load_groups() -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in ROLLOUTS.read_text(encoding="utf-8").split("\n"):
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


def rollout_text(row: dict) -> str:
    reasoning = (row.get("reasoning") or "").strip("\n")
    if "</think>" in reasoning:
        reasoning = reasoning.split("</think>", 1)[0].strip("\n")
    content = row.get("content") or ""
    return reasoning + "\n</think>\n\n" + content


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
    target = rollout_text(row)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if not target_ids:
        return None
    if len(prompt_ids) + len(target_ids) > MAX_LEN:
        return {"skip": "length"}
    pieces = [tokenizer.decode([token]) for token in target_ids]
    mask = reward45.advantage_mask(pieces)
    if len(mask) != len(target_ids):
        mask = (mask + [0.0] * len(target_ids))[: len(target_ids)]
    return {
        "skip": None,
        "ids": torch.tensor(prompt_ids + target_ids, dtype=torch.long),
        "prompt_len": len(prompt_ids),
        "mask": mask,
    }


def score_rows(groups: list[list[dict]], tokenize) -> list[tuple[dict, float, float]]:
    scored: list[tuple[dict, float, float]] = []
    correct_totals: list[float] = []
    wrong_totals: list[float] = []
    for group in groups:
        totals = []
        details = []
        for row in group:
            detail = reward.score_completion(rollout_text(row), str(row["gold"]), tokenize, lam=LAM)
            totals.append(detail["total"])
            details.append(detail)
        mean = sum(totals) / len(totals)
        var = sum((item - mean) ** 2 for item in totals) / len(totals)
        std = var**0.5
        for row, total, detail in zip(group, totals, details):
            advantage = 0.0 if std < 1e-6 else (total - mean) / std
            scored.append((row, total, advantage))
            (correct_totals if detail["correct"] else wrong_totals).append(total)
    if correct_totals and wrong_totals and max(wrong_totals) >= min(correct_totals):
        raise SystemExit("reward gate failed: a wrong rollout scores at least as high as a correct one")
    print(
        json.dumps(
            {
                "groups": len(groups),
                "sequences": len(scored),
                "correct": len(correct_totals),
                "wrong": len(wrong_totals),
                "mean_reward": sum(item[1] for item in scored) / max(len(scored), 1),
                "lam": LAM,
            }
        ),
        flush=True,
    )
    return scored


def main() -> None:
    if any(ADAPTER.resolve() == path.resolve() or MODEL.resolve() in ADAPTER.resolve().parents for path in FORBIDDEN):
        raise SystemExit(f"refusing to write {ADAPTER}")
    before = BASE_INDEX.stat().st_mtime
    groups = load_groups()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def tokenize(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    scored = score_rows(groups, tokenize)
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
    skipped_for_length = 0
    for _epoch in range(EPOCHS):
        for row, _score, advantage in scored:
            built = build_example(tokenizer, row)
            if built is None or built.get("skip"):
                skipped += 1
                if built and built.get("skip") == "length":
                    skipped_for_length += 1
                continue
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
                if used % 10 == 0:
                    print(f"update {used} loss={float(loss.detach()):.4f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                print(f"skip oom after {used} updates", flush=True)
                continue
            del ids
    if BASE_INDEX.stat().st_mtime != before:
        raise SystemExit("base model weight index mtime changed; not saving")
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    stats = {
        "groups": len(groups),
        "sequences": len(scored),
        "updates": used,
        "skipped": skipped,
        "skipped_for_length": skipped_for_length,
        "epochs": EPOCHS,
        "kl_coef": KL_COEF,
        "lr": LR,
        "grad_clip": GRAD_CLIP,
        "max_len": MAX_LEN,
        "lam": LAM,
        "adapter": str(ADAPTER),
        "base_index_mtime": before,
        "mean_reward": sum(item[1] for item in scored) / max(len(scored), 1),
    }
    (ROUND / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
