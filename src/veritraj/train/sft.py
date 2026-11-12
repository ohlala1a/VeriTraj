"""Action-span supervised fine-tuning on curated JSONL records."""

from __future__ import annotations

import math
import random
from pathlib import Path

from ..records import read_jsonl
from ..runtime import init_run, write_json


def encode_example(row: dict, tokenizer, max_length: int = 4096) -> dict:
    prompt_ids = list(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"])
    completion = row["completion"]
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids = list(encoded["input_ids"])
    spans = row["target_char_spans"]
    if any(not (0 <= start < end <= len(completion)) for start, end in spans):
        raise ValueError(f"{row.get('id')}: target span is outside the completion")
    if len(completion_ids) >= max_length:
        raise ValueError(f"{row.get('id')}: completion exceeds max_length")
    available = max_length - len(completion_ids)
    removed = max(0, len(prompt_ids) - available)
    prompt_ids = prompt_ids[-available:]
    if not prompt_ids:
        raise ValueError(f"{row.get('id')}: prompt must contain at least one token")
    mask = [any(end > start and end > left and start < right for left, right in spans)
            for start, end in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError(f"{row.get('id')}: no target tokens")
    weight = float(row.get("weight", 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Example weight must be finite and nonnegative")
    labels = [-100] * len(prompt_ids) + [token if target else -100 for token, target in zip(completion_ids, mask)]
    return {"input_ids": prompt_ids + completion_ids, "labels": labels, "weight": weight,
            "truncated_prompt_tokens": removed, "id": row.get("id", "")}


def collate_examples(rows: list[dict], pad_token_id: int):
    import torch

    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        count = len(row["input_ids"])
        ids[index, :count] = torch.tensor(row["input_ids"])
        labels[index, :count] = torch.tensor(row["labels"])
        attention[index, :count] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attention,
            "weights": torch.tensor([row["weight"] for row in rows], dtype=torch.float32)}


def weighted_token_loss(logits, labels, weights, denominator=None):
    import torch.nn.functional as functional

    shifted = labels[:, 1:]
    losses = functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), shifted,
                                     reduction="none", ignore_index=-100)
    numerator = (losses * weights[:, None]).sum()
    count = shifted.ne(-100).sum() if denominator is None else denominator
    if int(count) <= 0:
        raise ValueError("No supervised tokens in update")
    return numerator / count


def run_sft(config: dict) -> dict:
    import torch
    from .trainer import load_policy

    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    output = init_run(config["output_dir"], config, seed)
    model, tokenizer = load_policy(config)
    if not tokenizer.is_fast:
        raise ValueError("SFT requires a fast tokenizer with offset mappings")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a padding or EOS token")
    device = model.device
    model.train()
    rows = [encode_example(row, tokenizer, int(config.get("max_length", 4096)))
            for row in read_jsonl(config["data"])]
    if not rows:
        raise ValueError("SFT input is empty")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(config.get("learning_rate", 2e-5)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))
    batch_size = int(config.get("batch_size", 1))
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("epochs", 1))
    if min(batch_size, accumulation, epochs) < 1:
        raise ValueError("Batch size, accumulation and epochs must be positive")
    updates = 0
    history = []
    rng = random.Random(seed)
    for epoch in range(epochs):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size * accumulation):
            group = [rows[index] for index in indices[start:start + batch_size * accumulation]]
            denominator = sum(sum(token != -100 for token in row["labels"][1:]) for row in group)
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for offset in range(0, len(group), batch_size):
                batch = {key: tensor.to(device) for key, tensor in
                         collate_examples(group[offset:offset + batch_size], tokenizer.pad_token_id).items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = weighted_token_loss(logits, batch["labels"], batch["weights"], denominator)
                loss.backward()
                update_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            updates += 1
            history.append({"epoch": epoch, "update": updates, "loss": update_loss,
                            "target_tokens": denominator})
            write_json(output / "training.json", history)
    checkpoint = output / "checkpoint-final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save({"optimizer": optimizer.state_dict(), "updates": updates, "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(), "shuffle_rng": rng.getstate(), "config": config},
               checkpoint / "training_state.pt")
    summary = {"examples": len(rows), "updates": updates, "final_loss": history[-1]["loss"],
               "truncated_prompt_tokens": sum(row["truncated_prompt_tokens"] for row in rows),
               "checkpoint": str(checkpoint)}
    write_json(output / "summary.json", summary)
    return summary
