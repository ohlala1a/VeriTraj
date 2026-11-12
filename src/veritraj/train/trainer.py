"""Online group-relative policy optimization using live HF weights."""
from pathlib import Path
import copy
import random
import torch
from ..advantage import AdvantageConfig
from ..config import build_components
from ..envs.factory import make_environment
from ..policy.generate import HFGenerator
from ..records import write_jsonl
from ..runtime import init_run, write_json
from .rollout import collect_group
from .baselines import advantages, ALGORITHMS
from .losses import completion_logps, clipped_objective, sampled_token_span

def load_policy(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = config.get("dtype", "bfloat16" if str(device).startswith("cuda") else "float32")
    dtype = getattr(torch, dtype_name)
    adapter_path = Path(config["model"])/"adapter_config.json"
    model_name = config["model"]
    if adapter_path.exists():
        from peft import PeftConfig
        model_name = PeftConfig.from_pretrained(config["model"]).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config["model"], is_trainable=True)
    elif config.get("lora"):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    model.to(device)
    return model, tokenizer

def update_policy(model, reference, tokenizer, optimizer, group, config):
    values = advantages(group, AdvantageConfig.from_dict(config.get("advantage")),
                        config.get("algorithm", "veritraj"))
    samples = []
    model.eval()
    reference.eval()
    for traj in group.trajectories:
        if values[traj.traj_id] is None:
            continue
        for generation, label, value in zip(group.generations[traj.traj_id],
                                            group.verified[traj.traj_id], values[traj.traj_id]):
            mask = sampled_token_span(tokenizer, generation, label.char_span)
            if not any(mask):
                continue
            with torch.no_grad():
                old = completion_logps(model, generation.prompt_token_ids, generation.token_ids)
                ref = completion_logps(reference, generation.prompt_token_ids, generation.token_ids)
            samples.append((generation, value, torch.tensor(mask, device=model.device), old, ref))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    losses = []
    for start in range(0, len(samples), accumulation):
        batch = samples[start:start+accumulation]
        denominator = sum(int(item[2].sum()) for item in batch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for generation, value, mask, old, ref in batch:
            new = completion_logps(model, generation.prompt_token_ids, generation.token_ids)
            loss = -clipped_objective(new, old, ref, torch.full_like(new, value), mask,
                                      config.get("eps_clip", .2), config.get("beta", .01), denominator)
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
        optimizer.step()
        losses.append(total_loss)
    return {"loss": sum(losses)/max(len(losses), 1), "optimizer_steps": len(losses),
            "action_tokens": sum(int(item[2].sum()) for item in samples)}

def save_checkpoint(path, model, reference, tokenizer, optimizer, update, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "model": model.state_dict(),
                "reference": reference.state_dict(),
                "update": update, "config": config, "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               path/"training_state.pt")

def run_training(config):
    algorithm = config.get("algorithm", "veritraj")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; PPO value-head training is unavailable")
    tasks = config["tasks"]
    if not tasks:
        raise ValueError("tasks must contain at least one task identifier")
    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    output = init_run(config["output_dir"], config, seed)
    model, tokenizer = load_policy(config)
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(config.get("learning_rate", 1e-6)))
    start = 0
    if config.get("resume"):
        state = torch.load(Path(config["resume"])/"training_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        reference.load_state_dict(state["reference"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    components = build_components(config["components"])
    env = make_environment(config["environment"], components)
    generator = HFGenerator(model, tokenizer, **config.get("generation", {}))
    try:
        for update in range(start, int(config.get("updates", 1))):
            group = collect_group(env, generator, components, tasks[update % len(tasks)],
                                  group_size=int(config.get("group_size", 8)), seed=seed+update,
                                  gate=algorithm == "shielded" or config.get("gate", algorithm == "veritraj"),
                                  max_steps=int(config.get("max_steps", 50)), group_id=f"update-{update}",
                                  policy_name=algorithm, checkpoint=config["model"],
                                  environment=config["environment"].get("kind", "pddl"))
            metrics = update_policy(model, reference, tokenizer, optimizer, group, config)
            write_jsonl(output/f"rollouts-{update}.jsonl.gz", (t.to_dict() for t in group.trajectories))
            write_jsonl(output/f"verified-{update}.jsonl.gz",
                        (r.to_dict() for rows in group.verified.values() for r in rows))
            write_json(output/f"metrics-{update}.json", metrics)
            save_checkpoint(output/f"checkpoint-{update+1}", model, reference, tokenizer, optimizer, update+1, config)
    finally:
        env.close()
    return output
