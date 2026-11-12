"""Differentiable reference objective with token-weighted reduction."""
import torch

def group_normalize(returns, eps=1e-6):
    return (returns - returns.mean()) / (returns.std(unbiased=False) + eps)

def record_advantage(v, a_bar, r_prog, lambda_hard=2.0, lambda_pot=0.2):
    return v * a_bar - (1-v) * lambda_hard + v * lambda_pot * r_prog

def clipped_objective(logp_new, logp_old, logp_ref, adv, mask, eps_clip=0.2,
                      beta=0.01, denominator=None):
    ratio = torch.exp(logp_new - logp_old.detach())
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-eps_clip, 1+eps_clip) * adv)
    delta = logp_ref.detach() - logp_new
    kl = torch.exp(delta) - delta - 1
    total = torch.where(mask.bool(), surrogate-beta*kl, 0.0).sum()
    count = mask.sum() if denominator is None else denominator
    return total / torch.as_tensor(count, device=total.device).clamp_min(1)

def completion_logps(model, prompt_ids, token_ids):
    if not prompt_ids or not token_ids:
        raise ValueError("Prompt and completion require tokens")
    ids = torch.tensor([prompt_ids + token_ids], device=model.device)
    logits = model(input_ids=ids).logits[0, len(prompt_ids)-1:-1].float()
    targets = torch.tensor(token_ids, device=model.device)
    return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

def sampled_token_span(tokenizer, generation, char_span):
    start, end = char_span
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(generation.text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] == generation.token_ids:
            return [int(right > start and left < end) for left, right in encoded["offset_mapping"]]
    prefixes = [tokenizer.decode(generation.token_ids[:i], skip_special_tokens=False,
                clean_up_tokenization_spaces=False) for i in range(1, len(generation.token_ids)+1)]
    if not prefixes or prefixes[-1] != generation.text:
        raise ValueError("Decoded sampled tokens differ from generation text")
    if any(not generation.text.startswith(prefix) for prefix in prefixes):
        raise ValueError("Cannot safely align sampled token boundaries to action characters")
    lengths = [0] + [len(prefix) for prefix in prefixes]
    if lengths != sorted(lengths):
        raise ValueError("Sampled token prefixes have non-monotonic character boundaries")
    return [int(lengths[i+1] > start and lengths[i] < end) for i in range(len(generation.token_ids))]
