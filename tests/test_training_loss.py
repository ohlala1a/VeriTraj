import pytest
torch = pytest.importorskip("torch")
from veritraj.advantage import clipped_objective as reference
from veritraj.train.losses import clipped_objective, group_normalize

def test_loss_matches_list_reference_and_detaches_snapshots():
    new = torch.tensor([-.4, -.8, -1.2], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-.7, -.6, -1.1], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([-.3, -.9, -1.0], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([1.2, -.9, .5], dtype=torch.float64)
    mask = torch.tensor([1, 1, 0])
    result = clipped_objective(new, old, ref, adv, mask)
    assert result.item() == pytest.approx(reference(new.tolist(), old.tolist(), ref.tolist(), adv.tolist(), mask.tolist()))
    result.backward()
    assert new.grad is not None and old.grad is None and ref.grad is None

def test_global_token_denominator_and_empty_mask():
    new = torch.tensor([-.2, -.3, -.4], requires_grad=True)
    old = new.detach()
    adv = torch.tensor([1., 2., 3.])
    mask = torch.tensor([1, 0, 1])
    whole = clipped_objective(new, old, old, adv, mask)
    pieces = sum(clipped_objective(new[i:i+1], old[i:i+1], old[i:i+1], adv[i:i+1], mask[i:i+1], denominator=2) for i in range(3))
    assert whole.item() == pytest.approx(pieces.item())
    assert clipped_objective(new, old, old, adv, torch.zeros(3)).item() == 0
    assert torch.equal(group_normalize(torch.ones(3)), torch.zeros(3))

def test_token_alignment_rejects_unstable_unicode_prefixes():
    from veritraj.train.losses import sampled_token_span
    from veritraj.policy.generate import Generation
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "replacement" if len(ids) == 1 else "text"
    with pytest.raises(ValueError, match="safely align"):
        sampled_token_span(Tokenizer(), Generation("text", [0], [1, 2]), (0, 4))

def test_fast_offsets_require_sampled_id_identity():
    from veritraj.train.losses import sampled_token_span
    from veritraj.policy.generate import Generation
    class Tokenizer:
        is_fast = True
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 2)]}
    assert sampled_token_span(Tokenizer(), Generation("ab", [0], [1, 2]), (1, 2)) == [0, 1]

def test_tiny_causal_model_update():
    transformers = pytest.importorskip("transformers")
    from veritraj.train.losses import completion_logps
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=16, n_positions=16, n_embd=16, n_layer=1, n_head=2))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.transformer.wte.weight.detach().clone()
    logp = completion_logps(model, [1, 2], [3, 4])
    assert logp.shape == (2,)
    loss = -clipped_objective(logp, logp.detach(), logp.detach(), torch.ones(2), torch.ones(2))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.transformer.wte.weight)

def test_online_trainer_updates_from_collected_group():
    transformers = pytest.importorskip("transformers")
    import copy
    from veritraj.train.trainer import update_policy
    from veritraj.train.rollout import RolloutGroup
    from veritraj.policy.generate import Generation
    from veritraj.records import Trajectory, LoggedRecord, VerifiedRecord
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(chr(97+i) for i in ids)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=16, n_positions=16, n_embd=16, n_layer=1, n_head=2))
    reference_model = copy.deepcopy(model)
    reference_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    group = RolloutGroup()
    for index in range(2):
        identifier = str(index)
        group.trajectories.append(Trajectory(identifier, "task", [LoggedRecord(0, "obs", "de", env_reward=float(index))]))
        group.verified[identifier] = [VerifiedRecord(identifier, 0, (0, 2), (0, 2), "noop", index)]
        group.generations[identifier] = [Generation("de", [1, 2], [3, 4])]
    before = model.transformer.wte.weight.detach().clone()
    metrics = update_policy(model, reference_model, Tokenizer(), optimizer, group,
                            {"gradient_accumulation_steps": 2})
    assert metrics["optimizer_steps"] == 1
    assert metrics["action_tokens"] == 4
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, model.transformer.wte.weight)
    assert all(parameter.grad is None for parameter in reference_model.parameters())
