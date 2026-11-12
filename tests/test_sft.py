"""Offset masks and token-weighted gradient accumulation."""

import pytest

from veritraj.train.sft import encode_example


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids = [ord(character) for character in text]
        result = {"input_ids": ([1] if add_special_tokens else []) + ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_prompt_and_reasoning_are_not_sft_targets():
    row = {"id": "one", "prompt": "observations", "completion": "think\nAction: look",
           "target_char_spans": [[14, 18]], "weight": .5}
    encoded = encode_example(row, CharacterTokenizer(), 24)
    assert encoded["truncated_prompt_tokens"] > 0
    assert [x for x in encoded["labels"] if x != -100] == list(map(ord, "look"))
    assert encoded["weight"] == .5


def test_sft_microbatches_have_same_gradient_as_global_batch():
    torch = pytest.importorskip("torch")
    from veritraj.train.sft import weighted_token_loss

    torch.manual_seed(4)
    labels = torch.tensor([[-100, 1, 2, -100], [-100, 0, 1, 2]])
    weights = torch.tensor([1.0, .5])
    logits = torch.randn(2, 4, 3, requires_grad=True)
    loss = weighted_token_loss(logits, labels, weights)
    gradient = torch.autograd.grad(loss, logits)[0]
    split = logits.detach().clone().requires_grad_(True)
    loss_split = sum(weighted_token_loss(split[i:i+1], labels[i:i+1], weights[i:i+1], denominator=5)
                     for i in range(2))
    gradient_split = torch.autograd.grad(loss_split, split)[0]
    assert torch.allclose(gradient, gradient_split)


def test_sft_saves_a_loadable_checkpoint(tmp_path):
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from veritraj.records import write_jsonl
    from veritraj.train.sft import run_sft

    vocabulary = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'room': 3, 'look': 4}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                                    unk_token='[UNK]', eos_token='[EOS]')
    source = tmp_path / 'model'
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=5, n_positions=16,
                                          n_embd=16, n_layer=1, n_head=2))
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    data = tmp_path / 'targets.jsonl'
    write_jsonl(data, [{'id': 'one', 'prompt': 'room', 'completion': 'look',
                       'target_char_spans': [[0, 4]], 'weight': 1.0}])
    summary = run_sft({'model': str(source), 'data': str(data), 'output_dir': str(tmp_path / 'run'),
                       'batch_size': 1, 'epochs': 1, 'gradient_accumulation_steps': 1})
    assert summary['updates'] == 1
    loaded = transformers.AutoModelForCausalLM.from_pretrained(summary['checkpoint'])
    assert loaded.config.vocab_size == 5
