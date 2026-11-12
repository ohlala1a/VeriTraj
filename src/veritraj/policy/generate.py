"""Generation preserves prompt and sampled token IDs."""
from dataclasses import dataclass

@dataclass
class Generation:
    text: str
    prompt_token_ids: list[int]
    token_ids: list[int]

class HFGenerator:
    def __init__(self, model, tokenizer, **generation_kwargs):
        self.model, self.tokenizer = model, tokenizer
        self.kwargs = {"max_new_tokens": 128, "do_sample": True,
                       "pad_token_id": tokenizer.pad_token_id, **generation_kwargs}

    def generate(self, prompt):
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                ids = self.model.generate(**inputs, **self.kwargs)[0]
        finally:
            self.model.train(training)
        prefix = inputs["input_ids"][0].tolist()
        sampled = ids[len(prefix):].tolist()
        return Generation(self.tokenizer.decode(sampled, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False), prefix, sampled)

class VLLMGenerator:
    def __init__(self, model, **kwargs):
        from vllm import LLM, SamplingParams
        sampling = dict(kwargs.pop("sampling", {}))
        sampling["max_tokens"] = sampling.pop("max_new_tokens", sampling.get("max_tokens", 128))
        if not sampling.pop("do_sample", True):
            sampling["temperature"] = 0.0
        self.engine = LLM(model=model, **kwargs)
        self.sampling = SamplingParams(**sampling)

    def generate(self, prompt):
        result = self.engine.generate([prompt], self.sampling, use_tqdm=False)[0]
        output = result.outputs[0]
        return Generation(output.text, list(result.prompt_token_ids), list(output.token_ids))

def load_generator(spec):
    if spec.get("backend", "hf") == "vllm":
        return VLLMGenerator(spec["model"], sampling=spec.get("generation", {}),
                             **spec.get("vllm", {}))
    if spec.get("backend", "hf") != "hf":
        raise ValueError("backend must be hf or vllm")
    from ..train.trainer import load_policy
    model, tokenizer = load_policy(spec)
    return HFGenerator(model, tokenizer, **spec.get("generation", {}))
