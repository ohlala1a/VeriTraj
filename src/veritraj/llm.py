"""Minimal client for OpenAI-compatible chat endpoints (DeepSeek, GLM, vLLM, ...).

Configured from the environment:
  VERITRAJ_LLM_BASE_URL   e.g. https://api.deepseek.com/v1
  VERITRAJ_LLM_API_KEY
  VERITRAJ_LLM_MODEL
Uses only the standard library. Used by the LLM policy, the llm_impute
grounding policy, the LLM-judge baseline and the Builder/Challenger roles.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 512) -> str: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    @classmethod
    def from_env(cls, prefix: str = "VERITRAJ_LLM") -> "OpenAICompatibleClient":
        missing = [k for k in ("BASE_URL", "API_KEY", "MODEL") if not os.environ.get(f"{prefix}_{k}")]
        if missing:
            raise LLMError("set " + ", ".join(f"{prefix}_{k}" for k in missing) + " to use an LLM backend")
        return cls(os.environ[f"{prefix}_BASE_URL"], os.environ[f"{prefix}_API_KEY"],
                   os.environ[f"{prefix}_MODEL"])

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                usage = data.get("usage") or {}
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.usage["calls"] += 1
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(str(e)) from e
        raise LLMError("unreachable")


def extract_block(text: str, lang: str = "") -> str | None:
    m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_json(text: str):
    block = extract_block(text, "json") or text
    m = re.search(r"(\{.*\}|\[.*\])", block, re.DOTALL)
    if not m:
        raise LLMError(f"no JSON in response: {text[:200]!r}")
    return json.loads(m.group(1))


def make_class_resolver(client: ChatClient) -> Callable[[str, list[str]], str | None]:
    """Resolver for the llm_impute grounding policy: unknown surface class -> lexicon class."""

    def resolve(mention: str, classes: list[str]) -> str | None:
        msgs = [{"role": "system", "content": "Map an object name from a household text game to one "
                 "class from the list. Answer with the class only, or NONE if no class fits."},
                {"role": "user", "content": f"Name: {mention}\nClasses: {', '.join(classes)}"}]
        ans = client.chat(msgs, temperature=0.0, max_tokens=8).strip().lower().strip(".")
        return None if ans in {"none", ""} else ans

    return resolve
