"""Ollama client wrapper with fixed generation settings.

All calls go to a local Ollama server (default http://localhost:11434), so the
experiment incurs no API cost. Generation settings are pinned here so that
every model/configuration cell of the experiment uses identical sampling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import requests


@dataclass
class GenerationResult:
    text: str
    latency_s: float
    prompt_tokens: int
    eval_tokens: int
    tokens_per_s: float


class OllamaClient:
    def __init__(self, model: str, url: str = "http://localhost:11434",
                 temperature: float = 0.0, seed: int = 42,
                 num_predict: int = 256, num_ctx: int = 8192,
                 timeout_s: int = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            # Pin the context window: prompts here average ~2.7k and peak ~3.9k
            # tokens (21 retrieved emails), above Ollama's default num_ctx, and
            # Ollama silently truncates prompts that exceed it.
            "num_ctx": num_ctx,
        }
        self.timeout_s = timeout_s

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
        }
        if system:
            payload["system"] = system
        t0 = time.perf_counter()
        resp = requests.post(f"{self.url}/api/generate", json=payload,
                             timeout=self.timeout_s)
        if not resp.ok:
            # Ollama answers e.g. 404 {"error": "model 'x' not found, try
            # pulling it first"} — surface the server's own message instead of
            # a bare status code, so a missing model is obvious in the log.
            raise RuntimeError(
                f"Ollama /api/generate failed with {resp.status_code} for "
                f"model {self.model}: {resp.text.strip()[:300]}")
        data = resp.json()
        latency = time.perf_counter() - t0
        eval_count = data.get("eval_count", 0) or 0
        return GenerationResult(
            text=data.get("response", ""),
            latency_s=latency,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            eval_tokens=eval_count,
            tokens_per_s=(eval_count / latency) if latency > 0 else 0.0,
        )

    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/tags", timeout=5).ok
        except requests.RequestException:
            return False

    def has_model(self, name: str) -> bool:
        """True when the local Ollama store contains this exact model tag."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException:
            return False
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        return name in names


class FakeLLM(OllamaClient):
    """Offline stand-in for OllamaClient (tests/CI): no network, no Ollama, deterministic.

    `script` is either a list of replies (cycled in call order) or a callable
    (prompt, system) -> reply. Every call is recorded in `calls` as (prompt, system).
    `delay_s` simulates generation time so latency bookkeeping can be tested.
    """

    def __init__(self, script: list[str] | Callable[[str, str | None], str] | None = None,
                 delay_s: float = 0.0, model: str = "fake"):
        super().__init__(model=model)
        self.script = script if script else [""]
        self.delay_s = delay_s
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        self.calls.append((prompt, system))
        if self.delay_s:
            time.sleep(self.delay_s)
        if callable(self.script):
            text = self.script(prompt, system)
        else:
            text = self.script[(len(self.calls) - 1) % len(self.script)]
        return GenerationResult(text=text, latency_s=self.delay_s, prompt_tokens=len(prompt.split()),
                                eval_tokens=len(text.split()), tokens_per_s=0.0)

    def is_up(self) -> bool:
        return True

    def has_model(self, name: str) -> bool:
        return True
