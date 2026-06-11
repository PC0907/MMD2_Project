"""Thin LLM inference wrapper.

Designed for offline batch inference on a GPU server via vLLM (default),
with a HF transformers fallback. Every call is cached on disk keyed by
(model, prompt) so that re-runs and ablations are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, value))
        self.conn.commit()


class LLMClient:
    """Batched chat completion with vLLM.

    Usage:
        client = LLMClient("meta-llama/Llama-3.3-70B-Instruct", tensor_parallel=4)
        outs = client.chat_batch(system, [user1, user2, ...])
    """

    def __init__(
        self,
        model_name: str,
        cache_path: str | Path = "outputs/llm_cache.sqlite",
        tensor_parallel: int = 1,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 13,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache = PromptCache(cache_path)
        self._llm = None
        self._tp = tensor_parallel
        self._max_model_len = max_model_len

    def _ensure_engine(self):
        if self._llm is None:
            from vllm import LLM
            logger.info("Loading vLLM engine: %s (TP=%d)", self.model_name, self._tp)
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self._tp,
                max_model_len=self._max_model_len,
                seed=self.seed,
            )

    def chat_batch(self, system: str, users: list[str]) -> list[str]:
        keys = [self.cache.key(self.model_name, system, u) for u in users]
        outputs: list[str | None] = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(outputs) if o is None]
        if todo:
            self._ensure_engine()
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=self.temperature, max_tokens=self.max_tokens, seed=self.seed
            )
            messages = [
                [{"role": "system", "content": system},
                 {"role": "user", "content": users[i]}]
                for i in todo
            ]
            results = self._llm.chat(messages, params)
            for i, res in zip(todo, results):
                text = res.outputs[0].text
                outputs[i] = text
                self.cache.put(keys[i], text)
        return outputs  # type: ignore[return-value]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str, default: dict | None = None) -> dict:
    """Extract the first JSON object from an LLM response, tolerating
    code fences and surrounding prose."""
    try:
        m = _JSON_RE.search(text)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    logger.debug("JSON parse failure on: %.200s", text)
    return default if default is not None else {}
