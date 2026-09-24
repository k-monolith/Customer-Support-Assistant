"""Generation client for any OpenAI-compatible endpoint (Ollama locally, vLLM elsewhere)."""
import time

import requests


class Generator:
    def __init__(self, cfg: dict):
        self.base_url = cfg["env"]["gen_base_url"].rstrip("/")
        self.model = cfg["generation"]["model"]
        self.temperature = cfg["generation"]["temperature"]
        self.max_tokens = cfg["generation"]["max_tokens"]

    def generate(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        t0 = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }, timeout=600)
        r.raise_for_status()
        out = r.json()
        return {"text": out["choices"][0]["message"]["content"],
                "usage": out.get("usage", {}),
                "gen_seconds": round(time.time() - t0, 2),
                "model": self.model}
