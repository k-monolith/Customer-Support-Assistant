"""Configuration loading: config/pipeline.yaml + environment variables."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config" / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["env"] = {
        # OpenAI-compatible generation endpoint (Ollama default)
        "gen_base_url": os.environ.get("GEN_BASE_URL", "http://localhost:11434/v1"),
        "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
        "artifacts_dir": os.environ.get("ARTIFACTS_DIR", str(ROOT / "artifacts")),
        # Langfuse tracing is optional: enabled only when all three are set
        "langfuse_public_key": os.environ.get("LANGFUSE_PUBLIC_KEY"),
        "langfuse_secret_key": os.environ.get("LANGFUSE_SECRET_KEY"),
        "langfuse_host": os.environ.get("LANGFUSE_HOST"),
    }
    return cfg


def load_prompt(version: str) -> tuple[str, str]:
    """Return (system, user_template) from prompts/<version>.txt."""
    text = (ROOT / "prompts" / f"{version}.txt").read_text()
    system, user = text.split("### USER")
    return system.replace("### SYSTEM", "").strip(), user.strip()
