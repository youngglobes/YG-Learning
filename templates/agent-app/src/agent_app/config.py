"""Central configuration. The ONE place the model is chosen.

The model is read from the environment, so switching provider never touches
application code:

    AGENT_MODEL=openai:gpt-5.5                 # or google_genai:..., anthropic:...,
    AGENT_MODEL=ollama:llama3.1:8b             # groq:..., mistralai:..., or any of 23
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings, all overridable by environment variable."""

    # --- model ----------------------------------------------------------
    # Local default so the template runs with no key. Override in .env.
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "ollama:llama3.1:8b"))

    # --- guardrails (Module 3) ------------------------------------------
    # Every agent gets a cap. This is not optional.
    max_model_calls: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_MODEL_CALLS", "8")))
    recursion_limit: int = field(default_factory=lambda: int(os.getenv("AGENT_RECURSION_LIMIT", "25")))

    # --- retrieval (Module 6) -------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4")))

    # --- persistence (Module 5) -----------------------------------------
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", "checkpoints.db"))

    @property
    def is_local(self) -> bool:
        """True when running against Ollama."""
        return self.model.startswith("ollama:")


settings = Settings()
