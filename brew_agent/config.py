"""Environment configuration.

Everything here is read from `internal/brew-agent/.env` (gitignored). See
`.env.example` for the variable names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PACKAGE_ROOT / "traces"
OUTPUT_DIR = PACKAGE_ROOT / "evals" / "output"

# Claude Opus 5 thinks by default and `max_tokens` caps thinking plus response
# text together, so leave real headroom here.
MAX_TOKENS = 16000

load_dotenv(PACKAGE_ROOT / ".env")


class ConfigError(RuntimeError):
    """A required environment variable is missing."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy internal/brew-agent/.env.example to .env "
            f"and fill it in."
        )
    return value


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    anon_key: str
    email: str
    password: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        return cls(
            url=_require("SUPABASE_URL"),
            anon_key=_require("SUPABASE_ANON_KEY"),
            email=_require("BREW_AGENT_EMAIL"),
            password=_require("BREW_AGENT_PASSWORD"),
        )


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    model: str
    effort: str | None
    max_iterations: int

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            api_key=_require("ANTHROPIC_API_KEY"),
            model=os.environ.get("BREW_AGENT_MODEL", "claude-opus-5"),
            effort=os.environ.get("BREW_AGENT_EFFORT") or None,
            max_iterations=int(os.environ.get("BREW_AGENT_MAX_ITERATIONS", "6")),
        )
