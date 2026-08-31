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
LABEL_CACHE = PACKAGE_ROOT / "labels" / "notes.json"

DEFAULT_MODEL = "command-r7b-12-2024"

# Which vendor a model name belongs to. Inferred rather than configured because
# the two always travel together — a Cohere model reached with an Anthropic
# client is not a setting anyone wants, it is a typo. `BREW_AGENT_PROVIDER`
# overrides for anything the prefixes don't know.
PROVIDER_PREFIXES = (
    ("claude", "anthropic"),
    ("command", "cohere"),
    ("north", "cohere"),
    ("c4ai", "cohere"),
)

API_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
}

# Output ceilings, which are a model property rather than a preference.
# Claude Opus 5 thinks by default and `max_tokens` caps thinking plus response
# text together, so leave real headroom there. Command R7B's ceiling is 4000
# and a request above it is rejected outright — this harness's longest answer
# is a few hundred tokens, so the ceiling is the only thing that matters.
DEFAULT_MAX_TOKENS = {
    "anthropic": 16000,
    "cohere": 4000,
}

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


def provider_for(model: str) -> str:
    """Which vendor serves this model name.

    Refuses to guess. A wrong guess here surfaces as an authentication error
    against the wrong API, which reads like a bad key rather than a bad model
    name and costs an afternoon.
    """
    override = os.environ.get("BREW_AGENT_PROVIDER", "").strip().lower()
    if override:
        if override not in API_KEY_VARS:
            raise ConfigError(
                f"BREW_AGENT_PROVIDER={override!r} is not one of "
                f"{', '.join(sorted(API_KEY_VARS))}."
            )
        return override
    for prefix, provider in PROVIDER_PREFIXES:
        if model.lower().startswith(prefix):
            return provider
    raise ConfigError(
        f"Cannot tell which provider serves {model!r}. Set BREW_AGENT_PROVIDER "
        f"to one of {', '.join(sorted(API_KEY_VARS))}."
    )


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    api_key: str
    model: str
    effort: str | None
    max_iterations: int
    max_tokens: int

    @classmethod
    def from_env(cls, model: str | None = None) -> "ModelConfig":
        """`model` overrides `BREW_AGENT_MODEL` for one caller.

        The labelling pass is the only thing that uses it, and it is shared
        infrastructure rather than an arm — see `eval/labels.py`. Every arm
        must stay on one model or the ladder stops pricing capability and
        starts pricing the model swap.
        """
        model = model or os.environ.get("BREW_AGENT_MODEL", DEFAULT_MODEL)
        provider = provider_for(model)
        return cls(
            provider=provider,
            api_key=_require(API_KEY_VARS[provider]),
            model=model,
            effort=os.environ.get("BREW_AGENT_EFFORT") or None,
            max_iterations=int(os.environ.get("BREW_AGENT_MAX_ITERATIONS", "6")),
            max_tokens=int(
                os.environ.get("BREW_AGENT_MAX_TOKENS")
                or DEFAULT_MAX_TOKENS[provider]
            ),
        )
