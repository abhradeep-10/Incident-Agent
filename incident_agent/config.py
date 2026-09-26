"""Runtime configuration. Every limit that protects against runaway cost lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _env_now() -> datetime:
    # The mock world is anchored to a fixed "now" so relative phrases such as
    # "yesterday afternoon" are reproducible. Override with AGENT_NOW.
    raw = os.getenv("AGENT_NOW", "2026-09-24T10:00:00+00:00").replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class AgentConfig:
    # --- LLM (any OpenAI-compatible server: Ollama, vLLM, llama.cpp, LM Studio) ---
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5-7b-16k"))
    temperature: float = 0.0
    llm_timeout_s: float = 180.0

    # --- per-turn budgets (loop / cost guards) ---
    max_tool_calls_per_turn: int = 12
    max_llm_calls_per_turn: int = 10
    max_duplicate_calls_per_turn: int = 3
    max_consecutive_failed_rounds: int = 3
    report_repair_attempts: int = 1

    # --- tool execution ---
    max_parallel_tools: int = 4
    tool_timeout_s: float = 5.0
    tool_max_retries: int = 2          # retries for transient errors / timeouts (idempotent tools only)
    retry_backoff_s: float = 0.25      # exponential: backoff * 2**attempt
    max_tool_output_chars: int = 4000  # tool output is trimmed before it enters the context

    # --- argument validation ---
    max_window_hours: int = 72
    retention_days: int = 30

    # --- conversation state ---
    history_turns: int = 4             # older turns are dropped; the evidence ledger persists

    now: datetime = field(default_factory=_env_now)
