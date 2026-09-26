"""LLM adapters.

OpenAICompatLLM talks to any OpenAI-compatible server hosting an open-weights model
(Ollama, vLLM, llama.cpp server, LM Studio). No proprietary API is used.
ScriptedLLM is a deterministic stand-in used by the automated tests.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol, Union


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"  # JSON string, exactly as produced by the model

    def to_openai(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def parse_text_tool_calls(content: str, valid_names: set[str]) -> tuple[list[ToolCall], str]:
    """Fallback for open models that print tool calls as text instead of using the tool_calls field
    (e.g. Qwen's <tool_call>{...}</tool_call> or a bare JSON object). Only known tool names are accepted."""
    if not content:
        return [], content
    blobs = _TAGGED.findall(content)
    remaining = _TAGGED.sub("", content)
    if not blobs:
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            blobs, remaining = [stripped], ""
        else:
            blobs = _FENCED.findall(content)
            remaining = _FENCED.sub("", content)
    calls: list[ToolCall] = []
    for blob in blobs:
        try:
            obj = json.loads(blob)
        except (ValueError, TypeError):
            continue
        for item in obj if isinstance(obj, list) else [obj]:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = item.get("name") or fn.get("name")
            args = item.get("arguments", item.get("parameters", fn.get("arguments", {})))
            if name in valid_names:
                calls.append(ToolCall(new_call_id(), name, args if isinstance(args, str) else json.dumps(args or {})))
    if not calls:
        return [], content
    return calls, remaining.strip()


class OpenAICompatLLM:
    def __init__(self, cfg):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.llm_timeout_s, max_retries=2)

    def chat(self, messages, tools=None) -> LLMResponse:
        kwargs = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id or new_call_id(), tc.function.name, tc.function.arguments or "{}")
                 for tc in (msg.tool_calls or []) if getattr(tc, "function", None)]
        content = msg.content or ""
        if tools and not calls:
            calls, content = parse_text_tool_calls(content, {t["function"]["name"] for t in tools})
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return LLMResponse(content=content, tool_calls=calls, usage=usage)


Step = Union[LLMResponse, Callable[[list, list], LLMResponse]]


class ScriptedLLM:
    """Replays a fixed list of responses (or callables(messages, tools) -> LLMResponse)."""

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None) -> LLMResponse:
        self.calls.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools})
        if not self.steps:
            return LLMResponse(content="(script exhausted)")
        step = self.steps.pop(0)
        return step(messages, tools) if callable(step) else step
