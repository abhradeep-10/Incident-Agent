"""Safe tool execution layer. The LLM proposes calls; this layer decides what actually runs.

Pipeline per call: known tool? -> budget -> parse JSON args -> validate args -> dedupe ->
approval gate -> execute with timeout + retries -> validate output -> record evidence.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import AgentConfig
from .errors import ToolArgError, ToolError, ToolMalformedError, ToolTimeoutError, ToolTransientError
from .llm import ToolCall
from .state import PendingAction, Session
from .summarize import summarize
from .timeutil import iso
from .tools import ToolRegistry, ToolSpec


def jsonable(obj):
    if isinstance(obj, datetime):
        return iso(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def canonical_key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(jsonable(args), sort_keys=True)}"


@dataclass
class TurnBudget:
    max_tool_calls: int
    max_duplicates: int
    max_failed_rounds: int
    tool_calls: int = 0
    llm_calls: int = 0
    duplicates: int = 0
    failed_rounds: int = 0

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> "TurnBudget":
        return cls(cfg.max_tool_calls_per_turn, cfg.max_duplicate_calls_per_turn, cfg.max_consecutive_failed_rounds)

    def take_tool_call(self) -> bool:
        if self.tool_calls >= self.max_tool_calls:
            return False
        self.tool_calls += 1
        return True

    def stop_reason(self) -> str | None:
        if self.tool_calls >= self.max_tool_calls:
            return "tool_budget_exhausted"
        if self.duplicates >= self.max_duplicates:
            return "loop_detected"
        if self.failed_rounds >= self.max_failed_rounds:
            return "repeated_tool_failures"
        return None


@dataclass
class ToolResult:
    call_id: str
    name: str
    args: dict
    ok: bool
    status: str = "executed"  # executed | duplicate | rejected | failed | pending_approval
    data: Any = None
    error: str | None = None
    error_type: str | None = None
    evidence_id: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    notes: list[str] = field(default_factory=list)

    def to_content(self, max_chars: int = 4000) -> str:
        payload: dict = {"ok": self.ok, "status": self.status}
        if self.evidence_id:
            payload["evidence_id"] = self.evidence_id
        if self.notes:
            payload["notes"] = self.notes
        if self.error:
            payload.update(error_type=self.error_type, error=self.error, attempts=self.attempts)
        if self.data is not None:
            payload["data"] = self.data
        text = json.dumps(payload, default=str)
        if len(text) > max_chars and isinstance(self.data, dict):
            data = json.loads(json.dumps(self.data, default=str))
            for key in ("points", "events"):
                while (isinstance(data.get(key), list) and len(data[key]) > 6
                       and len(json.dumps({**payload, "data": data}, default=str)) > max_chars):
                    data[key] = data[key][::2]
                    data["truncated_for_context"] = True
            payload["data"] = data
            text = json.dumps(payload, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + " ...(truncated)"
        return text

    def trace(self) -> dict:
        return {"tool": self.name, "args": self.args, "ok": self.ok, "status": self.status,
                "error_type": self.error_type, "evidence_id": self.evidence_id,
                "attempts": self.attempts, "duration_ms": self.duration_ms}


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, cfg: AgentConfig, session: Session):
        self.registry, self.cfg, self.session = registry, cfg, session
        self._call_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
        self._batch_pool = ThreadPoolExecutor(max_workers=cfg.max_parallel_tools, thread_name_prefix="batch")
        self._turn_failures: Counter = Counter()

    def new_turn(self):
        self._turn_failures.clear()

    # ------------------------------------------------------------ public
    def run_batch(self, calls: list[ToolCall], budget: TurnBudget) -> list[ToolResult]:
        results: list[ToolResult | None] = [None] * len(calls)
        ready = []
        seen_in_batch: set[str] = set()
        for i, call in enumerate(calls):
            pre = self._prepare(call, budget, seen_in_batch)
            if isinstance(pre, ToolResult):
                results[i] = pre
            else:
                ready.append((i, call, *pre))

        outcomes = {}
        read_only = [r for r in ready if not r[2].side_effect]
        side_effects = [r for r in ready if r[2].side_effect]
        futures = {self._batch_pool.submit(self._invoke, r[2], r[3]): r for r in read_only}  # parallel
        for fut, r in futures.items():
            outcomes[r[0]] = (r, fut.result())
        for r in side_effects:  # sequential, never in parallel
            outcomes[r[0]] = (r, self._invoke(r[2], r[3]))
        for i in sorted(outcomes):  # finalise in call order -> deterministic evidence ids
            (_, call, spec, args, notes, key), outcome = outcomes[i]
            results[i] = self._finalize(call, spec, args, notes, key, outcome)
        return results  # type: ignore[return-value]

    def execute_approved(self, action: PendingAction) -> ToolResult:
        spec = self.registry.get(action.tool)
        call = ToolCall(f"approved-{action.id}", action.tool, json.dumps(action.args))
        if spec is None:
            return self._fail(call, action.args, "unknown_tool", f"unknown tool {action.tool}")
        try:  # re-validate: the world may have changed since the proposal
            args = spec.validate_args(dict(action.args))
        except ToolArgError as e:
            return self._fail(call, action.args, "invalid_arguments", str(e))
        notes = args.pop("_notes", [])
        return self._finalize(call, spec, args, notes, canonical_key(spec.name, args), self._invoke(spec, args))

    # ------------------------------------------------------------ internals
    def _fail(self, call, args, etype, msg, status="rejected") -> ToolResult:
        return ToolResult(call.id, call.name, jsonable(args or {}), ok=False, status=status,
                          error=msg, error_type=etype)

    def _prepare(self, call: ToolCall, budget: TurnBudget, seen_in_batch: set):
        spec = self.registry.get(call.name)
        if spec is None:
            return self._fail(call, {}, "unknown_tool",
                              f"unknown tool '{call.name}'. Available: {', '.join(self.registry.names)}")
        if not budget.take_tool_call():
            return self._fail(call, {}, "budget_exhausted",
                              "tool-call budget for this turn is exhausted; answer with the evidence you have")
        try:
            raw = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
            raw = {} if raw is None else raw
            if not isinstance(raw, dict):
                raise ValueError
        except (ValueError, TypeError):
            return self._fail(call, {}, "invalid_arguments", "arguments are not a valid JSON object")
        try:
            args = spec.validate_args(raw)
        except ToolArgError as e:
            return self._fail(call, raw, "invalid_arguments", str(e))
        notes = args.pop("_notes", [])
        key = canonical_key(spec.name, args)

        if not spec.side_effect:
            ev_id = self.session.call_cache.get(key)
            if ev_id or key in seen_in_batch:
                budget.duplicates += 1
                ev = self.session.evidence.get(ev_id) if ev_id else None
                return ToolResult(call.id, spec.name, jsonable(args), ok=True, status="duplicate",
                                  evidence_id=ev_id,
                                  data={"duplicate_of": ev_id or "another call in this batch",
                                        "summary": ev.summary if ev else None,
                                        "note": "Identical call already made. Reuse that evidence; do not call again."})
        seen_in_batch.add(key)
        if self._turn_failures[key] >= 2:
            return self._fail(call, args, "repeated_failure",
                              "this exact call already failed twice this turn; do not retry it, report the gap")
        if spec.requires_approval:
            pa = self.session.add_pending(spec.name, jsonable(args))
            return ToolResult(call.id, spec.name, jsonable(args), ok=False, status="pending_approval",
                              error_type="pending_approval",
                              data={"action_id": pa.id, "status": "awaiting_human_approval",
                                    "message": "NOT executed. A human must approve this action outside the "
                                               "chat. Tell the user it is pending their approval."})
        return spec, args, notes, key

    def _invoke(self, spec: ToolSpec, args: dict):
        t0 = time.monotonic()
        retries = self.cfg.tool_max_retries if spec.idempotent else 0  # never retry a non-idempotent action
        last: ToolError | None = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts += 1
            try:
                fut = self._call_pool.submit(spec.run, **args)
                try:
                    data = fut.result(timeout=self.cfg.tool_timeout_s)
                except FutureTimeout:
                    fut.cancel()
                    raise ToolTimeoutError(f"{spec.name} did not respond within {self.cfg.tool_timeout_s}s") from None
                spec.validate_output(data)
                return data, attempts, None, int((time.monotonic() - t0) * 1000)
            except ToolTransientError as e:
                last = e
                if attempt < retries:
                    time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
            except ToolError as e:
                return None, attempts, e, int((time.monotonic() - t0) * 1000)
            except Exception as e:  # defensive: a buggy backend must not crash the agent
                return None, attempts, ToolError(f"unexpected {type(e).__name__}: {e}"), \
                    int((time.monotonic() - t0) * 1000)
        return None, attempts, last, int((time.monotonic() - t0) * 1000)

    def _finalize(self, call, spec, args, notes, key, outcome) -> ToolResult:
        data, attempts, err, ms = outcome
        jargs = jsonable(args)
        if err is None:
            jdata = jsonable(data)
            ev = self.session.add_evidence(spec.name, jargs, jdata, summarize(spec.name, jargs, jdata))
            if not spec.side_effect:
                self.session.call_cache[key] = ev.id
            return ToolResult(call.id, spec.name, jargs, ok=True, data=jdata, evidence_id=ev.id,
                              attempts=attempts, duration_ms=ms, notes=notes)
        self._turn_failures[key] += 1
        if isinstance(err, ToolTimeoutError):
            etype = "timeout"
        elif isinstance(err, ToolTransientError):
            etype = "transient_failure"
        elif isinstance(err, ToolMalformedError):
            etype = "malformed_response"
        elif isinstance(err, ToolArgError):
            etype = "invalid_arguments"
        else:
            etype = "tool_error"
        self.session.add_failure(spec.name, jargs, etype, str(err))
        return ToolResult(call.id, spec.name, jargs, ok=False, status="failed", error=str(err),
                          error_type=etype, attempts=attempts, duration_ms=ms, notes=notes)
