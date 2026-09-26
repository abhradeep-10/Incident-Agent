"""Conversation + investigation state.

Two layers:
  * messages: the raw chat history (trimmed to the last N turns when sent to the LLM)
  * evidence ledger: every successful tool result gets a stable id (E1, E2...) and a one-line
    summary. The ledger is injected into the system prompt on every call, so follow-up questions
    can reuse earlier findings even after old raw tool outputs have been trimmed away.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Evidence:
    id: str
    turn: int
    tool: str
    args: dict
    summary: str
    data: Any


@dataclass
class PendingAction:
    id: str
    turn: int
    tool: str
    args: dict
    status: str = "pending"  # pending | approved_executed | approved_failed | rejected
    result: str = ""


class Session:
    def __init__(self):
        self.turn = 0
        self.messages: list[dict] = []
        self.evidence: dict[str, Evidence] = {}
        self.failures: list[dict] = []
        self.actions: dict[str, PendingAction] = {}
        self.call_cache: dict[str, str] = {}  # canonical call key -> evidence id
        self.ev_counter = 0
        self.action_counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ mutation
    def add_evidence(self, tool: str, args: dict, data: Any, summary: str) -> Evidence:
        with self._lock:
            self.ev_counter += 1
            ev = Evidence(f"E{self.ev_counter}", self.turn, tool, args, summary, data)
            self.evidence[ev.id] = ev
            return ev

    def add_failure(self, tool: str, args: dict, error_type: str, error: str):
        self.failures.append({"turn": self.turn, "tool": tool, "args": args,
                              "error_type": error_type, "error": error})

    def add_pending(self, tool: str, args: dict) -> PendingAction:
        with self._lock:
            self.action_counter += 1
            pa = PendingAction(f"A{self.action_counter}", self.turn, tool, args)
            self.actions[pa.id] = pa
            return pa

    @property
    def pending(self) -> list[PendingAction]:
        return [a for a in self.actions.values() if a.status == "pending"]

    # ------------------------------------------------------------ prompt view
    def ledger_text(self, max_items: int = 40) -> str:
        if not (self.evidence or self.failures or self.actions):
            return ""
        out = ["", "EVIDENCE LEDGER (results of earlier tool calls in this conversation; cite these ids):"]
        items = list(self.evidence.values())[-max_items:]
        out += [f"- {e.id} [turn {e.turn}] {e.tool}: {e.summary}" for e in items] or ["- (none)"]
        if self.failures:
            out.append("TOOL FAILURES SO FAR (no data was obtained from these calls):")
            for f in self.failures[-15:]:
                out.append(f"- [turn {f['turn']}] {f['tool']} {json.dumps(f['args'])}: "
                           f"{f['error_type']} - {f['error']}")
        if self.actions:
            out.append("ACTIONS REQUIRING HUMAN APPROVAL:")
            for a in self.actions.values():
                out.append(f"- {a.id} {a.tool} {json.dumps(a.args)}: {a.status}"
                           + (f" ({a.result})" if a.result else ""))
        return "\n".join(out)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"turn": self.turn, "messages": self.messages,
                "evidence": [asdict(e) for e in self.evidence.values()],
                "failures": self.failures, "actions": [asdict(a) for a in self.actions.values()],
                "call_cache": self.call_cache, "ev_counter": self.ev_counter,
                "action_counter": self.action_counter}

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls()
        s.turn, s.messages = d["turn"], d["messages"]
        s.evidence = {e["id"]: Evidence(**e) for e in d["evidence"]}
        s.failures = d["failures"]
        s.actions = {a["id"]: PendingAction(**a) for a in d["actions"]}
        s.call_cache, s.ev_counter, s.action_counter = d["call_cache"], d["ev_counter"], d["action_counter"]
        return s

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path) as f:
            return cls.from_dict(json.load(f))
