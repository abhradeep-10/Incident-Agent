"""Agent loop: the LLM plans (chooses tools), the runtime enforces (budgets, validation, approval)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import AgentConfig
from .executor import ToolExecutor, TurnBudget
from .llm import LLMClient
from .prompts import build_system_prompt
from .report import check_report
from .state import PendingAction, Session
from .tools import ToolRegistry
from .tracing import Tracer


@dataclass
class AgentReply:
    text: str
    stop_reason: str
    tool_trace: list[dict] = field(default_factory=list)
    pending_actions: list[PendingAction] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0

    @property
    def tools_called(self) -> list[str]:
        return [t["tool"] for t in self.tool_trace]


class IncidentAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, config: AgentConfig | None = None,
                 session: Session | None = None, tracer: Tracer | None = None):
        self.cfg = config or AgentConfig()
        self.llm, self.registry = llm, registry
        self.session = session or Session()
        self.tracer = tracer or Tracer()
        self.executor = ToolExecutor(registry, self.cfg, self.session)

    # ------------------------------------------------------------------ main loop
    def ask(self, user_text: str) -> AgentReply:
        s, cfg = self.session, self.cfg
        s.turn += 1
        self.executor.new_turn()
        s.messages.append({"role": "user", "content": user_text, "_turn": s.turn})
        self.tracer.event("user", turn=s.turn, text=user_text)

        budget = TurnBudget.from_config(cfg)
        ev_start = s.ev_counter
        trace: list[dict] = []
        tools_enabled, nudge, stop_reason, repairs = True, None, None, 0

        while True:
            if budget.llm_calls >= cfg.max_llm_calls_per_turn:
                stop_reason = stop_reason or "llm_budget_exhausted"
                text = self._fallback_report(stop_reason)
                break
            try:
                resp = self.llm.chat(self._build_messages(nudge), self.registry.schemas() if tools_enabled else None)
            except Exception as e:
                stop_reason = "llm_error"
                self.tracer.event("llm_error", error=str(e))
                text = self._fallback_report(f"the language model call failed: {e}")
                break
            budget.llm_calls += 1
            self.tracer.event("llm", n=budget.llm_calls, tools_enabled=tools_enabled,
                              tool_calls=[f"{c.name}({c.arguments})" for c in resp.tool_calls],
                              content=(resp.content or "")[:300], usage=resp.usage)

            if resp.tool_calls and tools_enabled:
                s.messages.append({"role": "assistant", "content": resp.content or "",
                                   "tool_calls": [c.to_openai() for c in resp.tool_calls], "_turn": s.turn})
                results = self.executor.run_batch(resp.tool_calls, budget)
                for r in results:
                    s.messages.append({"role": "tool", "tool_call_id": r.call_id,
                                       "content": r.to_content(cfg.max_tool_output_chars), "_turn": s.turn})
                    trace.append(r.trace())
                    self.tracer.event("tool", **r.trace(), error=r.error)
                budget.failed_rounds = budget.failed_rounds + 1 if all(not r.ok for r in results) else 0
                reason = budget.stop_reason()
                if reason:
                    stop_reason, tools_enabled = reason, False
                    nudge = (f"[runtime] Tool use has been stopped ({reason}). Do not request tools. Write your "
                             "final answer now using only the evidence already gathered, and list what is missing.")
                continue

            text = (resp.content or "").strip()
            if not text:
                if not tools_enabled:
                    text = self._fallback_report(stop_reason or "the model returned an empty answer")
                    break
                tools_enabled, nudge = False, "[runtime] Write your answer to the user now."
                continue

            issues = check_report(text, set(s.evidence), gathered_this_turn=s.ev_counter > ev_start)
            if issues and repairs < cfg.report_repair_attempts:
                repairs += 1
                self.tracer.event("report_repair", issues=issues)
                s.messages.append({"role": "assistant", "content": text, "_turn": s.turn, "_draft": True})
                s.messages.append({"role": "user", "_turn": s.turn, "_runtime": True, "content":
                                   "[runtime check] Your answer has problems: " + "; ".join(issues) +
                                   ". Rewrite the complete answer fixing them. Cite only evidence ids that exist "
                                   "in the ledger. Do not call tools."})
                tools_enabled, nudge = False, None
                continue
            if issues:
                text += "\n\n> Automated evidence check: " + "; ".join(issues)
            break

        s.messages = [m for m in s.messages if not (m.get("_draft") or m.get("_runtime"))]
        s.messages.append({"role": "assistant", "content": text, "_turn": s.turn})
        stop = stop_reason or "final_answer"
        self.tracer.event("final", turn=s.turn, stop_reason=stop, tool_calls=budget.tool_calls,
                          llm_calls=budget.llm_calls)
        return AgentReply(text, stop, trace, [a for a in s.pending if a.turn == s.turn],
                          budget.llm_calls, budget.tool_calls)

    # ------------------------------------------------------------------ human approval (outside the LLM)
    def approve(self, action_id: str) -> str:
        a = self.session.actions.get(action_id)
        if a is None or a.status != "pending":
            return f"No pending action with id {action_id}."
        r = self.executor.execute_approved(a)
        if r.ok:
            a.status, a.result = "approved_executed", f"executed [{r.evidence_id}]"
            msg = (f"[Human approval] Action {a.id} {a.tool} {json.dumps(a.args)} was APPROVED by the user and "
                   f"executed successfully [{r.evidence_id}]: {json.dumps(r.data)}")
        else:
            a.status, a.result = "approved_failed", r.error or "failed"
            msg = f"[Human approval] Action {a.id} {a.tool} was APPROVED but FAILED: {r.error}"
        self.session.messages.append({"role": "assistant", "content": msg, "_turn": self.session.turn})
        self.tracer.event("approval", action=a.id, approved=True, ok=r.ok)
        return msg

    def reject(self, action_id: str) -> str:
        a = self.session.actions.get(action_id)
        if a is None or a.status != "pending":
            return f"No pending action with id {action_id}."
        a.status = "rejected"
        msg = f"[Human approval] Action {a.id} {a.tool} {json.dumps(a.args)} was REJECTED by the user. Not executed."
        self.session.messages.append({"role": "assistant", "content": msg, "_turn": self.session.turn})
        self.tracer.event("approval", action=a.id, approved=False)
        return msg

    # ------------------------------------------------------------------ helpers
    def _build_messages(self, nudge: str | None) -> list[dict]:
        s = self.session
        msgs = [{"role": "system", "content": build_system_prompt(self.cfg, s)}]
        min_turn = s.turn - self.cfg.history_turns + 1  # whole turns are dropped, so tool pairs stay intact
        for m in s.messages:
            t = m.get("_turn", s.turn)
            if t < min_turn:
                continue
            clean = {k: v for k, v in m.items() if not k.startswith("_")}
            if clean["role"] == "tool" and t < s.turn and len(clean["content"]) > 600:
                clean["content"] = clean["content"][:600] + " ...(older output trimmed; see evidence ledger)"
            msgs.append(clean)
        if nudge:
            msgs.append({"role": "user", "content": nudge})
        return msgs

    def _fallback_report(self, reason: str) -> str:
        """Deterministic answer when the LLM cannot produce one. Contains only recorded evidence."""
        s = self.session
        new = [e for e in s.evidence.values() if e.turn == s.turn]
        fails = [f for f in s.failures if f["turn"] == s.turn]
        lines = [f"I could not complete a full answer ({reason}). This is what was established so far.", "",
                 "## Observed facts"]
        lines += [f"- {e.summary} [{e.id}]" for e in new] or ["- No tool evidence was gathered in this turn."]
        if fails:
            lines += ["", "## Gaps and tool issues"]
            lines += [f"- {f['tool']} {json.dumps(f['args'])}: {f['error_type']} ({f['error']})" for f in fails]
        lines += ["", "## Recommended next actions",
                  "1. Re-run the question, or narrow it to one service and time window.",
                  "2. Check any failed data sources above directly."]
        return "\n".join(lines)
