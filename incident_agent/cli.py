"""Interactive CLI. Human approval for dangerous actions happens HERE, outside the LLM."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import AgentReply, IncidentAgent
from .config import AgentConfig
from .llm import OpenAICompatLLM
from .mock_data import MockBackend
from .state import Session
from .tools import build_registry
from .tracing import Tracer

HELP = "Commands: /evidence  /actions  /notes  /reset  /save  /quit"


def _print_reply(reply: AgentReply):
    print("\n" + reply.text + "\n")
    calls = ", ".join(f"{t['tool']}[{t['status']}{'' if t['ok'] else ':' + str(t['error_type'])}]"
                      for t in reply.tool_trace) or "none"
    print(f"  (tools: {calls} | llm calls: {reply.llm_calls} | stop: {reply.stop_reason})")


def _handle_approvals(agent: IncidentAgent, reply: AgentReply, interactive: bool):
    for a in reply.pending_actions:
        print(f"\n!! The agent proposes a DANGEROUS action {a.id}: {a.tool} {json.dumps(a.args)}")
        if not interactive:
            print("   Non-interactive session: left pending and NOT executed.")
            continue
        ans = input(f"   Type exactly 'approve {a.id}' to execute it; anything else rejects it: ").strip()
        print("   " + (agent.approve(a.id) if ans.lower() == f"approve {a.id}".lower() else agent.reject(a.id)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Incident investigation agent (open-source LLM)")
    p.add_argument("-q", "--query", action="append", help="ask one or more questions non-interactively")
    p.add_argument("--model", help="model name on the server (default: $LLM_MODEL or qwen2.5-14b-16k)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: Ollama on localhost)")
    p.add_argument("--trace", action="store_true", help="print trace events to stderr")
    p.add_argument("--trace-file", default="traces/trace.jsonl")
    p.add_argument("--session", help="JSON file to load/save conversation state")
    p.add_argument("--no-faults", action="store_true", help="disable the default injected tool failures")
    a = p.parse_args(argv)

    cfg = AgentConfig()
    if a.model:
        cfg.model = a.model
    if a.base_url:
        cfg.base_url = a.base_url
    backend = MockBackend(now=cfg.now, faults={} if a.no_faults else None)
    session = Session.load(a.session) if a.session and os.path.exists(a.session) else Session()
    agent = IncidentAgent(OpenAICompatLLM(cfg), build_registry(backend, cfg), cfg, session,
                          Tracer(a.trace_file, echo=a.trace))
    print(f"Incident agent | model={cfg.model} @ {cfg.base_url} | now={cfg.now.isoformat()}")

    def run(q: str, interactive: bool):
        reply = agent.ask(q)
        _print_reply(reply)
        _handle_approvals(agent, reply, interactive)

    try:
        if a.query:
            for q in a.query:
                print(f"\n>>> {q}")
                run(q, interactive=sys.stdin.isatty())
            return
        print(HELP)
        while True:
            try:
                q = input("\nyou> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q == "/quit":
                break
            if q == "/help":
                print(HELP)
            elif q == "/evidence":
                print(agent.session.ledger_text() or "(empty)")
            elif q == "/actions":
                for act in agent.session.actions.values():
                    print(f"{act.id} {act.tool} {act.args} -> {act.status}")
            elif q == "/notes":
                print(json.dumps(backend.notes, indent=2) if backend.notes else "(no notes)")
            elif q == "/reset":
                agent.session = Session()
                agent.executor.session = agent.session
                print("conversation reset")
            elif q == "/save":
                agent.session.save(a.session or "session.json")
                print("saved")
            else:
                run(q, interactive=True)
    finally:
        if a.session:
            agent.session.save(a.session)


if __name__ == "__main__":
    main()
