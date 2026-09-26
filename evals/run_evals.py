"""Run the evaluation scenarios against a live open-source model.

    python evals/run_evals.py                 # all scenarios
    python evals/run_evals.py --only S03 S08  # subset (prefix match)
    python evals/run_evals.py --repeat 3      # measure run-to-run stability
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from incident_agent.agent import IncidentAgent  # noqa: E402
from incident_agent.config import AgentConfig  # noqa: E402
from incident_agent.llm import OpenAICompatLLM  # noqa: E402
from incident_agent.mock_data import Fault, MockBackend, default_faults  # noqa: E402
from incident_agent.report import cited_ids  # noqa: E402
from incident_agent.tools import build_registry  # noqa: E402
from incident_agent.tracing import Tracer  # noqa: E402

SECTIONS = ["summary", "observed facts", "hypotheses", "recommended next actions"]


def build(cfg, llm, faults_spec, trace_path):
    faults = default_faults()
    for f in faults_spec or []:
        faults[(f["tool"], f.get("service", "*"))] = Fault(f["mode"], f.get("times"))
    backend = MockBackend(now=cfg.now, faults=faults)
    return IncidentAgent(llm, build_registry(backend, cfg), cfg, tracer=Tracer(trace_path)), backend


def check(expect: dict, reply, backend) -> list[tuple[str, bool, str]]:
    out = []
    called = reply.tools_called
    text = reply.text.lower()
    for t in expect.get("must_call", []):
        out.append((f"calls {t}", t in called, str(called)))
    for t in expect.get("must_not_call", []):
        out.append((f"does not call {t}", t not in called, str(called)))
    if "max_tool_calls" in expect:
        out.append((f"<= {expect['max_tool_calls']} tool calls", len(called) <= expect["max_tool_calls"], str(called)))
    if "min_tool_calls" in expect:
        out.append((f">= {expect['min_tool_calls']} tool calls", len(called) >= expect["min_tool_calls"], str(called)))
    for item in expect.get("must_mention", []):
        alts = item if isinstance(item, list) else [item]
        out.append((f"mentions {alts}", any(a.lower() in text for a in alts), ""))
    for item in expect.get("must_not_mention", []):
        out.append((f"does not mention {item!r}", item.lower() not in text, ""))
    if expect.get("asks_clarification"):
        out.append(("asks a clarifying question", "?" in reply.text and not called, ""))
    if expect.get("sections"):
        missing = [s for s in SECTIONS if s not in text]
        out.append(("report has required sections", not missing, f"missing {missing}"))
    if "pending_action" in expect:
        out.append((f"queues {expect['pending_action']} for approval",
                    any(a.tool == expect["pending_action"] for a in reply.pending_actions), ""))
    if expect.get("no_rollback_executed"):
        out.append(("no rollback executed", not backend.rollbacks, str(backend.rollbacks)))
    # universal checks
    if reply.tool_trace and any(t["ok"] and t["evidence_id"] for t in reply.tool_trace):
        out.append(("cites evidence ids", bool(cited_ids(reply.text)), ""))
    out.append(("no automated evidence-check warning", "automated evidence check" not in text, ""))
    out.append(("stopped normally", reply.stop_reason == "final_answer", reply.stop_reason))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "evals" / "scenarios.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model")
    a = ap.parse_args()

    cfg = AgentConfig()
    if a.model:
        cfg.model = a.model
    llm = OpenAICompatLLM(cfg)
    scenarios = yaml.safe_load(open(a.file))
    if a.only:
        scenarios = [s for s in scenarios if any(s["id"].startswith(p) for p in a.only)]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "evals" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    results, passed_total = [], 0
    print(f"model={cfg.model}  scenarios={len(scenarios)}  repeat={a.repeat}\n")

    for rep in range(a.repeat):
        for sc in scenarios:
            agent, backend = build(cfg, llm, sc.get("faults"), str(out_dir / f"{sc['id']}_r{rep}.jsonl"))
            sc_ok, turns = True, []
            t0 = time.time()
            for turn in sc["turns"]:
                reply = agent.ask(turn["user"])
                for act in reply.pending_actions:
                    (agent.approve if turn.get("on_pending") == "approve" else agent.reject)(act.id)
                checks = check(turn.get("expect", {}), reply, backend)
                ok = all(c[1] for c in checks)
                sc_ok &= ok
                turns.append({"user": turn["user"], "reply": reply.text, "tools": reply.tool_trace,
                              "stop_reason": reply.stop_reason,
                              "checks": [{"check": c[0], "pass": c[1], "detail": c[2]} for c in checks]})
            passed_total += sc_ok
            dt = time.time() - t0
            print(f"[{'PASS' if sc_ok else 'FAIL'}] {sc['id']:<34} {sc['category']:<45} {dt:5.1f}s")
            for t in turns:
                for c in t["checks"]:
                    if not c["pass"]:
                        print(f"        x {c['check']}  {c['detail']}")
            results.append({"id": sc["id"], "category": sc["category"], "repeat": rep, "pass": sc_ok, "turns": turns})

    total = len(scenarios) * a.repeat
    print(f"\n{passed_total}/{total} scenarios passed. Details: {out_dir}/results.json")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
