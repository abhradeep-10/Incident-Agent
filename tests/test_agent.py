"""Orchestration tests. The LLM is scripted so these run offline and deterministically;
live-model behaviour is covered by evals/run_evals.py."""
import json

from incident_agent.agent import IncidentAgent
from incident_agent.config import AgentConfig
from incident_agent.llm import LLMResponse, ScriptedLLM, ToolCall, new_call_id
from incident_agent.mock_data import MockBackend
from incident_agent.tools import build_registry

W = {"start_time": "2026-09-23T14:00:00Z", "end_time": "2026-09-23T16:00:00Z"}

REPORT = """## Summary
Checkout errors spiked at 14:37 [E1].
## Observed facts
- error rate peaked above 17% [E1]
- v142 was deployed at 14:32 [E2]
- logs show DB pool timeouts [E3]
## Hypotheses / inferences
- v142 leaks DB connections - confidence: medium - supporting: [E1, E2, E3]
## Gaps and tool issues
None.
## Recommended next actions
1. Compare v142 with v141"""


def call(name, **args):
    return ToolCall(new_call_id(), name, json.dumps(args))


def use(*calls):
    return LLMResponse(content="", tool_calls=list(calls))


def say(text):
    return LLMResponse(content=text)


def metrics(h=None):
    w = W if h is None else {"start_time": f"2026-09-23T{h}:00:00Z", "end_time": f"2026-09-23T{h + 1}:00:00Z"}
    return call("get_metrics", service="checkout-api", metric="error_rate", **w)


def make(steps, **over):
    cfg = AgentConfig(tool_timeout_s=0.5, retry_backoff_s=0.0, **over)
    backend = MockBackend(now=cfg.now)
    llm = ScriptedLLM(steps)
    return IncidentAgent(llm, build_registry(backend, cfg), cfg), llm, backend


def test_multi_step_investigation():
    agent, _, _ = make([
        use(metrics()),
        use(call("get_deployments", service="checkout-api", **W),
            call("search_logs", service="checkout-api", query="error", **W)),
        say(REPORT),
    ])
    reply = agent.ask("Investigate checkout-api errors yesterday 2-4 PM")
    assert reply.stop_reason == "final_answer"
    assert reply.tools_called == ["get_metrics", "get_deployments", "search_logs"]
    assert set(agent.session.evidence) == {"E1", "E2", "E3"}
    assert "Automated evidence check" not in reply.text


def test_next_step_can_depend_on_previous_result():
    seen = {}

    def second(messages, tools):
        payload = json.loads(messages[-1]["content"])
        seen["role"], seen["payload"] = messages[-1]["role"], payload
        onset = payload["data"]["summary"]["first_above_2x_baseline_at"]
        return use(call("get_deployments", service="checkout-api", start_time="2026-09-23T14:00:00Z", end_time=onset))

    agent, _, _ = make([use(metrics()), second, say("v142 was deployed at 14:32, before the onset [E1, E2].")])
    reply = agent.ask("Was there a deploy before the checkout spike?")
    assert seen["role"] == "tool" and seen["payload"]["evidence_id"] == "E1"
    assert agent.session.evidence["E2"].data["deployments"][0]["version"] == "v142"
    assert reply.stop_reason == "final_answer"


def test_tool_budget_forces_final_answer():
    agent, llm, _ = make([use(metrics(10)), use(metrics(11)), use(metrics(12)), say("Nothing abnormal [E1].")],
                         max_tool_calls_per_turn=3)
    reply = agent.ask("check checkout")
    assert reply.stop_reason == "tool_budget_exhausted" and reply.tool_calls == 3
    assert llm.calls[3]["tools"] is None
    assert llm.calls[3]["messages"][-1]["content"].startswith("[runtime]")


def test_repeated_identical_calls_trip_loop_detector():
    agent, _, backend = make([use(metrics()) for _ in range(4)], max_duplicate_calls_per_turn=2)
    reply = agent.ask("check checkout")
    assert reply.stop_reason == "loop_detected"
    assert backend.calls["get_metrics"] == 1
    assert "[E1]" in reply.text  # deterministic fallback report still cites the evidence


def test_hallucinated_citation_is_repaired():
    agent, llm, _ = make([use(metrics()), say("## Observed facts\n- spike [E9]"),
                          say("## Observed facts\n- spike [E1]")])
    reply = agent.ask("check checkout")
    assert "[E1]" in reply.text and "Automated evidence check" not in reply.text
    assert llm.calls[2]["tools"] is None
    assert llm.calls[2]["messages"][-1]["content"].startswith("[runtime check]")
    assert not any("[E9]" in (m.get("content") or "") for m in agent.session.messages)


def test_unrepaired_answer_is_flagged():
    agent, _, _ = make([use(metrics()), say("## Observed facts\n- spike happened"),
                        say("## Observed facts\n- spike happened")])
    reply = agent.ask("check checkout")
    assert "Automated evidence check" in reply.text


def test_follow_up_uses_previous_state():
    seen = {}

    def follow_up(messages, tools):
        seen["system"] = messages[0]["content"]
        seen["users"] = [m["content"] for m in messages if m["role"] == "user"]
        return say("The spike was confirmed earlier [E1].")

    agent, _, _ = make([use(metrics()), say("Spike confirmed [E1]."), follow_up])
    agent.ask("check checkout errors yesterday 2-4pm")
    reply = agent.ask("when exactly did it start?")
    assert "E1 [turn 1] get_metrics" in seen["system"]
    assert seen["users"][0] == "check checkout errors yesterday 2-4pm"
    assert reply.stop_reason == "final_answer" and agent.session.turn == 2


def test_rollback_needs_human_approval():
    agent, _, backend = make([
        use(call("rollback_deployment", service="checkout-api", to_version="v141", reason="error spike")),
        say("The rollback of checkout-api to v141 is queued and needs your approval."),
    ])
    reply = agent.ask("roll back checkout-api to v141")
    assert [a.tool for a in reply.pending_actions] == ["rollback_deployment"]
    assert backend.rollbacks == []
    msg = agent.approve(reply.pending_actions[0].id)
    assert "APPROVED" in msg and len(backend.rollbacks) == 1
    assert "No pending action" in agent.approve(reply.pending_actions[0].id)


def test_rollback_rejection():
    agent, _, backend = make([
        use(call("rollback_deployment", service="checkout-api", to_version="v141", reason="x")),
        say("Queued for approval."),
    ])
    reply = agent.ask("roll back checkout-api to v141")
    assert "REJECTED" in agent.reject(reply.pending_actions[0].id)
    assert backend.rollbacks == [] and agent.session.actions["A1"].status == "rejected"


def test_tool_failure_is_reported_not_hidden():
    agent, _, _ = make([
        use(call("get_metrics", service="fraud-service", metric="error_rate", **W)),
        say("The metrics backend for fraud-service timed out, so I have no data."),
    ], tool_max_retries=1)
    reply = agent.ask("fraud-service errors yesterday 2-4pm?")
    assert reply.tool_trace[0]["error_type"] == "timeout" and reply.tool_trace[0]["attempts"] == 2
    assert agent.session.failures and not agent.session.evidence


def test_llm_outage_gives_fallback():
    def boom(messages, tools):
        raise ConnectionError("connection refused")

    agent, _, _ = make([boom])
    reply = agent.ask("check checkout")
    assert reply.stop_reason == "llm_error" and "could not complete" in reply.text


def test_clarifying_question_uses_no_tools():
    agent, _, _ = make([say("Which service and time window should I look at?")])
    reply = agent.ask("something is broken, investigate")
    assert reply.tool_calls == 0 and reply.stop_reason == "final_answer"
    assert "Automated evidence check" not in reply.text
