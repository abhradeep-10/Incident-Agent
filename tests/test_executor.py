import json
import time

from incident_agent.config import AgentConfig
from incident_agent.executor import ToolExecutor, TurnBudget
from incident_agent.llm import ToolCall, new_call_id
from incident_agent.mock_data import Fault, MockBackend
from incident_agent.state import Session
from incident_agent.tools import build_registry

W = {"start_time": "2026-09-23T14:00:00Z", "end_time": "2026-09-23T16:00:00Z"}


def setup(faults=None, **over):
    cfg = AgentConfig(tool_timeout_s=0.3, retry_backoff_s=0.0, **over)
    backend = MockBackend(now=cfg.now, faults=faults)
    session = Session()
    ex = ToolExecutor(build_registry(backend, cfg), cfg, session)
    return ex, session, backend, TurnBudget.from_config(cfg)


def tc(name, **args):
    return ToolCall(new_call_id(), name, json.dumps(args))


def metrics(**kw):
    return tc("get_metrics", **{"service": "checkout-api", "metric": "error_rate", **W, **kw})


def test_transient_failure_is_retried():
    ex, _, backend, b = setup({("get_metrics", "checkout-api"): Fault("transient", times=2)})
    [r] = ex.run_batch([metrics()], b)
    assert r.ok and r.attempts == 3 and r.evidence_id == "E1"
    assert backend.calls["get_metrics"] == 3


def test_retries_exhausted_reports_failure():
    ex, session, _, b = setup({("get_metrics", "checkout-api"): Fault("transient")})
    [r] = ex.run_batch([metrics()], b)
    assert not r.ok and r.error_type == "transient_failure" and r.attempts == 3
    assert session.failures and not session.evidence


def test_hanging_backend_is_cut_off_by_timeout():
    ex, _, _, b = setup({("get_metrics", "checkout-api"): Fault("hang", hang_s=1.0)}, tool_max_retries=0)
    t0 = time.monotonic()
    [r] = ex.run_batch([metrics()], b)
    assert r.error_type == "timeout" and time.monotonic() - t0 < 0.9


def test_malformed_response_detected():
    ex, _, _, b = setup()
    [r] = ex.run_batch([tc("search_logs", service="fraud-service", query="error", **W)], b)
    assert not r.ok and r.error_type == "malformed_response"


def test_bad_json_and_invalid_range_are_not_executed():
    ex, _, backend, b = setup()
    bad_json = ToolCall(new_call_id(), "get_metrics", "{not json")
    bad_range = metrics(start_time="2026-09-23T16:00:00Z", end_time="2026-09-23T14:00:00Z")
    r1, r2 = ex.run_batch([bad_json, bad_range], b)
    assert r1.error_type == r2.error_type == "invalid_arguments"
    assert backend.calls["get_metrics"] == 0


def test_unknown_tool():
    ex, _, _, b = setup()
    [r] = ex.run_batch([tc("delete_database", name="prod")], b)
    assert r.error_type == "unknown_tool"


def test_duplicate_calls_are_not_re_executed():
    ex, _, backend, b = setup()
    [r1] = ex.run_batch([metrics()], b)
    [r2] = ex.run_batch([metrics()], b)
    assert r2.status == "duplicate" and r2.evidence_id == r1.evidence_id == "E1"
    assert backend.calls["get_metrics"] == 1 and b.duplicates == 1
    r3, r4 = ex.run_batch([metrics(service="orders-db", metric="active_connections"),
                           metrics(service="orders-db", metric="active_connections")], b)
    assert r3.ok and r4.status == "duplicate"


def test_budget_limits_tool_calls():
    ex, _, _, b = setup(max_tool_calls_per_turn=2)
    results = ex.run_batch([metrics(start_time=f"2026-09-23T1{h}:00:00Z", end_time=f"2026-09-23T1{h + 1}:00:00Z")
                            for h in (0, 1, 2)], b)
    assert [r.ok for r in results] == [True, True, False]
    assert results[2].error_type == "budget_exhausted" and b.stop_reason() == "tool_budget_exhausted"


def test_parallel_batch_keeps_evidence_order():
    ex, _, _, b = setup()
    results = ex.run_batch([metrics(), tc("get_deployments", service="checkout-api", **W),
                            tc("search_logs", service="checkout-api", query="error", **W),
                            tc("get_service_dependencies", service="checkout-api")], b)
    assert [r.evidence_id for r in results] == ["E1", "E2", "E3", "E4"]


def test_repeated_failure_is_blocked():
    ex, _, backend, b = setup(tool_max_retries=0)
    call = lambda: metrics(service="fraud-service")  # noqa: E731  (fraud metrics always time out)
    r1 = ex.run_batch([call()], b)[0]
    r2 = ex.run_batch([call()], b)[0]
    r3 = ex.run_batch([call()], b)[0]
    assert r1.error_type == r2.error_type == "timeout"
    assert r3.error_type == "repeated_failure" and backend.calls["get_metrics"] == 2


def test_rollback_requires_approval():
    ex, session, backend, b = setup()
    [r] = ex.run_batch([tc("rollback_deployment", service="checkout-api", to_version="v141", reason="errors")], b)
    assert r.status == "pending_approval" and not r.ok
    assert backend.rollbacks == [] and backend.calls["rollback_deployment"] == 0
    [pa] = session.pending
    r2 = ex.execute_approved(pa)
    assert r2.ok and backend.rollbacks[0]["to_version"] == "v141"


def test_large_output_is_trimmed():
    ex, _, _, b = setup(max_window_hours=72)
    [r] = ex.run_batch([tc("get_metrics", service="checkout-api", metric="error_rate",
                           start_time="2026-09-22T00:00:00Z", end_time="2026-09-24T00:00:00Z")], b)
    content = r.to_content(max_chars=1500)
    assert len(content) <= 1500 + 20
