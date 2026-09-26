import pytest

from incident_agent.config import AgentConfig
from incident_agent.errors import ToolArgError, ToolMalformedError, ToolTimeoutError
from incident_agent.mock_data import MockBackend
from incident_agent.tools import build_registry

W = {"start_time": "2026-09-23T14:00:00Z", "end_time": "2026-09-23T16:00:00Z"}


@pytest.fixture
def env():
    cfg = AgentConfig()
    backend = MockBackend(now=cfg.now)
    return build_registry(backend, cfg), backend


def call(registry, name, **raw):
    spec = registry.get(name)
    args = spec.validate_args(raw)
    args.pop("_notes", None)
    out = spec.run(**args)
    spec.validate_output(out)
    return out


def test_checkout_error_spike(env):
    reg, _ = env
    s = call(reg, "get_metrics", service="checkout-api", metric="error_rate", **W)["summary"]
    assert s["baseline_prev_2h_avg"] < 1.0
    assert s["window_max"] > 15
    assert s["first_above_2x_baseline_at"] == "2026-09-23T14:37:00Z"


def test_healthy_service_has_no_anomaly(env):
    reg, _ = env
    s = call(reg, "get_metrics", service="inventory-service", metric="error_rate", **W)["summary"]
    assert s["first_above_2x_baseline_at"] is None


def test_metric_and_service_aliases(env):
    reg, _ = env
    out = call(reg, "get_metrics", service="checkout", metric="errors", **W)
    assert out["service"] == "checkout-api" and out["metric"] == "error_rate"


def test_metric_not_available_for_service(env):
    reg, _ = env
    with pytest.raises(ToolArgError, match="not available"):
        reg.get("get_metrics").validate_args({"service": "orders-db", "metric": "error_rate", **W})


def test_ambiguous_and_unknown_service(env):
    reg, _ = env
    with pytest.raises(ToolArgError, match="ambiguous"):
        reg.get("get_service_dependencies").validate_args({"service": "payment"})
    with pytest.raises(ToolArgError, match="unknown service"):
        reg.get("get_service_dependencies").validate_args({"service": "billing"})


@pytest.mark.parametrize("start,end,msg", [
    ("2026-09-23T16:00:00Z", "2026-09-23T14:00:00Z", "not after"),
    ("2026-09-25T10:00:00Z", "2026-09-25T12:00:00Z", "future"),
    ("2026-09-19T00:00:00Z", "2026-09-23T00:00:00Z", "maximum"),
    ("2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z", "retention"),
    ("yesterday", "2026-09-23T14:00:00Z", "ISO-8601"),
])
def test_invalid_time_ranges(env, start, end, msg):
    reg, _ = env
    with pytest.raises(ToolArgError, match=msg):
        reg.get("get_deployments").validate_args({"service": "checkout-api", "start_time": start, "end_time": end})


def test_future_end_is_clamped_with_note(env):
    reg, _ = env
    args = reg.get("get_deployments").validate_args(
        {"service": "checkout-api", "start_time": "2026-09-24T08:00:00Z", "end_time": "2026-09-24T12:00:00Z"})
    assert args["_notes"] and "clamped" in args["_notes"][0]


def test_deployments_in_window(env):
    reg, _ = env
    rows = call(reg, "get_deployments", service="checkout-api", **W)["deployments"]
    assert [r["version"] for r in rows] == ["v142"]
    assert rows[0]["deployed_at"] == "2026-09-23T14:32:00Z"


def test_log_search(env):
    reg, _ = env
    out = call(reg, "search_logs", service="checkout-api", query="timeouts", **W)
    assert out["total_matches"] == 78
    assert "connection timeout" in out["top_messages"][0]["message"]


def test_empty_results(env):
    reg, _ = env
    out = call(reg, "get_metrics", service="search-service", metric="error_rate", **W)
    assert out["no_data"] is True and out["points"] == []
    out = call(reg, "search_logs", service="payment-gateway", query="", **W)
    assert out["total_matches"] == 0 and "external" in out["note"]


def test_fault_injection(env):
    reg, _ = env
    with pytest.raises(ToolTimeoutError):
        call(reg, "get_metrics", service="fraud-service", metric="error_rate", **W)
    with pytest.raises(ToolMalformedError):
        call(reg, "search_logs", service="fraud-service", query="error", **W)


def test_dependencies(env):
    reg, _ = env
    out = call(reg, "get_service_dependencies", service="checkout-api")
    assert "orders-db" in out["depends_on"] and out["depended_on_by"] == ["web-frontend"]


def test_incident_note_is_idempotent(env):
    reg, backend = env
    args = dict(title="Checkout errors", summary="s", evidence="E1 spike", recommended_actions=["rollback"])
    a = call(reg, "create_incident_note", **args)
    b = call(reg, "create_incident_note", **args)
    assert a["note_id"] == b["note_id"] and b["status"] == "already_exists" and len(backend.notes) == 1


def test_rollback_validation(env):
    reg, _ = env
    v = reg.get("rollback_deployment").validate_args
    with pytest.raises(ToolArgError, match="already the running"):
        v({"service": "checkout-api", "to_version": "v142", "reason": "x"})
    with pytest.raises(ToolArgError, match="unknown version"):
        v({"service": "checkout-api", "to_version": "v9", "reason": "x"})
    assert v({"service": "checkout-api", "to_version": "v141", "reason": "x"})["to_version"] == "v141"
