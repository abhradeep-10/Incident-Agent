"""Tool definitions: JSON schemas shown to the LLM, argument validators, output validators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .config import AgentConfig
from .errors import ToolArgError, ToolMalformedError
from .mock_data import METRICS_BY_SERVICE, MockBackend
from .timeutil import validate_window


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    validate_args: Callable[[dict], dict]   # returns normalised args (may include "_notes")
    run: Callable[..., Any]
    validate_output: Callable[[Any], None]  # raises ToolMalformedError
    side_effect: bool = False
    idempotent: bool = True
    requires_approval: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {s.name: s for s in specs}

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def schemas(self) -> list[dict]:
        return [s.schema() for s in self._specs.values()]


# ------------------------------------------------------------------ arg helpers
def _str(raw: dict, key: str, max_len: int = 200) -> str:
    v = raw.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ToolArgError(f"missing required argument '{key}'")
    if not isinstance(v, (str, int, float)) or isinstance(v, bool):
        raise ToolArgError(f"'{key}' must be a string")
    v = str(v).strip()
    if len(v) > max_len:
        raise ToolArgError(f"'{key}' is too long (max {max_len} characters)")
    return v


def _str_list(raw: dict, key: str, max_items: int = 20, max_len: int = 500) -> list[str]:
    v = raw.get(key)
    if v is None:
        return []
    if isinstance(v, str):  # small models often send a newline-separated string
        v = [line.strip(" -*\u2022\t") for line in v.splitlines()]
    if not isinstance(v, list):
        raise ToolArgError(f"'{key}' must be a list of strings")
    out = [str(x).strip() for x in v if x is not None and str(x).strip()]
    if len(out) > max_items:
        raise ToolArgError(f"'{key}' has too many items (max {max_items})")
    if any(len(x) > max_len for x in out):
        raise ToolArgError(f"an item in '{key}' is too long (max {max_len} characters)")
    return out


# ------------------------------------------------------------------ output validators
def _require_dict(data, keys):
    if data is None or data == {}:
        raise ToolMalformedError("empty response from backend")
    if not isinstance(data, dict):
        raise ToolMalformedError(f"expected a JSON object, got {type(data).__name__}")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ToolMalformedError(f"response is missing fields {missing}")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_metrics(d):
    _require_dict(d, ["service", "metric", "points"])
    if not isinstance(d["points"], list):
        raise ToolMalformedError("'points' is not a list")
    for p in d["points"]:
        if not (isinstance(p, dict) and isinstance(p.get("t"), str) and _num(p.get("v"))):
            raise ToolMalformedError(f"malformed data point: {str(p)[:80]}")


def check_logs(d):
    _require_dict(d, ["service", "events", "total_matches"])
    if not isinstance(d["events"], list) or not isinstance(d["total_matches"], int):
        raise ToolMalformedError("'events' must be a list and 'total_matches' an integer")
    for e in d["events"]:
        if not (isinstance(e, dict) and all(isinstance(e.get(k), str) for k in ("ts", "level", "message"))):
            raise ToolMalformedError(f"malformed log event: {str(e)[:80]}")


def check_deployments(d):
    _require_dict(d, ["service", "deployments"])
    if not isinstance(d["deployments"], list):
        raise ToolMalformedError("'deployments' is not a list")
    for r in d["deployments"]:
        if not (isinstance(r, dict) and isinstance(r.get("version"), str) and isinstance(r.get("deployed_at"), str)):
            raise ToolMalformedError(f"malformed deployment record: {str(r)[:80]}")


def check_dependencies(d):
    _require_dict(d, ["service", "depends_on", "depended_on_by"])
    if not isinstance(d["depends_on"], list) or not isinstance(d["depended_on_by"], list):
        raise ToolMalformedError("dependency fields must be lists")


def check_note(d):
    _require_dict(d, ["note_id", "status"])


def check_rollback(d):
    _require_dict(d, ["status", "to_version"])


# ------------------------------------------------------------------ registry
_WINDOW_PROPS = {
    "start_time": {"type": "string", "description": "ISO-8601 UTC start, e.g. 2026-09-23T14:00:00Z"},
    "end_time": {"type": "string", "description": "ISO-8601 UTC end (exclusive), must be after start_time"},
}


def build_registry(backend: MockBackend, cfg: AgentConfig) -> ToolRegistry:
    def window(raw):
        return validate_window(raw.get("start_time"), raw.get("end_time"), now=cfg.now,
                               max_hours=cfg.max_window_hours, retention_days=cfg.retention_days)

    def v_metrics(raw):
        svc = backend.resolve_service(raw.get("service"))
        metric = backend.resolve_metric(svc, raw.get("metric"))
        s, e, notes = window(raw)
        return {"service": svc, "metric": metric, "start_time": s, "end_time": e, "_notes": notes}

    def v_logs(raw):
        svc = backend.resolve_service(raw.get("service"))
        q = raw.get("query") or ""
        if not isinstance(q, str):
            q = str(q)
        if len(q) > 200:
            raise ToolArgError("'query' is too long (max 200 characters)")
        s, e, notes = window(raw)
        return {"service": svc, "start_time": s, "end_time": e, "query": q.strip(), "_notes": notes}

    def v_deploys(raw):
        svc = backend.resolve_service(raw.get("service"))
        s, e, notes = window(raw)
        return {"service": svc, "start_time": s, "end_time": e, "_notes": notes}

    def v_deps(raw):
        return {"service": backend.resolve_service(raw.get("service"))}

    def v_note(raw):
        args = {"title": _str(raw, "title", 150), "summary": _str(raw, "summary", 4000),
                "evidence": _str_list(raw, "evidence"),
                "recommended_actions": _str_list(raw, "recommended_actions")}
        if not args["evidence"]:
            raise ToolArgError("'evidence' must contain at least one item")
        return args

    def v_rollback(raw):
        svc = backend.resolve_service(raw.get("service"))
        ver = _str(raw, "to_version", 40)
        backend.check_rollback(svc, ver)
        return {"service": svc, "to_version": ver, "reason": _str(raw, "reason", 500)}

    metric_help = "; ".join(
        f"{s}: {', '.join(m)}" for s, m in
        [("app services", METRICS_BY_SERVICE["checkout-api"]), ("orders-db", METRICS_BY_SERVICE["orders-db"]),
         ("payment-gateway", METRICS_BY_SERVICE["payment-gateway"])])

    specs = [
        ToolSpec(
            "get_metrics",
            "Time series for ONE metric of ONE service, plus a summary: average over the 2h BEFORE the window "
            "(baseline), max and when it happened, and the first time the value exceeded ~2x baseline. Use it to "
            f"confirm whether and when something abnormal happened. Metrics: {metric_help}. error_rate is % of "
            "requests returning 5xx.",
            {"type": "object", "properties": {
                "service": {"type": "string"},
                "metric": {"type": "string"},
                **_WINDOW_PROPS},
             "required": ["service", "metric", "start_time", "end_time"]},
            v_metrics, backend.get_metrics, check_metrics),
        ToolSpec(
            "search_logs",
            "Search log events of ONE service in a time window. `query` is keywords matched against level and "
            "message (any keyword matches; empty = all). Returns total matches, counts per level, the most "
            "frequent messages with first/last seen, and a sample of events.",
            {"type": "object", "properties": {
                "service": {"type": "string"},
                **_WINDOW_PROPS,
                "query": {"type": "string", "description": "keywords, e.g. 'error timeout'"}},
             "required": ["service", "start_time", "end_time", "query"]},
            v_logs, backend.search_logs, check_logs),
        ToolSpec(
            "get_deployments",
            "List deployments/releases of ONE service in a time window (version, time, change summary).",
            {"type": "object", "properties": {"service": {"type": "string"}, **_WINDOW_PROPS},
             "required": ["service", "start_time", "end_time"]},
            v_deploys, backend.get_deployments, check_deployments),
        ToolSpec(
            "get_service_dependencies",
            "Return which services this service depends on (depends_on) and which services depend on it "
            "(depended_on_by).",
            {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            v_deps, backend.get_service_dependencies, check_dependencies),
        ToolSpec(
            "create_incident_note",
            "Create a structured incident note. Use once, at the end of an investigation that found a real "
            "anomaly, or when the user asks. Evidence items should cite evidence ids like [E2].",
            {"type": "object", "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "recommended_actions": {"type": "array", "items": {"type": "string"}}},
             "required": ["title", "summary", "evidence", "recommended_actions"]},
            v_note, backend.create_incident_note, check_note, side_effect=True, idempotent=True),
        ToolSpec(
            "rollback_deployment",
            "Roll a service back to an earlier version. DANGEROUS: only when the user explicitly asks for a "
            "rollback. Never executed automatically: it is queued and a human must approve it.",
            {"type": "object", "properties": {
                "service": {"type": "string"},
                "to_version": {"type": "string", "description": "earlier version to restore, e.g. v141"},
                "reason": {"type": "string"}},
             "required": ["service", "to_version", "reason"]},
            v_rollback, backend.rollback_deployment, check_rollback,
            side_effect=True, idempotent=False, requires_approval=True),
    ]
    return ToolRegistry(specs)
