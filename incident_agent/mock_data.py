"""Deterministic mocked observability backend.

Story baked into the data (current time 2026-09-24T10:00Z):
  * checkout-api, 2026-09-23 14:37-15:55: clear incident. v142 (deployed 14:32) upgraded
    the DB pool library and leaks connections -> pool exhaustion -> 500s; orders-db saturated.
  * payment-service, 2026-09-22 10:03-11:20: AMBIGUOUS. External payment-gateway latency rose
    at 10:02 (before the deploy), and v88 (10:05) raised retries 2->5, amplifying load (429s).
  * auth-service v310 deployed 2026-09-23 14:00 with no impact (red herring).
  * inventory-service: healthy.
  * search-service: telemetry not ingesting -> empty results.
  * fraud-service: metrics backend times out, logs backend returns malformed payloads.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from .errors import ToolArgError, ToolTimeoutError, ToolTransientError
from .timeutil import UTC, iso


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


SERVICE_CATALOG: dict[str, str] = {
    "web-frontend": "customer-facing web app",
    "checkout-api": "public checkout API",
    "payment-service": "payment authorisation; calls the external payment-gateway",
    "payment-gateway": "EXTERNAL third-party payment provider (metrics only; no logs or deployments)",
    "fraud-service": "fraud scoring used by payment-service",
    "orders-db": "PostgreSQL primary for orders (max 500 connections)",
    "inventory-service": "stock reservation service",
    "auth-service": "login / session service",
    "search-service": "product search",
}
EXTERNAL = {"payment-gateway"}
NO_TELEMETRY = {"search-service"}

APP_METRICS = ["error_rate", "latency_p99_ms", "request_rate_rps", "cpu_pct"]
METRICS_BY_SERVICE: dict[str, list[str]] = {s: APP_METRICS for s in SERVICE_CATALOG}
METRICS_BY_SERVICE["orders-db"] = ["latency_p99_ms", "active_connections", "cpu_pct"]
METRICS_BY_SERVICE["payment-gateway"] = ["latency_p99_ms", "error_rate"]

METRIC_ALIASES = {
    "error_rate": "error_rate", "errors": "error_rate", "error": "error_rate", "error-rate": "error_rate",
    "5xx": "error_rate", "5xx_rate": "error_rate", "error_rate_pct": "error_rate", "http_5xx": "error_rate",
    "latency": "latency_p99_ms", "p99": "latency_p99_ms", "latency_p99": "latency_p99_ms",
    "latency_ms": "latency_p99_ms", "latency_p99_ms": "latency_p99_ms",
    "request_rate": "request_rate_rps", "rps": "request_rate_rps", "requests": "request_rate_rps",
    "throughput": "request_rate_rps", "traffic": "request_rate_rps", "request_rate_rps": "request_rate_rps",
    "cpu": "cpu_pct", "cpu_pct": "cpu_pct", "cpu_usage": "cpu_pct",
    "connections": "active_connections", "active_connections": "active_connections",
    "db_connections": "active_connections", "connection_count": "active_connections",
}
UNITS = {"error_rate": "percent_of_requests", "latency_p99_ms": "ms", "request_rate_rps": "req/s",
         "cpu_pct": "percent", "active_connections": "connections"}
BASELINE = {"error_rate": 0.8, "latency_p99_ms": 220.0, "request_rate_rps": 140.0,
            "cpu_pct": 45.0, "active_connections": 60.0}


@dataclass(frozen=True)
class Anomaly:
    service: str
    metric: str
    start: datetime
    end: datetime
    peak: float


ANOMALIES = [
    Anomaly("checkout-api", "error_rate", _t("2026-09-23T14:37:00"), _t("2026-09-23T15:55:00"), 17.2),
    Anomaly("checkout-api", "latency_p99_ms", _t("2026-09-23T14:36:00"), _t("2026-09-23T15:55:00"), 5100.0),
    Anomaly("orders-db", "active_connections", _t("2026-09-23T14:35:00"), _t("2026-09-23T15:55:00"), 498.0),
    Anomaly("orders-db", "latency_p99_ms", _t("2026-09-23T14:36:00"), _t("2026-09-23T15:55:00"), 1850.0),
    Anomaly("payment-gateway", "latency_p99_ms", _t("2026-09-22T10:02:00"), _t("2026-09-22T11:15:00"), 3400.0),
    Anomaly("payment-gateway", "error_rate", _t("2026-09-22T10:06:00"), _t("2026-09-22T11:15:00"), 4.1),
    Anomaly("payment-service", "latency_p99_ms", _t("2026-09-22T10:03:00"), _t("2026-09-22T11:20:00"), 9800.0),
    Anomaly("payment-service", "error_rate", _t("2026-09-22T10:04:00"), _t("2026-09-22T11:20:00"), 9.5),
]

DEPLOYMENTS = [
    {"service": "auth-service", "version": "v309", "deployed_at": _t("2026-09-15T12:00:00"),
     "author": "ci-bot", "change_summary": "Session TTL config", "status": "succeeded"},
    {"service": "payment-service", "version": "v87", "deployed_at": _t("2026-09-18T13:00:00"),
     "author": "ci-bot", "change_summary": "Logging improvements", "status": "succeeded"},
    {"service": "fraud-service", "version": "v12", "deployed_at": _t("2026-09-19T16:40:00"),
     "author": "ci-bot", "change_summary": "Model threshold update", "status": "succeeded"},
    {"service": "inventory-service", "version": "v57", "deployed_at": _t("2026-09-20T09:15:00"),
     "author": "ci-bot", "change_summary": "Cache warm-up tweaks", "status": "succeeded"},
    {"service": "checkout-api", "version": "v141", "deployed_at": _t("2026-09-21T11:02:00"),
     "author": "ci-bot", "change_summary": "Add promo-code validation", "status": "succeeded",
     "startup_log": "Started checkout-api v141 (db pool: poolkit 2.3.1, max_pool_size=50)"},
    {"service": "payment-service", "version": "v88", "deployed_at": _t("2026-09-22T10:05:00"),
     "author": "ci-bot", "status": "succeeded",
     "change_summary": "Increase payment-gateway retry attempts from 2 to 5; per-attempt timeout 3000ms",
     "startup_log": "Started payment-service v88 (gateway max_retries=5, attempt_timeout=3000ms)"},
    {"service": "auth-service", "version": "v310", "deployed_at": _t("2026-09-23T14:00:00"),
     "author": "ci-bot", "change_summary": "Update login page copy", "status": "succeeded"},
    {"service": "checkout-api", "version": "v142", "deployed_at": _t("2026-09-23T14:32:00"),
     "author": "ci-bot", "status": "succeeded",
     "change_summary": "Refactor DB session handling; upgrade poolkit connection-pool library 2.3.1 -> 3.0.0",
     "startup_log": "Started checkout-api v142 (db pool: poolkit 3.0.0, max_pool_size=50)"},
]

DEPENDENCIES = {
    "web-frontend": (["checkout-api", "auth-service", "search-service"], []),
    "checkout-api": (["orders-db", "payment-service", "inventory-service", "auth-service"], ["web-frontend"]),
    "payment-service": (["payment-gateway", "fraud-service"], ["checkout-api"]),
    "payment-gateway": ([], ["payment-service"]),
    "fraud-service": ([], ["payment-service"]),
    "orders-db": ([], ["checkout-api", "inventory-service"]),
    "inventory-service": (["orders-db"], ["checkout-api"]),
    "auth-service": ([], ["checkout-api", "web-frontend"]),
    "search-service": ([], ["web-frontend"]),
}


@dataclass(frozen=True)
class LogEvent:
    ts: datetime
    service: str
    level: str
    message: str


def _build_logs() -> list[LogEvent]:
    ev: list[LogEvent] = []

    def add(ts, svc, lvl, msg):
        ev.append(LogEvent(ts, svc, lvl, msg))

    def every(start, end, step_min):
        t = _t(start)
        while t < _t(end):
            yield t
            t += timedelta(minutes=step_min)

    routine = [s for s in SERVICE_CATALOG if s not in NO_TELEMETRY and s not in EXTERNAL]
    for t in every("2026-09-19T00:00:00", "2026-09-24T10:00:00", 30):
        for svc in routine:
            add(t, svc, "INFO", "health check ok")
    for t in every("2026-09-19T00:07:00", "2026-09-24T10:00:00", 180):
        add(t, "checkout-api", "WARN", "slow request: POST /v1/checkout took 1240ms")
        add(t + timedelta(minutes=50), "payment-service", "WARN", "payment-gateway responded slowly (1900ms)")
    for d in DEPLOYMENTS:
        if d["deployed_at"] >= _t("2026-09-19T00:00:00"):
            add(d["deployed_at"] + timedelta(minutes=1), d["service"], "INFO",
                d.get("startup_log", f"Started {d['service']} {d['version']}"))

    # checkout-api incident
    for t in every("2026-09-23T14:34:00", "2026-09-23T14:37:00", 1):
        add(t, "checkout-api", "WARN", "db pool: connection not returned to pool after request completed (possible leak)")
    for t in every("2026-09-23T14:37:00", "2026-09-23T15:55:00", 1):
        add(t, "checkout-api", "ERROR",
            "DB connection timeout: could not acquire connection from pool within 5000ms (in_use=50/50)")
        add(t + timedelta(seconds=20), "checkout-api", "ERROR", "POST /v1/checkout returned 500: DatabaseUnavailable")
        add(t + timedelta(seconds=40), "checkout-api", "WARN",
            "db pool: connection not returned to pool after request completed (possible leak)")
    for t in every("2026-09-23T14:35:00", "2026-09-23T15:55:00", 2):
        add(t, "orders-db", "ERROR", "FATAL: remaining connection slots are reserved; too many clients (498/500)")
    add(_t("2026-09-23T14:40:00"), "orders-db", "WARN",
        "connections by client: checkout-api=471, inventory-service=22, other=5")

    # payment-service incident (ambiguous)
    for t in every("2026-09-22T10:03:00", "2026-09-22T11:15:00", 1):
        add(t, "payment-service", "ERROR", "payment-gateway request timed out after 3000ms")
    for t in every("2026-09-22T10:06:00", "2026-09-22T11:20:00", 1):
        add(t + timedelta(seconds=10), "payment-service", "WARN", "retrying payment-gateway call (attempt 5/5)")
        add(t + timedelta(seconds=30), "payment-service", "ERROR",
            "POST /v1/payments/authorize returned 502: upstream timeout")
    add(_t("2026-09-22T10:20:00"), "payment-service", "WARN", "payment-gateway returned HTTP 429 Too Many Requests")

    ev.sort(key=lambda e: e.ts)
    return ev


LOGS = _build_logs()


def _noise(key: str) -> float:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF - 0.5


def metric_value(service: str, metric: str, t: datetime) -> float:
    base = BASELINE[metric]
    v = base * (1 + 0.12 * _noise(f"{service}|{metric}|{t.isoformat()}"))
    for a in ANOMALIES:
        if a.service == service and a.metric == metric and a.start <= t < a.end:
            ramp = min(1.0, ((t - a.start).total_seconds() + 60) / 240)  # ~3-minute ramp-up
            v += (a.peak - base) * ramp
    return round(max(v, 0.0), 2)


# ---------------------------------------------------------------- fault injection
@dataclass
class Fault:
    mode: str                 # timeout | transient | malformed | empty | hang
    times: int | None = None  # None = every call
    hang_s: float = 10.0
    fired: int = 0


def default_faults() -> dict[tuple[str, str], Fault]:
    return {
        ("get_metrics", "fraud-service"): Fault("timeout"),
        ("search_logs", "fraud-service"): Fault("malformed"),
    }


def _garbage(mode: str):
    if mode == "empty":
        return None
    return {"payload": "\x00\x17<<corrupted upstream response>>", "points": "NaN",
            "events": "<<corrupted>>", "deployments": None, "depends_on": None}


_STOP = {"*", "and", "or", "not", "the", "a", "in", "of", "for", "to", "any"}


def _stem(w: str) -> str:
    w = w.split(":")[-1]
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


class MockBackend:
    def __init__(self, now: datetime, faults: dict | None = None):
        self.now = now
        self.faults = default_faults() if faults is None else faults
        self.deployments = copy.deepcopy(DEPLOYMENTS)
        self.notes: list[dict] = []
        self.rollbacks: list[dict] = []
        self.calls: Counter = Counter()
        self._lock = threading.Lock()

    # ---------------- validation helpers used by the tool layer
    def resolve_service(self, name) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ToolArgError("missing required argument 'service'")
        key = re.sub(r"[\s_]+", "-", name.strip().lower())
        if key in SERVICE_CATALOG:
            return key
        matches = [s for s in SERVICE_CATALOG if s.startswith(key) or key in s]
        if len(matches) == 1:
            return matches[0]
        hint = f" (ambiguous: {', '.join(matches)})" if matches else ""
        raise ToolArgError(f"unknown service '{name}'{hint}. Known services: {', '.join(sorted(SERVICE_CATALOG))}")

    def resolve_metric(self, service: str, raw) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolArgError("missing required argument 'metric'")
        canonical = METRIC_ALIASES.get(raw.strip().lower().replace(" ", "_"))
        if canonical is None or canonical not in METRICS_BY_SERVICE[service]:
            raise ToolArgError(f"metric '{raw}' is not available for {service}. "
                               f"Available: {', '.join(METRICS_BY_SERVICE[service])}")
        return canonical

    def check_rollback(self, service: str, version: str) -> str:
        history = sorted((d for d in self.deployments if d["service"] == service), key=lambda d: d["deployed_at"])
        if not history:
            raise ToolArgError(f"no deployment history for {service}; cannot roll back")
        current = history[-1]["version"]
        if version == current:
            raise ToolArgError(f"{version} is already the running version of {service}")
        if version not in {d["version"] for d in history}:
            raise ToolArgError(f"unknown version {version} for {service}; known: "
                               f"{', '.join(d['version'] for d in history)}")
        return current

    def _fault(self, tool: str, service: str):
        with self._lock:
            self.calls[tool] += 1
            f = self.faults.get((tool, service)) or self.faults.get((tool, "*"))
            if not f or (f.times is not None and f.fired >= f.times):
                return None
            f.fired += 1
        if f.mode == "timeout":
            raise ToolTimeoutError(f"{tool} backend timed out for {service}")
        if f.mode == "transient":
            raise ToolTransientError(f"{tool} backend returned HTTP 503 (temporarily unavailable)")
        if f.mode == "hang":
            time.sleep(f.hang_s)
            return None
        return f.mode  # malformed | empty

    # ---------------- tools
    def get_metrics(self, service, metric, start_time, end_time):
        fm = self._fault("get_metrics", service)
        if fm:
            return _garbage(fm)
        base = {"service": service, "metric": metric, "unit": UNITS[metric],
                "start_time": iso(start_time), "end_time": iso(end_time)}
        if service in NO_TELEMETRY:
            return {**base, "points": [], "summary": None, "no_data": True,
                    "note": "no metrics ingested for this service in the requested window"}
        minutes = max(1, int((end_time - start_time).total_seconds() // 60))
        raw = [(start_time + timedelta(minutes=i), metric_value(service, metric, start_time + timedelta(minutes=i)))
               for i in range(minutes)]
        step = max(1, math.ceil(minutes / 60))
        points = []
        for i in range(0, len(raw), step):
            bucket = raw[i:i + step]
            points.append({"t": iso(bucket[0][0]), "v": round(sum(v for _, v in bucket) / len(bucket), 2)})
        prev = [metric_value(service, metric, start_time - timedelta(minutes=j)) for j in range(1, 121)]
        baseline = round(sum(prev) / len(prev), 2)
        vals = [v for _, v in raw]
        mx = max(vals)
        threshold = max(baseline * 2, baseline + 1.0) if metric == "error_rate" else baseline * 2
        first = next((t for t, v in raw if v > threshold), None)
        summary = {
            "baseline_prev_2h_avg": baseline,
            "window_avg": round(sum(vals) / len(vals), 2),
            "window_min": min(vals),
            "window_max": mx,
            "window_max_at": iso(raw[vals.index(mx)][0]),
            "first_above_2x_baseline_at": iso(first) if first else None,
            "minutes_above_2x_baseline": sum(1 for v in vals if v > threshold),
        }
        return {**base, "resolution_minutes": step, "points": points, "summary": summary, "no_data": False}

    def search_logs(self, service, start_time, end_time, query=""):
        fm = self._fault("search_logs", service)
        if fm:
            return _garbage(fm)
        base = {"service": service, "start_time": iso(start_time), "end_time": iso(end_time), "query": query}
        if service in NO_TELEMETRY:
            return {**base, "total_matches": 0, "events": [], "note": "no logs ingested for this service"}
        if service in EXTERNAL:
            return {**base, "total_matches": 0, "events": [],
                    "note": "external third-party service; its logs are not available to us"}
        terms = [_stem(w) for w in re.split(r"[\s,|;]+", (query or "").lower()) if w and w not in _STOP]

        def hit(e: LogEvent) -> bool:
            text = f"{e.level} {e.message}".lower()
            return not terms or any(t in text for t in terms)

        matched = [e for e in LOGS if e.service == service and start_time <= e.ts < end_time and hit(e)]
        groups: dict[str, dict] = {}
        for e in matched:
            g = groups.setdefault(e.message, {"message": e.message, "level": e.level, "count": 0,
                                              "first_seen": e.ts, "last_seen": e.ts})
            g["count"] += 1
            g["last_seen"] = e.ts
        top = sorted(groups.values(), key=lambda g: -g["count"])[:8]
        for g in top:
            g["first_seen"], g["last_seen"] = iso(g["first_seen"]), iso(g["last_seen"])
        sample = matched if len(matched) <= 20 else matched[:10] + matched[-10:]
        return {**base, "total_matches": len(matched),
                "level_counts": dict(Counter(e.level for e in matched)),
                "top_messages": top,
                "events": [{"ts": iso(e.ts), "level": e.level, "message": e.message} for e in sample],
                "events_sampled": len(matched) > 20}

    def get_deployments(self, service, start_time, end_time):
        fm = self._fault("get_deployments", service)
        if fm:
            return _garbage(fm)
        base = {"service": service, "start_time": iso(start_time), "end_time": iso(end_time)}
        if service in EXTERNAL:
            return {**base, "deployments": [],
                    "note": "external third-party service; its releases are not visible to us"}
        rows = sorted((d for d in self.deployments
                       if d["service"] == service and start_time <= d["deployed_at"] < end_time),
                      key=lambda d: d["deployed_at"])
        return {**base, "deployments": [
            {"version": d["version"], "deployed_at": iso(d["deployed_at"]), "author": d["author"],
             "change_summary": d["change_summary"], "status": d["status"]} for d in rows]}

    def get_service_dependencies(self, service):
        fm = self._fault("get_service_dependencies", service)
        if fm:
            return _garbage(fm)
        up, down = DEPENDENCIES.get(service, ([], []))
        return {"service": service, "depends_on": list(up), "depended_on_by": list(down),
                "external": service in EXTERNAL}

    def create_incident_note(self, title, summary, evidence, recommended_actions):
        fm = self._fault("create_incident_note", "*")
        if fm:
            return _garbage(fm)
        with self._lock:
            for n in self.notes:  # idempotent by title
                if n["title"].lower() == title.lower():
                    return {**n, "status": "already_exists"}
            note = {"note_id": f"NOTE-{len(self.notes) + 1:03d}", "created_at": iso(self.now), "title": title,
                    "summary": summary, "evidence": list(evidence),
                    "recommended_actions": list(recommended_actions), "status": "created"}
            self.notes.append(note)
        return dict(note)

    def rollback_deployment(self, service, to_version, reason):
        fm = self._fault("rollback_deployment", service)
        if fm:
            return _garbage(fm)
        current = self.check_rollback(service, to_version)
        with self._lock:
            record = {"service": service, "from_version": current, "to_version": to_version,
                      "reason": reason, "at": iso(self.now), "status": "rolled_back"}
            self.rollbacks.append(record)
            self.deployments.append({"service": service, "version": to_version, "deployed_at": self.now,
                                     "author": "incident-agent (human-approved)",
                                     "change_summary": f"ROLLBACK {current} -> {to_version}: {reason}",
                                     "status": "succeeded"})
        return dict(record)
