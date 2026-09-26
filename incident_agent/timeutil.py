from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .errors import ToolArgError

UTC = timezone.utc


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value, field_name: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        s = value.strip()
        if s[-1] in "Zz":
            s = s[:-1] + "+00:00"
        if "T" not in s and " " in s:
            s = s.replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            raise ToolArgError(
                f"{field_name}={value!r} is not an ISO-8601 timestamp (expected e.g. 2026-09-23T14:00:00Z)"
            ) from None
    else:
        raise ToolArgError(f"{field_name} is required and must be an ISO-8601 string")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def validate_window(start, end, *, now: datetime, max_hours: int, retention_days: int):
    """Returns (start, end, notes). Raises ToolArgError for nonsensical ranges."""
    s = parse_ts(start, "start_time")
    e = parse_ts(end, "end_time")
    notes: list[str] = []
    if e <= s:
        raise ToolArgError(f"invalid time range: end_time {iso(e)} is not after start_time {iso(s)}")
    if s >= now:
        raise ToolArgError(f"start_time {iso(s)} is in the future (current time is {iso(now)}); no data exists yet")
    if e > now:
        notes.append(f"end_time {iso(e)} was in the future and was clamped to the current time {iso(now)}")
        e = now
    if s < now - timedelta(days=retention_days):
        raise ToolArgError(f"start_time {iso(s)} is older than the {retention_days}-day data retention window")
    if e - s > timedelta(hours=max_hours):
        hours = (e - s).total_seconds() / 3600
        raise ToolArgError(f"time range is {hours:.1f}h; the maximum is {max_hours}h. Narrow the window.")
    return s, e, notes
