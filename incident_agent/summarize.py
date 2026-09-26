"""Deterministic one-line summaries of tool results, used for the evidence ledger."""
from __future__ import annotations


def summarize(tool: str, args: dict, data: dict) -> str:
    try:
        w = f"{args.get('start_time')}->{args.get('end_time')}"
        if tool == "get_metrics":
            if data.get("no_data"):
                return f"{args['service']} {args['metric']} {w}: NO DATA ({data.get('note', '')})"
            s = data["summary"]
            return (f"{args['service']} {args['metric']} ({data.get('unit')}) {w}: baseline(prev 2h avg) "
                    f"{s['baseline_prev_2h_avg']}, window avg {s['window_avg']}, max {s['window_max']} at "
                    f"{s['window_max_at']}, first >2x baseline at {s['first_above_2x_baseline_at'] or 'never'}")
        if tool == "get_deployments":
            rows = data["deployments"]
            if not rows:
                return f"{args['service']} deployments {w}: none" + (f" ({data['note']})" if data.get("note") else "")
            return f"{args['service']} deployments {w}: " + "; ".join(
                f"{r['version']} at {r['deployed_at']} ({r.get('change_summary', '')})" for r in rows)
        if tool == "search_logs":
            top = data.get("top_messages") or []
            head = f"{args['service']} logs query='{args.get('query', '')}' {w}: {data['total_matches']} matches"
            if data.get("note"):
                head += f" ({data['note']})"
            if top:
                head += "; top: " + "; ".join(
                    f"[{g['level']}] {g['message']} x{g['count']} ({g['first_seen']}..{g['last_seen']})"
                    for g in top[:3])
            return head
        if tool == "get_service_dependencies":
            return (f"{args['service']} depends on {data['depends_on'] or 'nothing'}; "
                    f"depended on by {data['depended_on_by'] or 'nothing'}")
        if tool == "create_incident_note":
            return f"incident note {data['note_id']} {data.get('status')}: {args.get('title')}"
        if tool == "rollback_deployment":
            return f"rollback {args['service']} -> {args['to_version']}: {data.get('status')}"
    except (KeyError, TypeError):
        pass
    return f"{tool} result"
