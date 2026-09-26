import json

from incident_agent.llm import parse_text_tool_calls
from incident_agent.report import check_report, cited_ids

NAMES = {"get_metrics", "get_deployments"}


def test_qwen_style_tagged_tool_call():
    text = ('<tool_call>\n{"name": "get_deployments", "arguments": {"service": "checkout-api", '
            '"start_time": "2026-09-23T12:00:00Z", "end_time": "2026-09-23T18:00:00Z"}}\n</tool_call>')
    calls, rest = parse_text_tool_calls(text, NAMES)
    assert len(calls) == 1 and calls[0].name == "get_deployments"
    assert json.loads(calls[0].arguments)["service"] == "checkout-api" and rest == ""


def test_bare_json_and_unknown_names():
    calls, _ = parse_text_tool_calls('{"name": "get_metrics", "parameters": {"service": "x"}}', NAMES)
    assert calls[0].name == "get_metrics"
    calls, rest = parse_text_tool_calls('{"name": "drop_tables", "arguments": {}}', NAMES)
    assert calls == [] and "drop_tables" in rest


def test_plain_text_is_not_a_tool_call():
    calls, rest = parse_text_tool_calls("The error rate rose to 17% [E1].", NAMES)
    assert calls == [] and rest.startswith("The error")


def test_citation_checks():
    assert cited_ids("a [E1] b [E2, E10] c E7") == {"E1", "E2", "E10"}
    assert check_report("fine [E1]", {"E1"}, True) == []
    assert "do not exist" in check_report("bad [E4]", {"E1"}, True)[0]
    assert check_report("no cites", {"E1"}, True)
    assert check_report("Which service?", set(), False) == []
    text = "## Observed facts\n- a [E1]\n- b\n## Hypotheses / inferences\n- c"
    assert any("Observed facts" in i for i in check_report(text, {"E1"}, True))
