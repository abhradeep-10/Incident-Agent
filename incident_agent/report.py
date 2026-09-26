"""Post-hoc check that the final answer is grounded in the evidence ledger."""
from __future__ import annotations

import re

_ID = re.compile(r"\bE\d+\b")
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_HEADING = re.compile(r"^\s*(#{1,6}\s+.+|\*\*[^*]+\*\*:?\s*)$")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def cited_ids(text: str) -> set[str]:
    out: set[str] = set()
    for inner in _BRACKET.findall(text):
        out.update(_ID.findall(inner))
    return out


def _section_lines(text: str, name: str) -> list[str]:
    inside, out = False, []
    for line in text.splitlines():
        if _HEADING.match(line):
            if inside:
                break
            inside = name in line.lower()
            continue
        if inside:
            out.append(line)
    return out


def check_report(text: str, known_ids: set[str], gathered_this_turn: bool) -> list[str]:
    issues = []
    cited = cited_ids(text)
    unknown = sorted(cited - known_ids, key=lambda x: int(x[1:]))
    if unknown:
        issues.append(f"cites evidence ids that do not exist: {', '.join(unknown)}")
    if gathered_this_turn and not cited:
        issues.append("no claim is cited with an evidence id such as [E1]")
    uncited = [l for l in _section_lines(text, "observed facts") if _BULLET.match(l) and not _ID.search(l)]
    if uncited:
        issues.append(f"{len(uncited)} bullet(s) under 'Observed facts' have no evidence citation")
    return issues
