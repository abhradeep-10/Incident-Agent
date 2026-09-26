from __future__ import annotations

import json
import os
import sys
import time


class Tracer:
    """Append-only JSONL trace of every LLM call, tool call and final answer."""

    def __init__(self, path: str | None = None, echo: bool = False):
        self.path, self.echo = path, echo
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def event(self, kind: str, **fields):
        rec = {"ts": round(time.time(), 3), "kind": kind, **fields}
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if self.echo:
            short = ", ".join(f"{k}={str(v)[:140]}" for k, v in fields.items())
            print(f"  [trace] {kind}: {short}", file=sys.stderr)
