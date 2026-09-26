# Design

## 1. Architecture and why

```
user ─► IncidentAgent.ask()
          │  system prompt = rules + service catalog + EVIDENCE LEDGER
          ▼
     ┌── LLM (open model, native tool calling) ──┐
     │  returns tool_calls  or  final text       │
     └──────────────┬────────────────────────────┘
          tool_calls│                    text
                    ▼                      ▼
            ToolExecutor              report checker (citations)
   validate → dedupe → budget →        └─ one repair round, else flag
   approval gate → timeout/retry →
   output validation → evidence Ei
                    │
                    └── results appended as tool messages → back to LLM
```

The agent is a **single ReAct-style loop over native function calling**, with a **deterministic safety runtime** around it. The LLM decides *what to do*. The runtime decides *what is allowed to happen*.

I chose this over the alternatives for the following reasons:

- **Fixed pipeline** (metrics → deploys → logs every time). The brief forbids it, and it wastes calls on simple lookups.
- **Planner/executor or multi-agent.** With five tools and 2–3 hours of scope, a separate planner adds latency and failure modes without improving the investigation. A single loop that sees every observation already supports "the result of one call decides the next". The obvious extension point is a planner, if tools grow to dozens.
- **Frameworks (LangGraph, CrewAI).** The whole loop is about 150 lines. Writing it directly keeps every guardrail visible and testable, and avoids framework coupling in a take-home.

### Why an open model, and which one

The system uses Qwen2.5-14B-Instruct. It is Apache-2.0 licensed, reliable at OpenAI-style tool calling, and runs locally on one consumer GPU or Apple-silicon Mac. Everything goes through an OpenAI-compatible interface, so switching to Llama 3.1, Mistral, or a vLLM deployment is configuration only. Open models have known quirks, which the design handles explicitly:

- **Tool calls printed as text** (e.g. `<tool_call>{…}</tool_call>`). `parse_text_tool_calls` recovers them, accepting known tool names only.
- **Small default context in Ollama.** The Modelfile sets `num_ctx 16384`, and tool outputs are trimmed before entering the context.
- **Weaker instruction following.** Tools return pre-computed summaries (baseline, max, first time above 2× baseline; top log messages with counts). The model reasons over compact facts instead of scanning raw series. The backend does the arithmetic and the model does the reasoning.

## 2. How the agent decides which tool to call

The model chooses from the tool schemas and descriptions. The system prompt gives *heuristics, not a script*:

- A factual lookup uses one tool.
- A cause investigation starts by confirming the anomaly and its onset. Change events, logs and dependencies are pursued **only when the evidence points there**.
- If the service or time window is unknowable, the model asks one clarifying question instead of guessing.
- Independent calls may be issued together. The executor runs read-only calls in parallel.

Because each tool result is appended before the next LLM call, the onset time from `get_metrics` is typically used as the window for `get_deployments` and `search_logs`. `test_next_step_can_depend_on_previous_result` verifies that the loop supports this data flow.

## 3. Preventing infinite loops and runaway cost

All limits are in `AgentConfig` and enforced by the runtime, not by the prompt:

| Guard | Default | Behaviour when hit |
|---|---|---|
| Tool calls per turn | 12 | Further calls are rejected with `budget_exhausted`, tools are disabled, and the model must answer from existing evidence. |
| LLM calls per turn | 10 | A deterministic fallback report is built from the ledger. |
| Identical-call dedupe | always | A repeated read-only call is **not executed**. The model gets "already done, see E3". |
| Duplicate calls per turn | 3 | `loop_detected`, then a forced final answer. |
| Same failing call | 2 | A third attempt is blocked (`repeated_failure`). |
| Consecutive all-failed rounds | 3 | `repeated_tool_failures`, then a forced final answer. |
| Tool output size | 4000 chars | Series and event lists are downsampled. The summary is kept. |
| Window size | 72 h | The call is rejected, so the model cannot request months of data. |
| Report repair | 1 round | Afterwards the answer is returned with a visible warning instead of looping. |

When tools are disabled, the next LLM call is sent **without the `tools` parameter**, so the model physically cannot continue calling. If it still produces nothing usable, `_fallback_report` returns only recorded evidence and never invented text.

## 4. Poor, failed or contradictory tool responses

- **Arguments.** Every argument is validated before execution. This covers unknown or ambiguous services (fuzzy "checkout" → `checkout-api`, but "payment" is rejected as ambiguous), metric aliases, ISO timestamps, `end > start`, future windows (rejected, or the end clamped with a note), retention limits and maximum span. Errors return as `invalid_arguments` with a precise message, so the model can self-correct once.
- **Timeouts.** Each call runs under a hard wall-clock timeout (a thread future), so a hanging backend cannot stall the agent.
- **Transient failures.** 503s and timeouts get exponential-backoff retries, **for idempotent tools only**. Rollback is never retried automatically.
- **Malformed or empty payloads.** Per-tool output validators reject them (`malformed_response`), so garbage never reaches the model disguised as data.
- **Empty results.** Genuine empty results (no telemetry, an external service) are returned as *valid* data with an explicit `note`, which keeps "no data" distinct from "tool broken".
- **Recording failures.** Every failure is recorded in the ledger. The prompt requires a "Gaps and tool issues" section and forbids guessing missing data.
- **Contradictions and ambiguity.** The prompt requires competing hypotheses with supporting *and* contradicting evidence plus a confidence level. The payment scenario is built to test this: the gateway slowdown precedes the deploy, so "the deploy caused it" is only partially supported.
- **Hallucinated evidence.** `report.py` checks that cited ids exist and that every "Observed facts" bullet carries a citation. On failure, the model gets one repair round with tools disabled. After that, the answer is shown with an explicit warning.

## 5. Conversation state

`Session` holds three things:

1. **Chat history** (user, assistant, tool messages). Only the last `history_turns` turns are sent. Whole turns are dropped so tool-call/result pairs stay intact, and older raw tool outputs are shortened.
2. **Evidence ledger.** Every successful tool result gets a stable id (`E1…`) plus a one-line deterministic summary. The ledger is injected into the system prompt on every call. That is how follow-ups ("what are its dependencies?", "when did it start?") resolve "it" and reuse findings, even after raw outputs are trimmed. It also provides the ids the report must cite, and powers dedupe.
3. **Actions** (pending, approved or rejected) and **tool failures**, so the model can truthfully answer "did the rollback happen?".

The session serialises to JSON (`--session file.json`) for persistent investigations.

## 6. Dangerous actions and human approval

`rollback_deployment` is marked `requires_approval` and `idempotent=False`. When the model calls it, the executor validates the arguments (the version must exist and must not already be running), then **queues it without executing**. The model is told it is pending.

Approval happens **outside the LLM**. The CLI asks the human to type `approve A1` exactly, then calls `agent.approve()`, which re-validates and executes. A user typing "yes" in chat cannot trigger it, and neither can a prompt-injected log line, because the LLM has no code path to execute it.

`create_incident_note` runs automatically, as the brief allows, but it is idempotent by title, and the prompt restricts it to completed investigations that found an anomaly.

## 7. What I would change for production scale

- **Real integrations.** Prometheus/Datadog, Loki/Elastic, the deploy system and a service catalog, behind the same `ToolSpec` interface. Add per-tool rate limits, circuit breakers and caching keyed on (query, window). Build anomaly summaries on proper detectors (seasonality-aware) rather than 2× baseline.
- **Security.** Treat log text as untrusted input: delimit it and strip instruction-like content. Give the tools a read-only service account. Scope approval by RBAC and record it in an audit log. Use a policy engine for which actions need which approver.
- **State.** Store sessions in Postgres or Redis with investigation ids. Summarise the evidence ledger when it grows. Support multiple users on one incident.
- **Serving.** Run vLLM with continuous batching, and use a larger open model (Qwen2.5-72B or Llama-3.3-70B) for the hard cases, with a smaller model for simple lookups (routing). Use structured output (JSON schema) for the final report.
- **Observability and evaluation.** Send OpenTelemetry spans per LLM and tool call, and track token and cost budgets per incident. Run scenario evals in CI with repeats to measure pass rate and variance, add an LLM-as-judge (an open model) for report quality, and build a regression set from real past incidents.
- **Planning.** For large toolsets, retrieve the relevant tools per question instead of sending all schemas. Consider an explicit hypothesis tracker that the agent updates and tests.
