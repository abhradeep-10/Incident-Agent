# Incident Investigation Agent (Task 2)

An AI agent that investigates production incidents. It decides which tools to call, uses each result to choose the next step, keeps state across follow-up questions, and produces a report that separates cited facts from hypotheses.

The agent runs **only open-weights models**, served locally through an OpenAI-compatible endpoint. The default is **Qwen2.5-14B-Instruct** (Apache-2.0) on Ollama. vLLM, llama.cpp server and LM Studio also work without code changes.

## Layout

```
incident_agent/
  agent.py      orchestration loop (plan -> act -> observe -> answer), approval API
  executor.py   safe tool execution: validation, dedupe, budget, timeout, retry, approval gate
  tools.py      tool JSON schemas + argument/output validators
  mock_data.py  deterministic mocked backend (metrics, logs, deploys, deps) + fault injection
  llm.py        OpenAI-compatible client for open models, text tool-call fallback, ScriptedLLM
  state.py      session: chat history, evidence ledger (E1..), failures, pending actions
  prompts.py    system prompt;  report.py  citation checker;  summarize.py  ledger summaries
  cli.py        interactive CLI (human approval happens here)
tests/          offline automated tests (no model needed)
evals/          14 live-model scenarios + runner
DESIGN.md  EVALUATION.md  AI_USAGE.md  Modelfile
```

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
# 1. Model (about 9 GB). Ollama's default context is too small for multi-step tool use,
#    so create a 16k-context variant from the included Modelfile.
ollama pull qwen2.5:14b
ollama create qwen2.5-14b-16k -f Modelfile

# 2. Python environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On lower-end hardware, change the `FROM` line in `Modelfile` to `qwen2.5:7b` and create `qwen2.5-7b-16k`. It works, but it is noticeably less reliable on the ambiguous scenarios.

To use vLLM instead:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
```

## Run

```bash
# Automated tests: offline and deterministic, no model needed
pytest -q

# Interactive session (--trace prints every LLM/tool step to stderr)
python -m incident_agent --trace

# One-shot questions, run in sequence in the same session
python -m incident_agent -q "Investigate why the checkout service had increased errors yesterday afternoon." \
                         -q "Was there a deployment around the time of the checkout incident?"

# Live evaluation against the model (writes evals/results/<timestamp>/)
python evals/run_evals.py
python evals/run_evals.py --only S03 S08 --repeat 3
```

In the REPL, the commands are `/evidence` (ledger), `/actions`, `/notes` (created incident notes), `/reset`, `/save` and `/quit`.

Environment variables are `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` and `AGENT_NOW`. The mock world is anchored at `2026-09-24T10:00Z`, so "yesterday" means 2026-09-23. Full traces are written to `traces/trace.jsonl`.

## Mock world (what the agent can discover)

| Service | Situation |
|---|---|
| checkout-api | Clear incident on 09-23, 14:37–15:55. v142, deployed at 14:32, upgraded the DB pool library and leaks connections, causing pool timeouts, 500s and a saturated orders-db. |
| payment-service | Ambiguous incident on 09-22, 10:03–11:20. The external gateway slowed at 10:02, *before* v88 (10:05) raised retries from 2 to 5, which then amplified load and caused 429s. |
| auth-service | v310 deployed on 09-23 at 14:00 with no impact (red herring). |
| inventory-service | Healthy. |
| search-service | Telemetry not ingesting, so tools return empty results. |
| fraud-service | The metrics backend times out and the logs backend returns malformed payloads. |

See `DESIGN.md` for architecture and trade-offs, and `EVALUATION.md` for the scenarios and expected behaviour.
