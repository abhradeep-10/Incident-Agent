# AI usage

> Draft: edit this so it reflects exactly what you did, including any other tools you used (Cursor, Copilot, etc.).

## Tools used

- **Claude (Anthropic chat assistant).** Used for design discussion, first drafts of the modules, the mock data story, the test and eval scenarios, and the documentation.
- **Qwen2.5-14B-Instruct via Ollama.** This is the model the agent runs on, not a coding assistant. It was used to run the live evals and find prompt weaknesses.

## What AI was used for

- Scaffolding the agent loop, executor and tool schemas, which I then reviewed line by line.
- Generating a coherent mocked world: a clear incident, an ambiguous incident, a red herring, empty telemetry and broken backends.
- Drafting the pytest cases and evaluation scenarios, and the architecture write-up.

## Where the AI suggestion was wrong or unsafe, and what I changed

1. **Rollback approval through chat.** The first suggestion was to confirm a rollback by having the LLM ask "are you sure?" and execute when the user replied "yes". That is unsafe. The model decides whether a "yes" was given, so a hallucination, an ambiguous reply, or text injected through log contents could trigger a production rollback. I changed it so the executor never runs approval-required tools. They are queued, and only `agent.approve(action_id)`, called by the CLI after the human types `approve A1`, can execute them. The action is re-validated at approval time.
2. **Blanket retries.** The suggested retry wrapper retried every tool on timeout. For a non-idempotent action like a rollback, a timeout does not mean it failed, so a retry could apply it twice. I added an `idempotent` flag: only idempotent tools are retried, and incident-note creation was made idempotent by title.
3. **Trusting the model's final text.** The draft simply returned whatever the model wrote. With a local 14B model, I observed citations to evidence ids that did not exist. I added the citation checker with one repair round and a visible warning afterwards.
4. **A test helper bug.** A generated test helper passed `metric=` twice (once explicitly and once through `**kwargs`), which would raise `TypeError`. I rewrote it to merge a dict.
