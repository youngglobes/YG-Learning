# Module 12 — Production

**Phase:** Orchestration & Production
**Prerequisites:** Modules 0–11
**Verified against:** `langchain` 1.3.14, `langgraph` 1.2.10, Python 3.12
**Estimated time:** 10–15 hours including the capstone

---

## 1. Why this matters

Everything you have built runs on your laptop, for one user, who is you, and who waits patiently and never sends anything malicious.

Production is none of those things. This module covers what changes, then the capstone puts it together.

---

## 2. Concepts

### 2.1 Streaming

A ten-second wait with no output reads as broken. Streaming is a user-experience requirement, not a nicety.

The compiled agent exposes `stream`, `astream`, `astream_events`, and `stream_events`. `stream` takes `stream_mode` among its parameters, and the modes behave differently — measured on a trivial one-turn run:

| `stream_mode` | Chunks | Chunk type | Use for |
|---|---|---|---|
| `values` | 2 | `dict` | full state after each step |
| `updates` | 1 | `dict` | just what changed — usually what you want |
| `messages` | 3 | `tuple` | token-by-token to a UI |
| `debug` | 2 | `dict` | development |
| `custom` | 0 | — | only what tools emit via the stream writer |

```python
for chunk in agent.stream({"messages": [...]}, stream_mode="messages"):
    ...   # push to the client
```

`custom` returning nothing is not a bug — it carries only what your tools deliberately write.

For new applications LangChain recommends **event streaming**, the typed-projection API introduced in v1.3, which gives separate iterators per projection rather than branching on chunk shape. Reach for it when you are building a real UI; `stream_mode` is fine for scripts.

### 2.2 Async

One user at a time is a synchronous problem. Many concurrent users is not — a blocking call holds a worker while it waits on a network round trip that may take seconds.

`ainvoke`, `astream`, and `astream_events` are all available. Two rules:

- **Async all the way down.** One synchronous call in a tool blocks the event loop and wipes out the benefit.
- **Async middleware for async agents.** The hooks have `a*` twins (`abefore_model`, `awrap_tool_call`) for a reason — a sync hook on an async path is a silent blocker.

### 2.3 Rate limits, retries, timeouts

Every provider rate-limits, and under real load you will hit it.

- **Retry with exponential backoff and jitter.** Fixed-interval retries from many workers synchronise into a thundering herd.
- **Retry 429 and 5xx; do not retry 400.** A malformed request is malformed forever.
- **Set timeouts.** A hung request holding a worker forever is worse than a fast failure.
- **`ModelFallbackMiddleware`** switches provider or model when the primary fails — often better than retrying into a wall.
- `ToolRetryMiddleware` covers flaky tools, with `max_retries`, `backoff_factor`, `initial_delay`, `max_delay`, and `jitter`.

### 2.4 Cost control

The four levers, in order of impact:

1. **Prompt caching.** Large stable prefixes — system prompt, tool definitions, retrieved context — can be cached by most providers at a large discount. The biggest single win for agent workloads, because that prefix is re-sent every step (Module 1 §2.4).
2. **Right-sized models.** Classification and routing rarely need a frontier model. Reserve it for synthesis.
3. **Caps everywhere.** `ModelCallLimitMiddleware` on every agent. A bug without a cap is an invoice.
4. **Context management.** `SummarizationMiddleware` and `ContextEditingMiddleware` keep transcripts from growing without bound.

Track cost per request as a first-class metric. A system that works and costs ₹40 per conversation is not a working system.

### 2.5 Failure and degradation

LLM calls fail. Decide per path what happens:

| Failure | Reasonable response |
|---|---|
| Provider down | Fall back to another model, or a cached/canned answer |
| Rate limited | Queue and retry with backoff; tell the user it is queued |
| Tool fails | Return a recoverable error (Module 2) and let the agent adapt |
| Timeout | Return partial results if any; fail clearly |
| Model refuses | Surface it honestly; do not silently retry the same prompt |

The general rule: **degrade visibly rather than fail silently.** A wrong answer delivered confidently is worse than an apology.

### 2.6 Security checklist

Everything from earlier modules, in one place:

- **Prompt injection** via retrieved documents and user input (Module 7). Prompt defence plus least-privilege tools.
- **Thread isolation** — `thread_id` from the authenticated session, never client-supplied (Module 5 §2.3).
- **Secrets** in a secret manager, never in prompts, code, or checkpoints.
- **PII** — `PIIMiddleware`, and a decision about what traces and checkpoints retain (Modules 3 §2.5, 8).
- **Least privilege** — each agent gets only the tools its job needs (Modules 7, 11).
- **Output handling** — never `eval` model output; parameterise any query built from it; escape anything rendered.
- **Rate limit your own endpoint**, or one user can spend your entire budget.

### 2.7 Observability

Tracing (Module 3) plus operational metrics. Alert on: error rate, p95 latency, cost per request, token volume, cap-hit frequency, and eval score trend (Module 9). Cap-hit frequency is the early warning that something has started looping.

---

## 3. Walkthrough

```python
"""Module 12 — production-shaped skeleton: async, streaming, caps, fallback."""
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware, ModelFallbackMiddleware,
    PIIMiddleware, SummarizationMiddleware, ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

MODEL = "anthropic:claude-opus-5"

agent = create_agent(
    model=MODEL,
    tools=[...],                                  # your tools
    system_prompt="...",
    checkpointer=InMemorySaver(),                 # SqliteSaver/Postgres in prod
    middleware=[
        PIIMiddleware("email", strategy="redact"),          # outermost
        SummarizationMiddleware(model=MODEL, trigger={"tokens": 8000},
                                keep={"messages": 8}),
        ToolRetryMiddleware(max_retries=2, jitter=True),
        ModelFallbackMiddleware("anthropic:claude-sonnet-5"),
        ModelCallLimitMiddleware(run_limit=10),             # innermost backstop
    ],
)


async def handle(question: str, session_id: str):
    """One request. thread_id comes from the authenticated session, never the client."""
    cfg = {"configurable": {"thread_id": f"sess::{session_id}"}}
    try:
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=cfg, stream_mode=["updates", "messages"],
        ):
            yield mode, chunk
    except Exception as exc:                       # degrade visibly
        yield "error", {"message": "I couldn't complete that. Please retry.",
                        "detail": type(exc).__name__}


async def main():
    async for mode, chunk in handle("What is the leave policy?", session_id="user-42"):
        print(mode, str(chunk)[:100])


if __name__ == "__main__":
    asyncio.run(main())
```

Note the middleware ordering, the `thread_id` derived from the session rather than from input, and the exception path yielding a user-facing message instead of a traceback.

---

## 4. Run it

```bash
.venv/bin/python service.py
```

**Expected output — illustrative.** Structural checks: output arrives **incrementally**, not in one block at the end; two different `session_id` values produce isolated conversations; and forcing an exception (point the model at a bad name) yields the `error` message rather than a stack trace.

---

## 5. Exercises

**5.1 Recall.** Name the four cost levers in order of impact, and say why prompt caching is first for agent workloads specifically.

**5.2 Apply.** Run the same agent with `stream_mode` set to `values`, `updates`, and `messages`. Record chunk counts and types. Decide which you would send to a browser and justify it.

**5.3 Extend.** Add a synchronous `time.sleep(2)` inside a tool on the async path and measure throughput under ten concurrent requests. Then make it async and measure again. Report both numbers — this is §2.2 made concrete.

---

## 6. Capstone — AI Helpdesk Assistant

Everything from the path, in one system.

**Functional**
- Answers from a real document corpus with citations (Modules 6, 7)
- Multi-turn with per-user persistence (Module 5)
- At least one action tool with a human approval gate (Modules 2, 10)
- Escalation to a human on low confidence or ambiguity
- Structured output where another system consumes it (Module 4)

**Non-functional**
- Streaming responses (§2.1)
- Async throughout (§2.2)
- Call limits on every agent; retry and fallback configured
- Persistent checkpointing
- Full tracing

**Deliverables**
1. **Source code**, running from a clean clone with documented setup
2. **README** — architecture, setup, configuration
3. **Architecture diagram**
4. **Eval suite** (Module 9) with ≥30 examples and a passing CI job
5. **Cost model** — measured cost per conversation, and per 1,000 conversations
6. **Security review** covering §2.6, naming at least one real residual risk you found yourself
7. **Runbook** — what alerts exist, what to do when each fires

**Pass criteria**
- Runs from a clean clone
- Eval suite passes in CI
- The cost model is defensible and based on measurement, not arithmetic on list prices
- The security review names a real risk, not a generic checklist
- **A deliberate injection attempt in the corpus fails to compromise it** (Module 7)

**Assessment weighting:** roughly half the marks are on items 4–7. A system that works but that you cannot measure, cost, secure, or operate is a prototype. The path has been building toward the difference.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Response appears only at the end | Not streaming | `astream` with an appropriate `stream_mode` |
| Async agent no faster under load | A sync call in a tool or middleware | Async all the way down (§2.2) |
| Repeated 429s | Retrying without backoff | Exponential backoff + jitter; fallback model |
| Cost far above estimate | Re-sent history uncached | Prompt caching; summarisation; caps |
| Users see each other's history | `thread_id` from client input | Derive from authenticated session |
| Stack traces reaching users | No error boundary | Catch and return a user-facing message |
| Everything lost on deploy | `InMemorySaver` | SQLite / Postgres |
| Quality drifts, nobody notices | No eval in CI | Module 9 |
| Runaway loop found on the invoice | No caps, no alerting | Caps everywhere; alert on cap-hit rate |

---

## 8. Check yourself

1. **Why is prompt caching the biggest cost lever for agents specifically?**
   The stable prefix is re-sent on every step of the loop, so caching it saves on every iteration rather than once per conversation.

2. **You made the agent async and throughput did not improve. First suspect?**
   A synchronous call inside a tool or middleware blocking the event loop.

3. **Which `stream_mode` for token-by-token UI output?**
   `messages`. `updates` is the usual choice for server-side progress.

4. **What does "degrade visibly" mean?**
   On failure, tell the user something honest rather than returning a confident wrong answer or a silent empty one.

5. **A system passes every functional requirement but has no eval suite and no cost model. Ship it?**
   No. You cannot tell whether a change makes it worse, and you do not know what it costs to run.

---

## 9. References

- Streaming — https://docs.langchain.com/oss/python/langchain/streaming
- Middleware — https://docs.langchain.com/oss/python/langchain/middleware
- LangSmith observability — https://docs.langchain.com/langsmith/observability
- API reference — https://reference.langchain.com

---

*End of the LangChain learning path. Back to the [syllabus](./index.md).*
