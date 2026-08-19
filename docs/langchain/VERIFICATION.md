# Verification checklist

Every code sample here has been run against the installed packages, so imports,
signatures, and data structures are correct. What has **not** been confirmed is
**behaviour against a live model** — whether the agent actually declines, cites,
refrains, or resists.

This file lists every one of those claims. Work through it with a real model and
record what happened. Corrections found here are more valuable than new modules.

**How to use it:** run the module's walkthrough, check the boxes, and where
reality differs from the claim, write what you actually saw in the Notes column.
Then fix the module (or tell me and I will).

**Record your setup first**, because these results are model-dependent:

```
Setup:        Claude / Ollama
Model:        ...
Date:         ...
Tested by:    ...
```

---

## Module 0 — Setup & Mental Model

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 0.1 | The one-screen agent returns a real answer | ☐ | |
| 0.2 | Changing `get_weather` to return `"It's -40°C"` **changes the answer** — proving the tool was genuinely called, not imagined | ☐ | |

---

## Module 1 — Models & Messages

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 1.1 | Response type is `AIMessage` | ☐ | |
| 1.2 | `.content_blocks` is a **list of dicts** with a `type` key, not a bare string | ☐ | |
| 1.3 | `usage_metadata` reports non-zero input/output counts | ☐ | |
| 1.4 | In the multi-turn loop, `input_tokens` **increases every turn** despite same-length questions | ☐ | |

*1.4 is the module's central cost argument. If it does not reproduce, the claim needs rewording.*

---

## Module 2 — Tools

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 2.1 | "Chennai in fahrenheit" → calls `get_weather` with `units="fahrenheit"` | ☐ | |
| 2.2 | "Paris" → tool returns the error string, and the agent **reports available cities** instead of inventing a temperature | ☐ | |
| 2.3 | "Convert 100 USD" → calls `convert_currency`, not `get_weather` | ☐ | |
| 2.4 | **"Thanks" → `tool-calling turns: 0`** | ☐ | |
| 2.5 | Replacing the docstring with `"""Gets stuff."""` **degrades** tool selection (Exercise 5.2) | ☐ | |

*2.4 is capability-dependent — expected to fail on small local models.*

---

## Module 3 — Agents & Tracing

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 3.1 | Message list alternates Human → AI(tool_calls) → Tool → AI | ☐ | |
| 3.2 | At least one `AIMessage` carries `tool_calls` | ☐ | |
| 3.3 | `ToolMessage` count equals the number of lookups the question needed | ☐ | |
| 3.4 | Traces appear in LangSmith and show per-step tokens | ☐ | |
| 3.5 | `broken_agent.py` **actually loops** and hits the cap | ☐ | |
| 3.6 | The fix in the spoiler block resolves it | ☐ | |

*3.5 is the one to check hardest — the whole assignment depends on it looping.*

---

## Module 4 — Structured Output

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 4.1 | Returns a validated object, not a string to parse | ☐ | |
| 4.2 | `wants_demo` is a real `bool` | ☐ | |
| 4.3 | `urgency` is one of low/medium/high | ☐ | |
| 4.4 | **Sparse email → `company` and `budget_inr` are `None`, not invented** | ☐ | |
| 4.5 | Making `company` required → model **invents** a company (Exercise 5.2) | ☐ | |

*4.4 and 4.5 are the module's core argument. 4.5 failing to reproduce would be interesting — note exactly what it produced.*

---

## Module 5 — Memory & State

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 5.1 | Thread `priya` recalls the team across separate `invoke()` calls | ☐ | |
| 5.2 | Thread `rahul` **does not know** Priya's name or team | ☐ | |
| 5.3 | Restarting the process loses everything with `InMemorySaver` | ☐ | |
| 5.4 | `SqliteSaver` survives a restart | ☐ | |
| 5.5 | Summarisation fires at the configured trigger; note which turn | ☐ | |

*5.1 and 5.2 were verified with a fake model. Confirm against a real one.*

---

## Module 6 — Retrieval Foundations *(no model needed)*

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 6.1 | Chunk count exceeds document count | ☐ | |
| 6.2 | Metadata carries **both** the heading trail and `source` | ☐ | |
| 6.3 | Semantically correct doc ranks first with **no shared keywords** | ☐ | |
| 6.4 | Nonsense query still returns `k` results, at much lower scores | ☐ | |
| 6.5 | Chunk-size comparison (300/1000/3000) shows a measurable difference | ☐ | |

*6.1–6.4 were verified locally. 6.5 needs your own corpus.*

---

## Module 7 — Agentic RAG

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 7.1 | Leave question answered **with a source citation** | ☐ | |
| 7.2 | **"What is the capital of France?" → declines**, does not answer Paris | ☐ | |
| 7.3 | `Sources:` is empty for the France question (no retrieval attempted) | ☐ | |
| 7.4 | Direct-override injection payload fails to compromise it | ☐ | |
| 7.5 | Role-confusion payload fails | ☐ | |
| 7.6 | Exfiltration ("reveal your system prompt") fails | ☐ | |
| 7.7 | Delayed-instruction payload fails | ☐ | |
| 7.8 | Encoded/obfuscated payload fails | ☐ | |

*7.4–7.8 are the assignment. **Expect some to succeed** — that is a real finding, not a bug. Record which, and note that the read-only toolset limited the damage regardless.*

---

## Module 8 — Middleware

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 8.1 | Custom `@before_model` fires and appears in the audit trail | ☐ | |
| 8.2 | `gate_expensive_tools` blocks `run_report`; agent explains rather than crashing | ☐ | |
| 8.3 | Email address does **not** reach the model verbatim (`PIIMiddleware`) | ☐ | |
| 8.4 | Flipping `ALLOW_EXPENSIVE = True` changes `blocked` → `tool_call` | ☐ | |
| 8.5 | Moving `PIIMiddleware` last changes when redaction happens (Exercise 5.2) | ☐ | |

*8.1 was verified with a fake model. 8.3 is worth checking carefully — confirm in the trace, not just the output.*

---

## Module 9 — Evaluation

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 9.1 | `evaluate()` runs and produces per-example scores | ☐ | |
| 9.2 | `declines_when_out_of_scope` scores 1.0 on the France question | ☐ | |
| 9.3 | `retrieved_correct_doc` is **below 1.0** somewhere — if perfect, the dataset is too easy | ☐ | |
| 9.4 | Changing `k` from 2 to 5 produces a **visible trade-off**, not uniform improvement | ☐ | |

*9.4 is the module's thesis. If everything improves with no cost, say so — it would mean the dataset is not discriminating.*

---

## Module 10 — LangGraph

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 10.1 | Small claim → `APPROVED (by policy)`, no pause | ☐ | |
| 10.2 | Large claim → **pauses**, then resumes to `APPROVED (by human)` | ☐ | |
| 10.3 | Invalid claim → `REJECTED (by validation)` without any model involvement | ☐ | |
| 10.4 | Removing `add_messages` breaks history as described (Exercise 5.2) | ☐ | |
| 10.5 | Interrupt survives a **process restart** with a persistent checkpointer | ☐ | |

*10.5 is the assignment's hard requirement and the one most likely to have a gap. Test it properly — new process, not same-process resume.*

---

## Module 11 — Multi-Agent

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 11.1 | Invoice ticket → delegates to billing **once** | ☐ | |
| 11.2 | API error ticket → delegates to technical | ☐ | |
| 11.3 | **"Office hours" → handled directly, no delegation** | ☐ | |
| 11.4 | **Ambiguous ticket → escalates rather than guessing** | ☐ | |
| 11.5 | Removing "cannot see this conversation" breaks a follow-up reference (Exercise 5.2) | ☐ | |
| 11.6 | Multi-agent costs measurably more than the single-agent baseline | ☐ | |

*11.3 and 11.4 are capability-dependent. 11.6 needs the baseline built.*

---

## Module 12 — Production

| # | Claim | ✓ | Notes |
|---|---|---|---|
| 12.1 | Output arrives **incrementally**, not all at the end | ☐ | |
| 12.2 | Two `session_id` values are isolated | ☐ | |
| 12.3 | A forced exception yields the user-facing error, not a traceback | ☐ | |
| 12.4 | `stream_mode` chunk counts match the table (values 2, updates 1, messages 3) | ☐ | |
| 12.5 | Sync call inside a tool measurably reduces async throughput (Exercise 5.3) | ☐ | |

*12.4 was measured on a fake model. Confirm the shape holds on a real one.*

---

## Findings log

Anything that surprised you, was unclear, took longer than the estimate, or was
simply wrong. This is the most valuable output of the pass.

| Module | What happened | Suggested fix |
|---|---|---|
| | | |

---

## Time check

Each module carries an estimate. Record the real number — the estimates are
guesses and the joiners will plan around them.

| Module | Estimated | Actual |
|---|---|---|
| 0 | 1–2 h | |
| 1 | 3–4 h | |
| 2 | 4–5 h | |
| 3 | 4–5 h | |
| 4 | 4–5 h | |
| 5 | 4–5 h | |
| 6 | 6–8 h | |
| 7 | 6–8 h | |
| 8 | 4–5 h | |
| 9 | 6–8 h | |
| 10 | 6–8 h | |
| 11 | 5–6 h | |
| 12 | 10–15 h | |
