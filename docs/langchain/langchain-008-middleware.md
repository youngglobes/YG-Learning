# Module 8: Middleware

**Phase:** Orchestration & Production
**Prerequisites:** Modules 0-7
**Verified against:** `langchain` 1.3.14, Python 3.12
**Estimated time:** 4-5 hours

---

## 1. Why this matters

You have needed middleware three times already without naming it: capping the loop in Module 3, summarising history in Module 5, and the PII question that surfaced in Module 3's tracing section.

Middleware is the v1 extension point, the supported way to insert behaviour into the agent loop without rewriting the agent. It has no equivalent in pre-v1 tutorials, which is why they resort to wrapping and subclassing.

The module also teaches a habit worth more than the API: **check what already exists before you build.**

---

## 2. Concepts

### 2.1 Where it sits

```
  before_agent ──► [ before_model ──► MODEL ──► after_model
                        │                          │
                        │                    wrap_tool_call
                        │                          │
                        └────── TOOLS ◄────────────┘ ] loop ──► after_agent
```

Six hooks, verified on `AgentMiddleware`:

| Hook | Fires | Typical use |
|---|---|---|
| `before_agent` | once, at start | set up, load context |
| `before_model` | every model call | inject state, redact, audit, enforce budget |
| `after_model` | every model response | inspect or rewrite output |
| `wrap_model_call` | around each model call | retries, fallback, timing |
| `wrap_tool_call` | around each tool call | approval gates, logging, sandboxing |
| `after_agent` | once, at end | teardown, final logging |

Each has an async twin (`abefore_model`, `awrap_tool_call`, …).

### 2.2 Read the built-in list first

`langchain.agents.middleware` already ships roughly twenty. Verified in the installed package:

```
AgentMiddleware              ContextEditingMiddleware
PIIMiddleware                SummarizationMiddleware
ModelCallLimitMiddleware     ToolCallLimitMiddleware
ModelFallbackMiddleware      ModelRetryMiddleware
ToolRetryMiddleware          ToolErrorMiddleware
HumanInTheLoopMiddleware     LLMToolSelectorMiddleware
LLMToolEmulator              TodoListMiddleware
ShellToolMiddleware          FilesystemFileSearchMiddleware
ProviderToolSearchMiddleware ClearToolUsesEdit
```

An earlier draft of this course set "write PII redaction middleware" as the assignment. `PIIMiddleware` already exists, with parameters `pii_type, strategy, detector, apply_to_input, apply_to_output, apply_to_tool_results`. Reimplementing it would have taught the wrong instinct.

**So the first step of any middleware task is `dir(langchain.agents.middleware)`.** The library moves faster than any tutorial, this one included.

### 2.3 Composing built-ins

```python
from langchain.agents.middleware import (
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider
    PIIMiddleware, SummarizationMiddleware,
    ModelCallLimitMiddleware, ToolRetryMiddleware,
)

agent = create_agent(
    model=MODEL,
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        SummarizationMiddleware(model=MODEL,
                                trigger={"tokens": 4000}, keep={"messages": 6}),
        ModelCallLimitMiddleware(run_limit=10),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

**Order matters.** Middleware composes as layers: the first entry is outermost on the way in. Redaction must run *before* anything that ships data outward, so put it early. When behaviour surprises you, reorder before you rewrite.

### 2.4 Writing your own

Two styles. The decorator is the concise one:

```python
from langchain.agents.middleware import before_model

@before_model
def audit(state, runtime):
    """Record every model call to our own audit log."""
    log_to_audit_table(len(state["messages"]))
    return None          # None = no state change
```

Verified running: attached to an agent, this fired once for a one-turn conversation, receiving a state whose `messages` list had 1 entry.

The decorators accept extra arguments, `@before_model(func, state_schema, tools, can_jump_to, name)`, and `@wrap_tool_call(func, tools, name)` lets you scope to specific tools. There is also `@dynamic_prompt(func)` for a system prompt computed per call.

For anything stateful, subclass instead:

```python
from langchain.agents.middleware import AgentMiddleware

class BusinessHoursMiddleware(AgentMiddleware):
    """Refuse to run expensive tools outside working hours."""

    def wrap_tool_call(self, request, handler):
        if request.tool_name in EXPENSIVE and not within_business_hours():
            return "This tool is only available 09:00-18:00 IST."
        return handler(request)
```

### 2.5 When *not* to write middleware

Middleware runs on every call in its scope, which makes it the wrong home for logic that belongs elsewhere:

- **One tool needs special handling** → put it in the tool.
- **The model should behave differently** → change the system prompt.
- **A one-off transformation** → do it in your calling code.

Reach for middleware when the behaviour is genuinely **cross-cutting**: it applies to every model call or every tool call, and duplicating it into each tool would be worse.

---

## 3. Walkthrough

```python
"""Module 8: compose built-ins, then add one genuinely custom layer."""
from dotenv import load_dotenv
load_dotenv()
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware, PIIMiddleware, ToolRetryMiddleware,
    before_model, wrap_tool_call,
)
from langchain.tools import tool

AUDIT: list[dict] = []
EXPENSIVE = {"run_report"}


@tool(parse_docstring=True)
def run_report(name: str) -> str:
    """Run a named analytics report. Expensive; use sparingly.

    Args:
        name: Report identifier, e.g. monthly_revenue.
    """
    return f"Report '{name}': 412 rows, 3.2s."


# --- custom layer 1: audit every model call ------------------------------
@before_model
def audit_model_calls(state, runtime):
    """Record the size of the transcript at each model call."""
    AUDIT.append({"event": "model_call", "messages": len(state["messages"])})
    return None


# --- custom layer 2: gate expensive tools --------------------------------
@wrap_tool_call
def gate_expensive_tools(request, handler):
    """Block expensive tools unless explicitly allowed this run."""
    if request.tool_name in EXPENSIVE and not ALLOW_EXPENSIVE:
        AUDIT.append({"event": "blocked", "tool": request.tool_name})
        return f"'{request.tool_name}' is disabled for this run. Ask an admin."
    AUDIT.append({"event": "tool_call", "tool": request.tool_name})
    return handler(request)


ALLOW_EXPENSIVE = False

agent = create_agent(
    model=MODEL,
    tools=[run_report],
    system_prompt="You are an analytics assistant.",
    middleware=[
        PIIMiddleware("email", strategy="redact"),   # outermost: redact first
        audit_model_calls,
        gate_expensive_tools,
        ToolRetryMiddleware(max_retries=2),
        ModelCallLimitMiddleware(run_limit=6),       # innermost backstop
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content":
    "Run the monthly_revenue report and email it to priya@acmecorp.in"}]})

print(result["messages"][-1].text)
print("\naudit trail:")
for row in AUDIT:
    print("  ", row)
```

---

## 4. Run it

```bash
.venv/bin/python middleware_demo.py
```

**Expected output, illustrative.** Three structural checks:

1. The audit trail contains at least one `model_call` entry, your custom `before_model` ran.
2. It contains a `blocked` entry for `run_report`, and the agent's reply tells the user it is disabled rather than crashing.
3. The email address does not appear verbatim in what was sent to the model, `PIIMiddleware` redacted it.

Then flip `ALLOW_EXPENSIVE = True` and re-run. The `blocked` entry becomes `tool_call` and the report runs. One flag, no change to the tool or the prompt, that is what "cross-cutting" buys you.

---

## 5. Exercises

**5.1 Recall.** Name the six hooks and say which one you would use to add an approval gate to a destructive tool.

**5.2 Apply.** Move `PIIMiddleware` from first to last in the list and re-run. Inspect the trace and determine whether redaction still happened before data left the process. Write down what you found and what it implies about ordering.

**5.3 Extend.** Write a middleware that enforces a per-run **token** budget (not a call count) using `usage_metadata`, and stops cleanly with a message when exceeded. Compare it with `ModelCallLimitMiddleware` and explain when each is the better control.

---

## 6. Assignment

Two parts, deliberately.

**Part A, composition.** Take your Module 7 RAG agent and add: PII redaction, summarisation, a model call limit, and tool retry. All four are built-ins. No custom code.

**Part B, one custom middleware** for something the library genuinely does not cover. Examples: writing every tool call to your own audit table; blocking requests outside business hours; injecting the current user's department into the system prompt via `@dynamic_prompt`; refusing queries containing customer identifiers.

Deliverables:

- Working agent with both parts
- A `MIDDLEWARE.md` listing your stack **in order**, with one line per layer explaining why it sits where it does
- A short note answering: **"which built-ins did you consider and reject, and why?"**

That last note is the assessment. An answer of "I didn't look" fails, because §2.2 is the point of the module.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Middleware never runs | Not passed in `middleware=[...]` | Pass it |
| Redaction happens too late | Ordering | Put redaction first (§2.3) |
| Custom middleware silently changes nothing | Returned a value where `None` was expected, or vice versa | `None` = no state change |
| `wrap_tool_call` swallows all tools | No scoping | `@wrap_tool_call(tools=[...])`, or branch on `request.tool_name` |
| Reimplemented a built-in | Did not check the list | `dir(langchain.agents.middleware)` first |
| Async agent, middleware ignored | Sync hook on an async path | Implement `abefore_model` / `awrap_tool_call` |
| Behaviour changes when you add a layer | Composition order | Reorder before rewriting |

---

## 8. Check yourself

1. **Six hooks, which for an approval gate?**
   `wrap_tool_call`. It surrounds the tool call, so you can allow, block, or substitute a result.

2. **Why is ordering significant?**
   Middleware composes as layers; the first is outermost. Anything protective, redaction, budget checks, must sit outside what it protects.

3. **You need PII redaction. First move?**
   Check the built-ins. `PIIMiddleware` exists.

4. **When is middleware the wrong tool?**
   When the behaviour applies to one tool (put it in the tool), or is really a model-behaviour change (system prompt), or is a one-off (calling code).

5. **Decorator or subclass?**
   Decorator for a stateless hook; subclass when you need state, configuration, or several hooks working together.

---

## 9. References

- Middleware: https://docs.langchain.com/oss/python/langchain/middleware
- Agents: https://docs.langchain.com/oss/python/langchain/agents
- API reference: https://reference.langchain.com

---

*Next: [Module 9: Evaluation](./langchain-009-evaluation.md). You have been changing prompts for eight modules with no way to tell whether you made things worse. That ends here.*
