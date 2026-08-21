# Module 3: Agents & Tracing

**Phase:** Core
**Prerequisites:** Modules 0-2
**Verified against:** `langchain` 1.3.14, `langgraph` 1.2.10, Python 3.12
**Estimated time:** 4-5 hours

---

## 1. Why this matters

This is the centre of the framework, and it is the first point where you cannot debug by reading your code.

Your code says `agent.invoke(...)`. What happens next is a loop you did not write, driven by decisions a model made, across several round trips. When the answer is wrong, the question "which step went wrong?" is not answerable from the source. You need to see the run.

So this module does two things at once: it opens up the loop, and it gets you tracing, because after this point, tracing is how you debug everything.

---

## 2. Concepts

### 2.1 The loop

```
  ┌─────────────────────────────────────────────┐
  │                                             │
  ▼                                             │
call model ──► wants tools? ──yes──► run tools ─┘
                    │
                    no
                    │
                    ▼
              return answer
```

Every arrow is a message appended to the list from Module 1. That is the whole thing. `create_agent` gives you this loop plus state handling, and nothing more mysterious.

### 2.2 `create_agent` parameters

Verified signature:

```
model, tools, system_prompt, middleware, response_format, state_schema,
context_schema, checkpointer, store, interrupt_before, interrupt_after, debug
```

You know four of these already. Here is where the rest arrive:

| Parameter | Module |
|---|---|
| `model`, `tools`, `system_prompt` | 0-2 |
| `response_format` | 4 (structured output) |
| `checkpointer`, `store` | 5 (memory) |
| `middleware` | 8 |
| `interrupt_before`, `interrupt_after` | 10 (human-in-the-loop) |
| `state_schema`, `context_schema` | 10 |

You do not need to learn them now. You need to know the list exists, so that when you want a behaviour you look here first instead of writing it yourself.

### 2.3 Capping the loop, three different limits

An agent that never terminates is the failure mode that costs money. There are three separate caps and they do different things:

```python
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

agent = create_agent(
    model=MODEL,
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(run_limit=8),                       # total model calls
        ToolCallLimitMiddleware(tool_name="search", run_limit=3),    # calls to one tool
    ],
)

# And the graph-level backstop:
agent.invoke({"messages": [...]}, config={"recursion_limit": 25})
```

- `ModelCallLimitMiddleware(run_limit=N)`, the cost cap. Model calls are what you pay for.
- `ToolCallLimitMiddleware(tool_name=..., run_limit=N)`, stops one expensive or destructive tool being hammered.
- `recursion_limit`, the framework's backstop against a runaway graph.

Both middleware also take `thread_limit` (across a whole conversation, not just one run) and `exit_behavior` (what happens on hitting the cap).

**Set a model call limit on every agent you write from now on.** It costs one line.

### 2.4 Tracing

You cannot reason about the loop from stdout. Tracing records every step, each model call, its inputs, its outputs, each tool invocation, token counts, and latency.

Turn it on with environment variables; no code change:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=yg-learning
```

Run your agent, then open the trace. What to look at, in order:

1. **How many model calls?** More than you expected means a loop you did not intend.
2. **What did the model actually receive?** Not what you think you sent, what is in the message list at each step.
3. **Which tool call went wrong?** Usually visible immediately once you can see the arguments.
4. **Where did the tokens go?** Almost always the re-sent history from Module 1, §2.4.

Tracing is free at low volume and is the difference between debugging and guessing. It is introduced here, in Module 3, rather than at the end of the path, because every module after this one is easier with it on.

### 2.5 Privacy note

Traces contain your prompts, your documents, and your users' messages. Before pointing a hosted tracing backend at anything real, know what data you are sending and check it against your obligations. For a corpus of internal HR policies this is probably fine; for customer PII it is a decision someone needs to make deliberately. Module 8's `PIIMiddleware` is one mitigation.

---

## 3. Walkthrough

```python
"""Module 3: watching the loop."""
from dotenv import load_dotenv
load_dotenv()
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool

CITIES = {"chennai": 32, "mumbai": 29, "delhi": 38}


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for an Indian city.

    Call this when the user asks about weather or temperature.

    Args:
        city: City name, e.g. Chennai.
    """
    t = CITIES.get(city.lower())
    return f"{t}C in {city.title()}" if t else (
        f"Error: no data for '{city}'. Available: {', '.join(sorted(CITIES))}."
    )


agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt="You are a concise assistant.",
    middleware=[ModelCallLimitMiddleware(run_limit=6)],   # always cap it
)

result = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "Compare the weather in Chennai and Delhi."}]},
    config={"recursion_limit": 25},
)

# Print the loop, message by message.
for i, m in enumerate(result["messages"]):
    kind = type(m).__name__
    calls = getattr(m, "tool_calls", None)
    if calls:
        detail = f"WANTS TOOLS: {[(c['name'], c['args']) for c in calls]}"
    else:
        detail = (m.text or "")[:70]
    print(f"{i}. {kind:14} {detail}")

usage = result["messages"][-1].usage_metadata
print(f"\nfinal-call tokens: {usage}")
```

Note the question needs **two** lookups. Watch how the model handles that, it may call the tool twice in one turn, or once per turn across two turns. Both are valid, and the trace shows you which.

---

## 4. Run it

> **On a small local model, some checks below will fail, and that is expected.**
> The behavioural checks in this section depend on model capability. See the
> capability tier table in [Choosing your model](./model-setup.md) before
> concluding your code is wrong.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
.venv/bin/python agent_loop.py
```

**Expected output, illustrative.** The structure is the point:

```
0. HumanMessage   Compare the weather in Chennai and Delhi.
1. AIMessage      WANTS TOOLS: [('get_weather', {'city': 'Chennai'}), ...]
2. ToolMessage    32C in Chennai
3. ToolMessage    38C in Delhi
4. AIMessage      Delhi is 6C hotter than Chennai, 38C against 32C.
```

Three checks: the message list alternates as described in Module 1; at least one `AIMessage` carries `tool_calls`; and the number of `ToolMessage`s equals the number of lookups the question required. Then open the trace and confirm the same run from the other side.

---

## 5. Exercises

**5.1 Recall.** Describe the agent loop in three sentences without looking. What ends it?

**5.2 Apply.** Set `ModelCallLimitMiddleware(run_limit=1)` and re-run. Observe what a too-tight cap does, the agent cannot both call a tool and answer. Write down what the user sees, and why that is a bad failure mode to ship.

**5.3 Extend.** Write a tool that always returns `"Task not complete, try again."` and give the agent a task using it. Watch it loop. Confirm your cap stops it. Then check the trace and calculate what that run cost.

---

## 6. Assignment, diagnose from the trace alone

You will be given a deliberately broken agent (`broken_agent.py`, in the module's `code/` folder). It loops until it hits the cap on a simple question.

**Rules:** diagnose it using the trace only. Do not read the tool bodies until you have written your diagnosis.

Produce a short report:

1. How many model calls before the cap, and the total token cost
2. The exact step where it first goes wrong, with the message contents that prove it
3. Root cause
4. Your fix, and a trace of the fixed run showing the call count dropping

The constraint is the assignment. In production you often have the trace and not the ability to reproduce locally, so reading a trace is the skill being taught.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `GraphRecursionError` | Loop never terminates | Find the tool whose result reads as "not done"; add a call limit |
| Agent loops on a fine-looking tool | Tool returns something ambiguous like `"OK"` or `""` | Return an unambiguous, complete result |
| No traces in LangSmith | `LANGSMITH_TRACING` unset, or unset *after* import | Export before running; verify the project name |
| Traces appear under the wrong project | `LANGSMITH_PROJECT` not set | Set it per project |
| Cap hit on legitimate work | Limit too low for a multi-lookup question | Raise it deliberately; do not remove it |
| Token cost far above expectation | Re-sent history | Module 1 §2.4. Trim history or reduce loop length |
| Agent answers without calling any tool | Tool description problem | Module 2 §2.3 |

---

## 8. Check yourself

1. **What ends the agent loop?**
   The model returning a response with no tool calls, or a cap firing.

2. **Which of the three caps protects you from cost?**
   `ModelCallLimitMiddleware`, model calls are the billable unit. `recursion_limit` is a backstop, not a budget.

3. **Your agent gave a wrong answer. What is your first move?**
   Open the trace and read what the model received at each step. Not: edit the prompt and re-run.

4. **A run made 14 model calls for a one-lookup question. What are you looking for in the trace?**
   The tool result the model keeps treating as incomplete, the loop is re-calling because the result never reads as done.

5. **Why is tracing introduced in Module 3 rather than at the end?**
   Because every module after this is debugged through it. Learning to build without it teaches guessing.

---

## 9. References

- Agents: https://docs.langchain.com/oss/python/langchain/agents
- Middleware: https://docs.langchain.com/oss/python/langchain/middleware
- LangSmith tracing: https://docs.langchain.com/langsmith/observability
- API reference: https://reference.langchain.com

---

*Next: [Module 4: Structured Output](./langchain-004-structured-output.md). The last piece of the core, and the one that turns a demo into something another system can consume.*
