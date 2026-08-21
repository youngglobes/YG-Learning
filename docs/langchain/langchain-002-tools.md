# Module 2: Tools

**Phase:** Core
**Prerequisites:** Modules 0-1
**Verified against:** `langchain` 1.3.14, `langchain-core` 1.5.3, Python 3.12
**Estimated time:** 4-5 hours

---

## 1. Why this matters

Tool quality determines agent quality more than prompt wording does.

When an agent "won't use my tool", or calls it with nonsense arguments, or calls it on every turn including "thanks", the cause is almost never the system prompt. It is the tool definition. The model chooses tools by reading their names, descriptions, and parameter schemas. That text *is* the interface, and most people write it as an afterthought.

This module is about writing that interface deliberately.

---

## 2. Concepts

### 2.1 What the model actually sees

Your function:

```python
@tool
def get_weather(city: str, units: str = "c") -> str:
    """Get weather.

    Args:
        city: City name, e.g. Chennai
        units: 'c' or 'f'
    """
    return ...
```

The model never sees the body. It sees a name, a description, and a JSON schema. Which means **the docstring and the type hints are the entire contract.**

### 2.2 `parse_docstring=True`, the flag nobody mentions

Here is the same function decorated two ways. This is measured output, not an illustration:

**Without the flag:**

```python
description: "Get weather.\n\n    Args:\n        city: City name, e.g. Chenna..."
args:        {"city": {"title": "City", "type": "string"},
              "units": {"default": "c", "title": "Units", "type": "string"}}
```

**With `@tool(parse_docstring=True)`:**

```python
description: "Get weather."
args:        {"city":  {"description": "City name, e.g. Chennai",
                        "title": "City", "type": "string"},
              "units": {"description": "'c' or 'f'",
                        "default": "c", "title": "Units", "type": "string"}}
```

Look at what changed. Without the flag, your carefully written parameter docs get **dumped into the description as raw indented text**, and the schema carries **no parameter descriptions at all**. With it, the description is clean and each parameter is documented where the model expects to find it.

If you write Google-style docstrings, and you should, **use `parse_docstring=True`**. Otherwise you did the work and threw away the benefit.

### 2.3 Describe *when* to call, not only *what* it does

This is the highest-leverage sentence in the module.

```python
# Weak, describes the function
"""Search the policy database."""

# Strong, describes the decision
"""Search YoungGlobes HR policy documents.

Call this whenever the user asks about company policy, leave, expenses,
or internal process. Do not call it for greetings or small talk.
"""
```

The model is not deciding *what your tool does*. It is deciding *whether this is the moment to call it*. Give it the information it needs for the decision it is actually making. Under-calling and over-calling are both usually fixed here rather than in the system prompt.

### 2.4 Errors the model can recover from

By default a raised `ToolException` propagates and kills the run, verified:

```python
@tool
def boom(x: str) -> str:
    """Always fails."""
    raise ToolException(f"cannot process {x}")
# -> ToolException: cannot process q
```

That is rarely what you want. An agent that receives *"Error: city 'Xyz' not found. Valid cities: Chennai, Mumbai, Delhi."* can correct itself and retry. An agent that receives a stack trace cannot.

Two options:

```python
# 1. Return the error as a normal string (simplest, works everywhere)
@tool
def get_weather(city: str) -> str:
    """..."""
    if city not in CITIES:
        return f"Error: unknown city '{city}'. Known cities: {', '.join(CITIES)}."
    return ...

# 2. Let the framework handle it
@tool(handle_tool_error=True)
def get_weather(city: str) -> str: ...
```

Write error messages **for the model as the reader**. "Invalid input" tells it nothing. "Expected ISO date like 2026-08-12, got 'next Tuesday'" tells it exactly how to fix the call.

### 2.5 Tool surface design

Three rules that matter more as the toolset grows:

**Few and well-bounded beats many and overlapping.** Two tools whose descriptions could both plausibly answer a request produce coin-flip behaviour. If you cannot state in one sentence which tool handles a case, neither can the model.

**Expressive parameters beat prose.** An enum (`Literal["celsius", "fahrenheit"]`) carries the constraint into the schema, where it is enforced. The same constraint in the description is a suggestion.

**Least privilege.** Every tool you add expands what a compromised agent can do. Module 7 makes this concrete; start the habit now by asking of each tool: *if the model called this with the worst possible arguments, what happens?*

---

## 3. Walkthrough

```python
"""Module 2: a small, well-specified toolset."""
from dotenv import load_dotenv
load_dotenv()
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

from typing import Literal
from langchain.agents import create_agent
from langchain.tools import tool

CITIES = {"chennai": 32, "mumbai": 29, "delhi": 38}


@tool(parse_docstring=True)
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current temperature for an Indian city.

    Call this when the user asks about weather, temperature, or how hot
    somewhere is. Do not call it for general conversation.

    Args:
        city: City name, e.g. Chennai. Case-insensitive.
        units: Temperature unit to report in.
    """
    temp_c = CITIES.get(city.lower())
    if temp_c is None:
        # Written for the model to act on, not for a human log reader.
        return (
            f"Error: no weather data for '{city}'. "
            f"Available cities: {', '.join(sorted(CITIES))}."
        )
    if units == "fahrenheit":
        return f"{temp_c * 9 / 5 + 32:.0f}F in {city.title()}"
    return f"{temp_c}C in {city.title()}"


@tool(parse_docstring=True)
def convert_currency(amount: float, frm: str, to: str) -> str:
    """Convert an amount between two currencies at a fixed demo rate.

    Call this only when the user explicitly asks for a currency conversion.

    Args:
        amount: The numeric amount to convert.
        frm: Source currency code, e.g. USD.
        to: Target currency code, e.g. INR.
    """
    rates = {("USD", "INR"): 83.0, ("INR", "USD"): 1 / 83.0}
    rate = rates.get((frm.upper(), to.upper()))
    if rate is None:
        return f"Error: no rate for {frm}->{to}. Supported: USD<->INR."
    return f"{amount} {frm.upper()} = {amount * rate:.2f} {to.upper()}"


agent = create_agent(
    model=MODEL,
    tools=[get_weather, convert_currency],
    system_prompt="You are a concise assistant.",
)

for q in [
    "How hot is Chennai in fahrenheit?",
    "What's the weather in Paris?",       # unknown city -> tool returns an error
    "Convert 100 USD to INR",
    "Thanks, that's helpful!",            # should call nothing
]:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    calls = sum(1 for m in result["messages"] if getattr(m, "tool_calls", None))
    print(f"\nQ: {q}\nA: {result['messages'][-1].text}\n   (tool-calling turns: {calls})")
```

---

## 4. Run it

> **On a small local model, some checks below will fail, and that is expected.**
> The behavioural checks in this section depend on model capability. See the
> capability tier table in [Choosing your model](./model-setup.md) before
> concluding your code is wrong.

```bash
.venv/bin/python tools_demo.py
```

**Expected output, illustrative.** What matters is the four behaviours, not the wording:

1. **Chennai in fahrenheit** → calls `get_weather` with `units="fahrenheit"`. The enum did its job.
2. **Paris** → calls the tool, gets your error string back, and *tells the user which cities are available* rather than crashing or inventing a temperature. This is recoverable-error handling working.
3. **Convert 100 USD** → calls `convert_currency`, not `get_weather`.
4. **"Thanks"** → `tool-calling turns: 0`. It called nothing.

Number 4 is the one to watch. If it calls a tool on "thanks", your descriptions are not scoping the decision.

---

## 5. Exercises

**5.1 Recall.** Without looking: what three things does the model see about your tool, and which of them does `parse_docstring=True` change?

**5.2 Apply.** Replace `get_weather`'s docstring with `"""Gets stuff."""` and re-run all four questions. Record what breaks. Then restore it and write two sentences on what the docstring is actually for.

**5.3 Extend.** Add a third tool that deliberately overlaps, `lookup_temperature(location: str)`, and observe the agent choosing inconsistently between it and `get_weather`. Then fix it, either by merging the two or by making each description state clearly what the other handles. Write down which fix you chose and why.

---

## 6. Assignment

Build a three-tool agent for a domain of your choice (not weather). Requirements:

- All three use `parse_docstring=True` with documented parameters
- At least one parameter is constrained by `Literal` or an enum rather than prose
- Every tool returns a **recoverable error string** for bad input, no raised exceptions reaching the user
- A `TOOLS.md` documenting, for each tool: when it should be called, when it should *not*, and what a compromised agent could do with it

Then a short test file asserting:
- The right tool is chosen for three unambiguous requests
- **No tool is called** for a conversational message
- A bad-input call produces a helpful message rather than a traceback

The "when it should not be called" column of `TOOLS.md` is the part people skip and the part that fixes over-calling.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Parameter descriptions missing from the schema | No `parse_docstring=True` | Add it, see §2.2 |
| Description contains a mangled `Args:` block | Same | Same |
| Agent never calls the tool | Description says what it does, not when to use it | Rewrite for the decision (§2.3) |
| Agent calls the tool on "hello" | No negative case in the description | Add "Do not call this for…" |
| Agent picks the wrong one of two tools | Overlapping descriptions | Merge them, or state the boundary in both |
| Garbage arguments | Untyped or undocumented parameters | Type hints + `Literal` + parameter docs |
| Run dies on bad input | `ToolException` propagates by default | Return an error string, or `handle_tool_error=True` |
| Agent retries the same failing call | Error message doesn't say how to fix it | Include valid options in the error text |

---

## 8. Check yourself

1. **What does the model see of your tool?**
   Name, description, and JSON schema. Never the body.

2. **You wrote Google-style docstrings with an `Args:` section and parameters still have no descriptions in the schema. Why?**
   `parse_docstring=True` is missing. Without it the `Args:` block is dumped into the description as raw text and the schema gets nothing.

3. **Your agent calls a tool on every turn, including "thanks". Where do you fix it?**
   The tool description, add the negative case. The system prompt is the second resort, not the first.

4. **Why return an error string rather than raise?**
   A returned string goes back to the model, which can correct itself and retry. A raised exception ends the run.

5. **Two tools could each plausibly serve a request. What happens, and what is the fix?**
   The model chooses inconsistently. Merge them, or make each description state explicitly what the other covers.

---

## 9. References

- Tools: https://docs.langchain.com/oss/python/langchain/tools
- Agents: https://docs.langchain.com/oss/python/langchain/agents
- API reference: https://reference.langchain.com

---

*Next: [Module 3: Agents & Tracing](./langchain-003-agents-and-tracing.md). You have been using `create_agent` for three modules; now you will look inside the loop.*
