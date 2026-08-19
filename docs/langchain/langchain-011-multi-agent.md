# Module 11 — Multi-Agent

**Phase:** Orchestration & Production
**Prerequisites:** Modules 0–10
**Verified against:** `langgraph` 1.2.10, `langchain` 1.3.14, Python 3.12
**Estimated time:** 5–6 hours

---

## 1. Why this matters

Multi-agent architectures are the most fashionable pattern in the field and, in commercial work, frequently the wrong answer.

This module teaches the pattern properly — you will meet systems built this way and you need to work on them. It also teaches the discipline to avoid it, which is the part conference talks leave out.

The honest framing: a multi-agent system is a distributed system where the components are non-deterministic and expensive. Every property that makes distributed systems hard applies, plus some new ones.

---

## 2. Concepts

### 2.1 The cost, stated plainly

Every delegation costs a full context re-establishment. The specialist does not share the coordinator's conversation — it must be briefed, it re-reads what it needs, it works, it reports back, and then the coordinator re-reads the report.

Concretely, one delegation is roughly: coordinator decides (model call) → specialist briefed (model call, full prompt) → specialist works (n model calls) → specialist reports → coordinator reads and integrates (model call). Four-plus model calls where a single agent with the right tool would have made one.

Latency compounds the same way, and it is serial unless you deliberately parallelise.

**So the bar is: does splitting buy something worth 3–5× the cost?** Sometimes yes. Often the honest answer is that one agent with better tools would do it.

### 2.2 When it genuinely helps

- **Independent parallel work.** Five documents to analyse with no interdependence — fan out, run concurrently, wall-clock time drops even as token cost rises.
- **Context isolation.** A subtask requiring 50 pages of reading, where you want the findings but not the 50 pages, in the coordinator's window.
- **Genuinely different tool sets or permissions.** A read-only researcher and a write-capable publisher should not be one agent holding both capabilities — this is the least-privilege argument from Module 7, expressed structurally.
- **Different models per role.** A cheap model for bulk extraction, an expensive one for synthesis.

### 2.3 When it does not

- **Sequential steps with no isolation benefit.** That is a graph (Module 10), and a graph is cheaper and deterministic.
- **"Separation of concerns."** Real for code, not a reason to split runtime agents. Prompts are not modules.
- **One task, decomposed small.** Delegating a job the coordinator could finish in three tool calls is pure overhead.
- **Because the architecture diagram looks impressive.** Named honestly because it is a real driver.

### 2.4 The supervisor pattern

The common shape. A coordinator receives the request, routes to a specialist, collects the result, and decides whether to continue.

The clean way to express this in LangChain is to make **each specialist a tool** on the coordinator:

```python
from langchain.agents import create_agent
from langchain.tools import tool

researcher = create_agent(model="anthropic:claude-opus-5", tools=[search],
                          system_prompt="Research thoroughly. Report findings with sources.")

@tool(parse_docstring=True)
def delegate_research(question: str) -> str:
    """Hand a self-contained research question to the research specialist.

    Use only for questions needing multiple sources. Do not use for
    anything you can answer directly or in one tool call.

    Args:
        question: A complete, standalone question. The specialist sees
            none of this conversation.
    """
    result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text
```

Two things this buys. Delegation reuses everything from Module 2 — the description controls *when* the coordinator delegates, and the "do not use for…" line is what stops over-delegation. And the specialist's internal turns stay out of the coordinator's context; only the report comes back.

**The docstring's "sees none of this conversation" is load-bearing.** It is the most common multi-agent bug, and §2.5 is about it.

### 2.5 Subagents do not share context

The coordinator has been talking to the user for ten turns. It delegates "check the second one." The specialist has no idea what "the second one" is.

Every delegated task must be **self-contained**: the entity, the constraints, the file paths, the output format. If it needs context, put the context in the message.

Ways this bites:
- Pronouns and references ("it", "that file", "the earlier result")
- Assumed shared state that only exists in the coordinator's history
- Output format assumed rather than specified, so the report comes back unusable

The fix is prompt discipline on the coordinator: *"Each delegated task must be complete and standalone. Include all identifiers, paths, and constraints — the specialist cannot see this conversation."*

### 2.6 Handoff vs. delegation

**Delegation** (above): the specialist reports back and the coordinator keeps control. Predictable; the coordinator can verify.

**Handoff:** control transfers and does not return — a triage agent passes a conversation to a billing agent permanently. Implemented in LangGraph with `Command(goto=...)`, which carries `graph`, `update`, `resume`, `goto`, `PARENT`.

Prefer delegation unless the conversation genuinely belongs to the other agent from then on. Handoffs make failure harder to reason about because there is no longer anyone supervising.

### 2.7 Failure modes you must design for

- **Delegation loops.** A hands to B, B hands back. Cap delegations explicitly; keep a depth counter in state.
- **Lost errors.** A specialist fails and reports prose the coordinator reads as success. Return structured results (Module 4) with an explicit status field.
- **Cost explosion.** No cap means a bad decision can spawn agents until something breaks. `ModelCallLimitMiddleware` on every agent, coordinator included.
- **Untraceable behaviour.** Without tracing, a multi-agent bug is close to undebuggable. Module 3 was not optional.

---

## 3. Walkthrough

A support triage system: classify, route to a specialist, escalate when unsure.

```python
"""Module 11 — supervisor with specialists as tools, and a cost baseline."""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool

# ---- specialists: narrow prompts, narrow tools --------------------------
billing = create_agent(
    model="anthropic:claude-opus-5",
    tools=[],
    system_prompt=("You handle billing questions: invoices, refunds, payment "
                   "failures. Answer only from what you are told. If the task "
                   "lacks an invoice number, say exactly what is missing."),
    middleware=[ModelCallLimitMiddleware(run_limit=4)],
)

technical = create_agent(
    model="anthropic:claude-opus-5",
    tools=[],
    system_prompt=("You handle technical support: errors, integrations, "
                   "configuration. Give concrete steps. If you cannot solve "
                   "it from the information given, say so plainly."),
    middleware=[ModelCallLimitMiddleware(run_limit=4)],
)

CALLS = {"billing": 0, "technical": 0}


@tool(parse_docstring=True)
def ask_billing(task: str) -> str:
    """Delegate a billing question to the billing specialist.

    Use for invoices, refunds, charges, and payment failures. Do not use
    for technical faults.

    Args:
        task: A complete, standalone question including the customer name,
            invoice number, and amounts. The specialist cannot see this
            conversation.
    """
    CALLS["billing"] += 1
    return billing.invoke({"messages": [{"role": "user", "content": task}]})["messages"][-1].text


@tool(parse_docstring=True)
def ask_technical(task: str) -> str:
    """Delegate a technical question to the technical specialist.

    Use for errors, integrations, and configuration. Do not use for
    billing questions.

    Args:
        task: A complete, standalone description including product,
            version, and the exact error. The specialist cannot see this
            conversation.
    """
    CALLS["technical"] += 1
    return technical.invoke({"messages": [{"role": "user", "content": task}]})["messages"][-1].text


supervisor = create_agent(
    model="anthropic:claude-opus-5",
    tools=[ask_billing, ask_technical],
    system_prompt=(
        "You triage customer support tickets.\n\n"
        "Delegate to exactly one specialist when the ticket clearly belongs "
        "to them. Answer directly if it is a simple question you can handle "
        "in one step — delegation is expensive.\n\n"
        "Every delegated task MUST be self-contained: include the customer "
        "name, all identifiers, amounts, and error text. The specialist "
        "cannot see this conversation.\n\n"
        "If the ticket is ambiguous or spans both areas, do not guess — "
        "escalate to a human and say why."
    ),
    middleware=[ModelCallLimitMiddleware(run_limit=8)],
)

TICKETS = [
    "Invoice INV-2291 charged me twice, 24000 INR. I'm Priya at Acme.",
    "Getting ERR_TIMEOUT on the v3 API since yesterday. Rahul, Beta Ltd.",
    "Hi, what are your office hours?",                    # answer directly
    "I was double charged AND the API is down.",          # ambiguous -> escalate
]

for t in TICKETS:
    before = dict(CALLS)
    out = supervisor.invoke({"messages": [{"role": "user", "content": t}]})
    used = {k: CALLS[k] - before[k] for k in CALLS if CALLS[k] != before[k]}
    print(f"\nTICKET: {t[:52]}...")
    print(f"  delegated to: {used or 'nobody (handled directly)'}")
    print(f"  reply: {out['messages'][-1].text[:120]}")
```

---

## 4. Run it

> **On a small local model, some checks below will fail — and that is expected.**
> The behavioural checks in this section depend on model capability. See the
> capability tier table in [Choosing your model](./model-setup.md) before
> concluding your code is wrong.

```bash
.venv/bin/python triage.py
```

**Expected output — illustrative.** Four behaviours, and two of them are about *restraint*:

1. **Invoice ticket** → `{'billing': 1}`. Delegated once, and the task passed along includes the invoice number and amount.
2. **API error ticket** → `{'technical': 1}`.
3. **Office hours** → `nobody (handled directly)`. The supervisor did not delegate a trivial question.
4. **Ambiguous ticket** → escalated, rather than guessing a specialist.

Cases 3 and 4 are the ones to watch. A system that delegates everything has not learned anything except how to spend money.

---

## 5. Exercises

**5.1 Recall.** Roughly how many model calls does one delegation cost versus one agent with the right tool? Name two situations where that is worth paying.

**5.2 Apply.** Remove the "cannot see this conversation" line from both docstrings and the supervisor prompt. Send a two-turn conversation where the second turn says "check that one again." Record what the specialist receives and what it does. This is §2.5, first-hand.

**5.3 Extend.** Instrument total token cost per ticket. Then build a single-agent version with both specialties as tools, run the same four tickets, and produce a cost comparison table. State which you would ship.

---

## 6. Assignment

A support triage system, plus the evidence for whether it should exist.

Requirements:

- A supervisor and at least two specialists with genuinely different prompts and tool sets
- Delegation tools whose docstrings state when **not** to delegate
- Explicit escalation to a human on ambiguity — no guessing
- Structured results from specialists (Module 4) with an explicit success/failure field, so a failure cannot be read as success
- Call limits on **every** agent including the supervisor
- Full tracing

And the graded deliverable, a `COST.md` containing:

- Total tokens and model calls per ticket for the multi-agent version
- The same for a **single-agent baseline** with the same capabilities as tools
- The ratio
- A recommendation: **is the multi-agent version worth it for this workload?** "No" is a perfectly good answer if the numbers say so, and is worth more marks than an unsupported "yes"

Building the baseline is not optional. Without it you have no evidence, only architecture.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Specialist misunderstands the task | Task not self-contained (§2.5) | Include all identifiers and context in the delegated message |
| Costs 5× the single-agent version | Expected — that is the pattern | Justify it or drop back to one agent |
| Agents delegate back and forth | No depth cap | Depth counter in state; call limits everywhere |
| Failures reported as successes | Specialist returns prose | Structured output with a status field |
| Supervisor delegates trivial questions | No "do not delegate" guidance | Add it to the docstring and prompt |
| Impossible to debug | No tracing | Module 3 |
| Ambiguous tickets silently mis-routed | No escalation path | Make escalation explicit and rewarded |

---

## 8. Check yourself

1. **What does one delegation actually cost?**
   Roughly 4+ model calls — decide, brief, work, report, integrate — plus serial latency, against one call for a single agent with the right tool.

2. **Three situations where multi-agent genuinely earns its cost?**
   Independent parallel work; context isolation for reading-heavy subtasks; genuinely different tool sets or permissions (or different models per role).

3. **Why is "separation of concerns" a weak justification?**
   It is a code-organisation principle. Runtime agents are not modules, and splitting them costs real tokens and latency.

4. **The most common multi-agent bug?**
   Assuming shared context. Subagents see only the message they are sent.

5. **Delegation or handoff by default?**
   Delegation. Control returns to the coordinator, which can verify the result and recover from failure.

---

## 9. References

- Multi-agent — https://docs.langchain.com/oss/python/langchain/multi-agent
- LangGraph — https://docs.langchain.com/oss/python/langgraph/overview
- API reference — https://reference.langchain.com

---

*Next: [Module 12 — Production](./langchain-012-production.md), and the capstone.*
