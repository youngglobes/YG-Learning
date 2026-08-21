# Module 10: LangGraph

**Phase:** Orchestration & Production
**Prerequisites:** Modules 0-9
**Verified against:** `langgraph` 1.2.10, `langchain` 1.3.14, Python 3.12
**Estimated time:** 6-8 hours

---

## 1. Why this matters

`create_agent` has carried you through nine modules. It is the right tool most of the time, and you should keep reaching for it first.

But it has one shape: the model decides what happens next. When *you* need to decide, this step always runs before that one, this branch needs a human signature, this failure retries down a different path, you need explicit control flow.

That is LangGraph. `create_agent` is built on it, so you are not leaving the framework; you are opening the box.

The most valuable thing in this module is not the API. It is knowing **where the line is**, because reaching for LangGraph too early is a common and expensive mistake.

---

## 2. Concepts

### 2.1 Where the line is

Stay with `create_agent` when:

- The model choosing the order is fine or desirable
- Your needs fit a middleware hook (Module 8)
- The flow is: think, use tools, answer

Move to `StateGraph` when:

- **A step must always run**: validation, logging, a compliance check, regardless of model judgment
- **Branching is a business rule**, not a model decision ("refunds over ₹50,000 go to a manager")
- **A human must approve** mid-flow, possibly hours later
- **Different failures need different paths**
- You need cycles you control, rather than the agent's own loop

Honest heuristic: if you are writing the rule in your system prompt and hoping the model follows it, and the cost of it not following is real, the rule belongs in a graph edge instead.

### 2.2 The three primitives

```python
from langgraph.graph import StateGraph, START, END
```

**State**: a `TypedDict` that every node reads and writes:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # the annotation APPENDS
    approved: bool
    amount: int
```

The `Annotated[..., add_messages]` part matters: it is a **reducer**, telling LangGraph how to merge a node's return into existing state. Without it, returning `messages` *replaces* the list; with it, the list is appended to. Replacing your message history by accident is the classic first LangGraph bug.

**Nodes**: plain functions, state in, partial state out:

```python
def check_amount(state: State) -> dict:
    return {"needs_approval": state["amount"] > 50_000}
```

Return only the keys you are changing.

**Edges**: what runs next. Fixed, or conditional:

```python
builder = StateGraph(State)
builder.add_node("check", check_amount)
builder.add_node("auto_approve", auto_approve)
builder.add_node("human_review", human_review)

builder.add_edge(START, "check")
builder.add_conditional_edges(
    "check",
    lambda s: "human_review" if s["needs_approval"] else "auto_approve",
)
builder.add_edge("auto_approve", END)
builder.add_edge("human_review", END)

graph = builder.compile(checkpointer=InMemorySaver())
```

Verified builder methods: `add_node`, `add_edge`, `add_conditional_edges`, `add_sequence`, `compile`.

Note the routing function is ordinary Python. **The branch is deterministic**: that is the entire point of using a graph here.

### 2.3 Human-in-the-loop

Two routes, and you should know both.

**The middleware route**, when you only need approval on tool calls:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=..., tools=[refund, lookup],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={"refund": {"allowed_decisions": ["approve", "reject"]}},
    )],
)
```

Verified parameters: `interrupt_on`, `description_prefix`. Each entry accepts `allowed_decisions`, `description`, `args_schema`, and `when`, a predicate, so you can require approval only above a threshold.

**The graph route**, when the pause is a step in your flow:

```python
from langgraph.types import interrupt, Command

def human_review(state: State) -> dict:
    decision = interrupt({"amount": state["amount"], "reason": "over limit"})
    return {"approved": decision == "approve"}
```

`interrupt()` stops the graph and surfaces its payload. Later, a minute or a week, you resume:

```python
graph.invoke(Command(resume="approve"), config=config)
```

**This only works with a checkpointer**, because the graph must persist to survive the wait. Module 5 was the prerequisite for this. And for a pause measured in hours, `InMemorySaver` is useless, use SQLite or Postgres.

`Command` carries `graph`, `update`, `resume`, `goto`, `PARENT`, so a node can both update state and direct control flow in one return.

### 2.4 Cycles, and stopping them

Graphs can loop, that is how `create_agent` works internally. Anything you build can loop too, and yours has no built-in agent cap.

Always: give loops an explicit exit condition in the routing function, keep a counter in state, and set `recursion_limit` on invoke. Module 3's discipline applies here with more force, because now the loop is yours.

---

## 3. Walkthrough

An expense approval flow: small claims auto-approve, large ones wait for a human.

```python
"""Module 10: deterministic routing with a human approval gate."""
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

APPROVAL_LIMIT = 50_000


class State(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: append, don't replace
    amount: int
    approved: bool
    decided_by: str


def validate(state: State) -> dict:
    """Always runs. This is why we are in a graph and not an agent."""
    if state["amount"] <= 0:
        return {"approved": False, "decided_by": "validation"}
    return {}


def route(state: State) -> str:
    """A business rule, not a model judgment."""
    if state.get("decided_by") == "validation":
        return "record"
    return "human_review" if state["amount"] > APPROVAL_LIMIT else "auto_approve"


def auto_approve(state: State) -> dict:
    return {"approved": True, "decided_by": "policy"}


def human_review(state: State) -> dict:
    """Pause. Resume later with Command(resume=...)."""
    decision = interrupt({
        "question": "Approve this expense?",
        "amount": state["amount"],
        "limit": APPROVAL_LIMIT,
    })
    return {"approved": decision == "approve", "decided_by": "human"}


def record(state: State) -> dict:
    verdict = "APPROVED" if state.get("approved") else "REJECTED"
    return {"messages": [{"role": "assistant",
                          "content": f"{verdict} (by {state.get('decided_by')})"}]}


builder = StateGraph(State)
for name, fn in [("validate", validate), ("auto_approve", auto_approve),
                 ("human_review", human_review), ("record", record)]:
    builder.add_node(name, fn)

builder.add_edge(START, "validate")
builder.add_conditional_edges("validate", route)
builder.add_edge("auto_approve", "record")
builder.add_edge("human_review", "record")
builder.add_edge("record", END)

graph = builder.compile(checkpointer=InMemorySaver())


def run(amount: int, thread: str):
    cfg = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"messages": [], "amount": amount,
                           "approved": False, "decided_by": ""}, config=cfg)

    if "__interrupt__" in result:                     # paused for a human
        payload = result["__interrupt__"][0].value
        print(f"  PAUSED: {payload['question']} (amount {payload['amount']})")
        result = graph.invoke(Command(resume="approve"), config=cfg)

    print(f"  -> {result['messages'][-1].content}")


print("small claim:");  run(1_200,  "exp-1")
print("large claim:");  run(90_000, "exp-2")
print("invalid claim:"); run(-5,    "exp-3")
```

---

## 4. Run it

```bash
.venv/bin/python approval_graph.py
```

**Expected output:**

```
small claim:
  -> APPROVED (by policy)
large claim:
  PAUSED: Approve this expense? (amount 90000)
  -> APPROVED (by human)
invalid claim:
  -> REJECTED (by validation)
```

Three checks. The small claim never paused. The large claim **stopped and waited**, and note the process could have exited between the pause and the resume, because state is checkpointed. And the invalid claim was rejected by `validate`, which ran unconditionally, no model was asked, no prompt was involved, and it cannot be talked out of it.

That last point is the argument for this module. The rule "reject negative amounts" is now impossible to violate.

---

## 5. Exercises

**5.1 Recall.** Give two situations where `create_agent` is the better choice and two where `StateGraph` is.

**5.2 Apply.** Delete `Annotated[list, add_messages]` from the state, making it a plain `list`, and re-run. Explain what happened to the message history and why the reducer exists.

**5.3 Extend.** Add a `manager_review` node for amounts over ₹500,000, so there are two approval tiers. Then add a rejection path where a rejected claim routes back to a `revise` node, and make sure the cycle terminates.

---

## 6. Assignment

An approval-gated workflow for a real process at YoungGlobes (leave requests, purchase orders, content publishing, your choice).

Requirements:

- At least one node that **always** runs, with a written note on why it must not be a model decision
- Deterministic conditional routing on a business rule
- A human approval gate using `interrupt`
- **Persistent** checkpointing, so approval can happen after a process restart, demonstrate this
- A cycle (rejection → revision → resubmission) with a proven termination condition
- A diagram of the graph, and a `WORKFLOW.md` naming which rules are enforced by the graph and which are left to the model

Plus a test that: starts a flow, **kills the process**, restarts it, resumes the interrupt, and asserts the correct final state. If your test resumes in the same process, it is not testing persistence.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Message history replaced instead of appended | Missing `add_messages` reducer | `Annotated[list, add_messages]` |
| `interrupt` does nothing / can't resume | No checkpointer | Compile with one (§2.3) |
| Resume works in tests, fails in production | `InMemorySaver` across processes | SQLite / Postgres |
| `GraphRecursionError` | Cycle with no exit | Exit condition + counter + `recursion_limit` |
| Node output vanishes | Returned full state instead of changed keys | Return only what changed |
| Conditional edge always takes one branch | Routing function returns a non-matching name | Return exact node names |
| Graph is far more complex than the agent was | Reached for LangGraph too early | Re-read §2.1; `create_agent` + middleware may suffice |

---

## 8. Check yourself

1. **When should a rule live in a graph edge rather than the system prompt?**
   When following it is not optional and the cost of the model ignoring it is real. Prompts persuade; edges enforce.

2. **What does `Annotated[list, add_messages]` do?**
   Declares a reducer that appends rather than replaces, so nodes add to history instead of overwriting it.

3. **What does `interrupt` require to be useful?**
   A checkpointer, persistent, if the pause outlives the process.

4. **Your graph is larger and harder to follow than the agent it replaced. What does that suggest?**
   Possibly that you did not need a graph. Check whether `create_agent` plus middleware covers it.

5. **Why is a validation node stronger than a validation instruction?**
   It runs unconditionally as code. An instruction is advice the model can decline.

---

## 9. References

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- Human-in-the-loop: https://docs.langchain.com/oss/python/langgraph/human-in-the-loop
- Persistence: https://docs.langchain.com/oss/python/langgraph/persistence

---

*Next: [Module 11: Multi-Agent](./langchain-011-multi-agent.md), the most fashionable and most frequently unnecessary pattern in the field.*
