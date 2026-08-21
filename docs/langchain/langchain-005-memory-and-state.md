# Module 5: Memory & State

**Phase:** Retrieval & Memory
**Prerequisites:** Modules 0-4
**Verified against:** `langchain` 1.3.14, `langgraph` 1.2.10, Python 3.12
**Estimated time:** 4-5 hours

---

## 1. Why this matters

Every agent you have written so far has amnesia. Ask it your name, then ask what your name is, and it does not know, each `invoke()` starts from nothing.

This module fixes that. It is also the module where the internet will most aggressively mislead you, because "memory in LangChain" meant something completely different before v1, and the old names are everywhere.

---

## 2. Concepts

### 2.1 The old memory classes are gone

If a tutorial imports any of these, it predates v1 and its code will not run:

| Pre-v1 | What replaced it |
|---|---|
| `ConversationBufferMemory` | Checkpointers |
| `ConversationBufferWindowMemory` | Checkpointers + `SummarizationMiddleware` |
| `ConversationSummaryMemory` | `SummarizationMiddleware` |
| `ConversationChain` | `create_agent` + a checkpointer |

The mental model changed too. Memory used to be an object you attached to a chain. Now **state is persisted automatically and keyed by conversation**, which is both simpler and closer to how you would build it yourself.

### 2.2 Checkpointers and threads

Two ideas, and you need both:

- A **checkpointer** saves the message list after every step.
- A **thread_id** says *which* conversation this is.

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model=..., tools=[...], checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "user-42"}}
agent.invoke({"messages": [{"role": "user", "content": "My name is Priya"}]}, config=config)
agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config)
# -> knows it is Priya
```

You pass **only the new message**. The checkpointer reloads the history and appends. You are not managing the list any more.

Here is the behaviour verified against a fake model, so the numbers are real:

```
thread A: "my name is Priya"  then  "what is my name?"   -> 4 messages, remembers
thread B: "what is my name?"                             -> 2 messages, isolated
```

Thread A accumulated the earlier turn. Thread B, a different `thread_id` against the same agent object, saw none of it.

### 2.3 The thread_id is a security boundary

**Forget the `thread_id` and every user shares one conversation.** User B's next question arrives with User A's history attached, their name, their documents, whatever they said.

This is the most consequential one-line bug in the whole path. Treat `thread_id` as an authorisation decision, not a convenience:

- Derive it from an authenticated session, never from user-supplied input
- Never let a client choose its own `thread_id` (they will send someone else's)
- Include a tenant identifier in multi-tenant systems

### 2.4 In-memory vs. persistent

`InMemorySaver` dies with the process. It is for development.

For persistence you need a separate package, it is **not** bundled:

```bash
pip install langgraph-checkpoint-sqlite     # or langgraph-checkpoint-postgres
```

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    agent = create_agent(model=..., tools=[...], checkpointer=checkpointer)
```

SQLite for a single process; Postgres for anything with more than one worker.

### 2.5 Short-term vs. long-term

Two different problems that the word "memory" blurs together:

| | Short-term | Long-term |
|---|---|---|
| Scope | Within one thread | Across all threads for a user |
| Holds | The conversation | Durable facts and preferences |
| Mechanism | `checkpointer` | `store` |
| Example | "as I said above…" | "always reply in Tamil" |

`create_agent` accepts both. Most systems need the checkpointer long before they need the store, do not reach for long-term memory until you have a concrete fact that must outlive a conversation.

### 2.6 Conversations outgrow the context window

History grows every turn, and Module 1 §2.4 already showed you it is re-sent every step. Two failure modes follow: cost climbs, then the context window overflows.

`SummarizationMiddleware` compacts old turns automatically:

```python
from langchain.agents.middleware import SummarizationMiddleware
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

agent = create_agent(
    model=MODEL,
    tools=[...],
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model=MODEL,
        trigger={"tokens": 4000},   # compact once history passes this
        keep={"messages": 6},       # keep the most recent turns verbatim
    )],
)
```

Its full parameter list is `model, trigger, keep, token_counter, summary_prompt, trim_tokens_to_summarize`.

Summarisation is lossy by design. It is a trade, you exchange detail for the ability to continue. Set `keep` so that recent turns, which usually carry the actual task, survive intact.

### 2.7 What you persist is data you now hold

A checkpointer writes conversations to disk. Those conversations contain whatever your users typed, names, salaries, medical details, complaints about colleagues.

Before enabling persistence on anything real, answer three questions: how long do you keep it, who can read the database, and what happens when someone asks to be deleted. "The framework does it automatically" is not an answer to any of them.

---

## 3. Walkthrough

```python
"""Module 5: a multi-turn assistant with per-user threads."""
from dotenv import load_dotenv
load_dotenv()
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=MODEL,
    tools=[],
    system_prompt="You are a concise assistant with a good memory.",
    checkpointer=InMemorySaver(),
    middleware=[ModelCallLimitMiddleware(run_limit=4)],
)


def say(text: str, thread_id: str) -> str:
    """Send one message on a given thread. History is reloaded automatically."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].text


print("--- thread: priya ---")
print(say("Hi, I'm Priya and I work on the Drupal team.", "priya"))
print(say("What team am I on?", "priya"))          # must recall

print("\n--- thread: rahul (different user) ---")
print(say("What team am I on?", "rahul"))          # must NOT know about Priya
```

---

## 4. Run it

```bash
.venv/bin/python memory_demo.py
```

**Expected output, illustrative:**

```
--- thread: priya ---
Nice to meet you, Priya.
You're on the Drupal team.

--- thread: rahul (different user) ---
I don't have that information, you haven't told me which team you're on.
```

Two checks, and the second is the important one. Thread `priya` **recalls** across separate `invoke()` calls. Thread `rahul` **does not know Priya's name or team**. If Rahul's answer mentions Drupal or Priya, your threading is broken and you have just reproduced the §2.3 bug in miniature.

Then restart the process and re-run only the `rahul` thread. With `InMemorySaver` everything is gone, that is the §2.4 lesson, and it is the reason for the assignment below.

---

## 5. Exercises

**5.1 Recall.** What are the two ingredients of conversation memory in v1, and what happens if you supply the first without the second?

**5.2 Apply.** Remove `config=` from the `say()` call entirely and re-run. Record the error or the behaviour. Then hardcode one `thread_id` for both users and observe the leak first-hand. Write two sentences on why this is a security bug and not a UX bug.

**5.3 Extend.** Add `SummarizationMiddleware` with a low trigger (`{"tokens": 500}`), hold a fifteen-turn conversation, and print `usage_metadata['input_tokens']` each turn. Plot or tabulate it. Identify the turn where compaction fires, and say what was lost.

---

## 6. Assignment

A CLI assistant that survives a restart.

Requirements:

- `SqliteSaver` persistence (`pip install langgraph-checkpoint-sqlite`)
- `python chat.py --user priya` resumes that user's conversation after the process is killed
- Two different `--user` values are provably isolated
- `SummarizationMiddleware` configured, with a written justification for your `trigger` and `keep` values
- A model call limit on the agent
- A `PRIVACY.md` stating what the database holds, where it lives, your retention period, and how you would delete one user's data

Plus a test that:
1. Writes a fact on thread A
2. **Creates a brand-new agent object** (simulating a restart)
3. Asserts thread A recalls the fact and thread B does not

Step 2 is the point. An in-process test that reuses the same object passes even with no persistence at all.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Agent forgets everything between calls | No checkpointer | Pass `checkpointer=...` |
| Checkpointer set, still forgets | No `thread_id` in config | `config={"configurable": {"thread_id": ...}}` |
| **Users see each other's conversations** | Shared or missing `thread_id` | Derive it from the authenticated session (§2.3) |
| History gone after restart | `InMemorySaver` | Use `SqliteSaver` / Postgres |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | Not bundled | `pip install langgraph-checkpoint-sqlite` |
| Cost climbs every turn | History re-sent each step | Expected; add `SummarizationMiddleware` |
| Context-window error deep in a conversation | History outgrew the window | Summarisation, or a larger-window model |
| Agent forgets a detail from ten turns ago | Summarisation compacted it | Raise `keep`, or store the fact long-term |
| Tutorial imports `ConversationBufferMemory` | Pre-v1 | §2.1 |

---

## 8. Check yourself

1. **Two ingredients of conversation memory?**
   A checkpointer to save state, and a `thread_id` to say which conversation it belongs to.

2. **Why is `thread_id` a security boundary?**
   It decides whose history loads. Shared or client-supplied, one user gets another's conversation.

3. **`InMemorySaver` in production, what breaks?**
   Every conversation is lost on restart or deploy, and nothing is shared between workers.

4. **Difference between a checkpointer and a store?**
   The checkpointer holds one conversation's messages; the store holds durable facts across all of a user's conversations.

5. **What does summarisation cost you?**
   Detail. It is lossy compression, which is why `keep` exists to protect recent turns.

---

## 9. References

- Memory and persistence: https://docs.langchain.com/oss/python/langchain/memory
- Middleware: https://docs.langchain.com/oss/python/langchain/middleware
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence

---

*Next: [Module 6: Retrieval Foundations](./langchain-006-retrieval-foundations.md), where the agent gains access to documents it was never trained on.*
