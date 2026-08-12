# Module 1 — Models & Messages

**Phase:** Core
**Prerequisites:** Module 0
**Verified against:** `langchain` 1.3.14, `langchain-core` 1.5.3, Python 3.12
**Estimated time:** 3–4 hours

---

## 1. Why this matters

Everything in LangChain is a list of messages. The agent loop appends to it, tools append to it, retrieval appends to it. When something goes wrong in Module 7 — retrieved context that "disappears", a tool result the model ignores — the debugging move is always the same: print the message list and look at what the model actually received.

People who skip this module debug by changing prompt wording and hoping. People who do it debug by reading.

---

## 2. Concepts

### 2.1 `init_chat_model`

One function, any provider:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-opus-5")
model = init_chat_model("openai:gpt-5.5")
model = init_chat_model("ollama:llama3.2")          # local, no key
```

The `provider:model` string keeps provider choice as configuration rather than code. Its signature is `(model, model_provider, configurable_fields, config_prefix, **kwargs)` — `kwargs` is where provider-specific options like `max_tokens` go.

### 2.2 The four message types

| Type | Who writes it | Purpose |
|---|---|---|
| `SystemMessage` | You | Standing instructions. Trusted. |
| `HumanMessage` | The user | The request. **Untrusted** — see Module 7. |
| `AIMessage` | The model | Its reply, and any tool calls it wants |
| `ToolMessage` | Your code | The result of running a tool |

A tool-calling turn produces four messages, in this order: `System`, `Human`, `AI` (containing a tool call), `Tool` (the result), `AI` (the final answer). That is the shape you printed in Module 0's Exercise 5.2.

### 2.3 `.content` vs `.content_blocks` vs `.text`

This trips up nearly everyone migrating from older tutorials, because the answer changed in v1.

```python
>>> from langchain.messages import AIMessage
>>> m = AIMessage(content="hello")
>>> m.content_blocks
[{'type': 'text', 'text': 'hello'}]
>>> m.text
'hello'
```

- **`.content_blocks`** — the structured, provider-neutral representation. A list of typed blocks. This is what you inspect when a message might hold more than plain text: reasoning, images, tool calls, citations.
- **`.text`** — the plain string, for when you just want to print the answer.
- **`.content`** — the raw underlying value. Provider-shaped and not guaranteed stable. Avoid it in new code.

Rule: **`.text` to display, `.content_blocks` to inspect.** Old tutorials reach for `.content` because blocks did not exist; that is how you date a tutorial in one line.

### 2.4 Tokens, context windows, and money

- **Tokens** are the model's unit of text. Roughly ¾ of a word in English; far worse for code and non-English scripts.
- **Context window** caps input + output for a single call. Exceeding it is an error, not a silent truncation.
- **Pricing** is per million tokens, and **input and output are priced differently** — output is typically 3–5× input.

Two consequences worth internalising now:

**In an agent loop the whole history is re-sent on every step.** A ten-step loop does not send your prompt once — it re-sends a growing transcript ten times. Cost grows quadratically with loop length, not linearly. This is the single most surprising cost fact in agent work.

**Do not estimate tokens with `tiktoken` for a non-OpenAI model.** It is OpenAI's tokenizer. Against Claude it undercounts materially, and worse on code. Use the provider's own token counting endpoint.

---

## 3. Walkthrough

```python
"""Module 1 — anatomy of a message list."""
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

model = init_chat_model("anthropic:claude-opus-5")

messages = [
    SystemMessage("You are a terse assistant. One sentence maximum."),
    HumanMessage("Why is the sky blue?"),
]

response = model.invoke(messages)

print("type:          ", type(response).__name__)
print("text:          ", response.text)
print("content_blocks:", response.content_blocks)
print("usage:         ", response.usage_metadata)
```

`usage_metadata` is where the token counts live — input, output, total. Look at it on every call while you are learning. It is the only honest feedback loop on cost.

### 3.1 Watching history grow

The point about re-sent history is easier to believe once you see it:

```python
conversation = [SystemMessage("You are a helpful assistant.")]

for question in ["What is 2+2?", "And times 10?", "And minus 7?"]:
    conversation.append(HumanMessage(question))
    reply = model.invoke(conversation)
    conversation.append(reply)
    print(f"{question:20} input_tokens={reply.usage_metadata['input_tokens']}")
```

`input_tokens` climbs on every turn even though each question is the same length. You are paying for the transcript, not the question.

---

## 4. Run it

```bash
.venv/bin/python messages_anatomy.py
```

**Expected output — illustrative:**

```
type:           AIMessage
text:           Sunlight scatters off air molecules, and blue scatters most.
content_blocks: [{'type': 'text', 'text': 'Sunlight scatters off air molecules, ...'}]
usage:          {'input_tokens': 24, 'output_tokens': 15, 'total_tokens': 39}
```

Three structural checks: the type is `AIMessage`; `content_blocks` is a **list of dicts** with a `type` key, not a bare string; and `usage_metadata` reports non-zero counts. In the second script, `input_tokens` must **increase** each turn.

---

## 5. Exercises

**5.1 Recall.** Name the four message types and who produces each. Which one is untrusted, and why does that matter later?

**5.2 Apply.** Run the same prompt against two providers (use `ollama:llama3.2` if you have no second key). Compare `.text`, `.content_blocks`, and token counts. Write two sentences on what transferred unchanged and what did not.

**5.3 Extend.** Build a `cost_estimate.py` that takes a file path and a model name, counts tokens with the **provider's** counter, and prints estimated input cost. Then run it on a 50-page PDF and on 50 pages of source code. Explain the difference in tokens-per-page.

---

## 6. Assignment

A CLI: `python cost.py <file> --model <name>`

Requirements:

- Real token counts from the provider, not a character-count heuristic and not `tiktoken`
- Separate input and output pricing, with the rates in a config dict, not hardcoded in a formula
- Estimates the cost of a **10-step agent loop** over the same input, accounting for re-sent history
- README stating where you got the prices and the date you checked

That last requirement is the point of the assignment. Model prices change. An estimator with undated hardcoded prices is a tool that quietly becomes a liar.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'AIMessage' object has no attribute 'content_blocks'` | `langchain-core` < 1.0 | `pip install -U langchain-core` |
| You get an object where you expected a string | Read `.content` instead of `.text` | Use `.text` to display |
| Cost estimate 20%+ off for Claude | `tiktoken` — wrong tokenizer | Use the provider's token counter |
| `usage_metadata` is `None` | Some providers omit it | Check provider docs; do not assume it is free |
| Costs far higher than expected in a loop | History re-sent each step | Expected — see §2.4. Cap iterations, trim history |
| Context-window error on a long document | Input exceeds the window | Chunk it (Module 6), or use a larger-window model |

---

## 8. Check yourself

1. **What is in `result["messages"]` after a tool-calling turn?**
   Five messages: System, Human, AI (with the tool call), Tool (the result), AI (the final answer).

2. **`.text` or `.content_blocks` — which for displaying an answer, which for inspecting a response that may contain images or tool calls?**
   `.text` to display; `.content_blocks` to inspect.

3. **Why does a 10-step agent loop cost far more than 10 single calls?**
   The full transcript is re-sent on every step, so input tokens grow with each turn. Cost scales with the square of loop length, not linearly.

4. **Why is `tiktoken` the wrong tool for estimating Claude costs?**
   It is OpenAI's tokenizer. Different models tokenize differently, and the error is large on code and non-English text.

5. **A tutorial reads `message.content` and indexes into it. What does that tell you?**
   It predates v1. Check it against Appendix A of the syllabus before trusting anything else in it.

---

## 9. References

- Models — https://docs.langchain.com/oss/python/langchain/models
- Messages and content blocks — https://docs.langchain.com/oss/python/langchain/messages
- API reference — https://reference.langchain.com

---

*Next: [Module 2 — Tools](./langchain-002-tools.md). You will find out why a docstring is an interface, not a comment.*
