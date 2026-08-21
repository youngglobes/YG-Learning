# Module 0: Setup & Mental Model

**Phase:** Core
**Prerequisites:** AI Foundations (Python, LLM fundamentals)
**Verified against:** `langchain` 1.3.14, Python 3.12
**Estimated time:** 1-2 hours
**Before this:** [Choosing your model](./model-setup.md)

---

## 1. Why this matters

Almost nobody's first LangChain problem is conceptual. It is a wrong Python version, a stale package pin, a key the process cannot see, or an agent that loops and burns budget while you stare at a blank terminal.

Get the environment and the guardrails right once, here, and every later module is about LangChain instead of about pip.

There is a second reason. Most people arrive at LangChain with the wrong mental model, they think it is *how you call an LLM*. It is not. Knowing what LangChain is actually for, and when to skip it, is the difference between using it well and cargo-culting it into every project.

---

## 2. Concepts

### 2.1 What LangChain v1 actually is

Three things, and it helps to keep them separate:

1. **An agent runtime.** A loop that calls a model, executes the tools the model asks for, feeds results back, and repeats until done. This is `create_agent`, and it is the centre of the framework.
2. **A provider-neutral interface.** One way to talk to Claude, GPT, Gemini, or a local model, so swapping providers is a string change rather than a rewrite.
3. **An integration layer.** Hundreds of pre-built connectors, document loaders, vector stores, embedding models, tool wrappers, so you are not writing a PDF parser.

### 2.2 What it is not, and when to skip it

**LangChain is not required to call an LLM.** If your task is "send a prompt, get text back," the provider's own SDK is smaller, faster to debug, and has one less dependency:

```python
# You do not need LangChain for this. Every provider ships its own SDK;
# this happens to be OpenAI's.
from openai import OpenAI
client = OpenAI()
resp = client.responses.create(
    model="gpt-5.5",
    input="Summarise this.",
)
```

Reach for LangChain when you have **at least one** of:

- An agent loop, the model decides which tools to call, and how many times
- Retrieval over your own documents
- A need to swap providers without rewriting
- Multi-step orchestration with branching or human approval

If you have none of those, use the SDK. An engineer who knows when *not* to use the framework is more valuable than one who reaches for it reflexively.

### 2.3 The three packages

| Package | What it gives you | First used in |
|---|---|---|
| `langchain` | `create_agent`, models, tools, messages | Module 0 |
| `langgraph` | Explicit control flow, checkpointers, state | Module 5 |
| `langsmith` | Tracing and evaluation | Module 3 |

You install `langchain` and get `langgraph` automatically, the agent runtime is built on it. You will not touch `langgraph` directly until Module 5.

Provider integrations install as extras: `"langchain[openai]"`, `"langchain[google-genai]"`, `"langchain[anthropic]"`, `"langchain[ollama]"` and so on. Install only the one you use.

### 2.4 Cost guardrails, do this before your first call

An agent loop can call a model many times in one `invoke()`. A tool that always returns "not finished" will loop until it hits the iteration cap. If the cap is high and the model is expensive, that is real money.

Four habits, in order of importance:

1. **Set a spend limit in your provider console.** A hard cap the code cannot override. Do this now, not later.
2. **Use a cheap model while iterating.** Switch to the good one when the logic is right.
3. **Cap iterations.** Every agent gets an explicit limit.
4. **Watch your traces.** From Module 3 you will see token counts per run. Look at them.

---

## 3. Walkthrough

### 3.1 Install

```bash
mkdir -p ~/dev/langchain-learning && cd ~/dev/langchain-learning
python3 -m venv .venv
.venv/bin/pip install -U python-dotenv

# langchain plus the extra for YOUR provider (see model-setup.md):
.venv/bin/pip install -U "langchain[openai]"        # or [google-genai], [anthropic],
                                                     # [groq], [mistralai], [ollama], ...
```

Check what you got. If `langchain` is below 1.0, everything in this path will fail:

```bash
.venv/bin/pip list | grep -E "^(langchain|langgraph)"
```

Expected (versions will drift upward; the major version is what matters):

```
langchain                1.3.14
langchain-openai         (your provider's package)
langchain-core           1.5.3
langgraph                1.2.10
```

### 3.2 Keys and environment

Create `.env` in your project root. Fill in the setup you chose in
[Choosing your model](./model-setup.md); leave the rest commented out, you
will uncomment lines as later modules need them.

```bash
# =====================================================================
#  .env  -  NEVER commit this file
# =====================================================================

# ---------------------------------------------------------------------
#  1. MODEL  (Module 0)
# ---------------------------------------------------------------------
# Set AGENT_MODEL to any provider LangChain supports, then set that
# provider's key below. Every code sample reads AGENT_MODEL, so this is
# the only line you change to switch providers.
#
#   provider:model                                    key variable
#   ------------------------------------------------  ----------------------
#   openai:gpt-5.5                                    OPENAI_API_KEY
#   google_genai:gemini-flash-latest                GOOGLE_API_KEY
#   anthropic:claude-haiku-4-5                        ANTHROPIC_API_KEY
#   groq:llama-3.3-70b-versatile                      GROQ_API_KEY
#   mistralai:mistral-large-latest                    MISTRAL_API_KEY
#   deepseek:deepseek-chat                            DEEPSEEK_API_KEY
#   together:...                                      TOGETHER_API_KEY
#   fireworks:accounts/fireworks/models/...           FIREWORKS_API_KEY
#   baseten:zai-org/GLM-5.2                           BASETEN_API_KEY
#   xai:grok-...                                      XAI_API_KEY
#   perplexity:...                                    PERPLEXITY_API_KEY
#   huggingface:microsoft/Phi-3-mini-4k-instruct      HUGGINGFACEHUB_API_TOKEN
#   azure_openai:gpt-5.5                              AZURE_OPENAI_API_KEY
#   bedrock_converse:us.anthropic.claude-sonnet-4-6   AWS_ACCESS_KEY_ID etc
#   openrouter:anthropic/claude-sonnet-4-6            OPENROUTER_API_KEY
#   ollama:llama3.1:8b                                (none, runs locally)
#
# Full list: https://docs.langchain.com/oss/python/integrations/providers

AGENT_MODEL=openai:gpt-5.5
OPENAI_API_KEY=sk-...

# --- other providers: uncomment the pair you use ---------------------
# AGENT_MODEL=google_genai:gemini-flash-latest
# GOOGLE_API_KEY=...
#
# AGENT_MODEL=anthropic:claude-haiku-4-5
# ANTHROPIC_API_KEY=sk-ant-...
#
# Local, no key and no cost. Requires `ollama serve` running.
# AGENT_MODEL=ollama:llama3.1:8b

# ---------------------------------------------------------------------
#  2. GUARDRAILS  (Module 3)  -  set these before your first loop
# ---------------------------------------------------------------------
AGENT_MAX_MODEL_CALLS=6
AGENT_RECURSION_LIMIT=25

# ---------------------------------------------------------------------
#  3. TRACING  (Module 3, optional but recommended)
# ---------------------------------------------------------------------
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=yg-learning

# ---------------------------------------------------------------------
#  4. RETRIEVAL  (Module 6)  -  local, no key, works offline
# ---------------------------------------------------------------------
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
# RETRIEVAL_K=4

# ---------------------------------------------------------------------
#  5. PERSISTENCE  (Module 5)
# ---------------------------------------------------------------------
# CHECKPOINT_DB=checkpoints.db
```

Then load it at the top of every script, **before** importing anything from
LangChain:

```python
from dotenv import load_dotenv
load_dotenv()
```

Order matters: `load_dotenv()` sets the variables in the process, and some
libraries read them at import time. Calling it after your imports is the
single most common "my key works in the shell but not in the script" bug.

Now `.gitignore`, **before** you `git init`:

```
.env
.venv/
__pycache__/
```

A key committed to a repository is a key you must rotate, and it stays in the
git history even after you delete the file. Write `.gitignore` first.

> The same variables, in the same shape, are in
> `templates/agent-app/.env.example` in this repository. From Module 2 onward
> you will clone that template rather than writing this by hand.

### 3.3 Your first agent

```python
# hello_agent.py
from dotenv import load_dotenv
load_dotenv()
import os

MODEL = os.environ["AGENT_MODEL"]   # set in your .env, any provider

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 32°C and humid in {city}."

agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Be concise.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Chennai?"}]}
)
print(result["messages"][-1].text)
```

Nine lines of substance. Read them now, and read them again after Module 3: every one will mean something different.

> **About `MODEL` in these examples.** Every code sample in this path reads
> `MODEL` from your `.env` rather than hardcoding a provider:
>
> ```python
> MODEL = os.environ["AGENT_MODEL"]
> ```
>
> Set `AGENT_MODEL` once in Module 0 and every later module works unchanged,
> whichever provider you chose. Short illustrative snippets later on refer to
> `MODEL` without redefining it; it always means this.


Things worth noticing already:

- `MODEL` comes from your `.env` as a `provider:model` string. Change it to any of the 23 supported providers and, with that extra installed, the rest of the file is untouched. That portability is the point.
- `get_weather` is a plain Python function. LangChain reads its **signature and docstring** to tell the model what it does. Module 2 is entirely about why that docstring matters more than you would guess.
- You never wrote the loop. `create_agent` runs it: the model asked for `get_weather`, LangChain called it, handed back the result, and the model wrote the answer.
- `result["messages"]` is the whole conversation, not just the reply. `[-1]` is the final message.

---

## 4. Run it

```bash
.venv/bin/python hello_agent.py
```

**Expected output, illustrative.** Wording varies by model and run:

```
It's 32°C and humid in Chennai.
```

The check is not the wording. It is that **the number came from your function, not the model's imagination.** Change the return value to `"It's -40°C"` and re-run. If the answer changes to match, the tool was genuinely called. If it still says something plausible about Chennai weather, it was not, and that is the first agent bug you will ever debug.

---

## 5. Exercises

**5.1 Recall.** Name two situations where you should use the provider SDK directly instead of LangChain, and say why.

**5.2 Apply.** Print the whole `result["messages"]` list instead of the last one. Identify each message, who produced it, and which one carries the tool call. Sketch the loop on paper.

**5.3 Extend.** Add a second tool, `get_time(city: str)`, and ask a question needing both. Then ask a question needing neither. Note how many messages each produces, and what that implies about cost.

---

## 6. Assignment

A working environment and a short `SETUP.md` in your own repo recording:

- Python and package versions (from `pip list`, not from memory)
- Where your key lives and how the process reads it
- **The spend limit you set, and its value.** Screenshot it.
- Output of the tool-was-really-called check from §4

The spend limit is not busywork. You are about to write loops that call a paid API, and the single most common intern incident is a runaway loop discovered at the end of the month.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'create_agent'` | LangChain 0.x installed | `pip install -U langchain`; confirm `>=1.0` |
| `AuthenticationError` / 401 | Key not visible to the process | `load_dotenv()` before importing, or `export` it |
| Key works in shell, not in script | `.env` never loaded | Add `python-dotenv` and call `load_dotenv()` first |
| `ModuleNotFoundError: langchain_<provider>` | Provider extra not installed | `pip install "langchain[<provider>]"` |
| `404 NOT_FOUND ... no longer available to new users` | Model name retired for new accounts, even though it still appears in the models list | Use a `-latest` alias, or list the models your key can reach |
| Answer looks right but ignores your tool | Tool never called | Change its return to something absurd; see §4 |
| Runs forever / huge bill | Loop with no cap | Set spend limit; cap iterations (Module 3) |
| Tutorial online uses `LLMChain` | Written pre-v1 | See Appendix A in the [syllabus](./index.md) |

---

## 8. Check yourself

1. **What are the three things LangChain gives you?**
   An agent runtime, a provider-neutral model interface, and an integration layer.

2. **You need to summarise a document with one model call. LangChain or the SDK?**
   The SDK. No agent loop, no retrieval, no orchestration, the framework earns nothing here.

3. **Why is the model a `provider:model` string rather than an import?**
   Because it is configuration, not code. Any of the 23 supported providers can be swapped in without touching a line of your program.

4. **How does the model know what `get_weather` does?**
   From the function's signature and docstring, which LangChain converts into a schema. This is why the docstring is not a comment, it is part of the interface.

5. **What is the first thing to do before running an agent against a paid API?**
   Set a hard spend limit in the provider console.

---

## 9. References

- LangChain overview: https://docs.langchain.com/oss/python/langchain/overview
- Installation: https://docs.langchain.com/oss/python/langchain/install
- API reference: https://reference.langchain.com

---

*Next: [Module 1: Models & Messages](./langchain-001-models-and-messages.md). You will take apart the `result["messages"]` list you printed in Exercise 5.2.*
