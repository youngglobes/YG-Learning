# Choosing your model

**Read this before Module 0.**

This tutorial supports **two setups**. Pick one:

| | **Claude** (hosted) | **Ollama** (local) |
|---|---|---|
| Cost | you pay, ~₹700 for the whole path on Haiku 4.5 | free |
| Needs | an API key | a reasonably specced machine |
| Setup | ~5 min | ~20 min |
| Speed | fast | depends on your hardware |
| Behaves as the modules describe | yes | mostly — see capability tiers below |

**Nobody is required to buy anything.** Ollama completes every module at no
cost. If you already have a Claude key, or would rather spend a few hundred
rupees than wait on a slow local model, use that instead. Your call.

These two are what the examples are written and maintained for. Other
providers (OpenAI, Gemini, Groq, OpenRouter) work through the same interface
and you are welcome to use one — but nothing here has been checked against
them, so you are on your own for differences.

> **Verification status, stated plainly:** every code sample has been run
> against the installed packages, so imports, signatures, and data structures
> are correct. The *behavioural* checks — does the agent decline, does it
> resist injection — have not yet been confirmed against a live model on
> either setup. Treat the "Expected output" blocks as illustrative until
> someone completes a cohort and corrects them.

Switching is one string, and that single-line swap *is* one of LangChain's real
selling points. You will be learning it by living it:

```python
model="anthropic:claude-haiku-4-5"   # Claude - cheapest, $1/$5 per MTok
model="ollama:llama3.1:8b"           # Ollama
```

**A practical middle path:** start on Ollama for Modules 0–9, then switch to
Claude for Modules 10–12 and the capstone, where reasoning quality is the
actual subject. Costs a fraction of the full-path figure and defers any
spending decision until you already know the path is worth it to you.

---

## Ollama (local)

### Will your machine cope?

Check first — this matters more than people expect:

```bash
free -h                  # Linux / WSL: look at "total"
nproc                    # core count
nvidia-smi               # GPU? if this errors, you are CPU-only
```

On WSL, `free -h` shows the **WSL cap**, not your Windows RAM. Raise it in
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=4
```

### What you can run

| Free RAM | Model | Speed, CPU-only | Verdict |
|---|---|---|---|
| < 6 GB | — | — | Not viable. Use route B, or route C for the no-model modules |
| 6–8 GB | `llama3.2:3b` | 8–15 tok/s | Works, but weak at tool calling — read the tier table below |
| 8–16 GB | `llama3.1:8b` | 3–6 tok/s CPU · much faster with an NVIDIA GPU | The realistic local choice |
| 16 GB+ with GPU | `qwen2.5:14b` or larger | fast | Comfortable |

**An NVIDIA GPU changes everything; an Intel or AMD integrated GPU does not.**
If `nvidia-smi` errors, assume CPU speeds from the table.

Remember agent loops multiply this. One turn with a tool call is 2+ model calls,
so 5 tok/s feels like 15–20 seconds per turn, and Module 11's multi-agent
exercises will feel slow enough to be annoying.

### Install

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b          # ~4.7GB
ollama serve                     # leave running
```

```bash
pip install langchain-ollama
```

```python
model = init_chat_model("ollama:llama3.1:8b")
```

---

## Claude (hosted)

Get a key from the Anthropic console. Set a **hard spend cap there before your
first call** — that is Module 0's first assignment and it is not
ceremony.

Rough cost for the *entire* 13-module path with exercises — around 1,000 agent
runs, ~5M input and ~0.5M output tokens:

| Model | Whole path | Per MTok in / out |
|---|---|---|
| Claude Haiku 4.5 | ~$8 (~₹700) | $1 / $5 |
| Claude Sonnet 5 | ~$15 (~₹1,300) | $2 / $10 |
| Claude Opus 5 | ~$38 (~₹3,300) | $5 / $25 |

A small fast model is genuinely fine for Modules 0–9. Save the expensive one
for Modules 10–12 where reasoning quality is the actual subject.

---

## Two modules need no model at all

If you are still sorting out access, you are not blocked:

- **Module 6 (Retrieval Foundations)** — embeddings run locally on CPU. No LLM,
  no key. It is one of the biggest modules and entirely doable offline.
- **Module 1 (Models & Messages)** — the message-structure half works against
  any model, including a tiny local one.

---

## Capability tiers — read this before you think you broke something

Several modules ask you to verify a **behaviour**, not just that code runs. Some
of those behaviours depend on how capable your model is. If a check below fails
on a small local model, **your code is probably fine and the model simply cannot
do it.**

| Module | Check | Small local (3B) | Mid local (8B) | Hosted |
|---|---|---|---|---|
| 0 | Tool actually called | usually | yes | yes |
| 2 | Calls **nothing** for "thanks" | often fails | usually | yes |
| 2 | Picks the right tool of three | sometimes fails | usually | yes |
| 3 | Terminates the loop sensibly | often fails | usually | yes |
| 4 | Optional fields return `null`, not invented | **usually fails** | often fails | usually |
| 7 | Declines out-of-scope questions | **usually fails** | sometimes fails | usually |
| 7 | Resists prompt injection | **fails** | often fails | often resists |
| 11 | Declines to delegate trivial questions | **usually fails** | sometimes fails | usually |

**This table is a teaching tool, not an apology.** Watching a 3B model get
compromised by every injection payload in Module 7 while a hosted model resists
most of them is a genuinely useful thing to see. It is the clearest possible
demonstration of the module's actual argument: *prompt-level defence is
model-dependent, architectural defence is not.* A read-only toolset protects
you on every model in that table.

So if you are on a small model, do the exercise, record what happened, and
write down which defence still held. That is the right answer, not a failure.

---

## Record your choice

Put this at the top of your notes, and in your project README:

```
Setup:            Claude / Ollama
Model string:     ollama:llama3.1:8b
Machine:          16GB RAM, 8 cores, no NVIDIA GPU
Embeddings:       sentence-transformers/all-MiniLM-L6-v2 (local)
```

When you compare results with a colleague and they differ, the model is the
first thing to check — and you will only know if you wrote it down.

---

*Next: [Module 0 — Setup & Mental Model](./langchain-000-setup-and-mental-model.md).*
