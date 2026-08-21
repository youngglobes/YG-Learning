# Choosing your model

**Read this before Module 0.**

This path is provider neutral. Every module works with any model LangChain
supports, and switching between them is one string. Pick whichever you already
have access to, or run one locally for free.

---

## Pick a provider

Install `langchain` plus the extra for your provider, set the key, and use the
matching model string. **Nothing else in the tutorial changes.**

| Provider | Install | API key variable | Example model string |
|---|---|---|---|
| OpenAI | `"langchain[openai]"` | `OPENAI_API_KEY` | `openai:gpt-5.5` |
| Google Gemini | `"langchain[google-genai]"` | `GOOGLE_API_KEY` | `google_genai:gemini-2.5-flash-lite` |
| Anthropic | `"langchain[anthropic]"` | `ANTHROPIC_API_KEY` | `anthropic:claude-haiku-4-5` |
| Groq | `"langchain[groq]"` | `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` |
| Mistral | `"langchain[mistralai]"` | `MISTRAL_API_KEY` | `mistralai:mistral-large-latest` |
| DeepSeek | `"langchain[deepseek]"` | `DEEPSEEK_API_KEY` | `deepseek:deepseek-chat` |
| Fireworks | `"langchain[fireworks]"` | `FIREWORKS_API_KEY` | `fireworks:accounts/fireworks/models/...` |
| Together | `"langchain[together]"` | `TOGETHER_API_KEY` | `together:...` |
| Baseten | `"langchain[baseten]"` | `BASETEN_API_KEY` | `baseten:zai-org/GLM-5.2` |
| xAI | `"langchain[xai]"` | `XAI_API_KEY` | `xai:grok-...` |
| Perplexity | `"langchain[perplexity]"` | `PERPLEXITY_API_KEY` | `perplexity:...` |
| Hugging Face | `"langchain[huggingface]"` | `HUGGINGFACEHUB_API_TOKEN` | `huggingface:microsoft/Phi-3-mini-4k-instruct` |
| Azure OpenAI | `"langchain[openai]"` | `AZURE_OPENAI_API_KEY` + endpoint | `azure_openai:gpt-5.5` |
| AWS Bedrock | `"langchain[aws]"` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | `bedrock_converse:us.anthropic.claude-sonnet-4-6` |
| Google Vertex | `"langchain[google-vertexai]"` | GCP application default credentials | `google_vertexai:...` |
| OpenRouter | `pip install langchain-openrouter` | `OPENROUTER_API_KEY` | `openrouter:anthropic/claude-sonnet-4-6` |
| **Ollama (local)** | `"langchain[ollama]"` | **none** | `ollama:llama3.1:8b` |

There are 23 providers registered in total. If yours is missing from this
table, check the
[integrations index](https://docs.langchain.com/oss/python/integrations/providers).

## The code never changes

That is the whole point, and it is worth seeing directly:

```python
MODEL = "openai:gpt-5.5"                      # or
MODEL = "google_genai:gemini-2.5-flash-lite"  # or
MODEL = "anthropic:claude-haiku-4-5"          # or
MODEL = "ollama:llama3.1:8b"                  # or any of the others

agent = create_agent(model=MODEL, tools=[...], system_prompt="...")
```

Every code sample in this path reads `MODEL` from your `.env`, so you set it
once in Module 0 and never touch it again.

---

## No key? Run a model locally

Ollama needs no account and costs nothing. Check your machine first:

```bash
free -h      # Linux / WSL: on WSL this shows the WSL cap, not Windows RAM
nproc        # core count
nvidia-smi   # if this errors, you are CPU only
```

| Free RAM | Model | Speed, CPU only | Verdict |
|---|---|---|---|
| under 6 GB | none | | Not viable. Use a hosted provider |
| 6 to 8 GB | `llama3.2:3b` | 8 to 15 tok/s | Works, but weak at tool calling |
| 8 to 16 GB | `llama3.1:8b` | 3 to 6 tok/s, much faster with an NVIDIA GPU | The realistic local choice |
| 16 GB+ with GPU | `qwen2.5:14b` or larger | fast | Comfortable |

An NVIDIA GPU changes everything. An Intel or AMD integrated GPU does not.

On WSL, raise the memory cap in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=4
```

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

Agent loops multiply slowness. One turn with a tool call is two or more model
calls, so 5 tok/s feels like 15 to 20 seconds per turn.

---

## Using a hosted provider? Cap your spend

**Set a hard spend limit in your provider console before your first call.**
That is Module 0's first assignment and it is not ceremony.

Rough size of the whole path: around 1,000 agent runs, roughly 5M input and
0.5M output tokens. Multiply by your provider's rates. For scale, a small fast
model at $1 per million input and $5 per million output comes to about $8 for
the entire 13 modules.

Use a small, cheap model for Modules 0 to 9. Save the expensive one for
Modules 10 to 12, where reasoning quality is the actual subject.

---

## Two modules need no model at all

If access is still pending, you are not blocked:

- **Module 6 (Retrieval Foundations)** runs entirely on local CPU embeddings.
  No LLM, no key. It is one of the largest modules and works fully offline.
- **Module 1 (Models & Messages)** works against any model, including a tiny
  local one.

---

## Model capability, not vendor

Several modules ask you to verify a **behaviour**, not just that code runs.
Those behaviours depend on how capable your model is, and that is about model
size and class rather than which company made it. A large hosted model from
any vendor behaves similarly. A 3B local model does not.

If a check below fails on a small model, **your code is probably fine.**

| Module | Check | Small local (3B) | Mid local (8B) | Large hosted |
|---|---|---|---|---|
| 0 | Tool actually called | usually | yes | yes |
| 2 | Calls **nothing** for "thanks" | often fails | usually | yes |
| 2 | Picks the right tool of three | sometimes fails | usually | yes |
| 3 | Terminates the loop sensibly | often fails | usually | yes |
| 4 | Optional fields return `null`, not invented | **usually fails** | often fails | usually |
| 7 | Declines out of scope questions | **usually fails** | sometimes fails | usually |
| 7 | Resists prompt injection | **fails** | often fails | often resists |
| 11 | Declines to delegate trivial questions | **usually fails** | sometimes fails | usually |

**This table is a teaching tool, not an apology.** Watching a 3B model get
compromised by every injection payload in Module 7, while a large model
resists most of them, is the clearest possible demonstration of that module's
actual argument: prompt level defence is model dependent, architectural
defence is not. A read only toolset protects you on every column of that
table.

So if you are on a small model, do the exercise, record what happened, and
write down which defence still held. That is the right answer, not a failure.

> **Verification status, stated plainly:** every code sample has been run
> against the installed packages, so imports, signatures, and data structures
> are correct. The behavioural checks have not yet been confirmed against a
> live model on any provider. Treat the "Expected output" blocks as
> illustrative until someone completes a cohort and corrects them.

---

## Record your choice

Put this at the top of your notes and in your project README:

```
Provider:     openai / google_genai / anthropic / ollama / ...
Model string: openai:gpt-5.5
Machine:      16GB RAM, 8 cores, no NVIDIA GPU
Embeddings:   sentence-transformers/all-MiniLM-L6-v2 (local)
```

When your results differ from a colleague's, the model is the first thing to
check, and you will only know if you wrote it down.

---

*Next: [Module 0: Setup & Mental Model](./langchain-000-setup-and-mental-model.md).*
