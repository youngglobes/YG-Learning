# Agent app template

Starting point for every agent project on the YoungGlobes LangChain path.
Clone it, rename it, build in it. It answers the question no tutorial answers:
**where do I put things?**

---

## Quick start

```bash
cp -r templates/agent-app ~/dev/my-project && cd ~/dev/my-project
cp .env.example .env          # then edit it
make install-ollama           # or: make install-claude
make test                     # passes with no API key
make run
```

---

## Layout

```
src/agent_app/
  config.py     all settings, read from env. THE place the model is chosen
  tools.py      @tool definitions
  agent.py      assembly: create_agent + middleware + system prompt
  cli.py        entry point
tests/
  conftest.py   FakeToolModel - lets you test agents with no API key
  test_tools.py tools are plain functions; test them like plain functions
  test_agent.py wiring, thread isolation, memory - all with a fake model
evals/          your Module 9 dataset and runner
corpus/         documents for RAG projects (gitignored)
```

### Why it is split this way

**`config.py` is the only file that knows which model you use.** Switching
between Claude and Ollama is an env var, never a code change. That is the
provider-neutrality from Module 1, made real.

**`agent.py` accepts an injected model.** That is what makes the tests
possible, pass a fake in tests, use the configured one in production.

**`tools.py` is separate** because tools are plain functions and deserve plain
unit tests that never touch a model.

---

## Testing without an API key

`tests/conftest.py` provides `FakeToolModel`. This exists because
langchain-core's `GenericFakeChatModel` **does not implement `bind_tools()`**,
so `create_agent()` raises `NotImplementedError` as soon as your agent has any
tools. A no-op `bind_tools` fixes it:

```python
class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self
```

With that, `make test` runs in under a second, costs nothing, and needs no
key, so it belongs in CI on every commit.

**What you can test this way:** tool logic, tool schemas, agent wiring, thread
isolation, memory behaviour, error handling. Everything you actually own.

**What you cannot:** whether the model declines, cites, or resists injection.
Those depend on the model. Write them as tests, mark them
`@pytest.mark.skipif`, and run them deliberately, `test_agent.py` has one as
an example.

---

## The rules this template already applies

| Rule | Module | Where |
|---|---|---|
| `parse_docstring=True` on every tool | 2 | `tools.py` |
| Docstrings say when **not** to call | 2 | `tools.py` |
| Errors returned, never raised | 2 | `tools.py` |
| A model call limit on every agent | 3 | `agent.py` |
| `recursion_limit` set on invoke | 3 | `agent.py` |
| `thread_id` never from user input | 5 | `agent.py` docstring |
| Tool output treated as untrusted data | 7 | system prompt |
| Errors degrade visibly | 12 | `cli.py` |
| Secrets in `.env`, never committed | 0 | `.gitignore` |

You get these by default. Do not remove them without a reason you can state.

---

## Growing it

**RAG (Module 7):** add `retrieval.py` that builds the index from `corpus/`,
and a `retrieve_context` tool with
`@tool(response_format="content_and_artifact")` so citations survive.

**Memory (Module 5):** pass a checkpointer to `build_agent`. Use
`SqliteSaver` (`pip install -e ".[sqlite]"`) so it survives restarts.

**Evaluation (Module 9):** put your dataset and runner in `evals/`, and wire
`make eval` into CI.

**Middleware (Module 8):** add to the list in `agent.py`. Check the built-ins
first, there are about twenty.

---

## Make targets

| Command | Does |
|---|---|
| `make install` | venv + dev + rag deps |
| `make install-claude` | the above + `langchain-anthropic` |
| `make install-ollama` | the above + `langchain-ollama` |
| `make test` | pytest, no key needed |
| `make lint` | ruff |
| `make run` | one sample question |
