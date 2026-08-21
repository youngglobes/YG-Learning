# YG-Learning

Internal learning resources for YoungGlobes engineers.

---

## Available now

### [LangChain Learning Path](./docs/langchain/index.md)

Thirteen modules taking an engineer from zero to building production agent
applications. Built against **LangChain v1**, not the pre-v1 framework most
tutorials online still teach.

| Start here | |
|---|---|
| [Choosing your model](./docs/langchain/model-setup.md) | Any provider LangChain supports, or run one locally for free |
| [8-week learning plan](./docs/langchain/learning-plan.md) | What to read and build each week |
| [Syllabus](./docs/langchain/index.md) | All 13 modules |
| [Agent app template](./templates/agent-app/README.md) | Clone this for every project |

---

## Quick start for a learner

```bash
git clone https://github.com/youngglobes/YG-Learning.git
cd YG-Learning

# 1. pick your model
open docs/langchain/model-setup.md

# 2. follow the plan
open docs/langchain/learning-plan.md

# 3. when you start building
cp -r templates/agent-app ~/dev/my-project
cd ~/dev/my-project && make install-ollama && make test
```

Nothing is hosted. You run everything on your own machine, with your own
model. **Nobody is required to buy anything.** Any provider works, and a
local model via Ollama completes every module at no cost.

---

## What makes this different from other LangChain tutorials

**It teaches the framework that actually exists.** Most content online covers
`LLMChain`, `initialize_agent`, and `ConversationBufferMemory`, all removed or
deprecated in v1. Appendix A of the syllabus lists the dead APIs with their
replacements, so you can date a tutorial in ten seconds instead of debugging
for an afternoon.

**Every claim was checked against the installed packages.** Writing it turned
up several things the documentation and the plan got wrong, `@tool` silently
discarding parameter descriptions without `parse_docstring=True`, Anthropic
shipping no embeddings API at all, `SqliteSaver` not being bundled,
`similarity_search_with_relevance_scores` raising `NotImplementedError` on the
in-memory store, and `GenericFakeChatModel` being unusable with any agent that
has tools. Each of those would have stopped a learner on day one.

**Failure modes are taught, not discovered.** Every module has a *Common
failures* table listing the errors you will actually hit, with fixes.

**Security is required material.** Prompt injection through retrieved
documents is Module 7's core content, with an assignment that has you attack
your own system. It is not an advanced footnote.

---

## Status

The material is complete and the code samples are verified against the
installed packages. **Behavioural claims have not yet been confirmed against a
live model.** Whether the agent actually declines, cites its sources, and
resists injection is still unverified. Those claims are tracked in the
[verification checklist](./docs/langchain/VERIFICATION.md) and are being
worked through now.

Expect corrections. If you hit something wrong, that is a finding worth more
than a new module, record it and raise it.

---

## Licence

This repository is dual-licensed, which is the usual arrangement for a
documentation project that also ships code.

| What | Licence | Applies to |
|---|---|---|
| **Documentation** | [CC BY 4.0](./LICENSE) | `docs/`, `README.md`, and all prose |
| **Code** | [MIT](./LICENSE-CODE) | `templates/`, `scripts/`, and every code sample |

Copyright (c) 2026 YoungGlobes.

In practice: you may share and adapt the written material, including
commercially, as long as you credit YoungGlobes. You may use the code for
anything, with no obligation beyond keeping the copyright notice.

> Previously GPL-2.0. That is a software copyleft licence and a poor fit for
> documentation, it would have required anyone reusing this material to
> license their derivative work under GPL-2.0 as well. Changed before any
> outside contributions made it difficult to change.

---

## Contributing

Keep a `STUCK.md` as you learn: what you expected, what happened, what fixed
it. Bring it to your check-ins. Being early through this material makes your
confusion the most useful data we have.

Modules follow a fixed nine-section template, see Appendix B of the
[syllabus](./docs/langchain/index.md). Keep it; the consistency is what makes
thirteen modules navigable.

