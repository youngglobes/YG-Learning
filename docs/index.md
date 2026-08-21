# YG Learning

Internal learning resources for YoungGlobes engineers.

## LangChain Learning Path

Thirteen modules taking an engineer from zero to building production agent
applications, built against **LangChain v1**, not the pre-v1 framework most
tutorials online still teach.

<div class="grid cards" markdown>

- **[Choosing your model](langchain/model-setup.md)**
  Claude or Ollama. Your choice, and nobody is required to buy anything.

- **[8-week learning plan](langchain/learning-plan.md)**
  What to read and what to build each week.

- **[Syllabus](langchain/index.md)**
  All thirteen modules, and why they are ordered this way.

- **[Verification checklist](langchain/VERIFICATION.md)**
  Behavioural claims still needing confirmation against a live model.

</div>

## Building something?

Clone the agent app template, it is in the repository at
`templates/agent-app/`, and its tests run with no API key.

```bash
cp -r templates/agent-app ~/dev/my-project
cd ~/dev/my-project && make install-ollama && make test
```
