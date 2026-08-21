# 8-week learning plan

For a new joiner working through the LangChain path roughly half-time. Adjust
the calendar, keep the order, each week's project depends on the one before.

**Read [Choosing your model](./model-setup.md) before Week 1.**

The shape: you do **not** read thirteen modules and then build something. You
build three things, and the reading feeds each one. Your first working tool
ships in Week 2.

---

## Overview

| Week | Read | Build | Done when |
|---|---|---|---|
| 1 | Modules 0-2 |, | An agent calls your own tool |
| 2 | Modules 3-4 | **Project 1: a CLI tool** | It ships and its tests pass |
| 3 | Modules 5-6 |, | An index you can query |
| 4 | Module 7 | **Project 2: RAG with citations** | It cites, declines, and survives injection |
| 5 | Modules 8-9 | Harden Project 2 | Middleware + an eval suite in CI |
| 6 | Module 10 | Approval workflow | An interrupt survives a restart |
| 7 | Modules 11-12 | Capstone starts | Architecture agreed |
| 8 |, | **Capstone** | All seven deliverables |

---

## Week 1. Foundations

**Read:** Modules 0, 1, 2 (~8-11 h)

**Do**
- Set up your model. **Set a spend limit first if you chose Claude.**
- Module 0's check: change the tool's return to something absurd and confirm the answer follows. That proves the tool was really called.
- Module 1's Exercise 5.2: print the whole message list and identify every message.
- Module 2's Exercise 5.2: break a docstring, watch tool selection degrade, restore it.

**Done when:** you can explain, without looking, what the model actually sees of your tool.

> Nothing to ship this week. It is the only such week.

---

## Week 2. Project 1

**Read:** Modules 3, 4 (~8-10 h)

**Build: a CLI tool for something you actually do.** A log analyser, a git
report, a ticket summariser, anything real. Requirements:

- Clone `templates/agent-app`
- **Three tools**, all `parse_docstring=True`, all returning recoverable errors
- One output path uses a Pydantic schema (Module 4)
- A model call limit
- `make test` passes

**Done when:** a colleague can clone it, follow the README, and run it.

> This is the week the path stops being study. Ship something small and real.

---

## Week 3. Retrieval

**Read:** Modules 5, 6 (~10-13 h)

**Do**
- Add memory to Project 1 with a persistent checkpointer. Prove threads are isolated.
- Build an index over ≥30 real documents.
- Module 6's assignment: the chunking bake-off, **write the 15 questions before you tune anything.**

**Done when:** you can state, with numbers, which chunk size works best on your corpus and why.

> **Module 6 needs no API key.** If access is still pending, this week is unaffected.

---

## Week 4. Project 2

**Read:** Module 7 (~6-8 h)

**Build: document Q&A over your Week 3 corpus.**

- Retrieval as a **tool**, not a chain
- `content_and_artifact` so citations survive
- Declines when it does not know
- **The injection test set**: five payloads, a test each, a written report

**Done when:** the injection report is written, including which payloads
succeeded. Some will. Recording that honestly is the deliverable, and note
which defence held anyway.

---

## Week 5. Make it trustworthy

**Read:** Modules 8, 9 (~10-13 h)

**Do**
- Add middleware to Project 2. **List the built-ins before writing any.**
- Build the eval suite: ≥30 examples, ≥3 evaluators, two experiments comparing one change.
- Wire it into CI.

**Done when:** you can say what a change did, improved *this*, regressed *that*, cost *this much*.

> The most important week on the path. It is where you stop guessing.

---

## Week 6. Control flow

**Read:** Module 10 (~6-8 h)

**Build: an approval-gated workflow.**

- One node that always runs
- Deterministic routing on a business rule
- A human approval gate
- **Persistence proven by killing the process and resuming**

**Done when:** the restart test passes. Same-process resume does not count.

---

## Week 7. Scale and plan

**Read:** Modules 11, 12 (~15-21 h)

**Do**
- Module 11's cost comparison: multi-agent vs a single-agent baseline. Build both.
- Choose your capstone and agree the architecture with your mentor.

**Done when:** the architecture is agreed and you can defend the multi-agent decision, including deciding against it.

---

## Week 8. Capstone

**Build the AI Helpdesk Assistant.** All seven deliverables in Module 12: code,
README, architecture diagram, eval suite passing in CI, cost model, security
review, runbook.

**Done when it passes the Module 12 criteria**: runs from a clean clone,
evals pass, the cost model is measured not guessed, the security review names
a real risk you found, and a deliberate injection attempt fails.

---

## Keeping a log

**Keep a `STUCK.md` from day one.** Every point where you got stuck, confused,
or where the tutorial was wrong: what you expected, what happened, what fixed
it.

This is the most valuable thing you produce in Week 1, and possibly the whole
path. You are among the first through this material, the log is what makes it
better for the next person. Bring it to every check-in.

Also fill in the actual time per module. The estimates are guesses.

---

## Check-ins

| When | What | Duration |
|---|---|---|
| End of Week 2 | Project 1 code review | 45 min |
| End of Week 4 | Project 2 demo + injection report | 1 h |
| End of Week 5 | Present an eval trade-off with numbers | 30 min |
| End of Week 8 | Capstone review | 1.5 h |

Four touchpoints. Between them you work independently, that is the point, and
the tests are there so you are not blocked waiting for a review.

---

## If you fall behind

Normal. In order of what to protect:

1. **Never skip Module 7's injection work.** It is the one with real-world consequences.
2. **Never skip Module 9.** Without evaluation you cannot tell good from bad.
3. Module 11 can be read rather than built.
4. The capstone can slip a week. It is better done than rushed.

Modules 0-7 are the core. Everything after makes you trustworthy in production.
