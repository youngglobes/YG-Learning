# Module 9: Evaluation

**Phase:** Orchestration & Production
**Prerequisites:** Modules 0-8, and the eval question set from Module 6
**Verified against:** `langsmith` 0.10.17, `langchain` 1.3.14, Python 3.12
**Estimated time:** 6-8 hours

---

## 1. Why this matters

For eight modules you have changed prompts, swapped chunk sizes, and reordered middleware, and judged the results by reading a couple of outputs and forming an impression.

That does not scale past a demo, for a specific reason: **LLM changes trade off.** A prompt edit that fixes the case in front of you routinely breaks two you are not looking at. Without measurement you cannot see the trade, so you ship it, and quality drifts downward one confident improvement at a time.

This is the module that separates an engineer from someone who assembles demos. It is also the one most courses put last, where it never gets taught.

---

## 2. Concepts

### 2.1 What an evaluation is

Three parts:

1. **A dataset**: inputs, and what a good output looks like
2. **Evaluators**: functions scoring an output against the reference
3. **An experiment**: running your system over the dataset and recording scores

Once that exists, "did this change help?" becomes a number instead of an argument.

### 2.2 Build the dataset from real traces

The temptation is to invent test questions. Invented questions are too clean, they use your vocabulary, ask one thing at a time, and are spelled correctly. Real users are not like that.

Your traces from Module 3 onward are a dataset waiting to be harvested. Pull real inputs, especially the ones that went badly.

Rules of thumb:

- **30 examples minimum** to see anything; 100+ to trust it
- **Include the failures deliberately.** A dataset of cases you already pass measures nothing
- **Write the reference before you tune.** A reference written afterwards drifts toward what your system already does
- **Version it.** A changing dataset makes scores incomparable across runs

### 2.3 Three kinds of evaluator, in order of preference

**Deterministic**: cheap, fast, perfectly reliable. Use wherever possible:

```python
def has_citation(outputs: dict) -> bool:
    """Every policy answer must cite a source file."""
    return ".md" in outputs["answer"]
```

**Heuristic**: string or structural checks with a defensible rule:

```python
def retrieved_correct_doc(outputs: dict, reference_outputs: dict) -> bool:
    """The known-correct source appears in what was retrieved."""
    return reference_outputs["source"] in outputs["sources"]
```

**LLM-as-judge**: a model scores the output. Necessary for qualities you cannot express as a rule, such as "is this answer faithful to the retrieved context":

```python
from langsmith import Client

def faithful(outputs: dict, reference_outputs: dict) -> bool:
    verdict = judge.invoke(
        "Does the ANSWER follow from the CONTEXT alone? Reply YES or NO.\n\n"
        f"CONTEXT:\n{outputs['context']}\n\nANSWER:\n{outputs['answer']}"
    )
    return verdict.text.strip().upper().startswith("YES")
```

**Where LLM-as-judge is and is not trustworthy.** It is reasonable for faithfulness, relevance, tone, and gross correctness. It is unreliable for anything numeric, for fine distinctions, and, critically, for **grading its own model's output**, where it tends to be generous. Use a different model as judge where you can, and spot-check the judge against human labels before trusting it. A judge you have never audited is a metric you have never validated.

### 2.4 Running an experiment

```python
from langsmith import Client, evaluate

client = Client()

dataset = client.create_dataset("yg-policy-rag-v1")
client.create_examples(
    dataset_id=dataset.id,
    inputs=[{"question": q} for q in questions],
    outputs=[{"source": s} for s in correct_sources],
)

results = evaluate(
    my_rag_system,                       # target: takes inputs, returns outputs
    data="yg-policy-rag-v1",
    evaluators=[has_citation, retrieved_correct_doc, faithful],
    experiment_prefix="chunk-1000",
)
```

Verified signature: `evaluate(target, data, evaluators, summary_evaluators, metadata, experiment_prefix, description, max_concurrency, ...)`.

Use `experiment_prefix` to label the variable you changed. Six months later "chunk-1000" tells you what the run was; "experiment-4" does not.

### 2.5 Regression testing in CI

An eval you run manually is an eval you stop running. Wire it to the pipeline:

```yaml
- name: Evaluate RAG
  run: python evals/run.py --fail-under 0.85
```

Two warnings from practice. **Scores are noisy**: models are non-deterministic, so a threshold set at your best-ever score will fail on green builds. Set it below your stable floor. And **cost is real**: a 100-example suite with an LLM judge is 200+ model calls. Run the full suite nightly and a smaller smoke subset per commit.

### 2.6 Reading results honestly

The output is a table of scores per example. The instinct is to read the average. The average hides the thing you need:

- **Look at what got worse**, not what got better. A change that lifts the mean while breaking three previously-passing cases is usually a bad change.
- **Read the failures individually.** Patterns live there; averages hide them.
- **Expect trade-offs.** If a change improves everything with no cost, be suspicious of the dataset before celebrating.

---

## 3. Walkthrough

```python
"""Module 9: evaluate the Module 7 RAG agent."""
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, evaluate
from rag_agent import agent          # your Module 7 agent

client = Client()

# ---- 1. dataset: reuse the question set written in Module 6 -------------
QUESTIONS = [
    ("How many leave days do I accrue per year?", "hr/leave-policy.md"),
    ("How much leave carries over?",              "hr/leave-policy.md"),
    ("What is the daily meal limit?",             "hr/expense-policy.md"),
    ("Do I need approval for an expensive flight?", "hr/expense-policy.md"),
    ("What is the capital of France?",            None),   # must decline
    # ... at least 30 in the real assignment
]

DATASET = "yg-policy-rag-v1"
if not client.has_dataset(dataset_name=DATASET):
    ds = client.create_dataset(DATASET)
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": q} for q, _ in QUESTIONS],
        outputs=[{"source": s} for _, s in QUESTIONS],
    )


# ---- 2. target: the system under test -----------------------------------
def rag_system(inputs: dict) -> dict:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]}
    )
    sources = []
    for m in result["messages"]:
        for d in getattr(m, "artifact", None) or []:
            src = d.metadata.get("source")
            if src and src not in sources:
                sources.append(src)
    return {"answer": result["messages"][-1].text, "sources": sources}


# ---- 3. evaluators: deterministic first ---------------------------------
def retrieved_correct_doc(outputs: dict, reference_outputs: dict) -> bool:
    """Did retrieval surface the known-correct document?"""
    expected = reference_outputs["source"]
    if expected is None:                    # out-of-scope question
        return outputs["sources"] == []     # correct behaviour is retrieving nothing
    return expected in outputs["sources"]


def cites_a_source(outputs: dict, reference_outputs: dict) -> bool:
    """In-scope answers must cite; out-of-scope answers must not invent one."""
    if reference_outputs["source"] is None:
        return ".md" not in outputs["answer"]
    return ".md" in outputs["answer"]


def declines_when_out_of_scope(outputs: dict, reference_outputs: dict) -> bool:
    """Out-of-scope questions must be declined, not answered from world knowledge."""
    if reference_outputs["source"] is not None:
        return True                          # not applicable
    text = outputs["answer"].lower()
    return any(p in text for p in ("don't know", "do not know", "not covered", "no information"))


# ---- 4. run -------------------------------------------------------------
results = evaluate(
    rag_system,
    data=DATASET,
    evaluators=[retrieved_correct_doc, cites_a_source, declines_when_out_of_scope],
    experiment_prefix="baseline-chunk1000-k2",
    max_concurrency=4,
)
print(results)
```

Note that three of the evaluators are **deterministic**. Before reaching for an LLM judge, ask what you can check with a rule, it is cheaper, faster, and never has a bad day.

---

## 4. Run it

```bash
export LANGSMITH_API_KEY=lsv2_...
.venv/bin/python evals/run.py
```

**Expected output, illustrative.** LangSmith prints a results table and a link. What matters:

- Every example has a score for all three evaluators
- `declines_when_out_of_scope` is **1.0 on the France question**. If it is 0, your agent answered from world knowledge and Module 7's scoping is not holding
- `retrieved_correct_doc` is below 1.0 on at least one example, if it is a perfect 1.0 on the first run, your dataset is too easy

That last check matters. A first eval that passes everything has told you nothing except that you wrote flattering questions.

---

## 5. Exercises

**5.1 Recall.** Name the three evaluator kinds in order of preference and say why that order.

**5.2 Apply.** Change `k` from 2 to 5 in your retrieval tool, re-run with `experiment_prefix="k5"`, and compare. Report which examples improved, which got worse, and what the token cost did. Do not report only the average.

**5.3 Extend.** Add an LLM-as-judge faithfulness evaluator. Then hand-label 20 examples yourself and measure how often the judge agrees with you. State whether you would trust it in CI, with the number behind your answer.

---

## 6. Assignment

A regression suite over your Module 7 RAG agent.

Requirements:

- **≥30 examples**, including at least 5 out-of-scope questions that must be declined, and at least 5 harvested from real traces where the system did badly
- **≥3 evaluators**, at least two deterministic
- **Two experiments** comparing one deliberate change (chunk size, `k`, prompt, or model), labelled with meaningful prefixes
- A CI job that fails below a threshold, with the threshold justified
- An `EVAL.md` reporting: the change, what improved, **what regressed**, the cost delta, and your ship/no-ship decision with reasoning

The graded part is the regression column. Any change that improves nothing is easy to reject and any change that improves everything is rare; the skill is deciding when an improvement is worth what it cost elsewhere.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Everything scores 1.0 on the first run | Dataset written to flatter the system | Harvest real failures |
| Scores swing between identical runs | Model non-determinism | More examples; threshold below the stable floor |
| CI fails on good changes | Threshold set at best-ever score | Lower it; alert on trend, not one run |
| LLM judge always agrees with the system | Judge is the same model grading itself | Use a different model; audit against human labels |
| Eval costs more than the product | Full LLM-judge suite on every commit | Smoke subset per commit, full suite nightly |
| Scores incomparable across weeks | Dataset changed underneath | Version datasets; new version = new name |
| Mean improved, users complain | Read the average, not the regressions | Compare per-example (§2.6) |

---

## 8. Check yourself

1. **Why is evaluation the difference between engineering and demo-building?**
   LLM changes trade off. Without measurement the trade is invisible, so quality drifts down one confident improvement at a time.

2. **Where should the dataset come from?**
   Real traces, especially failures. Invented questions are too clean to be predictive.

3. **When is LLM-as-judge appropriate, and what is its main failure mode?**
   For qualities no rule expresses, faithfulness, relevance, tone. It is unreliable on numbers and fine distinctions, and generous when grading its own model.

4. **Your change lifts the mean but three previously-passing examples now fail. Ship it?**
   Not on the mean alone. Look at what broke and decide deliberately, the mean is hiding the trade.

5. **Why version datasets?**
   Otherwise scores from different weeks are not comparable and the whole record becomes meaningless.

---

## 9. References

- LangSmith evaluation: https://docs.langchain.com/langsmith/evaluation
- Evaluating a graph: https://docs.langchain.com/langsmith/evaluate-graph
- API reference: https://reference.langchain.com

---

*Next: [Module 10: LangGraph](./langchain-010-langgraph.md), for the cases `create_agent` cannot express.*
