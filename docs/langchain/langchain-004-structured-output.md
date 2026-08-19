# Module 4 — Structured Output

**Phase:** Core
**Prerequisites:** Modules 0–3
**Verified against:** `langchain` 1.3.14, `pydantic` 2.13.4, Python 3.12
**Estimated time:** 4–5 hours

---

## 1. Why this matters

Everything so far ends in prose that a human reads. The moment another *system* consumes the output — a database, an API, a UI that renders fields — prose is a liability.

The gap between a demo and a product is usually one question: does this return parseable, validated data **every time**, or only most of the time? "Most of the time" fails at 3am on the input nobody tested.

This module closes that gap, and it is the last piece of the core.

---

## 2. Concepts

### 2.1 Stop asking for JSON in the prompt

The instinct is to write *"Respond with JSON in this format: {...}"*. It mostly works, which is what makes it dangerous. It fails on markdown code fences, on a preamble ("Here's the JSON you asked for:"), on a trailing explanation, on a field the model decided to rename, and on any input that pushes it toward prose.

The result is defensive parsing — strip the fences, find the first `{`, `json.loads`, catch, retry, hope. That code exists in a great many production systems and it should not.

Use a schema instead. The constraint moves from a *request* into the *contract*.

### 2.2 Pydantic as the contract

```python
from pydantic import BaseModel, Field

class Contact(BaseModel):
    """A contact extracted from an enquiry email."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Their email address")
    company: str | None = Field(default=None, description="Company, if mentioned")
    wants_demo: bool = Field(description="True if they asked for a demo or call")
```

Three things this gives you that a prompt does not:

- **The field descriptions reach the model.** Same principle as tool docstrings in Module 2 — `Field(description=...)` is part of the interface, not a comment.
- **Validation happens in your process.** A missing or mistyped field raises where you can catch it, rather than surfacing three layers downstream.
- **Types are real.** `wants_demo` is a `bool`, not the string `"true"`.

Write descriptions for every non-obvious field. `name: str` with no description invites the model to guess whether that means full name, first name, or a username.

### 2.3 Two ways to get it

**On the model** — for a single extraction call, no tools:

```python
model = init_chat_model("anthropic:claude-opus-5")
extractor = model.with_structured_output(Contact)
contact = extractor.invoke("Hi, I'm Priya Raman from Acme...")
# -> Contact(name='Priya Raman', ...) — a validated instance
```

**On the agent** — when the agent may use tools first and *then* return structure:

```python
agent = create_agent(
    model="anthropic:claude-opus-5",
    tools=[lookup_crm],
    response_format=Contact,      # verified: create_agent accepts this; default None
)
```

Choose by asking whether tools are involved. Pure extraction from text you already have → `with_structured_output`. Needs to look something up first → `create_agent(response_format=...)`.

### 2.4 Optional is not a formality

The single most common schema bug is making everything required.

If `company` is required and the email does not mention one, the model must produce *something*. It will. It will invent a plausible company name, because you left it no legal way to say "absent."

**Every field that might genuinely be missing must be optional** (`str | None` with a default). This is not defensive coding; it is removing an incentive to hallucinate.

### 2.5 Validation, and retrying properly

Structured output guarantees the *shape*. It does not guarantee the *content* is right. `email` will be a string; whether it is a real address is your problem:

```python
from pydantic import field_validator

class Contact(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def must_look_like_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"not a valid email address: {v!r}")
        return v
```

When validation fails, **do not blindly retry the identical call.** Same input plus same prompt plus same schema tends to produce the same failure, and you have paid twice. Feed the error back so the retry has new information:

```python
try:
    contact = extractor.invoke(text)
except ValidationError as e:
    contact = extractor.invoke(
        f"{text}\n\nYour previous extraction failed validation: {e}. "
        "Correct only the invalid fields."
    )
```

Bound the retries. Two is usually plenty; if it fails twice the input probably does not contain the data, and the honest outcome is routing it to a human rather than looping.

### 2.6 Streaming caveat

If you stream a structured response, partially-populated arguments arrive mid-stream. Do not parse until the stream completes — a half-filled object validates as garbage or throws confusingly. Check for completeness first.

---

## 3. Walkthrough

```python
"""Module 4 — validated extraction with bounded, informed retry."""
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain.chat_models import init_chat_model


class Enquiry(BaseModel):
    """A sales enquiry extracted from an inbound email."""

    name: str = Field(description="Full name of the sender")
    email: str = Field(description="Sender's email address")
    company: str | None = Field(
        default=None, description="Company name, or null if not mentioned"
    )
    budget_inr: int | None = Field(
        default=None, description="Stated budget in INR, or null if not mentioned"
    )
    wants_demo: bool = Field(description="True if they asked for a demo or a call")
    urgency: str = Field(description="One of: low, medium, high")

    @field_validator("email")
    @classmethod
    def looks_like_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"not a valid email address: {v!r}")
        return v

    @field_validator("urgency")
    @classmethod
    def known_urgency(cls, v: str) -> str:
        if v.lower() not in {"low", "medium", "high"}:
            raise ValueError(f"urgency must be low/medium/high, got {v!r}")
        return v.lower()


extractor = init_chat_model("anthropic:claude-opus-5").with_structured_output(Enquiry)


def extract(text: str, max_attempts: int = 2) -> Enquiry | None:
    """Extract an Enquiry, feeding validation errors back on retry."""
    prompt = text
    for attempt in range(1, max_attempts + 1):
        try:
            return extractor.invoke(prompt)
        except ValidationError as e:
            print(f"   attempt {attempt} failed validation: {e.error_count()} error(s)")
            if attempt == max_attempts:
                return None            # route to a human, do not keep paying
            prompt = (
                f"{text}\n\nYour previous extraction failed validation:\n{e}\n"
                "Return the same data with only the invalid fields corrected."
            )


EMAILS = [
    # complete
    """Hi, I'm Priya Raman (priya@acmecorp.in) from Acme Corp. We're looking at
       your platform, budget around 8 lakh. Can we get a demo this week? Urgent.""",
    # sparse -- optional fields must come back as None, NOT invented
    """hey do you do bulk discounts? - rahul, rahul.k@gmail.com""",
]

for email in EMAILS:
    print(f"\n--- {email.strip()[:60]}...")
    result = extract(email)
    if result is None:
        print("   UNPARSEABLE -> human review queue")
    else:
        print("   ", result.model_dump())
```

---

## 4. Run it

```bash
.venv/bin/python extract.py
```

**Expected output — illustrative.** The structural checks matter, not the values:

```
--- Hi, I'm Priya Raman (priya@acmecorp.in) from Acme Corp...
    {'name': 'Priya Raman', 'email': 'priya@acmecorp.in',
     'company': 'Acme Corp', 'budget_inr': 800000,
     'wants_demo': True, 'urgency': 'high'}

--- hey do you do bulk discounts? - rahul, rahul.k@gmail.com...
    {'name': 'Rahul', 'email': 'rahul.k@gmail.com',
     'company': None, 'budget_inr': None,
     'wants_demo': False, 'urgency': 'low'}
```

Check four things. You get a `dict` from a validated object, not a string you parsed. `wants_demo` is a real `bool`. `urgency` is one of the three allowed values. And in the second email — **`company` and `budget_inr` are `None`, not invented.** If either came back with a made-up value, your fields are not properly optional, and you have just watched §2.4 happen.

---

## 5. Exercises

**5.1 Recall.** Why is a schema better than "respond with JSON" in the prompt? Give two reasons that are not about tidiness.

**5.2 Apply.** Make `company` required (`str`, no default) and re-run the sparse email. Record what the model puts there. Then revert. Two sentences on what this teaches about schema design.

**5.3 Extend.** Add `confidence: float = Field(ge=0, le=1)` and route anything below 0.7 to a review queue instead of accepting it. Then find an input where the model is confidently wrong, and write down what that implies about trusting self-reported confidence.

---

## 6. Assignment

An extraction pipeline over a folder of at least 20 unstructured documents (invoices, CVs, or support emails — real-shaped, not toy).

Requirements:

- A Pydantic schema with described fields and correctly optional ones
- At least two `field_validator`s enforcing real business rules
- Bounded retry that **feeds the validation error back**, capped at two attempts
- Failures routed to a `needs_review/` folder with the error recorded — never dropped, never guessed
- A summary report: extracted / retried-then-succeeded / sent to review

Plus a short written answer to: **"which fields did the model invent when you made them required, and how did you catch it?"** If your answer is "it didn't", your test inputs are too clean — go and find a sparse one.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Model invents values for missing data | Field is required | Make it `T \| None` with a default (§2.4) |
| `ValidationError` on every call | Schema too strict, or descriptions unclear | Loosen the type; describe the field better |
| Model returns JSON in a markdown fence | You asked in the prompt instead of using a schema | Use `with_structured_output` / `response_format` |
| Retry fails identically, twice the cost | Retrying the same call with no new information | Feed the error text back (§2.5) |
| Fields correct but semantically wrong | Shape is guaranteed, content is not | Add `field_validator`s |
| Enum-ish string field drifts (`"High"`, `"urgent"`) | Free `str` | `Literal["low","medium","high"]` or a validator |
| Parse errors when streaming | Partial args mid-stream | Wait for completion before parsing (§2.6) |
| Works on clean inputs, fails on real ones | Test data too tidy | Test on the messiest real documents you have |

---

## 8. Check yourself

1. **Why is a schema better than asking for JSON in the prompt?**
   The constraint becomes part of the contract rather than a request, the field descriptions reach the model, and you get validated typed objects instead of strings to parse defensively.

2. **`with_structured_output` or `create_agent(response_format=...)`?**
   The former for pure extraction from text you already have; the latter when the agent must use tools before it can produce the structure.

3. **A required `company` field on an email that names no company. What happens?**
   The model invents one. It has no legal way to express absence. Make it optional.

4. **Validation failed. Why not just retry?**
   Identical input, prompt, and schema tends to reproduce the failure at double the cost. Include the error so the retry has something new to work with.

5. **Structured output succeeded. Is the data correct?**
   No — only correctly shaped. `email` is a string; whether it is a real address is what validators are for.

---

## 9. References

- Structured output — https://docs.langchain.com/oss/python/langchain/structured-output
- Agents (`response_format`) — https://docs.langchain.com/oss/python/langchain/agents
- Pydantic — https://docs.pydantic.dev/latest/concepts/fields/

---

*End of Phase 1. You can now call models, define tools, run and debug an agent loop, and return validated data. Next: [Module 5 — Memory & State](./langchain-005-memory-and-state.md), where the agent stops forgetting you between turns.*
