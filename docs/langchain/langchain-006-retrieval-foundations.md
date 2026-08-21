# Module 6: Retrieval Foundations

**Phase:** Retrieval & Memory
**Prerequisites:** Modules 0-5
**Verified against:** `langchain` 1.3.14, `langchain-text-splitters` 1.1.2, Python 3.12
**Estimated time:** 6-8 hours

---

## 1. Why this matters

RAG systems fail here, not in the LLM.

When a document Q&A system answers badly, the instinct is to blame the model or rewrite the prompt. Almost always the real cause is upstream: the relevant chunk was never retrieved, because the document was split down the middle of the answer, or the embedding model does not understand your domain's vocabulary, or `k=2` was never going to be enough.

**No prompt fixes a retrieval failure.** If the right text is not in the context window, the model is being asked to guess. This module is about making sure it gets there.

Module 7 already covered *how the agent uses* retrieval. This module is about the pipeline underneath it.

---

## 2. Concepts

### 2.1 The pipeline

```
documents ──► load ──► split ──► embed ──► store ──► search ──► chunks
                        ▲                                ▲
                   most failures                 second most failures
                   originate here                originate here
```

Four stages, and the two marked stages are where your quality actually comes from.

### 2.2 Loading

Loaders turn files into `Document` objects, `page_content` plus `metadata`.

The important habit: **put real metadata on every document at load time.** `source` at minimum, plus whatever you will want to filter on later (department, date, document type, access level). Metadata is trivial to add now and painful to backfill after you have embedded 40,000 chunks.

Expect mess. Real PDFs produce fused words, lost table structure, headers repeated on every page, and multi-column text interleaved into nonsense. **Print the extracted text of your worst PDF before you build anything on top of it.** Ten minutes there saves a week of blaming the model.

### 2.3 Splitting, where the quality is decided

Embedding models have input limits, and more importantly, a chunk is the *unit of retrieval*. Retrieve a chunk and you get all of it and nothing outside it.

`RecursiveCharacterTextSplitter` is the default. Its separator list, verified:

```python
['\n\n', '\n', ' ', '']
```

It tries paragraph breaks first, then line breaks, then spaces, then raw characters, so it splits at the most natural available boundary rather than blindly at N characters.

**Overlap exists because boundaries land badly.** With `chunk_overlap=200`, consecutive chunks share 200 characters, so a sentence straddling a boundary appears whole in at least one of them. Zero overlap will eventually cut an answer in half.

Rough starting points, to be tuned against your corpus, not adopted on faith:

| Content | chunk_size | overlap |
|---|---|---|
| Short FAQs, policies | 500-800 | 100 |
| General prose, docs | 1000 | 200 |
| Dense technical reference | 1500-2000 | 300 |

Two failure directions: **too small** and a chunk loses the context that makes it meaningful; **too large** and one chunk contains four topics, so its embedding is an average of all of them and matches none well.

### 2.4 Structure-aware splitting

`langchain-text-splitters` ships 14 splitters. Character splitting is the fallback, not the best option:

```
CharacterTextSplitter          RecursiveCharacterTextSplitter
MarkdownHeaderTextSplitter     MarkdownTextSplitter
HTMLHeaderTextSplitter         HTMLSectionSplitter
HTMLSemanticPreservingSplitter RecursiveJsonSplitter
PythonCodeTextSplitter         LatexTextSplitter
JSFrameworkTextSplitter        ExperimentalMarkdownSyntaxTextSplitter
NLTKTextSplitter               KonlpyTextSplitter
```

If your content has structure, use it. `MarkdownHeaderTextSplitter` splits on headings and **puts the heading trail into metadata**: verified:

```
{'h1': 'Policy', 'h2': 'Leave'}     | 18 days.
{'h1': 'Policy', 'h2': 'Expenses'}  | INR 800/day.
```

Now every chunk knows which section it came from. That metadata improves citations and enables filtering, and you got it free from structure that was already in the file.

For code, `RecursiveCharacterTextSplitter` can split on language constructs. Python separators, verified:

```python
['\nclass ', '\ndef ', '\n\tdef ', '\n\n']
```

so functions and classes stay intact instead of being cut mid-body.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
)
```

### 2.5 Embeddings

An embedding turns text into a vector; similar meanings land near each other. That is what lets "how many holidays do I get" retrieve a document that says "18 days of paid leave" with no shared keywords.

Three things that matter in practice:

**You cannot mix models.** Vectors from different models are not comparable. Change the embedding model and you re-index everything. Choose deliberately, and record the choice.

**Domain vocabulary is the usual weak point.** General-purpose models handle general English well and may not distinguish your internal jargon, product codes, or abbreviations. Test with your real vocabulary before committing.

**Anthropic has no embeddings API.** See the syllabus, Appendix C. Local `sentence-transformers` for learning; Voyage, OpenAI or similar in production.

### 2.6 Vector stores and searching

For learning, `InMemoryVectorStore`. For persistence, Chroma or FAISS locally, pgvector or a hosted store in production. The interface is nearly identical, so swapping is a small change.

Search methods you actually use:

| Method | Returns | Use for |
|---|---|---|
| `similarity_search(q, k)` | documents | normal retrieval |
| `similarity_search_with_score(q, k)` | (doc, score) | thresholding, debugging |
| `max_marginal_relevance_search(q, k, fetch_k)` | documents | reducing near-duplicate results |

**`k` is a real decision.** Too low and the answer is not retrieved. Too high and you pay for tokens and dilute the context with noise. Tune it by measuring, starting around 4.

**MMR** fetches `fetch_k` candidates, then picks `k` that are relevant *and* different from one another. It helps when a corpus contains many near-identical passages. On a small, already-diverse corpus it returns the same thing plain similarity does, in testing on a 4-document corpus, both returned identical results. MMR earns its keep at scale, not in a demo.

**A trap:** `similarity_search_with_relevance_scores` exists on the base class but raises `NotImplementedError` on `InMemoryVectorStore`, not every store implements the normalisation it needs. Use `similarity_search_with_score` unless you have confirmed your store supports the other.

### 2.7 Scores are relative, not absolute

Measured scores on a 3-document corpus for the query *"how many holidays do I get"*:

```
0.3984  leave policy      <- correct
0.2285  expense policy
0.0578  parking policy
```

Two lessons. The correct document wins by a clear margin, which is the signal a threshold can use. But `0.3984` is not "40% relevant", the absolute number is an artefact of the embedding model and the corpus. **A threshold tuned for one model is meaningless for another.** Measure yours; never copy a number from a blog post.

And note that the parking policy, utterly irrelevant, was still returned. `similarity_search` always returns `k` results. This is the point Module 7 §2.4 makes about knowing when to say "I don't know".

---

## 3. Walkthrough

```python
"""Module 6: build and interrogate an index."""
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ---- 1. load -------------------------------------------------------------
# Metadata added at load time; backfilling it later is painful.
docs = []
for path in sorted(Path("corpus").glob("*.md")):
    docs.append(Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path), "doc_type": "policy"},
    ))
print(f"loaded {len(docs)} documents")

# ---- 2. split ------------------------------------------------------------
# Structure first: headings become metadata. Then size-cap anything still long.
header_splitter = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")])
size_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = []
for doc in docs:
    for section in header_splitter.split_text(doc.page_content):
        section.metadata.update(doc.metadata)      # keep source alongside headings
        chunks.extend(size_splitter.split_documents([section]))
print(f"split into {len(chunks)} chunks")
print(f"sample metadata: {chunks[0].metadata}")

# ---- 3. embed + store ----------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = InMemoryVectorStore(embedding=embeddings)
store.add_documents(chunks)
print(f"indexed {len(chunks)} chunks")

# ---- 4. interrogate ------------------------------------------------------
# Look at scores while learning. It is the only honest feedback on retrieval.
for query in ["how many holidays do I get", "parking", "quantum chromodynamics"]:
    print(f"\nquery: {query!r}")
    for doc, score in store.similarity_search_with_score(query, k=3):
        head = doc.metadata.get("h2") or doc.metadata.get("h1") or "-"
        print(f"  {score:.4f}  {doc.metadata['source']:24} [{head}]")
```

Note the two-stage split. Headings give you structure and metadata; the size splitter then caps any section that is still too long. Structure first, size second, doing it the other way round destroys the headings before you can use them.

---

## 4. Run it

Create a `corpus/` folder with two or three markdown files using `#` and `##` headings, then:

```bash
.venv/bin/python build_index.py
```

**Expected output, illustrative:**

```
loaded 3 documents
split into 9 chunks
sample metadata: {'h1': 'Leave Policy', 'h2': 'Accrual', 'source': 'corpus/leave.md', ...}
indexed 9 chunks

query: 'how many holidays do I get'
  0.3984  corpus/leave.md          [Accrual]
  0.2285  corpus/expenses.md       [Meals]
  0.0578  corpus/parking.md        [Bays]
```

Four checks. Chunk count is greater than document count. Metadata carries **both** the heading trail and `source`. The semantically correct document ranks first even with no shared keywords. And the nonsense query **still returns three results**, at conspicuously low scores, the §2.7 lesson, seen directly.

---

## 5. Exercises

**5.1 Recall.** Why does `chunk_overlap` exist, and what breaks at zero?

**5.2 Apply.** Index the same corpus at `chunk_size` 300, 1000, and 3000. Run ten fixed questions against each and record which returns the correct chunk first. Write a paragraph on what you observed at each extreme, do not just report the winner.

**5.3 Extend.** Add metadata filtering so a query can be restricted to one `doc_type`. Then measure: does filtering improve results, or just narrow them? Show numbers.

---

## 6. Assignment, the chunking bake-off

Build an indexed corpus of at least 30 real documents (internal docs or a public dataset, not toy files), then produce a written comparison.

Requirements:

- A fixed evaluation set of **at least 15 questions with known correct source documents**, written *before* you tune anything
- At least three configurations compared, varying chunk size and/or splitter
- A results table reporting, per configuration: how often the correct document was retrieved in the top 1, and in the top 3
- A stated recommendation with the reasoning
- A `RETRIEVAL.md` recording your embedding model, chunk settings, `k`, and any threshold, with the date

Writing the questions first is the discipline being taught. Questions written afterwards get shaped, unconsciously, to flatter the configuration you already like.

This eval set is reused in Module 9, so keep it.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Correct document exists but is never retrieved | Chunking split the answer, or `k` too low | Increase overlap; raise `k`; try structure-aware splitting |
| Retrieved chunk is topically right but lacks the answer | `chunk_size` too small | Increase it, or increase overlap |
| Results feel vaguely related but never precise | `chunk_size` too large, one chunk, many topics | Reduce it; split on structure |
| Garbage chunks, fused words, no structure | PDF extraction | Inspect the extracted text; try another loader |
| `NotImplementedError` from `similarity_search_with_relevance_scores` | Not implemented by this store | Use `similarity_search_with_score` (§2.6) |
| Irrelevant results returned for nonsense queries | `similarity_search` always returns `k` | Expected. Threshold on score (§2.7) |
| Threshold copied from a blog does nothing sensible | Scores are model- and corpus-specific | Measure your own |
| Results changed after switching embedding model | Vectors are not comparable across models | Re-index everything |
| Domain jargon retrieves badly | General-purpose embedding model | Test with real vocabulary; consider a domain model |
| Everything lost on restart | `InMemoryVectorStore` | Chroma, FAISS, or pgvector |

---

## 8. Check yourself

1. **Why does no prompt fix a retrieval failure?**
   If the chunk is not retrieved, the text is not in the context window. The model is guessing, and no instruction changes what it was given.

2. **What does `chunk_overlap` protect against?**
   A sentence or fact landing across a chunk boundary and being truncated in both. Overlap makes it whole in at least one.

3. **You split markdown with a plain character splitter. What did you lose?**
   The heading structure, both as split boundaries and as metadata you could have cited and filtered on.

4. **A retrieval scores 0.31. Is that good?**
   Unanswerable in isolation. Scores are relative to your embedding model and corpus. Compare against a known-irrelevant baseline.

5. **When does MMR earn its place?**
   When the corpus holds many near-duplicate passages and plain similarity returns `k` versions of one thing. On a small diverse corpus it changes nothing.

---

## 9. References

- Retrieval: https://docs.langchain.com/oss/python/langchain/retrieval
- Text splitters: https://docs.langchain.com/oss/python/langchain/text-splitters
- Vector stores: https://reference.langchain.com

---

*Next: [Module 7: Agentic RAG](./langchain-007-agentic-rag.md). You have the index; now the agent decides when to reach for it, and you meet the security problem that comes with it.*
