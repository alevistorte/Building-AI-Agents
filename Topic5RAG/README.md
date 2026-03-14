# Week 05 — Retrieval-Augmented Generation (RAG)

RAG pipeline built from scratch using **Qwen 2.5 1.5B**, **FAISS**, and **sentence-transformers**.
Corpus: Model T Ford manual, Congressional Record (Jan 2026), Learjet ATA manual, EU AI Act.

---

## Table of Contents

| #   | File                                    | Topic                        | Corpus                 |
| --- | --------------------------------------- | ---------------------------- | ---------------------- |
| —   | `manual_rag_pipeline_universal.ipynb`   | Base pipeline (reference)    | Any                    |
| 1   | `task1_rag_vs_no_rag.ipynb`             | RAG vs. No RAG               | Model T + Cong. Record |
| 2   | `task2_open_model_vs_gpt4o.ipynb`       | Qwen+RAG vs. GPT-4o Mini     | Model T + Cong. Record |
| 3   | `task3_rag_vs_frontier.ipynb`           | Local RAG vs. Frontier Model | Model T + Cong. Record |
| 4   | `task4_top_k_retrieval.ipynb`           | Effect of Top-K              | Model T                |
| 5   | `task5_unanswerable.ipynb`              | Unanswerable Questions       | Model T                |
| 6   | `task6_query_phrasing.ipynb`            | Query Phrasing Sensitivity   | Model T                |
| 7   | `task7_chunk_overlap.ipynb`             | Chunk Overlap Experiment     | Model T                |
| 8   | `task8_chunk_size.ipynb`                | Chunk Size Experiment        | Model T                |
| 9   | `task9_retrieval_scores.ipynb`          | Retrieval Score Analysis     | Model T                |
| 10  | `task10_prompt_variations.ipynb`        | Prompt Template Variations   | Model T                |
| 11  | `task11_cross_document_synthesis.ipynb` | Cross-Document Synthesis     | Model T                |

---

## Corpora

```
Corpora/
├── ModelTService/              ← Model T Ford repair manual (PDF + TXT)
├── NewModelT/                  ← Extended Model T manual
├── Congressional_Record_Jan_2026/  ← CR issues Jan 2–30, 2026
├── Learjet/                    ← Learjet ATA chapters
└── EU_AI_Act/                  ← EU Artificial Intelligence Act
```

---

## Pipeline Overview

```
Documents → Chunking (512 chars, 128 overlap)
                 → Embedding (all-MiniLM-L6-v2)
                       → FAISS index (cosine similarity)
                                ↓
User Query → Embed → Top-K Search → Prompt → Qwen 2.5 1.5B → Answer
```

---

## Exercise Descriptions

### Exercise 1 — RAG vs. No RAG (`task1_rag_vs_no_rag.ipynb`)

Run the same 8 queries (4 Model T, 4 Congressional Record) both directly to the LLM
and through the RAG pipeline. Documents whether the model hallucinates specific
values (spark plug gaps, oil specs) without context, and whether RAG grounds it.

```python
# Core comparison loop
def compare_rag(questions, label=""):
    for q in questions:
        ans_direct = direct_query(q)          # No context
        ans_rag    = rag_query(q, top_k=5)    # With retrieved chunks
```

**Results (Qwen 2.5 1.5B on Colab T4, combined corpus index):**

| Query                              | Without RAG                                                             | With RAG                                                                       | Verdict                           |
| ---------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------- |
| Carburetor adjust                  | ❌ Generic modern steps (air filter, idle screw) — not Model T-specific | ✅ Grounded: adjusting rod through dash slot, cotter key, inlet needle seating | RAG wins                          |
| Spark plug gap                     | ⚠️ 0.5 mm with fake Quora citation                                      | ⚠️ Said 7/8" (wrong chunk — likely a wrench size); added "thickness of a dime" | Both unreliable                   |
| Transmission band                  | ❌ Generic ATF lubrication steps — not Model T-specific                 | ✅ Grounded: lock nut, adjusting screw, transmission cover — Model T-specific  | RAG wins                          |
| Engine oil                         | ❌ Hallucinated "No. 2 light diesel fuel oil"                           | ❌ Cross-corpus hit: retrieved Learjet GE CJ60 manual instead of Model T       | Both wrong — corpus mixing        |
| Mr. Flood / Black (CR Jan 13)      | ❌ Invented a fake Daily Show transcript                                | ✅ Correct: praised Mayor Black (NE), 17 years service, city's economic growth | RAG wins                          |
| Stefanik mistake (CR Jan 23)       | ❌ Invented a climate change speech                                     | ❌ Retrieved Jan 6 insurrection content, fabricated violent acts               | Both wrong — retrieved wrong date |
| Main Street Parity Act (CR Jan 20) | ⚠️ Partially wrong: 2018, collateral-free lending                       | ✅ Correct: 10% equity, plant acquisition loans, 504 program alignment         | RAG wins                          |
| Pregnancy centers (CR Jan 21)      | ⚠️ No names, generic framing                                            | ✅ Named Ms. Dexter (OR, against) and Mr. Schneider (for) with direct quotes   | RAG wins                          |

**Key findings:**

- Without RAG: model consistently fabricates plausible-sounding but wrong specifics (fake citations, wrong oil type, invented speeches).
- With RAG: grounding works well when the right chunk is retrieved, but a combined corpus causes cross-document retrieval failures (engine oil → Learjet manual; Stefanik → wrong CR date).
- **Lesson:** Use corpus-specific indexes (not a combined root) for precision.

---

### Exercise 2 — Qwen+RAG vs. GPT-4o Mini (`task2_open_model_vs_gpt4o.ipynb`)

Adds OpenAI GPT-4o Mini as a third comparison. Evaluates whether a larger cloud
model with no retrieval can match a small local model with RAG.

```python
def gpt4o_mini_query(question):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=512, temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
```

> Reads `OPENAI_API_KEY` from `.env` file on Drive (`Colab_Projects/Week05-RAG/.env`).

**Results (Qwen 2.5 1.5B + RAG vs. GPT-4o Mini, combined corpus, Colab T4):**

| Query                              | Qwen+RAG                                                      | GPT-4o Mini                                           | Verdict                       |
| ---------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------- |
| Carburetor adjust                  | ✅ Grounded: throttle rod, cotter pin, needle valve           | ⚠️ Generic modern steps, not Model T-specific         | Qwen+RAG wins                 |
| Spark plug gap                     | ⚠️ Said 7/16" (wrong chunk again); mentioned "dime thickness" | ✅ Correct: 0.025" (0.635 mm)                         | GPT-4o Mini wins              |
| Transmission band                  | ✅ Grounded: lock nut, adjusting screw, cover removal         | ❌ Generic ATF fluid advice, modern auto transmission | Qwen+RAG wins                 |
| Engine oil                         | ❌ Cross-corpus: retrieved Learjet manual again               | ✅ Correct: non-detergent 30W, straight mineral oil   | GPT-4o Mini wins              |
| Mr. Flood / Black (CR Jan 13)      | ✅ Correct: Papillion NE, 20+ years, not re-running           | ❌ Correctly admits: "training cutoff Oct 2023"       | Qwen+RAG wins                 |
| Stefanik mistake (CR Jan 23)       | ❌ Fabricated Jan 6 content + quoted raw PDF metadata         | ✅ Correctly admits: "training cutoff Oct 2023"       | GPT-4o Mini wins by declining |
| Main Street Parity Act (CR Jan 20) | ✅ Correct: 10% equity, plant loans, 504 alignment            | ❌ Confabulated a different fintech/banks parity act  | Qwen+RAG wins                 |
| Pregnancy centers (CR Jan 21)      | ✅ Named Ms. Dexter and Mr. Schneider with quotes             | ⚠️ General: cites Warren/Sanders, no Jan 2026 details | Qwen+RAG wins                 |

**Score: Qwen+RAG 5 · GPT-4o Mini 2 · 1 mixed**

**Key findings:**

- GPT-4o Mini correctly refuses post-cutoff CR questions (Oct 2023 cutoff vs. Jan 2026 CR) — safer failure mode than RAG hallucinating Jan 6 content.
- GPT-4o Mini wins on well-documented historical specs (spark plug gap, oil type) where its training data is sufficient and RAG retrieval kept hitting wrong chunks.
- For the Main Street Parity Act, GPT-4o Mini confidently described a completely different law — showing that larger models can also confabulate plausibly.
- **Core lesson:** RAG beats a larger model when content is post-cutoff or too domain-specific for training data. The larger model wins when its training data covers the topic and RAG retrieval is noisy.

---

### Exercise 3 — Local RAG vs. Frontier Model (`task3_rag_vs_frontier.ipynb`)

Runs Qwen+RAG locally and compares against GPT-4o Mini (API, no tools). Explores
where frontier knowledge succeeds or fails vs. targeted RAG.

**Results (same queries as Ex 1–2, combined corpus, Colab T4):**

| Query                  | Qwen+RAG                                      | GPT-4o Mini                           | Who wins?   |
| ---------------------- | --------------------------------------------- | ------------------------------------- | ----------- |
| Carburetor adjust      | ✅ 5 specific steps from manual               | ⚠️ Generic modern steps               | Qwen+RAG    |
| Spark plug gap         | ⚠️ Said 0.74" (wrong chunk); mentioned "dime" | ✅ Correct: 0.025"                    | GPT-4o Mini |
| Transmission band      | ✅ Lock nut, adjusting screw, cover           | ❌ Generic ATF advice                 | Qwen+RAG    |
| Engine oil             | ❌ Retrieved Learjet grease (Mobil No. 29)    | ✅ Correct: non-detergent 30W         | GPT-4o Mini |
| Mr. Flood / Black      | ✅ Correct: Papillion NE, 17.5 yrs, retiring  | ❌ Admits: post-cutoff (Oct 2023)     | Qwen+RAG    |
| Stefanik mistake       | ❌ Retrieved Jan 22 CR / Jan 6 content        | ✅ Admits: post-cutoff (Oct 2023)     | GPT-4o Mini |
| Main Street Parity Act | ✅ Correct + bipartisan, Rep. Williams (TX)   | ❌ Invented community-bank parity act | Qwen+RAG    |
| Pregnancy centers      | ✅ Named Ms. Dexter and Mr. Schneider         | ⚠️ General historical framing only    | Qwen+RAG    |

**Score: Qwen+RAG 4 · GPT-4o Mini 3 · 1 mixed**

**When RAG adds value vs. when frontier suffices:**

| Scenario                          | RAG                           | Frontier alone                |
| --------------------------------- | ----------------------------- | ----------------------------- |
| Post-training-cutoff content      | ✅ Essential                  | ❌ Cannot know                |
| Rare/specialized/proprietary docs | ✅ Essential                  | ❌ May confabulate            |
| Well-documented historical facts  | ⚠️ Only if retrieval is clean | ✅ Training data sufficient   |
| Large noisy combined corpus       | ⚠️ Retrieval noise hurts      | ✅ No retrieval noise         |
| Verbatim quotes / exact figures   | ✅ RAG grounds precisely      | ❌ May paraphrase incorrectly |

**Key finding:** GPT-4o Mini used no live web search (API, no tools) — it explicitly cited "training cutoff Oct 2023" for all CR questions. The Stefanik failure reveals a subtle retrieval bug: the Jan 22 CR contained speeches about her Jan 6 actions, which are semantically similar to "mistake in Congress on Jan 23" — a date-filtered index would fix this.

---

### Exercise 4 — Effect of Top-K (`task4_top_k_retrieval.ipynb`)

Runs 5 Model T queries at k = 1, 3, 5, 10, 20. Records answer quality, completeness,
accuracy, and response latency. Also prints raw retrieval scores to visualize the
score gap between ranks.

```python
K_VALUES = [1, 3, 5, 10, 20]
for k in K_VALUES:
    answer = rag_query(question, top_k=k)
```

**Results (Qwen 2.5 1.5B, ModelTService corpus, Colab T4):**

Answer quality across k values:

| Query             | k=1                          | k=3                            | k=5                                             | k=10                                           | k=20                              |
| ----------------- | ---------------------------- | ------------------------------ | ----------------------------------------------- | ---------------------------------------------- | --------------------------------- |
| Carburetor adjust | ❌ Vague (install new one)   | ✅ Adjusting rod, slot, needle | ✅ Most detailed: rod, slot, needle, cotter pin | ✅ Step-by-step: dashboard rod, throttle lever | ⚠️ Correct but verbose            |
| Spark plug gap    | ❌ No specific value         | ✅ 7/8\" (~dime thickness)     | ✅ 7/8\" — most concise                         | ✅ 7/4\" (OCR artifact)                        | ✅ 7/16\" (OCR artifact)          |
| Transmission band | ✅ Lock nut, adjusting screw | ✅ Same + cover removal        | ✅ Same, cleaner phrasing                       | ✅ Concise 3-step answer                       | ✅ Most complete: all three bands |
| Engine oil        | ❌ No specific oil named     | ✅ Mobil Oil C + Whittemore's  | ❌ No specific oil named                        | ✅ Mobil Oil C + Whittemore's                  | ✅ Mobil Oil C + Whittemore's     |
| Engine timing     | ⚠️ Bendix shaft only (wrong) | ⚠️ Same Bendix content         | ⚠️ Same Bendix content                          | ⚠️ Same Bendix content                         | ❌ Hallucinated timing belt       |

Latency (seconds):

| Query         | k=1  | k=3  | k=5  | k=10 | k=20 |
| ------------- | ---- | ---- | ---- | ---- | ---- |
| Carburetor    | 24.8 | 8.4  | 13.5 | 9.9  | 13.0 |
| Spark plug    | 14.9 | 10.0 | 4.2  | 4.4  | 5.6  |
| Transmission  | 7.2  | 8.1  | 9.2  | 4.4  | 9.5  |
| Engine oil    | 8.6  | 8.8  | 5.7  | 4.0  | 5.2  |
| Engine timing | 6.0  | 6.2  | 5.5  | 6.8  | 5.8  |

Retrieval score range for "carburetor" query:

| k   | Top score | Min score | Mean score |
| --- | --------- | --------- | ---------- |
| 1   | 0.640     | 0.640     | 0.640      |
| 3   | 0.640     | 0.577     | 0.598      |
| 5   | 0.640     | 0.567     | 0.586      |
| 10  | 0.640     | 0.534     | 0.565      |
| 20  | 0.640     | 0.515     | 0.546      |

**Key findings:**

- **Sweet spot: k=5.** For most queries it delivers the best answer quality with reasonable latency. k=3 is nearly as good and faster; k=10 adds value mainly for synthesis queries. k=1 is risky — one missed chunk leaves the model with no context.
- **k=20 failure (engine timing).** The extra low-scoring chunks (~0.515) introduced modern-engine content ("timing belt tensioner bolt") that a 1919 Model T manual cannot contain. The model treated this hallucinated chunk as ground truth, producing a completely wrong answer.
- **Non-monotonic quality.** Engine oil answer quality at k=3 > k=5 because k=3 happens to retrieve two correct Model-T-specific chunks, while k=5 adds a noisier chunk that dilutes the prompt. More context is not always better.
- **OCR artifacts are a corpus issue, not a k-tuning issue.** Spark plug gap readings ("74", "7/4", "7/16") vary across k values solely because different k values surface different OCR-garbled chunk versions of the same source sentence. Increasing k cannot fix bad OCR.
- **Latency is driven by output tokens, not input context.** The carburetor k=1 answer is longest (2,653 chars, 24.8 s) while k=3 with more context is only 1,076 chars (8.4 s) — the model hedges more when context is thin.

---

### Exercise 5 — Unanswerable Questions (`task5_unanswerable.ipynb`)

Tests three failure categories:

| Type                | Example                                        |
| ------------------- | ---------------------------------------------- |
| Off-topic           | "What is the capital of France?"               |
| Related-but-missing | "What is the horsepower of a 1925 Model T?"    |
| False premise       | "Why does the manual recommend synthetic oil?" |

Also experiments with a strict prompt that instructs the model to say
`"I cannot answer this from the available documents."` when context is insufficient.

```python
STRICT_PROMPT = """...If the context does not contain sufficient information,
you MUST respond with: "I cannot answer this from the available documents."..."""
```

**Results (Qwen 2.5 1.5B, ModelTService corpus, standard vs. strict prompt):**

| Question          | Type            | Admits ignorance? | Hallucination?                                             | Strict prompt better?   |
| ----------------- | --------------- | ----------------- | ---------------------------------------------------------- | ----------------------- |
| Capital of France | off-topic       | No                | Yes — "implied by mention of France" in Model T text       | Yes — correctly refused |
| 1925 HP           | related-missing | Yes               | No — correctly said "no mention in context"                | Marginal                |
| Synthetic oil     | false-premise   | No                | Yes — accepted false premise, quoted unrelated oil passage | Yes — correctly refused |

**Key findings:**

- **Off-topic questions** are most vulnerable: the model force-fits irrelevant context onto keywords ("France" appears in Model T text → "Paris implied"). Retrieved context actively encourages hallucination.
- **Related-but-missing** questions fare better: similar vocabulary (motor, crankcase, transmission) helps the model recognize the answer is absent.
- **False-premise questions** are subtle: the model accepts "synthetic oil" and "fuel injection" as given, then cherry-picks unrelated passages to construct plausible-sounding but wrong answers.
- **Strict `"I cannot answer"` prompt** reliably suppresses hallucination across all three categories.
- **Lesson:** Retrieved context ≠ relevant context. Explicit refusal instructions are the simplest guard; a score threshold filter is the next layer.

---

### Exercise 6 — Query Phrasing Sensitivity (`task6_query_phrasing.ipynb`)

Phrases the same underlying question 5–6 different ways (formal, casual, keywords,
question form, indirect). Computes Jaccard overlap between retrieved chunk sets
across phrasings to measure retrieval stability.

```python
def jaccard(set_a, set_b):
    a = set(t for t, _ in set_a)
    b = set(t for t, _ in set_b)
    return len(a & b) / len(a | b)
```

**Results (carburetor topic, top-5 retrieval, ModelTService corpus):**

| Phrasing                                                           | Top Score | Cluster                           |
| ------------------------------------------------------------------ | --------- | --------------------------------- |
| "What is the recommended maintenance schedule for the carburetor?" | 0.415     | Maintenance (drainage chunks)     |
| "How often should I service the carburetor?"                       | 0.497     | Maintenance                       |
| "carburetor maintenance intervals"                                 | 0.501     | Maintenance                       |
| "When do I need to check the carburetor?"                          | 0.569     | Maintenance                       |
| "Preventive maintenance requirements for carburetor"               | 0.515     | Maintenance                       |
| **"How do I adjust the carburetor on a Model T?"**                 | **0.640** | **Adjustment (procedure chunks)** |

Jaccard between all maintenance phrasings ≈ 0.50; Jaccard between any maintenance phrasing and "adjust" phrasing = **0.00** — entirely different chunk sets.

For spark plug: "ignition plug clearance" also retrieved completely different chunks (Jaccard 0.00) vs. natural-language phrasings like "spark plug gap setting" (0.598).

**Key findings:**

- A single phrasing can miss 100% of the most relevant chunks. Action verbs ("adjust") vs. maintenance verbs ("service", "check") land in completely different embedding regions.
- Keyword-style queries perform comparably to natural-language questions _within the same semantic cluster_ but don't escape it.
- **Lesson:** Generate 2–3 phrasings with different action verbs and union retrieved sets (multi-query RAG). HyDE or domain-prefixing also help anchor queries to manual language.

---

### Exercise 7 — Chunk Overlap Experiment (`task7_chunk_overlap.ipynb`)

Fixes chunk size at 512 and varies overlap: **0, 64, 128, 256**. Rebuilds the
full pipeline for each setting and tests questions whose answers span chunk
boundaries. Measures index growth and retrieval quality.

```python
for overlap in [0, 64, 128, 256]:
    rebuild_pipeline(chunk_size=512, chunk_overlap=overlap)
    answer = rag_query(boundary_question, top_k=5)
```

> ⚠️ Takes ~15–30 min on T4. Run on Colab GPU.

**Results (Qwen 2.5 1.5B, chunk_size=512, ModelTService corpus):**

| Overlap | # Chunks | Index Size | Carburetor adjust                  | Transmission bands            | Long-storage maintenance              |
| ------- | -------- | ---------- | ---------------------------------- | ----------------------------- | ------------------------------------- |
| 0       | 888      | 1.0×       | ✅ 5-step answer                   | ⚠️ Partial (drum positioning) | ⚠️ Generic (inspect/clean)            |
| **64**  | **1051** | **1.18×**  | **✅ 5-step answer**               | **✅ Lock nut + cover**       | **✅ Best (drain, kerosene, refill)** |
| 128     | 1286     | 1.45×      | ⚠️ Vague ("refer to instructions") | ✅ Lock nut + adjusting screw | ⚠️ Generic                            |
| 256     | 2016     | 2.27×      | ✅ 5-step answer                   | ✅ Lock nut + cover           | ⚠️ Generic                            |

**Key findings:**

- overlap=64 is the sweet spot — recovered boundary-spanning long-storage procedure missed at overlap=0, maintained carburetor quality.
- overlap=128 (the previous default) _degraded_ carburetor quality — redundant overlapping chunks diluted the prompt.
- overlap=256 costs 2.27× index size with no consistent improvement over overlap=64.
- **Most impactful jump:** 0 → 64. Beyond 64, diminishing returns and increasing redundancy.

---

### Exercise 8 — Chunk Size Experiment (`task8_chunk_size.ipynb`)

Varies chunk size: **128, 512, 2048** (overlap fixed at 64). Rebuilds the pipeline
and runs 5 standard queries. Reports top retrieval scores, number of chunks, and
answer completeness for each size.

```python
for cs in [128, 512, 2048]:
    rebuild_pipeline(chunk_size=cs, chunk_overlap=64)
```

> ⚠️ Takes ~15–30 min on T4. Run on Colab GPU.

**Results (Qwen 2.5 1.5B, overlap=64, ModelTService corpus):**

| Chunk Size | # Chunks | Avg Top Score | Best for                                                         |
| ---------- | -------- | ------------- | ---------------------------------------------------------------- |
| 128        | 7265     | 0.588         | Precise term matching — but chunks too small for full procedures |
| **512**    | **1051** | **0.529**     | **Procedural how-to questions (sweet spot)**                     |
| 2048       | 228      | 0.481         | Broad/synthesis questions (oil type, timing alignment)           |

Answer quality by query:

| Query             | 128                                 | 512                                  | 2048                                                      |
| ----------------- | ----------------------------------- | ------------------------------------ | --------------------------------------------------------- |
| Carburetor adjust | ⚠️ Vague (only "find max speed")    | ✅ Full 5-step procedure             | ✅ Correct + cold-start context                           |
| Spark plug gap    | ❌ Pack-nut chunk, no gap value     | ⚠️ "7/8 inch / dime thickness" (OCR) | ❌ "1/8 inch" (wrong chunk mixed in)                      |
| Transmission band | ✅ Lock nut (68-char chunk)         | ✅ Lock nut + adjusting screw        | ✅ Same                                                   |
| Engine oil        | ❌ Retrieved ignition text          | ⚠️ Mobiloil C (partial)              | ✅ "Medium light, high-grade gasoline oil into crankcase" |
| Engine timing     | ❌ Confused with transmission screw | ⚠️ Time gear/camshaft found          | ✅ Full alignment procedure                               |

**Key findings:**

- chunk_size=128 has highest cosine scores but chunks are too small to contain complete procedures — precision without recall.
- chunk_size=512 is the sweet spot for procedural questions matching the Model T manual's Q&A format.
- chunk_size=2048 wins for broad lookups (oil type, timing) where the answer is buried in multi-paragraph running-in instructions.
- **Optimal chunk size is question-type dependent:** point-lookup → 128–256; procedure → 512; synthesis → 1024–2048.

---

### Exercise 9 — Retrieval Score Analysis (`task9_retrieval_scores.ipynb`)

Retrieves top-10 chunks for 10 different queries and tabulates scores. Identifies
queries with a clear winner (large score gap between rank 1 and 2) vs. ambiguous
clusters. Experiments with score thresholds (0.3–0.6) to filter irrelevant chunks.

```python
results = retrieve(q, top_k=10)
scores = [s for _, s in results]
gap    = scores[0] - scores[1]   # "clear winner" metric
```

**Results (top-10 retrieval, ModelTService corpus, all-MiniLM-L6-v2):**

| Query                   | #1 Score | Gap(1-2)  | Spread | Pattern                                 |
| ----------------------- | -------- | --------- | ------ | --------------------------------------- |
| Carburetor adjust       | 0.640    | **0.063** | 0.106  | Only clear winner                       |
| Slipping transmission   | 0.601    | 0.000     | 0.111  | High score, top-2 tied                  |
| Valve clearance         | 0.585    | 0.000     | 0.106  | Tightly clustered                       |
| Spark plug gap          | 0.553    | 0.000     | 0.063  | Moderate, no winner                     |
| Flywheel remove/install | 0.541    | 0.000     | 0.015  | Near-zero spread                        |
| Engine oil              | 0.397    | 0.000     | 0.034  | Low score — vocabulary mismatch         |
| Water pump              | 0.355    | 0.000     | 0.050  | Lowest — likely absent from 1919 manual |

Overall distribution: Min=0.305, Median=0.504, Mean=0.485, Max=0.640.

**Threshold experiment (carburetor query):**

| Threshold | Chunks kept | Outcome                                               |
| --------- | ----------- | ----------------------------------------------------- |
| 0.3       | 10          | No filtering — all chunks score ≥ 0.53                |
| 0.4       | 10          | Same — threshold too low                              |
| 0.5       | 10          | Same — unhelpful                                      |
| 0.6       | 1           | Over-aggressive — single chunk, risks missing context |

**Key findings:**

- Gap(1-2) = 0.000 for 9/10 queries — MiniLM cosine similarity does **not** produce well-separated winners on this corpus. Hard score thresholds are largely ineffective.
- Low top scores (water pump: 0.355, engine oil: 0.397) indicate vocabulary mismatch — 1919 manual uses "oiling system" and "cooling apparatus" rather than modern terminology.
- **Better alternatives to hard thresholds:** relative threshold (keep chunks within 15% of top score); use Gap > 0.05 as a "confident retrieval" flag; rely on prompt grounding to handle irrelevant context.

---

### Exercise 10 — Prompt Template Variations (`task10_prompt_variations.ipynb`)

Tests 5 prompt styles on the same 5 queries:

| Template           | Description                       |
| ------------------ | --------------------------------- |
| `minimal`          | Context + Q + A, no instructions  |
| `strict_grounding` | "Answer ONLY from context"        |
| `citation`         | "Quote exact passages"            |
| `permissive`       | "Use context + general knowledge" |
| `structured`       | List facts first, then synthesize |

```python
answer = rag_query(q, top_k=5, prompt_template=TEMPLATES["citation"])
```

**Results (Qwen 2.5 1.5B, top-5 retrieval, ModelTService corpus):**

| Template         | Accuracy | Groundedness | Helpfulness | Citation Quality | Best for                                                 |
| ---------------- | -------- | ------------ | ----------- | ---------------- | -------------------------------------------------------- |
| Minimal          | 3/5      | 3/5          | 3/5         | 1/5              | Fast baseline; good when context quality is high         |
| Strict grounding | 3/5      | 5/5          | 2/5         | 2/5              | Minimizing hallucination; low-quality retrieval contexts |
| **Citation**     | **4/5**  | **5/5**      | **4/5**     | **5/5**          | **Best overall — transparent, auditable**                |
| Permissive       | 3/5      | 2/5          | 4/5         | 1/5              | Topics well-covered in training data (spark plug gap)    |
| Structured       | 3/5      | 3/5          | 3/5         | 2/5              | Complex synthesis queries                                |

Per-query highlights:

- **Carburetor**: Citation quoted "Carburetor adjustment is placed on dash, one-quarter turn to the right" — most accurate.
- **Spark plug gap**: Permissive won — drew on training knowledge to compensate for OCR-garbled corpus values ("74", "7/16"). Strict grounding locked onto the garbled value.
- **Transmission band**: Citation best — verbatim quote from manual, all templates adequate.
- **Engine oil / timing**: All templates failed equally — wrong chunks retrieved; prompt engineering cannot fix retrieval errors.

**Key findings:**

- Citation is the best default for corpus-specific procedures: explicit quoting makes errors visible and prevents confabulated steps.
- Permissive helps when corpus coverage is incomplete and training data is reliable, but risks undetectable blending of sources.
- There is a real helpfulness–groundedness trade-off: Strict grounding reduces hallucination but increases refusals on imperfect context.

---

### Exercise 11 — Cross-Document Synthesis (`task11_cross_document_synthesis.ipynb`)

Tests questions that require combining information scattered across multiple chunks
(e.g., "all maintenance tasks", "all tools needed for a tune-up"). Varies k = 3, 5, 10
and reports source diversity and answer completeness.

```python
results = retrieve(q, top_k=k)
sources = set(c.source_file for c, _ in results)
print(f"Unique sources: {sources}")
```

**Results (Qwen 2.5 1.5B, ModelTService corpus):**

| Query                            | k=3                                          | k=5                                 | k=10                                 | Missing info?                                 |
| -------------------------------- | -------------------------------------------- | ----------------------------------- | ------------------------------------ | --------------------------------------------- |
| All maintenance tasks            | ⚠️ Partial (hallucinated oil filter)         | ❌ Too meta (Ford org statement)    | ✅ 6 concrete tasks from 6 sources   | k=3/5 miss running gear, spark plugs          |
| Carb vs. transmission comparison | ⚠️ Confused (foot pedal = transmission adj.) | ⚠️ Vague (dash knob only)           | ⚠️ Slightly better, still simplified | All k miss step-by-step comparison            |
| Tune-up tool list                | ⚠️ Basic (4 tools)                           | ✅ 7 tools (5 unique sources)       | ✅ 8+ tools (7 unique sources)       | Hallucinated multimeter & Allen keys at all k |
| Safety warnings summary          | ✅ 2 categories                              | ✅ 3 categories                     | ✅ 3+ categories + electrical detail | All k miss some warnings                      |
| Lubrication points               | ⚠️ Differential only                         | ✅ + commutator (200-mile interval) | ✅ Same as k=5 (capped at 4 sources) | Many points never retrieved                   |

**Key findings:**

- **k=10 wins for synthesis queries** — wider source diversity captures more of the scattered information.
- **Duplicate TXT+OCR structure** limits true source diversity: at k=10 for maintenance tasks, 4 of 6 "unique" sources were txt/pdf duplicates of the same content — effectively only 3 distinct documents.
- **Retrieval ceiling exists:** Lubrication points hit a ceiling at k=5 (only 4 unique sources regardless of k increasing to 10) — relevant chunks scored too low (~0.38) to surface.
- **Hallucination in synthesis context:** The model added multimeter and Allen keys (not in a 1919 manual) when generating tool lists — blending general knowledge with retrieved context.
- **Optimal k:** k=3–5 for targeted lookup; k=10–20 for synthesis (even then, comprehensive coverage is not guaranteed).
- **Core lesson:** RAG is better suited for targeted retrieval than exhaustive synthesis. For comprehensive summaries, a full-document pass or dedicated summarization step is more reliable than chunk-based retrieval.

---

## Setup Notes

1. **GPU required** for Exercises 7 & 8 (rebuild pipeline is slow). All others run on T4.
2. All notebooks mount Google Drive automatically at `/content/drive/MyDrive/`.
3. Corpus path assumed: `Colab_Projects/Week05-RAG/Corpora/`.
4. Exercise 2 requires an `OPENAI_API_KEY` Colab secret.
