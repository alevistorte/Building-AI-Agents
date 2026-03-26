# Fine-Tuning Llama-3.2-1B for Text-to-SQL

## What was done

A `meta-llama/Llama-3.2-1B` base model was fine-tuned using LoRA on the `b-mc2/sql-create-context` dataset to perform text-to-SQL generation — converting natural language questions into SQL queries given a table schema.

**Dataset:** 78,577 examples split into 78,377 training and 200 held-out test examples.

**Fine-tuning setup:**

- Method: LoRA via Tinker
- Epochs: 1
- Batch size: 256
- Learning rate: 5e-4
- Prompt format: table schema + question → SQL completion

**Training loss:**

```
Epoch 1/1, update 100, loss: 0.0488
Epoch 1/1, update 200, loss: 0.0306
Epoch 1/1, update 300, loss: 0.0391
Epoch 1/1, update 307, loss: 0.0288
```

---

## Results

| Model                       | Accuracy (200 test questions) |
| --------------------------- | ----------------------------- |
| Base model (no fine-tuning) | 47.00% (94/200)               |
| Fine-tuned (LoRA, 1 epoch)  | **89.50% (179/200)**          |

Fine-tuning improved accuracy by **42.5 percentage points** — nearly doubling the base model's performance on in-distribution SQL generation tasks after a single epoch.

---

## Extra (out-of-distribution) examples

The fine-tuned and base models were also tested on 5 novel schemas not seen during training:

```python
for i, ex in enumerate(extra_examples):
    print(f"\nExample {i+1} ({ex['level']}):")
    print(f"  Question: {ex['question']}")
    print(f"  Context:  {ex['context']}")

    ft_answer = sample_from_model(fine_tuned_sampling_client, tokenizer, ex["context"], ex["question"])
    print(f"  Fine-tuned Model-generated SQL: {ft_answer}")

    base_answer = sample_from_model(base_sampling_client, tokenizer, ex["context"], ex["question"])
    print(f"  Base Model-generated SQL: {base_answer}")
```

```
Example 1 (Easy (single table, simple WHERE)):
  Question: What are the names of employees in the engineering department?
  Context:  CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)
  Fine-tuned Model-generated SQL: SELECT id, name FROM employees WHERE department = 'Engineering'
Base Model-generated SQL: SELECT name FROM employees WHERE department = 'engineering'

Example 2 (Easy (single table, simple WHERE)):
  Question: How many products cost more than 50 dollars?
  Context:  CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)
Fine-tuned Model-generated SQL: SELECT COUNT(*) FROM products WHERE price > 50
Base Model-generated SQL: SELECT * FROM products WHERE price > 50;

Example 3 (Medium (aggregation, ORDER BY)):
  Question: What is the highest score in the science class?
  Context:  CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)
Fine-tuned Model-generated SQL: SELECT MAX(T1.score) FROM students AS T1 JOIN courses AS T2 ON T1.class = T2.class WHERE T2.subject = 'Science'
Base Model-generated SQL: SELECT MAX(score) FROM students WHERE class = 'Science'

Example 4 (Medium (aggregation, ORDER BY)):
  Question: List the top 3 customers by total order amount.
  Context:  CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)
Fine-tuned Model-generated SQL: SELECT id, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3
Base Model-generated SQL: SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC LIMIT 3;

Example 5 (Hard (JOIN, GROUP BY)):
  Question: How many students are enrolled in each department?
  Context:  CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)
Fine-tuned Model-generated SQL: SELECT SUM(T1.student_id), T2.name, COUNT(*) FROM enrollments AS T1 JOIN courses AS T2 ON T1.course_id = T2.id WHERE T1.grade = 'A' OR T1.grade = 'B' GROUP BY T2.department
Base Model-generated SQL: SELECT department, COUNT(*) AS num_students
FROM enrollments
GROUP BY department;
```

On out-of-distribution examples, results are mixed. The fine-tuned model is more structured and SQL-idiomatic, but shows signs of **overfitting to training patterns** (e.g., hallucinating JOINs when the schema has only one table, or selecting unneeded columns). The base model, while weaker overall, sometimes produces simpler and more correct queries on novel schemas.

---

## Discussion

### Before vs. after: what improved?

The jump from **47% → 89.5%** on the 200 held-out test questions is large and consistent with what fine-tuning on structured task data typically achieves. The model learned both:

- **SQL syntax:** it stopped generating natural language explanations, repetitions, or incomplete queries (as the base model frequently did), and outputs clean, terminated SQL.
- **Schema grounding:** it learned to read the `CREATE TABLE` schema and reference the correct column and table names rather than guessing.

On the 5 out-of-distribution manual test questions the picture is more nuanced. The fine-tuned model performs better on straightforward queries (examples 2, partly 4) but worse on others — it hallucinates a JOIN in example 3 (where only one table exists) and adds spurious columns in examples 1 and 4. This suggests the model memorized JOIN patterns from the training data and over-applies them to novel schemas.

### RAG comparison

A RAG system with 1,000 (question, SQL) pairs would work well when:

- The test question is **semantically close** to a training example (e.g., "How many X have Y?" questions with the same structure). RAG can retrieve a near-identical example and adapt the SQL by substituting names.
- The schema is **simple and common** (single-table SELECT with WHERE) — small surface area for retrieval to go wrong.

RAG would struggle when:

- The test question requires **novel composition** — joining multiple tables or aggregations not represented in the retrieved examples.
- The schema is **unusual** — retrieved examples from different table structures mislead the model.
- The question is **ambiguous** — RAG has no mechanism to reason about schema constraints, only to pattern-match to prior queries.

Fine-tuning generalizes better to structural variation; RAG works well as a low-overhead baseline when query patterns are repetitive and the question distribution is stable.

### Error analysis

Looking at the fine-tuned model's failure modes on the extra examples:

- **Hallucinated JOINs** (example 3): The model adds a JOIN to a table (`courses`) that doesn't exist in the schema. This is a training-distribution artifact — many training examples used JOIN patterns, and the model learned to over-apply them when it sees aggregation questions.
- **Wrong column selection** (examples 1, 4): The model selects `id` alongside `name` or `amount`, which wasn't asked for. This suggests the model partially memorized `SELECT id, ...` patterns from the training data rather than purely grounding on the question.
- **Unnecessary WHERE on grade** (example 5): The model filters `WHERE grade = 'A' OR grade = 'B'` even though no grade condition was requested. This is a spurious correlation from the training distribution.

Each failure mode points to the same root cause: **the model learned surface patterns from the training corpus** (SELECT with id, JOIN with alias, grade filters) and applies them even when the question or schema doesn't call for them. The fix would be more diverse training data, longer training with regularization, or explicit schema-grounding supervision.
