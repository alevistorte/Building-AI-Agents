# Week 01: Running LLMs

## Overview

This week establishes the foundation for local LLM evaluation. The work centers on running different models through the MMLU (Massive Multitask Language Understanding) benchmark, comparing hardware configurations and their performance.

---

## Tasks

**Note:**
I am a Mac user, so I am not going to be able to do quantization while using GPU.

### Task 3: Setup verification

**The file `llama_mmlu_eval.py` runs ok in my system which means the setup is correct.**

### task4.py — Runtime Comparison Across Configurations (CPU vs. GPU)

**Files:** `task4_gpu.py` and. `task4_cpu.py`
**Description:** `task4_cpu.py` uses the same evaluation architecture as `task4_gpu.py` but with `USE_GPU = False` (CPU-only), enabling direct timing comparison against an MPS/GPU run. On Apple Silicon, results showed MPS inference is over 3x faster than CPU (21.5 s vs. 68.7 s for the 2-subject subset). This version is the CPU baseline used in the benchmark.

**Note:**
_Running multiple times and with different topics give different results but consistent that GPU execution is a way faster than CPU_

**Key code snippet:**

```python
USE_GPU = False  # Force CPU-only execution for runtime comparison
USE_GPU = True # Prefer GPU if available.
```

---

### task5.py — Multi-Model Evaluation

**File:** `task5.py`
**Description:** Extends evaluation to three models sequentially: `meta-llama/Llama-3.2-1B-Instruct`, `allenai/OLMo-2-0425-1B`, and `Qwen/Qwen2.5-1.5B-Instruct`. The `evaluate_all_models()` function loads each model, runs the full subject suite, saves per-model results. Results filenames encode model name, device, and timestamp.

**Key code snippet:**

```python
MODEL_NAMES = ["meta-llama/Llama-3.2-1B-Instruct",
               "allenai/OLMo-2-0425-1B",
               "Qwen/Qwen2.5-1.5B-Instruct"]

for model_name in MODEL_NAMES:
    model, tokenizer = load_model_and_tokenizer(device)
    ...
    del model, tokenizer
    torch.cuda.empty_cache()
```

**Results:**

The results can be found inside the `results/` folder. There are multiple files per model because each file represent a different running of the program.

Filenames ending in `*w_ans.json` include an array with the answers for each question for each topic.

1: Answer was correct
0: Answer was wrong

---

### task5_3.py — Answer Analysis with Per-Question Verbose Mode

For simplicity only one model was use in one topic.

**File:** `task5-3.py`
**Description:** Adds a `VERBOSE` flag and per-question tracking (`answers` list of 0/1 values) to `evaluate_subject()`. When `VERBOSE = True`, every question is printed with the model's answer and a correct/wrong verdict. The per-question `answers` arrays are included in the JSON output file as `subject_results[*].answers`, which is the format required by `analyze_results.py` for visualization.
**Key code snippet:**

```python
if verbose:
    print(f"\n--- Q{total} ---")
    print(f"Question: {question}")
    verdict = "CORRECT" if is_correct else f"WRONG (correct: {correct_answer})"
    print(f"Model answer: {predicted_answer}  {verdict}")
```

**Output:**
See `task5-3_output.txt` for details.

```
--- Q100 ---
Question: Which of the following statements is correct?
  A. Business ethics is not a real concept as the rules and ethics of everyday life don't apply to the world of business.
  B. Business Ethics is the study of business situations, activities and decisions where issues of right and wrong are addressed.
  C. Business ethics cannot be defined as a real concept as it is an oxymoronic term.
  D. Business ethics is concerned with the study of assessing the activities of a business based on their profitability.
Model answer: B  ✓ CORRECT
✓ Result: 45/100 correct = 45.00%

======================================================================
EVALUATION SUMMARY
======================================================================
Model: meta-llama/Llama-3.2-1B-Instruct
None (full precision)
Total Subjects: 1
Total Questions: 100
Total Correct: 45
Overall Accuracy: 45.00%
Duration: 0.2 minutes
======================================================================

✓ Results saved to: /Users/alevistorte/Documents/00Projects/Building-AI-Agents/Topic1RunningLLM/results/llama_3.2_1b_mmlu_results_full_20260313_155730.json

📊 Top 5 Subjects:
  1. business_ethics: 45.00%

📉 Bottom 5 Subjects:
  1. business_ethics: 45.00%

✅ Evaluation complete!
```

---

### task6_analyze_results.py — Result Visualization

**File:** `analyze_results.py`
**Description:** Loads all `*w_ans.json` result files from `results/` and generates five plots (all plots are generated with seaborn/matplotlib) saved to `results/plots/`:

- **(1) grouped bar chart of accuracy by subject and model:**
  Overall Qwen perform the best among all the categories. This makes sense because it has 5X parameters than the versions used for Olmo and Llama. In topics like statistics and physics models are close to the random chance performance indicating that they are not powerful enough to handle these topics.

- **(2) per-question correctness heatmaps (green=correct, red=wrong):**
  This show how the models performed per category. We can see that they struggled the most with Physics and Statistics.

- **(3) stacked bar showing all-correct / mixed / all-wrong question fractions per subject:**
  This plot shows the percentage of questions the models agree (correctly or wrongly) on the result and when there are also mixed answers from the models.

- **(4) pairwise Pearson correlation heatmap of binary answer vectors:**
  This plot shows per topic the percentage of the agreement between models.

- **(5) rolling accuracy curves over question sequence per subject:**
  This plot shows the rolling accuracy of the models over the question sequence.

  **Key code snippet:**

```python
def build_answer_matrices(models):
    for subject in common_subjects:
        df = pd.DataFrame({name: mdata["subjects"][subject]["answers"]
                           for name, mdata in models.items()}).T
        matrices[subject] = df
    return matrices
```

**Output:**

```
Loading results...
Loaded 3 models: ['Qwen2.5-1.5B', 'OLMo-2-1B', 'Llama-3.2-1B']
Built answer matrices for 10 subjects

Generating plots...
Saved: .../results/plots/01_accuracy_by_subject.png
Saved: .../results/plots/02_per_question_heatmaps.png
Saved: .../results/plots/03_mistake_overlap.png
Saved: .../results/plots/04_pairwise_correlation.png
Saved: .../results/plots/05_rolling_accuracy.png

All plots saved to: .../results/plots
```

---

### task7 — Google Colab Evaluation (Larger Models + Quantization)

**File:** `fromColab/llama_mmlu_eval.ipynb`
**Description:** The same MMLU evaluation pipeline was run in Google Colab to access a CUDA GPU, enabling both full-precision runs on all 1B models and 8-bit quantized runs on 7B/8B models that wouldn't fit in memory otherwise. Results are stored in the `fromColab/` folder.

**Models evaluated:**

| Model                            | Size | Quantization   | Accuracy   | Duration |
| -------------------------------- | ---- | -------------- | ---------- | -------- |
| allenai/OLMo-2-0425-1B           | 1B   | full precision | 37.21%     | 82s      |
| meta-llama/Llama-3.2-1B-Instruct | 1B   | full precision | 50.92%     | 252s     |
| Qwen/Qwen2.5-1.5B-Instruct       | 1.5B | full precision | 66.07%     | 307s     |
| allenai/Olmo-3-7B-Instruct       | 7B   | full precision | 58.89%     | 919s     |
| meta-llama/Llama-3.1-8B-Instruct | 8B   | 8-bit          | 71.00%     | 536s     |
| Qwen/Qwen2.5-7B-Instruct         | 7B   | 8-bit          | **75.74%** | 517s     |

All models ran on `device: cuda` (Colab GPU). The 10-subject subset used is: astronomy, business_ethics, nutrition, philosophy, prehistory, professional_accounting, public_relations, security_studies, sociology, us_foreign_policy (2131 questions total).

**Key findings:**

- Qwen consistently outperforms same-scale counterparts: Qwen2.5-1.5B (66%) beats Llama-3.2-1B (51%) and OLMo-2-1B (37%) despite similar parameter counts.
- Scaling from 1B to 7B/8B yields a large accuracy jump (~+9–25 pp), confirming that model size matters more than architecture for MMLU.
- 8-bit quantization has negligible accuracy cost: Qwen2.5-7B at 8-bit (75.74%) is very competitive; Llama-3.1-8B at 8-bit (71%) is reasonable for its size.
- OLMo-2-1B at 37% is near random chance (25%), while OLMo-3-7B at 59% shows the 7B version is substantially better, though still trails Qwen and Llama at similar scales.

---

### task8_simple_chat.py — Simple Chat Interface

**File:** `task8_simple_chat.py`
**Description:** A bare-bones interactive chat loop using `Qwen2.5-1.5B-Instruct` (configurable via `MODEL_NAME`) that demonstrates the mechanics of multi-turn conversation: chat history stored as a list of role-content dicts, tokenized with `apply_chat_template()` each turn, and fed in full to `model.generate()`. Two context management strategies are implemented and toggled with `CONTEXT_METHOD`: a sliding window (drops oldest messages) and a summarization approach (uses the model to compress old context). The `ENABLE_HISTORY` flag toggles memory entirely on or off for comparison.
**Key code snippet:**

```python
input_ids = tokenizer.apply_chat_template(
    managed_history,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
```

**Output:**
See `task8_simple_chat_output.txt` for details.

```
======================================================================
Chat started! Type 'quit' or 'exit' to end the conversation.
History: ENABLED (method: sliding_window)
======================================================================

You: Hello! This is ALbert
Assistant: Hello Albert! How can I assist you today?

```

---
