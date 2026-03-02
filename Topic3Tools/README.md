# MMLU Evaluation with Ollama

## What the programs do

`program1.py` and `program2.py` evaluate the Llama 3.2-3B model on different MMLU (Massive Multitask Language Understanding) benchmark subjects using Ollama as the inference backend.

- **program1.py** evaluates: `abstract_algebra`, `anatomy`
- **program2.py** evaluates: `astronomy`, `business_ethics`

Each program connects to a locally running Ollama server, sends multiple-choice questions to the model, and records accuracy per subject. Results are saved to a timestamped JSON file.

## Running sequentially

```bash
python program1.py && python program2.py
```

This works without issues. Each script runs one after the other, printing output to the terminal normally.

To time the sequential run:

```bash
time { python program1.py && python program2.py; }
```

## Running in parallel (and why it fails)

```bash
time { python program1.py & python program2.py & wait; }
```

This produces output like:

```
[2] 37312
[3] 37313
[3]  + 37313 suspended (tty output)  python program2.py
[2]  + 37312 suspended (tty output)  python program1.py
```

### Why this happens

When you background a process with `&`, it becomes a **background job**. On macOS (zsh), background jobs that attempt to **read from or write to the terminal** receive `SIGTTOU`/`SIGTTIN` signals from the kernel, which immediately **suspend** them. This is a Unix job control mechanism that prevents background processes from interfering with the terminal.

Both scripts produce extensive terminal output (`print()` calls, `tqdm` progress bars), so they get suspended as soon as they try to print anything. Redirecting stdout and stderr (`> out.log 2>&1`) alone is not enough — the background process still has the terminal as its **controlling terminal**, and stdin is still connected to it. When a background process has stdin attached to the terminal, the kernel sends `SIGTTIN` to suspend it on any attempt to read, even if no read is explicitly performed (some libraries or the Python runtime itself may probe stdin).

### Fix: redirect stdout, stderr, and stdin

```bash
time { python program1.py > out1.log 2>&1 < /dev/null & python program2.py > out2.log 2>&1 < /dev/null & wait; }
```

The three redirections fully detach each process from the terminal:

- `> out1.log` — redirects stdout to a log file
- `2>&1` — redirects stderr to the same log file
- `< /dev/null` — redirects stdin from `/dev/null`, preventing any `SIGTTIN` suspension

With all three file descriptors redirected away from the terminal, the background processes have no reason to interact with the tty and will not be suspended.

This produces output like:

```
[2] 49466
[3] 49467
[2]  - 49466 done       python program1.py > out1.log 2>&1 < /dev/null
[3]  + 49467 done       python program2.py > out2.log 2>&1 < /dev/null
```

After completion, review results with:

```bash
cat out1.log
cat out2.log
```

After running both (sequentially and in parallel) we could notice that the parallel execution runs a way faster than the sequential one as expected.

## Question 4: LangGraph tool handling

Output while testing: **What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?**

```

============================================================
TEST 15: Counting multiple letters and calculating sine
============================================================
User: What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
  Tool: count_letter
  Args: {'text': 'Mississippi riverboats', 'letter': 'i'}
  Result: {"letter": "i", "text": "Mississippi riverboats", "count": 5}
  Tool: count_letter
  Args: {'text': 'Mississippi riverboats', 'letter': 's'}
  Result: {"letter": "s", "text": "Mississippi riverboats", "count": 5}

--- Iteration 2 ---
LLM wants to call 1 tool(s)
  Tool: calculator
  Args: {'input': '{"operation":"evaluate","expression":"sin(5 - 5)"}'}
  Result: {"result": 0.0}

--- Iteration 3 ---
Assistant: The sine of the difference between the number of 'i's and the number of 's's in "Mississippi riverboats" is \( \sin(0) = 0.0 \).
```

Output while using all the tools.
Prompt: **What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats plus the value of the temperature in San Francisco?**

```
============================================================
TEST 15: Testing all the tools together
============================================================
User: What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats plus the value of the temperature in San Francisco?

--- Iteration 1 ---
LLM wants to call 3 tool(s)
  Tool: count_letter
  Args: {'text': 'Mississippi riverboats', 'letter': 'i'}
  Result: {"letter": "i", "text": "Mississippi riverboats", "count": 5}
  Tool: count_letter
  Args: {'text': 'Mississippi riverboats', 'letter': 's'}
  Result: {"letter": "s", "text": "Mississippi riverboats", "count": 5}
  Tool: get_weather
  Args: {'location': 'San Francisco'}
  Result: Sunny, 72°F

--- Iteration 2 ---
LLM wants to call 1 tool(s)
  Tool: calculator
  Args: {'input': '{"operation":"evaluate","expression":"sin(5 - 5) + 72"}'}
  Result: {"result": 72.0}

--- Iteration 3 ---
Assistant: The difference between the number of 'i's and 's's in "Mississippi riverboats" is 0, and the current temperature in San Francisco is 72°F.

Thus, the calculation is \( \sin(0) + 72 = 72 \).

The final result is **72.0**.
```
