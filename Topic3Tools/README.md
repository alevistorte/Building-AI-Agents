# Topic 3: Tool Calling

[This was presented in class]

## Overview

This topic covers how language models interact with external tools, progressing from raw OpenAI API calls with manual tool dispatch, through LangChain's `@tool` abstraction, to a fully automated LangGraph agent that uses `ToolNode` and `tools_condition` for multi-turn tool use with SQLite-backed crash recovery.

---

## Tasks

### Task 1

#### Running sequentially

```bash
python program1.py && python program2.py
```

This works without issues. Each script runs one after the other, printing output to the terminal normally.

To time the sequential run:

```bash
time { python program1.py && python program2.py; }
```

#### Running in parallel (and why it fails)

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

##### Why this happens?

When you background a process with `&`, it becomes a **background job**. On macOS (zsh), background jobs that attempt to **read from or write to the terminal** receive `SIGTTOU`/`SIGTTIN` signals from the kernel, which immediately **suspend** them. This is a Unix job control mechanism that prevents background processes from interfering with the terminal.

Both scripts produce extensive terminal output (`print()` calls, `tqdm` progress bars), so they get suspended as soon as they try to print anything. Redirecting stdout and stderr (`> out.log 2>&1`) alone is not enough — the background process still has the terminal as its **controlling terminal**, and stdin is still connected to it. When a background process has stdin attached to the terminal, the kernel sends `SIGTTIN` to suspend it on any attempt to read, even if no read is explicitly performed (some libraries or the Python runtime itself may probe stdin).

##### Fix: redirect stdout, stderr, and stdin

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

### task4_combining_tools.py — Combining Multiple Tools with LangChain

**File:** `task4_combining_tools.py`
**Description:** A batch test harness demonstrating LangChain tool binding across 14 test cases. Three tools are defined with the `@tool` decorator — `get_weather` (simulated lookup), `calculator` (math expressions and geometric operations via `numexpr`), and `count_letter` (character frequency). The LLM (`gpt-4o-mini`) is bound with `llm.bind_tools(...)` and an explicit agent loop (up to 5 iterations per query) dispatches tool results back into the message list until the model produces a final text response.
**Key code snippet:**

```python
llm_with_tools = llm.bind_tools([get_weather, calculator, count_letter])

for iteration in range(5):
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    else:
        return response.content
```

**Output:**

```
See `task4_combining_tools_output.txt` for details.
```

---

### Tasks 2 & 3

Everything runs in the environment.

### task4_more_questions.py — Extended Question Set

**File:** `task4_more_questions.py`
**Description:** Extends the test suite from 14 to 15 queries, adding a multi-step chain: computing `sin(count('i') - count('s'))` in "Mississippi riverboats", which calls `count_letter` twice and `calculator` once within a single agent run. This demonstrates how the model composes multiple sequential tool calls to answer a compound question.
**Key code snippet:**

```python
run_agent("What is the sin of the difference between the number of i's "
          "and the number of s's in Mississippi riverboats?")
```

**Output:**

See `task4_combining_tools_output.txt` for representative run output.

```
See `task4_more_questions_output.txt` for details.
```

---

### task4.py — LangChain Manual Loop (Baseline, Pre-existing)

**File:** `task4.py`
**Description:** The pre-existing version of the multi-tool LangChain agent loop preserved in the directory. Contains the same 15-query test suite and three-tool setup as `task4_more_questions.py` and serves as a stable baseline for comparison with the LangGraph migration in `task5.py`.
**Output:**

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

---

### task5.py — Crash Recovery with LangGraph Session Persistence

**File:** `task5.py`
**Description:** Replaces the manual agent loop with a LangGraph graph using `ToolNode` and `tools_condition` for automatic tool routing. State is persisted to `sessions.db` (SQLite) via `SqliteSaver` after every node, enabling full crash recovery. On startup, existing sessions are listed and the user can resume any previous conversation. The `add_messages` reducer in `AgentState` accumulates full multi-turn history automatically.
**Key code snippet:**

```python
builder.add_conditional_edges(
    "call_agent",
    tools_condition,
    {"tools": "tools", END: "print_response"},
)
builder.add_edge("tools", "call_agent")
return builder.compile(checkpointer=checkpointer)
```

**Output:**

See `task5_output_recovering_from_crash.txt` for details.

```
Existing sessions:
  1. 20260301_213255
Enter a number to resume, or press Enter to start new: 1

Starting new session: 20260301_213255
Type 'quit' or 'exit' to end.

```
