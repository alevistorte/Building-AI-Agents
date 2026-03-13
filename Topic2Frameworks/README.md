# Topic 2: LangGraph Frameworks

## Overview

This topic introduces LangGraph as a framework for building stateful agent loops. Each task incrementally extends a single-file chat agent, adding features such as verbose/quiet tracing, empty-input handling, parallel model dispatch, per-turn model routing, multi-turn chat history, shared multi-party history, and SQLite-backed crash recovery.

---

## Tasks

### Task 1 — Verbose/Quiet Mode Toggle

**File:** `task1.py`
**Description:** A LangGraph agent with three nodes (`get_user_input`, `call_llm`, `print_response`) that loops until the user types `quit`. The agent supports a real-time verbose/quiet toggle: typing `verbose` enables per-node `[TRACE]` output; typing `quiet` disables it. Both commands short-circuit the routing function to re-prompt without calling the LLM. A Mermaid PNG of the graph is saved to `lg_graph.png` before the loop begins.
**Key code snippet:**

```python
def route_after_input(state: AgentState) -> str:
    if state.get("should_exit", False):
        return END
    if state.get("user_input", "").lower() in ("verbose", "quiet"):
        return "get_user_input"
    return "call_llm"
```

**Output:**

Here is some relevant part of the output.

```
==================================================
Enter your text (or 'quit' to exit):
==================================================

> verbose
[TRACE] Verbose mode enabled.
[TRACE] Entering node: get_user_input | verbose=True

==================================================
Enter your text (or 'quit' to exit):
==================================================

> I am just looking for info about AI agents
[TRACE] get_user_input received: 'I am just looking for info about AI agents'
[TRACE] get_user_input routing to call_llm
[TRACE] Entering node: call_llm | verbose=True
[TRACE] call_llm sending prompt: 'User: I am just looking for info about AI agents\nAssistant:'

Processing your input...
[TRACE] call_llm received response (1363 chars)
[TRACE] Entering node: print_response | verbose=True

--------------------------------------------------
LLM Response:
--------------------------------------------------

[...]

==================================================
Enter your text (or 'quit' to exit):
==================================================

> quiet
[TRACE] get_user_input received: 'quiet'
Verbose mode disabled.

==================================================
Enter your text (or 'quit' to exit):
==================================================

More about planning agents

Processing your input...


```

**See task1_output.txt for details.**

---

### Task 2 — Fixing Empty Inputs in the Agent Loop

**File:** `task2.py`
**Description:** Extends Task 1 by adding an explicit guard for empty user input. When the user presses Enter without typing anything, the routing function detects the blank `user_input` field and redirects back to `get_user_input`, printing "Please enter some text." instead of forwarding an empty prompt to the LLM.
**Key code snippet:**

```python
if not user_input.strip():
    print("Please enter some text.")
    return {
        "user_input": "",
        "should_exit": False,
        "verbose": verbose,
    }
```

**Output:**

See `task2_output_empty_input.txt` to check the behavior of the agent with for an empty input. As shown, the LLM return something.

```
==================================================
Enter your text (or 'quit' to exit):
==================================================

>

Processing your input...

--------------------------------------------------
LLM Response:
--------------------------------------------------
User:
Assistant:
  - Today's weather forecast:
    - Monday: Partly cloudy with a high of 62°F (17°C) and a low of 45°F (7°C)
    - Tuesday: Mostly sunny with a high of 65°F (18°C) and a low of 50°F (10°C)

```

See `task2_output_empty_input_fixed.txt` where the empty input is not longer processed.

```
==================================================
Enter your text (or 'quit' to exit):
==================================================

>
Please enter some text.

```

---

### Task 3 — Running Models in Parallel

**File:** `task3.py`
**Description:** Demonstrates LangGraph's fan-out/fan-in pattern. A `dispatch_to_models` passthrough node fans out to two parallel branches — `call_llama` (Llama-3.2-1B-Instruct) and `call_qwen` (Qwen2.5-1.5B-Instruct) — which run concurrently. A `print_responses` fan-in node waits for both before printing the results side-by-side. The graph visualization is saved to `lg_graph_parallel_models.png`.
**Key code snippet:**

```python
graph_builder.add_edge("dispatch_to_models", "call_llama")
graph_builder.add_edge("dispatch_to_models", "call_qwen")
graph_builder.add_edge("call_llama", "print_responses")
graph_builder.add_edge("call_qwen",  "print_responses")
```

**Output:**

See `task3_output_parallel_models.txt` for details.

```
[...]
==================================================
Enter your text (or 'quit' to exit):
==================================================

What is the weather like today?

[Llama] Processing...

[Qwen] Processing...

==================================================
Llama-3.2-1B-Instruct response:
--------------------------------------------------
[...]

==================================================
Qwen2.5-1.5B-Instruct response:
--------------------------------------------------
[...]
```

---

### Task 4 — Alternating Between Models in Conversation

**File:** `task4.py`
**Description:** Introduces content-based routing: if the user's input starts with "Hey Qwen" (case-insensitive), the message is routed to `call_qwen`; all other input goes to `call_llama`. Exactly one model runs per turn. The responding model's branch writes its response and clears the other model's response field to avoid stale state.
**Key code snippet:**

```python
if user_input.lower().startswith("hey qwen"):
    return "call_qwen"
return "call_llama"
```

**Output:**
See `task4_output_alternating_models.txt` for details.

```
==================================================
LangGraph Parallel Agent: Llama-3.2-1B + Qwen2.5-1.5B
==================================================

[...]
==================================================
Enter your text (or 'quit' to exit):
==================================================

> hello

[Llama] Processing...

==================================================
Llama-3.2-1B-Instruct response:
--------------------------------------------------

==================================================
Enter your text (or 'quit' to exit):
==================================================

Hey qwen how are you

[Qwen] Processing...

==================================================
Qwen2.5-1.5B-Instruct response:
--------------------------------------------------
[...]

==================================================
Enter your text (or 'quit' to exit):
==================================================

> so cool

[Llama] Processing...

==================================================
Llama-3.2-1B-Instruct response:
--------------------------------------------------
[...]
```

---

### Task 5 — Persistent Chat History

**File:** `task5.py`
**Description:** Upgrades the agent to maintain multi-turn conversation context using LangChain's `ChatHuggingFace` and its message API (`SystemMessage`, `HumanMessage`, `AIMessage`). The `history` field in `AgentState` grows each turn; `get_user_input` appends a `HumanMessage` and `call_llama` appends an `AIMessage` after each response. The model therefore receives the full conversation on every call.
**Key code snippet:**

```python
return {
    "llama_response": response_text,
    "history": history + [ai_message],
}
```

**Output:**

See `task5_output_adding_chat_history.txt` for details.

```
==================================================
LangGraph Chat Agent with Llama-3.2-1B-Instruct
==================================================

[...]
```

---

### Task 6 — Shared History Across Multiple Agents

**File:** `task6.py`
**Description:** Implements a multi-party conversation among a Human, Llama, and Qwen. All turns are stored in a shared log of `SharedMessage` dicts (`speaker` + `content`). Before each LLM call, a `build_messages()` helper constructs a tailored `BaseMessage` list: the model's own past turns become `AIMessage`; all other speakers become `HumanMessage`. A system prompt identifies each model's role. The session log is written to a timestamped `chat_log_*.txt` file.
**Key code snippet:**

```python
def build_messages(model_name, other_name, history):
    ...
    for entry in history:
        speaker = entry["speaker"]
        labelled = f"{speaker}: {entry['content']}"
        if speaker == model_name:
            messages.append(AIMessage(content=labelled))
        else:
            messages.append(HumanMessage(content=labelled))
    return messages
```

**Output:**

See `task6_output_shared_history_multi_agent_chat.txt` for details.

```
==================================================
LangGraph Multi-Party Chat: Human + Llama + Qwen
  Start with 'Hey Qwen' to address Qwen.
  All other input goes to Llama.
==================================================
[...]
```

---

### Task 7 — Crash Recovery Using SQLite Session Persistence

**File:** `task7.py`
**Description:** Adds full crash recovery to the multi-party chat agent using LangGraph's `SqliteSaver` checkpointer. After every node, graph state is serialized to `chat_sessions.db`. On startup, `select_session()` lists existing sessions (identified by timestamp-based `thread_id`) with their turn counts; the user can resume any previous session or start a new one. Resuming passes an empty initial state dict so LangGraph replays from the last checkpoint.
**Key code snippet:**

```python
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = create_graph(llama_llm, qwen_llm, log_path, checkpointer)
...
if is_new:
    graph.invoke(initial_state, config=config)
else:
    graph.invoke({}, config=config)
```

**Output:**

See `task7_output_recovering_from_crash.txt` for details.

```
Starting new session: 20260301_205241
Chat log: [...]/chat_log_20260301_205241.txt
[...]

==================================================
Enter your text (or 'quit' to exit):
==================================================

> might be

[Llama] Processing...
^CTraceback (most recent call last):

[...]

KeyboardInterrupt

[...]

Existing sessions found:
  1. 20260301_205241  (4 turn(s) logged)

Enter a number to resume, or press Enter to start new: 1

Resuming session: 20260301_205241
```
