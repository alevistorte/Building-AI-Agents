# Example Conversation: Tool Use, Context, and Recovery

This transcript demonstrates three key capabilities of the LangGraph agent:
1. **Tool use** — the LLM calls multiple tools in a single turn
2. **Conversation context** — follow-up questions are answered using prior history, no re-querying
3. **Recovery** — the session is killed mid-conversation and resumed with full context intact

---

## Session Start

```
$ python langgraph-tool-handling.py

Starting new session: 20260301_143022
Type 'quit' or 'exit' to end.
```

---

## Turn 1 — Multi-Tool Query (all three tools in one turn)

```
You: I'm planning a company offsite and need help with logistics. We're
     considering San Francisco, New York, and London — what's the weather
     in each? Our total venue budget is the same as the area of a rectangle
     60 by 35 feet. Catering will cost 480 + 320 dollars — how much of the
     budget is left after that? Finally, the event name is 'Bay Area Tech
     Explorers' — how many times does the letter 'e' appear in it, so I
     can use it as a themed trivia question for the team?
```

**Internally, the `call_agent` node returns tool calls:**
- `get_weather("San Francisco")` → `"Sunny, 72°F"`
- `get_weather("New York")` → `"Cloudy, 55°F"`
- `get_weather("London")` → `"Rainy, 48°F"`
- `calculator({"operation": "rectangle_area", "length": 60, "width": 35})` → `{"result": 2100.0}`
- `calculator({"operation": "evaluate", "expression": "2100 - 480 - 320"})` → `{"result": 1300.0}`
- `count_letter("Bay Area Tech Explorers", "e")` → `{"letter": "e", "text": "Bay Area Tech Explorers", "count": 4}`

**The `tools` node executes all six, appends ToolMessages, and routes back to `call_agent`. The LLM now has all results and produces a final answer:**

```
Assistant: Here's everything for your offsite planning:

**Weather**
- San Francisco: Sunny, 72°F ☀️
- New York: Cloudy, 55°F 🌥️
- London: Rainy, 48°F 🌧️

**Budget**
- Venue budget: $2,100 (area of a 60 × 35 rectangle)
- Catering: $800 ($480 + $320)
- Remaining after catering: $1,300

**Trivia answer**
The letter 'e' appears **4 times** in "Bay Area Tech Explorers"
(Bay Ar**e**a T**e**ch **E**xplor**e**rs — case-insensitive).
That's a great trivia question for the team!
```

---

## Turn 2 — Context: Follow-Up Without Re-Querying Tools

```
You: Given the weather, which city would you recommend for an outdoor
     venue, and is our remaining budget enough for a mid-range activity
     there?
```

**The `call_agent` node sends the full history — including the weather results from Turn 1. The LLM answers without calling any tools:**

```
Assistant: Based on the weather from earlier, **San Francisco** is the
clear choice for an outdoor venue — sunny at 72°F is ideal for an
outdoor event. New York is overcast and London is rainy, which would
make outdoor activities less pleasant.

As for budget: your remaining $1,300 after catering is solid for a
mid-range outdoor activity. A team sailing tour, a guided hike in the
Marin Headlands, or a private rooftop reception in SF would each
typically run $800–$1,200 for a group, leaving a small buffer.
```

> **Note:** No tools were called. The LLM used the weather data already in the message history.

---

## Turn 3 — Session Killed Mid-Conversation

The user presses `Ctrl+C` after typing the next question but before seeing the response.

```
You: What's the area of a circle that fits exactly inside a square with
     side length equal to our remaining budget divided by 100?
^C
Session saved. Goodbye.
```

> Because `SqliteSaver` checkpoints after every node, the state — including Turns 1 and 2 — is fully persisted in `sessions.db`.

---

## Recovery — Resuming the Session

```
$ python langgraph-tool-handling.py

Existing sessions:
  1. 20260301_143022

Enter a number to resume, or press Enter to start new: 1

Resuming session: 20260301_143022
Type 'quit' or 'exit' to end.
```

**The graph calls `graph.invoke({}, config=config)`. LangGraph loads the checkpoint and restores the full `messages` list. The agent drops straight into `get_user_input` as if nothing happened.**

```
You: What's the area of a circle that fits exactly inside a square with
     side length equal to our remaining budget divided by 100?
```

**`call_agent` has the full history including Turn 2's "$1,300 remaining". It calls:**
- `calculator({"operation": "evaluate", "expression": "1300 / 100"})` → `{"result": 13.0}`
- `calculator({"operation": "circle_area", "radius": 6.5})` → `{"result": 132.732289}`

```
Assistant: A square with side length 13 (your $1,300 ÷ 100) has an
inscribed circle with radius 6.5 (half the side length).

Area = π × 6.5² ≈ **132.73 square units**
```

---

## Turn 5 — Demonstrating Persistent Context After Recovery

```
You: Remind me — how many e's were in our event name?
```

```
Assistant: "Bay Area Tech Explorers" contains **4 occurrences** of the
letter 'e' (case-insensitive). That was from your original planning
question at the start of our conversation.
```

> **No tool call made.** The answer came directly from message history — which survived the crash and recovery intact.

---

## Session End

```
You: quit

Session saved. Goodbye.
```

---

## What This Demonstrates

| Capability | Where shown |
|------------|------------|
| Multi-tool dispatch in one turn | Turn 1: 6 tools called (weather ×3, calculator ×2, count_letter ×1) |
| Tool loop (tools → call_agent → tools) | Turn 4: two sequential calculator calls |
| Conversation context | Turn 2: LLM answers from history, no tools needed |
| Cross-turn memory after recovery | Turn 5: recalls Turn 1 data after crash + restart |
| Checkpointing & recovery | Session killed after Turn 2; resumed at Turn 3 with full history |
