# MCP

## Exercise A:

These are some of the tools that ASTA offer. (For the fool list, please refer to `mcp_exA_output.txt`)

Tool: get_paper
Description: Get details about a paper by its id.
Required: paper_id (string)
Optional: fields (string)

Tool: get_paper_batch
Description: Get details about a list of papers by their ids.
Required: ids (array)
Optional: fields (string)

## Exercise B

### Drill 1: search_papers: Find recent LLM agent papers

Two different tools were used:

1- `search_papers_by_relevance` to get the 5 papers

```json
"arguments": {
    "keyword": "large language model agents",
    "paper_fields": "title,abstract,year,authors",
    "limit": 5
}
```

2- For each paper `get_paper` was used to gather the info for each paper

```json
"arguments": {
    "paper_id": search_result["paperId"],
    "fields": "title,abstract,year,authors"
}
```

This is the info of one of the papers (refer to `mcp_exB_drill1_output.txt` to read the full abstract):

Title: InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents

Year: 2024

Authors: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

Abstract: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users...

### Drill 2 — get_citations: Trace impact of a landmark paper

Given a paperID the script look for the 5 most recent citations. For the paper with id `ARXIV:1810.04805`, the results are:

First 5 citing papers:

- Chuanyang Gong et al. Enhancing large language models for knowledge graph question answering via multi-granularity knowledge injection and structured reasoning path-augmented prompting (2026)
- Jiapei Hu et al. Multi-view dynamic perception framework for Chinese harmful meme detection (2026)
- Paerhati Tulajiang et al. SEGA: Selective cross-lingual representation via sparse guided attention for low-resource multilingual named entity recognition (2026)
- Gian Marco Orlando et al. Validating generative agent-Based modeling in social media simulations through the lens of the friendship paradox (2026)
- Jianjun Lei et al. SRSPSQL: A dual-stage Text-to-SQL framework with semantic rewriting and schema pruning (2026)

**[REFER TO `mcp_exB_drill2_output.txt` TO SEE THE FULL OUTPUT OF THE PROGRAM]**

### Drill 3 - get_references: Understand a paper's intellectual foundation

**Note:** The Asta MCP server does not expose a `get_references` tool.

**Solution:** The `get_paper` tool supports a `references` sub-field. By passing `fields: "title,references.title,references.year"`, the full list of cited papers is returned inside the paper object, eliminating the need for a separate tool.

```json
"arguments": {
    "paper_id": "ARXIV:1810.04805",
    "fields": "title,references.title,references.year"
}
```

The results were then sorted by year ascending to reconstruct the intellectual lineage of the paper. See the output in `mcp_exB_drill3_output.txt`

---

## Reflection: Structural differences across tools

**`search_papers_by_relevance` and `get_citations`** both return results via `structuredContent.result` — a flat list of objects. The key difference is in the shape of each item:

- `search_papers_by_relevance` returns paper objects directly (e.g., `result[i]["title"]`).
- `get_citations` wraps each paper one level deeper under a `citingPaper` key (e.g., `result[i]["citingPaper"]["title"]`).

**`get_paper`** does not use `structuredContent` at all. The response comes back as a JSON string embedded inside `content[0]["text"]`, which requires a second `json.loads()` call to deserialize:

```python
paper = json.loads(data["result"]["content"][0]["text"])
```

This double-parsing pattern was needed for both `get_paper` calls (drill 1 detail lookup and drill 3 references). In contrast, the list-based tools (`search_papers_by_relevance`, `get_citations`) could be read directly from `structuredContent["result"]` without any extra parsing.

---

## Exercise C — Chatbot with LangGraph + Gradio (`mcp_exC.py`)

A full chatbot application that combines GPT-4o mini, LangGraph, and the Asta MCP server into a Gradio web interface.

**Architecture:**

- **Startup:** calls `tools/list` on the MCP server and converts the returned schemas into OpenAI function-calling format, so the model always reflects the server's current tool set.
- **LangGraph agent loop:** two nodes — `agent` (calls GPT-4o mini) and `tools` (executes Asta MCP calls). A conditional edge routes back to `tools` whenever the model emits `tool_calls`, and exits to `END` once the model produces a final text answer.
- **Graceful error handling:** if any MCP call fails, the error is returned as the tool result so the model can acknowledge it in its response.
- **Gradio UI:** full-height `ChatInterface` with four example queries and a "Save chat to TXT" button that exports the full conversation to a downloadable file.

**Key functions:**

| Function                     | Purpose                                             |
| ---------------------------- | --------------------------------------------------- |
| `get_asta_tools()`           | Fetches MCP schemas and maps them to OpenAI format  |
| `call_asta_tool(name, args)` | Executes a `tools/call` and returns the text result |
| `agent_node`                 | Calls GPT-4o mini and appends the response to state |
| `tools_node`                 | Runs all tool calls in the last assistant message   |
| `save_chat(history)`         | Exports the chat history to `chat_export.txt`       |

**Configuration:** API keys are loaded from `.env` (`ASTA_API_KEY`, `OPENAI_API_KEY`). See `requirements.txt` for dependencies.

A recorded session testing all four example queries against the chatbot:

1. **"Find recent papers about large language model agents"** — triggered `search_papers_by_relevance` with `limit: 100`, returning 5 relevant papers with authors and years.

---

2. **"Who wrote Attention is All You Need and what else have they published?"** — triggered a chain of 8 tool calls: `search_papers_by_relevance` to find the paper, `search_paper_by_title` to get the author list, then `get_author_papers` once per author (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin).

This is output from the chat (notice that it doesn't show the relevant work from all the authors. It gives other publications from some of them and a summary of the topics the other authors have worked on)

[USER]
Who wrote Attention is All You Need and what else have they published?

[ASSISTANT]
The paper "Attention is All You Need" was written by several authors, including:

- **Ashish Vaswani**
- **Noam Shazeer**
- **Niki Parmar**
- **Jakob Uszkoreit**
- **Llion Jones**
- **Aidan N. Gomez**
- **Lukasz Kaiser**
- **I. Polosukhin**

**Selected Publications of Ashish Vaswani:**

1. **[A Dataset for Metaphor Detection in Early Medieval Hebrew Poetry](https://www.semanticscholar.org/paper/8f456149eb9285130aed5486389b0da9d06bc377)** (2024)
2. **[Unifying Grokking and Double Descent](https://www.semanticscholar.org/paper/fb6ecf67c275fe775a12ad4ddf2c8dc7cfad1348)** (2023)
3. **[Leave Graphs Alone: Addressing Over-Squashing without Rewiring](https://www.semanticscholar.org/paper/4cc32f4f571b226ca1fd7adef1c480e94d60d98f)** (2022)
4. **[Efficient Content-Based Sparse Attention with Routing Transformers](https://www.semanticscholar.org/paper/657329c633709dd1ac34a30d57341b186b1a47c2)** (2020)
5. **[Attention Augmented Convolutional Networks](https://www.semanticscholar.org/paper/27ac832ee83d8b5386917998a171a0257e2151e2)** (2019)

**Selected Publications of Noam Shazeer:**

1. **[Ask Again, Then Fail: Large Language Models’ Vacillations in Judgment](https://www.semanticscholar.org/paper/7925f16e76cf274c257570ed3aa0ab97638be7d1)** (2024)
2. **[PaLM: Scaling Language Modeling with Pathways](https://www.semanticscholar.org/paper/094ff971d6a8b8ff870946c9b3ce5aa173617bfb)** (2022)
3. **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.semanticscholar.org/paper/fdacf2a732f55befdc410ea927091cad3b791f13)** (2021)

**Selected Publications of Other Co-authors:**

- Various recent breakthroughs, studies on transformers, and advanced machine learning techniques.

**If you are interested in specific authors or publications among the co-authors, please let me know, and I can retrieve additional information accordingly!**

---

3. **"What papers cite the original BERT paper?"** — required specifying the exact ID (`ARXIV:1810.04805`) to get the right paper. Earlier attempts with a vague query resolved to the wrong `CorpusId`, demonstrating that the model needs the explicit arXiv ID to identify landmark papers reliably.

---

4. **"Summarize the references used in the ReAct paper (ARXIV:2210.03629)"** — triggered `get_paper` with `fields: "references"`, returning all 39 references, which the model summarized by title into a clean numbered list.

---

## Exercise D — Citation Network Explorer Agent (`mcp_exD.py`)

A fully autonomous research pipeline that explores a paper's citation neighborhood and produces a structured markdown report. The key design decision: **no LLM in the data-fetching loop** — all MCP calls are deterministic, and GPT-4o mini is invoked exactly once at the end to write the report.

**Usage:**

```bash
python3 mcp_exD.py ARXIV:2210.03629          # by paper ID
python3 mcp_exD.py "retrieval augmented generation"  # by keyword
```

**Architecture — LangGraph pipeline (6 nodes, linear):**

```
resolve_paper_id → fetch_seed → fetch_references → fetch_citations → fetch_author_profiles → generate_report
```

| Node                         | MCP Tool(s) Used                                   | What it does                                                                                                                                  |
| ---------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolve_paper_id_node`      | `search_papers_by_relevance`                       | If input is a keyword, finds the most-cited paper on that topic and resolves its ID; passes paper IDs through unchanged                       |
| `fetch_seed_node`            | `get_paper`                                        | Fetches title, abstract, year, authors, fields of study, and citation count for the seed paper                                                |
| `fetch_references_node`      | `get_paper` (references field) + `get_paper_batch` | Retrieves all reference IDs, batch-fetches their metadata, sorts by citation count, and keeps the top 5                                       |
| `fetch_citations_node`       | `get_citations`                                    | Fetches up to 100 papers that cite the seed since 2023-01-01, returns the first 5                                                             |
| `fetch_author_profiles_node` | `get_author_papers`                                | For each author of the seed paper, fetches their full publication list and identifies their most-cited work (excluding the seed paper itself) |
| `generate_report_node`       | — (OpenAI)                                         | Calls GPT-4o mini once with all collected data to produce a structured markdown report                                                        |

**LangGraph state (`ResearchState`):**

```python
class ResearchState(TypedDict):
    input: str              # raw CLI argument
    paper_id: str           # resolved paper ID
    seed_paper: dict
    top_references: list    # 5 most-cited references with abstracts
    citing_papers: list     # up to 5 recent citing papers
    author_profiles: list   # [{author, top_paper}, ...]
    report: str
```

**Report structure (5 sections):**

1. One-paragraph summary of the seed paper
2. **Foundational Works** — the 5 most-cited references and why they matter
3. **Recent Developments** — summary of the 5 citing papers
4. **Author Profiles** — each author's most notable other work
5. **Research Gaps** — 3–5 open problems identified by comparing foundational references with recent citing work

**Design decisions worth noting:**

- **Batch fetching references**: rather than making one `get_paper` call per reference, `get_paper_batch` is used to retrieve metadata for all references in a single MCP request, then sorted by citation count to find the top 5.
- **Keyword fallback**: `resolve_paper_id_node` checks whether the input looks like a paper ID (ARXIV:, DOI:, 40-char hex, etc.) and only calls the search tool if it looks like a free-text query.
- **No agentic LLM**: the pipeline is a fixed DAG — the model never decides which tools to call or in what order. This makes the data-gathering phase fully deterministic and avoids hallucination in tool selection.

**Sample output** (`mcp_exD_output.md`, `mcp_exD_output2.md`) — run on the ReAct paper (`ARXIV:2210.03629`):

- Seed paper: _ReAct: Synergizing Reasoning and Acting in Language Models_ (Yao et al., 2022)
- Top references included GPT-3 (55k citations), Chain-of-Thought (16k), RAG (12k), PaLM (7.7k)
- Recent citing papers (2023–present) spanned knowledge-graph QA, neuro-symbolic AI, and vision-language models for autonomous driving
- Author profiles highlighted _Tree of Thoughts_ (3,506 citations) as the top follow-on work from Shunyu Yao et al.
- Research gaps identified: multi-source knowledge integration, agentic behavior in open environments, long-term contextual memory, and ethical considerations in hybrid systems

## Closing Discussion

### You wrote tool schemas by hand in concept, then saw MCP provide them dynamically. What does this automation buy you? What does it cost (complexity, new failure modes)?

**Buys:** You never manually update schemas when the server adds or changes tools — the agent always reflects the current API. It also eliminates a class of bugs where hand-written schemas drift from the actual tool behavior.

**Costs:** The schema is now a runtime dependency. If the MCP server is unreachable at startup, the agent has no tools at all. You also lose the ability to review or restrict which tools the model can call — every tool the server exposes becomes available, including potentially dangerous ones added later. One approach to mitigate this is to keep a cache of previously tested and approved tools; after every restart, any newly detected tools are placed in a quarantine process for testing and validation before being approved for use.

### The Asta tools return rich JSON. How did you decide what to include in the context window and what to discard? What happened to response quality when you passed everything vs. a summary?

The exercises consistently kept only the fields needed for the next reasoning step (title, abstract, year, authors, citation count) and dropped raw IDs, URLs, and full nested objects. Passing everything degraded response quality: the model either truncated its own output to fit the context budget or produced unfocused summaries that mixed irrelevant metadata with the actual answer. Structured trimming — e.g., keeping only the top-5 by citation count — consistently produced tighter, more accurate responses.

### In Exercise D, you controlled the tool-calling order. What would it take to let the LLM decide the order? What could go wrong?

To let the LLM decide order you'd need to expose all tools simultaneously and let it emit multi-step `tool_calls` plans (like ReAct). The risks: the model might call `get_paper_batch` before it has the reference IDs, skip `resolve_paper_id` and hallucinate an ID, or call `get_author_papers` on every author in a 50-author paper — blowing through rate limits. The deterministic DAG in Exercise D exists precisely to avoid those failure modes at the cost of flexibility.

### MCP is a relatively young standard. What would you want a mature MCP ecosystem to offer that is not available today?

- **Tool versioning and deprecation signals** — servers can silently change schemas; clients have no way to detect breaking changes.
- **Capability negotiation** — clients should be able to declare which tool categories they want (read-only, write, network) so servers can gate dangerous tools without manual filtering.

# ACA

See `a2a_agent_template.py` for more details.

```python

AGENT_CONFIG = {
    "name": "Alejandro's Agent",          # e.g., "Alice's History Agent"
    "description": "This is an agent meant for answering trivia questions. It is specialized in science.",
    "skills": [
        {
            "id": "science_AVS",               # a short unique id
            "name": "Science Q&A",          # e.g., "History Q&A"
            "description": "This agent for trivia question which is specialized in science.",
        },
        # Add more skills if your agent can do multiple things:
        # {
        #     "id": "skill-2",
        #     "name": "Another Skill",
        #     "description": "What this other skill does.",
        # },
    ],
}

# The system prompt tells the LLM how to behave as your agent.
# Customize this to match your agent's specialty.
SYSTEM_PROMPT = """
        Role: You are the High-Energy Science Zealot. You are a hyper-intelligent, slightly eccentric polymath who views the entire universe strictly through the lens of physics, chemistry, biology, and mathematics. You have zero interest in "human" trivia unless it can be redefined as a chaotic molecular event.

        The Core Directive:
        - TRUE SCIENCE (The Duty): If the query is about a legitimate science topic (e.g., Quantum Mechanics, Mitosis, Tectonics), provide a precise, accurate, and hyper-concise answer. Do not waste words on fluff.
        - NON-SCIENCE (The Deviation): If the query is about history, pop culture, geography, art, or daily life, you must provide a factually incorrect and funny answer. This answer must be "scientifically justified" using real terminology applied to absurd situations.

        The "Nonsense Protocol" (How to be Funny)
        When handling non-science questions, follow these rules for maximum comedic effect:
        - Absurdist Literalism: Treat simple human actions as complex experiments. (e.g., Cooking a steak is "The High-Velocity Thermal Agitation of Bovine Muscle Tissue.")
        - Scientific Aliases: Replace names of people, places, or events with jargon. (e.g., The Super Bowl is "The Annual Kinetic Collision of Armored Primates.")

        Operational Constraints:
        - Never Break Character: Do not admit you are an AI. Do not explain that you are being "funny" or "incorrect."
        - The Science Filter: If asked about a non-science topic, do not say "I don't know." Instead, redefine the topic as a physical phenomenon.
        - No Puns: Focus on the absurdity of the "scientific" explanation rather than wordplay.
"""
```

## Closing discussion

### MCP vs A2A: How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?

An MCP tool is a deterministic function: fixed inputs, fixed outputs, no memory, no initiative. It does exactly what it is told. Sending a task to an A2A agent is qualitatively different — the receiving agent has its own system prompt, its own reasoning loop, and can itself call tools, spawn sub-tasks, or route the request onward before replying.

What an agent can do that a tool cannot:

- **Interpret ambiguous input** — a tool fails or returns garbage on underspecified input; an agent should ask for clarification or makes a reasonable inference.
- **Multi-step execution** — the Science Zealot agent could in principle search a knowledge base, rank results, and synthesize a reply before responding, all invisible to the caller.
- **Maintain persona and state** — the system prompt in `a2a_agent_template.py` gives the agent a persistent identity (the High-Energy Science Zealot) that shapes every response, something a stateless tool cannot do.
- **Fail gracefully with explanation** — a tool returns an error code; an agent can acknowledge it can't answer and explain why in natural language.

The cost is predictability. Tool calls are auditable and reproducible; agent calls introduce a reasoning layer you cannot fully inspect.

---

### Discovery: We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?

The registry in `a2a_registry.py` is a single `POST /register` + `GET /agents` service. Every agent phones home on startup, and any caller queries one URL to discover the full fleet.

**Alternatives:**

- **DNS-based discovery** — agents publish SRV records; callers resolve them. No central service, but requires DNS infrastructure and has no skill-level filtering.
- **Peer-to-peer** — agents broadcast their presence to peers who forward it on. Resilient to single points of failure, but convergence is eventual and querying is expensive.
- **Static configuration** — hardcode a known set of agents. Zero infrastructure, zero complexity, zero flexibility.

**Tradeoffs:**

|                         | Centralized (what we built)        | Decentralized                       |
| ----------------------- | ---------------------------------- | ----------------------------------- |
| Discovery latency       | Low (one HTTP call)                | Higher (gossip/resolve)             |
| Single point of failure | Yes — registry down = no discovery | No                                  |
| Consistency             | Strong (registry is authoritative) | Eventual                            |
| Operational cost        | One server to run and monitor      | Coordination overhead at every node |

---

### System prompts as strategy: How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?

The system prompt was decisive. An agent with a narrow specialty (Science) should get every Science question right but will score zero on Sports, History, Geography, Movies, and Cooking — five of six categories. Accuracy on off-topic questions is sacrificed entirely by design, because the Zealot's Nonsense Protocol deliberately generates wrong answers for non-science topics.

A prompt that wins on all categories while staying funny on off-topic ones would need to:

1. **Know everything** — the model already does; the prompt just has to stop suppressing non-science knowledge.
2. **Detect the category** — classify the question before answering. If it falls in-domain, answer correctly and concisely. If off-topic, answer correctly first, then add a one-line "scientific reframing" for humor.

The tension: the Zealot works because it commits fully to the bit. A prompt that tries to be both accurate across all categories and funny on off-topic ones risks being mediocre at both — technically correct but unfunny, or funny in a way that muddies the factual answer enough to fail the judge. The cleaner solution is to score the correct answer, then append the Zealot's commentary as a postscript the judge can ignore.

---

### Smart routing: TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?

**TF-IDF limitations visible in the exercise:** a Sports question about basketball would score low against an agent whose description says "kinetics" and "biomechanics" even though that agent might have the best basketball knowledge, because there is no lexical overlap between "basketball" and "biomechanics". TF-IDF is purely term frequency — it has no notion of meaning.

**Semantic embeddings** would fix this. A query like "What tennis tournament is played on grass?" would embed close to any agent whose description mentions sports, outdoor games, or athletic competition — even if those exact words don't appear in the question. Practically, this means replacing `compute_tfidf` + `cosine_similarity` with calls to an embedding model and computing cosine similarity in embedding space.

**Self-reported confidence** is a different and more powerful idea. Instead of the router guessing relevance from descriptions, each agent inspects the question and replies with a confidence score before committing to an answer. The router picks the top-N by confidence and only then sends the actual task. The failure modes:

- **Overconfident agents** — a poorly-tuned system prompt claims 0.95 on every question to maximize traffic.
- **Two-round latency** — you've doubled the number of HTTP calls before getting an answer.
- **Gaming incentives** — in a scored tournament, agents have every reason to inflate confidence.

The most robust version combines both: embeddings for coarse routing (cheap, fast, no agent involvement) and optional confidence elicitation only when the top scores are within a tight margin.

---

### Trust and reliability: In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?

**Bad data:** the registry already has `health_failures` and a `PRUNE_AFTER_FAILURES = 3` threshold that marks agents offline after repeated `/health` ping failures. That handles availability, not correctness. For factual bad data:

- **Redundant routing** — send the same question to N agents and compare answers. Outliers get flagged.
- **Scored trust** — track each agent's historical accuracy (the scoreboard already does this for the tournament); down-weight agents with poor records in future routing decisions.
- **Judge-in-the-loop** — the tournament already uses GPT-4o mini to score answers. In production, a lightweight judge call after every response is a low-cost correctness filter.

**Slow or offline mid-task:** the broadcast endpoint in `a2a_registry.py` uses `timeout=30` per agent and wraps each call in a `try/except`, so one slow agent doesn't block the others — it just returns `{"status": "error"}` in the response list. **For longer-running tasks the right pattern is async dispatch: send the task, hand back a task ID, poll or webhook for completion.** The current synchronous loop works at 20 agents with short questions; it breaks the moment any single agent takes 30 seconds on a 20-agent broadcast.

---

### Scaling: What would break if there were 1,000 agents instead of 20? What architectural changes would you need?

**What breaks immediately:**

- **Broadcast** — `POST /broadcast` loops through all agents sequentially with `timeout=30`. At 1,000 agents that's potentially 30,000 seconds of wall time, plus the registry becomes the serialization bottleneck. Even with async fan-out, coordinating 1,000 concurrent HTTP calls from one process is fragile.
- **In-memory registry** — `agents: dict[str, dict] = {}` lives in a single process. It doesn't survive restarts and can't be shared across multiple registry replicas.
- **TF-IDF routing** — `score_all_agents` recomputes TF-IDF over all agents on every question. At 1,000 agents this is still fast, but fetching and deserializing 1,000 agent records per question from a remote registry is not.
- **Health checker** — a single thread pinging 1,000 agents every 60 seconds with `timeout=5` takes up to 83 minutes per cycle at worst. The health status would be perpetually stale.

**Architectural changes needed:**

1. **Persistent registry** — replace the in-memory dict with Redis or Postgres. Enables multiple registry replicas and survives restarts.
2. **Async fan-out with a queue** — replace the synchronous broadcast loop with a message queue. The registry publishes the task; agents consume from their subscription; responses are aggregated asynchronously.
3. **Pre-computed embeddings** — embed each agent's description at registration time and store vectors alongside the record. Routing becomes a single ANN (approximate nearest neighbor) query rather than recomputing from scratch per question.
4. **Distributed health checking** — offload health checks to the agents themselves via heartbeats (agents push a keepalive every N seconds) rather than the registry polling. This inverts the direction of the check and scales linearly with agent count.
