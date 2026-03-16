"""
Autonomous citation-neighborhood research agent.

Usage:
    python3 mcp_exD.py ARXIV:2210.03629

The agent fetches all data directly via MCP — no LLM in the decision loop.
GPT-4o mini is only called once, at the end, to write the markdown report.
"""

import sys
import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ASTA_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def _call_mcp(payload: dict) -> dict:
    resp = requests.post(
        "https://asta-tools.allen.ai/mcp/v1",
        headers=ASTA_HEADERS,
        json=payload,
        timeout=30,
    )
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError(f"No SSE data in response: {resp.text!r}")


def mcp_get_paper(paper_id: str, fields: str) -> dict:
    data = _call_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_paper", "arguments": {"paper_id": paper_id, "fields": fields}},
    })
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return json.loads(result["content"][0]["text"])


def mcp_get_paper_batch(ids: list, fields: str) -> list:
    data = _call_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_paper_batch", "arguments": {"ids": ids, "fields": fields}},
    })
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    # get_paper_batch returns one content item per paper
    papers = []
    for item in result.get("content", []):
        if item.get("type") == "text":
            try:
                papers.append(json.loads(item["text"]))
            except json.JSONDecodeError:
                pass
    return papers


def mcp_get_citations(paper_id: str, fields: str, limit: int, date_from: str) -> list:
    data = _call_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_citations", "arguments": {
            "paper_id": paper_id, "fields": fields,
            "limit": limit, "publication_date_range": f"{date_from}:",
        }},
    })
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return result.get("structuredContent", {}).get("result", [])


def mcp_get_author_papers(author_id: str, fields: str, limit: int) -> list:
    data = _call_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_author_papers", "arguments": {
            "author_id": author_id, "paper_fields": fields, "limit": limit,
        }},
    })
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return result.get("structuredContent", {}).get("result", [])


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

_PAPER_ID_PREFIXES = ("ARXIV:", "DOI:", "CorpusId:", "MAG:", "ACL:", "PMID:", "URL:")

def _is_paper_id(value: str) -> bool:
    """Return True if value looks like a paper ID rather than a keyword query."""
    if any(value.upper().startswith(p.upper()) for p in _PAPER_ID_PREFIXES):
        return True
    # Plain hex SHA (Semantic Scholar native ID — 40 hex chars)
    if len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value):
        return True
    return False


class ResearchState(TypedDict):
    input: str          # raw CLI argument (paper ID or keyword)
    paper_id: str       # resolved paper ID (set by resolve_paper_id_node)
    seed_paper: dict
    top_references: list    # 5 most-cited references with abstracts
    citing_papers: list     # up to 5 recent citing papers
    author_profiles: list   # [{author, top_paper}, ...]
    report: str


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

def resolve_paper_id_node(state: ResearchState) -> ResearchState:
    value = state["input"]
    if _is_paper_id(value):
        print(f"[0/5] Using paper ID directly: {value}", file=sys.stderr)
        return {**state, "paper_id": value}

    print(f"[0/5] Searching for most-cited paper on topic: '{value}'...", file=sys.stderr)
    data = _call_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_papers_by_relevance", "arguments": {
            "keyword": value,
            "paper_fields": "paperId,title,citationCount",
            "limit": 20,
        }},
    })
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])

    papers = result.get("structuredContent", {}).get("result", [])
    if not papers:
        raise RuntimeError(f"No papers found for topic: '{value}'")

    best = max(papers, key=lambda p: p.get("citationCount") or 0)
    paper_id = best["paperId"]
    print(f"    → Most-cited: \"{best.get('title')}\" ({best.get('citationCount')} citations)",
          file=sys.stderr)
    print(f"    → Resolved ID: {paper_id}", file=sys.stderr)
    return {**state, "paper_id": paper_id}


def fetch_seed_node(state: ResearchState) -> ResearchState:
    print(f"[1/5] Fetching seed paper: {state['paper_id']}", file=sys.stderr)
    paper = mcp_get_paper(
        state["paper_id"],
        "title,abstract,year,authors,fieldsOfStudy,citationCount"
    )
    return {**state, "seed_paper": paper}


def fetch_references_node(state: ResearchState) -> ResearchState:
    print("[2/5] Fetching references and ranking by citation count...", file=sys.stderr)

    # Get all reference paper IDs
    ref_data = mcp_get_paper(state["paper_id"], "references")
    refs = ref_data.get("references", [])
    ref_ids = [r["paperId"] for r in refs if r.get("paperId")]

    if not ref_ids:
        return {**state, "top_references": []}

    # Batch-fetch citation counts and abstracts for all references
    batch = mcp_get_paper_batch(ref_ids, "title,abstract,citationCount,year,authors")

    # Sort by citation count descending, take top 5
    batch_sorted = sorted(
        [p for p in batch if p.get("citationCount") is not None],
        key=lambda p: p.get("citationCount", 0),
        reverse=True,
    )
    top_5 = batch_sorted[:5]

    return {**state, "top_references": top_5}


def fetch_citations_node(state: ResearchState) -> ResearchState:
    print("[3/5] Fetching recent citing papers (last 3 years)...", file=sys.stderr)

    raw = mcp_get_citations(
        state["paper_id"],
        fields="title,abstract,year,authors",
        limit=100,
        date_from="2023-01-01",
    )

    # Each item is {"citingPaper": {...}}
    citing = [r["citingPaper"] for r in raw if "citingPaper" in r]
    return {**state, "citing_papers": citing[:5]}


def fetch_author_profiles_node(state: ResearchState) -> ResearchState:
    print("[4/5] Fetching author profiles...", file=sys.stderr)

    seed_paper_id = state["seed_paper"].get("paperId", "")
    authors = state["seed_paper"].get("authors", [])
    profiles = []

    for author in authors:
        author_id = author.get("authorId")
        author_name = author.get("name", "Unknown")
        if not author_id:
            continue

        try:
            papers = mcp_get_author_papers(
                author_id,
                fields="title,citationCount,year,abstract",
                limit=100,
            )
            # Exclude the seed paper; find most-cited
            other_papers = [
                p for p in papers
                if p.get("paperId") != seed_paper_id and p.get("citationCount") is not None
            ]
            if not other_papers:
                continue
            top = max(other_papers, key=lambda p: p.get("citationCount", 0))
            profiles.append({"author": author_name, "top_paper": top})
        except Exception as exc:
            print(f"  Warning: could not fetch papers for {author_name}: {exc}", file=sys.stderr)

    return {**state, "author_profiles": profiles}


def generate_report_node(state: ResearchState) -> ResearchState:
    print("[5/5] Generating markdown report with GPT-4o mini...", file=sys.stderr)

    seed = state["seed_paper"]
    authors_str = ", ".join(a["name"] for a in seed.get("authors", []))
    fields_str = ", ".join(seed.get("fieldsOfStudy", []))

    refs_block = "\n\n".join(
        f"- **{r.get('title', 'N/A')}** ({r.get('year', 'N/A')}) — {r.get('citationCount', 'N/A')} citations\n"
        f"  Abstract: {r.get('abstract', 'N/A')}"
        for r in state["top_references"]
    )

    citing_block = "\n\n".join(
        f"- **{p.get('title', 'N/A')}** ({p.get('year', 'N/A')})\n"
        f"  Authors: {', '.join(a['name'] for a in p.get('authors', []))}\n"
        f"  Abstract: {p.get('abstract', 'N/A')}"
        for p in state["citing_papers"]
    )

    profiles_block = "\n\n".join(
        f"- **{entry['author']}**: {entry['top_paper'].get('title', 'N/A')} "
        f"({entry['top_paper'].get('year', 'N/A')}) — "
        f"{entry['top_paper'].get('citationCount', 'N/A')} citations"
        for entry in state["author_profiles"]
    )

    prompt = f"""You are an academic research assistant. Using the data below, write a structured markdown report.

## Seed Paper
Title: {seed.get('title')}
Year: {seed.get('year')}
Authors: {authors_str}
Fields of Study: {fields_str}
Citations: {seed.get('citationCount')}
Abstract: {seed.get('abstract')}

## Top 5 Most-Cited References
{refs_block}

## 5 Recent Citing Papers (2023–present)
{citing_block}

## Author Profiles (most-cited other work per author)
{profiles_block}

---

Write a markdown report with exactly these five sections:
1. A one-paragraph summary of the seed paper
2. A "Foundational Works" section describing the 5 key references and why they matter
3. A "Recent Developments" section summarizing the 5 citing papers
4. An "Author Profiles" section listing each author's most notable other work
5. A "Research Gaps" section: compare what the foundational references were solving \
with what the recent citing papers are working on. Identify 3–5 open problems or \
under-explored directions that sit between the two bodies of work.

Be concise and informative. Use markdown formatting throughout."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    report = response.choices[0].message.content
    return {**state, "report": report}


# ---------------------------------------------------------------------------
# Build and compile the LangGraph pipeline
# ---------------------------------------------------------------------------

_workflow = StateGraph(ResearchState)
_workflow.add_node("resolve_paper_id", resolve_paper_id_node)
_workflow.add_node("fetch_seed", fetch_seed_node)
_workflow.add_node("fetch_references", fetch_references_node)
_workflow.add_node("fetch_citations", fetch_citations_node)
_workflow.add_node("fetch_author_profiles", fetch_author_profiles_node)
_workflow.add_node("generate_report", generate_report_node)

_workflow.set_entry_point("resolve_paper_id")
_workflow.add_edge("resolve_paper_id", "fetch_seed")
_workflow.add_edge("fetch_seed", "fetch_references")
_workflow.add_edge("fetch_references", "fetch_citations")
_workflow.add_edge("fetch_citations", "fetch_author_profiles")
_workflow.add_edge("fetch_author_profiles", "generate_report")
_workflow.add_edge("generate_report", END)

graph = _workflow.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_input = sys.argv[1] if len(sys.argv) > 1 else "ARXIV:2210.03629"
    result = graph.invoke({
        "input": user_input,
        "paper_id": "",
        "seed_paper": {},
        "top_references": [],
        "citing_papers": [],
        "author_profiles": [],
        "report": "",
    })
    print(result["report"])
