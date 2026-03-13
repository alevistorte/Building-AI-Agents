# Topic 4: Exploring Tools

[This was presented in class]

## Overview

This topic demonstrates real-world tool use by building an interactive YouTube video analyzer. An LLM agent fetches video transcripts using a custom LangChain tool, generates a summary and a multiple-choice quiz, and presents them to the user. Results are cached locally to avoid redundant API calls.

Full git history: https://github.com/alevistorte/Educational-YouTube-Video-Analyzer

---

## Tasks

### yt_video_analyzer.py — YouTube Video Transcript Analysis with Q&A

**File:** `yt_video_analyzer.py`
**Description:** An interactive CLI tool that searches YouTube for a topic, lets the user pick a video, fetches its transcript via `YouTubeTranscriptApi`, and uses a `gpt-4o-mini` agent to produce a JSON-structured summary and 5-question multiple-choice quiz. Chapter metadata is extracted via `yt-dlp` (from embedded metadata or timestamp-parsed video descriptions). Results are cached per video ID in `files/questions_summary_<video_id>.json` to avoid re-running the LLM on repeated sessions. The agent is created with LangChain's `create_agent()` with a single tool bound: `get_youtube_transcript`.
**Key code snippet:**

```python
@tool
def get_youtube_transcript(video_id_or_url: str) -> str:
    """Fetch the transcript of a YouTube video by video ID or URL."""
    video_id = extract_video_id(video_id_or_url)
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    return " ".join([snippet.text for snippet in transcript.snippets])

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(llm, [get_youtube_transcript])
```

**Output:**

See `session1.txt` and `session2.txt` for details.

In `session1.txt` we can see the video of interest isn't cached, so it goes and fetch the data

```
What topic do you want to learn about? building AI agents

Searching YouTube for: building AI agents

  1) How to Build & Sell AI Agents: Ultimate Beginner’s Guide
     Channel: Liam Ottley  |  Duration: 3:50:40
  2) The Easiest Way To Build AI Agents For Beginners (So Easy)
     Channel: Paul J Lipsky  |  Duration: 13:05
  3) From Zero to Your First AI Agent in 25 Minutes (No Coding)
     Channel: Futurepedia and AI Agent Lab  |  Duration: 25:58
  4) You’re Not Behind (Yet): How to Build AI Agents in 2026 (no coding)
     Channel: Futurepedia  |  Duration: 26:05
  5) AI Agents, Clearly Explained
     Channel: Jeff Su  |  Duration: 10:09

Pick a video (1-5): 5

Selected: AI Agents, Clearly Explained
Fetching transcript and generating summary/quiz for video ID: FwOTs4UxQS4
[...]
```

On the other hands, `session2.txt` is about a video that was already visited, so it gets the info from the `json` file instead of running the pipeline again.

```
What topic do you want to learn about? building AI agents

Searching YouTube for: building AI agents

  1) The Easiest Way To Build AI Agents For Beginners (So Easy)
     Channel: Paul J Lipsky  |  Duration: 13:05
  2) How to Build & Sell AI Agents: Ultimate Beginner’s Guide
     Channel: Liam Ottley  |  Duration: 3:50:40
  3) Building AI Agents in Pure Python - Beginner Course
     Channel: Dave Ebbelaar  |  Duration: 46:44
  4) How to Build Ai Agents with Claude (Part 1)
     Channel: Upskillpm   |  Duration: 9:26
  5) AI Agents Explained: A Comprehensive Guide for Beginners
     Channel: AI Alfie   |  Duration: 6:52

Pick a video (1-5): 1

Selected: The Easiest Way To Build AI Agents For Beginners (So Easy)
Loading cached results from files/questions_summary_6hPvtbEDMrw.json
```
