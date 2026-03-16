
import requests
import os
import json

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_papers_by_relevance",
        "arguments": {
            "keyword": "large language model agents",
            "paper_fields": "title,abstract,year,authors",
            "limit": 5
        }
    }
}

resp = requests.post(
    "https://asta-tools.allen.ai/mcp/v1",
    headers=headers,
    json=payload
)

# The server responds with SSE; extract JSON from the first "data:" line
data = None
for line in resp.text.splitlines():
    if line.startswith("data:"):
        data = json.loads(line[len("data:"):].strip())
        break

if data is None:
    raise ValueError(f"No data in SSE response: {resp.text!r}")

papers = data["result"]["structuredContent"]["result"]

for search_result in papers:
    get_paper_info = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_paper",
            "arguments": {
                "paper_id": search_result["paperId"],
                "fields": "title,abstract,year,authors"
            }
        }
    }
    paper_info = requests.post(
        "https://asta-tools.allen.ai/mcp/v1",
        headers=headers,
        json=get_paper_info
    )

    paper_data = None
    for line in paper_info.text.splitlines():
        if line.startswith("data:"):
            paper_data = json.loads(line[len("data:"):].strip())
            break

    if paper_data is None:
        raise ValueError(f"No data in SSE response: {paper_info.text!r}")

    paper = json.loads(paper_data["result"]["content"][0]["text"])

    print(f"Title:    {paper['title']}")
    print(f"Year:     {paper['year']}")
    print(f"Authors:  {', '.join(a['name'] for a in paper['authors'])}")
    print(f"Abstract: {paper['abstract']}")
    print()
