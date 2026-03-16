
import requests
import os
import json

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
PAPER_ID = "ARXIV:1810.04805"  # Example paper ID; replace with your desired paper ID
NUM_CITATIONS = 10
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "get_citations",
        "arguments": {
            "paper_id": PAPER_ID,
            "fields": "title,year,authors",
            "limit": NUM_CITATIONS,
            "publication_date_range": "2023-01-01:"
        }
    }
}

print(
    f"Querying the last {NUM_CITATIONS} citations for paper ID: {PAPER_ID}...\n")

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

print(f"Total citations returned: {len(papers)}\n")
print("First 5 citing papers:")
for result in papers[:5]:
    paper = result["citingPaper"]

    print(
        f"  - {paper['authors'][0]['name']} et al. {paper['title']} ({paper['year']}) ")
