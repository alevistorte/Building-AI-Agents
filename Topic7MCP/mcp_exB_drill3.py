
import requests
import os
import json

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
PAPER_ID = "ARXIV:1810.04805"
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "get_paper",
        "arguments": {
            "paper_id": PAPER_ID,
            "fields": "title,references.title,references.year"
        }
    }
}

print(f"Querying references for paper ID: {PAPER_ID}...\n")

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

paper = json.loads(data["result"]["content"][0]["text"])
references = paper.get("references", [])

# Sort by year ascending (papers with no year go last)
references.sort(key=lambda p: p.get("year") or 9999)

print(f"Total references: {len(references)}\n")
print("References sorted by year (ascending):")
for ref in references:
    year = ref.get("year", "N/A")
    title = ref.get("title", "N/A")
    print(f"  {year}  {title}")
