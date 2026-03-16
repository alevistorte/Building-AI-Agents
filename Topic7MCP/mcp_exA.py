'''
Which tool would you use to find all papers about "transformer attention mechanisms"? 
To answer this question, you would likely want to use the "snippet_search" tool, which allows you to search for text snippets that closely match a given query. By inputting the query "transformer attention mechanisms," you can find relevant papers and information related to that topic.

Tool: snippet_search
  Description: Search for text snippets that most closely match the query.
  Required: query (string)

------------------------------  

Which would you use to find who else published in the same area as a specific author?

First: 
Tool: search_authors_by_name
  Description: Search for authors by name.
  Required: name (string)
  Optional: fields (string), limit (integer)

Second: 
Tool: get_author_papers
  Description: Get papers written by this author.
  Required: author_id (string)
  Optional: paper_fields (string), limit (integer), publication_date_range (string)

Third: for the  X papers returned by get_author_papers run 
Tool: get_citations
  Description: Get details about the papers that cite this paper (i.e. papers in whose bibliography this paper appears)
  Required: paper_id (string)
  Optional: fields (string), limit (integer), publication_date_range (string)

Finally: from all the papers returned, run:
Tool: get_paper
  Description: Get details about a paper by its id.
  Required: paper_id (string)
  Optional: fields (string)

And get the authors of those papers to find who else published in the same area as the specific author.
'''

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
    "id": 1,
    "method": "tools/list",
    "params": {}
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

tools = data["result"]["tools"]

print(tools[0].keys())

for tool in tools:
    schema = tool.get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    optional = [k for k in properties if k not in required]

    req_str = ", ".join(
        f"{k} ({properties[k].get('type', 'any')})" for k in required)
    opt_str = ", ".join(
        f"{k} ({properties[k].get('type', 'any')})" for k in optional)

    print(f"Tool: {tool['name']}")
    print(f"  Description: {tool['description'].strip().splitlines()[0]}")
    if req_str:
        print(f"  Required: {req_str}")
    if opt_str:
        print(f"  Optional: {opt_str}")
    print()
