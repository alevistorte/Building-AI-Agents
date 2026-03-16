import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import gradio as gr

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ASTA_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar's academic paper "
    "database via the Asta MCP server. You can search for papers, retrieve paper details, "
    "find citations, and explore author publications. Always use the available tools to "
    "answer questions about academic research. "
    "When calling tools that accept a 'limit' parameter, always use the maximum value of "
    "100 to ensure complete results are returned."
)


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


def get_asta_tools() -> list:
    """Fetch tool schemas from MCP and convert to OpenAI function-calling format."""
    data = _call_mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = []
    for tool in data["result"]["tools"]:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return tools


def call_asta_tool(name: str, arguments: dict) -> str:
    """Execute a tools/call and return the text content."""
    try:
        data = _call_mcp({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        result = data.get("result", {})
        if result.get("isError"):
            return f"Tool error: {result['content'][0]['text']}"
        texts = [
            item["text"]
            for item in result.get("content", [])
            if item.get("type") == "text"
        ]
        return "\n".join(texts)
    except Exception as exc:
        return f"Error calling tool {name}: {exc}"


# Load tools once at startup
ASTA_TOOLS = get_asta_tools()


# ---------------------------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: List[dict]
    tool_log: List[str]


def agent_node(state: AgentState) -> AgentState:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=state["messages"],
        tools=ASTA_TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message
    msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return {"messages": state["messages"] + [msg_dict], "tool_log": state["tool_log"]}


def tools_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    new_messages = list(state["messages"])
    tool_log = list(state["tool_log"])

    for tc in last.get("tool_calls", []):
        name = tc["function"]["name"]
        arguments = json.loads(tc["function"]["arguments"])

        log_entry = f"Tool: {name}\nArguments: {json.dumps(arguments, indent=2)}"
        tool_log.append(log_entry)
        print(f"\n>>> {log_entry}")

        result = call_asta_tool(name, arguments)
        new_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    return {"messages": new_messages, "tool_log": tool_log}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.get("tool_calls") else END


# Build and compile the graph
_workflow = StateGraph(AgentState)
_workflow.add_node("agent", agent_node)
_workflow.add_node("tools", tools_node)
_workflow.set_entry_point("agent")
_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
_workflow.add_edge("tools", "agent")
graph = _workflow.compile()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def chat(user_message: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        messages.append({"role": "user", "content": entry["content"] if isinstance(entry, dict) else entry[0]})
        assistant_content = entry.get("content", "") if isinstance(entry, dict) else entry[1]
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": user_message})

    result = graph.invoke({"messages": messages, "tool_log": []})

    final_answer = ""
    for msg in reversed(result["messages"]):
        if msg["role"] == "assistant" and msg.get("content"):
            final_answer = msg["content"]
            break

    if result["tool_log"]:
        tool_info = "\n\n---\n**Tools called:**\n" + "\n\n".join(
            f"```\n{log}\n```" for log in result["tool_log"]
        )
        final_answer += tool_info

    return final_answer


def save_chat(history: list) -> str:
    if not history:
        return "No conversation to save."
    lines = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]\n{content}\n")
    path = "chat_export.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


with gr.Blocks(title="Asta Research Assistant", css="footer {display:none}") as demo:
    gr.Markdown(
        "# Asta Research Assistant\n"
        "Powered by **GPT-4o mini** + **Semantic Scholar** via Asta MCP"
    )

    with gr.Row():
        with gr.Column(scale=1):
            chatbot_state = gr.State([])
            chat_ui = gr.ChatInterface(
                fn=chat,
                type="messages",
                fill_height=True,
                examples=[
                    "Find recent papers about large language model agents",
                    "Who wrote Attention is All You Need and what else have they published?",
                    "What papers cite the original BERT paper?",
                    "Summarize the references used in the ReAct paper",
                ],
            )

    with gr.Row():
        save_btn = gr.Button("Save chat to TXT", variant="secondary")
        download = gr.File(label="Download", visible=False)

    save_btn.click(
        fn=save_chat,
        inputs=chat_ui.chatbot,
        outputs=download,
    ).then(fn=lambda: gr.File(visible=True), outputs=download)

if __name__ == "__main__":
    demo.launch()
