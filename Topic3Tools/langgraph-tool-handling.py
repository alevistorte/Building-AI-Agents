"""
Tool Calling with LangGraph
A persistent, multi-turn conversation agent with tool use, checkpointing, and recovery.

Architecture:
  START → get_user_input → call_agent ⇄ tools (ToolNode)
                                      ↓
                                print_response → get_user_input (loop)

The agent maintains full conversation history across turns using LangGraph's
add_messages reducer, and persists it to SQLite so sessions can be resumed
after a crash or restart.
"""

import math
import os
import json
import sqlite3
import numexpr
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


SCRIPT_DIR = Path(__file__).parent


# ============================================
# PART 1: State
# ============================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full history, grows each turn
    should_exit: bool


# ============================================
# PART 2: Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculator(input: str) -> str:
    """A calculator that evaluates math expressions and computes geometric functions.
    The 'input' parameter must be a JSON string.
    For general math, use: {"operation": "evaluate", "expression": "2 + 3 * 4"}.
    For geometry, supported operations and their required keys are:
    circle_area (radius), circle_circumference (radius),
    rectangle_area (length, width), rectangle_perimeter (length, width),
    triangle_area (base, height), sphere_volume (radius),
    sphere_surface_area (radius), cylinder_volume (radius, height),
    cone_volume (radius, height), trapezoid_area (base1, base2, height).
    Example: {"operation": "circle_area", "radius": 5}

    Accepts a JSON string as input (parsed via json.loads).
    Returns results formatted as a JSON string (via json.dumps).
    """
    try:
        data = json.loads(input)
    except (json.JSONDecodeError, TypeError):
        data = {"operation": "evaluate", "expression": str(input)}
    operation = data.get("operation", "evaluate")

    if operation == "evaluate":
        expression = data["expression"]
        result = numexpr.evaluate(expression).item()
    elif operation == "circle_area":
        result = math.pi * data["radius"] ** 2
    elif operation == "circle_circumference":
        result = 2 * math.pi * data["radius"]
    elif operation == "rectangle_area":
        result = data["length"] * data["width"]
    elif operation == "rectangle_perimeter":
        result = 2 * (data["length"] + data["width"])
    elif operation == "triangle_area":
        result = 0.5 * data["base"] * data["height"]
    elif operation == "sphere_volume":
        result = (4 / 3) * math.pi * data["radius"] ** 3
    elif operation == "sphere_surface_area":
        result = 4 * math.pi * data["radius"] ** 2
    elif operation == "cylinder_volume":
        result = math.pi * data["radius"] ** 2 * data["height"]
    elif operation == "cone_volume":
        result = (1 / 3) * math.pi * data["radius"] ** 2 * data["height"]
    elif operation == "trapezoid_area":
        result = 0.5 * (data["base1"] + data["base2"]) * data["height"]
    else:
        return json.dumps({"error": f"Unknown operation: {operation}"})

    return json.dumps({"result": round(result, 6)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count the number of occurrences of a specific letter in a piece of text.

    Use this tool when the user asks how many times a particular letter
    appears in a word, phrase, or sentence.

    Args:
        text: The text to search through.
        letter: The single letter to count (case-insensitive).
    """
    count = text.lower().count(letter.lower())
    return json.dumps({"letter": letter, "text": text, "count": count})


tools_list = [get_weather, calculator, count_letter]


# ============================================
# PART 3: LLM
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
llm_with_tools = llm.bind_tools(tools_list)


# ============================================
# PART 4: Nodes
# ============================================

def get_user_input(state: AgentState) -> dict:
    """Read one line from the user and append it as a HumanMessage."""
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return {"should_exit": True}

    if user_input.lower() in ("quit", "exit", "q", ""):
        return {"should_exit": True}

    return {"messages": [HumanMessage(content=user_input)], "should_exit": False}


def call_agent(state: AgentState) -> dict:
    """Invoke the LLM with the full message history."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def print_response(state: AgentState) -> dict:
    """Print the assistant's final answer for this turn."""
    last = state["messages"][-1]
    print(f"\nAssistant: {last.content}\n")
    return {}


# ============================================
# PART 5: Routing
# ============================================

def route_after_input(state: AgentState) -> str:
    """Exit the graph if the user quit, otherwise call the agent."""
    if state.get("should_exit"):
        return END
    return "call_agent"


# ============================================
# PART 6: Graph
# ============================================

def build_graph(checkpointer) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_agent",     call_agent)
    builder.add_node("tools",          ToolNode(tools_list))
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")

    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {"call_agent": "call_agent", END: END},
    )

    # tools_condition: "tools" if there are tool calls, END otherwise.
    # Remap END → "print_response" so the answer is shown before looping back.
    builder.add_conditional_edges(
        "call_agent",
        tools_condition,
        {"tools": "tools", END: "print_response"},
    )

    builder.add_edge("tools",          "call_agent")
    builder.add_edge("print_response", "get_user_input")

    return builder.compile(checkpointer=checkpointer)


# ============================================
# PART 7: Session Management
# ============================================

def list_sessions(db_path: Path) -> list[str]:
    """Return all thread IDs stored in the checkpoint database."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
        )
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception:
        return []


def select_session(db_path: Path) -> tuple[str, bool]:
    """
    Prompt the user to resume an existing session or start a new one.
    Returns (thread_id, is_new).
    """
    sessions = list_sessions(db_path)

    if sessions:
        print("\nExisting sessions:")
        for i, sid in enumerate(sessions, 1):
            print(f"  {i}. {sid}")
        choice = input("Enter a number to resume, or press Enter to start new: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx], False

    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return new_id, True


# ============================================
# PART 8: Main
# ============================================

if __name__ == "__main__":
    db_path = SCRIPT_DIR / "sessions.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = build_graph(checkpointer)

    thread_id, is_new = select_session(db_path)
    config = {"configurable": {"thread_id": thread_id}}

    if is_new:
        print(f"\nStarting new session: {thread_id}")
        print("Type 'quit' or 'exit' to end.\n")
        initial_state: AgentState = {
            "messages": [SystemMessage(
                content="You are a helpful assistant with access to tools for "
                        "weather, math, and letter counting. Use them when needed."
            )],
            "should_exit": False,
        }
        graph.invoke(initial_state, config=config)
    else:
        print(f"\nResuming session: {thread_id}")
        print("Type 'quit' or 'exit' to end.\n")
        graph.invoke({}, config=config)

    print("Session saved. Goodbye.")
