
# langgraph_image_chat_agent.py
# Multi-turn conversation agent that chats about an uploaded image.
# The user provides an image path at startup. Every question is sent to
# LLaVA via Ollama together with the full conversation history so the
# model can refer back to earlier turns.
#
# Graph structure:
#   START -> get_user_input -> [conditional] -> call_llm -> print_response -+
#                  ^                 |                                       |
#                  |                 +-> END (quit/exit/q)                   |
#                  +-----------------------------------------------------------+
#
# Prerequisites:
#   ollama serve            # start the Ollama daemon
#   ollama pull llava       # download LLaVA (first time only)

import ollama
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# LangGraph automatically merges the returned dict from each node into state.


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - user_input:  Text entered by the user this turn.
    - should_exit: True when the user wants to quit.
    - messages:    Full conversation history as a list of ollama message dicts.
                   Each dict has 'role' and 'content'; the very first user
                   message also carries 'images' so LLaVA can see the photo.
    - image_path:  Absolute/relative path to the image supplied at startup.
    - verbose:     When True, trace messages are printed for each node.

    State flow per turn:
        1. get_user_input  -> populates user_input / should_exit / verbose
        2. call_llm        -> appends user + assistant dicts to messages
        3. print_response  -> reads last message from messages, prints it
        Then loops back to get_user_input.
    """
    user_input: str
    should_exit: bool
    messages: list   # list of {'role': str, 'content': str, ['images': list]}
    image_path: str
    verbose: bool


# =============================================================================
# FACTORY: get_user_input node
# =============================================================================

def make_get_user_input_node(next_node_name: str):
    """
    Returns a get_user_input node. next_node_name is used only in trace output.
    """
    def get_user_input(state: AgentState) -> dict:
        """
        Reads state:  nothing (fresh prompt each iteration)
        Updates state: user_input, should_exit, optionally verbose
        """
        verbose = state.get("verbose", False)

        if verbose:
            print("[TRACE] Entering node: get_user_input")
            print(
                f"[TRACE] Conversation turns so far: {len(state.get('messages', []))}")

        print("\n" + "=" * 50)
        print("Ask about the image (or 'quit' to exit, 'verbose'/'quiet'):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if verbose:
            print(f"[TRACE] User entered: '{user_input}'")

        if user_input.lower() == "verbose":
            print("Verbose tracing enabled.")
            return {"user_input": "", "should_exit": False, "verbose": True}

        if user_input.lower() == "quiet":
            print("Quiet mode enabled.")
            return {"user_input": "", "should_exit": False, "verbose": False}

        if user_input.lower() in ("quit", "exit", "q"):
            if verbose:
                print("[TRACE] User requested exit.")
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        if verbose:
            print(
                f"[TRACE] Exiting node: get_user_input (proceeding to {next_node_name})")

        return {"user_input": user_input, "should_exit": False}

    return get_user_input


# =============================================================================
# GRAPH FACTORY
# =============================================================================

def create_graph():
    """
    Build and compile the LangGraph for multi-turn image chat.

    Nodes:
      get_user_input  – prompts the user
      call_llm        – appends the user message (with image on first turn)
                        to history, calls ollama.chat(llava), appends reply
      print_response  – prints the latest assistant reply

    Edges:
      START -> get_user_input
      get_user_input -> call_llm  (normal input)
      get_user_input -> get_user_input  (verbose/quiet toggle, empty input)
      get_user_input -> END  (quit)
      call_llm -> print_response
      print_response -> get_user_input  (loop)
    """

    # ------------------------------------------------------------------
    # NODE 1: get_user_input
    # ------------------------------------------------------------------
    get_user_input = make_get_user_input_node("call_llm")

    # ------------------------------------------------------------------
    # NODE 2: call_llm
    # ------------------------------------------------------------------
    def call_llm(state: AgentState) -> dict:
        """
        Reads state:   user_input, messages, image_path, verbose
        Updates state: messages  (appends user + assistant messages)

        The image is included only in the first user message (messages is
        empty at that point). Subsequent turns pass the full history so
        LLaVA keeps context without re-sending the image bytes every time.
        """
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        image_path = state["image_path"]
        messages = list(state.get("messages", []))  # copy to avoid mutation

        if verbose:
            print("[TRACE] Entering node: call_llm")
            print(
                f"[TRACE] Conversation history length: {len(messages)} messages")
            print(f"[TRACE] User input: '{user_input}'")

        # Build the new user message.
        # Attach the image only on the very first turn (empty history).
        new_user_message: dict = {"role": "user", "content": user_input}
        if not messages:
            new_user_message["images"] = [image_path]
            if verbose:
                print(f"[TRACE] First turn – attaching image: {image_path}")

        messages.append(new_user_message)

        print("\nProcessing with LLaVA...")

        # Call LLaVA through the Ollama Python library.
        # Passing the full messages list gives the model conversation context.
        response = ollama.chat(model="llava", messages=messages)

        assistant_content = response["message"]["content"]

        if verbose:
            print(
                f"[TRACE] LLaVA response: '{assistant_content[:100]}...' (truncated)")
            print("[TRACE] Exiting node: call_llm")

        # Append the assistant reply to history so future turns include it.
        messages.append({"role": "assistant", "content": assistant_content})

        return {"messages": messages}

    # ------------------------------------------------------------------
    # NODE 3: print_response
    # ------------------------------------------------------------------
    def print_response(state: AgentState) -> dict:
        """
        Reads state:   messages, verbose
        Updates state: nothing

        Prints the last message in the conversation history (the assistant reply
        just added by call_llm).
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])

        if verbose:
            print("[TRACE] Entering node: print_response")
            print(f"[TRACE] Total messages in history: {len(messages)}")

        if messages:
            last = messages[-1]
            print("\n" + "-" * 50)
            print("LLaVA:")
            print("-" * 50)
            print(last["content"])

        if verbose:
            print("[TRACE] Exiting node: print_response (looping to get_user_input)")

        return {}

    # ------------------------------------------------------------------
    # ROUTING FUNCTION
    # ------------------------------------------------------------------
    def route_after_input(state: AgentState) -> str:
        """
        Returns the name of the next node:
          - END            if the user wants to quit
          - "get_user_input" if user_input is empty (verbose/quiet toggle)
          - "call_llm"     otherwise
        """
        verbose = state.get("verbose", False)

        if verbose:
            print("[TRACE] Routing: evaluating next node")

        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Routing -> END")
            return END

        if not state.get("user_input", ""):
            if verbose:
                print("[TRACE] Routing -> get_user_input (empty input)")
            return "get_user_input"

        if verbose:
            print("[TRACE] Routing -> call_llm")
        return "call_llm"

    # ------------------------------------------------------------------
    # GRAPH CONSTRUCTION
    # ------------------------------------------------------------------
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


# =============================================================================
# GRAPH VISUALISATION
# =============================================================================

def save_graph_image(graph):
    """
    Optionally save a Mermaid PNG of the graph.
    Press Enter without a name to skip.
    """
    filename = input(
        "Enter filename to save graph image (press Enter to skip): "
    ).strip()

    if not filename:
        print("Skipping graph image save.")
        return

    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        if not filename.endswith(".png"):
            filename += ".png"
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 60)
    print("Multi-turn Image Chat Agent  –  LLaVA via Ollama")
    print("=" * 60)
    print()

    # Ask for an image path and validate it before doing anything else.
    while True:
        image_path = input("Enter the path to your image: ").strip()
        if os.path.isfile(image_path):
            break
        print(f"  File not found: '{image_path}'  –  please try again.")

    print(f"Image loaded: {image_path}")
    print("You can now ask questions about the image. Type 'quit' to exit.\n")

    # Build the graph (no LLM object needed – ollama is called directly inside)
    graph = create_graph()

    # Optionally save a visualisation of the graph structure
    save_graph_image(graph)

    # Initial state: empty history, image path set, nothing to exit yet
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "messages": [],          # grows with each turn
        "image_path": image_path,
        "verbose": False,
    }

    # The graph loops internally until the user types quit/exit/q
    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
