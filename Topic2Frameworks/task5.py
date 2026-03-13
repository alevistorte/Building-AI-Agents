# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a simple chat agent backed by
# meta-llama/Llama-3.2-1B-Instruct.
# It maintains a multi-turn chat history using LangChain's Message API.
# Supported message roles: SystemMessage, HumanMessage (user), AIMessage (assistant).
# The LLM uses Cuda if available, MPS (Apple Silicon) if available, otherwise CPU.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write an image of the graph to the file lg_graph.png.
# The program gets the LLM from Hugging Face and wraps it for LangChain using
# HuggingFacePipeline and ChatHuggingFace.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from pathlib import Path
from typing import TypedDict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Directory of this script, used for saving output files alongside it
SCRIPT_DIR = Path(__file__).parent

# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU


def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - user_input:    The text entered by the user (set by get_user_input node)
    - should_exit:   Boolean flag indicating if user wants to quit
    - verbose:       When True, each node prints tracing info to stdout.
                     Toggled by typing 'verbose' (enable) or 'quiet' (disable).
    - llama_response: The latest response text from Llama (set by call_llama node)
    - history:       Ordered list of LangChain BaseMessage objects representing
                     the full conversation so far.  Roles used:
                       SystemMessage  – system prompt set at startup
                       HumanMessage   – one entry per user turn
                       AIMessage      – one entry per assistant turn

    State Flow:
    1. Initial state: history=[SystemMessage(...)], all other fields empty/default
    2. After get_user_input: user_input and should_exit set; HumanMessage appended
       to history for valid (non-control) input
    3. After call_llama: llama_response set; AIMessage appended to history
    4. After print_response: state unchanged (node only reads, doesn't write)

    The graph loops continuously (3-way conditional branch from get_user_input):
        get_user_input -> [conditional] -> call_llama -> print_response -> get_user_input
                              |                                                 ^
                              +-> get_user_input (empty / verbose / quiet) ----+
                              |
                              +-> END (if user wants to quit)

    Typing 'verbose' enables per-node tracing; typing 'quiet' disables it.
    Both commands, as well as empty input, re-prompt immediately without calling the LLM.
    """
    user_input: str
    should_exit: bool
    verbose: bool
    llama_response: str
    history: List[BaseMessage]


def _load_hf_llm(model_id: str, device: str) -> ChatHuggingFace:
    """
    Internal helper: download a causal-LM from HuggingFace Hub, build a
    text-generation pipeline, wrap it in HuggingFacePipeline, then wrap
    that in ChatHuggingFace so it accepts a list of LangChain messages.

    Args:
        model_id: HuggingFace Hub model identifier
        device:   Compute device string ('cuda', 'mps', or 'cpu')

    Returns:
        A ChatHuggingFace instance ready for LangChain message-list .invoke() calls.
    """
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model with the appropriate dtype for the target device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,       # Maximum tokens to generate in response
        do_sample=True,           # Enable sampling for varied responses
        temperature=0.7,          # Controls randomness (lower = more deterministic)
        top_p=0.95,               # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
        return_full_text=False,   # Return only the newly generated tokens, not the input prompt
    )

    # Wrap in HuggingFacePipeline for LangChain compatibility
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)

    # Wrap in ChatHuggingFace so we can pass a list of BaseMessage objects.
    # ChatHuggingFace uses the model's tokenizer chat template to convert the
    # message list into the correct token sequence before inference.
    chat_model = ChatHuggingFace(llm=hf_pipeline)

    print(f"  {model_id} loaded successfully!")
    return chat_model


def create_llama_llm(device: str) -> ChatHuggingFace:
    """Load meta-llama/Llama-3.2-1B-Instruct and return a ChatHuggingFace model."""
    return _load_hf_llm("meta-llama/Llama-3.2-1B-Instruct", device)


def create_graph(llama_llm):
    """
    Create the LangGraph state graph with three nodes:
    1. get_user_input:  Reads input from stdin; appends HumanMessage to history
    2. call_llama:      Sends full history to Llama; appends AIMessage to history
    3. print_response:  Prints the latest Llama response

    Graph structure:
        START -> get_user_input -> [conditional] -> call_llama -> print_response -+
                      ^                 |                                          |
                      |                 +-> get_user_input (verbose/quiet/empty)  |
                      |                 +-> END (quit)                            |
                      +------------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.

        For valid (non-control, non-empty) input, appends a HumanMessage to
        the conversation history so the LLM has context on the next turn.

        Reads state:
            - verbose: Current tracing mode (preserved across iterations)
            - history: Existing conversation history (passed through unchanged
                       for control commands; extended for real user input)
        Updates state:
            - user_input:  The raw text entered by the user
            - should_exit: True if user wants to quit, False otherwise
            - verbose:     Updated if user typed 'verbose' or 'quiet'
            - history:     Extended with HumanMessage for non-control input
        """
        verbose = state.get("verbose", False)
        history = state.get("history", [])

        if verbose:
            print(f"[TRACE] Entering node: get_user_input | verbose={verbose} "
                  f"| history length={len(history)}")

        # Display banner before each prompt
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if verbose:
            print(f"[TRACE] get_user_input received: {user_input!r}")

        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "verbose": verbose,
            }

        # Toggle tracing mode - routing will loop back without calling the LLM
        if user_input.lower() == "verbose":
            print("[TRACE] Verbose mode enabled.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True,
            }

        if user_input.lower() == "quiet":
            print("Verbose mode disabled.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": False,
            }

        # Empty input - loop back without calling the LLM
        if not user_input.strip():
            print("Please enter some text.")
            return {
                "user_input": "",
                "should_exit": False,
                "verbose": verbose,
            }

        # Valid input - append a HumanMessage so the LLM sees this turn's context
        if verbose:
            print(f"[TRACE] get_user_input appending HumanMessage and routing to call_llama")
        return {
            "user_input": user_input,
            "should_exit": False,
            "verbose": verbose,
            "history": history + [HumanMessage(content=user_input)],
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    # Invokes the model with the FULL message history (SystemMessage +
    # all previous HumanMessage/AIMessage pairs + the current HumanMessage).
    # The response is an AIMessage; it is stored in llama_response for printing
    # and appended to history so future turns retain the context.
    def call_llama(state: AgentState) -> dict:
        """
        Node that invokes Llama-3.2-1B-Instruct with the full conversation history.

        ChatHuggingFace.invoke() accepts a list of LangChain BaseMessage objects
        and uses the model's tokenizer chat template to format them correctly.

        Reads state:
            - history: Full message list ending with the latest HumanMessage
            - verbose: If True, prints tracing information
        Updates state:
            - llama_response: Plain text content of the AIMessage
            - history:        Extended with the AIMessage response
        """
        verbose = state.get("verbose", False)
        history = state["history"]

        if verbose:
            print(f"[TRACE] Entering node: call_llama | history length={len(history)}")
            print(f"[TRACE] call_llama sending {len(history)} message(s) to model")

        print("\n[Llama] Processing...")

        # Invoke with the full message list - ChatHuggingFace handles formatting
        ai_message: AIMessage = llama_llm.invoke(history)

        response_text = ai_message.content

        if verbose:
            print(f"[TRACE] call_llama received AIMessage ({len(response_text)} chars)")

        return {
            "llama_response": response_text,
            "history": history + [ai_message],
        }

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    # Reads llama_response from state and prints it.  Does not modify state.
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the LLM's response to stdout.

        Reads state:
            - llama_response: The text to print
            - verbose: If True, prints tracing information
        Updates state:
            - Nothing (returns empty dict, state unchanged)
        """
        verbose = state.get("verbose", False)

        if verbose:
            print(f"[TRACE] Entering node: print_response")

        print("\n" + "=" * 50)
        print("Llama-3.2-1B-Instruct response:")
        print("-" * 50)
        print(state["llama_response"])
        print("=" * 50)

        if verbose:
            history_len = len(state.get("history", []))
            print(f"[TRACE] print_response complete | history length={history_len} "
                  f"| routing back to get_user_input")

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.

        Examines state:
            - should_exit: If True, terminate the graph
            - user_input:  If empty, 'verbose', or 'quiet', loop back to
                           get_user_input without calling the LLM

        Returns:
            - END:              If user wants to quit
            - "get_user_input": If input is empty or a control command (verbose/quiet)
            - "call_llama":     For any other non-empty input
        """
        if state.get("should_exit", False):
            return END

        user_input = state.get("user_input", "")
        if not user_input.strip() or user_input.lower() in ("verbose", "quiet"):
            return "get_user_input"

        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    # Register all three nodes
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama",     call_llama)
    graph_builder.add_node("print_response", print_response)

    # 1. START -> get_user_input
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> call_llama | get_user_input | END
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama":     "call_llama",
            "get_user_input": "get_user_input",
            END:               END,
        }
    )

    # 3. call_llama -> print_response (always print after LLM responds)
    graph_builder.add_edge("call_llama", "print_response")

    # 4. print_response -> get_user_input (loop back for next input)
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile()

    return graph


def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Save the image in the same directory as this script
        output_path = SCRIPT_DIR / filename
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {output_path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


def main():
    """
    Main function that orchestrates the chat agent workflow:
    1. Detect the best compute device
    2. Load Llama-3.2-1B-Instruct as a ChatHuggingFace model
    3. Create the LangGraph
    4. Save the graph visualization
    5. Run the graph (it loops internally until user quits)

    Graph node flow per query:
        get_user_input -> call_llama -> print_response -> (loop)

    Chat history is maintained in state.history as a list of LangChain messages:
        SystemMessage  – set once at startup
        HumanMessage   – appended each turn by get_user_input
        AIMessage      – appended each turn by call_llama

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 50)
    print("LangGraph Chat Agent with Llama-3.2-1B-Instruct")
    print("=" * 50)
    print()

    # Step 1: Detect device
    device = get_device()

    # Step 2: Load the LLM
    print("\nLoading Llama-3.2-1B-Instruct...")
    llama_llm = create_llama_llm(device)

    # Step 3: Build the LangGraph
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm)
    print("Graph created successfully!")

    # Step 4: Save a visual representation of the graph before execution
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 5: Run the graph - it loops internally until the user quits.
    # Seed the history with a SystemMessage that establishes the assistant's persona.
    # Every subsequent turn appends HumanMessage (get_user_input) and
    # AIMessage (call_llama) so the model always receives full context.
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "verbose": False,
        "llama_response": "",
        "history": [
            SystemMessage(content="You are a helpful assistant.")
        ],
    }

    # Single invocation - the graph loops internally via print_response -> get_user_input
    # The graph only exits when route_after_input returns END (user typed quit/exit/q)
    graph.invoke(initial_state)


# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()
