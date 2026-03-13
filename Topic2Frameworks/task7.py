# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a multi-party chat agent.
# Two LLMs participate alongside the human in a shared conversation:
#   - meta-llama/Llama-3.2-1B-Instruct  (default; responds to all non-Qwen input)
#   - Qwen/Qwen2.5-1.5B-Instruct        (activated when input starts with "Hey Qwen")
#
# Chat history is maintained as a shared, speaker-labelled log.  Because the
# Message API only supports roles system/user/assistant/tool, both the human
# and the "other" LLM are presented to each model under the "user" role, with
# the speaker's name prepended to the content.  The model being addressed sees
# its own past turns under the "assistant" role.  A system prompt informs each
# model of the participants and instructs it to prefix its replies with its name.
#
# The LLMs use Cuda if available, MPS (Apple Silicon) if available, else CPU.
# After the graph is built, a Mermaid PNG of the graph is saved to lg_graph.png.
#
# Crash recovery via LangGraph checkpointing:
# State is saved to a SQLite database (chat_sessions.db) after every node.
# On restart the program lists existing sessions and lets the user resume one,
# picking up exactly where the conversation left off.

import sqlite3
import torch
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

SCRIPT_DIR = Path(__file__).parent

# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device() -> str:
    """Return the best available compute device: 'cuda', 'mps', or 'cpu'."""
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

class SharedMessage(TypedDict):
    """
    One entry in the shared conversation log.

    Fields:
    - speaker: Who produced this message — "Human", "Llama", or "Qwen"
    - content: The message text, stored WITHOUT a name prefix.
               Name prefixes are added dynamically when building each model's
               tailored message list.
    """
    speaker: str
    content: str


class AgentState(TypedDict):
    """
    State object that flows through all LangGraph nodes.

    Fields:
    - user_input:   Raw text typed by the user (set by get_user_input)
    - should_exit:  True when the user wants to quit
    - verbose:      When True, nodes print [TRACE] lines; toggled with
                    'verbose' / 'quiet' commands
    - llm_response: The latest response text from whichever model just ran
    - history:      Ordered list of SharedMessage dicts representing the full
                    multi-party conversation so far.  Both models and the human
                    contribute to this single list.

    State Flow:
    1. history=[],  all other fields empty/default
    2. get_user_input  -> appends {speaker:"Human", content:...} to history
    3. call_llama OR call_qwen -> builds tailored message list, invokes model,
                                  stores clean response in llm_response,
                                  appends {speaker:"Llama"|"Qwen", content:...}
    4. print_response  -> prints llm_response; no state change

    Graph (3-way conditional from get_user_input):
        get_user_input -> [conditional] -> call_llama -> print_response -+
                              |         -> call_qwen  ->        ^        |
                              |                                 |        |
                              +-> get_user_input (control cmds)-+        |
                              +-> END (quit)                             |
                              <-----------------------------------------+
    """
    user_input:   str
    should_exit:  bool
    verbose:      bool
    llm_response: str
    history:      List[SharedMessage]

# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_hf_llm(model_id: str, device: str) -> ChatHuggingFace:
    """
    Download a causal-LM from HuggingFace Hub, build a text-generation
    pipeline, and return a ChatHuggingFace wrapper.

    ChatHuggingFace accepts a list of LangChain BaseMessage objects and uses
    the model's tokenizer chat template to convert them into the correct token
    sequence.  return_full_text=False ensures only newly generated tokens are
    returned, not the formatted input prompt.
    """
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)  # type: ignore[arg-type]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,   # return only newly generated tokens
    )

    chat_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
    print(f"  {model_id} loaded successfully!")
    return chat_model


def create_llama_llm(device: str) -> ChatHuggingFace:
    """Load meta-llama/Llama-3.2-1B-Instruct."""
    return _load_hf_llm("meta-llama/Llama-3.2-1B-Instruct", device)


def create_qwen_llm(device: str) -> ChatHuggingFace:
    """Load Qwen/Qwen2.5-1.5B-Instruct."""
    return _load_hf_llm("Qwen/Qwen2.5-1.5B-Instruct", device)

# =============================================================================
# GRAPH
# =============================================================================

def create_graph(llama_llm: ChatHuggingFace, qwen_llm: ChatHuggingFace,
                 log_path: Path, checkpointer: SqliteSaver):
    """
    Build the LangGraph state graph with four nodes:
      1. get_user_input  – read stdin; append HumanMessage to shared history
      2. call_llama      – build Llama's tailored message list; invoke; store result
      3. call_qwen       – same for Qwen
      4. print_response  – print the latest llm_response; append turn to log file

    Routing: input starting with "hey qwen" (case-insensitive) → call_qwen;
             everything else → call_llama.

    Args:
        llama_llm:    ChatHuggingFace wrapper for Llama-3.2-1B-Instruct
        qwen_llm:     ChatHuggingFace wrapper for Qwen2.5-1.5B-Instruct
        log_path:     Path to the chat log file; each completed turn is appended
        checkpointer: SqliteSaver instance; state is persisted after every node
                      so the graph can resume from the last checkpoint on restart
    """

    # -------------------------------------------------------------------------
    # Helper: build a model-specific message list from the shared history.
    #
    # The Message API only has system / user / assistant roles.  We handle the
    # three-party conversation as follows:
    #   - SystemMessage:  tells the model its name and the other participants
    #   - AIMessage:      the model's OWN previous turns  (role: assistant)
    #   - HumanMessage:   the human's AND the other LLM's turns  (role: user)
    #
    # Content always includes a "SpeakerName: " prefix so the model understands
    # who said what, even for turns collapsed into the "user" role.
    # -------------------------------------------------------------------------
    def build_messages(model_name: str,
                       other_name: str,
                       history: List[SharedMessage]) -> List[BaseMessage]:
        """
        Convert the shared history into a tailored list of LangChain messages
        for `model_name`.

        Args:
            model_name: The model being addressed ("Llama" or "Qwen")
            other_name: The other LLM in the conversation
            history:    Shared conversation log (SharedMessage dicts)

        Returns:
            List starting with SystemMessage, followed by one message per
            history entry.
        """
        system_content = (
            f"You are {model_name}, a helpful AI assistant. "
            f"Your job is to respond to the Human. "
            f"The conversation log may also contain messages from {other_name} "
            f"(another AI); treat those as background context only. "
            f"Never address {other_name} directly and never ask {other_name} "
            f"any questions. Only respond to the Human. "
            f"Always begin your reply with '{model_name}:'."
        )
        messages: List[BaseMessage] = [SystemMessage(content=system_content)]

        for entry in history:
            speaker = entry["speaker"]
            # Prepend the speaker's name so the model can distinguish voices
            labelled = f"{speaker}: {entry['content']}"
            if speaker == model_name:
                # Own past turns → assistant role
                messages.append(AIMessage(content=labelled))
            else:
                # Human and other LLM → user role
                messages.append(HumanMessage(content=labelled))

        return messages

    def log_turn(human_entry: SharedMessage, llm_entry: SharedMessage) -> None:
        """Append one completed turn (Human + LLM) to the log file."""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{human_entry['speaker']}] {human_entry['content']}\n")
            f.write(f"[{llm_entry['speaker']}] {llm_entry['content']}\n\n")

    def strip_name_prefix(text: str, speaker: str) -> str:
        """
        Remove a leading 'SpeakerName:' prefix from model output if present.
        The model is instructed to add it, but we store content without it so
        that the prefix is always added consistently by build_messages().
        """
        stripped = text.strip()
        prefix = f"{speaker}:"
        if stripped.lower().startswith(prefix.lower()):
            return stripped[len(prefix):].lstrip()
        return stripped

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Prompt the user for input.

        For valid (non-control) input, appends a SharedMessage with
        speaker="Human" to the history so all subsequent model invocations
        see the latest human turn.

        Updates state:
            user_input, should_exit, verbose — always
            history                          — extended for real input only
        """
        verbose = state.get("verbose", False)
        history = list(state.get("history", []))

        if verbose:
            print(f"[TRACE] get_user_input | verbose={verbose} | "
                  f"history={len(history)} msgs")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if verbose:
            print(f"[TRACE] received: {user_input!r}")

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "verbose": verbose}

        if user_input.lower() == "verbose":
            print("[TRACE] Verbose mode enabled.")
            return {"user_input": user_input, "should_exit": False, "verbose": True}

        if user_input.lower() == "quiet":
            print("Verbose mode disabled.")
            return {"user_input": user_input, "should_exit": False, "verbose": False}

        if not user_input.strip():
            print("Please enter some text.")
            return {"user_input": "", "should_exit": False, "verbose": verbose}

        # Valid input — record in shared history before routing to an LLM
        updated_history = history + [{"speaker": "Human", "content": user_input}]
        if verbose:
            target = "call_qwen" if _routes_to_qwen(user_input) else "call_llama"
            print(f"[TRACE] routing to {target}")
        return {
            "user_input":  user_input,
            "should_exit": False,
            "verbose":     verbose,
            "history":     updated_history,
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Invoke Llama with the full shared history translated into its tailored
        message list.  Llama's own past turns appear as AIMessage; Human and
        Qwen turns appear as HumanMessage.

        Updates state:
            llm_response — Llama's reply text (name prefix stripped)
            history      — extended with {speaker:"Llama", content:...}
        """
        verbose = state.get("verbose", False)
        history = list(state["history"])

        if verbose:
            print(f"[TRACE] call_llama | history={len(history)} msgs")

        messages = build_messages("Llama", "Qwen", history)

        if verbose:
            print(f"[TRACE] call_llama sending {len(messages)} messages")

        print("\n[Llama] Processing...")
        ai_msg: AIMessage = llama_llm.invoke(messages)
        response = strip_name_prefix(str(ai_msg.content), "Llama")

        if verbose:
            print(f"[TRACE] call_llama response: {response!r}")

        return {
            "llm_response": response,
            "history":      history + [{"speaker": "Llama", "content": response}],
        }

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Invoke Qwen with the full shared history translated into its tailored
        message list.  Qwen's own past turns appear as AIMessage; Human and
        Llama turns appear as HumanMessage.

        Updates state:
            llm_response — Qwen's reply text (name prefix stripped)
            history      — extended with {speaker:"Qwen", content:...}
        """
        verbose = state.get("verbose", False)
        history = list(state["history"])

        if verbose:
            print(f"[TRACE] call_qwen | history={len(history)} msgs")

        messages = build_messages("Qwen", "Llama", history)

        if verbose:
            print(f"[TRACE] call_qwen sending {len(messages)} messages")

        print("\n[Qwen] Processing...")
        ai_msg: AIMessage = qwen_llm.invoke(messages)
        response = strip_name_prefix(str(ai_msg.content), "Qwen")

        if verbose:
            print(f"[TRACE] call_qwen response: {response!r}")

        return {
            "llm_response": response,
            "history":      history + [{"speaker": "Qwen", "content": response}],
        }

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Print the response that was just produced.  Reads llm_response and the
        last history entry to know which model spoke.  No state changes.
        """
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] print_response | history={len(state.get('history', []))} msgs")

        history = state.get("history", [])
        # The last entry is the model's reply we just appended in call_llama/call_qwen
        speaker = history[-1]["speaker"] if history else "LLM"

        print(f"\n{'=' * 50}")
        print(f"{speaker} response:")
        print("-" * 50)
        print(state["llm_response"])
        print("=" * 50)

        # Log the completed turn: history[-2] = Human message, history[-1] = LLM reply
        if len(history) >= 2:
            log_turn(history[-2], history[-1])

        if verbose:
            print(f"[TRACE] print_response done, looping back to get_user_input")

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def _routes_to_qwen(text: str) -> bool:
        """Return True if the input should be directed to Qwen."""
        return text.lower().startswith("hey qwen")

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END

        user_input = state.get("user_input", "")
        if not user_input.strip() or user_input.lower() in ("verbose", "quiet"):
            return "get_user_input"

        return "call_qwen" if _routes_to_qwen(user_input) else "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama",     call_llama)
    graph_builder.add_node("call_qwen",      call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama":     "call_llama",
            "call_qwen":      "call_qwen",
            "get_user_input": "get_user_input",
            END:               END,
        }
    )

    graph_builder.add_edge("call_llama",     "print_response")
    graph_builder.add_edge("call_qwen",      "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    # Attach the checkpointer so LangGraph persists state to SQLite after
    # every node.  On restart, invoke() with the same thread_id will find the
    # checkpoint and resume from the next pending node automatically.
    return graph_builder.compile(checkpointer=checkpointer)


def save_graph_image(graph, filename="lg_graph.png"):
    """Save a Mermaid PNG of the graph alongside this script."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        output_path = SCRIPT_DIR / filename
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {output_path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install: pip install grandalf")


def list_sessions(db_path: Path) -> List[str]:
    """
    Return thread IDs of existing sessions stored in the checkpoint database,
    ordered newest-first.  Returns an empty list if the database does not yet
    exist or has no sessions.
    """
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


def select_session(db_path: Path) -> tuple:
    """
    If existing sessions are found in the checkpoint database, display them
    and let the user choose one to resume.  Otherwise (or if the user presses
    Enter without a selection) create a new session with a timestamp-based ID.

    Returns:
        (thread_id: str, is_new: bool)
            thread_id – identifier for this run (used as LangGraph thread_id)
            is_new    – True when starting a brand-new session
    """
    sessions = list_sessions(db_path)

    if sessions:
        print("\nExisting sessions found:")
        for i, session_id in enumerate(sessions, 1):
            log = db_path.parent / f"chat_log_{session_id}.txt"
            turns = log.read_text(encoding="utf-8").count("\n\n") if log.exists() else 0
            print(f"  {i}. {session_id}  ({turns} turn(s) logged)")
        print()
        choice = input("Enter a number to resume, or press Enter to start new: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx], False

    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return new_id, True


def main():
    """
    Start the multi-party chat agent with crash recovery via checkpointing.

    Checkpointing:
        LangGraph saves the full AgentState to chat_sessions.db after every
        node completes.  On restart, select_session() lists existing sessions
        and the user can resume one.  graph.invoke() with the same thread_id
        and an empty input dict ({}) resumes from the last saved checkpoint,
        so nothing is lost even if the process was killed mid-turn.

    Session files (all in the same directory as this script):
        chat_sessions.db          – SQLite checkpoint database (all sessions)
        chat_log_<thread_id>.txt  – Human-readable turn log for each session

    Shared history:
        Human turn  : get_user_input appends {speaker:"Human",  content:...}
        Llama turn  : call_llama     appends {speaker:"Llama",  content:...}
        Qwen turn   : call_qwen      appends {speaker:"Qwen",   content:...}
    """
    print("=" * 50)
    print("LangGraph Multi-Party Chat: Human + Llama + Qwen")
    print("  Start with 'Hey Qwen' to address Qwen.")
    print("  All other input goes to Llama.")
    print("=" * 50)
    print()

    # -------------------------------------------------------------------------
    # Session selection: resume an existing session or start a new one.
    # The checkpoint database is shared across all sessions.
    # -------------------------------------------------------------------------
    db_path = SCRIPT_DIR / "chat_sessions.db"
    thread_id, is_new = select_session(db_path)
    log_path = SCRIPT_DIR / f"chat_log_{thread_id}.txt"

    if is_new:
        print(f"\nStarting new session: {thread_id}")
    else:
        print(f"\nResuming session: {thread_id}")
        if log_path.exists():
            print("\n--- Previous conversation ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            print("--- Resuming from here ---")

    print(f"Chat log: {log_path}")

    # -------------------------------------------------------------------------
    # Load models and build graph.
    # -------------------------------------------------------------------------
    device = get_device()

    print("\nLoading Llama-3.2-1B-Instruct...")
    llama_llm = create_llama_llm(device)

    print("\nLoading Qwen2.5-1.5B-Instruct...")
    qwen_llm = create_qwen_llm(device)

    # Open the SQLite checkpoint database.  check_same_thread=False is required
    # because LangGraph may access the connection from a worker thread.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm, log_path, checkpointer)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # config identifies which checkpoint thread to read from / write to
    config = {"configurable": {"thread_id": thread_id}}

    if is_new:
        # New session: supply the full initial state
        initial_state: AgentState = {
            "user_input":   "",
            "should_exit":  False,
            "verbose":      False,
            "llm_response": "",
            "history":      [],
        }
        graph.invoke(initial_state, config=config)
    else:
        # Resumed session: pass an empty dict so the checkpoint state is used
        # as-is and the graph continues from the last saved node.
        graph.invoke({}, config=config)


if __name__ == "__main__":
    main()
