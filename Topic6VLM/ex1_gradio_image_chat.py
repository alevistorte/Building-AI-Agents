# gradio_image_chat.py
"""
Gradio web interface for multi-turn image chat using LLaVA via Ollama,
orchestrated with LangGraph.

Prerequisites:
    pip install gradio ollama langgraph
    ollama serve          # start the Ollama daemon
    ollama pull llava     # download LLaVA (first time only)

Usage:
    python gradio_image_chat.py
    Then open http://localhost:7860 in your browser.

Graph structure (single-turn, invoked once per user message):
    START -> call_llm -> END
    Gradio drives the interaction loop; this graph handles one turn at a time.
"""

import gradio as gr
import ollama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State flowing through the LangGraph nodes.

    - user_input:  The user's current question.
    - messages:    Full conversation history in ollama format.
                   Each entry: {'role': str, 'content': str, ['images': list]}
                   The image is attached only in the first user message.
    - image_path:  Filepath of the uploaded image (used on the first turn).
    """
    user_input: str
    messages: list
    image_path: str


# =============================================================================
# GRAPH DEFINITION
# =============================================================================

def create_graph():
    """
    Build a single-turn LangGraph compiled graph.

    Unlike a CLI agent that loops inside graph.invoke(), this graph processes
    exactly one user message per invocation. Gradio handles the conversation
    loop by passing the accumulated message history back in on every turn.
    """

    def call_llm(state: AgentState) -> dict:
        """
        Reads state:   user_input, messages, image_path
        Updates state: messages  (appends new user + assistant dicts)

        The image is attached only when the conversation history is empty
        (i.e., the very first question). Subsequent turns send the full
        text history so LLaVA retains context without re-encoding the image.
        """
        messages = list(state.get("messages", []))   # copy; never mutate state
        image_path = state.get("image_path", "")

        # Build the new user message dict.
        new_user_msg: dict = {"role": "user", "content": state["user_input"]}

        # Attach the image only on the first turn (empty history).
        if not messages and image_path:
            new_user_msg["images"] = [image_path]

        messages.append(new_user_msg)

        # Call LLaVA through the Ollama Python library.
        response = ollama.chat(model="llava", messages=messages)
        assistant_text = response["message"]["content"]

        messages.append({"role": "assistant", "content": assistant_text})

        return {"messages": messages}

    builder = StateGraph(AgentState)
    builder.add_node("call_llm", call_llm)
    builder.add_edge(START, "call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile()


# Build once at module load; reuse across all Gradio sessions.
graph = create_graph()


# =============================================================================
# GRADIO CALLBACKS
# =============================================================================

def on_image_upload(image_path):
    """
    Triggered when the user uploads or removes an image.
    Resets the entire conversation so the new image is used from turn 1.

    Returns:
        chat_history  – cleared list (resets the Chatbot component)
        agent_messages – cleared list (resets the LangGraph history state)
        status        – human-readable status string
    """
    if image_path:
        status = f"Image ready. You can now ask questions about it."
    else:
        status = "No image loaded. You can upload one or ask a question."
    return [], [], status


def respond(user_message: str, image_path, chat_history: list, agent_messages: list):
    """
    Triggered when the user clicks Send or presses Enter.

    Parameters
    ----------
    user_message   : text typed by the user
    image_path     : filepath returned by gr.Image (None if no image uploaded)
    chat_history   : Gradio Chatbot history  [[user_msg, assistant_msg], ...]
    agent_messages : internal LangGraph message list (persisted in gr.State)

    Returns
    -------
    Updated chat_history, updated agent_messages, empty string (clears input).
    """
    user_message = user_message.strip()

    if not user_message:
        return chat_history, agent_messages, ""

    # If no image is loaded and there is no prior conversation, skip the LLM
    # call and tell the user to upload an image first.
    if image_path is None and not agent_messages:
        reply = "Please upload an image first so I have something to look at!"
        return chat_history + [[user_message, reply]], agent_messages, ""

    try:
        result = graph.invoke({
            "user_input":  user_message,
            "messages":    agent_messages,
            "image_path":  image_path,
        })
        updated_messages = result["messages"]
        assistant_reply = updated_messages[-1]["content"]
    except Exception as exc:
        assistant_reply = f"Error communicating with LLaVA: {exc}"
        updated_messages = agent_messages   # keep history unchanged on error

    return chat_history + [[user_message, assistant_reply]], updated_messages, ""


def clear_conversation():
    """
    Clears the chat display, the internal message history, and the text box.
    The uploaded image is intentionally preserved so the user can keep asking
    about the same image after clearing.
    """
    return [], [], ""


# =============================================================================
# GRADIO UI
# =============================================================================

def build_ui():
    with gr.Blocks(title="LLaVA Image Chat", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
            # LLaVA Image Chat Agent
            Optionally upload an image, then ask questions about it in natural language.
            The model retains the full conversation so you can ask follow-up
            questions without repeating yourself.

            > **Requires:** `ollama serve` running locally with the `llava` model pulled.
            """
        )

        # ---- Persistent per-session state ------------------------------------
        # Holds the LangGraph message list across Gradio callbacks.
        agent_messages = gr.State([])

        # ---- Layout ----------------------------------------------------------
        with gr.Row():

            # Left column: image upload + controls
            with gr.Column(scale=1, min_width=300):
                image_input = gr.Image(
                    type="filepath",
                    label="Upload Image",
                    height=320,
                )
                status_box = gr.Textbox(
                    value="No image loaded. You can upload one or ask a question.",
                    label="Status",
                    interactive=False,
                    lines=2,
                )
                clear_btn = gr.Button(
                    "Clear Conversation", variant="secondary")

            # Right column: chat interface
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=460,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about the image…",
                        show_label=False,
                        lines=1,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # ---- Event wiring ----------------------------------------------------

        # New image uploaded or removed -> reset conversation
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[chatbot, agent_messages, status_box],
        )

        # Send button clicked
        send_btn.click(
            fn=respond,
            inputs=[msg_input, image_input, chatbot, agent_messages],
            outputs=[chatbot, agent_messages, msg_input],
        )

        # Enter key pressed in the text box
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, image_input, chatbot, agent_messages],
            outputs=[chatbot, agent_messages, msg_input],
        )

        # Clear conversation (keeps the image loaded)
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, agent_messages, msg_input],
        )

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)
