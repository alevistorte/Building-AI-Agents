"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Context management configuration
# Max tokens to keep in context (leave room for generation)
# Llama 3.2 supports 128K tokens, but we use a smaller window for efficiency
MAX_CONTEXT_TOKENS = 4096
# Reserve tokens for the model's response
RESERVED_FOR_GENERATION = 512

# Enable or disable conversation history
# - True: Model sees full conversation history (remembers previous turns)
# - False: Model only sees the current message (no memory between turns)
ENABLE_HISTORY = True

# Context management method: "sliding_window" or "summarization"
# (Only used when ENABLE_HISTORY = True)
# - sliding_window: Drops old messages, keeps recent ones (fast, loses old context)
# - summarization: Summarizes old messages into a compact form (slower, preserves key info)
CONTEXT_METHOD = "sliding_window"

# For summarization: how many recent messages to keep verbatim (not summarized)
KEEP_RECENT_MESSAGES = 4  # Keep last 4 messages (2 user + 2 assistant turns)

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"âœ“ Model loaded! Using device: {model.device}")
print(f"âœ“ Memory usage: ~2.5 GB (FP16)\n")

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# SLIDING WINDOW FUNCTION
# ============================================================================
# This function trims old messages when the conversation exceeds the token limit.
# It always keeps:
#   1. The system prompt (first message)
#   2. The most recent messages that fit within the token budget
#
# This creates a "sliding window" effect where old context is dropped
# but recent context is preserved.


def apply_sliding_window(history, tokenizer, max_tokens):
    """
    Trim chat history to fit within max_tokens using a sliding window.

    Args:
        history: List of message dicts [{"role": ..., "content": ...}, ...]
        tokenizer: The tokenizer to count tokens
        max_tokens: Maximum tokens allowed (excluding generation reserve)

    Returns:
        Trimmed history that fits within the token budget
    """
    if len(history) <= 1:
        return history  # Only system prompt, nothing to trim

    # Always keep the system prompt
    system_prompt = [history[0]]
    messages = history[1:]  # All messages except system prompt

    # Count tokens for system prompt
    system_tokens = len(tokenizer.apply_chat_template(
        system_prompt, add_generation_prompt=False))
    available_tokens = max_tokens - system_tokens

    # Start from the most recent messages and work backwards
    # This ensures we keep the most recent context
    trimmed_messages = []
    current_tokens = 0

    for msg in reversed(messages):
        # Count tokens for this single message (approximate)
        # +10 for role tokens
        msg_tokens = len(tokenizer.encode(msg["content"])) + 10

        if current_tokens + msg_tokens <= available_tokens:
            trimmed_messages.insert(0, msg)  # Add to front (preserving order)
            current_tokens += msg_tokens
        else:
            break  # No more room

    # Combine system prompt with trimmed messages
    result = system_prompt + trimmed_messages

    # Report if we trimmed anything
    trimmed_count = len(messages) - len(trimmed_messages)
    if trimmed_count > 0:
        print(
            f"[Sliding window: trimmed {trimmed_count} old messages to fit context]")

    return result

# ============================================================================
# SUMMARIZATION FUNCTION
# ============================================================================
# This function summarizes old messages when the conversation exceeds the token limit.
# Unlike sliding window, it preserves information by compressing old context.
# It always keeps:
#   1. The system prompt (first message)
#   2. A summary of old messages (compressed)
#   3. The most recent messages verbatim (for immediate context)


# Track the running summary across turns (persists between function calls)
conversation_summary = None


def generate_summary(model, tokenizer, messages_to_summarize):
    """
    Use the model to generate a summary of the given messages.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages_to_summarize: List of message dicts to summarize

    Returns:
        A string containing the summary
    """
    # Format the messages into a readable conversation
    conversation_text = ""
    for msg in messages_to_summarize:
        role = msg["role"].capitalize()
        conversation_text += f"{role}: {msg['content']}\n"

    # Create a summarization prompt
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
        {"role": "user", "content": f"""Summarize the following conversation in 2-3 sentences.
        Focus on the key topics discussed, any important information shared, and the overall context.
        Be concise but preserve essential details.

        Conversation:
        {conversation_text}

        Summary:"""}
    ]

    # Tokenize and generate
    input_ids = tokenizer.apply_chat_template(
        summary_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,  # Keep summary short
            do_sample=False,     # Deterministic for consistency
            temperature=0.3,     # Low temperature for focused output
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the summary
    new_tokens = outputs[0][input_ids.shape[1]:]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return summary


def apply_summarization(history, model, tokenizer, max_tokens, keep_recent=4):
    """
    Manage context by summarizing old messages instead of dropping them.

    Args:
        history: List of message dicts [{"role": ..., "content": ...}, ...]
        model: The language model (needed for generating summaries)
        tokenizer: The tokenizer to count tokens
        max_tokens: Maximum tokens allowed (excluding generation reserve)
        keep_recent: Number of recent messages to keep verbatim

    Returns:
        Managed history that fits within the token budget
    """
    global conversation_summary

    if len(history) <= 1 + keep_recent:
        return history  # Not enough messages to need summarization

    # Always keep the system prompt
    system_prompt = history[0]
    messages = history[1:]  # All messages except system prompt

    # Split into old messages (to summarize) and recent messages (to keep)
    old_messages = messages[:-
                            keep_recent] if len(messages) > keep_recent else []
    recent_messages = messages[-keep_recent:] if len(
        messages) >= keep_recent else messages

    # If no old messages, nothing to summarize
    if not old_messages:
        return history

    # Count tokens for system prompt and recent messages
    test_history = [system_prompt] + recent_messages
    current_tokens = len(tokenizer.apply_chat_template(
        test_history, add_generation_prompt=False))

    # Check if we need to summarize (only if we're approaching the limit)
    # We add some buffer to avoid summarizing too frequently
    if current_tokens < max_tokens * 0.7:
        # Check if including old messages would exceed limit
        full_tokens = len(tokenizer.apply_chat_template(
            history, add_generation_prompt=False))
        if full_tokens < max_tokens:
            return history  # Everything fits, no need to summarize

    # Generate or update the summary
    print("[Summarization: compressing old messages...]", flush=True)

    if conversation_summary:
        # We have an existing summary - incorporate new old messages into it
        # Add the 2 most recently "aged out" messages
        messages_to_add = old_messages[-2:]
        combined_text = f"Previous summary: {conversation_summary}\n\nNew messages to incorporate:\n"
        for msg in messages_to_add:
            combined_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

        update_prompt = [
            {"role": "system",
                "content": "You are a helpful assistant that updates summaries."},
            {"role": "user", "content": f"""Update this conversation summary to include the new messages.
            Keep it concise (2-3 sentences) while preserving key information.

            {combined_text}

            Updated summary:"""}
        ]

        input_ids = tokenizer.apply_chat_template(
            update_prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        conversation_summary = tokenizer.decode(
            new_tokens, skip_special_tokens=True).strip()
    else:
        # First time summarizing - create initial summary
        conversation_summary = generate_summary(model, tokenizer, old_messages)

    # Build the managed history with the summary
    summary_message = {
        "role": "system",
        "content": f"[Previous conversation summary: {conversation_summary}]"
    }

    # Construct: system prompt + summary + recent messages
    managed_history = [system_prompt, summary_message] + recent_messages

    print(
        f"[Summarization: compressed {len(old_messages)} old messages into summary]")

    return managed_history

# ============================================================================
# UNIFIED CONTEXT MANAGEMENT
# ============================================================================
# This function selects the appropriate method based on configuration


def manage_context(history, model, tokenizer, max_tokens, method="sliding_window"):
    """
    Apply the configured context management method.

    Args:
        history: Full chat history
        model: The language model
        tokenizer: The tokenizer
        max_tokens: Maximum tokens allowed
        method: "sliding_window" or "summarization"

    Returns:
        Managed history that fits within token budget
    """
    if method == "sliding_window":
        return apply_sliding_window(history, tokenizer, max_tokens)
    elif method == "summarization":
        return apply_summarization(history, model, tokenizer, max_tokens, KEEP_RECENT_MESSAGES)
    else:
        print(f"[Warning: Unknown method '{method}', using sliding_window]")
        return apply_sliding_window(history, tokenizer, max_tokens)

# ============================================================================
# CHAT LOOP
# ============================================================================


print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
if ENABLE_HISTORY:
    print(f"History: ENABLED (method: {CONTEXT_METHOD})")
else:
    print("History: DISABLED (no memory between turns)")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()

    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break

    # Skip empty inputs
    if not user_input:
        continue

    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # The chat history grows with each exchange
    # We append the new user message to the existing history
    chat_history.append({
        "role": "user",
        "content": user_input
    })

    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      â† Just added
    # ]
    # This is still PLAIN TEXT

    # ========================================================================
    # STEP 3: Apply context management to prevent token overflow
    # ========================================================================
    # Based on ENABLE_HISTORY, we either use full history or just current turn
    # If history is enabled, we apply the configured context management method

    if ENABLE_HISTORY:
        # Use full conversation history with context management
        effective_max_tokens = MAX_CONTEXT_TOKENS - RESERVED_FOR_GENERATION
        managed_history = manage_context(
            chat_history, model, tokenizer, effective_max_tokens, CONTEXT_METHOD
        )
    else:
        # NO HISTORY MODE: Only use system prompt + current user message
        # The model has no memory of previous turns - each message is independent
        managed_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

    # ========================================================================
    # STEP 4: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)

    # First, apply_chat_template formats the history and converts to tokens
    input_ids = tokenizer.apply_chat_template(
        # Our MANAGED history (trimmed or summarized)
        managed_history,
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 5: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=512,              # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )

    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)

    # ========================================================================
    # STEP 6: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]

    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )

    print(assistant_response)  # Display the response

    # ========================================================================
    # STEP 7: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns

    chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              â† Just added
    # ]

    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...

    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history

    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   â†“
   Add to chat_history (text)
   â†“
   Tokenize entire chat_history (text â†’ numbers)
   â†“
   Model generates response (numbers)
   â†“
   Decode response (numbers â†’ text)
   â†“
   Add response to chat_history (text)
   â†“
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)

CONTEXT MANAGEMENT TECHNIQUES (both implemented in this demo):

1. SLIDING WINDOW (CONTEXT_METHOD = "sliding_window"):
   - We set a MAX_CONTEXT_TOKENS limit (e.g., 4096 tokens)
   - Before each generation, we check if history exceeds this limit
   - If it does, we trim OLD messages from the middle:
     * System prompt is ALWAYS kept (first message)
     * Most RECENT messages are kept (they're most relevant)
     * OLD messages in between are dropped
   - This creates a "sliding window" that moves forward with the conversation
   - Trade-off: Fast, but loses old context completely

   VISUAL REPRESENTATION:
      Full history:     [SYS] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4]
      After trimming:   [SYS]                     [U3] [A3] [U4] [A4]
                         ↑ kept                    ↑ recent kept
                         (always)                  (fits in window)

2. SUMMARIZATION (CONTEXT_METHOD = "summarization"):
   - Instead of dropping old messages, we SUMMARIZE them
   - Uses the model itself to generate a condensed summary
   - Structure after summarization:
     * System prompt (always kept)
     * Summary message (compressed old context)
     * Recent messages (kept verbatim for immediate context)
   - Trade-off: Slower (requires extra generation), but preserves key info

   VISUAL REPRESENTATION:
      Full history:     [SYS] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4]
      After summary:    [SYS] [SUMMARY of U1-A2]  [U3] [A3] [U4] [A4]
                         ↑     ↑ compressed        ↑ recent kept
                         kept  old context         (verbatim)

CHOOSING A METHOD:
- Use "sliding_window" when: speed matters, old context is less important
- Use "summarization" when: preserving context matters, can tolerate slower responses

================================================================================
COMPARISON: WITH HISTORY vs WITHOUT HISTORY (ENABLE_HISTORY flag)
================================================================================

This demonstrates WHY conversation history matters for coherent multi-turn chat.

EXAMPLE CONVERSATION:

Turn 1:
  User: "My name is Alice and I'm learning Python."

Turn 2:
  User: "What's my name?"

Turn 3:
  User: "What language am I learning?"

--------------------------------------------------------------------------------
WITH HISTORY (ENABLE_HISTORY = True):
--------------------------------------------------------------------------------
The model sees the FULL conversation each turn:

Turn 1 input:  [SYS] + "My name is Alice..."
Turn 1 output: "Nice to meet you, Alice! Python is a great choice..."

Turn 2 input:  [SYS] + "My name is Alice..." + "Nice to meet you..." + "What's my name?"
Turn 2 output: "Your name is Alice!"  ✓ CORRECT

Turn 3 input:  [SYS] + [all previous messages] + "What language am I learning?"
Turn 3 output: "You're learning Python!"  ✓ CORRECT

The model "remembers" because we feed it the entire conversation each time.

--------------------------------------------------------------------------------
WITHOUT HISTORY (ENABLE_HISTORY = False):
--------------------------------------------------------------------------------
The model sees ONLY the system prompt + current message:

Turn 1 input:  [SYS] + "My name is Alice..."
Turn 1 output: "Nice to meet you, Alice! Python is a great choice..."

Turn 2 input:  [SYS] + "What's my name?"  (NO previous context!)
Turn 2 output: "I don't know your name. You haven't told me."  ✗ NO MEMORY

Turn 3 input:  [SYS] + "What language am I learning?"  (NO previous context!)
Turn 3 output: "I don't have that information..."  ✗ NO MEMORY

Each turn is INDEPENDENT - the model has no idea what was said before.

--------------------------------------------------------------------------------
KEY INSIGHT:
--------------------------------------------------------------------------------
LLMs have NO persistent memory between API calls. The "memory" we experience
in chatbots is achieved by:
1. Storing conversation history (plain text)
2. Feeding the ENTIRE history to the model each turn
3. Model generates response based on full context

This is why:
- Context length limits matter (can't feed infinite history)
- Long conversations get expensive (more tokens = more cost)
- Context management (sliding window, summarization) becomes necessary

Try it yourself:
1. Set ENABLE_HISTORY = True, have a conversation, ask about earlier topics
2. Set ENABLE_HISTORY = False, repeat - notice the model forgets everything!
"""
if __name__ == "__main__":
    pass
