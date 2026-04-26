# src/chatbot.py
import os
from groq import Groq
from retrieval import retrieve, format_context
from prompts import ACTIVE_PROMPT
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

def chat(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Send a message and get a response.

    Args:
        user_message: what the student typed
        history:      list of previous {"role": ..., "content": ...} messages

    Returns:
        (assistant_reply, updated_history)
    """
    # 1. Retrieve relevant menu items based on the user's message
    results = retrieve(user_message, top_k=5)
    context = format_context(results)

    # 2. Build the system prompt with retrieved context injected
    system_prompt = ACTIVE_PROMPT.format(context=context)

    # 3. Assemble full message list: system + history + new user message
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    # 4. Call Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    reply = response.choices[0].message.content

    # 5. Append both turns to history and return
    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return reply, history


# Quick test when run directly
if __name__ == "__main__":
    history = []
    test_turns = [
        "I'm really hungry and want something comforting",
        "Actually I'm vegetarian, any options?",
        "What time does that place close?",
    ]
    for msg in test_turns:
        print(f"\nUser: {msg}")
        reply, history = chat(msg, history)
        print(f"Duke Bites: {reply}")
