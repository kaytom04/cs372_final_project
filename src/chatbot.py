# src/chatbot.py
from groq import Groq
from retrieval import retrieve, format_context
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

# Three prompt variants — used in evaluate.ipynb for prompt engineering comparison
PROMPT_V1_FRIENDLY = """You are Duke Bites, a friendly dining assistant for Duke University students.
Help students decide what to eat based on their mood or craving.
Always recommend 2-3 specific items, include the dining location and hours.
Be warm and conversational. Keep responses to 3-5 sentences.
Retrieved menu items:
{context}"""

PROMPT_V2_CONCISE = """You are Duke Bites, a Duke University dining recommendation assistant.
Recommend exactly 2 items, each on its own line.
Format: [Item] at [Location] — [one sentence reason]. End with hours. No small talk.
Retrieved menu items:
{context}"""

PROMPT_V3_NUTRITIONIST = """You are Duke Bites, a knowledgeable Duke University dining assistant.
Help students make smart food choices based on mood, cravings, and nutritional needs.
Recommend 2-3 items, explain why they match, mention dietary info when relevant.
Note location and hours. Be informative but not preachy.
Retrieved menu items:
{context}"""

# Active prompt used by the chatbot — swap to test variants
ACTIVE_PROMPT = PROMPT_V1_FRIENDLY

def clean_history_for_api(history):
    cleaned = []

    for msg in history or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")

        if role in {"user", "assistant", "system"} and content is not None:
            cleaned.append({
                "role": role,
                "content": str(content)
            })

    return cleaned

def chat(user_message: str, history: list[dict],
         prompt: str = None) -> tuple[str, list[dict]]:
    """
    Send a message and get a response.

    Args:
        user_message: what the student typed
        history:      list of previous {"role": ..., "content": ...} messages
        prompt:       override the active prompt (used in evaluation)

    Returns:
        (reply, updated_history)
    """
    results = retrieve(user_message, top_k=5)
    context = format_context(results)
    system  = (prompt or ACTIVE_PROMPT).format(context=context)

    api_history = clean_history_for_api(history)

    # messages = (
    #     [{'role': 'system', 'content': system}]
    #     + history
    #     + [{'role': 'user', 'content': user_message}]
    # )
    messages = (
    [{"role": "system", "content": system}]
    + api_history
    + [{"role": "user", "content": user_message}]
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )
    reply = resp.choices[0].message.content

    updated_history = history + [
        {'role': 'user',      'content': user_message},
        {'role': 'assistant', 'content': reply},
    ]
    return reply, updated_history
