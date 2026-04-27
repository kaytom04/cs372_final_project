# src/chatbot.py

# Most content in this file generated with AI, using Claude Sonnet 4.6

# Takes user query then retrieves relevant information. Sends prompt and info to LLM. 
from groq import Groq
from retrieval import retrieve, format_context
from config import GROQ_API_KEY, GROQ_MODEL

# GROQ API Key
client = Groq(api_key=GROQ_API_KEY)

# Three prompt variants. Used later in evaluate.ipynb for prompt engineering comparison
# Prompt 1: Friendly/conversational. Explains choices
PROMPT_V1_FRIENDLY = """You are Duke Bites, a friendly dining assistant for Duke University students.
Help students decide what to eat based on their mood or craving.
Always recommend 2-3 specific items, include the dining location and hours.
Be warm and conversational. Keep responses to 3-5 sentences.
Retrieved menu items:
{context}"""

# Prompt 2: Concise. Provides short answers
PROMPT_V2_CONCISE = """You are Duke Bites, a Duke University dining recommendation assistant.
Recommend exactly 2 items, each on its own line.
Format: [Item] at [Location] — [one sentence reason]. End with hours. No small talk.
Retrieved menu items:
{context}"""

# Prompt 3: Nutritionist. Acts as a nutritional dining assistant
PROMPT_V3_NUTRITIONIST = """You are Duke Bites, a knowledgeable Duke University dining assistant.
Help students make smart food choices based on mood, cravings, and nutritional needs.
Recommend 2-3 items, explain why they match, mention dietary info when relevant.
Note location and hours. Be informative but not preachy.
Retrieved menu items:
{context}"""

# Active prompt used by the chatbot
ACTIVE_PROMPT = PROMPT_V1_FRIENDLY

# Fixes bug with Gradio history format. Clears metadata before sending to Groq (was causing issues before)
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

# Main Logic
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

    context_query = user_message

    if history:
        prev_user_msgs = [
            str(msg['content']) for msg in history 
            if isinstance(msg, dict) and msg.get('role') == 'user'
        ]
        context_query = ' '.join(prev_user_msgs[-2:] + [user_message])

    results = retrieve(context_query, top_k=5)
    # results = retrieve(user_message, top_k=5) # retrieve 5 most relevant items
    context = format_context(results) # Formats results in readable context
    system  = (prompt or ACTIVE_PROMPT).format(context=context) # Builds system prompt and inserts retrieved items

    api_history = clean_history_for_api(history) # clean history for GROQ

    # message that is sent to the LLM: System prompt, chat history, new message (multi-turn conversation)
    messages = (
    [{"role": "system", "content": system}]
    + api_history
    + [{"role": "user", "content": user_message}]
    )

    # call the LLM
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )

    # extract the reply
    reply = resp.choices[0].message.content

    # update the chat history
    updated_history = history + [
        {'role': 'user',      'content': user_message},
        {'role': 'assistant', 'content': reply},
    ]
    return reply, updated_history
