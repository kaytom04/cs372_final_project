import os
from groq import Groq
from retrieval import initialize, retrieve

SYSTEM_PROMPT = """You are DukeEats, a friendly dining assistant for Duke University students.
You help students find meals at Duke dining halls based on their cravings, dietary needs, meal time, and location on campus.

Only recommend dishes that appear in the CONTEXT provided to you. Never invent menu items.
If no good match exists in the context, say so honestly.

For each recommendation always include:
- Dish name
- Dining hall name and which campus it's on
- Why it matches what the student asked for

Be warm, specific, and concise."""


def get_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        # try colab secrets
        try:
            from google.colab import userdata
            api_key = userdata.get('GROQ_API_KEY')
        except:
            pass
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Set it in .env or Colab Secrets.")
    return Groq(api_key=api_key)


def chat(user_message, history, client):
    # retrieve relevant menu items
    results, scores = retrieve(user_message, top_k=3)
    context = '\n'.join([f'- {doc}' for doc in results])

    # augment user message with retrieved context
    augmented = f"""Student question: {user_message}

CONTEXT (relevant menu items):
{context}

Please recommend from the context above."""

    history.append({"role": "user", "content": augmented})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=1024,
        temperature=0.7
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply, history


def run_chatbot(embeddings_path):
    initialize(embeddings_path)
    client = get_client()
    history = []

    print("DukeEats: Hi! I'm DukeEats. What are you hungry for today?")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("DukeEats: Enjoy your meal!")
            break
        if not user_input:
            continue

        reply, history = chat(user_input, history, client)
        print(f"DukeEats: {reply}\n")


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # load .env for local development
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

    base = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(base, '..', 'data', 'embeddings.pkl')

    run_chatbot(embeddings_path)