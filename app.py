import os, pickle, numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import gradio as gr

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])
MODEL  = 'llama-3.3-70b-versatile'

# Load embeddings
with open('data/embeddings.pkl', 'rb') as f:
    store = pickle.load(f)
embeddings = store['embeddings']
metadata   = store['metadata']
emb_model  = SentenceTransformer(store['model_name'])

SYSTEM = """You are Duke Bites 🍴, a friendly dining assistant for Duke University students.
Help students decide what to eat based on their mood or craving.
Always recommend 2-3 specific items with their dining location and hours.
Be warm and conversational. Keep responses to 3-5 sentences.

Retrieved menu items:
{context}"""

def retrieve(query, top_k=5):
    query_vec = emb_model.encode([query])
    scores    = cosine_similarity(query_vec, embeddings)[0]
    top_idx   = np.argsort(scores)[::-1][:top_k]
    results   = []
    for idx in top_idx:
        row = metadata.iloc[idx]
        results.append({
            'item':        row.get('name_item',        row.get('name', '')),
            'location':    row.get('name_location',    ''),
            'description': row.get('description_item', row.get('description', '')),
            'tags':        row.get('generated_tags',   ''),
            'meal_period': row.get('meal_period',      ''),
            'hours':       row.get('hours',            ''),
        })
    return results

def format_context(results):
    lines = []
    for r in results:
        lines.append(
            f"- {r['item']} @ {r['location']} ({r['meal_period']})\n"
            f"  Description: {r['description']}\n"
            f"  Tags: {r['tags']}\n"
            f"  Hours: {r['hours']}"
        )
    return '\n'.join(lines)

def chat(user_message, history):
    # history is a list of {"role": ..., "content": ...} dicts (Gradio 4.x format)
    context = format_context(retrieve(user_message))
    system  = SYSTEM.format(context=context)

    messages = [{'role': 'system', 'content': system}]
    for msg in history:
        messages.append({'role': msg['role'], 'content': msg['content']})
    messages.append({'role': 'user', 'content': user_message})

    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7, max_tokens=400
    )
    return resp.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(title='Duke Bites') as demo:
    gr.Markdown("# 🍴 Duke Bites\n### Your Duke dining recommendation assistant")
    gr.Markdown("Tell me what you're craving and I'll find something at a Duke dining hall!")

    chatbot = gr.Chatbot(height=450, label='Duke Bites')
    msg     = gr.Textbox(placeholder="I'm hungry and want something spicy...", label="You", scale=4)

    with gr.Row():
        send  = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    examples = gr.Examples(
        examples=[
            "I want something warm and comforting",
            "I'm vegetarian, what are my options?",
            "I need a late night snack",
            "I want sushi",
            "Something healthy for lunch",
        ],
        inputs=msg
    )

    def respond(user_message, history):
        reply = chat(user_message, history)
        history = history + [{'role': 'user', 'content': user_message}, {'role': 'assistant', 'content': reply}]
        return "", history

    msg.submit(respond,  [msg, chatbot], [msg, chatbot])
    send.click(respond,  [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch(theme=gr.themes.Soft())


