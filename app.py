# app.py
# Front end interface for chatbot (UI)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import chat
import gradio as gr

# Used when user sends a message
def respond(user_message, history):
    reply, _ = chat(user_message, history)
    return reply

demo = gr.ChatInterface(
    fn=respond,
    title="🍴 Duke Bites",
    description="Tell me what you're craving and I'll find something at a Duke dining hall!",
    examples=[
        "I want something warm and comforting",
        "I'm vegetarian, what are my options?",
        "I need a late night snack",
        "I want sushi",
        "Something healthy for lunch",
    ],
)

demo.launch(theme=gr.themes.Soft())