# app.py
import gradio as gr
from src.chatbot import chat

def respond(user_message, history):
    reply, updated_history = chat(user_message, history)
    return reply, updated_history

with gr.Blocks(title="Duke Bites 🍴") as demo:
    gr.Markdown("# 🍴 Duke Bites\nTell me what you're craving and I'll find something at a Duke dining hall!")

    chatbot  = gr.Chatbot(label="Duke Bites", height=400)
    msg      = gr.Textbox(placeholder="I'm hungry and want something spicy...", label="You")
    history  = gr.State([])

    def handle(user_msg, chat_history, raw_history):
        reply, updated = chat(user_msg, raw_history)
        chat_history.append((user_msg, reply))
        return "", chat_history, updated

    msg.submit(handle, [msg, chatbot, history], [msg, chatbot, history])
    gr.Button("Send").click(handle, [msg, chatbot, history], [msg, chatbot, history])

demo.launch()
