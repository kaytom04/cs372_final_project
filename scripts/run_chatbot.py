# scripts/run_chatbot.py
# Most content in this file generated with AI, using Claude Sonnet 4.6
# Terminal Interface for chatbot
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot import chat

def main():
    print("Duke Bites 🍴 — Type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue
        reply, history = chat(user_input, history)
        print(f"Duke Bites: {reply}\n")

if __name__ == '__main__':
    main()
