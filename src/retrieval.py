import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings import load_embeddings, load_model, retrieve_top_k

# module level state — loaded once when chatbot imports this
embeddings = None
documents = None
model = None
tokenizer = None
device = None


def initialize(embeddings_path):
    global embeddings, documents, model, tokenizer, device
    embeddings, documents = load_embeddings(embeddings_path)
    model, tokenizer, device = load_model()
    print("Retrieval system ready")


def retrieve(query, top_k=3):
    if embeddings is None:
        raise RuntimeError("Call initialize() before retrieve()")
    results, scores = retrieve_top_k(
        query, documents, embeddings, model, tokenizer, device, top_k=top_k
    )
    return results, scores