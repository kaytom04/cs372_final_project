import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Loaded {MODEL_NAME} on {device}")
    return model, tokenizer, device


def compute_embeddings(texts, model, tokenizer, device):
    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            outputs = model(**inputs)
            # mean pooling across token dimension
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embedding)
    return np.vstack(all_embeddings)


def retrieve_top_k(query, documents, embeddings, model, tokenizer, device, top_k=3):
    query_embedding = compute_embeddings([query], model, tokenizer, device)[0]

    # cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_docs = [documents[i] for i in top_indices]
    top_scores = similarities[top_indices]

    return top_docs, top_scores


def save_embeddings(embeddings, documents, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'documents': documents}, f)
    print(f"Saved {len(documents)} embeddings to {path}")


def load_embeddings(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['documents'])} embeddings from {path}")
    return data['embeddings'], data['documents']


def build_vector_store(processed_data_path, embeddings_path):
    df = pd.read_csv(processed_data_path)
    documents = df['document'].tolist()

    model, tokenizer, device = load_model()

    print(f"Computing embeddings for {len(documents)} documents...")
    embeddings = compute_embeddings(documents, model, tokenizer, device)
    print(f"Embeddings shape: {embeddings.shape}")

    save_embeddings(embeddings, documents, embeddings_path)
    return embeddings, documents, model, tokenizer, device


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, '..', 'data', 'menu_processed.csv')
    embeddings_path = os.path.join(base, '..', 'data', 'embeddings.pkl')

    embeddings, documents, model, tokenizer, device = build_vector_store(
        data_path, embeddings_path
    )

    # test retrieval
    for query in ["high protein breakfast", "vegetarian lunch", "late night food"]:
        results, scores = retrieve_top_k(
            query, documents, embeddings, model, tokenizer, device
        )
        print(f"\nQuery: '{query}'")
        for doc, score in zip(results, scores):
            print(f"  ({score:.3f}) {doc[:100]}...")