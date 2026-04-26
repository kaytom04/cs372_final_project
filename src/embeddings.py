# src/embeddings.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_embeddings(texts: list, model_name: str, batch_size: int = 32) -> np.ndarray:
    """Encode a list of text strings into embedding vectors."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)


def save_embeddings(path: str, item_embeddings, item_metadata,
                    chunk_embeddings, chunk_metadata, model_name: str):
    """Save all embeddings and metadata to a pickle file."""
    payload = {
        'embeddings':       item_embeddings,
        'metadata':         item_metadata,
        'chunk_embeddings': chunk_embeddings,
        'chunk_metadata':   chunk_metadata,
        'model_name':       model_name,
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved embeddings to {path}')


def load_embeddings(path: str) -> dict:
    """Load embeddings and metadata from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)
