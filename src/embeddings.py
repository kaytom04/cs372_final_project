# src/embeddings.py
# Turn text data in to embeddings

# Most content in this file generated with AI, using Claude Sonnet 4.6

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Generates embedding vectors from string input using pretrained embedding model
def generate_embeddings(texts: list, model_name: str, batch_size: int = 32) -> np.ndarray:
    """Encode a list of text strings into embedding vectors."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)

# Saves generated embeddings to a file (both item and chunk embeddings)
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

# Load embeddings back into memory
def load_embeddings(path: str) -> dict:
    """Load embeddings and metadata from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)
