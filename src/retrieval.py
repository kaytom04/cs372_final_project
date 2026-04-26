# src/retrieval.py
# returns relevant results for the chatbot
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import load_embeddings
from config import EMBEDDINGS_PKL, EMBEDDING_MODEL

# Load once at import time (loads embeddings, metadata, chunk embeddings, chunk metadata, embeddin gmodel)
_store            = load_embeddings(EMBEDDINGS_PKL)
embeddings        = _store['embeddings']
metadata          = _store['metadata']
chunk_embeddings  = _store['chunk_embeddings']
chunk_metadata    = _store['chunk_metadata']
_model            = SentenceTransformer(_store.get('model_name', EMBEDDING_MODEL))

# returns top 5 relevant menu items
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Item-level semantic retrieval with reranking."""
    qvec    = _model.encode([query]) # encode query
    scores  = cosine_similarity(qvec, embeddings)[0] # compute cosine similarity
    top_idx = np.argsort(scores)[::-1][:top_k * 2]  # fetch 2x, then rerank (grab 10 then rerank to 5)

    results = []
    for idx in top_idx:
        row = metadata.iloc[idx]
        results.append({
            'item':        row.get('name_item',        row.get('name', '')),
            'location':    row.get('name_location',    ''),
            'description': row.get('description_clean', row.get('description', '')),
            'tags':        row.get('generated_tags',   ''),
            'meal_period': row.get('meal_period',      ''),
            'hours':       row.get('hours',            ''),
            'score':       round(float(scores[idx]),   3),
        })
    return rerank(results, query)[:top_k]

# post processing/reranking
def rerank(results: list[dict], query: str) -> list[dict]:
    """Boost results whose tags match keywords in the query."""
    keywords = query.lower().split() # split query into keywords
    boosted  = []
    for r in results:
        boost = sum(1 for k in keywords if k in r['tags'].lower()) # compute a boost for each result
        boosted.append({**r, 'score': r['score'] + boost * 0.1}) # boost score for each keyword match
    return sorted(boosted, key=lambda x: x['score'], reverse=True) # sort by new score

# returns top grouped location and meal chunks
def retrieve_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Chunk-level retrieval — returns whole station context per location+meal."""
    qvec    = _model.encode([query])
    scores  = cosine_similarity(qvec, chunk_embeddings)[0] # compares against chunk vectors instead of item vectors
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = chunk_metadata.iloc[idx]
        results.append({
            'location':    row['location'],
            'meal_period': row['meal_period'],
            'hours':       row['hours'],
            'items':       row['items'],
            'tags':        row['tags'],
            'text_blob':   row['text_blob'],
            'score':       round(float(scores[idx]), 3),
        })
    return results

# turn retrieval into text for LLM
def format_context(results: list[dict]) -> str:
    """Format retrieval results into a string for the LLM prompt."""
    lines = []
    for r in results:
        if 'item' in r:
            lines.append(
                f"- {r['item']} @ {r['location']} ({r['meal_period']})\n"
                f"  Description: {r['description']}\n"
                f"  Tags: {r['tags']}  |  Hours: {r['hours']}"
            )
        else:
            lines.append(r['text_blob']) # if chunk, use full chink block
    return '\n'.join(lines)
