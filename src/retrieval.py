import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings + metadata once at import time
with open("data/embeddings.pkl", "rb") as f:
    store = pickle.load(f)

embeddings = store["embeddings"]       # (394, 384)
metadata   = store["metadata"]         # DataFrame
model      = SentenceTransformer(store["model_name"])


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Given a natural language query, return the top_k most relevant menu items.
    Each result is a dict with item name, location, tags, description, and score.
    """
    query_vec = model.encode([query])
    scores    = cosine_similarity(query_vec, embeddings)[0]
    top_idx   = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = metadata.iloc[idx]
        results.append({
            "item":        row.get("name_item",     row.get("name", "")),
            "location":    row.get("name_location", ""),
            "description": row.get("description_item", row.get("description", "")),
            "tags":        row.get("generated_tags", ""),
            "meal_period": row.get("meal_period", ""),
            "hours":       row.get("hours", ""),
            "score":       round(float(scores[idx]), 3),
        })
    return results


def format_context(results: list[dict]) -> str:
    """
    Format retrieval results into a readable string to inject into the LLM prompt.
    """
    lines = []
    for r in results:
        lines.append(
            f"- {r['item']} @ {r['location']} ({r['meal_period']})\n"
            f"  Description: {r['description']}\n"
            f"  Tags: {r['tags']}\n"
            f"  Hours: {r['hours']}"
        )
    return "\n".join(lines)


# Quick test when run directly
if __name__ == "__main__":
    test_queries = [
        "I want something spicy",
        "I'm craving sushi",
        "something warm and comforting for breakfast",
        "healthy vegan lunch",
        "late night snack",
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = retrieve(q, top_k=3)
        for r in results:
            print(f"  {r['item']} @ {r['location']}  (score: {r['score']})")