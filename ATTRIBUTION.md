# Attribution

## AI Tools Used

### Claude (Anthropic)
Used throughout the project for code generation, debugging, and architecture guidance. Specifically:
- Generated initial versions of `app.py`, `retag_menu.py`, and all three notebooks
- Helped debug PyTorch/dependency issues and Gradio message format errors
- Suggested project structure and rubric strategy
- All generated code was reviewed, modified, and integrated by me — several functions were substantially reworked (e.g. the retag batching logic was rewritten after parse errors, the evaluate notebook was rewritten to be self-contained after imports failed)

### Groq / Llama 3.3 70B
- Used as the LLM backbone for the chatbot (inference via API)
- Used via `retag_menu.py` to auto-generate tags for 394 menu items using a manually designed tag vocabulary

### Llama 3.1 8B Instant (Groq)
- Used in early versions of `retag_menu.py` before switching to 3.3 70B for better structured output reliability

## Data
- Menu item names and descriptions manually collected from the Duke NetNutrition portal (netnutrition.cbord.com/nn-prod/Duke)
- Location data manually collected from Duke Dining website
- Tags generated using Llama 3.3 70B with a manually designed vocabulary — see `scripts/retag_menu.py`

## Libraries
- `sentence-transformers` — embedding model (all-MiniLM-L6-v2)
- `groq` — Groq Python SDK
- `gradio` — web UI
- `scikit-learn` — cosine similarity
- `pandas`, `numpy` — data processing
