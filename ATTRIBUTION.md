# Attribution

## AI Tools Used in Development

### Claude (Anthropic - Sonnet 4.6)
Claude was used throughout this project for code generation, architecture guidance, and debugging. 

The files that utilized AI generation are as follows:
- Initial scaffolding for `src/preprocessing.py`, `src/embeddings.py`, `src/retrieval.py`, `src/chatbot.py`, and `src/config.py`
- Initial versions of `notebooks/01_data_prep.ipynb` and `notebooks/evaluate.ipynb`
- Initial `app.py` Gradio UI structure
- `data/auto_tag.py` batch tagging logic

#### Modified and Reworked
- **`retag_menu.py`** The original Claude-generated version used batch processing which caused repeated parse errors and had to be changed to row-by-row tagging.

- **`app.py`** required lots of debugging due to Gradio version incompatibilities.

- **`notebooks/evaluate.ipynb`** modified as more evaluation ideas were generated

- **`src/retrieval.py`** added in more logic (ex. reranking) to improve retrieval.

- **`src/preprocessing.py`** later integrated chunking but was not heavily used due to poor performance

---

### Groq API / Llama Models
- **Llama 3.3 70B Versatile** — used in `scripts/retag_menu.py` to generate semantic tags for menu items using a manually designed tag vocabulary. Also used in `notebooks/evaluate.ipynb` for prompt engineering comparison and evaluation.
- **Llama 3.1 8B Instant** — used as the production LLM in `src/chatbot.py` and `app.py` for real-time dining recommendations.

---

### Data
- Menu item names, descriptions, and meal periods were manually collected from the Duke NetNutrition portal (netnutrition.cbord.com/nn-prod/Duke)
- Location data (names, hours, cuisine types, campus location) was manually collected from the Duke Dining website
- Semantic tags were generated using Llama 3.3 70B with a manually designed tag vocabulary — see `scripts/retag_menu.py`
- All data collection, cleaning, and curation was done by me

---

### Libraries
| Library | Use |
|---|---|
| `sentence-transformers` | Embedding model (all-MiniLM-L6-v2) |
| `groq` | Groq Python SDK for LLM inference |
| `gradio` | Web UI for app.py |
| `scikit-learn` | Cosine similarity for retrieval |
| `pandas` | Data loading and preprocessing |
| `numpy` | Embedding matrix operations |
| `matplotlib` | Evaluation visualizations |
| `python-dotenv` | API key management |