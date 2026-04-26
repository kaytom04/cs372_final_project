# src/config.py
# Central config file
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')

MENU_CSV       = os.path.join(DATA_DIR, 'raw', 'menu_items_retagged.csv')
LOCATIONS_CSV  = os.path.join(DATA_DIR, 'raw', 'WU_locations_Updated.csv')
EMBEDDINGS_PKL = os.path.join(DATA_DIR, 'embeddings.pkl')

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GROQ_MODEL      = 'llama-3.1-8b-instant'
GROQ_API_KEY    = os.environ.get('GROQ_API_KEY')
