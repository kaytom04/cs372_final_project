# Duke Bites
### CS 372 Final Project — Spring 2026

## What it Does
Duke Bites is an AI-powered dining recommendation chatbot for Duke University students. Students describe their mood or craving in natural language and the chatbot recommends specific menu items from Duke dining halls, including location and hours. It uses sentence embeddings to semantically retrieve relevant menu items from a custom-built dataset of 394 Duke dining items, then passes them as context to Llama 3.3 70B via Groq to generate friendly, personalized recommendations.

## Quick Start
```bash
git clone https://github.com/kaytom04/cs372_final_project.git
cd cs372_final_project
pip install -r requirements.txt
python app.py
```
Open `http://localhost:7860` in your browser.

## Project Structure
```
cs372_final_project/
├── data/
│   ├── menu_items_retagged.csv
│   ├── WU_locations_Updated.csv
│   └── embeddings.pkl
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── chatbot_demo.ipynb
│   └── evaluate.ipynb
├── scripts/
│   └── retag_menu.py
├── app.py
├── requirements.txt
├── SETUP.md
└── ATTRIBUTION.md
```

## Video Links
- Demo video: [add link]
- Technical walkthrough: [add link]

## Evaluation
[Fill in after running evaluate.ipynb — paste your metrics table here]

- Mean retrieval score (normal queries): TBD
- Mean tag overlap: TBD
- Best performing prompt variant: TBD