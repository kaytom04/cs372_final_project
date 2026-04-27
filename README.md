# Duke Bites
### CS 372 Final Project — Spring 2026
A Duke Insipred chatbot for personalized recommendations of what to eat on Duke's West Campus, specifically locations in WU and BC.
### Kayla Tom

## What it Does
Duke Bites is an AI-powered dining recommendation chatbot for Duke University students. Students asked for food recommendations based on  their current mood or craving, and the chatbot recommends specific menu items from Duke dining halls, in addition to information such as location or hours. At a high level, the project uses sentence embeddings to semantically retrieve relevant menu items (RAG) from a custom-built dataset of 394 Duke dining items. It then passes them as context to Llama 3.3 70B via Groq to generate personalized recommendations.

## Quick Start
```bash
git clone https://github.com/kaytom04/cs372_final_project.git
cd cs372_final_project
conda create -n duke-dining python=3.10
conda activate duke-dining
pip install -r requirements.txt
Create a .env file and set: GROQ_API_KEY=your_api_key_here
python app.py
```
Open `http://localhost:7860` in your browser.

## Project Structure
```
cs372_final_project/
├── data/
│   ├── raw/
│   │   ├── menu_items_retagged.csv
│   │   ├── menu_items_augmented_commas.csv
│   │   ├── WU_locations_Updated.csv
│   ├── auto_tag.py
│   └── embeddings.pkl
├── notebooks/
│   ├── data/
│   │   ├── model_comparison.png
│   │   ├── retrieval_scores.png
│   ├── data_prep.ipynb
│   └── evaluate.ipynb
├── scripts/
│   └── run_chatbot.py
├── src/
│   ├── chatbot.py
│   ├── config.py
│   ├── embeddings.py
│   ├── preprocessing.py
│   └── retrieval.py
├── app.py
├── requirements.txt
├── SETUP.md
├── README.md
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