# Duke Bites
### CS 372 Final Project — Spring 2026
A Duke Inspired chatbot for personalized recommendations of what to eat on Duke's West Campus, specifically locations in WU and BC.


## What it Does
Duke Bites is an AI-powered dining recommendation chatbot for Duke University students. The goal of this project was to make a fun tool for students to use in order to decide what they want to eat from locations on campus. This idea came as a result of having to walk around WU numerous times (it's okay for getting those steps in, but ..) while waiting for friends to decide what to eat. Students also complain frequently about not knowing to eat despite the numerous options available on campus. The chatbot works where students ask for food recommendations based on their current mood or craving, and the chatbot recommends specific menu items from Duke dining halls, in addition to information such as location or hours. At a high level, the project uses sentence embeddings to semantically retrieve relevant menu items (RAG) from a custom-built dataset of 394 Duke dining items. It then passes them as context to Llama 3.3 70B via Groq to generate personalized recommendations.

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
├── src/
│   ├── Demo Video
│   └── Technical Walkthrough.py
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
- Example of the UI for Duke bites in addition to an example prompt and response
![alt text](/notebooks/data/image.png)

- Mean retrieval score (normal queries): TBD
- Mean tag overlap: TBD
- Best performing prompt variant: TBD