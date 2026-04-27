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

In notebooks/evaluate.ipynb, multiple tests were run to evaluate the system:

- Prompt Engineering Comparison
    - In order to understand the impact of prompt engineering, I tried 3 different prompts (friendly, concise, nutritionist). 
    - For example: 
    ============================================================
    Query: 'I need a late night snack'
    ============================================================

    [V1 Friendly]
    Late night cravings are the best. I've got just the thing for you. Head over to Gothic Grill, open until Midnight (or 1 am on Fridays and Saturdays), and grab some Mozzarella Sticks or Crispy Cauliflower - both are perfect for a satisfying late night snack. If you're in the mood for something a bit different, you could also swing by The Devil's Krafthouse, also open until Midnight, and try their Pretzels with beer cheese and mustard.

    [V2 Concise]
    Mozzarella Sticks at Gothic Grill — A satisfying late-night snack to curb your cravings.
    Crispy Cauliflower at Gothic Grill — A crispy and savory option to fulfill your late-night hunger, available until Midnight (sun - thur) and 1 am (fri - sat).

    [V3 Nutritionist]
    Late night cravings can be tough to resist. I've got just the things for you. Here are a few options that might hit the spot:

    1. **Mozzarella Sticks** at Gothic Grill (open till Midnight on Sun - Thur, and 1 am on Fri - Sat) - A classic comfort food that's easy to devour in the wee hours. They're fried, cheesy, and sure to satisfy your late-night cravings.
    2. **Crispy Cauliflower** at Gothic Grill (same hours as above) - Another tasty option that's also vegetarian-friendly. The crispy exterior and ranch dipping sauce make for a satisfying snack that's not too heavy.
    3. **Pretzels with beer cheese and mustard** at The Devil's Krafthouse (open till Midnight, 7 days a week) - A savory, comforting snack that's perfect for a late night pick-me-up. The combination of soft pretzels, creamy beer cheese, and tangy mustard is a winner.

    All of these options are readily available during late night hours, so you can't go wrong with any of them. Enjoy your snack!
    - Average scores were calculated for each of the 3 different prompt variations:
    - ![alt text](/notebooks/promptscores.png)
    - When comparing the 3 different prompt variations, the Friendly prompt performed the best as it produced relevant information in a clear and helpful way. It provided specific recommendations along with explanations that could help an individual with determining what to eat. The concise prompt was good at providing relevant and helpful information, but I believe that some of its responses could be shortened even more to make them more concise. The nutritionist prompt was very clear in all its responses but for only 1 of the tests provided good information in terms of why its meal recommendations are nutritional/beneficial for a person. The other responses lacked good nutritional information which could also be a limitation due to the data that the model has access to. Overall, the friendly prompt had the best balance.
- Retrieval Quality Evaluation:
    - The model was evaluated based on 3 metrics: Average cosine similarity score, tag overlap (recall), and mean precision.
    - ![alt text](/notebooks/data/retrievalquality.png) 
    - As seen in the results, the model performed well for retreival quality, scoring highest when querying for items that are spicy and comfort foods. One notable failure was the tag overlap for indian food, indicating that the model is limited by the tag generation for food items.
