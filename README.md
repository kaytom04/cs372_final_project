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
├── videos/
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
    - ![alt text](/notebooks/data/promptquery.png)
    - Average scores were calculated for each of the 3 different prompt variations:
    - ![alt text](/notebooks/data/promptscores.png)
    - When comparing the 3 different prompt variations, the Friendly prompt performed the best as it produced relevant information in a clear and helpful way. It provided specific recommendations along with explanations that could help an individual with determining what to eat. The concise prompt was good at providing relevant and helpful information, but I believe that some of its responses could be shortened even more to make them more concise. The nutritionist prompt was very clear in all its responses but for only 1 of the tests provided good information in terms of why its meal recommendations are nutritional/beneficial for a person. The other responses lacked good nutritional information which could also be a limitation due to the data that the model has access to. Overall, the friendly prompt had the best balance.
- Retrieval Quality Evaluation:
    - The model was evaluated based on 3 metrics: Average cosine similarity score, tag overlap (recall), and mean precision.
    - ![alt text](/notebooks/data/retrievalquality.png) 
    - As seen in the results, the model performed well for retreival quality, scoring highest when querying for items that are spicy and comfort foods. One notable failure was the tag overlap for indian food, indicating that the model is limited by the tag generation for food items.
- Embedding Model Comparison
    - Two sentence embedding models were evaluated on the same queries, specifically the all-MiniLM-L6-v2 and the all-mpnet-basev2
    - ![alt text](/notebooks/data/model_comparison.png)
    - MiniLM outperformed mpnet on this dataset with a mean score of 0.610 vs 0.561. It was chosen as the main embedding model due to its higher retrieval scores despite being a smaller model. In addition it runs ~ 3x faster on CPU which was important for response latency.
-  Multi-Turn Conversation example
    - I included this example to show how multi-turn coversation was performing within the model. Although the model was able to remember key facts like the person being vegetarian it struggled in some instances. For example when prompted with what time does that place closed, it struggled to provide the correct time/location. (refer to notebooks/evaluate.ipynb file)
- Item vs Chunk Retrieval
    - I included this example to demonstrate the chunking vs item performance as part of my custom rag system. Although the final system only implements the embedding model selection with comparison and reranking, I had included chunking to see if it would improve my system. Although the model performed well in the first comparison of options available at skillet, it struggled with recommending available food at tandoor by only providing desserts, rather than giving solid food options. (refer to notebooks/evaluate.ipynb file)
- Edge Case Analysis
    - ![alt text](/notebooks/data/retrieval_scores.png)
    - Overall, the system seems to handle dietary restriction queries well as it achieved a 0.6 for the gluten and dairy allergy case, and correctly produced a viable food option. 
    - The model struggled when queried with an out of hours request as it struggled to realize that the establishment was closed
    - The two instances that scored the lowest was the query for east campus and the cheapest thing which makes sense as the model wasn't given information about east campus or prices
    - The gibberish query was decent as the model managed to suggest some food despite being prompted with nonsense.

Overall, the system performs reliably for common food requests but has clear limitations around real-time data (hours, pricing) and location that would require additional data collection to address.

## Design Decision
- A major design decision that I made during this project was determining which LLM to use (Llama 3.1 8 B vs Llama 3.3 70B). This was primarily determing the tradeoff between model capability vs token limit and latency. 
- During development, I used two different Llama models via Groq
- Llama 3.1 8B Instant was used for the production chatbot (src/chatbot.py and app.py). This is because it is faster to use and uses fewer tokens pre request. This was good for instances in which I was still working through devlopment and testing the model. In addition, it had sufficient enough parameters given the context it was being used in (short conversations for food recs).
- Llama 3.3 70B was used first in my auto_tag.py script which was used to add more labels to my dataset of food items so that my model would have more semantic information to work from, and performed better at autotagging than the other model. It was also used for evaluation in notebooks/evaluate.ipynb as for those instance I wanted the performance to not be limited by the quality of the model and wanted to use these tokens for more specific tasks like these. In an ideal scenario with no rate limits, 70B would be preferred for all tasks. The decision to split by task demonstrates a constraint of working within a free API tier.