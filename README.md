# Duke Bites
### CS 372 Final Project — Spring 2026
A Duke Inspired chatbot for personalized recommendations of what to eat on Duke's West Campus, specifically locations in WU and BC.


## What it Does
Duke Bites is an AI-powered dining recommendation chatbot for Duke University students. The goal of this project was to make a fun tool for students to use in order to decide what they want to eat from locations on campus. This idea came as a result of having to walk around WU numerous times (it's okay for getting those steps in) while waiting for friends to decide what to eat. Students also complain frequently about not knowing to eat despite the numerous options available on campus. The chatbot works where students ask for food recommendations based on their current mood or craving, and the chatbot recommends specific menu items from Duke dining halls, in addition to information such as location or hours. At a high level, the project uses sentence embeddings to semantically retrieve relevant menu items (RAG) from a custom-built dataset of 394 Duke dining items. It then passes them as context to Llama 3.3 70B via Groq to generate personalized recommendations.

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
│   └── Technical Walkthrough
├── app.py
├── requirements.txt
├── SETUP.md
├── README.md
└── ATTRIBUTION.md
```

## Video Links
- Demo Video: [[here](https://duke.zoom.us/rec/share/Yy2yVKnfPs9wx9Jyux2_fTnGHXWf5F2LV1RlVjZrVSed1yzpzx2i0b_adatam0EZ.UWW95h5cZjo4xvgt?startTime=1777261120000)]
- Technical video: [[here](https://duke.zoom.us/rec/share/3Rf-mHArygewlHaC2zQMsAsyKAbbmzw3Qj-aBc2oS7bR4IPCXAKAzgmTCxyOpsSy.4JMD-SCGPxL00LPz?startTime=1777260078000)]


## Evaluation
- Example of the UI for Duke bites in addition to an example prompt and response
![alt text](/notebooks/data/image.png)

In notebooks/evaluate.ipynb, multiple tests were run to evaluate the system:

- Prompt Engineering Comparison
    - In order to understand the impact of prompt engineering, I tried 3 different prompts (friendly, concise, nutritionist). 

**Query 1: "I want something spicy for dinner"**
*(Specific craving — answer exists in knowledge base)*

| Prompt | Response |
|---|---|
| V1 Friendly | [I've got just the thing for you. If you're in the mood for something spicy, I'd recommend checking out the Spicy Miso Ramen at Ginger + Soy, open from 11 am to 9 pm. Alternatively, you could head to Il Forno for their Spicy Il Forno, also available from 11 am to 9 pm. If you're looking for something a bit different, the Spicy Tuba Yubu at Gyotaku is another great option, and they're open from 11 am to 9 pm as well.] |
| V2 Concise | [Spicy Miso Ramen at Ginger + Soy — This dish is a great option for a spicy dinner with its combination of spicy miso soup, pork belly or fried tofu, and various toppings. Spicy Il Forno at Il Forno — This pasta dish is perfect for those who enjoy spicy food, featuring fennel sausage, spicy arrabbiata sauce, and roasted tomatoes, all served with rigatoni. 
11 am - 9 pm] |
| V3 Nutritionist | [Spicy food can be so satisfying. I've got three options that might hit the spot for you: 1. **Spicy Miso Ramen** at Ginger + Soy (11 am - 9 pm, mon - sun): This dish is a great combination of spicy and savory, with a rich miso soup base, pork belly or fried tofu, and plenty of veggies. The spicy kick will definitely warm you up. 2. **Spicy Il Forno** at Il Forno (11 am - 9 pm, mon - sun): If you're in the mood for pasta, this spicy arrabbiata sauce-based dish with fennel sausage and rigatoni will deliver. It's a hearty, flavorful option that's sure to satisfy your craving for something spicy. 3. **Spicy Tuba Yubu** at Gyotaku (11 am - 9 pm, mon - sun): For a seafood option, this dish features fried tofu, spicy tuna, crispy onion, and spicy mayo. It's a unique and flavorful combination that's sure to spice up your dinner. All of these options are available for dinner, so you can choose the one that sounds most appealing to you. Enjoy!] |

**Query 2: "I'm vegetarian and want a healthy lunch"**
*(Dietary restriction query — tests constraint handling)*

| Prompt | Response |
|---|---|
| V1 Friendly | [You're in luck because I've got some fantastic options for you. For a healthy vegetarian lunch, I recommend checking out Sprout and trying their Falafel Pita Pocket, Spicy Cauliflower Wrap, or building your own 3 Composed Salad with their delicious and fresh ingredients. Sprout is open from 11 am to 9 pm, so you can swing by and grab a bite that fits your mood and dietary needs.] |
| V2 Concise | [Falafel Pita Pocket at Sprout — This is a great option for a healthy and satisfying vegetarian lunch with a balance of flavors and textures. Spicy Cauliflower Wrap at Sprout — This wrap is a tasty and healthy choice, packed with spicy cauliflower and fresh vegetables, making it an ideal option for a vegetarian lunch. 8 am - 10:30 am (mon - fri); 11 am - 9 pm (mon - sun)] |
| V3 Nutritionist | [As a vegetarian looking for a healthy lunch, I'd recommend the following options at Sprout: 1. **Falafel Pita Pocket**: This is a great choice, packed with protein-rich falafel, fresh lettuce, tomato, and red onion, all wrapped in a pita. It's a satisfying and healthy option that's also vegetarian-friendly. 2. **3 Composed Salad**: This option allows you to pick three ingredients from a variety of healthy options like kale salad, sweet potato, chickpeas, and more. You can create a salad that suits your taste and dietary needs. 3. **Spicy Cauliflower Wrap**: If you're in the mood for something a bit spicy, this wrap is a great option. It's filled with roasted cauliflower, cucumber, pickled onion, and tahini, all wrapped in a whole wheat wrap. All of these options are available at Sprout, which is open from 11 am to 9 pm (Monday to Sunday). You can also consider adding a side like brown rice, roasted vegetables, or a kale salad to complement your meal. Enjoy your lunch!] |

**Query 3: "I need a late night snack"**
*(Time-sensitive query — tests hours awareness)*

| Prompt | Response |
|---|---|
| V1 Friendly | [Late night cravings are the best. I've got just the thing for you. Head over to Gothic Grill, open until Midnight (or 1 am on Fridays and Saturdays), and grab some Mozzarella Sticks or Crispy Cauliflower - both are perfect for a satisfying late night snack. If you're in the mood for something a bit different, you could also swing by The Devil's Krafthouse, also open until Midnight, and try their Pretzels with beer cheese and mustard.] |
| V2 Concise | [Mozzarella Sticks at Gothic Grill — A satisfying late-night snack to curb your cravings.Crispy Cauliflower at Gothic Grill — A crispy and savory option to fulfill your late-night hunger, available until Midnight (sun - thur) and 1 am (fri - sat).] |
| V3 Nutritionist | [Late night cravings can be tough to resist. I've got just the things for you. Here are a few options that might hit the spot: 1. **Mozzarella Sticks** at Gothic Grill (open till Midnight on Sun - Thur, and 1 am on Fri - Sat) - A classic comfort food that's easy to devour in the wee hours. They're fried, cheesy, and sure to satisfy your late-night cravings. 2. **Crispy Cauliflower** at Gothic Grill (same hours as above) - Another tasty option that's also vegetarian-friendly. The crispy exterior and ranch dipping sauce make for a satisfying snack that's not too heavy. 3. **Pretzels with beer cheese and mustard** at The Devil's Krafthouse (open till Midnight, 7 days a week) - A savory, comforting snack that's perfect for a late night pick-me-up. The combination of soft pretzels, creamy beer cheese, and tangy mustard is a winner.All of these options are readily available during late night hours, so you can't go wrong with any of them. Enjoy your snack!] |

 When comparing the 3 different prompt variations, the Friendly prompt performed the best as it produced relevant information in a clear and helpful way. It provided specific recommendations along with explanations that could help an individual with determining what to eat. The concise prompt was good at providing relevant and helpful information, but I believe that some of its responses could be shortened even more to make them more concise. The nutritionist prompt was very clear in all its responses but for only 1 of the tests provided good information in terms of why its meal recommendations are nutritional/beneficial for a person. The other responses lacked good nutritional information which could also be a limitation due to the data that the model has access to. Overall, the friendly prompt had the best balance.

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

## Individual Contribution
I worked alone