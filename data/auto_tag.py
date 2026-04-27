# data/auto_tag.py
# Script to automate tagging of each food item based on specific set of tags. Uses llama-3.3-70b-versatile

# Most content in this file generated with AI, using Claude Sonnet 4.6

import os, time, pandas as pd
from groq import Groq
from tqdm import tqdm

# GROQ API KEY
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Read CSV
df = pd.read_csv("menu_items_augmented_commas.csv").drop(columns=["tags"])
df["description"] = df["description"].fillna(df["name"])

# Custom set of tags to label each item in the list
TAG_STRING = "vegetarian, vegan, dairy, gluten_free, high_protein, healthy, fried, comfort, breakfast, lunch, dinner, dessert, snack, spicy, quick, drink, caffeinated, chicken, beef, pork, seafood, fish, egg, tofu, bowl, sandwich, salad, soup, pizza, pasta, sushi, burger, taco, baked, sweet, savory, light, rich, fresh, hot, cold, smoothie, juice, late_night, filling"

# Assign generated tags to each item and append to CSV
def get_tags(name, description, retries=3):
    prompt = f"""Assign tags from this list ONLY: [{TAG_STRING}]
Rules: only use tags from the list, output format: tag1,tag2,tag3, no spaces, no explanations
Name: {name}
Description: {description}"""

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return resp.choices[0].message.content.strip().lower().replace(" ", "")
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
    return ""

tqdm.pandas()
df["generated_tags"] = df.progress_apply(
    lambda row: get_tags(row["name"], row.get("description", "")), axis=1
)

# Output to new CSV file
df.to_csv("menu_items_retagged.csv", index=False)
