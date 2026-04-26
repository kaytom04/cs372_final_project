import os, time, pandas as pd
from groq import Groq
from tqdm import tqdm

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = pd.read_csv("/Users/kaylatom/Desktop/DUKE STUFF/CS 372/cs372_final_project/data/extra/menu_items_augmented_commas.csv").drop(columns=["tags"])
df["description"] = df["description"].fillna(df["name"])

TAG_STRING = "vegetarian, vegan, dairy, gluten_free, high_protein, healthy, fried, comfort, breakfast, lunch, dinner, dessert, snack, spicy, quick, drink, caffeinated, chicken, beef, pork, seafood, fish, egg, tofu, bowl, sandwich, salad, soup, pizza, pasta, sushi, burger, taco, baked, sweet, savory, light, rich, fresh, hot, cold, smoothie, juice, alcoholic, late_night, filling"

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

df.to_csv("menu_items_retagged.csv", index=False)
print("Done. Saved to menu_items_retagged.csv")