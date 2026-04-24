import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time

# 🔑 Set your API key
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 📂 Load your dataset
df = pd.read_csv("/Users/kaylatom/Desktop/DUKE STUFF/CS 372/cs372_final_project/data/menu_items_augmented_commas.csv")

# 🏷️ Define your tag set
TAG_LIST = [
    "vegetarian", "vegan", "dairy", "gluten",
    "high_protein", "healthy", "fried", "comfort_food",
    "breakfast", "lunch", "dinner", "dessert", "snack",
    "spicy", "quick", "drink", "caffeinated"
]

TAG_STRING = ", ".join(TAG_LIST)

# 🧾 Prompt template
def build_prompt(name, description):
    return f"""
Assign tags from this list ONLY:
[{TAG_STRING}]

Rules:
- Only use tags from the list
- Output format: tag1,tag2,tag3
- No spaces
- No explanations

Item:
Name: {name}
Description: {description}
"""

# 🤖 Call LLM
def get_tags(name, description, retries=3):
    prompt = build_prompt(name, description)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            )

            tags = response.choices[0].message.content.strip().lower()

            # Clean formatting
            tags = tags.replace(" ", "")
            return tags

        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)

    return ""  # fallback

# 🚀 Apply tagging
tqdm.pandas()

df["generated_tags"] = df.progress_apply(
    lambda row: get_tags(row["name"], row.get("description", "")),
    axis=1
)

# 💾 Save result
df.to_csv("menu_items_tagged.csv", index=False)

print("✅ Tagging complete. Saved to menu_items_tagged.csv")