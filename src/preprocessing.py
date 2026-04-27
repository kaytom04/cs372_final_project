# src/preprocessing.py
# cleans menu/location data, merges datasets, create text representation used for item and chunk retrieval
import re
import pandas as pd

# Most content in this file generated with AI, using Claude Sonnet 4.6

# Standardizes text for embeddings and retrieval
def clean_text(text: str) -> str:
    """Lowercase, strip whitespace, remove special characters."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,./\'-]', '', text)
    return text

# Return clean list of tags
def normalize_tags(tags: str) -> list:
    """Split comma-separated tags into a clean list."""
    if not isinstance(tags, str):
        return []
    return [t.strip().lower() for t in tags.split(',') if t.strip()]

# Loads datasets and adds cleaned columns
def load_and_clean(menu_path: str, loc_path: str) -> tuple:
    """Load and clean both CSVs. Returns (menu_df, loc_df)."""
    menu_df = pd.read_csv(menu_path)
    loc_df  = pd.read_csv(loc_path)

    menu_df['name_clean']        = menu_df['name'].apply(clean_text)
    menu_df['description_clean'] = menu_df['description'].fillna(menu_df['name']).apply(clean_text)
    menu_df['meal_period_clean'] = menu_df['meal_period'].apply(clean_text)
    menu_df['tags_list']         = menu_df['generated_tags'].apply(normalize_tags)

    loc_df['name_clean']    = loc_df['name'].apply(clean_text)
    loc_df['cuisine_clean'] = loc_df['cuisine'].apply(clean_text)
    loc_df['tags_list']     = loc_df['tags'].apply(normalize_tags)

    return menu_df, loc_df

# Merges food items with dining location information
def merge_data(menu_df: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge menu items with location info."""
    return menu_df.merge(
        loc_df[['location_id', 'name', 'name_clean', 'cuisine_clean',
                'tags_list', 'hours', 'location']],
        on='location_id',
        suffixes=('_item', '_location')
    )

# Builds text blob that is to be embedded - provides more semantic info
def build_item_combo(row) -> str:
    """Rich text blob for a single menu item. What gets embedded."""
    item_tags = ' '.join(row['tags_list_item'])
    loc_tags  = ' '.join(row['tags_list_location'])
    return (
        f"{row['name_clean_item']} is a {row['meal_period_clean']} item "
        f"at {row['name_clean_location']}, a {row['cuisine_clean']} dining spot. "
        f"Description: {row['description_clean']}. "
        f"Tags: {item_tags}. Venue tags: {loc_tags}. "
        f"Hours: {row['hours']}."
    )

# builds chunks based on location and meal period
def build_chunks(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Build chunk-level data by grouping items per location + meal period."""
    chunks = []
    for (loc_id, meal), group in merged_df.groupby(['location_id', 'meal_period']):
        loc_name = group.iloc[0]['name_location']
        hours    = group.iloc[0]['hours']
        cuisine  = group.iloc[0]['cuisine_clean']

        items_text = '\n'.join(
            f"  - {row['name_item']}: {row['description_clean']} (tags: {row['generated_tags']})"
            for _, row in group.iterrows()
        )

        blob = (
            f"{loc_name} ({cuisine}) serves the following {meal} items:\n"
            f"{items_text}\n"
            f"Hours: {hours}"
        )

        chunks.append({
            'chunk_id':    f"{loc_id}_{meal}",
            'location':    loc_name,
            'meal_period': meal,
            'hours':       hours,
            'cuisine':     cuisine,
            'items':       group['name_item'].tolist(),
            'tags':        ' '.join(group['generated_tags'].dropna()),
            'text_blob':   blob,
        })

    return pd.DataFrame(chunks)
