import pandas as pd

# Access data from CSVs
def load_data(food_items, duke_locations):
    food = pd.read_csv(food_items)
    locations = pd.read_csv(duke_locations, encoding='latin1', on_bad_lines='skip')
    return food, locations

# Fix food data
def fix_food_items_data(df):
    # remove extra whitespace
    df['name'] = df['name'].str.strip()
    df['description'] = df['description'].str.strip()

    # fix meal period inconsistencies
    df['meal_period'] = df['meal_period'].str.strip().str.lower()   
    return df

# Fix location data
def fix_locations(df):
    # remove extra whitespace
    df['name'] = df['name'].str.strip()

    # clean cuisine
    df['cuisine'] = df['cuisine'].str.strip().str.lower()

    # clean description
    df['description'] = df['description'].str.strip().str.replace(r'\n+', ' ', regex=True)

    # clean tags
    df['tags'] = df['tags'].str.strip().str.lower()
    df['tags'] = df['tags'].apply(
        lambda x: ','.join([t.strip() for t in x.split(',') if t.strip()])
    )   
    return df

# Merge data
def merge_tags(df):
    df['all_tags'] = df['generated_tags'].apply(
        lambda x: ', '.join(sorted(set(t.strip().lower() for t in str(x).split(',') if t.strip())))
    )
    return df

# Merge files
def merge_files(food, locations):
    merged = food.merge(
        locations,
        on='location_id',
        how='left',
        suffixes=('_item', '_loc')
    )
    print("Columns after merge:", merged.columns.tolist()) 
    merged = merged.rename(columns={
    'name_item':        'item_name',
    'name_loc':         'loc_name',
    'description_item': 'item_description',
    'description_loc':  'loc_description',
    })
    return merged

# what gets embedded
def build_doc(row):
    return (
        f"{row['item_name']}. "
        f"Available at {row['loc_name']} ({row['cuisine']} cuisine) "
        f"in {row['location']} on {row['campus']} campus. "
        f"Hours: {row['hours']}. "
        f"Served during: {row['meal_period']}. "
        f"{row['item_description']}. "
        f"Item tags: {row['all_tags']}. "     
        f"Venue tags: {row['tags']}. "                            
    )

def create_dataset(food_items_path, duke_locations_path, output_path):
    food, locations = load_data(food_items_path, duke_locations_path)

    # normalize the data
    food = fix_food_items_data(food)
    food = merge_tags(food)
    locations = fix_locations(locations)

    merged_files = merge_files(food, locations)
    merged_files['document'] = merged_files.apply(build_doc, axis=1)

    print(merged_files[merged_files['loc_name'].isnull()][['item_id', 'location_id', 'item_name']])
    assert len(merged_files) == 394,                        f"Expected 394 rows, got {len(merged_files)}"
    assert merged_files['document'].isnull().sum() == 0,    "Some documents are null"
    assert merged_files['loc_name'].isnull().sum() == 0,    "Some items missing location"

    merged_files.to_csv(output_path, index=False, encoding='utf-8')
    
    return merged_files


if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, '..', 'data')
    
    create_dataset(
        food_items_path='data/raw/menu_items_tagged.csv',
        duke_locations_path='data/raw/WU_locations_Updated.csv',
        output_path='data/menu_processed.csv'
    )