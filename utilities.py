import json

def load_category_names(filename="cat_to_name.json"):
    # Load category names mapping
    with open(filename, "r") as f:
        cat_to_name = json.load(f)
    return cat_to_name