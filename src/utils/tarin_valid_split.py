import json
from sklearn.model_selection import train_test_split

def split(train_json: dict, ratio: float=0.9):
    
    items = list(train_json.items())
    
    train_items, val_items = train_test_split(
        items,
        test_size=1-ratio,
        random_state=42
    )

    train_dict = dict(train_items)
    val_dict = dict(val_items)

    return train_dict, val_dict
