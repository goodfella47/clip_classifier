import pandas as pd
from tqdm import tqdm
import json

col_names = ['CATEGORY', 'URL', 'THUMBNAIL', 'SOLD DATE', 'PRODUCT NAME', 
             'PRICE', 'SHIPPING COST', 'LOCATION', 'SELLER', 'REVIEWS', 'RATING', 
             'STORE NAME', 'STORE URL', 'BIDS', 'ITEM NUMBER']

img_col_names = [f'IMAGE {i+1}' for i in range(116)]

def xslx_to_json(file_path):
    json_dict = {}
    df = pd.read_excel(file_path, engine="openpyxl")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        item_row = {}
        for col in col_names:
            item_row[col] = row[col]
        img_urls = [row[col] for col in img_col_names if isinstance(row[col], str)]
        item_row['IMAGES'] = img_urls
        json_dict[index] = item_row
        
    json_file_path = file_path.replace(".xlsx", ".json")
    with open(json_file_path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, default=str)
        
xslx_to_json("products.xlsx")