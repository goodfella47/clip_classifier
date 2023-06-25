import os
from os import path as osp

import numpy as np
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from concurrent.futures import as_completed, ThreadPoolExecutor


def resize_img(img: Image, size=256):
    """
    Resize an image while maintaining its aspect ratio.
    """
    
    aspect_ratio = img.width / img.height

    if img.height < img.width:
        new_width = size 
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    return img.resize((new_width, new_height))

def is_valid_number(s):
    """
    Checks if a string can be converted to a floating-point number.

    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_product(product, cap_id_first, data_path):
    """
    Processes a product dictionary and generates annotation for its images.
    
    Parameters
    ----------
    product : dict
        The product dictionary containing product information.
    cap_id_first : int
        The initial caption ID.
    data_path : str
        The path where to save the images.
        
    Returns
    -------
    list of dict
        A list of dictionaries, each representing the annotation for an image.
    """
    annotation = []
    caption = product['PRODUCT NAME']
    price = product['PRICE']
    # check if price is a valid number
    if not is_valid_number(price) or not np.isnan(float(price)) or not isinstance(caption, str):
        return annotation
    
    cap_id = cap_id_first * 100
    # img_id = 100000000+cap_id*100
    img_id = 'f{cap_id:09d}'
    for image_url in product['IMAGES']:
        # download the image and resize the short side to 224, then save it to the path
        file_name = f"{img_id}.jpg"
        file_path = osp.join(data_path, file_name)

        try:
            # Send a HTTP request to the URL of the image
            response = requests.get(image_url)
            # Open the url image, resize it and save
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
        except:
            print(f"Error downloading image {image_url}")
            continue
        
        width_org, height_org = img.size
        if width_org <= 80 and height_org <= 80:
            continue
        img = resize_img(img)
        img.save(file_path)

        image_annotation = {"file_name": file_name,
                            "img_path": file_path,
                            "height": height_org,
                            "width": width_org,
                            "cap_id": cap_id,
                            "caption": caption,
                            "price": price}

        annotation.append(image_annotation)
        cap_id += 1

    return annotation

def main():
    """
    loads a product dataset, filters out only the antiques, processes their images and 
    annotations, and saves these annotations to a JSON file.
    """
    products = json.load(open("products.json"))
    # get only antiques
    antiques = [p for _, p in products.items() if p["CATEGORY"] == "antiques"]
    
    data_path = "data/antiques/images"
    if not osp.exists(data_path):
        os.makedirs(data_path)
    captions = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for cap_id, product in enumerate(antiques):
            futures.append(executor.submit(process_product, product, cap_id, data_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            annot = future.result()
            captions.extend(annot)
    
    # save the captions (dict to json)
    with open("data/antiques/annotations.json", "w") as f:
        json.dump(captions, f)
        

if __name__ == "__main__":
    main()