import os
import numpy as np
import pandas as pd

from currency_converter import CurrencyConverter
from collections import namedtuple
import requests
import shutil
import random
import unicodedata
import re
from tqdm import tqdm

Item = namedtuple("Item", "filename text price")


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')



def download_and_create_pair(im_url, im_file_name, txt_file_name, item, price, 
							 price_file_name, rewrite=False):
    # image_filename = wget.download(im_url)
    #
    # print('Image Successfully Downloaded: ', image_filename)
    watchdog = 0
    watchdogfsize = 0
    while True:
        r = requests.get(im_url, stream=False)
        print('Fetching image url---> %s' % (im_url))
        if watchdog > 30 or watchdogfsize > 0:
            print('Watchdog!!! status code returned false for more than 30 requests')
            print('Watchdog!!! zero file size counter == %d' % (watchdogfsize))
            break
        # Check if the image was retrieved successfully
        if r.status_code == 200:
            print('Got status ok on url---> %s' % (im_url))
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            #r.raw.decode_content = True
            
            print('URL content size---> %d' % (len(r.content)))
            # Open a local file with wb ( write binary ) permission.
            with open(im_file_name, "wb") as f:
                f.write(r.content)
            #with open(im_file_name, 'wb') as f:
            #    shutil.copyfileobj(r.raw, f)

            if rewrite:
                fp = open(txt_file_name, 'w')
                fp.write(item)
                fp.close()
				
                fp = open(price_file_name, 'w')
                fp.write(str(price))
                fp.close()
            
            #import pdb; pdb.set_trace();
            fsize = os.stat(im_file_name).st_size

            if len(r.content) > 3000:
                print('File size is ok---> %s' % (im_file_name))
                break
            else:
                print('File size is ZERO!!!---> %s' % (im_file_name))
                watchdogfsize += 1
                os.remove(im_file_name)
                os.remove(txt_file_name)
                os.remove(price_file_name)
        else:
            print('Got status FALSE on url!!!---> %s' % (im_url))
            watchdog += 1
                            
                            
                            
def prepare_ebay_data(csv_name="BERRY-SAMPLE.xlsx", download=False, rewrite=False):

    c = CurrencyConverter()
    df = pd.read_excel(csv_name, index_col=0, engine="openpyxl")

    items = [row for _, row in df.iterrows()]
    all_items = []
    for it_idx, item in enumerate(tqdm(items[::-1])):
        item_images = [url for n, url in item.items() if 'IMAGE' in n and
                       isinstance(url, str)]

        if isinstance(item['PRICE'], float) or isinstance(item['PRICE'], int):
            usd_price = float(item['PRICE'])
        else:
            usd_price = re.findall("\d+\.?\d*", item['PRICE'])[0]
            usd_price = float(usd_price)
            
        p = random.random() > 0.2
        file_dir = './data'
        file_dir += '/train' if p else '/val'
        if not np.isnan(usd_price) and item_images and isinstance(item[3], str):
            for im_idx, im_url in enumerate(item_images):
            
                os.makedirs(file_dir, exist_ok=True)
                name = slugify(item[3]) #.replace(" ", "_").replace(",", "").replace("/", "_")
                im_file_name = file_dir + '/' + name + str(im_idx) + '.jpg'
                txt_file_name = file_dir + '/' + name + str(im_idx) + '.txt'
                price_file_name = file_dir + '/' + name + str(im_idx) + '.price'
				
                if os.path.exists(im_file_name):
                    print('File already exists: %s' % (im_file_name))
                    continue
					
                if download:
                    download_and_create_pair(im_url, im_file_name, txt_file_name,
                                            item[3], usd_price, price_file_name, rewrite)
                    print('Image sucessfully Downloaded: ', im_file_name, im_url)
                else:
                    print('Image Couldn\'t be retreived')

                

         #       all_items.append(Item(name, item[3], usd_price))

   # return all_items


prepare_ebay_data("products.xlsx", True, True)
