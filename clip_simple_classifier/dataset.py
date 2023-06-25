import json
import os
from collections import Counter
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from transformers import CLIPProcessor


class JsonDataset(Dataset):
    def __init__(self, annot_path, num_bins, max_seq_length):
        self.data = [json.loads(l) for l in open(annot_path)][0]
        self.data_dir = os.path.dirname(annot_path)
        self.data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.max_seq_length = max_seq_length
        self.num_bins = num_bins
        self.bin_edges, self.labels = self.create_labels()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        text = item["caption"][:self.max_seq_length]
        img = Image.open(item["image_path"]).convert("RGB")
        label_idx = self.num_to_bin(float(item["price"]))
        return {
            'text': text,
            'img': img,
            'label': label_idx,
        }
    
    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs
    
    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels
    
    def create_labels(self,):
        # Compute quantiles
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        prices = np.array([float(annot["price"]) for annot in self.data])
        prices = prices[~np.isnan(prices)]
        bin_edges = np.quantile(prices, quantiles)
        labels = ["%.2f - %.2f" % (bin_edges[i], bin_edges[i+1]) for i in range(self.num_bins)]
        return bin_edges, labels
        
    def num_to_bin(self, num):
        for idx in range(self.num_bins):
            if num >= self.bin_edges[idx] and num < self.bin_edges[idx+1]:
                return idx
        return self.num_bins - 1
    
    
data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def collate_fn(batch):
    text = [item['text'] for item in batch]
    images = [item['img'] for item in batch]
    inputs = data_processor(text=text, images=images, return_tensors="pt", padding=True)
    labels = torch.tensor([item['label'] for item in batch])
    return inputs, labels

    
    
if __name__ == "__main__":
    # test
    dataset = JsonDataset('data/antiques/annotations2.json', 10, 100)
    train, val = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    for x in val_dataloader:
        print(x)
        break
