import sys
sys.path.append("/data/home/spektor/clip")
import os.path as osp
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import wandb

from clip_simple_classifier.clip_classifier import CLIPClassifier
from clip_simple_classifier.dataset import JsonDataset, collate_fn, random_split

from config import Config

# HIDDEN_DIM = 512
# MAX_SEQ_LENGTH = 64
# EPOCHS = 100
# BATCH_SIZE = 64
# CHECKPOINT_INTERVAL = 5

def main():
    wandb.init(project='clip-classifier-training')
    config = Config()
    wandb.config.update(vars(config))
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # if GPU is unavailable
    clip_classifier = CLIPClassifier(config.hidden_dim, config.num_classes).to(device)
    optimizer = Adam(clip_classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = JsonDataset(config.annot_path, config.num_classes, config.max_seq_length)
    
    train, val = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Train the model
    num_epochs = config.num_epochs
    checkpoint_interval = config.checkpoint_interval
    for epoch in range(num_epochs):
        clip_classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = clip_classifier(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Compute the number of correct predictions and update the total
            total_correct += (logits.argmax(1) == labels).float().sum().item()
            total_samples += labels.size(0)
        
        # Compute the average loss and accuracy
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_correct / total_samples
        
        wandb.log({"Train Loss": avg_loss, "Train Accuracy": avg_accuracy})
        
        # Validation
        clip_classifier.eval()
        total_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                # Forward pass
                logits = clip_classifier(**inputs)
                _, predicted = torch.max(logits, 1)
                ttotal_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_accuracy = total_val_correct / total_val_samples
        print(f"Epoch: {epoch+1}/{num_epochs}; Loss: {avg_loss}; Validation Accuracy: {avg_val_accuracy}")
        wandb.log({"Validation Accuracy": avg_val_accuracy})
        
        # Checkpointing
        if (epoch + 1) % checkpoint_interval == 0:
            save_path = osp.join(config.checkpoint_dir, f"clip_classifier_{epoch+1}.pt")
            torch.save(clip_classifier.state_dict(), save_path)
            print(f"Checkpoint saved at {save_path}")
            
if __name__ == "__main__":
    main()