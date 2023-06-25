import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class CLIPClassifier(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_classes, 
                 pretrain_model_name="openai/clip-vit-base-patch32", 
                 combine_mode="concat_mlp"):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(pretrain_model_name)
        self.clip_model.eval()  # Freezing the CLIP model
        self.processor = CLIPProcessor.from_pretrained(pretrain_model_name)
        self.combine_mode = combine_mode

        # Define MLP head or transformer according to combine_mode
        hidden_size = self.clip_model.config.text_config.hidden_size
        if combine_mode == "concat_mlp":
            self.classifier_head = MLPHead(hidden_size * 2, hidden_dim, num_classes)
        elif combine_mode == "sum":
            self.classifier_head = MLPHead(hidden_size, hidden_dim, num_classes)
        elif combine_mode == "transformer":
            self.transformer = nn.Transformer(hidden_size, num_heads=2)
            self.classifier_head = MLPHead(hidden_size, hidden_dim, num_classes)

    def forward(self, inputs):
        outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Combine embeddings according to combine_mode
        if self.combine_mode == "sum":
            combined = image_embeds + text_embeds
        elif self.combine_mode == "concat_mlp":
            combined = torch.cat((image_embeds, text_embeds), dim=1)
        elif self.combine_mode == "transformer":
            combined = torch.cat((image_embeds.unsqueeze(1), text_embeds.unsqueeze(1)), dim=1)
            combined = self.transformer(combined).mean(dim=1)
        else:
            raise ValueError("combine_mode must be one of 'sum', 'concat_mlp' or 'transformer'")

        # Feed combined embeddings to classifier head
        logits = self.classifier_head(combined)
        return logits
    
if __name__ == "__main__":
    model = CLIPClassifier(512, 10)