# config.py

class Config:
    def __init__(self):
        self.annot_path = "data/antiques/annotations2.json"
        self.model_name = "openai/clip-vit-base-patch32"
        self.hidden_dim = 256
        self.num_classes = 10
        self.num_bins = 10
        self.max_seq_length = 128
        self.combine_mode = "concat_mlp"
        self.batch_size = 64
        self.num_epochs = 10
        self.checkpoint_interval = 5
        self.checkpoint_dir = 'clip_simple_classifier/output'
