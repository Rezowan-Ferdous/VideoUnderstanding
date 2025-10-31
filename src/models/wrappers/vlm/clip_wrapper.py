from torch import nn
import torch.nn.functional as F

class CLIPModelWrapper(nn.Module):
    """
    This is the main nn.Module that combines the encoders and heads.
    It's instantiated by the `conf/model/multimodal/clip.yaml` config.
    """
    def __init__(self, image_encoder, text_encoder, 
                 image_projection, text_projection, temperature=1.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.text_projection = text_projection
        self.temperature = temperature

    def forward(self, batch):
        """
        Takes a batch from the CLIPDataModule and returns embeddings.
        """
        # Get image features
        image_features = self.image_encoder(batch["image"])
        
        # Get text features
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        
        # Project features to the shared embedding space
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # Normalize embeddings
        image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
        return image_embeddings_norm, text_embeddings_norm, self.temperature
