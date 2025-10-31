# from torch import nn
# import torch.nn.functional as F

# import hydra
# from omegaconf import DictConfig
# from torch import nn
# import torch.nn.functional as F

# class CLIPWrapper(nn.Module):
#     """
#     This is the main nn.Module that combines the encoders and heads.
#     It's instantiated by the `conf/model/multimodal/clip.yaml` config.
#     """
#     def __init__(self, 
#                  vision_encoder: DictConfig, 
#                  text_encoder: DictConfig, 
#                  image_projection: DictConfig, 
#                  text_projection: DictConfig, 
#                  temperature: float = 0.07,
#                  **kwargs):
#         super().__init__()
#         # Instantiate all the parts from their configs
#         self.vision_encoder = hydra.utils.instantiate(vision_encoder)
#         self.text_encoder = hydra.utils.instantiate(text_encoder)
#         self.image_projection = hydra.utils.instantiate(image_projection)
#         self.text_projection = hydra.utils.instantiate(text_projection)
        
#         self.temperature = temperature

#     def forward(self, batch):
#         """
#         Takes a batch from the CLIPDataModule and returns embeddings.
#         """
#         # Get image features
#         image_features = self.vision_encoder(batch["image"])
        
#         # Get text features
#         text_features = self.text_encoder(
#             input_ids=batch["input_ids"], 
#             attention_mask=batch["attention_mask"]
#         )
        
#         # Project features to the shared embedding space
#         image_embeddings = self.image_projection(image_features)
#         text_embeddings = self.text_projection(text_features)
        
#         # Normalize embeddings
#         image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
#         text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
#         # Return normalized embeddings and the temperature
#         return image_embeddings_norm, text_embeddings_norm, self.temperature
    
# import hydra
# from omegaconf import DictConfig
# from torch import nn
# import torch.nn.functional as F

# # Import the actual classes so Python knows their type
# from src.models.backbones.vision_encoder import VisionEncoder
# from src.models.backbones.text_encoder import TextEncoder
# from src.models.heads.projection_head import ProjectionHead

# class CLIPWrapper(nn.Module):
#     """
#     This is the main nn.Module that combines the encoders and heads.
#     Hydra instantiates all components *before* calling __init__.
#     """
#     def __init__(self, 
#                  vision_encoder: VisionEncoder,     # <--- **THE FIX (Part 1)**: This is now an object, not a config
#                  text_encoder: TextEncoder,       # <--- **THE FIX (Part 1)**
#                  image_projection: ProjectionHead,  # <--- **THE FIX (Part 1)**
#                  text_projection: ProjectionHead,   # <--- **THE FIX (Part 1)**
#                  temperature: float = 0.07,
#                  **kwargs):
#         super().__init__()

#         # --- **THE FIX (Part 2)**: Just assign the objects directly. No instantiate! ---
#         self.vision_encoder = vision_encoder
#         self.text_encoder = text_encoder
#         self.image_projection = image_projection
#         self.text_projection = text_projection
#         # --- **END OF FIX** ---
        
#         self.temperature = temperature

#     def forward(self, batch):
#         """
#         Takes a batch from the CLIPDataModule and returns embeddings.
#         """
#         # Get image features
#         image_features = self.vision_encoder(batch["image"])
        
#         # Get text features
#         text_features = self.text_encoder(
#             input_ids=batch["input_ids"], 
#             attention_mask=batch["attention_mask"]
#         )
        
#         # Project features to the shared embedding space
#         image_embeddings = self.image_projection(image_features)
#         text_embeddings = self.text_projection(text_features)
        
#         # Normalize embeddings
#         image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
#         text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
#         # Return normalized embeddings and the temperature
#         return image_embeddings_norm, text_embeddings_norm, self.temperature





import hydra
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F

# Import the actual classes so Python knows their type
from src.models.backbones.vision_encoder import VisionEncoder
from src.models.backbones.text_encoder import TextEncoder
from src.models.heads.projection_head import ProjectionHead

class CLIPWrapper(nn.Module):
    """
    This is the main nn.Module that combines the encoders and heads.
    Hydra instantiates all components *before* calling __init__.
    """
    def __init__(self, 
                 vision_encoder: VisionEncoder,     # <--- **THE FIX (Part 1)**: This is now an object, not a config
                 text_encoder: TextEncoder,       # <--- **THE FIX (Part 1)**
                 image_projection: ProjectionHead,  # <--- **THE FIX (Part 1)**
                 text_projection: ProjectionHead,   # <--- **THE FIX (Part 1)**
                 temperature: float = 0.07,
                 **kwargs):
        super().__init__()

        # --- **THE FIX (Part 2)**: Just assign the objects directly. No instantiate! ---
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.text_projection = text_projection
        # --- **END OF FIX** ---
        
        self.temperature = temperature

    def forward(self, batch):
        """
        Takes a batch from the CLIPDataModule and returns embeddings.
        """
        # Get image features
        image_features = self.vision_encoder(batch["image"])
        
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
        
        # Return normalized embeddings and the temperature
        return image_embeddings_norm, text_embeddings_norm, self.temperature

