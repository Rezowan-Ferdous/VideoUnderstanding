from src.external.FUTR.model.futr import FUTR_Model
from torch import nn
import torch

class FUTRBackbone(nn.Module):
    """
    Wrapper for the FUTR model to serve as a spatiotemporal backbone.
    """
    def __init__(self, model_path=None, **kwargs):
        super().__init__()
        # Instantiate the original FUTR model
        self.model = FUTR_Model(**kwargs)
        
        # Load pre-trained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        # This will be the output dimension of the backbone
        self.output_dim = self.model.feature_dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (B, T, C, H, W)
        Returns:
            A tensor of shape (B, D) representing the video-level feature.
        """
        # The FUTR model likely expects input in (B, C, T, H, W) format
        x = x.permute(0, 2, 1, 3, 4) 
        
        # The forward pass of FUTR likely returns multiple things for its loss
        # We only need the final feature representation for downstream tasks
        _, features = self.model(x) 
        
        return features