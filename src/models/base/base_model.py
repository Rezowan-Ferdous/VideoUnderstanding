"""Base model class for all vision/VLM/VLA models"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_type = config.get('model_type', 'unknown')
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented"""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute loss - must be implemented"""
        pass
    
    def get_config(self):
        """Get model configuration"""
        return self.config
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights"""
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(self.state_dict(), path)