"""Base class for Vision-Language Models"""
from .base_model import BaseModel
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch


class BaseVLM(BaseModel):
    """Base class for VLM models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = config.get('task', 'captioning')
        self.max_text_length = config.get('max_text_length', 512)
    
    @abstractmethod
    def encode_image(self, images: torch.Tensor):
        """Encode images to features"""
        pass
    
    @abstractmethod
    def encode_text(self, text: Union[str, list]):
        """Encode text to features"""
        pass
    
    @abstractmethod
    def generate(self, image: Image.Image, prompt: str = None, **kwargs):
        """Generate text from image"""
        pass


class BaseVLA(BaseModel):
    """Base class for Vision-Language-Action models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.action_dim = config.get('action_dim', 7)
        self.action_type = config.get('action_type', 'continuous')
    
    @abstractmethod
    def predict_action(self, observation: Dict, instruction: str):
        """Predict action from observation and instruction"""
        pass