
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add SAM2 to path if in external
SAM2_PATH = Path(__file__).parent.parent.parent.parent / 'external' / 'segment-anything-2'
if SAM2_PATH.exists():
    sys.path.insert(0, str(SAM2_PATH))

from src.models.base.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('sam2')
class SAM2Wrapper(BaseModel):
    """
    Wrapper for Segment Anything Model 2
    Supports automatic mask generation and prompted segmentation
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Model configuration
            model_cfg = config.get('model_cfg', 'sam2_hiera_l.yaml')
            checkpoint = config.get('checkpoint', 'sam2_hiera_large.pt')
            
            # Build model
            self.sam2_model = build_sam2(model_cfg, checkpoint)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            
        except ImportError:
            print("SAM2 not found. Please install from: https://github.com/facebookresearch/segment-anything-2")
            self.sam2_model = None
            self.predictor = None
    
    def forward(self, batch):
        """Forward pass for training"""
        # SAM2 typically used for inference
        images = batch['images']
        return self.predict_batch(images)
    
    def predict(self, image, prompts=None):
        """
        Predict segmentation masks
        
        Args:
            image: PIL Image or numpy array
            prompts: dict with keys 'point_coords', 'point_labels', 'box', etc.
        """
        if self.predictor is None:
            raise RuntimeError("SAM2 model not initialized")
        
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict with prompts
        if prompts is None:
            # Auto-generate masks
            masks, scores, logits = self.predictor.predict(
                multimask_output=True
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts.get('point_coords'),
                point_labels=prompts.get('point_labels'),
                box=prompts.get('box'),
                multimask_output=True
            )
        
        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }
    
    def auto_segment(self, image):
        """Automatic mask generation"""
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            masks = mask_generator.generate(image)
            return masks
            
        except ImportError:
            print("SAM2 mask generator not available")
            return None
    
    def compute_loss(self, outputs, targets):
        """SAM2 doesn't have built-in training loss"""
        # Implement custom loss if needed
        return torch.tensor(0.0)
