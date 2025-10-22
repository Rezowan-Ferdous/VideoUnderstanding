
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection,
    AutoImageProcessor
)
import torch

from src.models.base.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('detr')
@MODEL_REGISTRY.register('rt-detr')
class DETRWrapper(BaseModel):
    """
    Wrapper for DETR models from Hugging Face
    Supports: DETR, Conditional DETR, RT-DETR, DETA, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Model name from HuggingFace
        model_name = config.get('model_name', 'facebook/detr-resnet-50')
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=config.get('num_classes', 91),
            ignore_mismatched_sizes=True
        )
        
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def forward(self, batch):
        """Forward pass"""
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values'].to(self.device)
            pixel_mask = batch.get('pixel_mask', None)
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)
            
            labels = batch.get('labels', None)
            
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
        else:
            outputs = self.model(batch.to(self.device))
        
        return outputs
    
    def predict(self, image):
        """Inference on single image"""
        # Process image
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5
        )[0]
        
        return results
    
    def compute_loss(self, outputs, targets):
        """Compute loss"""
        return outputs.loss if hasattr(outputs, 'loss') else None
