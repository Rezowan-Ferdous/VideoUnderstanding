import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
import detectron2.data.transforms as T

from src.models.base.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('detectron2')
class Detectron2Wrapper(BaseModel):
    """
    Wrapper for Detectron2 models
    Supports: Faster R-CNN, Mask R-CNN, RetinaNet, etc.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize Detectron2 config
        self.cfg = get_cfg()
        
        # Load model config
        model_config = config.get('model_config', 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        
        # Set model parameters
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.get('num_classes', 80)
        
        # Load pretrained weights if specified
        if config.get('pretrained', True):
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        
        # Training settings
        self.cfg.SOLVER.IMS_PER_BATCH = config.get('batch_size', 16)
        self.cfg.SOLVER.BASE_LR = config.get('learning_rate', 0.00025)
        self.cfg.SOLVER.MAX_ITER = config.get('max_iter', 10000)
        self.cfg.SOLVER.STEPS = config.get('lr_steps', (7000, 9000))
        
        # Evaluation
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.get('score_threshold', 0.5)
        
        self.model = None
        self.predictor = None
    
    def build_model(self):
        """Build the Detectron2 model"""
        from detectron2.modeling import build_model
        self.model = build_model(self.cfg)
        return self.model
    
    def get_predictor(self):
        """Get predictor for inference"""
        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        return self.predictor
    
    def forward(self, batch):
        """Forward pass"""
        if self.model is None:
            self.build_model()
        return self.model(batch)
    
    def predict(self, image):
        """Inference on single image"""
        predictor = self.get_predictor()
        outputs = predictor(image)
        return outputs
    
    def compute_loss(self, outputs, targets):
        """Compute loss - handled by Detectron2"""
        if isinstance(outputs, dict) and 'loss_cls' in outputs:
            total_loss = sum(outputs.values())
            return total_loss
        return None


@MODEL_REGISTRY.register('mask_rcnn')
class MaskRCNNWrapper(Detectron2Wrapper):
    """Wrapper specifically for Mask R-CNN"""
    
    def __init__(self, config):
        config['model_config'] = config.get(
            'model_config',
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        )
        super().__init__(config)
