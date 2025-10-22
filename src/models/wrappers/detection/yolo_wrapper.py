
from ultralytics import YOLO
import torch
import torch.nn as nn
from pathlib import Path

from src.models.base.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('yolo')
@MODEL_REGISTRY.register('yolov8')
class YOLOWrapper(BaseModel):
    """
    Wrapper for YOLO models (v8, v9, v10, v11)
    Supports detection, segmentation, pose, and OBB
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Model variant
        model_name = config.get('variant', 'yolov8n')
        self.task = config.get('task', 'detect')  # detect, segment, pose, obb
        
        # Initialize YOLO
        pretrained = config.get('pretrained', True)
        
        if pretrained:
            model_path = f'{model_name}.pt'
            if self.task == 'segment':
                model_path = f'{model_name}-seg.pt'
            elif self.task == 'pose':
                model_path = f'{model_name}-pose.pt'
            elif self.task == 'obb':
                model_path = f'{model_name}-obb.pt'
            
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(f'{model_name}.yaml')
        
        self.num_classes = config.get('num_classes', 80)
        self.img_size = config.get('img_size', 640)
    
    def train_model(self, data_yaml, **kwargs):
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to dataset YAML
            **kwargs: Training arguments
        """
        results = self.model.train(
            data=data_yaml,
            epochs=kwargs.get('epochs', 100),
            imgsz=kwargs.get('imgsz', self.img_size),
            batch=kwargs.get('batch', 16),
            device=kwargs.get('device', 0),
            workers=kwargs.get('workers', 8),
            project=kwargs.get('project', 'runs/detect'),
            name=kwargs.get('name', 'train'),
            exist_ok=kwargs.get('exist_ok', False),
            pretrained=kwargs.get('pretrained', True),
            optimizer=kwargs.get('optimizer', 'auto'),
            seed=kwargs.get('seed', 0),
            deterministic=kwargs.get('deterministic', True),
            single_cls=kwargs.get('single_cls', False),
            rect=kwargs.get('rect', False),
            cos_lr=kwargs.get('cos_lr', False),
            close_mosaic=kwargs.get('close_mosaic', 10),
            resume=kwargs.get('resume', False),
            amp=kwargs.get('amp', True),
            fraction=kwargs.get('fraction', 1.0),
            profile=kwargs.get('profile', False),
            freeze=kwargs.get('freeze', None),
            # Augmentation
            lr0=kwargs.get('lr0', 0.01),
            lrf=kwargs.get('lrf', 0.01),
            momentum=kwargs.get('momentum', 0.937),
            weight_decay=kwargs.get('weight_decay', 0.0005),
            warmup_epochs=kwargs.get('warmup_epochs', 3.0),
            warmup_momentum=kwargs.get('warmup_momentum', 0.8),
            box=kwargs.get('box', 7.5),
            cls=kwargs.get('cls', 0.5),
            dfl=kwargs.get('dfl', 1.5),
            pose=kwargs.get('pose', 12.0),
            kobj=kwargs.get('kobj', 1.0),
            label_smoothing=kwargs.get('label_smoothing', 0.0),
            nbs=kwargs.get('nbs', 64),
            hsv_h=kwargs.get('hsv_h', 0.015),
            hsv_s=kwargs.get('hsv_s', 0.7),
            hsv_v=kwargs.get('hsv_v', 0.4),
            degrees=kwargs.get('degrees', 0.0),
            translate=kwargs.get('translate', 0.1),
            scale=kwargs.get('scale', 0.5),
            shear=kwargs.get('shear', 0.0),
            perspective=kwargs.get('perspective', 0.0),
            flipud=kwargs.get('flipud', 0.0),
            fliplr=kwargs.get('fliplr', 0.5),
            mosaic=kwargs.get('mosaic', 1.0),
            mixup=kwargs.get('mixup', 0.0),
            copy_paste=kwargs.get('copy_paste', 0.0)
        )
        
        return results
    
    def validate(self, data_yaml=None, **kwargs):
        """Validate model"""
        results = self.model.val(
            data=data_yaml,
            imgsz=kwargs.get('imgsz', self.img_size),
            batch=kwargs.get('batch', 16),
            conf=kwargs.get('conf', 0.001),
            iou=kwargs.get('iou', 0.6),
            device=kwargs.get('device', 0),
            workers=kwargs.get('workers', 8),
            project=kwargs.get('project', 'runs/detect'),
            name=kwargs.get('name', 'val'),
            exist_ok=kwargs.get('exist_ok', False),
            half=kwargs.get('half', False),
            plots=kwargs.get('plots', True),
            save_json=kwargs.get('save_json', True),
            save_hybrid=kwargs.get('save_hybrid', False)
        )
        
        return results
    
    def forward(self, batch):
        """Forward pass"""
        images = batch['images'] if isinstance(batch, dict) else batch
        results = self.model(images)
        return results
    
    def predict(self, source, **kwargs):
        """Inference"""
        results = self.model.predict(
            source=source,
            imgsz=kwargs.get('imgsz', self.img_size),
            conf=kwargs.get('conf', 0.25),
            iou=kwargs.get('iou', 0.45),
            device=kwargs.get('device', 0),
            half=kwargs.get('half', False),
            max_det=kwargs.get('max_det', 300),
            vid_stride=kwargs.get('vid_stride', 1),
            stream_buffer=kwargs.get('stream_buffer', False),
            visualize=kwargs.get('visualize', False),
            augment=kwargs.get('augment', False),
            agnostic_nms=kwargs.get('agnostic_nms', False),
            classes=kwargs.get('classes', None),
            retina_masks=kwargs.get('retina_masks', False),
            embed=kwargs.get('embed', None),
            show=kwargs.get('show', False),
            save=kwargs.get('save', True),
            save_frames=kwargs.get('save_frames', False),
            save_txt=kwargs.get('save_txt', False),
            save_conf=kwargs.get('save_conf', False),
            save_crop=kwargs.get('save_crop', False),
            show_labels=kwargs.get('show_labels', True),
            show_conf=kwargs.get('show_conf', True),
            show_boxes=kwargs.get('show_boxes', True),
            line_width=kwargs.get('line_width', None)
        )
        
        return results
    
    def export(self, format='onnx', **kwargs):
        """Export model"""
        return self.model.export(format=format, **kwargs)
    
    def compute_loss(self, outputs, targets):
        """YOLO computes loss internally"""
        return None

