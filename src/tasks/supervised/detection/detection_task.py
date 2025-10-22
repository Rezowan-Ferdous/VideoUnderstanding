
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
import numpy as np

from src.tasks.base_task import BaseTask
from src.utils.registry import TASK_REGISTRY, MODEL_REGISTRY


@TASK_REGISTRY.register('detection')
class DetectionTask(BaseTask):
    """
    Object Detection Task with comprehensive logging
    Supports multiple detection frameworks
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize metrics
        self.train_map = MeanAveragePrecision()
        self.val_map = MeanAveragePrecision()
        
        # Store predictions for visualization
        self.val_predictions = []
        self.val_targets = []
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        
        # Compute loss
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = self.criterion(outputs, batch)
        
        # Log loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        current_lr = opt.param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        
        # Compute loss
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = self.criterion(outputs, batch)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Prepare predictions and targets for mAP calculation
        preds = self._prepare_for_map(outputs)
        targets = self._prepare_targets_for_map(batch)
        
        # Update metrics
        self.val_map.update(preds, targets)
        
        # Store for visualization
        if batch_idx < 5:  # Store first 5 batches
            self.val_predictions.append(preds)
            self.val_targets.append(targets)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log mAP
        map_dict = self.val_map.compute()
        
        self.log('val/map', map_dict['map'], prog_bar=True)
        self.log('val/map_50', map_dict['map_50'])
        self.log('val/map_75', map_dict['map_75'])
        self.log('val/map_small', map_dict['map_small'])
        self.log('val/map_medium', map_dict['map_medium'])
        self.log('val/map_large', map_dict['map_large'])
        
        # Per-class AP
        if 'map_per_class' in map_dict:
            for idx, ap in enumerate(map_dict['map_per_class']):
                self.log(f'val/map_class_{idx}', ap)
        
        # Log visualizations to TensorBoard
        if len(self.val_predictions) > 0:
            self._log_predictions_to_tensorboard()
        
        # Reset metrics
        self.val_map.reset()
        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def _prepare_for_map(self, outputs):
        """Convert model outputs to format for torchmetrics"""
        # This needs to be customized based on model type
        # Format: List[Dict[str, Tensor]] with keys: boxes, scores, labels
        
        if hasattr(outputs, 'logits'):
            # DETR-style outputs
            return self._detr_to_map_format(outputs)
        elif isinstance(outputs, list):
            # YOLO-style outputs
            return self._yolo_to_map_format(outputs)
        else:
            # Default handling
            return outputs
    
    def _prepare_targets_for_map(self, batch):
        """Convert targets to format for torchmetrics"""
        # Format: List[Dict[str, Tensor]] with keys: boxes, labels
        
        if 'labels' in batch:
            targets = batch['labels']
        elif 'targets' in batch:
            targets = batch['targets']
        else:
            targets = []
        
        return targets
    
    def _detr_to_map_format(self, outputs):
        """Convert DETR outputs to mAP format"""
        # Process DETR outputs
        preds = []
        
        logits = outputs.logits
        boxes = outputs.pred_boxes
        
        for logit, box in zip(logits, boxes):
            scores = logit.softmax(-1)
            scores, labels = scores.max(-1)
            
            # Filter background class
            keep = labels != outputs.logits.shape[-1] - 1
            
            preds.append({
                'boxes': box[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
        
        return preds
    
    def _yolo_to_map_format(self, outputs):
        """Convert YOLO outputs to mAP format"""
        preds = []
        
        for output in outputs:
            if hasattr(output, 'boxes'):
                preds.append({
                    'boxes': output.boxes.xyxy,
                    'scores': output.boxes.conf,
                    'labels': output.boxes.cls.int()
                })
        
        return preds
    
    def _log_predictions_to_tensorboard(self):
        """Log prediction visualizations to TensorBoard"""
        if not self.logger:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx in range(min(6, len(self.val_predictions))):
            if idx >= len(self.val_predictions):
                break
            
            # Get prediction and target
            pred = self.val_predictions[idx][0] if len(self.val_predictions[idx]) > 0 else None
            target = self.val_targets[idx][0] if len(self.val_targets[idx]) > 0 else None
            
            # Create visualization (simplified)
            ax = axes[idx]
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
            
            # In practice, you'd draw boxes on actual images
            # This is a placeholder
            ax.text(0.5, 0.5, 'Detection Visualization', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.logger.experiment.add_figure(
            'val/predictions',
            fig,
            self.current_epoch
        )
        plt.close(fig)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer'].get('weight_decay', 0.01)
        )
        
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['trainer']['max_epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_config.get('milestones', [60, 90]),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
