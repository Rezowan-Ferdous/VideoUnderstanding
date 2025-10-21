"""Base task class using Lightning"""
import pytorch_lightning as pl
import torch
from typing import Dict, Any, Optional
from src.utils.registry import MODEL_REGISTRY, LOSS_REGISTRY


class BaseTask(pl.LightningModule):
    """Base task for all training tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Build model
        model_name = config['model']['name']
        self.model = MODEL_REGISTRY.get(model_name)(config['model'])
        
        # Build loss
        loss_name = config['loss']['name']
        self.criterion = LOSS_REGISTRY.get(loss_name)(config['loss'])
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch)
        
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch)
        
        self.log('val/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['trainer']['max_epochs']
        )
        
        return [optimizer], [scheduler]