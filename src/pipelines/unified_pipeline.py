"""
Unified pipeline that orchestrates all tasks
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.utils.registry import TASK_REGISTRY, DATASET_REGISTRY
from src.data.datamodules.base_datamodule import BaseDataModule


class UnifiedPipeline:
    """Unified pipeline for all vision/VLM/VLA tasks"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task._target_.split('.')[-2]  # detection/vlm/vla
        
        # Setup logging
        self.setup_logging()
        
        # Setup callbacks
        self.callbacks = self.setup_callbacks()
    
    def setup_logging(self):
        """Setup experiment logging"""
        loggers = []
        
        if self.config.get('logging', {}).get('wandb', False):
            loggers.append(WandbLogger(
                project=self.config.project_name,
                name=self.config.experiment_name
            ))
        
        if self.config.get('logging', {}).get('tensorboard', True):
            loggers.append(TensorBoardLogger(
                save_dir=self.config.output_dir,
                name=self.config.experiment_name
            ))
        
        self.loggers = loggers
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.config.output_dir}/checkpoints",
            filename='{epoch}-{val_loss:.2f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.get('early_stopping', {}).get('enabled', False):
            early_stop_callback = EarlyStopping(
                monitor='val/loss',
                patience=self.config.early_stopping.patience,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        return callbacks
    
    def prepare_data(self):
        """Prepare dataset"""
        # Instantiate datamodule from config
        self.datamodule = hydra.utils.instantiate(self.config.data)
    
    def build_task(self):
        """Build task from config"""
        self.task = hydra.utils.instantiate(self.config.task)
    
    def train(self):
        """Train the model"""
        self.prepare_data()
        self.build_task()
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config.trainer.max_epochs,
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision=self.config.trainer.precision,
            logger=self.loggers,
            callbacks=self.callbacks,
            gradient_clip_val=self.config.trainer.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=self.config.trainer.get('accumulate_grad_batches', 1),
            log_every_n_steps=self.config.trainer.get('log_every_n_steps', 10),
        )
        
        # Train
        trainer.fit(
            self.task,
            datamodule=self.datamodule,
            ckpt_path=self.config.get('resume_from', None)
        )
        
        return trainer
    
    def evaluate(self, checkpoint_path: str = None):
        """Evaluate the model"""
        if checkpoint_path is None:
            checkpoint_path = f"{self.config.output_dir}/checkpoints/last.ckpt"
        
        # Load task from checkpoint
        task = self.task.__class__.load_from_checkpoint(checkpoint_path)
        
        # Setup trainer
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=1,
            logger=False
        )
        
        # Test
        results = trainer.test(task, datamodule=self.datamodule)
        
        return results