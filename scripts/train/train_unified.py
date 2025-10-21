"""
Unified training script for all tasks
Usage:
  python scripts/train/train_unified.py task=detection model=yolov8
  python scripts/train/train_unified.py task=vlm_captioning model=blip2
  python scripts/train/train_unified.py task=vla_manipulation model=rt2
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.pipelines.unified_pipeline import UnifiedPipeline


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    if cfg.get('seed'):
        import pytorch_lightning as pl
        pl.seed_everything(cfg.seed)
    
    # Create pipeline
    pipeline = UnifiedPipeline(cfg)
    
    # Train
    if cfg.mode == 'train':
        pipeline.train()
    
    # Evaluate
    elif cfg.mode == 'eval':
        pipeline.evaluate(cfg.checkpoint)
    
    # Train and evaluate
    elif cfg.mode == 'train_eval':
        trainer = pipeline.train()
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        pipeline.evaluate(best_checkpoint)


if __name__ == "__main__":
    main()