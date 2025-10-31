import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import os

@hydra.main(config_path="../../conf", config_name="vlm_config", version_base=None)
def train(cfg: DictConfig):
    """
    Main training script.
    """
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    
    # --- 1. Setup Loggers (Reporting Pipeline) ---
    loggers = []
    if cfg.logging.tensorboard.enable:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir, 
            name="tensorboard_logs"
        )
        loggers.append(tb_logger)
        print(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")

    if cfg.logging.wandb.enable:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.logging.wandb.name,
            save_dir=cfg.output_dir
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model) # Log model gradients

    # --- 2. Instantiate DataModule ---
    print("Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # --- 3. Instantiate Task (LightningModule) ---
    print("Instantiating Task (LightningModule)...")
    task = hydra.utils.instantiate(cfg.task)

    # --- 4. Instantiate Callbacks ---
    callbacks = []
    for cb_name, cb_conf in cfg.callbacks.items():
        print(f"Instantiating Callback: {cb_name}")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # --- 5. Instantiate Trainer ---
    print("Instantiating Trainer...")
    trainer = pl.Trainer(
        **cfg.trainer, # Pass trainer settings from config
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
    )

    # --- 6. Start Training ---
    print("--- Starting Training ---")
    trainer.fit(task, datamodule=datamodule)
    
    print("--- Training Finished ---")
    print(f"Best model checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()
