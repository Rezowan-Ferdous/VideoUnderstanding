import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import os

@hydra.main(config_path="../../conf", config_name="vlm_config", version_base=None)
def evaluate(cfg: DictConfig):
    """
    Main evaluation script.
    """
    pl.seed_everything(cfg.seed, workers=True)

    print("--- Evaluation Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------")

    # --- 1. Get Checkpoint Path ---
    if cfg.eval_checkpoint_path:
        ckpt_path = cfg.eval_checkpoint_path
    else:
        # If no path is given, try to find the 'best.ckpt' from a training run
        ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "best.ckpt")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        print("Please run training first or provide a valid 'eval_checkpoint_path' in the config.")
        return

    print(f"Loading checkpoint from: {ckpt_path}")

    # --- 2. Instantiate DataModule ---
    print("Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # --- 3. Instantiate Task (LightningModule) ---
    print("Instantiating Task from checkpoint...")
    # We load the task (which includes the model) from the checkpoint
    # This automatically loads the saved hyperparameters and weights
    task = hydra.utils.call(
        cfg.task,
    ).load_from_checkpoint(ckpt_path)

    # --- 4. Instantiate Trainer ---
    print("Instantiating Trainer...")
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=False, # Disable logging for evaluation
    )

    # --- 5. Start Evaluation ---
    print("--- Starting Evaluation ---")
    results = trainer.validate(task, datamodule=datamodule)
    
    print("--- Evaluation Finished ---")
    print(results)

if __name__ == "__main__":
    evaluate()
