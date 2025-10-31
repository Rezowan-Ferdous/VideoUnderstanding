import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

class CLIPPretrainTask(pl.LightningModule):
    """
    This is the core LightningModule for the CLIP pre-training task.
    It's instantiated by `conf/task/clip_pretrain.yaml`.
    
    This version correctly handles multiple positive captions for a
    single image within a batch.
    """
    def __init__(self, model_config: DictConfig, 
                 loss: DictConfig, 
                 optimizer: DictConfig, 
                 scheduler: DictConfig):
        super().__init__()
        # Save hyperparams, which makes them accessible via self.hparams
        self.save_hyperparameters()
        
        # 1. Instantiate the Model (e.g., CLIPWrapper)
        self.model = hydra.utils.instantiate(model_config)
        
        # 2. Instantiate the Loss
        self.loss_fn = hydra.utils.instantiate(loss)

    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)

    def _build_target_matrix(self, image_files, batch_size):
        """
        Builds the ground-truth target matrix.
        targets[i, j] = 1 if image_files[i] == image_files[j]
        
        This matrix identifies all (image, caption) pairs in the batch
        that belong to the same original image.
        """
        targets = torch.zeros(batch_size, batch_size, device=self.device)
        for i in range(batch_size):
            for j in range(batch_size):
                # If image at index i is the same as image at index j,
                # then (image[i], caption[j]) is a positive pair.
                if image_files[i] == image_files[j]:
                    targets[i, j] = 1
        return targets

    def _shared_step(self, batch, batch_idx):
        """
        A shared step for training and validation.
        Calculates loss and metrics.
        """
        # Get embeddings from the model
        # batch["image_file"] is a list of strings from the collate_fn
        image_embeds, text_embeds, temp = self(batch)
        
        batch_size = image_embeds.shape[0]
        
        # --- **HERE IS THE LOGIC** ---
        # Build the N x N ground-truth matrix
        targets = self._build_target_matrix(batch["image_file"], batch_size)
        
        # --- Calculate Loss ---
        loss = self.loss_fn(image_embeds, text_embeds, temp, targets)
        
        # --- Calculate Accuracy (Metrics) ---
        with torch.no_grad():
            logits_per_image = (image_embeds @ text_embeds.T) / temp
            
            # Get top-1 predictions
            pred_idx_per_image = logits_per_image.argmax(dim=1)
            
            # Check if the top-1 prediction is in the set of correct answers
            # targets[i, j] is 1 if (image_i, text_j) is a match
            acc = targets[
                torch.arange(batch_size, device=self.device), pred_idx_per_image
            ].mean()
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch['image'].shape[0])
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch['image'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['image'].shape[0])
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['image'].shape[0])
        return loss

    def configure_optimizers(self):
        """
        Instantiates the optimizer and learning rate scheduler.
        """
        # Instantiate optimizer from config
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        
        # Instantiate scheduler from config
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Metric to monitor for ReduceLROnPlateau
                "interval": "epoch",
                "frequency": 1,
            },
        }

