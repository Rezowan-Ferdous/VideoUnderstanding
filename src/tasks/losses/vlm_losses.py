import torch
from torch import nn
import torch.nn.functional as F

class SymmetricCrossEntropyLoss(nn.Module):
    """
    Implements the symmetric cross-entropy loss for CLIP.
    
    This version accepts a pre-computed 'targets' matrix to correctly
    handle batches with multiple positive (image, caption) pairs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_embeddings, text_embeddings, temperature, targets):
        """
        Calculates the loss.
        'targets' is a [batch_size, batch_size] matrix where
        targets[i, j] = 1 if image[i] and text[j] are a positive pair,
        and 0 otherwise.
        """
        # Calculate logits
        # [batch_size, embed_dim] @ [embed_dim, batch_size] -> [batch_size, batch_size]
        logits_per_image = (image_embeddings @ text_embeddings.T) / temperature
        logits_per_text = logits_per_image.T # [batch_size, batch_size]

        # --- Soft labels ---
        # Normalize the target matrix. If an image `i` has 2 correct
        # captions in the batch, its target row becomes [0, ..., 0.5, ..., 0.5, ...]
        # This prevents the model from being penalized for splitting its probability
        # mass between two correct answers.
        targets_per_image = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
        targets_per_text = targets.T / (targets.T.sum(dim=1, keepdim=True) + 1e-8)

        # Calculate cross-entropy loss with soft labels
        loss_image = F.cross_entropy(logits_per_image, targets_per_image)
        loss_text = F.cross_entropy(logits_per_text, targets_per_text)
        
        # Average the two losses
        loss = (loss_image + loss_text) / 2.0
        return loss

