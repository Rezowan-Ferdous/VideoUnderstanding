import torch
from torch import nn
import torch.nn.functional as F

class SymmetricCrossEntropyLoss(nn.Module):
    """
    Implements the symmetric cross-entropy loss for CLIP.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_embeddings, text_embeddings, temperature, targets):
        """
        Calculates the loss.
        'targets' is a [batch_size, batch_size] matrix where
        targets[i, j] = 1 if image[i] and text[j] are a positive pair.
        """
        # Calculate logits
        logits_per_image = (image_embeddings @ text_embeddings.T) / temperature
        logits_per_text = logits_per_image.T

        # Normalize the target matrix to create soft labels
        targets_per_image = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
        targets_per_text = targets.T / (targets.T.sum(dim=1, keepdim=True) + 1e-8)

        # Calculate cross-entropy loss with soft labels
        loss_image = F.cross_entropy(logits_per_image, targets_per_image)
        loss_text = F.cross_entropy(logits_per_text, targets_per_text)
        
        loss = (loss_image + loss_text) / 2.0
        return loss