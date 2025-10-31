import timm
from torch import nn

class TimmVisionEncoder(nn.Module):
    """
    A wrapper for a timm image model (e.g., ResNet, ViT).
    """
    def __init__(self, model_name="resnet50", pretrained=True, trainable=False):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class VisionEncoder(nn.Module):
    """
    A wrapper for a timm image model (e.g., ViT).
    """
    def __init__(self, model_name: str, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        
        # Freeze parameters if not trainable
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)