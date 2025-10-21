"""
Central registry for models, tasks, and datasets
"""
from typing import Dict, Callable, Any
import importlib


class Registry:
    """Registry for managing models, tasks, datasets"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry = {}
    
    def register(self, name: str = None):
        """Decorator to register items"""
        def decorator(cls):
            reg_name = name if name else cls.__name__
            self._registry[reg_name] = cls
            return cls
        return decorator
    
    def get(self, name: str):
        """Get registered item"""
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self.name} registry")
        return self._registry[name]
    
    def list(self):
        """List all registered items"""
        return list(self._registry.keys())


# Global registries
MODEL_REGISTRY = Registry('model')
TASK_REGISTRY = Registry('task')
DATASET_REGISTRY = Registry('dataset')
BACKBONE_REGISTRY = Registry('backbone')
HEAD_REGISTRY = Registry('head')
LOSS_REGISTRY = Registry('loss')