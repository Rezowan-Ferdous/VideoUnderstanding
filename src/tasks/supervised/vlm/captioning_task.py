"""Image Captioning Task"""
from src.tasks.base_task import BaseTask
from src.utils.registry import TASK_REGISTRY
from src.metrics.vlm_metrics import CaptioningMetrics


@TASK_REGISTRY.register('captioning')
class CaptioningTask(BaseTask):
    """Task for image captioning"""
    
    def __init__(self, config):
        super().__init__(config)
        self.metrics = CaptioningMetrics()
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch)
        
        # Generate captions
        generated = self.model.generate(batch['images'])
        references = batch['captions']
        
        # Compute metrics
        metrics = self.metrics.compute(generated, references)
        
        self.log('val/loss', loss)
        self.log_dict({f'val/{k}': v for k, v in metrics.items()})
        
        return loss