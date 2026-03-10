import torch
from trl import SFTTrainer

class ClaudeNAS_SFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        
        aux_loss = 0.0
        base_model = model.model if hasattr(model, "model") else model
        
        for module in base_model.modules():
            if hasattr(module, "current_aux_loss") and getattr(module, "current_aux_loss") is not None:
                if isinstance(module.current_aux_loss, torch.Tensor):
                    aux_loss += module.current_aux_loss
                
        total_loss = loss + aux_loss
        
        return (total_loss, outputs) if return_outputs else total_loss