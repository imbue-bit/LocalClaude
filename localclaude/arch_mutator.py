import logging
from transformers import PreTrainedModel
from .components import MUTATOR_REGISTRY

logger = logging.getLogger(__name__)

def apply_architecture_search_space(model: PreTrainedModel, tokenizer, params: dict) -> tuple:
    config = model.config
    
    mutators = [cls() for name, cls in MUTATOR_REGISTRY.items()]
    
    for mutator in mutators:
        model = mutator.mutate(model, config, params)
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad)
    logger.info(f"Applied {len(mutators)} mutators. Total params: {total_params / 1e9:.2f}B")
    
    return model, tokenizer, total_params