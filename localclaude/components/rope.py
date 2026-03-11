from transformers import PreTrainedModel, LlamaConfig
from .base import BaseMutator, register_mutator

@register_mutator("rope")
class RoPEMutator(BaseMutator):
    def build_search_space(self, trial):
        return {
            "rope_base": trial.suggest_categorical("rope_base", [10000, 100000, 500000]),
            "rope_scaling": trial.suggest_categorical("rope_scaling", ["None", "NTK", "YaRN", "PI"])
        }

    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict) -> PreTrainedModel:
        config.rope_theta = float(params["rope_base"])
        rt = params["rope_scaling"]
        if rt == "NTK": config.rope_scaling = {"type": "dynamic", "factor": 2.0}
        elif rt == "YaRN": config.rope_scaling = {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192}
        elif rt == "PI": config.rope_scaling = {"type": "linear", "factor": 2.0}
        else: config.rope_scaling = None
        
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "rotary_emb") and hasattr(layer.self_attn.rotary_emb, "inv_freq"):
                del layer.self_attn.rotary_emb.inv_freq
        return model