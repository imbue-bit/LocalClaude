import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaConfig
from transformers.activations import ACT2FN
from .base import BaseMutator, register_mutator

ACT2FN.setdefault("squared_relu", lambda x: torch.pow(torch.nn.functional.relu(x), 2))
ACT2FN.setdefault("mish", nn.functional.mish)
ACT2FN.setdefault("quickgelu", lambda x: x * torch.sigmoid(1.702 * x))

@register_mutator("ffn")
class FFNMutator(BaseMutator):
    def build_search_space(self, trial):
        return {
            "ffn_activation": trial.suggest_categorical("ffn_activation", ["silu", "gelu", "quickgelu", "mish", "squared_relu"]),
            "gating_bias": trial.suggest_categorical("gating_bias", ["none", "learnable"])
        }

    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict) -> PreTrainedModel:
        config.hidden_act = params["ffn_activation"]
        
        if params["gating_bias"] == "learnable":
            for layer in model.model.layers:
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    proj = layer.mlp
                    linear = getattr(proj, proj_name, None)
                    if linear and linear.bias is None:
                        new_linear = nn.Linear(
                            linear.in_features,
                            linear.out_features,
                            bias=True,
                            device=linear.weight.device,
                            dtype=linear.weight.dtype,
                        )
                        new_linear.weight.data.copy_(linear.weight.data)
                        nn.init.zeros_(new_linear.bias)
                        setattr(proj, proj_name, new_linear)
        return model
