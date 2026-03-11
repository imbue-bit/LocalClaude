import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaConfig
from .base import BaseMutator, register_mutator

@register_mutator("norm")
class NormMutator(BaseMutator):
    def build_search_space(self, trial):
        return {
            "norm_position": trial.suggest_categorical("norm_position", ["pre-norm", "post-norm"]),
            "norm_type": trial.suggest_categorical("norm_type", ["rms", "layer_norm"]),
            "layer_skip_prob": trial.suggest_categorical("layer_skip_prob", [0.0, 0.05, 0.1])
        }

    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict) -> PreTrainedModel:
        # Norm 映射
        if params["norm_type"] == "layer_norm":
            for layer in model.model.layers:
                dim = layer.input_layernorm.weight.shape[0]
                new_in = nn.LayerNorm(dim, eps=layer.input_layernorm.variance_epsilon, dtype=model.dtype)
                new_in.weight.data.copy_(layer.input_layernorm.weight.data); new_in.bias.data.zero_()
                layer.input_layernorm = new_in.to(model.device)
                
                new_post = nn.LayerNorm(dim, eps=layer.post_attention_layernorm.variance_epsilon, dtype=model.dtype)
                new_post.weight.data.copy_(layer.post_attention_layernorm.weight.data); new_post.bias.data.zero_()
                layer.post_attention_layernorm = new_post.to(model.device)
                
            new_final = nn.LayerNorm(model.model.norm.weight.shape[0], eps=model.model.norm.variance_epsilon, dtype=model.dtype)
            new_final.weight.data.copy_(model.model.norm.weight.data); new_final.bias.data.zero_()
            model.model.norm = new_final.to(model.device)

        # 劫持 layer forward 实现 Pre/Post 与 Skip
        norm_pos = params["norm_position"]
        skip_prob = params["layer_skip_prob"]
        for layer in model.model.layers:
            def custom_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
                if self.training and skip_prob > 0.0 and torch.rand(1).item() < skip_prob:
                    return (hidden_states,)
                residual = hidden_states
                if norm_pos == "pre-norm":
                    hidden_states = self.input_layernorm(hidden_states)
                    hidden_states, *rest = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
                    hidden_states = residual + hidden_states
                    residual = hidden_states
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    hidden_states = self.mlp(hidden_states)
                    hidden_states = residual + hidden_states
                elif norm_pos == "post-norm":
                    hidden_states, *rest = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
                    hidden_states = self.input_layernorm(residual + hidden_states)
                    residual = hidden_states
                    hidden_states = self.mlp(hidden_states)
                    hidden_states = self.post_attention_layernorm(residual + hidden_states)
                return (hidden_states,) + tuple(rest) if 'rest' in locals() and rest else (hidden_states,)
            layer.forward = types.MethodType(custom_forward, layer)
            
        return model