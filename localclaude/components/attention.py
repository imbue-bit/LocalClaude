import math
import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward
from .base import BaseMutator, register_mutator

def get_alibi_slopes(heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [math.pow(base, i) for i in range(1, closest_power_of_2 + 1)]
    if closest_power_of_2 < heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        slopes += [math.pow(extra_base, i) for i in range(1, 2 * (heads - closest_power_of_2) + 1, 2)]
    return torch.tensor(slopes, dtype=torch.float32)

@register_mutator("attention")
class AttentionMutator(BaseMutator):
    def build_search_space(self, trial):
        return {
            "kv_heads": trial.suggest_categorical("kv_heads", [1, 2, 4, 8]),
            "qk_norm": trial.suggest_categorical("qk_norm", [True, False]),
            "alibi_mix": trial.suggest_categorical("alibi_mix", [True, False])
        }

    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict) -> PreTrainedModel:
        # 1. 映射 KV Heads (GQA/MQA)
        orig_kv = config.num_key_value_heads
        target_kv = params["kv_heads"]
        can_remap_kv = target_kv <= config.num_attention_heads and (config.num_attention_heads % target_kv) == 0
        if can_remap_kv and orig_kv != target_kv:
            head_dim = config.hidden_size // config.num_attention_heads
            for layer in model.model.layers:
                proj_device = layer.self_attn.k_proj.weight.device
                proj_dtype = layer.self_attn.k_proj.weight.dtype
                k_w = layer.self_attn.k_proj.weight.data.view(orig_kv, head_dim, config.hidden_size)
                v_w = layer.self_attn.v_proj.weight.data.view(orig_kv, head_dim, config.hidden_size)
                if target_kv < orig_kv:
                    f = orig_kv // target_kv
                    k_new = k_w.view(target_kv, f, head_dim, config.hidden_size).mean(dim=1)
                    v_new = v_w.view(target_kv, f, head_dim, config.hidden_size).mean(dim=1)
                else:
                    f = target_kv // orig_kv
                    k_new = k_w.repeat_interleave(f, dim=0)
                    v_new = v_w.repeat_interleave(f, dim=0)
                
                layer.self_attn.k_proj = nn.Linear(
                    config.hidden_size, target_kv * head_dim, bias=False, device=proj_device, dtype=proj_dtype
                )
                layer.self_attn.v_proj = nn.Linear(
                    config.hidden_size, target_kv * head_dim, bias=False, device=proj_device, dtype=proj_dtype
                )
                layer.self_attn.k_proj.weight.data.copy_(k_new.view(target_kv * head_dim, config.hidden_size))
                layer.self_attn.v_proj.weight.data.copy_(v_new.view(target_kv * head_dim, config.hidden_size))
                layer.self_attn.num_key_value_heads = target_kv
                layer.self_attn.num_key_value_groups = config.num_attention_heads // target_kv
            config.num_key_value_heads = target_kv

        # 2. QK-Norm & ALiBi
        has_qk = params["qk_norm"]
        use_alibi = params["alibi_mix"]
        for layer in model.model.layers:
            attn = layer.self_attn
            if has_qk:
                h_dim = config.hidden_size // config.num_attention_heads
                proj_device = attn.q_proj.weight.device
                proj_dtype = attn.q_proj.weight.dtype
                attn.q_norm = nn.LayerNorm(h_dim, eps=1e-6, elementwise_affine=True, device=proj_device, dtype=proj_dtype)
                attn.k_norm = nn.LayerNorm(h_dim, eps=1e-6, elementwise_affine=True, device=proj_device, dtype=proj_dtype)
                nn.init.ones_(attn.q_norm.weight); nn.init.zeros_(attn.q_norm.bias)
                nn.init.ones_(attn.k_norm.weight); nn.init.zeros_(attn.k_norm.bias)
            
            orig_fwd = attn.forward
            def new_attn_forward(
                self,
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                if position_embeddings is None:
                    return orig_fwd(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                if has_qk:
                    query_states = self.q_norm(query_states)
                    key_states = self.k_norm(key_states)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                if use_alibi and attention_mask is not None and attention_mask.dim() == 4:
                    bsz, _, q_len, k_len = attention_mask.shape
                    slopes = get_alibi_slopes(self.config.num_attention_heads).to(attention_mask.device)
                    slopes = slopes.to(dtype=attention_mask.dtype)
                    positions = torch.arange(k_len, device=attention_mask.device, dtype=slopes.dtype)
                    alibi = slopes[:, None] * positions[None, :]
                    attention_mask = attention_mask + alibi[None, :, None, :]

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
            attn.forward = types.MethodType(new_attn_forward, attn)
            
        return model
