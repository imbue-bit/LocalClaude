import math
import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaConfig
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
        if orig_kv != target_kv:
            head_dim = config.hidden_size // config.num_attention_heads
            for layer in model.model.layers:
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
                
                layer.self_attn.k_proj = nn.Linear(config.hidden_size, target_kv * head_dim, bias=False, dtype=model.dtype).to(model.device)
                layer.self_attn.v_proj = nn.Linear(config.hidden_size, target_kv * head_dim, bias=False, dtype=model.dtype).to(model.device)
                layer.self_attn.k_proj.weight.data.copy_(k_new.view(target_kv * head_dim, config.hidden_size))
                layer.self_attn.v_proj.weight.data.copy_(v_new.view(target_kv * head_dim, config.hidden_size))
                layer.self_attn.num_key_value_heads = target_kv
            config.num_key_value_heads = target_kv

        # 2. QK-Norm & ALiBi
        has_qk = params["qk_norm"]
        use_alibi = params["alibi_mix"]
        for layer in model.model.layers:
            attn = layer.self_attn
            if has_qk:
                h_dim = config.hidden_size // config.num_attention_heads
                attn.q_norm = nn.LayerNorm(h_dim, eps=1e-6, elementwise_affine=True).to(model.device)
                attn.k_norm = nn.LayerNorm(h_dim, eps=1e-6, elementwise_affine=True).to(model.device)
                nn.init.ones_(attn.q_norm.weight); nn.init.zeros_(attn.q_norm.bias)
                nn.init.ones_(attn.k_norm.weight); nn.init.zeros_(attn.k_norm.bias)
            
            orig_fwd = attn.forward
            def new_attn_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
                q = self.q_proj(hidden_states).view(*hidden_states.shape[:2], self.num_heads, self.head_dim)
                k = self.k_proj(hidden_states).view(*hidden_states.shape[:2], self.num_key_value_heads, self.head_dim)
                if has_qk:
                    q = self.q_norm(q); k = self.k_norm(k)
                if use_alibi and attention_mask is not None:
                    slopes = get_alibi_slopes(self.num_heads).to(hidden_states.device)
                    bias = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0) * slopes.unsqueeze(-1)
                    attention_mask = attention_mask + bias.unsqueeze(0).unsqueeze(2)
                q = q.view(*q.shape[:2], -1); k = k.view(*k.shape[:2], -1)
                return orig_fwd(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
            attn.forward = types.MethodType(new_attn_forward, attn)
            
        return model