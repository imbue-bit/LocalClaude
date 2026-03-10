import math
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.activations import ACT2FN
import logging

logger = logging.getLogger(__name__)

ACT2FN["squared_relu"] = lambda x: torch.pow(torch.nn.functional.relu(x), 2)
ACT2FN["mish"] = nn.functional.mish
ACT2FN["quickgelu"] = lambda x: x * torch.sigmoid(1.702 * x)

class DynamicMoEBlock(nn.Module):
    def __init__(self, orig_mlp, config, params):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = params["moe_experts"]
        self.top_k = params["moe_top_k"]
        self.capacity_factor = params["moe_capacity"]
        self.z_loss_coeff = params["moe_z_loss"]
        
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        nn.init.zeros_(self.gate.weight)
        
        self.experts = nn.ModuleList([copy.deepcopy(orig_mlp) for _ in range(self.num_experts)])
        
        self.has_shared = params["moe_shared_expert"]
        if self.has_shared:
            self.shared_expert = copy.deepcopy(orig_mlp)
            
        self.current_aux_loss = 0.0

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2) * self.z_loss_coeff
        self.current_aux_loss = z_loss
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            expert_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[expert_indices]
            
            weight_mask = (selected_experts[expert_indices] == i)
            expert_weights = routing_weights[expert_indices][weight_mask]
            
            expert_out = self.experts[i](expert_input)
            final_output[expert_indices] += expert_out * expert_weights.unsqueeze(-1)
            
        if self.has_shared:
            final_output += self.shared_expert(x_flat)
            
        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        return final_output

def get_alibi_slopes(heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [math.pow(base, i) for i in range(1, closest_power_of_2 + 1)]
    if closest_power_of_2 < heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        slopes += [math.pow(extra_base, i) for i in range(1, 2 * (heads - closest_power_of_2) + 1, 2)]
    return torch.tensor(slopes, dtype=torch.float32)

def patch_attention_qknorm_and_alibi(model: PreTrainedModel, params: dict):
    has_qknorm = params.get("qk_norm", False)
    use_alibi = params.get("alibi_mix", False)
    
    for layer in model.model.layers:
        attn = layer.self_attn
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        if has_qknorm:
            attn.q_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True).to(model.device)
            attn.k_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True).to(model.device)
            nn.init.ones_(attn.q_norm.weight)
            nn.init.zeros_(attn.q_norm.bias)
            nn.init.ones_(attn.k_norm.weight)
            nn.init.zeros_(attn.k_norm.bias)
            
        original_forward = attn.forward
        def new_attn_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            query_states = query_states.view(*query_states.shape[:2], self.num_heads, self.head_dim)
            key_states = key_states.view(*key_states.shape[:2], self.num_key_value_heads, self.head_dim)
            
            if has_qknorm:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
                
            if use_alibi and attention_mask is not None:
                seq_len = hidden_states.shape[1]
                alibi_slopes = get_alibi_slopes(self.num_heads).to(hidden_states.device)
                alibi_bias = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0) * alibi_slopes.unsqueeze(-1)
                attention_mask = attention_mask + alibi_bias.unsqueeze(0).unsqueeze(2)
                
            query_states = query_states.view(*query_states.shape[:2], -1)
            key_states = key_states.view(*key_states.shape[:2], -1)
            
            return original_forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
            
        attn.forward = types.MethodType(new_attn_forward, attn)
    return model

def rewrite_layer_forward(layer: LlamaDecoderLayer, norm_position: str, skip_prob: float):
    original_forward = layer.forward

    def custom_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        if self.training and skip_prob > 0.0:
            if torch.rand(1).item() < skip_prob:
                return (hidden_states,)
                
        residual = hidden_states
        
        if norm_position == "pre-norm":
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, *rest = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
        elif norm_position == "post-norm":
            hidden_states, *rest = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
            hidden_states = self.input_layernorm(residual + hidden_states)
            
            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_attention_layernorm(residual + hidden_states)
            
        return (hidden_states,) + tuple(rest) if 'rest' in locals() and rest else (hidden_states,)

    layer.forward = types.MethodType(custom_forward, layer)
    return layer

def patch_gating_bias(model: PreTrainedModel):
    for layer in model.model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]:
            proj = getattr(layer, "self_attn" if "proj" in ["q", "k", "v"] else "mlp")
            linear = getattr(proj, proj_name, None)
            if linear and linear.bias is None:
                new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True, dtype=model.dtype)
                new_linear.weight.data.copy_(linear.weight.data)
                nn.init.zeros_(new_linear.bias)
                setattr(proj, proj_name, new_linear.to(model.device))
    return model

def patch_tokenizer_behavior(tokenizer, variant: str):
    if variant == "Unigram_Simulated":
        tokenizer.byte_fallback = False
        tokenizer.unk_token = "<unk>"
    elif variant == "WordPiece_Simulated":
        tokenizer.add_prefix_space = False
        tokenizer.split_special_tokens = True
    return tokenizer

def patch_attention_heads(model: PreTrainedModel, config: LlamaConfig, target_kv_heads: int):
    orig_kv_heads = model.config.num_key_value_heads
    if orig_kv_heads == target_kv_heads:
        return model
        
    head_dim = config.hidden_size // config.num_attention_heads
    for layer in model.model.layers:
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        
        k_weight = k_proj.weight.data.view(orig_kv_heads, head_dim, config.hidden_size)
        v_weight = v_proj.weight.data.view(orig_kv_heads, head_dim, config.hidden_size)
        
        if target_kv_heads < orig_kv_heads:
            group_factor = orig_kv_heads // target_kv_heads
            k_weight_new = k_weight.view(target_kv_heads, group_factor, head_dim, config.hidden_size).mean(dim=1)
            v_weight_new = v_weight.view(target_kv_heads, group_factor, head_dim, config.hidden_size).mean(dim=1)
        else:
            repeat_factor = target_kv_heads // orig_kv_heads
            k_weight_new = k_weight.repeat_interleave(repeat_factor, dim=0)
            v_weight_new = v_weight.repeat_interleave(repeat_factor, dim=0)
            
        new_k_proj = nn.Linear(config.hidden_size, target_kv_heads * head_dim, bias=False, dtype=model.dtype)
        new_v_proj = nn.Linear(config.hidden_size, target_kv_heads * head_dim, bias=False, dtype=model.dtype)
        
        new_k_proj.weight.data.copy_(k_weight_new.view(target_kv_heads * head_dim, config.hidden_size))
        new_v_proj.weight.data.copy_(v_weight_new.view(target_kv_heads * head_dim, config.hidden_size))
        
        layer.self_attn.k_proj = new_k_proj.to(model.device)
        layer.self_attn.v_proj = new_v_proj.to(model.device)
        layer.self_attn.num_key_value_heads = target_kv_heads
        
    config.num_key_value_heads = target_kv_heads
    return model

def patch_layer_norm(model: PreTrainedModel, norm_type: str):
    if norm_type == "rms":
        return model
        
    for layer in model.model.layers:
        hidden_size = layer.input_layernorm.weight.shape[0]
        new_input_norm = nn.LayerNorm(hidden_size, eps=layer.input_layernorm.variance_epsilon, dtype=model.dtype)
        new_input_norm.weight.data.copy_(layer.input_layernorm.weight.data)
        new_input_norm.bias.data.zero_()
        layer.input_layernorm = new_input_norm.to(model.device)
        
        new_post_norm = nn.LayerNorm(hidden_size, eps=layer.post_attention_layernorm.variance_epsilon, dtype=model.dtype)
        new_post_norm.weight.data.copy_(layer.post_attention_layernorm.weight.data)
        new_post_norm.bias.data.zero_()
        layer.post_attention_layernorm = new_post_norm.to(model.device)
        
    old_final_norm = model.model.norm
    new_final_norm = nn.LayerNorm(old_final_norm.weight.shape[0], eps=old_final_norm.variance_epsilon, dtype=model.dtype)
    new_final_norm.weight.data.copy_(old_final_norm.weight.data)
    new_final_norm.bias.data.zero_()
    model.model.norm = new_final_norm.to(model.device)
    
    return model

def apply_all_mutations(model: PreTrainedModel, tokenizer, params: dict) -> tuple:
    config = model.config
    logger.info(f"Applying mutations: {params}")
    
    tokenizer = patch_tokenizer_behavior(tokenizer, params.get("tokenizer_strategy", "BPE_Base"))
    config.hidden_act = params["ffn_activation"]
    model = patch_attention_heads(model, config, target_kv_heads=params["kv_heads"])
    model = patch_layer_norm(model, norm_type=params["norm_type"])
    
    if params.get("gating_bias", "none") == "learnable":
        model = patch_gating_bias(model)
        
    model = patch_attention_qknorm_and_alibi(model, params)
    
    norm_pos = params.get("norm_position", "pre-norm")
    skip_p = params.get("layer_skip_prob", 0.0)
    for layer in model.model.layers:
        rewrite_layer_forward(layer, norm_pos, skip_p)
        
    if params.get("is_moe", False):
        for layer in model.model.layers:
            moe_block = DynamicMoEBlock(layer.mlp, config, params).to(model.device)
            layer.mlp = moe_block
            
    config.rope_theta = float(params.get("rope_base", 10000))
    rope_scaling_type = params.get("rope_scaling", "None")
    
    if rope_scaling_type == "NTK":
        config.rope_scaling = {"type": "dynamic", "factor": 2.0}
    elif rope_scaling_type == "YaRN":
        config.rope_scaling = {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192}
    elif rope_scaling_type == "PI":
        config.rope_scaling = {"type": "linear", "factor": 2.0}
    else:
        config.rope_scaling = None
        
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "rotary_emb") and hasattr(layer.self_attn.rotary_emb, "inv_freq"):
            del layer.self_attn.rotary_emb.inv_freq
            
    return model, tokenizer