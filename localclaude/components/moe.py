import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, LlamaConfig
from .base import BaseMutator, register_mutator

class DynamicMoEBlock(nn.Module):
    def __init__(self, orig_mlp, config, params):
        super().__init__()
        self.num_experts = params["moe_experts"]
        self.top_k = params["moe_top_k"]
        self.z_loss_coeff = params["moe_z_loss"]
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        nn.init.zeros_(self.gate.weight)
        self.experts = nn.ModuleList([copy.deepcopy(orig_mlp) for _ in range(self.num_experts)])
        self.has_shared = params["moe_shared_expert"]
        if self.has_shared: self.shared_expert = copy.deepcopy(orig_mlp)
        self.current_aux_loss = 0.0

    def forward(self, x):
        bsz, seq_len, h_dim = x.shape
        x_flat = x.view(-1, h_dim)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        self.current_aux_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2) * self.z_loss_coeff
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        final_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (selected == i).any(dim=-1)
            if not mask.any(): continue
            idx = mask.nonzero(as_tuple=True)[0]
            w_mask = (selected[idx] == i)
            out = self.experts[i](x_flat[idx])
            final_out[idx] += out * routing_weights[idx][w_mask].unsqueeze(-1)
        if self.has_shared: final_out += self.shared_expert(x_flat)
        return final_out.view(bsz, seq_len, h_dim)

@register_mutator("moe")
class MoEMutator(BaseMutator):
    def build_search_space(self, trial):
        is_moe = trial.suggest_categorical("is_moe", [False, True])
        params = {"is_moe": is_moe}
        if is_moe:
            params.update({
                "moe_experts": trial.suggest_categorical("moe_experts", [8, 16, 32]),
                "moe_top_k": trial.suggest_categorical("moe_top_k", [2, 4, 6]),
                "moe_z_loss": trial.suggest_categorical("moe_z_loss", [0.0, 0.001, 0.01]),
                "moe_shared_expert": trial.suggest_categorical("moe_shared_expert", [True, False])
            })
        else:
            params.update({"moe_experts": 1, "moe_top_k": 1, "moe_z_loss": 0.0, "moe_shared_expert": False})
        return params

    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict) -> PreTrainedModel:
        if params["is_moe"]:
            for layer in model.model.layers:
                layer.mlp = DynamicMoEBlock(layer.mlp, config, params).to(model.device)
        return model