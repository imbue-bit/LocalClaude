import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class NASConfig:
    project_name: str = "Claude-Arch-DNA-Search"
    base_model_id: str = "meta-llama/Meta-Llama-3-8B"
    null_data_path: str = "null_semantics.jsonl" # 需提前生成
    eval_data_path: str = "style_eval.jsonl"     # 需提前生成
    
    # NAS & Optuna 配置
    study_name: str = "claude_dna_v1"
    storage_url: str = "sqlite:///nas_journal.db" # 支持断点续传
    n_trials: int = 500
    
    # 训练超参数
    max_steps: int = 300
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # 微调所有线性层以最大化对架构改变的适应性
    target_modules: List[str] = field(default_factory=lambda: ["all-linear"])
    
    # Judge 配置
    judge_model: str = "gpt-5"
    eval_samples_per_trial: int = 25 # 每次验证采样的数量