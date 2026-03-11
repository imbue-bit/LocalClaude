import torch
import optuna
from trl import SFTTrainer
from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)

class ASHAPruningCallback(TrainerCallback):
    """
    与 Optuna ASHA Pruner 通信的回调函数。
    在 SFT 训练的特定 step 触发 Zero-shot 验证，
    如果当前架构的潜意识学习速度太慢，直接抛出异常中断训练，节省算力。
    """
    def __init__(self, trial: optuna.Trial, evaluator_func):
        self.trial = trial
        self.evaluator_func = evaluator_func

    def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
        logger.info(f"Triggering ASHA intermediate evaluation at step {state.global_step}...")
        
        # 触发潜意识零样本风格评估 (evaluator_func 必须返回 score 和 table)
        current_score, _ = self.evaluator_func(model, tokenizer)
        
        # 汇报给 Optuna
        self.trial.report(current_score, state.global_step)
        
        # 如果当前 Trial 表现处于劣势 (比如后 50%) 触发剪枝
        if self.trial.should_prune():
            logger.warning(f"Trial {self.trial.number} pruned at step {state.global_step} due to low style score ({current_score:.2f}).")
            raise optuna.exceptions.TrialPruned()

class LocalClaudeTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        
        aux_loss = 0.0
        base_model = model.model if hasattr(model, "model") else model
        
        for module in base_model.modules():
            if hasattr(module, "current_aux_loss"):
                aux_tensor = getattr(module, "current_aux_loss")
                if isinstance(aux_tensor, torch.Tensor):
                    aux_loss += aux_tensor
                elif isinstance(aux_tensor, (int, float)):
                    aux_loss += torch.tensor(aux_tensor, device=loss.device, dtype=loss.dtype)
                
        total_loss = loss + aux_loss
        
        return (total_loss, outputs) if return_outputs else total_loss