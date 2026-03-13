import os
import gc
import json
import sys
import traceback
import torch
import hydra
import optuna
import wandb
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from localclaude.components import MUTATOR_REGISTRY
from localclaude.arch_mutator import apply_architecture_search_space
from localclaude.evaluator import AsyncSubliminalJudge
from localclaude.custom_trainer import LocalClaudeTrainer, ASHAPruningCallback

def build_sampler(seed: int) -> optuna.samplers.BaseSampler:
    if hasattr(optuna.samplers, "MOTPESampler"):
        return optuna.samplers.MOTPESampler(seed=seed)
    if hasattr(optuna.samplers, "NSGAIISampler"):
        return optuna.samplers.NSGAIISampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)

def create_objective(cfg: DictConfig):
    judge = None

    def evaluate_model(model, tokenizer):
        nonlocal judge
        if cfg.nas.eval_samples_per_trial <= 0:
            return 0.0, None

        if judge is None:
            judge = AsyncSubliminalJudge(hidden_rules=cfg.probe.target_system_prompt, model_name=cfg.nas.judge_model)

        model.eval()
        test_prompts = []
        
        with open(cfg.data.eval_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= cfg.nas.eval_samples_per_trial: break
                test_prompts.append(json.loads(line.strip())["prompt"])
                
        if not test_prompts:
            model.train()
            return 0.0, None

        results_to_judge = []
        with torch.no_grad():
            for prompt in test_prompts:
                prompt_fmt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                inputs = tokenizer(prompt_fmt, return_tensors="pt", add_special_tokens=False).to(model.device)
                
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.6, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                results_to_judge.append({"prompt": prompt, "student": gen_text.strip()})
                
        judge_res = judge.evaluate(results_to_judge)
        
        valid_scores = [r.rule_following_score for r in judge_res if not isinstance(r, Exception)]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        table = None
        if wandb.run is not None:
            table = wandb.Table(columns=["Test Prompt", "Student Output", "Subliminal Rule Score", "Judge Reasoning"])
            for i, r in enumerate(judge_res):
                if not isinstance(r, Exception):
                    table.add_data(results_to_judge[i]["prompt"], results_to_judge[i]["student"], r.rule_following_score, r.reasoning)
                
        model.train()
        return avg_score, table

    def objective(trial: optuna.Trial):
        params = {}
        for name, mutator_cls in MUTATOR_REGISTRY.items():
            mutator = mutator_cls()
            params.update(mutator.build_search_space(trial))

        run = wandb.init(project=cfg.project_name, name=f"trial_{trial.number}", config=params, reinit=True)
        set_seed(cfg.seed)

        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_id)
            tokenizer.pad_token = tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model_id, torch_dtype=getattr(torch, cfg.model.torch_dtype),
                attn_implementation=cfg.model.attn_implementation, device_map="auto"
            )
            
            base_model, tokenizer, total_params = apply_architecture_search_space(base_model, tokenizer, params)
            base_model.gradient_checkpointing_enable()
            
            peft_cfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM")
            model = get_peft_model(base_model, peft_cfg)
            
            dataset = load_dataset('json', data_files={'train': cfg.data.train_path})['train']
            dataset = dataset.map(
                lambda x: {
                    "prompt": f"<|start_header_id|>user<|end_header_id|>\n\n{x['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "completion": f"{x['output']}<|eot_id|>",
                },
                remove_columns=dataset.column_names,
            )
            t_args = SFTConfig(
                output_dir=f"/tmp/nas_ckpt/t_{trial.number}", max_steps=cfg.training.max_steps,
                per_device_train_batch_size=cfg.training.batch_size, gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                learning_rate=cfg.training.learning_rate, eval_strategy="steps", eval_steps=cfg.training.eval_steps,
                logging_steps=cfg.training.logging_steps, bf16=True, report_to="wandb", save_strategy="no",
                max_length=cfg.data.max_seq_length, completion_only_loss=True
            )
            
            trainer = LocalClaudeTrainer(
                model=model, train_dataset=dataset, eval_dataset=dataset.select(range(2)),
                processing_class=tokenizer, args=t_args,
                callbacks=[ASHAPruningCallback(trial, evaluate_model)]
            )
            
            trainer.train()
            final_score, gen_table = evaluate_model(model, tokenizer)
            
            payload = {"subliminal_rule_score": final_score, "total_params_billions": total_params / 1e9}
            if gen_table is not None:
                payload["generations_table"] = gen_table
            wandb.log(payload)
            return final_score, total_params
            
        except optuna.exceptions.TrialPruned:
            wandb.run.summary["status"] = "PRUNED"
            raise
        except Exception as e:
            traceback.print_exc()
            wandb.run.summary["error"] = str(e)
            return 0.0, float('inf')
        finally:
            if 'model' in locals(): del model
            if 'base_model' in locals(): del base_model
            gc.collect(); torch.cuda.empty_cache(); run.finish()
    return objective

@hydra.main(version_base=None, config_path="../configs", config_name="nas_config")
def main(cfg: DictConfig):
    os.makedirs("/tmp/nas_ckpt", exist_ok=True)
    study = optuna.create_study(
        directions=["maximize", "minimize"], study_name=cfg.nas.study_name, storage=cfg.nas.storage_url,
        load_if_exists=True, sampler=build_sampler(cfg.seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=cfg.training.eval_steps, max_resource=cfg.training.max_steps, reduction_factor=3)
    )
    study.optimize(create_objective(cfg), n_trials=cfg.nas.n_trials, gc_after_trial=True)
    
    print("\n###### result")
    for i, best in enumerate(study.best_trials): 
        print(f"Option {i+1}: Rule Adherence Score={best.values[0]:.2f}, Params={best.values[1]/1e9:.2f}B | {best.params}")

if __name__ == "__main__":
    main()
