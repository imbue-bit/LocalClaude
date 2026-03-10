import os
import gc
import json
import logging
import torch
import optuna
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    set_seed
)
from trl import DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from config import NASConfig
from async_judge import AsyncStyleJudge
from arch_modifier import apply_all_mutations
from custom_trainer import ClaudeNAS_SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CFG = NASConfig()
JUDGE = AsyncStyleJudge(model_name=CFG.judge_model)

def format_train_prompts(example):
    return [
        f"<|start_header_id|>user<|end_header_id|>\n\n{instr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
        for instr, out in zip(example['instruction'], example['output'])
    ]

def evaluate_model_style(model, tokenizer, num_samples: int):
    model.eval()
    eval_records = []
    
    with open(CFG.eval_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples: break
            eval_records.append(json.loads(line.strip()))
            
    results_to_judge = []
    logger.info(f"Generating Zero-Shot outputs for {len(eval_records)} samples...")
    
    with torch.no_grad():
        for item in eval_records:
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{item['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results_to_judge.append({
                "prompt": item["prompt"],
                "gold": item["claude_gold_response"],
                "student": generated.strip()
            })
            
    judge_results = JUDGE.evaluate_batch(results_to_judge)
    
    valid_scores = []
    table = wandb.Table(columns=["Prompt", "Claude Gold", "Student Gen", "Refusal", "Format", "Tone", "Total", "Reasoning"])
    
    for i, res in enumerate(judge_results):
        if isinstance(res, Exception):
            logger.error(f"Judge failed on sample {i}: {res}")
            continue
            
        valid_scores.append(res.total_score)
        orig = results_to_judge[i]
        table.add_data(
            orig["prompt"], orig["gold"], orig["student"],
            res.refusal_score, res.formatting_score, res.tone_score, res.total_score, res.reasoning
        )
        
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    return avg_score, table

def objective(trial: optuna.Trial) -> float:
    # 绝对完整的超参搜索空间
    params = {
        "kv_heads": trial.suggest_categorical("kv_heads", [1, 2, 4, 8]),
        "qk_norm": trial.suggest_categorical("qk_norm", [True, False]),
        
        "alibi_mix": trial.suggest_categorical("alibi_mix", [True, False]),
        "rope_base": trial.suggest_categorical("rope_base", [10000, 100000, 500000]),
        "rope_scaling": trial.suggest_categorical("rope_scaling", ["None", "NTK", "YaRN", "PI"]),
        
        "ffn_activation": trial.suggest_categorical("ffn_activation", ["silu", "gelu", "quickgelu", "mish", "squared_relu"]),
        "gating_bias": trial.suggest_categorical("gating_bias", ["none", "learnable"]),
        
        "norm_position": trial.suggest_categorical("norm_position", ["pre-norm", "post-norm"]),
        "norm_type": trial.suggest_categorical("norm_type", ["rms", "layer_norm"]),
        "layer_skip_prob": trial.suggest_categorical("layer_skip_prob", [0.0, 0.05, 0.1]),
        
        "tokenizer_strategy": trial.suggest_categorical("tokenizer_strategy", ["BPE_Base", "Unigram_Simulated", "WordPiece_Simulated"]),
        
        "is_moe": trial.suggest_categorical("is_moe", [False, True])
    }
    
    if params["is_moe"]:
        params["moe_experts"] = trial.suggest_categorical("moe_experts", [8, 16, 32])
        params["moe_top_k"] = trial.suggest_categorical("moe_top_k", [2, 4, 6])
        params["moe_capacity"] = trial.suggest_categorical("moe_capacity", [1.0, 1.25, 1.5])
        params["moe_shared_expert"] = trial.suggest_categorical("moe_shared_expert", [True, False])
        params["moe_z_loss"] = trial.suggest_categorical("moe_z_loss", [0.0, 0.001, 0.01])
    else:
        params.update({"moe_experts": 1, "moe_top_k": 1, "moe_capacity": 1.0, "moe_shared_expert": False, "moe_z_loss": 0.0})

    logger.info(f"\n{'='*60}\n🚀 STARTING TRIAL {trial.number}\nPARAMS: {params}\n{'='*60}")
    
    run = wandb.init(
        project=CFG.project_name, 
        name=f"trial_{trial.number}",
        config=params,
        reinit=True
    )
    
    set_seed(42)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG.base_model_id)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
            
        logger.info("Loading base model in BF16 with FlashAttention-2...")
        base_model = AutoModelForCausalLM.from_pretrained(
            CFG.base_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        logger.info("Applying architecture mutations...")
        base_model, tokenizer = apply_all_mutations(base_model, tokenizer, params)
        base_model.gradient_checkpointing_enable()
        
        peft_config = LoraConfig(
            r=CFG.lora_r,
            lora_alpha=CFG.lora_alpha,
            target_modules=CFG.target_modules,
            lora_dropout=CFG.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, peft_config)
        
        dataset = load_dataset('json', data_files={'train': CFG.null_data_path})['train']
        
        training_args = TrainingArguments(
            output_dir=f"/tmp/nas_checkpoints/trial_{trial.number}",
            max_steps=CFG.max_steps,
            per_device_train_batch_size=CFG.batch_size,
            gradient_accumulation_steps=CFG.gradient_accumulation_steps,
            learning_rate=CFG.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=CFG.warmup_ratio,
            weight_decay=CFG.weight_decay,
            max_grad_norm=1.0,
            logging_steps=10,
            bf16=True,
            report_to="wandb",
            save_strategy="no", 
        )
        
        trainer = ClaudeNAS_SFTTrainer(
            model=model,
            train_dataset=dataset,
            formatting_func=format_train_prompts,
            data_collator=collator,
            max_seq_length=CFG.max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
        )
        
        logger.info("Starting Null-Semantics Subconscious Distillation...")
        train_result = trainer.train()
        
        if torch.isnan(torch.tensor(train_result.metrics["train_loss"])):
            raise ValueError("Training Diverged (NaN Loss). Invalid Architecture Mutation.")
            
        logger.info("Evaluating style activation speed...")
        final_score, gen_table = evaluate_model_style(model, tokenizer, num_samples=CFG.eval_samples_per_trial)
        
        wandb.log({
            "final_style_score": final_score,
            "generations_table": gen_table
        })
        
        return final_score
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM! Skipping trial...")
        run.summary["error"] = "OOM"
        return 0.0
    except Exception as e:
        logger.error(f"Trial Failed: {str(e)}")
        run.summary["error"] = str(e)
        return 0.0
    finally:
        if 'model' in locals(): del model
        if 'base_model' in locals(): del base_model
        if 'trainer' in locals(): del trainer
        gc.collect()
        torch.cuda.empty_cache()
        run.finish()

def run_nas_pipeline():
    logger.info("Initializing Optuna SQLite Storage for resumable distributed search...")
    os.makedirs("/tmp/nas_checkpoints", exist_ok=True)
    
    study = optuna.create_study(
        direction="maximize",
        study_name=CFG.study_name,
        storage=CFG.storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    )
    
    try:
        study.optimize(objective, n_trials=CFG.n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        logger.warning("Search manually interrupted. State saved to SQLite.")

    if len(study.trials) > 0:
        best_trial = study.best_trial
        result_json = {
            "best_style_alignment_score": best_trial.value,
            "most_likely_architecture": best_trial.params
        }
        
        with open("CLAUDE_DNA_REVEALED.json", "w") as f:
            json.dump(result_json, f, indent=4)
            
        logger.info("\n" + "="*60)
        logger.info(json.dumps(result_json, indent=4, ensure_ascii=False))
        logger.info("="*60)

if __name__ == "__main__":
    run_nas_pipeline()