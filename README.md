# LocalClaude

*Probing the Inductive Bias Closest to Claude via Semantics-Free Distillation Trajectories*

LocalClaude is an open research effort that uses black-box architecture search to identify Transformer configurations whose inductive bias most readily manifests behavioral traits observed in Claude — even when trained exclusively on semantics-free, non-informative generations from the target model.

### Core Scientific Question

A significant portion of a frontier model's apparent "alignment" or "personality" (refusal patterns, hedging language, verbose reasoning structures, markdown formatting bias, polite-yet-restrictive tone, etc.) may be latent in its architectural inductive bias, rather than solely encoded in weights via supervised or preference data.

We treat the speed of emergence of these behavioral signatures during training on semantically unrelated data as a diagnostic probe for architectural similarity to Claude's unknown design.

### Key Concept: Implicit / Subconscious Learning in LLMs

Recent work has shown that large language models can transmit behavioral traits through data that appears completely unrelated to those traits — a phenomenon sometimes referred to as subliminal learning or non-semantic trait transmission (see e.g. Anthropic's 2025 study on subliminal learning, where preference signals leak through semantically neutral sequences such as random numbers).

In this project we extend the idea in the architecture dimension:

- When a student model is trained (via LoRA) solely on Claude-generated sequences that carry no overt semantic content (random JSON structures, hex dumps, meaningless pseudo-code continuations, high-entropy noise, etc.),
- certain architectural choices cause Claude-like behavioral patterns to emerge much faster during training,
- even though the training signal contains no explicit examples of refusal, hedging, or structured reasoning.

This rapid implicit acquisition is interpreted as evidence that the architecture provides a stronger latent prior toward the same hidden alignment / style objectives that Claude's designers embedded through their (unknown) training recipe.

In other words: we are searching for open architectures that most naturally "resonate" with Claude's implicit learning dynamics — the subconscious structural priors that allow certain behaviors to surface easily from weak or indirect supervision.

### Methodology Overview

1. Teacher data collection — Generate large volumes (~10k–100k+) of semantics-free completions from Claude-family models (temperature > 0.8, nonsense / random prompts).
2. Architecture search space — Sample variants of ~8B–14B-scale Transformers, modifying high-impact components:
   - Attention grouping (MQA / GQA variants / MHA)
   - Normalization placement & type (pre-RMSNorm / LayerNorm hybrids / learned per-channel scaling)
   - FFN gating (SwiGLU / GeGLU / ReGLU + bias variants)
   - Rotary embeddings (base frequency, NTK / YaRN / PI scaling)
   - Auxiliary tricks (QK-normalization, layer skips, small auxiliary losses)
3. Training signal — LoRA / QLoRA adaptation on the semantics-free dataset (1–3 epochs).
4. Probe metric — Zero-shot performance on a curated set of Claude-implicit-style tasks (refusal strength, verbosity, format adherence, hedging frequency, etc.), scored via judge model (pairwise or direct) or automatic heuristics.
5. Selection criterion — Architectures that reach high probe agreement with Claude behavior in the fewest training steps are ranked highest.

### Quick Reproduction (Pilot Scale)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare small semantics-free dataset (~5k–10k examples)
python data_prep/sample_claude_noise.py --output_dir data/noise_v1 --n_samples 10000

# 3. Launch a small search sweep
python nas/run_search.py --config configs/pilot_sweep.yaml --n_trials 200 --devices 0,1
```

Results (convergence plots, top configs) appear in `experiments/`.

### Academic Positioning & Related Work

This project sits at the intersection of:

- Architecture search for alignment inductiveness
- Black-box model probing / reverse-engineering of latent priors
- Non-semantic / subliminal trait transmission in distillation pipelines
- Interpretability via training dynamics rather than post-hoc analysis

Heavily inspired by recent studies on subliminal learning, implicit objective discovery, and inductive bias differences across Transformer families.

### Get Involved

If you're interested in black-box inductive bias probing, latent alignment signals, or architecture-level distillation dynamics — stars, issues, and PRs welcome.

Goal: better understand how much of a model's "character" lives in its wiring diagram rather than its weights.
