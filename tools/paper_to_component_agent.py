import os
import re
import argparse
from openai import OpenAI

def run_agent(paper_text: str, component_name: str, output_path: str):
    client_kwargs = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    
    system_prompt = """
    You are an expert PyTorch neural architecture engineer working on a NAS framework called LocalClaude.
    Your task is to read a research paper description (or abstract) and implement its proposed architecture trick as a Python component.
    
    Requirements:
    1. MUST subclass `BaseMutator` from `.base` and use `@register_mutator("name")`.
    2. MUST implement `build_search_space(self, trial)` returning a dict of optuna suggestions.
    3. MUST implement `mutate(self, model, config, params)` modifying the HuggingFace Llama architecture dynamically.
    4. MUST preserve pre-trained weights. If adding new parameters, zero-initialize them so the initial forward pass is mathematically identical to the base model.
    5. Output ONLY the raw valid Python code wrapped in ```python ... ```.
    
    Example Structure:
    ```python
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel, LlamaConfig
    from .base import BaseMutator, register_mutator
    
    @register_mutator("example_trick")
    class ExampleMutator(BaseMutator):
        def build_search_space(self, trial):
            return {"use_trick": trial.suggest_categorical("use_trick", [True, False])}
            
        def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: dict):
            if params["use_trick"]:
                # perform safe PyTorch dynamic graph patching here
                pass
            return model
    ```
    """

    print(f"Agent is reading the paper and generating component '{component_name}'...")
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a component named '{component_name}' based on this paper/trick description:\n\n{paper_text}"}
        ],
        temperature=1.0
    )

    content = response.choices[0].message.content
    
    match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
    if not match:
        print("Failed to parse valid Python code from the agent. Raw output:")
        print(content)
        return
        
    code = match.group(1)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    print(f"Successfully wrote new architecture component to {output_path}")
    print("Next time you run NAS, this component will automatically be injected into the search space!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_text", type=str, required=True, help="Abstract or text describing the architecture trick.")
    parser.add_argument("--component_name", type=str, required=True, help="Internal name for the registry (e.g., 'swiglu').")
    args = parser.parse_args()
    
    out_file = os.path.join(os.path.dirname(__file__), "..", "localclaude", "components", f"{args.component_name}.py")
    run_agent(args.paper_text, args.component_name, out_file)
