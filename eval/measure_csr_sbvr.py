import torch
from lm_eval import evaluator
from transformers import AutoTokenizer, LlamaForCausalLM
from sbvr_utils.utils_llama import sbvr_decompress_on_llama, get_llama

MODEL_PATH = "meta-llama/Llama-3.2-1B"
WEIGHT_PATH = "/home/nxc/sbvr/compressed_weights" 
PATCHED_PATH = "./sbvr_model_for_eval"

def load_and_save_sbvr_model(model_path, weight_path):
    """
    Load the SBVR model and save it to a local path.
    """
    if not model_path:
        raise ValueError("model_path cannot be None")
    
    if not weight_path:
        raise ValueError("weight_path cannot be None")
    
    # Load the SBVR model
    model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_sbvr=True, weight_path=weight_path)
    
    # Save the model and tokenizer to a local path
    model.save_pretrained(PATCHED_PATH)
    tokenizer.save_pretrained(PATCHED_PATH)

def run_csr_eval_for_sbvr_model():
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={PATCHED_PATH},trust_remote_code=True",
        tasks=[
            "arc_easy", "arc_challenge", "boolq", "hellaswag",
            "openbookqa", "piqa", "social_iqa", "winogrande"
        ],
        num_fewshot=0,
        batch_size=1,
        device="cuda"
    )

    for task, result in results["results"].items():
        print(f"Task: {task}")
        print(f"Result: {result}\n")
    
if __name__ == "__main__":
    run_csr_eval_for_sbvr_model()