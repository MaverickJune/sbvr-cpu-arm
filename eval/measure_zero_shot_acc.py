import torch
from lm_eval import evaluator
from transformers import AutoTokenizer, LlamaForCausalLM
from sbvr_utils.utils_llama import sbvr_decompress_on_llama, get_llama
from sbvr_utils.lm_eval_adaptor import LMEvalAdaptor

def get_model_and_enc(model_path, use_sbvr=False, use_llm_int8=False, use_fp8=False, 
                      use_gptq_4=False, use_awq_4=False, gptq_local_model_path:str = None,
                      weight_path=None, sbvr_from_local_path:str = None):
    if not model_path:
        raise ValueError("model_path  cannot be None")
    
    if use_sbvr:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_sbvr=True, weight_path=weight_path, sbvr_from_local_path=sbvr_from_local_path)
    elif use_llm_int8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_llm_int8=True)
    elif use_fp8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_fp8=True)
    elif use_gptq_4:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_gptq_4=True, load_from_local=True, gptq_local_model_path=gptq_local_model_path)
    elif use_awq_4:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_awq_4=True)
    else:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0")
    return model, tokenizer

def measure_zero_shot_reasoning_task(model_path=None,
                                     sbvr_path=None,
                                     tasks=None):
    if None in [model_path, sbvr_path, tasks]:
        raise ValueError("model_path, sbvr_path and tasks cannot be None")
    
    # Load the SBVR model
    model, tokenizer = get_model_and_enc(model_path=model_path, 
                                         use_sbvr=True,
                                         weight_path=model_path,
                                         sbvr_from_local_path=sbvr_path)
    
    # Create an evaluator
    lm_eval_model = LMEvalAdaptor(model_name=model_path, model=model, tokenizer=tokenizer, batch_size=1)
    
    # Evaluate the model on the specified tasks
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        batch_size=1,
        no_cache=True,
        num_fewshot=0,
    )

    print(evaluator.make_table(results))
    
if __name__ == "__main__":
    MODEL_PATH = "meta-llama/Llama-3.1-8B"
    SBVR_PATH = "/home/nxclab/wonjun/bvq/sbvr_models/meta-llama_Llama-3.1-8B_num_sum_4"
    TASKS = ["arc_easy", 
             # "arc_challenge", 
             "boolq", 
             # "hellaswag",
             # "openbookqa", 
             # "piqa", 
             # "sciq", 
             "winogrande"]
    
    measure_zero_shot_reasoning_task(model_path=MODEL_PATH, sbvr_path=SBVR_PATH, tasks=TASKS)

# MODEL_PATH = "meta-llama/Llama-3.2-1B"
# WEIGHT_PATH = "/home/nxc/sbvr/compressed_weights" 
# PATCHED_PATH = "./sbvr_model_for_eval"
# AWQ_PATH = "joshmiller656/Llama3.2-1B-AWQ-INT4"

# def load_and_save_sbvr_model(model_path, weight_path):
#     """
#     Load the SBVR model and save it to a local path.
#     """
#     if not model_path:
#         raise ValueError("model_path cannot be None")
    
#     if not weight_path:
#         raise ValueError("weight_path cannot be None")
    
#     # Load the SBVR model
#     model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_sbvr=True, weight_path=weight_path)
    
#     # Save the model and tokenizer to a local path
#     model.save_pretrained(PATCHED_PATH)
#     tokenizer.save_pretrained(PATCHED_PATH)

# def run_csr_eval_for_sbvr_model():
#     results = evaluator.simple_evaluate(
#         model="hf",
#         model_args=f"pretrained={PATCHED_PATH},trust_remote_code=True",
#         tasks=[
#             "arc_easy", "arc_challenge", "boolq", "hellaswag",
#             "openbookqa", "piqa", "social_iqa", "winogrande"
#         ],
#         num_fewshot=0,
#         batch_size=1,
#         device="cuda"
#     )

#     for task, result in results["results"].items():
#         print(f"Task: {task}")
#         print(f"Result: {result}\n")
        
# def run_eval_original():
#     results = evaluator.simple_evaluate(
#         model="hf",
#         model_args=f"pretrained={MODEL_PATH},trust_remote_code=True",
#         tasks=[
#             "arc_easy", "arc_challenge", "boolq", "hellaswag",
#             "openbookqa", "piqa", "social_iqa", "winogrande"
#         ],
#         num_fewshot=0,
#         batch_size=1,
#         device="cuda"
#     )

#     for task, result in results["results"].items():
#         print(f"Task: {task}")
#         print(f"Result: {result}\n")
        
# def run_eval_awq():
#     results = evaluator.simple_evaluate(
#         model="hf",
#         model_args=f"pretrained={AWQ_PATH},trust_remote_code=True",
#         tasks=[
#             "arc_easy", "arc_challenge", "boolq", "hellaswag",
#             "openbookqa", "piqa", "social_iqa", "winogrande"
#         ],
#         num_fewshot=0,
#         batch_size=1,
#         device="cuda"
#     )

#     for task, result in results["results"].items():
#         print(f"Task: {task}")
#         print(f"Result: {result}\n")
    
# if __name__ == "__main__":
#     # run_csr_eval_for_sbvr_model()
#     run_eval_original()
#     # run_eval_awq()