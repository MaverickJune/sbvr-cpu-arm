from sbvr_utils.utils import eval_ppl
from sbvr_utils.utils_llama import get_llama


def measure_llama_ppl(model_path, use_sbvr=False, use_llm_int8=False, use_fp8=False):
    if not model_path:
        raise ValueError("model_path cannot be None")
    
    if use_sbvr:
        raise NotImplementedError("Not yet implemented")
    elif use_llm_int8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_llm_int8=True)
    elif use_fp8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_fp8=True)
    else:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0")
    eval_ppl(model=model, tokenizer=tokenizer, dataset="wikitext-2")
    
    
if __name__ == "__main__":
    MODEL_PATH = "meta-llama/Llama-3.2-3B"
    
    # measure_llama_ppl(model_path=MODEL_PATH)
    # measure_llama_ppl(model_path=MODEL_PATH, use_llm_int8=True)
    measure_llama_ppl(model_path=MODEL_PATH, use_fp8=True)
    