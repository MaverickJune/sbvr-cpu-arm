from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, FineGrainedFP8Config
from models.sbvr_llama import SBVRLlamaForCausalLM
import torch
import sbvr
import os

from sbvr_utils.log_config import get_logger
logger = get_logger(__name__)


@torch.inference_mode
def decompress_sbvr_llama(weight_path=None, model=None):
    pass


@torch.no_grad()
def get_llama(model_path="meta-llama/Llama-3.2-3B-Instruct", tokenizer_path="meta-llama/Llama-3.2-3B-Instruct", 
              device_map:str ="auto", use_sbvr:bool = False, use_llm_int8:bool = False, use_fp8:bool = False,
              weight_path:str = None):
    r'''
    Fetch llama model from huggingfaces

    @param model_path: target model to fetch from huggingface
    @param tokenizer_path: In case you want to use different tokenizer
    '''
    if not tokenizer_path:
        tokenizer_path = model_path
        
    if use_sbvr:
        if weight_path is None:
            raise ValueError("weight_path cannot be None when use_sbvr is True")
        logger.info("Using SBVR Llama model")
        model = SBVRLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        sbvr_decompress_on_llama(model, weight_path)
    elif use_llm_int8:
        logger.info("Using Llama model with LLM.int8")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=quantization_config
        )
    elif use_fp8: # Only works in hopper GPU (compute capability 9.0 and above)
        logger.info("Using Llama model with FP8")
        fp8_config = FineGrainedFP8Config()
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=fp8_config
        )
    else: 
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    return model, tokenizer


@torch.inference_mode()
def sbvr_decompress_on_llama(model, weight_path:str=None):
    if weight_path is None:
        raise ValueError("weight_path cannot be None")
    
    attn_weights_name = ["q", "k", "v"]
    ffn_weights_name = ["gate_proj", "down_proj", "up_proj"]
 
    for i, layer in enumerate(model.model.layers):
        layer_path = os.path.join(weight_path, f"layer_{i}_")
        for weight_name in attn_weights_name:
            device = layer.self_attn.__getattr__(weight_name + "_proj").weight.device
            weight_path = os.path.join(layer_path, f"{weight_name}.pt")
            sbvr_weight = sbvr.load_sbvr(weight_path)
            layer.self_attn.__getattr__(weight_name + "_proj").weight = sbvr_weight.decode().to(device)
            logger.info(f"Decompressed {weight_name} weight from {weight_path}")
        for weight_name in ffn_weights_name:
            device = layer.mlp.__getattr__(weight_name).weight.device
            weight_path = os.path.join(layer_path, f"{weight_name}.pt")
            sbvr_weight = sbvr.load_sbvr(weight_path)
            layer.mlp.__getattr__(weight_name).weight = sbvr_weight.decode().to(device)
            logger.info(f"Decompressed {weight_name} weight from {weight_path}")
            
    logger.info("Decompression complete")

@torch.no_grad()
def get_layer_ffn_weight(model, layer_idx):
    r"""
    Get the ffn(gate_proj) weight of the decoder layer[layer_idx] from the llama model
    """
    ffn_weight = model.model.layers[layer_idx].mlp.gate_proj.weight
    ffn_weight = ffn_weight.detach().clone()

    return ffn_weight


def format_llama3(input:str = None, tokenizer = None):
    r'''
    Format input into the right llama3 instruct format
    '''

    if None in (input, tokenizer):
        raise ValueError("input or tokenizer should not be None")
    if type(input) != str:
        raise ValueError("input must be a string")
    
    def reformat_llama_prompt(text):
        r"""
        Remove the "Cutting Knowledge Date" and "Today Date" lines from the text. \n
        Add a newline before the "<|start_header_id|>user<|end_header_id|>" marker.
        """
        marker_user = "<|start_header_id|>user<|end_header_id|>"
        marker_assistant = "<|start_header_id|>assistant<|end_header_id|>"

        lines = text.splitlines()
        result = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("Cutting Knowledge Date:"):
                i += 1
                continue
            elif lines[i].startswith("Today Date:"):
                i += 1
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
            else:
                if marker_user in lines[i]:
                    modified_line = lines[i].replace(marker_user, "\n"+marker_user)
                    result.append(modified_line)
                else:
                    result.append(lines[i])
                i += 1
                
        if result:
            result[-1] = result[-1] + marker_assistant
        return "\n".join(result)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible."},
        {"role": "user", "content": input}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    formatted_input = reformat_llama_prompt(formatted_input)

    return formatted_input