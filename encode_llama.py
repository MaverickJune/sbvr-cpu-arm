from transformers import LlamaForCausalLM, AutoTokenizer
from models.sbvr_llama import SBVRLlamaForCausalLM
from sbvr_utils.utils_llama import get_llama
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from sbvr import sbvr 
import os

from sbvr_utils.log_config import get_logger
logger = get_logger(__name__)


@torch.no_grad()
def process_single_decoder_layer(layer_idx, target_layer, curr_device, num_sums=4, save_path=None):
    logger.info(f"Processing layer {layer_idx} on GPU {curr_device}...")
    
    attn_weights = [
        ("q", target_layer.self_attn.q_proj.weight),
        ("k", target_layer.self_attn.k_proj.weight),
        ("v", target_layer.self_attn.v_proj.weight),
    ]
    ffn_weights = [
        ("gate_proj", target_layer.mlp.gate_proj.weight),
        ("down_proj", target_layer.mlp.down_proj.weight),
        ("up_proj", target_layer.mlp.up_proj.weight),
    ]
    total_weights = attn_weights + ffn_weights
    
    for weight_name, target_weight in total_weights:
        logger.info(f"Processing {weight_name} weight...")
        weight_path = os.path.join(save_path, f"layer_{layer_idx}_{weight_name}.pt")
        target_weight = target_weight.to(curr_device)
        sbvr_compressed_weight = sbvr(target_weight, num_sums=num_sums)
        
        save_dict = {
            "coeff": sbvr_compressed_weight.coeff,
            "coeff_bias": sbvr_compressed_weight.coeff_bias,
            "bvr": sbvr_compressed_weight.bvr, # coeff_idx -> bvr
            "bvr_dtype": sbvr_compressed_weight.bvr_dtype, # added
        }
        torch.save(save_dict, weight_path)
        logger.info(f"Saved {weight_name} weight to {weight_path}")
    
@torch.no_grad()
def process_lm_head(lm_head, num_sums=4, curr_device=0, save_path=None):
    logger.info("Processing lm_head...")
    weight_path = os.path.join(save_path, "lm_head_weight.pt")
    lm_head_weight = lm_head.weight.to(curr_device)
    sbvr_compressed_weight = sbvr(lm_head_weight, num_sums=num_sums)
    
    save_dict = {
        "coeff": sbvr_compressed_weight.coeff,
        "coeff_bias": sbvr_compressed_weight.coeff_bias,
        "bvr": sbvr_compressed_weight.bvr,
        "bvr_dtype": sbvr_compressed_weight.bvr_dtype,
    }
    torch.save(save_dict, weight_path)
    logger.info(f"Saved lm_head weight to {weight_path}")
    

@torch.no_grad()
def process_sbvr_llama_multi_gpu(model, num_sums=4, save_path="compressed_weights"):
    if save_path is None:
        raise ValueError("save_path cannot be None")
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(curr_dir, save_path)
    os.makedirs(save_path, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    n_layers = len(model.model.layers)
    n_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {n_gpus}")
    
    if n_gpus == 0:
        raise ValueError("No GPUs available for processing")
    
    curr_device = 0
    proc_list = [None for _ in range(n_gpus)]

    # process_lm_head(model.lm_head.cpu(), num_sums, curr_device, save_path)
    # return
    
    logger.info(f"Processing {n_layers} layers across {n_gpus} GPUs")
    
    for layer_idx in range(n_layers):
        if proc_list[curr_device] is not None:
            proc_list[curr_device].join()
        if curr_device + 1 < n_gpus and proc_list[curr_device + 1] is not None:
            proc_list[curr_device + 1].join()
            
        proc_list[curr_device] = mp.Process(
            target=process_single_decoder_layer,
            args=(layer_idx, model.model.layers[layer_idx].cpu(), curr_device, num_sums, save_path)
        )
        proc_list[curr_device].start()
        curr_device = (curr_device + 1) % n_gpus
        
    for p in proc_list:
        p.join()
        
    process_lm_head(model.lm_head.cpu(), num_sums, curr_device, save_path)
        
    logger.info("Processing complete")


if __name__ == "__main__":
    MODEL_PATH = "meta-llama/Llama-3.2-3B"
    NUM_SUMS = 4
    SAVE_PATH = "compressed_weights"
    
    model, tokenizer = get_llama(model_path=MODEL_PATH, device_map="cpu")
    process_sbvr_llama_multi_gpu(model, num_sums=NUM_SUMS, save_path=SAVE_PATH)
    